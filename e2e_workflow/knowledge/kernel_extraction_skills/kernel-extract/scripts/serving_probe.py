#!/usr/bin/env python3
"""Execute ONE sampling point: arm a bounded profile window on the live service,
drive the intended shape through it, and verify the trace contains what was asked for.

Step three of log -> sampling points -> capture. Reads a point from
`plan_sampling_points.py` output and produces a trace plus a journal recording what
actually happened, which `coverage_ledger.py` consumes.

This is a DIAGNOSTIC probe. Its synthetic requests are chosen to hit an observed shape;
they do not replace the measured replay's weights. Per-kernel costs measured here are
combined with the replay's observed instance counts to get whole-workload mass.

Four rules, each of which was learned by a window failing
---------------------------------------------------------
1. REACH STEADY STATE BEFORE ARMING, AND CONFIRM IT.
   A window armed as load starts captures the ramp: batch 1, no steady-state expert or
   attention mix. "I sent N requests" is not confirmation -- the scheduler decides what
   is resident. This probe scrapes the server's own occupancy counter and refuses to arm
   until it reads the intended batch with non-zero generation throughput.

2. ARM ONCE. NEVER profile_by_stage.
   With a shape-discovery hook installed (TraceLens patches torch.profiler.profile.start),
   arming is SYNCHRONOUS and costs minutes -- 110-134 s measured on an 8-rank MXFP4
   deployment. profile_by_stage arms twice, so a two-stage window needs ~4 minutes of
   arming before any budget is spent on steps. A window budgeted at 150 s wall-clock
   that arms twice captures zero steps and reports no error. Budget the arming cost
   explicitly (--arm-timeout) and let the step counter close the window.

3. CLOSE THE WINDOW EXPLICITLY, INCLUDING ON EXCEPTION.
   An unclosed window rolls into the next request and silently attributes someone
   else's work to this point. /stop_profile runs in a finally block.

4. VERIFY THE TRACE, DO NOT ASSUME IT.
   A window can close successfully and contain the wrong phase, or no device time at
   all. (A graph-capture trace, for instance, records dispatch structure and shapes but
   essentially no kernel events -- it cannot carry a census.) After the window closes,
   this probe parses the trace and asserts the intended phase and shape are actually in
   it, and reports what it found either way.

A multi-node server (bench_e2e.sh PROFILE_PROBE_POINTS) runs these same rules with
--mode multinode: requests go to the router (--base-url), the window opens on the
point's engine (--profile-url) and occupancy is that engine's /metrics (or its log,
--occupancy-log, when the leg could not add --enable-metrics), every rank writes a trace
(--expect-traces), and the drives keep a PD engine's batch intact (see --mode).

Stdlib only.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observed_workload import parse_line  # noqa: E402


def post(base, path, obj=None, timeout=1800):
    data = json.dumps(obj if obj is not None else {}).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode()


def scrape_metrics(base, timeout=30):
    """(num_running_reqs, gen_throughput) from the server's own counters.

    The scheduler, not the client, decides what is resident. This is the authoritative
    occupancy signal; a request count sent from outside is not.
    """
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=timeout) as fh:
            text = fh.read().decode()
    except Exception as exc:
        return None, None, str(exc)
    out = {}
    for line in text.splitlines():
        for key in ("sglang:num_running_reqs", "sglang:gen_throughput",
                    "num_running_reqs", "gen_throughput"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                try:
                    out[key.split(":")[-1]] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out.get("num_running_reqs"), out.get("gen_throughput"), None


class LogOccupancy:
    """The same (num_running_reqs, gen_throughput) from the scheduler's own decode rows,
    for a server without /metrics. Parsed by observed_workload's dialects, the parser the
    replay weights come from. Only rows logged after this probe started count; a row is
    logged every decode_log_interval forwards, so the latest one stands until the next."""

    def __init__(self, path, rank=0):
        self.path, self.rank = path, rank
        self.offset = os.path.getsize(path)
        self.last = (None, None)

    def __call__(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError as exc:
            return None, None, str(exc)
        complete = chunk[:chunk.rfind(b"\n") + 1]
        self.offset += len(complete)
        for line in complete.decode("utf-8", "replace").splitlines():
            rec = parse_line(line)
            if rec and rec["kind"] == "decode" and rec["rank"] == self.rank:
                self.last = (rec["running"], rec["gen_tps"])
        return self.last[0], self.last[1], None


    rnd = random.Random(seed)
    return [rnd.randint(1000, 150000) for _ in range(n)]


def generate(base, ids, max_new_tokens, journal, label, timeout=1800):
    body = {"input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0,
                                "ignore_eos": True}}
    t0 = time.time()
    try:
        post(base, "/generate", body, timeout=timeout)
        journal.append({"event": "request_done", "label": label,
                        "prompt_tokens": len(ids), "max_new_tokens": max_new_tokens,
                        "wall_s": round(time.time() - t0, 2)})
        return True
    except Exception as exc:
        journal.append({"event": "request_failed", "label": label, "error": str(exc),
                        "wall_s": round(time.time() - t0, 2)})
        return False


def wait_for_occupancy(occupancy, want_bs, journal, timeout, poll=2.0, tolerance=0,
                       alive=None):
    """Block until the scheduler reports the intended batch, actively generating.

    `occupancy()` -> (bs, tps, err): scrape_metrics or a LogOccupancy.
    `alive()` -> False once a driven request has returned: a LogOccupancy row outlives
    the batch it describes, so without it an idle engine reads as the intended batch.
    Returns (ok, observed_bs, observed_tps). Never arms on a guess.
    """
    t0 = time.time()
    last = (None, None)
    while time.time() - t0 < timeout:
        if alive is not None and not alive():
            journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs,
                            "observed_bs": last[0], "gen_throughput": last[1],
                            "waited_s": round(time.time() - t0, 1),
                            "consequence": "a driven request returned before the batch "
                                           "was confirmed; window NOT armed"})
            return False, last[0], last[1]
        bs, tps, err = occupancy()
        last = (bs, tps)
        if err is None and bs is not None:
            if abs(bs - want_bs) <= tolerance and (tps or 0) > 0:
                journal.append({"event": "steady_state_confirmed", "want_bs": want_bs,
                                "observed_bs": bs, "gen_throughput": tps,
                                "waited_s": round(time.time() - t0, 1)})
                return True, bs, tps
        time.sleep(poll)
    journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs,
                    "observed_bs": last[0], "gen_throughput": last[1],
                    "waited_s": round(time.time() - t0, 1),
                    "consequence": "window NOT armed; arming here would capture the ramp "
                                   "or an unrelated batch and label it as this point"})
    return False, last[0], last[1]


def start_profile(base, out_dir, num_steps, journal, timeout, record_shapes=False,
                  with_stack=None):
    """Arm once. profile_by_stage is deliberately not offered -- see rule 2."""
    body = {"output_dir": out_dir, "num_steps": int(num_steps),
            "activities": ["CPU", "GPU"], "profile_by_stage": False}
    if record_shapes:
        body["record_shapes"] = True
    if with_stack is not None:
        body["with_stack"] = with_stack == "true"
    t0 = time.time()
    txt = post(base, "/start_profile", body, timeout=timeout)
    arm_s = round(time.time() - t0, 1)
    journal.append({"event": "armed", "arm_cost_s": arm_s, "num_steps": int(num_steps),
                    "profile_by_stage": False, "response": txt[:200]})
    if arm_s > 60:
        journal.append({"event": "note",
                        "text": "arming took %.0f s -- a shape-discovery hook is making "
                                "profiler start synchronous. Budget for this; it is not "
                                "a hang." % arm_s})
    return arm_s


def stop_profile(base, journal, timeout=1800):
    try:
        txt = post(base, "/stop_profile", {}, timeout=timeout)
        journal.append({"event": "stopped", "response": txt[:200]})
        return True
    except Exception as exc:
        journal.append({"event": "stop_FAILED", "error": str(exc),
                        "consequence": "the window may still be open and will attribute "
                                       "subsequent work to this point. Stop it by hand."})
        return False


def trace_files(trace_dir):
    return {p for p in glob.glob(os.path.join(trace_dir, "**", "*"), recursive=True)
            if os.path.isfile(p)}


def traces_written(trace_dir, before):
    """Files not present when the probe began. Not mtime: on a shared mount the writer's
    clock is another machine's."""
    out = [{"path": p, "bytes": os.path.getsize(p)} for p in trace_files(trace_dir) - before]
    return sorted(out, key=lambda d: -d["bytes"])


def wait_for_traces(trace_dir, before, expect, journal, settle=30, quiet=180, timeout=1800,
                    poll=5):
    """Every rank exports its own file after /stop_profile returns. Done when `expect` new
    files are non-empty and their byte total has not moved for `settle` s; no new bytes for
    `quiet` s with files missing means a rank died, and the verdict describes what is there."""
    t0, last, still = time.time(), -1, 0
    while True:
        written = traces_written(trace_dir, before)
        full = sum(1 for w in written if w["bytes"] > 0)
        total = sum(w["bytes"] for w in written)
        still = still + poll if total == last else 0
        last = total
        waited = time.time() - t0
        if (full >= expect and still >= settle) or still >= quiet or waited >= timeout:
            journal.append({"event": "traces_settled" if full >= expect else "traces_INCOMPLETE",
                            "expected": expect, "non_empty": full, "total_bytes": total,
                            "waited_s": round(waited, 1)})
            return full >= expect
        time.sleep(poll)


def verify_trace(paths, want_phase, want_shape_key, want_shape, journal, strict=False):
    """Rule 4: assert the trace contains the phase and shape that were asked for.
    `strict` also fails an idle window and a prefix-cache answer, and marks stray steps."""
    try:
        from parse_profile import _step_span_candidates, _classify_step
    except Exception as exc:
        journal.append({"event": "verify_SKIPPED", "error": str(exc)})
        return {"verified": None, "reason": "parse_profile unavailable: %s" % exc}

    import gzip
    found, kernels, scanned, prefill_tokens = {}, 0, 0, 0
    for rec in paths[:4]:
        p = rec["path"]
        if not (p.endswith(".json") or p.endswith(".json.gz") or p.endswith(".pt.trace.json")
                or p.endswith(".pt.trace.json.gz")):
            continue
        try:
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            continue
        scanned += 1
        events = data.get("traceEvents", data if isinstance(data, list) else [])
        kernels += sum(1 for e in events
                       if isinstance(e, dict) and e.get("cat") in ("kernel", "Kernel"))
        file_prefill_tokens = 0
        for e, nm in _step_span_candidates(events):
            c = _classify_step(nm)
            if c is None:
                continue
            phase = "PREFILL" if c[0] else "DECODE"
            shape = c[1] if c[0] else c[2]
            found.setdefault(phase, {}).setdefault(shape, 0)
            found[phase][shape] += 1
            if c[0]:
                file_prefill_tokens += c[1] or 0
        # Every rank traces the same steps; summing files would count them once per rank.
        prefill_tokens = max(prefill_tokens, file_prefill_tokens)

    res = {"files_scanned": scanned, "kernel_events": kernels,
           "step_spans_by_phase_shape": {k: dict(sorted(v.items())) for k, v in found.items()},
           "want": {"phase": want_phase, want_shape_key: want_shape}}
    got = found.get(want_phase, {})
    res["phase_present"] = bool(got)
    res["exact_shape_present"] = want_shape in got
    if strict and scanned and not found:
        res["verified"] = False
        res["reason"] = ("no step spans at all: the window ran no forward, so the engine "
                         "was idle while it was open")
    elif kernels == 0 and scanned:
        res["verified"] = False
        res["reason"] = ("trace has step spans but ~no kernel events -- it records "
                         "dispatch structure only and cannot carry a census. A "
                         "graph-capture trace looks exactly like this.")
    elif not got:
        res["verified"] = False
        res["reason"] = "no %s step spans in the trace; the window captured another phase" % want_phase
    elif strict and want_phase == "PREFILL" and want_shape and prefill_tokens * 2 < want_shape:
        # Summed over steps so a chunked prefill of the planned size still counts in full.
        res["verified"] = False
        res["prefill_tokens_computed"] = prefill_tokens
        res["reason"] = ("PREFILL computed %d of the planned %s=%s tokens (steps %s): the "
                         "prompt was mostly answered from the prefix cache, so this trace "
                         "does not measure the planned shape."
                         % (prefill_tokens, want_shape_key, want_shape, sorted(got)[:12]))
    elif not res["exact_shape_present"]:
        res["verified"] = "partial"
        res["reason"] = ("%s captured, but not at %s=%s. Present: %s. The measurement is "
                         "real; it just stands for a different shape than planned, and "
                         "the ledger must say so."
                         % (want_phase, want_shape_key, want_shape, sorted(got)[:12]))
    elif strict and len(got) > 1:
        # The census divides the window's total by its step count, so a stray step (a
        # router's /health is a 1-token generate on sglang) would halve a 1-step point.
        res["verified"] = "partial"
        res["reason"] = ("%s at %s=%s present, but the window also holds %s steps at %s, "
                         "and the census would average them in."
                         % (want_phase, want_shape_key, want_shape, want_phase,
                            sorted(s for s in got if s != want_shape)[:12]))
    else:
        res["verified"] = True
        res["reason"] = "%s at %s=%s present, %d kernel events" % (
            want_phase, want_shape_key, want_shape, kernels)
    journal.append({"event": "verify", **res})
    return res


# --------------------------------------------------------------------------- #
# (pause, body, resume) per server: while paused both still admit and queue new requests.
# sglang: scheduler.py's event loop receives before it checks _engine_paused. vllm: only
# mode=keep queues (abort/wait reject), and the route exists only with VLLM_SERVER_DEV_MODE=1.
PAUSE_APIS = (("/pause_generation", {"mode": "in_place"}, "/continue_generation"),
              ("/pause?mode=keep", None, "/resume"))


def pause_scheduler(engine, journal):
    """The resume path of the first pause API the engine accepts, else None."""
    for pause, body, resume in PAUSE_APIS:
        try:
            post(engine, pause, body, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                continue
            journal.append({"event": "pause_failed", "path": pause, "error": str(exc)})
            return None
        except Exception as exc:
            journal.append({"event": "pause_failed", "path": pause, "error": str(exc)})
            return None
        journal.append({"event": "scheduler_paused", "path": pause})
        return resume
    journal.append({"event": "pause_unavailable", "note": "no pause API on %s: the first "
                    "request is scheduled alone and the point may verify partial" % engine})
    return None


def drive_prefill(base, point, args, journal):
    """One request per intended sequence, each long enough to produce the token count."""
    toks = int(point["new_tokens"])
    nseq = int(point.get("running") or 1)
    # max_new_tokens=1 keeps the request in prefill: we are measuring the context
    # forward, and a long generation would mix decode into the window.
    with ThreadPoolExecutor(max_workers=nseq) as ex:
        for i in range(nseq):
            ex.submit(generate, base, rand_ids(toks, args.seed + i), 1, journal,
                      "prefill_seq%d" % i, args.request_timeout)


def drive_prefill_batched(base, point, args, journal, engine=None):
    """`new_seq` requests whose prompts add up to the batch's `new_tokens`, in ONE step.

    Sent concurrently, the first request reaches an idle scheduler alone and runs as its
    own step (14 x 512 runs as 512 + 6656 on sglang and vllm alike). So the point's engine
    is paused while they are sent and resumed once they are queued.
    """
    nseq = max(1, int(point.get("new_seq") or 1))
    toks, extra = divmod(int(point["new_tokens"]), nseq)
    resume = pause_scheduler(engine or base, journal) if nseq > 1 else None
    # max_new_tokens=1 keeps the request in prefill: we are measuring the context
    # forward, and a long generation would mix decode into the window.
    with ThreadPoolExecutor(max_workers=nseq) as ex:
        try:
            for i in range(nseq):
                n = max(1, toks + (i < extra))
                ex.submit(generate, base, rand_ids(n, args.seed + i, getattr(args, "vocab", None)), 1, journal,
                          "prefill_seq%d" % i, args.request_timeout)
            if resume:
                time.sleep(args.gather_s)
        finally:
            if resume:
                post(engine or base, resume, timeout=60)
                journal.append({"event": "scheduler_resumed", "path": resume,
                                "gathered_s": args.gather_s})


def drive_decode(base, point, args, journal):
    """Hold `bs` requests resident and generating, then arm. Returns the futures so the
    caller can keep them alive across the window."""
    bs = int(point["batch_size"])
    ctx = int(point.get("context_tokens") or 1024)
    ex = ThreadPoolExecutor(max_workers=bs)
    for i in range(bs):
        ex.submit(generate, base, rand_ids(ctx, args.seed + 100 + i),
                  args.decode_tokens, journal, "decode_req%d" % i, args.request_timeout)
    return ex


DECODE_TAIL_TOKENS = 256


def drive_decode_closed_loop(base, point, args, journal, stop):
    """Hold `bs` requests resident until `stop` is set: `bs` workers, each sending its next
    request the moment its last one returns, so the batch outlives any arm cost.

    Every prompt is one shared prefix plus its own tail, and the prefix is prefilled once
    before the workers start. A full-context prefill takes about a minute at long ISL and a
    PD prefill engine runs them one after another, so `bs` fresh prompts reach decode a
    minute apart and are never resident together; a tail's prefill takes well under a second.
    The decode shape is unchanged: each request still attends over its whole context."""
    bs = int(point["batch_size"])
    ctx = int(point.get("context_tokens") or 1024)
    vocab = getattr(args, "vocab", None)
    tail = min(DECODE_TAIL_TOKENS, max(1, ctx // 4))
    prefix = rand_ids(ctx - tail, args.seed + 100, vocab)
    if not generate(base, prefix, 1, journal, "decode_prefix", args.request_timeout):
        raise RuntimeError("the shared decode prefix could not be prefilled")

    def worker(i):
        n = 0
        while not stop.is_set():
            ids = prefix + rand_ids(tail, args.seed + 1000 * (i + 1) + n, vocab)
            if not generate(base, ids, args.decode_tokens, journal, "decode_req%d" % i,
                            args.request_timeout):
                return
            n += 1

    ex = ThreadPoolExecutor(max_workers=bs)
    for i in range(bs):
        ex.submit(worker, i)
    return ex


def decode_requests_done(journal):
    return sum(1 for e in journal if e.get("event") == "request_done"
               and str(e.get("label", "")).startswith("decode_req"))


def decode_requests_failed(journal):
    return sum(1 for e in journal if e.get("event") == "request_failed"
               and str(e.get("label", "")).startswith("decode_req"))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="plan_sampling_points.py output")
    ap.add_argument("--point-id", required=True)
    ap.add_argument("--base-url", default=os.environ.get("PROBE_BASE_URL",
                                                         "http://127.0.0.1:30000"))
    ap.add_argument("--profile-url", default=None,
                    help="where /start_profile and /stop_profile go, if not --base-url "
                         "(PD: the engine of this point's phase; requests go to the router)")
    ap.add_argument("--occupancy-log", default=None,
                    help="read occupancy from this scheduler log instead of /metrics "
                         "(the decode engine's log on a server without --enable-metrics)")
    ap.add_argument("--expect-traces", type=int, default=0,
                    help="after the window closes, wait for this many trace files (one per "
                         "rank) to finish exporting before verifying")
    ap.add_argument("--trace-settle", type=float, default=30)
    ap.add_argument("--trace-quiet", type=float, default=180)
    ap.add_argument("--record-shapes", action="store_true")
    ap.add_argument("--with-stack", choices=("true", "false"), default=None,
                    help="sent only when given; Python stacks can make the export block "
                         "the scheduler")
    ap.add_argument("--trace-dir", required=True,
                    help="host-visible directory the server writes traces to "
                         "(SGLANG_TORCH_PROFILER_DIR)")
    ap.add_argument("--server-trace-dir", default=None,
                    help="same directory as the SERVER sees it, if it differs (container)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="forwards to capture; the scheduler closes the window on this")
    ap.add_argument("--arm-timeout", type=float, default=600,
                    help="budget for the synchronous arm; 110-134 s is normal with a "
                         "shape hook installed")
    ap.add_argument("--steady-timeout", type=float, default=300)
    ap.add_argument("--window-timeout", type=float, default=900)
    ap.add_argument("--gather-s", type=float, default=2.0,
                    help="PREFILL: how long the paused engine queues the point's requests "
                         "before it is resumed")
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--mode", choices=("single", "multinode"), default="single",
                    help="'multinode' (bench_e2e.sh's multinode launcher): PREFILL sends new_seq "
                         "sequences into one paused step; DECODE keeps batch_size shared-prefix "
                         "requests resident until the window closes; the trace check also fails "
                         "prefix-cache hits and stray steps")
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, help="journal JSON for this point")
    args = ap.parse_args()

    plan = json.load(open(args.points))
    pts = [p for p in plan["points"] if p["point_id"] == args.point_id]
    if not pts:
        raise SystemExit("serving_probe: no point %r in %s. Have: %s"
                         % (args.point_id, args.points,
                            ", ".join(p["point_id"] for p in plan["points"])))
    point = pts[0]
    phase = point["phase"]

    journal = [{"event": "begin", "point": point, "base_url": args.base_url,
                "profile_url": profile_url, "occupancy": args.occupancy_log or "/metrics",
                "seed": args.seed,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    t_start = time.time()
    before = trace_files(args.trace_dir)
    server_dir = args.server_trace_dir or args.trace_dir
    num_steps = args.num_steps or (32 if phase == "DECODE" else 8)
    ok = False
    verify = {}
    ex = None
    stop_decode = threading.Event()

    multinode = args.mode == "multinode"
    try:
        # /metrics of the engine the window opens on: a PD router serves no engine counters.
        occupancy = (LogOccupancy(args.occupancy_log) if args.occupancy_log
                     else lambda: scrape_metrics(profile_url))
        if phase == "DECODE":
            # Steady state FIRST (rule 1): put the batch on, confirm the scheduler has
            # admitted it and is generating, and only then pay for the arm.
            want = int(point["batch_size"])
            if multinode:
                ex = drive_decode_closed_loop(args.base_url, point, args, journal, stop_decode)
                alive = lambda: decode_requests_failed(journal) == 0
            else:
                ex = drive_decode(args.base_url, point, args, journal)
                alive = None
            confirmed, obs_bs, _ = wait_for_occupancy(
                occupancy, want, journal, args.steady_timeout, alive=alive)
            if not confirmed:
                raise RuntimeError(
                    "steady state not reached: wanted bs=%s, scheduler reported %s. Not "
                    "arming." % (want, obs_bs))
            start_profile(profile_url, server_dir, num_steps, journal, args.arm_timeout,
                          args.record_shapes, args.with_stack)
            deadline = time.time() + args.window_timeout
            if multinode:
                # sglang closes the window itself once num_steps forwards have run and then
                # exports, well inside one generation; a server whose profiler has no step
                # counter is stopped once every resident request has turned over.
                done0 = decode_requests_done(journal)
                while (time.time() < deadline and not traces_written(args.trace_dir, before)
                       and decode_requests_done(journal) - done0 < want):
                    if decode_requests_failed(journal):
                        raise RuntimeError("a decode request failed while the window was open")
                    time.sleep(5)
            else:
                # The scheduler closes the window itself once num_steps forwards have run;
                # the batch is already resident, so the steps are the intended shape.
                while time.time() < deadline:
                    time.sleep(5)
                    bs, tps, _ = occupancy()
                    if bs == 0:
                        journal.append({"event": "batch_drained", "note": "requests finished "
                                        "before the window closed; shorten --decode-tokens "
                                        "or lower --num-steps"})
                        break
        else:
            # Prefill is a single forward per request: arm first, then send, because
            # there is no resident state to confirm. The window is still bounded by
            # num_steps and still closed explicitly.
            start_profile(profile_url, server_dir, num_steps, journal, args.arm_timeout,
                          args.record_shapes, args.with_stack)
            if multinode:
                drive_prefill_batched(args.base_url, point, args, journal, profile_url)
            else:
                drive_prefill(args.base_url, point, args, journal)
        ok = True
    except Exception as exc:
        journal.append({"event": "probe_failed", "error": "%s: %s"
                        % (type(exc).__name__, exc)})
    finally:
        # Rule 3: always, even on the failure path.
        stop_profile(profile_url, journal)
        stop_decode.set()
        if ex is not None:
            # Closed loop: a request still decoding would sit in the next point's batch.
            ex.shutdown(wait=multinode)

    # A rank set missing a rank is not the server: the point fails whatever TP-0 shows.
    complete = not (ok and args.expect_traces) or wait_for_traces(
        args.trace_dir, before, args.expect_traces, journal,
        args.trace_settle, args.trace_quiet)
    ok = ok and complete
    written = traces_written(args.trace_dir, before)
    journal.append({"event": "traces_written", "count": len(written),
                    "total_bytes": sum(w["bytes"] for w in written),
                    "files": written[:12]})
    if written:
        verify = verify_trace(
            written, phase,
            "new_tokens" if phase == "PREFILL" else "batch_size",
            point.get("new_tokens") if phase == "PREFILL" else point.get("batch_size"),
            journal, strict=multinode)
    else:
        journal.append({"event": "verify", "verified": False,
                        "reason": "no trace files appeared under %s" % args.trace_dir})
        verify = {"verified": False}

    result = {
        "point_id": args.point_id,
        "point": point,
        "status": "ok" if (ok and verify.get("verified") is True) else
                  "partial" if (ok and verify.get("verified") == "partial") else "failed",
        "trace_verified": verify.get("verified"),
        "verify_reason": verify.get("reason"),
        "wall_s": round(time.time() - t_start, 1),
        "traces": written[:12],
        "journal": journal,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("%s: %s (%s) in %.0f s -> %s"
          % (args.point_id, result["status"], result["verify_reason"],
             result["wall_s"], args.out))
    # A failed point is evidence, not an error to hide; preserve the journal and let the
    # caller decide whether to spend its one recapture.
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
