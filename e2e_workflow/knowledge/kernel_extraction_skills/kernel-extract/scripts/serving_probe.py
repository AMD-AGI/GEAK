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

5. DETECT A STALLED WINDOW FROM THE CLIENT, DO NOT WAIT FOR THE SERVER TO DIE.
   A profiler-armed decode window has been seen to freeze one TP rank: the streams stop,
   the NCCL watchdog fires only after 600 s, and every request then hangs until its own
   timeout (30 min). The streamed chunk counts already prove progress, so if no request
   receives a chunk for --stall-timeout seconds the point is failed as `stalled`, the
   window is closed with a short timeout, and the caller can restart the server instead
   of waiting out the watchdog.

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


def post(base, path, obj=None, timeout=1800):
    data = json.dumps(obj if obj is not None else {}).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode()


METRIC_KEYS = ("num_running_reqs", "gen_throughput",
               # Cumulative counters. `generation_tokens_total` advances ONLY during
               # decode and `prompt_tokens_total` ONLY during prefill, so the pair says
               # which phase the server is actually in -- which occupancy alone does not:
               # a long prompt still being chunk-ingested reports its request as running.
               "generation_tokens_total", "prompt_tokens_total")


def scrape_metrics(base, timeout=30):
    """The server's own counters as a dict, plus an error string.

    The scheduler, not the client, decides what is resident. This is the authoritative
    occupancy signal; a request count sent from outside is not.
    """
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=timeout) as fh:
            text = fh.read().decode()
    except Exception as exc:
        return {}, str(exc)
    out = {}
    for line in text.splitlines():
        for key in METRIC_KEYS:
            for name in ("sglang:" + key, key):
                if line.startswith(name + "{") or line.startswith(name + " "):
                    try:
                        out[key] = float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
    return out, None


def rand_ids(n, seed):
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
    except Exception as exc:
        journal.append({"event": "request_failed", "label": label, "error": str(exc),
                        "wall_s": round(time.time() - t0, 2)})


def generate_stream(base, ids, max_new_tokens, journal, label, progress, timeout=1800):
    """Like generate(), but streamed, counting chunks into progress[label].

    The first chunk arrives when the request's prefill has finished (its first token);
    every later chunk is one decode step. So `progress[label] >= 2` is a client-side
    proof that this request is decoding, independent of which counters the server
    exports or when it increments them.
    """
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0,
                                "ignore_eos": True}}
    req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    progress[label] = 0
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            for raw in fh:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    progress[label] += 1
        journal.append({"event": "request_done", "label": label,
                        "prompt_tokens": len(ids), "max_new_tokens": max_new_tokens,
                        "stream_chunks": progress[label],
                        "wall_s": round(time.time() - t0, 2)})
    except Exception as exc:
        journal.append({"event": "request_failed", "label": label, "error": str(exc),
                        "stream_chunks": progress.get(label),
                        "wall_s": round(time.time() - t0, 2)})
    finally:
        progress[label] = -1    # finished: no longer resident


def start_stream(*args):
    """Run generate_stream on a DAEMON thread. ThreadPoolExecutor workers are joined at
    interpreter exit, so one request stuck on a wedged server would keep the probe alive
    for the whole request timeout after the point has already been failed as stalled."""
    t = threading.Thread(target=generate_stream, args=args, daemon=True)
    t.start()
    return t


class StallError(RuntimeError):
    """No streamed request advanced for the stall timeout (rule 5)."""


class StallWatch:
    """Track streamed progress and raise StallError once nothing has advanced for stall_s.

    `progress` is the label -> chunk-count map filled by generate_stream (-1 = finished).
    Progress is any change of that map: a new chunk, a request finishing, or a new request.
    stall_s <= 0 disables the check.
    """

    def __init__(self, progress, stall_s, journal, where):
        self.progress, self.stall_s = progress, float(stall_s or 0)
        self.journal, self.where = journal, where
        self.snap, self.t_change = dict(progress), time.time()

    def check(self):
        now = time.time()
        cur = dict(self.progress)
        if cur != self.snap:
            self.snap, self.t_change = cur, now
            return
        resident = [k for k, c in cur.items() if c >= 0]
        if self.stall_s > 0 and resident and now - self.t_change > self.stall_s:
            self.journal.append({
                "event": "stalled", "where": self.where,
                "no_progress_s": round(now - self.t_change, 1),
                "stall_timeout_s": self.stall_s, "stream_chunks": cur,
                "consequence": "point failed; the server is likely wedged (e.g. a TP rank "
                               "stuck in a collective) -- restart it before the next point"})
            raise StallError("no streamed progress for %.0f s during %s"
                             % (now - self.t_change, self.where))


def wait_for_occupancy(base, want_bs, journal, timeout, poll=2.0, tolerance=0,
                       progress=None):
    """Block until the scheduler reports the intended batch AND is past prefill.

    Occupancy alone is not steady state. A request with a 234k-token prompt is counted
    as running from the moment it is admitted, while the scheduler spends the next ten
    seconds chunk-ingesting it -- so `num_running_reqs == want and gen_throughput > 0`
    went true 2.1 s in, the window was armed on prefill chunks, and the capture carried
    no decode step spans at all.

    `generation_tokens_total` / `prompt_tokens_total` cannot close that gap: SGLang's
    tokenizer manager increments both in observe_one_finished_request, i.e. only when a
    request FINISHES, so during steady decode neither moves and a gate on them never
    fires. When the caller streams its requests, `progress` (label -> chunks received)
    is the signal instead: every resident request has streamed >= 2 chunks, i.e. it is
    past prefill and has taken at least one decode step.

    Returns (ok, observed_bs, observed_tps). Never arms on a guess.
    """
    t0 = time.time()
    prev, last = None, {}
    while time.time() - t0 < timeout:
        cur, err = scrape_metrics(base)
        last = cur
        bs, tps = cur.get("num_running_reqs"), cur.get("gen_throughput")
        if err is None and bs is not None and abs(bs - want_bs) <= tolerance \
                and progress is not None:
            chunks = list(progress.values())
            decoding = sum(1 for c in chunks if c >= 2)
            if decoding >= want_bs and not any(c == -1 for c in chunks):
                journal.append({
                    "event": "steady_state_confirmed", "want_bs": want_bs,
                    "observed_bs": bs, "gen_throughput": tps,
                    "waited_s": round(time.time() - t0, 1),
                    "basis": "occupancy + every request streamed >= 2 chunks",
                    "stream_chunks": dict(progress)})
                return True, bs, tps
        elif err is None and bs is not None and abs(bs - want_bs) <= tolerance:
            gen, pro = cur.get("generation_tokens_total"), cur.get("prompt_tokens_total")
            if gen is None or pro is None:
                # Older/leaner metric sets: fall back to the weaker signal, and say so,
                # because this is exactly the condition that produced an empty capture.
                if (tps or 0) > 0:
                    journal.append({
                        "event": "steady_state_confirmed", "want_bs": want_bs,
                        "observed_bs": bs, "gen_throughput": tps,
                        "waited_s": round(time.time() - t0, 1),
                        "basis": "occupancy+throughput only",
                        "caveat": "generation_tokens_total/prompt_tokens_total absent, so "
                                  "'past prefill' could not be confirmed; verify_trace is "
                                  "the only remaining guard against a prefill window"})
                    return True, bs, tps
            elif prev is not None:
                decoding = gen - prev.get("generation_tokens_total", gen) >= max(1, want_bs)
                ingesting = pro - prev.get("prompt_tokens_total", pro) > 0
                if decoding and not ingesting:
                    journal.append({
                        "event": "steady_state_confirmed", "want_bs": want_bs,
                        "observed_bs": bs, "gen_throughput": tps,
                        "waited_s": round(time.time() - t0, 1),
                        "basis": "occupancy + generation advancing + prefill quiet",
                        "generated_tokens_delta": gen - prev["generation_tokens_total"],
                        "prompt_tokens_delta": pro - prev["prompt_tokens_total"]})
                    return True, bs, tps
            prev = cur
        time.sleep(poll)
    journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs,
                    "observed_bs": last.get("num_running_reqs"),
                    "gen_throughput": last.get("gen_throughput"),
                    "waited_s": round(time.time() - t0, 1),
                    "consequence": "window NOT armed; arming here would capture the ramp "
                                   "or an unrelated batch and label it as this point"})
    return False, last.get("num_running_reqs"), last.get("gen_throughput")


def start_profile(base, out_dir, num_steps, journal, timeout):
    """Arm once. profile_by_stage is deliberately not offered -- see rule 2."""
    body = {"output_dir": out_dir, "num_steps": int(num_steps),
            "activities": ["CPU", "GPU"], "profile_by_stage": False}
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


def traces_written(trace_dir, since_ts):
    out = []
    for p in glob.glob(os.path.join(trace_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.getmtime(p) >= since_ts:
            out.append({"path": p, "bytes": os.path.getsize(p)})
    return sorted(out, key=lambda d: -d["bytes"])


def verify_trace(paths, want_phase, want_shape_key, want_shape, journal):
    """Rule 4: assert the trace contains the phase and shape that were asked for."""
    try:
        from parse_profile import _step_span_candidates, _classify_step
    except Exception as exc:
        journal.append({"event": "verify_SKIPPED", "error": str(exc)})
        return {"verified": None, "reason": "parse_profile unavailable: %s" % exc}

    import gzip
    found, kernels, scanned = {}, 0, 0
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
        for e, nm in _step_span_candidates(events):
            c = _classify_step(nm)
            if c is None:
                continue
            phase = "PREFILL" if c[0] else "DECODE"
            shape = c[1] if c[0] else c[2]
            found.setdefault(phase, {}).setdefault(shape, 0)
            found[phase][shape] += 1

    res = {"files_scanned": scanned, "kernel_events": kernels,
           "step_spans_by_phase_shape": {k: dict(sorted(v.items())) for k, v in found.items()},
           "want": {"phase": want_phase, want_shape_key: want_shape}}
    got = found.get(want_phase, {})
    res["phase_present"] = bool(got)
    res["exact_shape_present"] = want_shape in got
    if kernels == 0 and scanned:
        res["verified"] = False
        res["reason"] = ("trace has step spans but ~no kernel events -- it records "
                         "dispatch structure only and cannot carry a census. A "
                         "graph-capture trace looks exactly like this.")
    elif not got:
        res["verified"] = False
        res["reason"] = "no %s step spans in the trace; the window captured another phase" % want_phase
    elif not res["exact_shape_present"]:
        res["verified"] = "partial"
        res["reason"] = ("%s captured, but not at %s=%s. Present: %s. The measurement is "
                         "real; it just stands for a different shape than planned, and "
                         "the ledger must say so."
                         % (want_phase, want_shape_key, want_shape, sorted(got)[:12]))
    else:
        res["verified"] = True
        res["reason"] = "%s at %s=%s present, %d kernel events" % (
            want_phase, want_shape_key, want_shape, kernels)
    journal.append({"event": "verify", **res})
    return res


# --------------------------------------------------------------------------- #
def warm_prefix(base, prefixes, args, journal):
    """Put each sequence's cached prefix in the radix cache BEFORE the measured extend.

    Without this the probe sends a fresh prompt and the server reports
    `#new-token: N, #cached-token: 0`, while the point it claims to stand for was
    measured at `#new-token: N, #cached-token: 127040`. The served workload hits prefix
    cache 97% of the time, so the uncached version runs the same new tokens against a
    context roughly fifty times shorter -- and attention, the indexer and the KV gather
    all scale with that context, not with N. The measurement is then real and about a
    workload nobody runs.
    """
    if not any(prefixes):
        return
    with ThreadPoolExecutor(max_workers=max(1, len(prefixes))) as ex:
        for i, ids in enumerate(prefixes):
            if ids:
                ex.submit(generate, base, list(ids), 1, journal,
                          "prefix_warm%d" % i, args.request_timeout)
    journal.append({"event": "prefix_warmed",
                    "sequences": [len(p) for p in prefixes],
                    "note": "the measured extend below should report these as cached"})


def prepare_prefill(base, point, args, journal):
    """Plan ONE observed prefill batch and warm its cached prefixes; send nothing measured.

    Must run BEFORE the window is armed. The warm requests are themselves long uncached
    prefills (a 127k-token prefix is ~8 chunked forwards at chunk 16384), so warming
    inside the window spends the whole num_steps budget on the warm-up and the measured
    extend never lands in the trace.

    `new_tokens` and `cached_tokens` are BATCH totals read from the scheduler log's
    `#new-token` / `#cached-token`, and `running` is how many sequences that batch held.
    Sending `running` requests of `new_tokens` each made the scheduler merge them into a
    single extend of `running x new_tokens`, so every multi-sequence point measured two
    or three times the token count it was labelled with. Split the batch totals across
    the sequences instead.
    """
    total_new = max(1, int(point["new_tokens"]))
    total_cached = max(0, int(point.get("cached_tokens") or 0))
    # The extend's width is `#new-seq` (point["new_seq"]). `running` is `#running-req`,
    # the OTHER requests resident at the time: splitting by it turned a one-sequence
    # 3717-token extend into two separately scheduled 1858/1859 extends.
    nseq = max(1, int(point.get("new_seq") or 1))

    per_new = [total_new // nseq] * nseq
    per_cached = [total_cached // nseq] * nseq
    for i in range(total_new - sum(per_new)):
        per_new[i] += 1                      # keep the batch total exact
    for i in range(total_cached - sum(per_cached)):
        per_cached[i] += 1

    # Distinct prefixes per sequence: they are separate requests in the served batch and
    # a shared prefix would collapse their cached mass onto one radix path.
    prefixes = [rand_ids(n, args.seed + 500 + i) if n else []
                for i, n in enumerate(per_cached)]
    warm_prefix(base, prefixes, args, journal)

    journal.append({"event": "prefill_plan", "batch_new_tokens": total_new,
                    "batch_cached_tokens": total_cached, "sequences": nseq,
                    "per_sequence_new": per_new, "per_sequence_cached": per_cached})
    return prefixes, per_new


def send_prefill(base, prefixes, per_new, args, journal, poll=2.0):
    """Send the measured extend: each warmed prefix plus its new tokens, concurrently.
    Streamed, so a wedged server is caught by the stall watch (rule 5) instead of
    holding the probe for the full request timeout."""
    # max_new_tokens=1 keeps the request in prefill: we are measuring the context
    # forward, and a long generation would mix decode into the window.
    progress = {"prefill_seq%d" % i: 0 for i in range(len(per_new))}
    watch = StallWatch(progress, args.stall_timeout, journal, "prefill_window")
    for i, n in enumerate(per_new):
        ids = prefixes[i] + rand_ids(n, args.seed + i)
        start_stream(base, ids, 1, journal, "prefill_seq%d" % i, progress,
                     args.request_timeout)
    while any(c != -1 for c in progress.values()):
        time.sleep(poll)
        watch.check()


def drive_decode(base, point, args, journal):
    """Hold `bs` requests resident and generating, then arm. The requests run on daemon
    threads, so they stay alive across the window without pinning the process after it;
    returns the per-request stream progress that wait_for_occupancy and the stall watch
    gate on."""
    bs = int(point["batch_size"])
    ctx = int(point.get("context_tokens") or 1024)
    progress = {}
    for i in range(bs):
        start_stream(base, rand_ids(ctx, args.seed + 100 + i), args.decode_tokens,
                     journal, "decode_req%d" % i, progress, args.request_timeout)
    return progress


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="plan_sampling_points.py output")
    ap.add_argument("--point-id", required=True)
    ap.add_argument("--base-url", default=os.environ.get("PROBE_BASE_URL",
                                                         "http://127.0.0.1:30000"))
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
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--stall-timeout", type=float, default=180,
                    help="fail the point as `stalled` when no streamed request advances "
                         "for this many seconds inside the window (rule 5); 0 disables. "
                         "Keep it above the server's trace-export pause")
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
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    t_start = time.time()
    server_dir = args.server_trace_dir or args.trace_dir
    num_steps = args.num_steps or (32 if phase == "DECODE" else 8)
    ok = False
    stalled = False
    verify = {}

    try:
        if phase == "DECODE":
            # Steady state FIRST (rule 1): put the batch on, confirm the scheduler has
            # admitted it and is generating, and only then pay for the arm.
            progress = drive_decode(args.base_url, point, args, journal)
            want = int(point["batch_size"])
            confirmed, obs_bs, _ = wait_for_occupancy(
                args.base_url, want, journal, args.steady_timeout, progress=progress)
            if not confirmed:
                raise RuntimeError(
                    "steady state not reached: wanted bs=%s, scheduler reported %s. Not "
                    "arming." % (want, obs_bs))
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout)
            # The scheduler closes the window itself once num_steps forwards have run;
            # the batch is already resident, so the steps are the intended shape.
            watch = StallWatch(progress, args.stall_timeout, journal, "decode_window")
            deadline = time.time() + args.window_timeout
            while time.time() < deadline:
                time.sleep(5)
                watch.check()
                metrics, _ = scrape_metrics(args.base_url)
                bs = metrics.get("num_running_reqs")
                if bs == 0:
                    journal.append({"event": "batch_drained", "note": "requests finished "
                                    "before the window closed; shorten --decode-tokens "
                                    "or lower --num-steps"})
                    break
        else:
            # Warm the cached prefixes OUTSIDE the window, then arm, then send only the
            # measured extend. The window is still bounded by num_steps and still
            # closed explicitly.
            prefixes, per_new = prepare_prefill(args.base_url, point, args, journal)
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout)
            send_prefill(args.base_url, prefixes, per_new, args, journal)
        ok = True
    except Exception as exc:
        stalled = isinstance(exc, StallError)
        journal.append({"event": "probe_failed", "error": "%s: %s"
                        % (type(exc).__name__, exc)})
    finally:
        # Rule 3: always, even on the failure path. A wedged server will not answer
        # /stop_profile either, so do not wait the full request timeout on it.
        stop_profile(args.base_url, journal, timeout=60 if stalled else 1800)

    written = traces_written(args.trace_dir, t_start)
    journal.append({"event": "traces_written", "count": len(written),
                    "total_bytes": sum(w["bytes"] for w in written),
                    "files": written[:12]})
    if written:
        verify = verify_trace(
            written, phase,
            "new_tokens" if phase == "PREFILL" else "batch_size",
            point.get("new_tokens") if phase == "PREFILL" else point.get("batch_size"),
            journal)
    else:
        journal.append({"event": "verify", "verified": False,
                        "reason": "no trace files appeared under %s" % args.trace_dir})
        verify = {"verified": False}

    result = {
        "point_id": args.point_id,
        "point": point,
        "status": "ok" if (ok and verify.get("verified") is True) else
                  "partial" if (ok and verify.get("verified") == "partial") else "failed",
        "stalled": stalled,
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
