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

Stdlib only.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
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


_REALTIME_RE = re.compile(r'^sglang:realtime_tokens_total\{(?P<labels>[^}]*)\}\s+(?P<val>\S+)')
FRAMEWORK = "sglang"
SERVED_MODEL = None
ITERATION_LOG_DIR = None
ID_HI = 150000


def scrape_metrics(base, timeout=30):
    """The server's own counters as a dict, plus an error string (None on success).

    The scheduler, not the client, decides what is resident, so occupancy comes from
    here. Two phase counters are read alongside it: `decode_tokens` advances only while
    decode runs and `prefill_tokens` only while prompts are ingested. Both come from
    `realtime_tokens_total{mode=...}`, which the scheduler updates as iterations run.
    `generation_tokens_total` / `prompt_tokens_total` are the fallback: on sglang
    0.5.19-0.5.20 they are bumped only when a request FINISHES, so during a long
    generation they stand still and a probe waiting on them never arms.
    """
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=timeout) as fh:
            text = fh.read().decode()
    except Exception as exc:
        return {}, str(exc)
    out, realtime = {}, {}
    for line in text.splitlines():
        if FRAMEWORK == "vllm":
            for metric, key in (("vllm:num_requests_running", "num_running_reqs"),
                                ("vllm:generation_tokens_total", "generation_tokens_total"),
                                ("vllm:prompt_tokens_total", "prompt_tokens_total")):
                if line.startswith(metric + "{") or line.startswith(metric + " "):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        m = _REALTIME_RE.match(line)
        if m:
            mode = re.search(r'mode="([^"]+)"', m.group("labels"))
            try:
                key = mode.group(1) if mode else "?"
                realtime[key] = realtime.get(key, 0.0) + float(m.group("val"))
            except ValueError:
                pass
            continue
        for key in ("num_running_reqs", "gen_throughput", "generation_tokens_total",
                    "prompt_tokens_total"):
            for name in ("sglang:" + key, key):
                if line.startswith(name + "{") or line.startswith(name + " "):
                    try:
                        out[key] = float(line.rsplit(" ", 1)[1])
                    except ValueError:
                        pass
    if "decode" in realtime:
        out["decode_tokens"] = realtime["decode"]
        out["prefill_tokens"] = (realtime.get("prefill_compute", 0.0)
                                 + realtime.get("prefill_cache", 0.0))
        out["phase_counters"] = "realtime_tokens_total"
    elif "generation_tokens_total" in out and "prompt_tokens_total" in out:
        out["decode_tokens"] = out["generation_tokens_total"]
        out["prefill_tokens"] = out["prompt_tokens_total"]
        out["phase_counters"] = "generation/prompt_tokens_total (request-scoped)"
    return out, None


def rand_ids(n, seed):
    rnd = random.Random(seed)
    return [rnd.randint(0, ID_HI) for _ in range(n)]


def generate(base, ids, max_new_tokens, journal, label, timeout=1800, rid=None):
    body = {"input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0,
                                "ignore_eos": True}}
    if rid:
        body["rid"] = rid
    endpoint = "/generate"
    if FRAMEWORK == "vllm":
        endpoint = "/v1/completions"
        body = {"model": SERVED_MODEL, "prompt": ids, "max_tokens": max_new_tokens,
                "temperature": 0.0, "ignore_eos": True}
    t0 = time.time()
    try:
        post(base, endpoint, body, timeout=timeout)
        journal.append({"event": "request_done", "label": label,
                        "prompt_tokens": len(ids), "max_new_tokens": max_new_tokens,
                        "wall_s": round(time.time() - t0, 2)})
    except Exception as exc:
        journal.append({"event": "request_failed", "label": label, "error": str(exc),
                        "wall_s": round(time.time() - t0, 2)})


def wait_for_occupancy(base, want_bs, journal, timeout, poll=0.5, tolerance=0):
    """Block until the scheduler holds the intended batch AND is past prefill.

    Occupancy alone is not steady state: a request with a 234k-token prompt counts as
    running from admission while the scheduler spends the next ten seconds chunk-ingesting
    it, and a window armed then carried no decode step at all. Arm only once the decode
    counter has advanced by at least one token per resident request since the last
    reading AND the prefill counter has not moved. Counters that update in steps (every
    decode_log_interval iterations) simply take a few polls to show the advance.

    Returns (ok, observed_bs, observed_tps). Never arms on a guess.
    """
    t0 = time.time()
    if FRAMEWORK == "vllm" and ITERATION_LOG_DIR:
        while time.time() - t0 < timeout:
            rows = []
            for path in glob.glob(os.path.join(ITERATION_LOG_DIR, "rank0_pid*.jsonl")):
                with open(path, "rb") as fh:
                    fh.seek(max(0, os.path.getsize(path) - 65536))
                    lines = fh.read().splitlines()
                for line in lines[-12:]:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d["time"] >= t0:
                        if "error" in d:
                            raise RuntimeError(d["error"])
                        rows.append(d)
            rows.sort(key=lambda d: d["time"])
            latest = rows[-3:]
            if len(latest) == 3 and time.time() - latest[-1]["time"] < 3 and all(
                    len(d["requests"]) == want_bs and
                    all(r["phase"] == "DECODE" for r in d["requests"]) for d in latest):
                journal.append({"event": "steady_state_confirmed", "want_bs": want_bs,
                                "observed_bs": want_bs, "basis": "three consecutive observed vLLM decode iterations",
                                "last_iteration": latest[-1]})
                return True, want_bs, None
            time.sleep(poll)
        journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs})
        return False, None, None
    prev, last = None, {}
    while time.time() - t0 < timeout:
        cur, err = scrape_metrics(base)
        last = cur or last
        bs, tps = cur.get("num_running_reqs"), cur.get("gen_throughput")
        if err is None and bs is not None and abs(bs - want_bs) <= tolerance:
            if "decode_tokens" not in cur:
                # No phase counter at all: the weaker signal, labelled as such, because
                # it is exactly what produced prefill windows labelled as decode.
                if (tps or 0) > 0:
                    journal.append({
                        "event": "steady_state_confirmed", "want_bs": want_bs,
                        "observed_bs": bs, "gen_throughput": tps,
                        "waited_s": round(time.time() - t0, 1),
                        "basis": "occupancy+throughput only",
                        "caveat": "no phase counter on this server; verify_trace is the only "
                                  "guard against a prefill window"})
                    return True, bs, tps
            elif prev is not None and "decode_tokens" in prev:
                dec = cur["decode_tokens"] - prev["decode_tokens"]
                pre = cur["prefill_tokens"] - prev["prefill_tokens"]
                if dec >= max(1, want_bs) and pre <= 0:
                    journal.append({
                        "event": "steady_state_confirmed", "want_bs": want_bs,
                        "observed_bs": bs, "gen_throughput": tps,
                        "waited_s": round(time.time() - t0, 1),
                        "basis": "occupancy + decode advancing + prefill quiet",
                        "counters": cur.get("phase_counters"),
                        "decode_tokens_delta": dec, "prefill_tokens_delta": pre})
                    return True, bs, tps
                if dec == 0 and pre == 0:
                    # A step-updated counter has not ticked yet: keep the older reading
                    # so the next tick is compared against a quiet interval.
                    time.sleep(poll)
                    continue
            prev = cur
        else:
            prev = None
        time.sleep(poll)
    journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs,
                    "observed_bs": last.get("num_running_reqs"),
                    "gen_throughput": last.get("gen_throughput"),
                    "counters": last.get("phase_counters"),
                    "waited_s": round(time.time() - t0, 1),
                    "consequence": "window NOT armed; arming here would capture the ramp "
                                   "or an unrelated batch and label it as this point"})
    return False, last.get("num_running_reqs"), last.get("gen_throughput")


def start_profile(base, out_dir, num_steps, journal, timeout):
    """Arm once. profile_by_stage is deliberately not offered -- see rule 2."""
    body = {"output_dir": out_dir, "num_steps": int(num_steps),
            "activities": ["CPU", "GPU"], "profile_by_stage": False}
    if FRAMEWORK == "vllm":
        body = {"profile_prefix": os.path.basename(out_dir), "max_iterations": int(num_steps)}
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


def wait_traces_settled(trace_dir, since_ts, journal, timeout, expect_files=0, poll=1.0,
                        stable_reads=3):
    """True once trace files exist (at least expect_files) and stopped growing.

    The scheduler closes the window itself after num_steps forwards and then writes one
    file per rank; that, not a fixed timeout, is when the capture is over. Settled means
    `stable_reads` identical readings `poll` apart -- a short poll with a few readings
    instead of two 5-second ones, which was ~10 s of every point's ~20.
    """
    t0, last, same = time.time(), None, 0
    while time.time() - t0 < timeout:
        w = traces_written(trace_dir, since_ts)
        sig = tuple((x["path"], x["bytes"]) for x in w)
        same = same + 1 if (sig == last and w) else 0
        if w and len(w) >= max(1, expect_files) and same >= stable_reads - 1:
            journal.append({"event": "window_closed", "files": len(w),
                            "waited_s": round(time.time() - t0, 1)})
            return True
        last = sig
        time.sleep(poll)
    journal.append({"event": "window_close_TIMEOUT", "waited_s": round(time.time() - t0, 1)})
    return False


def traces_written(trace_dir, since_ts):
    if FRAMEWORK == "vllm":
        # vLLM's per-session prefix changes names, not its configured output directory.
        tag = os.path.basename(trace_dir)
        for src in glob.glob(os.path.join(os.path.dirname(trace_dir), tag + "*.trace.json*")):
            dst = os.path.join(trace_dir, os.path.basename(src))
            suffix = os.path.basename(src)[len(tag):]
            if suffix and suffix[0].isdigit():
                continue
            if os.path.isfile(src) and os.path.getmtime(src) >= since_ts and not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)
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
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, help="journal JSON for this point")
    ap.add_argument("--framework", choices=["sglang", "vllm"], default="sglang")
    ap.add_argument("--served-model")
    ap.add_argument("--iteration-log-dir")
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--expect-files", type=int, default=1)
    args = ap.parse_args()
    global FRAMEWORK, SERVED_MODEL, ITERATION_LOG_DIR, ID_HI
    FRAMEWORK, SERVED_MODEL = args.framework, args.served_model
    ITERATION_LOG_DIR = args.iteration_log_dir
    if FRAMEWORK == "vllm" and not SERVED_MODEL:
        ap.error("--served-model is required for vLLM")
    if args.vocab_size is not None and args.vocab_size < 1:
        ap.error("--vocab-size must be positive")
    ID_HI = min(150000, args.vocab_size - 1) if args.vocab_size else 31999
    os.makedirs(args.trace_dir, exist_ok=True)

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
    verify = {}
    ex = None

    try:
        if phase == "DECODE":
            # Steady state FIRST (rule 1): put the batch on, confirm the scheduler has
            # admitted it and is generating, and only then pay for the arm.
            ex = drive_decode(args.base_url, point, args, journal)
            want = int(point["batch_size"])
            confirmed, obs_bs, _ = wait_for_occupancy(
                args.base_url, want, journal, args.steady_timeout)
            if not confirmed:
                raise RuntimeError(
                    "steady state not reached: wanted bs=%s, scheduler reported %s. Not "
                    "arming." % (want, obs_bs))
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout)
            # The scheduler closes the window itself once num_steps forwards have run;
            # the batch is already resident, so the steps are the intended shape.
            if FRAMEWORK == "vllm":
                if not wait_traces_settled(args.trace_dir, t_start, journal,
                                           args.window_timeout, args.expect_files):
                    raise RuntimeError("vLLM profile window did not produce the expected traces")
            else:
                deadline = time.time() + args.window_timeout
                while time.time() < deadline:
                    time.sleep(5)
                    metrics, _ = scrape_metrics(args.base_url)
                    bs = metrics.get("num_running_reqs")
                    if bs == 0:
                        journal.append({"event": "batch_drained", "note": "requests finished "
                                        "before the window closed; shorten --decode-tokens "
                                        "or lower --num-steps"})
                        break
        else:
            # Prefill is a single forward per request: arm first, then send, because
            # there is no resident state to confirm. The window is still bounded by
            # num_steps and still closed explicitly.
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout)
            drive_prefill(args.base_url, point, args, journal)
        ok = True
    except Exception as exc:
        journal.append({"event": "probe_failed", "error": "%s: %s"
                        % (type(exc).__name__, exc)})
    finally:
        # Rule 3: always, even on the failure path.
        stop_profile(args.base_url, journal)
        if ex is not None:
            ex.shutdown(wait=False)

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
