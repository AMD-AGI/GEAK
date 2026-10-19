#!/usr/bin/env python3
"""Serving log + config  ->  the observed phase-instance distribution of a measured leg.

This is step one of turning a completed AgentX baseline into profile sampling points.
It answers "what did this deployment actually execute", in instances, from the log of
the run being attributed -- not from a parametric ISL/OSL model.

Why this is not optional
------------------------
The alternative is `parse_profile.analytic_regime_calls`, which assumes every request
has the same length and decode runs OSL steps. On a conversational agentic corpus that
is wrong by whole multiples: it cannot see prefix-cache hits (routinely >90% of prefill
tokens), speculative accept length, or the batch-size distribution. Every downstream
number -- which shapes to sample, what mass each sample stands for, what coverage a
unit test achieves -- inherits that error. So the counts come from the log.

What it produces (`observed_workload.json`)
-------------------------------------------
  prefill   one record per logged batch (they are logged 1:1), plus histograms over
            new-token count, cached-token count, batch composition and occupancy.
  decode    decode batches are logged once every `decode_log_interval` iterations, so
            each row stands for many iterations. The expansion is reconstructed and
            then CROSS-CHECKED against an independent quantity -- completed output
            tokens divided by mean accept length. Both numbers are reported with their
            disagreement; a reader who does not believe the expansion can see the check
            rather than take it on faith.
  joint     the (phase, shape, occupancy) distribution `plan_sampling_points.py` samples.

Dialects
--------
Scheduler log formats differ between stacks and between versions of the same stack. A
regex that matches none of them yields an empty distribution, which looks exactly like
a quiet leg. So the formats live in a registry and a log that matches nothing is an
error, not an empty result.

Stdlib only.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from collections import Counter

# --------------------------------------------------------------------------- #
# Log dialects.
#
# Each entry parses ONE scheduler log line into a normalized record. Add a dialect
# here when a new stack or version shows up; do not loosen an existing regex until
# it matches two formats, because then it reports fields from the wrong one.
#
# Normalized prefill record:
#   rank, ts, new_seq, new_token, cached_token, running, queue, cuda_graph
# Normalized decode record:
#   rank, ts, running, token, accept_len, gen_tps, cuda_graph, queue
# --------------------------------------------------------------------------- #

SGLANG_PREFILL_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?) (?:TP|DP)(?P<rank>\d+)[^\]]*\] Prefill batch[,.] "
    r"#new-seq: (?P<new_seq>\d+), #new-token: (?P<new_token>\d+), "
    r"#cached-token: (?P<cached_token>\d+), "
    r"(?:token usage: (?P<usage>[\d.]+), )?"
    r"#running-req: (?P<running>\d+), #queue-req: (?P<queue>\d+)"
    r"(?:, #pending-token: (?P<pending>\d+))?"
    r"(?:, cuda graph: (?P<graph>\w+))?"
)

SGLANG_DECODE_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?) (?:TP|DP)(?P<rank>\d+)[^\]]*\] Decode batch[,.] "
    r"#running-req: (?P<running>\d+), #token: (?P<token>\d+), "
    r"(?:token usage: (?P<usage>[\d.]+), )?"
    r"(?:accept len: (?P<accept_len>[\d.]+), )?"
    r"(?:accept rate: (?P<accept_rate>[\d.]+), )?"
    r"(?:cuda graph: (?P<graph>\w+), )?"
    r"gen throughput \(token/s\): (?P<gen_tps>[\d.]+), #queue-req: (?P<queue>\d+)"
)


def _ts(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError("unparsable timestamp %r" % s)


def _b(s):
    return str(s).lower() == "true" if s is not None else None


def _sglang_prefill(line):
    m = SGLANG_PREFILL_RE.match(line)
    if not m:
        return None
    return dict(kind="prefill", rank=int(m.group("rank")), ts=_ts(m.group("ts")),
                new_seq=int(m.group("new_seq")), new_token=int(m.group("new_token")),
                cached_token=int(m.group("cached_token")),
                running=int(m.group("running")), queue=int(m.group("queue")),
                cuda_graph=_b(m.group("graph")))


def _sglang_decode(line):
    m = SGLANG_DECODE_RE.match(line)
    if not m:
        return None
    acc = m.group("accept_len")
    return dict(kind="decode", rank=int(m.group("rank")), ts=_ts(m.group("ts")),
                running=int(m.group("running")), token=int(m.group("token")),
                accept_len=float(acc) if acc else 1.0,
                gen_tps=float(m.group("gen_tps")), queue=int(m.group("queue")),
                cuda_graph=_b(m.group("graph")))


LOG_DIALECTS = [
    ("sglang-scheduler", [_sglang_prefill, _sglang_decode]),
]

# A line that is plausibly a scheduler batch line. Used only to tell "this log has
# batch lines we failed to parse" (an error) from "this log has none" (a fact).
BATCH_HINT_RE = re.compile(r"\](?:\s*)(?:Prefill|Decode) batch[,.]")


def parse_log(paths, rank=0, window=None):
    pre, dec, hints, parsed_any = [], [], 0, False
    unparsed = {}
    for path in paths:
        op = __import__("gzip").open if path.endswith(".gz") else open
        with op(path, "rb") as fh:
            for raw in fh:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if BATCH_HINT_RE.search(line):
                    hints += 1
                rec = None
                for _dn, parsers in LOG_DIALECTS:
                    for p in parsers:
                        try:
                            rec = p(line)
                        except ValueError:
                            rec = None
                        if rec:
                            break
                    if rec:
                        break
                if rec is None:
                    if BATCH_HINT_RE.search(line) and len(unparsed) < 5:
                        unparsed[line[:200]] = 1
                    continue
                parsed_any = True
                if rec["rank"] != rank:
                    continue
                if window and not (window[0] <= rec["ts"] <= window[1]):
                    continue
                (pre if rec["kind"] == "prefill" else dec).append(rec)

    if hints and not parsed_any:
        raise SystemExit(
            "observed_workload: found %d scheduler batch lines but no registered log "
            "dialect parsed any of them (%s). An empty distribution here is "
            "indistinguishable from an idle leg, so this is an error. Add a dialect to "
            "LOG_DIALECTS. Examples:\n  %s"
            % (hints, ", ".join(d for d, _ in LOG_DIALECTS),
               "\n  ".join(sorted(unparsed)))
        )
    return pre, dec


# --------------------------------------------------------------------------- #
# Decode iteration expansion.
# --------------------------------------------------------------------------- #
def expand_decode(rows, interval):
    """Attribute an iteration count to each logged decode row.

    The scheduler logs one row per `decode_log_interval` iterations, and resets the
    counter whenever the batch composition changes, so the row that closes a run
    covers the remainder rather than a full interval. Without per-row iteration
    counts in the log the honest estimator is `interval` per row; the row-level
    residual is bounded by one interval per decode run, and the whole-leg total is
    cross-checked below against completed output tokens.
    """
    out = []
    for i, d in enumerate(rows):
        prev = rows[i - 1] if i else None
        same_run = bool(prev) and prev["running"] == d["running"]
        out.append(dict(d, ts=d["ts"].isoformat(), iterations=interval,
                        token_delta=(d["token"] - prev["token"]) if same_run else None,
                        continues_run=same_run))
    return out


def crosscheck_decode(expanded, completed_output_tokens):
    """Two independent routes to the same quantity: request-steps executed.

      route A (log structure)  = SUM over rows of iterations x batch
      route B (work completed) = completed output tokens / mean accept length

    They share no term. Agreement is evidence the expansion is right; disagreement
    is reported rather than reconciled.
    """
    steps_a = sum(r["iterations"] * r["running"] for r in expanded)
    n = sum(r["iterations"] for r in expanded)
    wa = sum(r["iterations"] * r["accept_len"] for r in expanded)
    accept_mean = wa / n if n else 1.0
    out = {"route_a_request_steps_from_log_structure": steps_a,
           "iterations": n, "accept_len_mean_iteration_weighted": round(accept_mean, 6)}
    if completed_output_tokens:
        steps_b = completed_output_tokens / accept_mean if accept_mean else 0
        out.update(
            route_b_request_steps_from_output_tokens=round(steps_b, 1),
            completed_output_tokens=completed_output_tokens,
            disagreement_pct=round(100.0 * abs(steps_a - steps_b) / max(1.0, steps_b), 4),
        )
    else:
        out["route_b_request_steps_from_output_tokens"] = None
        out["note"] = ("no completed-output-token count supplied (--completed-output-tokens "
                       "or --baseline-json); the expansion is UNCHECKED")
    return out


# --------------------------------------------------------------------------- #
# Bucketing: the joint distribution sampling points are drawn from.
# --------------------------------------------------------------------------- #
def log_buckets(values, per_decade=4):
    """Log-spaced bucket edges. Prefill token counts span 1..chunk_size (4-5 decades)
    and are strongly right-skewed, so linear bins put ~everything in bin 0."""
    vals = [v for v in values if v > 0]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    n = max(1, int(math.ceil(per_decade * math.log10(max(hi / lo, 1.0000001)))))
    return [lo * (hi / lo) ** (i / n) for i in range(n + 1)]


def bucketize(value, edges):
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return max(0, len(edges) - 2)


def prefill_joint(prefills):
    """Prefill instances grouped by new-token magnitude, each carrying the occupancy
    and cache state it ran under -- all three change what the kernels do."""
    edges = log_buckets([p["new_token"] for p in prefills])
    groups = {}
    for p in prefills:
        b = bucketize(p["new_token"], edges) if edges else 0
        g = groups.setdefault(b, {"instances": 0, "new_token": [], "cached_token": [],
                                  "new_seq": Counter(), "running": Counter()})
        g["instances"] += 1
        g["new_token"].append(p["new_token"])
        g["cached_token"].append(p["cached_token"])
        g["new_seq"][p["new_seq"]] += 1
        g["running"][p["running"]] += 1
    out = []
    total = sum(g["instances"] for g in groups.values()) or 1
    for b in sorted(groups):
        g = groups[b]
        nt = sorted(g["new_token"])
        ct = sorted(g["cached_token"])
        out.append({
            "bucket": b,
            "instances": g["instances"],
            "instance_share_pct": round(100.0 * g["instances"] / total, 4),
            "new_token_min": nt[0], "new_token_median": nt[len(nt) // 2],
            "new_token_max": nt[-1],
            "new_token_sum": sum(nt),
            "cached_token_median": ct[len(ct) // 2],
            "new_seq_mode": g["new_seq"].most_common(1)[0][0],
            "running_mode": g["running"].most_common(1)[0][0],
            "running_hist": dict(sorted(g["running"].items())),
        })
    # token mass, not just instance count: one 32k chunk is not one 700-token chunk
    tok_total = sum(o["new_token_sum"] for o in out) or 1
    for o in out:
        o["new_token_mass_share_pct"] = round(100.0 * o["new_token_sum"] / tok_total, 4)
    return out


def decode_joint(expanded):
    """Decode iterations grouped by running batch size -- the only shape axis that
    varies under CUDA graph, since every batch is padded to a captured size."""
    g = {}
    for r in expanded:
        d = g.setdefault(r["running"], {"iterations": 0, "rows": 0,
                                        "accept": 0.0, "ctx": []})
        d["iterations"] += r["iterations"]
        d["rows"] += 1
        d["accept"] += r["iterations"] * r["accept_len"]
        if r["running"]:
            d["ctx"].append(r["token"] / r["running"])
    total = sum(d["iterations"] for d in g.values()) or 1
    out = []
    for bs in sorted(g):
        d = g[bs]
        ctx = sorted(d["ctx"])
        out.append({
            "batch_size": bs,
            "iterations": d["iterations"],
            "rows_logged": d["rows"],
            "iteration_share_pct": round(100.0 * d["iterations"] / total, 4),
            "accept_len_mean": round(d["accept"] / d["iterations"], 4) if d["iterations"] else None,
            "per_req_context_median": round(ctx[len(ctx) // 2], 1) if ctx else None,
            "per_req_context_min": round(ctx[0], 1) if ctx else None,
            "per_req_context_max": round(ctx[-1], 1) if ctx else None,
        })
    return out


# aiperf writes `Phase profiling (profiling) started | ...` and `... (profiling) complete`:
# capitalised, with the phase repeated in parentheses. The event must follow the name (or
# its parenthetical) directly, so `Phase profiling (profiling) sending complete` -- emitted
# when the send loop drains, long before the phase ends -- does not match.
PHASE_RE = re.compile(
    r"phase\s+['\"]?(?P<ph>\w+)['\"]?(?:\s*\([^)]*\))?\s+(?P<ev>started|complete)\b",
    re.IGNORECASE)
DATED_TS_RE = re.compile(r"(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d(?:\.\d+)?)")
# The same client log stamps its lines with the time of day only.
TIME_TS_RE = re.compile(r"\b(\d\d:\d\d:\d\d(?:\.\d+)?)\b")


def _serving_log_dates(paths):
    """Dates appearing in the serving log(s), to anchor a client log's time-only stamps."""
    dates = set()
    for path in paths:
        if path.endswith(".gz"):
            with __import__("gzip").open(path, "rb") as fh:
                for raw in fh:
                    m = DATED_TS_RE.search(raw.decode("utf-8", "replace"))
                    if m:
                        dates.add(m.group(1))
            continue
        with open(path, "rb") as fh:
            head = fh.read(1 << 20).decode("utf-8", "replace")
            dates.update(m[0] for m in DATED_TS_RE.findall(head))
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - (1 << 20)))
            tail = fh.read().decode("utf-8", "replace")
            dates.update(m[0] for m in DATED_TS_RE.findall(tail))
    return sorted(dates)


def read_window(args):
    if args.window_start and args.window_end:
        return (_ts(args.window_start), _ts(args.window_end))
    if args.bench_log:
        # The measured leg is delimited in the client's own log; using it keeps the
        # distribution scoped to the leg that produced the baseline number, excluding
        # warm-up and drain.
        found = {}
        for line in open(args.bench_log, "r", errors="replace"):
            m = PHASE_RE.search(line)
            if not m or m.group("ph").lower() != args.bench_phase.lower():
                continue
            dated = DATED_TS_RE.search(line)
            if dated:
                found[m.group("ev").lower()] = ("dated", f"{dated.group(1)} {dated.group(2)}")
                continue
            bare = TIME_TS_RE.search(line)
            if bare:
                found[m.group("ev").lower()] = ("time", bare.group(1))
        start, end = found.get("started"), found.get("complete")
        if not (start and end):
            raise SystemExit(
                "observed_workload: --bench-log given but phase %r start/complete not found "
                "in it. Pass --window-start/--window-end explicitly." % args.bench_phase)
        if start[0] == "dated" and end[0] == "dated":
            return (_ts(start[1]), _ts(end[1]))

        # Time-only stamps carry no date, so anchor them on the serving log's own dates.
        # A leg that crosses midnight ends on the following day, which is why `end` is
        # rolled forward rather than rejected.
        dates = _serving_log_dates(args.log)
        if not dates:
            raise SystemExit(
                "observed_workload: --bench-log phase lines carry a time of day but no date, "
                "and the serving log has no dated timestamp to anchor them to. Pass "
                "--window-start/--window-end explicitly.")
        if len(dates) > 1:
            # Guessing here would silently scope the distribution to the wrong day, which
            # looks exactly like a correct run with fewer batches. Refuse instead.
            raise SystemExit(
                "observed_workload: the client log's phase stamps carry a time of day but no "
                "date, and the serving log spans %d dates (%s), so the anchor is ambiguous. "
                "Pass --window-start/--window-end explicitly."
                % (len(dates), ", ".join(dates)))
        day = dates[0]
        a = _ts("%s %s" % (day, start[1])) if start[0] == "time" else _ts(start[1])
        b = _ts("%s %s" % (day, end[1])) if end[0] == "time" else _ts(end[1])
        if b < a:
            b += dt.timedelta(days=1)   # the leg crossed midnight
        return (a, b)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", nargs="+", required=True,
                    help="serving/scheduler log(s) of the measured run")
    ap.add_argument("--rank", type=int, default=0,
                    help="which rank's scheduler log to read; ranks log redundantly")
    ap.add_argument("--decode-log-interval", type=int, default=None,
                    help="server's decode_log_interval; read from --server-info if omitted")
    ap.add_argument("--server-info", default=None,
                    help="JSON from /get_server_info, for decode_log_interval and graph sizes")
    ap.add_argument("--bench-log", default=None,
                    help="client bench log delimiting the measured leg")
    ap.add_argument("--bench-phase", default="profiling")
    ap.add_argument("--window-start", default=None)
    ap.add_argument("--window-end", default=None)
    ap.add_argument("--completed-output-tokens", type=int, default=None,
                    help="output tokens the client completed in the leg; cross-checks "
                         "the decode expansion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    info = {}
    if args.server_info and os.path.exists(args.server_info):
        info = json.load(open(args.server_info))
    interval = args.decode_log_interval or info.get("decode_log_interval")
    if not interval:
        raise SystemExit(
            "observed_workload: decode_log_interval is unknown. Every decode row stands "
            "for that many iterations, so guessing it scales the whole decode mass. "
            "Pass --decode-log-interval or --server-info.")

    window = read_window(args)
    pre, dec = parse_log(args.log, rank=args.rank, window=window)
    if not pre and not dec:
        raise SystemExit(
            "observed_workload: 0 batch lines for rank %d%s. Check --rank and the window."
            % (args.rank, " in the window" if window else ""))

    expanded = expand_decode(dec, interval)
    check = crosscheck_decode(expanded, args.completed_output_tokens)

    new_tok = sum(p["new_token"] for p in pre)
    cached_tok = sum(p["cached_token"] for p in pre)

    out = {
        "source_logs": list(args.log),
        "rank_scope": args.rank,
        "measured_window": {
            "start": window[0].isoformat() if window else None,
            "end": window[1].isoformat() if window else None,
            "duration_s": (window[1] - window[0]).total_seconds() if window else None,
            "source": ("bench log phase %r" % args.bench_phase) if args.bench_log
                      else ("explicit" if window else "whole log"),
        },
        "decode_log_interval": interval,
        "counts": {
            "prefill_batches": len(pre),
            "decode_rows_logged": len(dec),
            "decode_iterations": sum(r["iterations"] for r in expanded),
        },
        "prefill": {
            "total_new_tokens": new_tok,
            "total_cached_tokens": cached_tok,
            "prefix_cache_hit_pct": round(100.0 * cached_tok / max(1, new_tok + cached_tok), 4),
            "cuda_graph_any": any(p["cuda_graph"] for p in pre),
            "buckets": prefill_joint(pre),
        },
        "decode": {
            "cuda_graph_all": all(p["cuda_graph"] for p in dec) if dec else None,
            "crosscheck": check,
            "buckets": decode_joint(expanded),
        },
        "raw_prefill_rows": [dict(p, ts=p["ts"].isoformat()) for p in pre],
        "raw_decode_rows": expanded,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)

    d = check.get("disagreement_pct")
    print("prefill batches %d (%d new tok, %.1f%% prefix-cache hit) | decode %d iterations"
          % (len(pre), new_tok, out["prefill"]["prefix_cache_hit_pct"],
             out["counts"]["decode_iterations"]))
    if d is None:
        print("!! decode expansion is UNCHECKED: pass --completed-output-tokens")
    else:
        print("decode expansion cross-check: %.2f%% disagreement between log structure "
              "and completed tokens" % d)
        if d > 5.0:
            print("!! >5%: the expansion and the completed work disagree materially. "
                  "Do not plan sampling points off this until it is understood.")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
