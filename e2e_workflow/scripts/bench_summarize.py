#!/usr/bin/env python3
"""Write bench_summary.json + the E2E_SUMMARY line for bench_e2e.sh.

Two input shapes, ONE acceptance contract:

  --from-runs      one server's per-round rows (bench_runs.jsonl).  Used by the legacy and
                   warm_server lifecycles.  With WARM_SERVER_ROUNDS set, a "round" is a sample.
  --from-replicas  one isolated_server leg's replica_*/selected_summary.json files, each already
                   summarized by a nested bench_e2e.sh run.  A "replica" is a sample.

Both emit status / requested_* / successful_* / usable_for_acceptance / observed_median, because
every caller (director:validate, the integrator A/B) reads one shape whichever lifecycle produced
the number.  Keeping the two emitters in one file is the point: when they lived in two heredocs
inside bench_e2e.sh, adding a contract field to one and not the other produced two different
bench_summary.json shapes with nothing to catch it.

Throughput basis: OUTPUT-only tok/s by default, matching the Hyperloom orchestrator's
baseline/explore collectors (they read output_throughput).  E2E_METRIC=total switches to total
(input+output).  Baseline and candidate read the same key, so the accept RATIO is basis-consistent;
metric_basis records which was used.  Values are aggregate, NOT divided by TP.
"""
import argparse
import glob
import json
import os
import statistics
import sys

TOTAL_KEYS = ("total_token_throughput", "total_throughput", "total_token_throughput_tok_s")
OUTPUT_KEYS = ("output_throughput", "output_token_throughput", "output_throughput_tok_s")


def _is_total():
    return (os.environ.get("E2E_METRIC") or "output").strip().lower() in (
        "total", "total_token", "total_throughput")


def _num(d, *keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _med3(xs):
    return round(statistics.median(xs), 3) if xs else None


def _spread_pct(xs):
    """Max-min as a % of the median. 0.0 for a single sample: no spread, not unknown."""
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return round(100.0 * (max(xs) - min(xs)) / m, 2) if m else 0.0


def _contract(requested, successful, observed, usable):
    """The fields every caller gates on, spelled the same way in both modes."""
    return {
        "requested_replicas": requested,
        "successful_replicas": successful,
        "status": "complete" if successful == requested and requested > 0 else "incomplete",
        "usable_for_acceptance": usable,
        "observed_median": observed,
    }


def _emit(summary, out_path, tail):
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"E2E_SUMMARY {summary.get('metric_basis') or 'unknown'}="
          f"{summary['throughput_tok_s_median']} "
          f"spread={summary['throughput_tok_s_spread_pct']}% " + tail)


def from_runs(args):
    keys = TOTAL_KEYS if _is_total() else OUTPUT_KEYS

    def read(path):
        xs = []
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    v = _num(d, *keys)
                    if v is not None:
                        xs.append(v)
        except FileNotFoundError:
            pass
        return xs

    tps, ttft, tpot = [], [], []
    with open(args.runs) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            v = _num(d, *keys)
            if v is not None:
                tps.append(v)
            for src, dst in ((("median_ttft_ms", "mean_ttft_ms"), ttft),
                             (("median_tpot_ms", "mean_tpot_ms"), tpot)):
                x = _num(d, *src)
                if x is not None:
                    dst.append(x)
    cold = read(args.cold) if args.cold else []
    med, spread = _med3(tps), _spread_pct(tps)
    total = _is_total()
    summary = {
        # Canonical, metric-neutral throughput of the SELECTED basis. Downstream reads this +
        # metric_basis. The output_*-named pair below is a legacy alias, populated ONLY in output
        # mode so nobody silently reads total throughput under an "output" name.
        "throughput_tok_s_median": med,
        "throughput_tok_s_spread_pct": spread,
        "output_throughput_tok_s_median": None if total else med,
        "output_throughput_tok_s_spread_pct": None if total else spread,
        "ttft_ms_median": _med3(ttft),
        "tpot_ms_median": _med3(tpot),
        "runs": len(tps),
        "all_throughput": tps,
        # Optional diagnostic cold round (BENCH_COLD_FINAL=1): one fresh-server round with
        # JIT/graph-capture costs included, same metric basis as the hot median. None by default.
        "cold_output_throughput_tok_s": _med3(cold),
        "cold_runs": len(cold),
        "metric_basis": "aggregate_total_token_tok_s" if total else "aggregate_output_tok_s",
        "measurement_mode": ("isolated_server_replica"
                             if os.environ.get("GEAK_ISOLATED_REPLICA") == "1"
                             else "legacy_same_server"),
        "effective_config_digest": os.environ.get("EFFECTIVE_CONFIG_DIGEST") or None,
    }
    tail = (f"ttft_ms={summary['ttft_ms_median']} tpot_ms={summary['tpot_ms_median']} "
            f"runs={summary['runs']} ")
    warm = os.environ.get("WARM_SERVER_ROUNDS")
    if warm:
        try:
            requested = int(warm)
        except ValueError:
            requested = 0
        summary["measurement_mode"] = "warm_server"
        summary["measurement_purpose"] = os.environ.get("MEASUREMENT_PURPOSE") or None
        summary.update(_contract(requested, len(tps), med,
                                 bool(len(tps) == requested and requested > 0 and med)))
        # A round and a replica are both "one sample" to the contract, but only these names say
        # which one this leg actually took.
        summary["requested_rounds"] = requested
        summary["successful_rounds"] = len(tps)
        # Samples from ONE server bound client noise; boot-to-boot variance needs isolated_server.
        summary["dispersion_basis"] = "within_server_rounds"
        tail += (f"status={summary['status']} "
                 f"usable_for_acceptance={str(summary['usable_for_acceptance']).lower()} ")
    _emit(summary, args.out, tail + f"measurement_mode={summary['measurement_mode']}")


def from_replicas(args):
    summaries, replicas = [], []
    for path in sorted(glob.glob(os.path.join(args.dir, "replica_*", "selected_summary.json"))):
        rdir = os.path.dirname(path)
        index = int(os.path.basename(rdir).split("_")[-1])
        if index > args.requested:      # a stale replica dir from a longer previous run
            continue
        with open(path) as fh:
            summaries.append(json.load(fh))
        try:
            attempt = int(open(os.path.join(rdir, "selected_attempt")).read().strip())
        except (OSError, ValueError):
            attempt = None
        replicas.append({"replica": index, "attempt": attempt,
                         "throughput_tok_s": summaries[-1].get("throughput_tok_s_median")})

    def col(key):
        return [float(s[key]) for s in summaries
                if isinstance(s.get(key), (int, float)) and not isinstance(s.get(key), bool)]

    tps = col("throughput_tok_s_median")
    med, spread = _med3(tps), _spread_pct(tps)
    bases = {s.get("metric_basis") for s in summaries if s.get("metric_basis")}
    basis = next(iter(bases)) if len(bases) == 1 else None
    is_output = basis == "aggregate_output_tok_s"
    summary = {
        "requested": args.requested,
        "successful": args.successful,
        **_contract(args.requested, args.successful, med,
                    args.successful == args.requested and med is not None),
        "measurement_mode": "isolated_server",
        "measurement_purpose": args.purpose,
        "effective_config_digest": args.digest or None,
        "throughput_tok_s_median": med,
        "throughput_tok_s_spread_pct": spread,
        "output_throughput_tok_s_median": med if is_output else None,
        "output_throughput_tok_s_spread_pct": spread if is_output else None,
        "ttft_ms_median": _med3(col("ttft_ms_median")),
        "tpot_ms_median": _med3(col("tpot_ms_median")),
        "runs": args.successful,
        "all_throughput": tps,
        "metric_basis": basis,
        "replicas": replicas,
    }
    _emit(summary, os.path.join(args.dir, "bench_summary.json"),
          f"requested={args.requested} successful={args.successful} "
          f"status={summary['status']} "
          f"usable_for_acceptance={str(summary['usable_for_acceptance']).lower()} "
          "measurement_mode=isolated_server")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("from-runs")
    r.add_argument("runs"); r.add_argument("out"); r.add_argument("cold", nargs="?")
    r.set_defaults(fn=from_runs)
    p = sub.add_parser("from-replicas")
    p.add_argument("dir"); p.add_argument("requested", type=int)
    p.add_argument("successful", type=int); p.add_argument("purpose")
    p.add_argument("digest", nargs="?", default="")
    p.set_defaults(fn=from_replicas)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
