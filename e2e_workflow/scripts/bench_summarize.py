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
(input+output).  THREE interactivity axes exist and are NOT interchangeable:
E2E_METRIC=e2e_norm_intvty_p50 is the TTFT-inclusive median Hyperloom keeps AgentX candidates
on, E2E_METRIC=e2e_norm_intvty_p90 is the same family's P90 tail, which Hyperloom holds as a
guard, and E2E_METRIC=p90_intvty_inferencex is the decode-only axis the InferenceX pareto
publishes.  See the _BASES block below before using any of them.  Baseline and candidate read
the same key, so the accept RATIO is basis-consistent; metric_basis records which was used.
Values are aggregate, NOT divided by TP.

``throughput_tok_s_median`` carries whichever axis was selected, so on any interactivity axis
it is not a tok/s figure at all -- read it with metric_basis, never by its name alone.
"""
import argparse
import glob
import json
import os
import statistics
import sys

TOTAL_KEYS = ("total_token_throughput", "total_throughput", "total_token_throughput_tok_s")
OUTPUT_KEYS = ("output_throughput", "output_token_throughput", "output_throughput_tok_s")
#: THREE DIFFERENT INTERACTIVITY AXES, and they are NOT interchangeable. All are "tok/s/user",
#: all higher-is-better, and on the GLM-5.2-MXFP4 AgentX trace the two P90 axes read 81.9 and
#: 119.7 -- a factor of 1.46. Never compare a candidate read on one against a reference read on
#: another.
#:
#:   e2e_norm_intvty_p50: aiperf's summary P50 of the per-request rate OSL/E2EL_s, the
#:       median request's own token rate, i.e. InferenceX's ``p50_e2e_norm_intvty``. TTFT and
#:       queue wait are INSIDE the window. This is the axis Hyperloom keeps AgentX candidates
#:       on (``GRADED_INTVTY_P50``), so it is the one its handoff selects.
#:
#:   e2e_norm_intvty_p90 (also ``intvty``, ``interactivity``)  ->  1 / P90(E2EL/OSL):
#:       aiperf's summary P10 of the same per-request rate, the slow-tail request's. This is
#:       InferenceX's ``p90_e2e_norm_intvty``. Hyperloom holds it as a guard
#:       (``GRADED_INTVTY``) rather than keeping candidates on it. The two short tokens predate
#:       the P50 axis and are kept only for callers that pinned them: in InferenceX's own
#:       vocabulary a bare ``intvty`` is the decode-only axis below, so no new token is short.
#:
#:   p90_intvty_inferencex  ->  1000 / P90(ITL), decode only. TTFT and queue wait are OUTSIDE
#:       the window. This is InferenceX's ``p90_intvty`` -- the axis its public pareto plots
#:       as "P90 Interactivity (tok/s/user)" -- and it is derived here exactly as InferenceX's
#:       own ingestion derives it (``infx/results/fixed_sequence.py``: p90_tpot_ms is mapped
#:       through ``1000.0 / p90_tpot_ms``). Derived from a field the rows already carry, so it
#:       costs no extra measurement.
#:
#: Hyperloom computes both e2e_norm values; this script only selects which already-computed
#: field to read, so rows from a mapper that predates the P50 field have no P50 reading. The
#: handoff names the axis Hyperloom keeps candidates on, so GEAK's accept gate and Hyperloom's
#: KEEP gate read the same number.
INTVTY_P50_KEYS = ("e2e_norm_intvty_p50",)
INTVTY_KEYS = ("e2e_norm_intvty_p90",)
P90_INTVTY_KEYS = ("p90_tpot_ms", "tpot_p90_ms")

OUTPUT_BASIS = "aggregate_output_tok_s"
TOTAL_BASIS = "aggregate_total_token_tok_s"
INTVTY_P50_BASIS = "e2e_norm_intvty_p50"
INTVTY_BASIS = "e2e_norm_intvty_p90"
P90_INTVTY_BASIS = "p90_intvty_inferencex"


def _recip_ms_to_rate(ms):
    """P90 ITL in ms -> the InferenceX per-user token rate. Non-positive is no reading."""
    return 1000.0 / ms if ms > 0 else None


#: E2E_METRIC value -> (rows to read, metric_basis to record, per-row transform or None).
#: Spelled in Hyperloom's axis vocabulary so the handoff and this summary can be compared as
#: strings on both sides. Each interactivity axis carries its OWN basis string on purpose: a
#: summary read on one can then never be mistaken for another downstream.
_BASES = {
    "output": (OUTPUT_KEYS, OUTPUT_BASIS, None),
    "total": (TOTAL_KEYS, TOTAL_BASIS, None),
    "total_token": (TOTAL_KEYS, TOTAL_BASIS, None),
    "total_throughput": (TOTAL_KEYS, TOTAL_BASIS, None),
    "e2e_norm_intvty_p50": (INTVTY_P50_KEYS, INTVTY_P50_BASIS, None),
    "intvty": (INTVTY_KEYS, INTVTY_BASIS, None),
    "interactivity": (INTVTY_KEYS, INTVTY_BASIS, None),
    "e2e_norm_intvty_p90": (INTVTY_KEYS, INTVTY_BASIS, None),
    "p90_intvty_inferencex": (P90_INTVTY_KEYS, P90_INTVTY_BASIS, _recip_ms_to_rate),
}

#: Interactivity basis -> the guards Hyperloom's KEEP verdict holds beside it, as guard basis ->
#: rows to read. A median gain is kept only while the P90 tail and output throughput each stay
#: inside the noise band, so a candidate that buys the median by shedding either is not a win;
#: each guard is summarized from the same rows as the objective. Total token throughput is
#: measured, not guarded. The throughput bases carry none: there the objective IS the guard.
#: A guard lands as ``guard_<basis>_median``, so a handoff naming a guard by its basis (see
#: ``workload_spec.acceptance``) finds the field without a table of its own.
_GUARDS = {
    INTVTY_P50_BASIS: {INTVTY_BASIS: INTVTY_KEYS, OUTPUT_BASIS: OUTPUT_KEYS},
    INTVTY_BASIS: {OUTPUT_BASIS: OUTPUT_KEYS},
    P90_INTVTY_BASIS: {OUTPUT_BASIS: OUTPUT_KEYS},
}


def _basis():
    """``(keys, metric_basis, transform)`` for E2E_METRIC; an unknown axis is fatal.

    Falling back to output here would be the expensive failure: an orchestrator that asks for an
    axis this build cannot measure would get a summary labelled with the axis it did NOT request,
    and accept candidates graded against a reference read on another one.
    """
    raw = (os.environ.get("E2E_METRIC") or "output").strip().lower()
    try:
        return _BASES[raw]
    except KeyError:
        raise SystemExit("bench_summarize: E2E_METRIC=%r is not an axis this build measures "
                         "(known: %s)" % (raw, ", ".join(sorted(_BASES))))


def _num(d, *keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _axis_num(d, keys, transform):
    """The selected axis's value for one row, transformed into the graded units.

    Transforming per row rather than after the median keeps ``_spread_pct`` expressed in the
    units the accept gate compares; the median itself is unaffected either way, the transform
    being monotone.
    """
    v = _num(d, *keys)
    if v is None:
        return None
    return transform(v) if transform else v


def _med3(xs):
    return round(statistics.median(xs), 3) if xs else None


def _spread_pct(xs):
    """Max-min as a % of the median. 0.0 for a single sample: no spread, not unknown."""
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return round(100.0 * (max(xs) - min(xs)) / m, 2) if m else 0.0


def _guard_fields(guards):
    """``guard_<basis>_median`` and ``guard_<basis>_spread_pct`` for each guard's samples."""
    fields = {}
    for basis, xs in guards.items():
        fields["guard_" + basis + "_median"] = _med3(xs)
        fields["guard_" + basis + "_spread_pct"] = _spread_pct(xs)
    return fields


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
    keys, basis, transform = _basis()
    is_output, guard_keys = basis == OUTPUT_BASIS, _GUARDS.get(basis, {})

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
                    v = _axis_num(d, keys, transform)
                    if v is not None:
                        xs.append(v)
        except FileNotFoundError:
            pass
        return xs

    tps, ttft, tpot = [], [], []
    guards = {name: [] for name in guard_keys}
    with open(args.runs) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            v = _axis_num(d, keys, transform)
            if v is not None:
                tps.append(v)
            for name, gkeys in guard_keys.items():
                g = _num(d, *gkeys)
                if g is not None:
                    guards[name].append(g)
            for src, dst in ((("median_ttft_ms", "mean_ttft_ms"), ttft),
                             (("median_tpot_ms", "mean_tpot_ms"), tpot)):
                x = _num(d, *src)
                if x is not None:
                    dst.append(x)
    cold = read(args.cold) if args.cold else []
    med, spread = _med3(tps), _spread_pct(tps)
    summary = {
        # Canonical, metric-neutral throughput of the SELECTED basis. Downstream reads this +
        # metric_basis. The output_*-named pair below is a legacy alias, populated ONLY in output
        # mode so nobody silently reads total throughput under an "output" name.
        "throughput_tok_s_median": med,
        "throughput_tok_s_spread_pct": spread,
        "output_throughput_tok_s_median": med if is_output else None,
        "output_throughput_tok_s_spread_pct": spread if is_output else None,
        # The guards Hyperloom holds beside an interactivity objective (see _GUARDS); none on the
        # throughput bases, so those keep their exact field set.
        **_guard_fields(guards),
        "ttft_ms_median": _med3(ttft),
        "tpot_ms_median": _med3(tpot),
        "runs": len(tps),
        "all_throughput": tps,
        # Optional diagnostic cold round (BENCH_COLD_FINAL=1): one fresh-server round with
        # JIT/graph-capture costs included, same metric basis as the hot median. None by default.
        "cold_output_throughput_tok_s": _med3(cold),
        "cold_runs": len(cold),
        "metric_basis": basis,
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
            with open(os.path.join(rdir, "selected_attempt")) as fh:
                attempt = int(fh.read().strip())
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
    is_output = basis == OUTPUT_BASIS
    guards = {name: col("guard_" + name + "_median") for name in _GUARDS.get(basis, {})}
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
        **_guard_fields(guards),
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
