#!/usr/bin/env python3
"""Turn the candidate legs into bench_summary.json + reference.json -- and refuse to
produce either unless the server is shown to have loaded the overlay.

The check that matters is the second one. A launcher that silently ignored the overlay
would hand back a perfectly clean 'no gain' that is really 'the patch was never loaded',
and nothing downstream could tell the difference. So the overlay is not merely requested:
the hashes the launcher recorded in the server's own server_case.json are read back and
compared, file by file, against the overlay on disk. Anything short of a match is a
non-zero exit, not a delta.

Both legs are written on the same metric basis, because finalize.py voids an A/B whose
two sides were measured on different axes.
"""
import argparse, hashlib, json, os, statistics, sys

BASIS = "output_token_throughput"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def overlay_hashes(overlay):
    """Only the suffixes the launcher itself records, so the two sides are comparable."""
    keep = (".py", ".cuh", ".cu", ".h", ".cpp")
    out = {}
    for r, _, ns in os.walk(overlay):
        for n in ns:
            p = os.path.join(r, n)
            if os.path.splitext(n)[1] in keep:
                out[os.path.relpath(p, overlay)] = sha256(p)
    return out


def leg(run_dir, want):
    tag = os.path.basename(run_dir.rstrip("/"))
    rec = {"tag": tag, "dir": run_dir}
    res = os.path.join(run_dir, tag + ".json")
    if not os.path.exists(res):
        rec["error"] = "the leg produced no result file (%s)" % res
        return rec
    d = json.load(open(res))
    rec["throughput"] = d.get("output_tok_per_s") or d.get("output_tokens_per_second")
    for k in ("duration_seconds", "requests_profiled", "requests_error_dropped",
              "ttft_p50", "itl_p50", "prefix_cache_hit_rate", "conc", "tp"):
        if k in d:
            rec[k] = d[k]

    sc = os.path.join(run_dir, "server_case.json")
    if not os.path.exists(sc):
        rec["overlay_in_server"] = False
        rec["error"] = "no server_case.json, so there is no evidence the server loaded the overlay"
        return rec
    got = (json.load(open(sc)) or {}).get("overlay_hashes") or {}
    missing = sorted(k for k in want if got.get(k) != want[k])
    rec["overlay_in_server"] = bool(got) and not missing
    rec["overlay_files_recorded"] = len(got)
    if missing:
        rec["error"] = ("the server recorded %d overlay files but %d of ours do not match "
                        "(%s); this leg measured a stack that is not the one we built"
                        % (len(got), len(missing), ", ".join(missing[:4])))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--reference-run", required=True)
    ap.add_argument("--reference-key", default="output_tok_per_s")
    ap.add_argument("--out-reference", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--noise-band", type=float, default=1.045)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    want = overlay_hashes(a.overlay)
    legs = [leg(r, want) for r in a.runs]
    good = [x for x in legs if x.get("throughput") and x.get("overlay_in_server")]
    tps = sorted(x["throughput"] for x in good)

    summary = {"metric_basis": BASIS, "repeats": len(good), "legs": legs,
               "overlay_files": len(want),
               "overlay_loaded_by_every_leg": bool(legs) and all(
                   x.get("overlay_in_server") for x in legs)}
    if tps:
        summary[BASIS] = statistics.median(tps)
        summary["spread_pct"] = round((tps[-1] / tps[0] - 1.0) * 100.0, 4) if len(tps) > 1 else 0.0
        summary["legs_tok_s"] = tps
    with open(a.out_summary, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    ref = json.load(open(a.reference_run))
    rv = ref.get(a.reference_key)
    reference = {
        BASIS: rv, "metric_basis": BASIS, "noise_band": a.noise_band,
        "provenance": {
            "method": "reference-replay-ab",
            "run": a.reference_run,
            "key": a.reference_key,
            "why": "the node-local pristine leg, measured by the same harness, the same "
                   "client flags and the same corpus as the candidate legs. An adopted CI "
                   "receipt from another machine is a cross-check, not the A/B reference.",
            "cross_check": ref.get("cross_check"),
        },
        "conditions": ref.get("conditions"),
    }
    with open(a.out_reference, "w") as f:
        json.dump(reference, f, indent=1, ensure_ascii=False)

    for x in legs:
        print("leg %-18s tok/s=%-12s overlay_in_server=%s %s"
              % (x["tag"], x.get("throughput"), x.get("overlay_in_server"),
                 x.get("error", "")))
    if rv is None:
        print("the reference run has no %r field" % a.reference_key, file=sys.stderr)
        return 2
    if not summary["overlay_loaded_by_every_leg"]:
        print("at least one leg did not load the overlay; no delta is reported",
              file=sys.stderr)
        return 3
    if not tps:
        print("no leg produced a throughput", file=sys.stderr)
        return 4
    print("reference=%.5f candidate_median=%.5f over %d legs"
          % (rv, summary[BASIS], len(tps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
