#!/usr/bin/env python3
"""Time-domain rollups with explicit case/kernel coverage. No model corrections.

Pass every requested case, including unavailable rows, or provide --cases.
Partial per-kernel rows must provide covered_measured_us; never extrapolate their
reference over the unmodeled part of an operation.
"""
import argparse
import json
import math
from pathlib import Path
from counter_metrics import attribute_change


def _number(x, positive=False):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and (x > 0 if positive else x >= 0)


def _weight(row, key):
    value = row.get(key)
    if value is None:
        value = row.get("case_weight")
    if value is None:
        value = row.get("share_of_op_calls")
    value = 1.0 if value is None else float(value)
    if not _number(value):
        raise ValueError(f"invalid case weight: {value}")
    return value


def fold_kernels(per_kernel, eff_of, measured_us, missing="drop"):
    """Return covered reference time, kernel coverage, and missing entries.

For drop, matching measured time is measured_us * coverage. For legacy zero,
matching time remains measured_us. Never assign parent TL work to child kernels.
"""
    if missing not in ("drop", "zero") or not _number(measured_us, True):
        raise ValueError("invalid missing policy or measured time")
    total = covered = num = 0.
    unscored = []
    for entry in per_kernel:
        t = entry["ms"]
        if not _number(t):
            raise ValueError("invalid kernel duration")
        total += t
        eff = eff_of(entry)
        if not _number(eff):
            unscored.append((entry.get("kernel", "?"), t))
        else:
            num += t * eff
            covered += t
    if not total or not covered:
        return None, 0., unscored
    return measured_us * num / total, covered / total, unscored


def _key(row):
    return row["kernel"], row["regime"], row["case"]


def _keyed(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate case: {key}")
        result[key] = row
    return result


def with_manifest(rows, cases):
    found, requested = _keyed(rows), _keyed(cases)
    if found.keys() - requested.keys():
        raise ValueError("rows outside requested case manifest")
    result = []
    for key, case in requested.items():
        row = found.get(key)
        if row and _weight(row, "w") != _weight(case, "w"):
            raise ValueError(f"case weight differs from manifest: {key}")
        result.append({**case, **(row or {"unavailable_reason": "missing_case"}), '_request_manifest':True})
    return result


def fold_cases(rows, roof_key="roof_us", meas_key="measured_us", weight_key="w", cov_key="coverage", target_efficiency=None):
    if target_efficiency is not None and (not _number(target_efficiency, True) or target_efficiency > 1):
        raise ValueError("target efficiency must be in (0,1]")
    groups, seen = {}, set()
    for r in rows:
        key = r["kernel"], r["regime"]
        if r.get("case") is not None:
            identity = (*key, r["case"])
            if identity in seen:
                raise ValueError(f"duplicate case: {identity}")
            seen.add(identity)
        a = groups.setdefault(key, {"w_meas": 0., "w_roof": 0., "w_requested": 0., "w_scored": 0.,
            "n": 0, "n_total": 0, "n_unscored": 0, "missing_cases": [], "suspect_cases": [], "cov": [], "intervals": [],'manifest_verified':True})
        a['manifest_verified'] &= bool(r.get('_request_manifest'))
        w = _weight(r, weight_key)
        a["w_requested"] += w
        a["n_total"] += 1
        if not w:
            continue
        roof, measured = r.get(roof_key), r.get(meas_key)
        if r.get('unavailable_reason'):
            roof=None
        cov = r.get("kernel_coverage", r.get(cov_key))
        if cov is not None and (not _number(cov) or cov > 1):
            raise ValueError("kernel coverage must be in [0,1]")
        if cov is not None and cov < 1 and r.get("missing_policy", "drop") == "drop":
            measured = r.get("covered_measured_us")
        if not _number(roof) or not _number(measured, True):
            a["n_unscored"] += 1
            a["missing_cases"].append({"case": r.get("case"), "weight": w,
                "reason": r.get("unavailable_reason", "missing_or_invalid_reference_or_time")})
            continue
        a["n"] += 1
        a["w_scored"] += w
        a["w_meas"] += w * measured
        a["w_roof"] += w * roof
        if cov is not None:
            a["cov"].append(cov)
        if r.get("suspect") or roof > measured:
            a["suspect_cases"].append(r.get("case"))
        samples = r.get("measured_us_reps") or []
        a["intervals"].append((w * min(samples), w * max(samples)) if samples and cov in (None, 1)
                              and all(_number(t, True) for t in samples) else None)
    for a in groups.values():
        a["pct"] = a["w_roof"] / a["w_meas"] if a["w_meas"] else None
        a["case_coverage"] = a["w_scored"] / a["w_requested"] if a["w_requested"] else None
        a['case_coverage_basis']='requested_manifest' if a.pop('manifest_verified') else 'provided_rows_only; completeness not independently verified'
        a["kernel_coverage"] = min(a["cov"]) if a["cov"] else None
        a["coverage"] = a["kernel_coverage"]  # compatibility alias is kernel-only
        a["suspect"] = bool(a["suspect_cases"])
        a["distance_to_reference"] = 1 / a["pct"] if a["pct"] and not a["suspect"] else None
        a["status"] = ("UNAVAILABLE" if a["pct"] is None else "PARTIAL" if a["case_coverage"] < 1
            or (a["kernel_coverage"] is not None and a["kernel_coverage"] < 1) else "SUSPECT" if a["suspect"] else "OK")
        samples = a.pop("intervals")
        a.pop("cov")
        a["sample_envelope"] = ([a["w_roof"] / sum(x[1] for x in samples), a["w_roof"] / sum(x[0] for x in samples)]
                                if samples and all(x is not None for x in samples) else None)
        if target_efficiency is not None and a["status"] == "OK":
            if a['case_coverage_basis']!='requested_manifest':
                a['status']='UNVERIFIED_CASE_COVERAGE'
                continue
            lo, hi = a["sample_envelope"] or [a["pct"], a["pct"]]
            a["status"] = ("BORDERLINE" if lo < target_efficiency <= hi else
                "AT_OR_ABOVE_TARGET" if lo >= target_efficiency else "BELOW_TARGET")
    return groups


def fold_ratio(rows, num_key, den_key, weight_key="w"):
    groups = {}
    for r in rows:
        if not _number(r.get(num_key)) or not _number(r.get(den_key)):
            raise ValueError("missing/invalid ratio component")
        a = groups.setdefault((r["kernel"], r["regime"]), {"num": 0., "den": 0., "n": 0})
        w = _weight(r, weight_key)
        a["num"] += w * r[num_key]; a["den"] += w * r[den_key]; a["n"] += 1
    for a in groups.values():
        a["ratio"] = a["num"] / a["den"] if a["den"] else None
    return groups


def compare_cases(baseline, candidate, roof_key="roofline_us", allow_condition_mismatch=False):
    base, cand = _keyed(baseline), _keyed(candidate)
    aligned_b, aligned_c, missing = [], [], []
    for key in sorted(base.keys() | cand.keys()):
        x, y = base.get(key), cand.get(key)
        if x is None or y is None:
            missing.append(key)
            template = x or y
            empty = {k: template[k] for k in ("kernel", "regime", "case")}
            empty["w"] = _weight(template, "w")
            x, y = x or empty, y or empty
        if _weight(x, "w") != _weight(y, "w"):
            raise ValueError(f"different comparison weights: {key}")
        if x.get('measurement_contract') and y.get('measurement_contract') and x['measurement_contract']!=y['measurement_contract'] and not allow_condition_mismatch:
            raise ValueError(f'measurement conditions differ: {key}; use --allow-condition-mismatch only for labeled diagnostics')
        if not all(_number(r.get("measured_us"), True) and _number(r.get(roof_key)) for r in (x, y)):
            x, y = {**x, roof_key: None}, {**y, roof_key: None}
        aligned_b.append(x); aligned_c.append(y)
    b, c = fold_cases(aligned_b, roof_key=roof_key), fold_cases(aligned_c, roof_key=roof_key)
    result = {}
    for key in b:
        pairs = [(x, y) for x, y in zip(aligned_b, aligned_c) if (x['kernel'], x['regime']) == key and x.get(roof_key) is not None]
        latency_pairs=[(base[k],cand[k]) for k in base.keys() & cand.keys() if k[:2]==key
                       and _number(base[k].get('measured_us'),True) and _number(cand[k].get('measured_us'),True)]
        time_base=sum(_weight(x,'w')*x['measured_us'] for x,y in latency_pairs)
        time_candidate=sum(_weight(y,'w')*y['measured_us'] for x,y in latency_pairs)
        latency_weight=sum(_weight(x,'w') for x,y in latency_pairs)
        result['/'.join(key)] = {"baseline": b[key], "candidate": c[key],
            "comparison_status":'CONDITION_MISMATCH' if any(x.get('measurement_contract')!=y.get('measurement_contract') for x,y in latency_pairs)
                else 'MATCHED' if latency_pairs and all(x.get('measurement_contract') for x,y in latency_pairs) else 'UNVERIFIED_CONDITIONS',
            "ab_speedup": time_base / time_candidate if time_candidate else None,
            "latency_case_coverage":latency_weight / b[key]['w_requested'] if b[key]['w_requested'] else None,
            "tl_ratio_is_speedup": bool(pairs) and len(pairs)==len(latency_pairs)
                and all(math.isclose(x[roof_key], y[roof_key], rel_tol=1e-12) and
                        all(r.get('kernel_coverage',r.get('coverage')) in (None,1) for r in (x,y)) for x,y in pairs),
            "missing_arms": [k[2] for k in missing if k[:2] == key],
            "per_case_work_attribution": {x['case']:attribute_change(x,y) for x,y in latency_pairs
                                           if 'bytes' in x or 'bytes' in y}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', required=True, type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--allow-condition-mismatch',action='store_true')
    p.add_argument('--cases', type=Path, help='Full requested case manifest: list or {cases: [...]}')
    p.add_argument('--roof-key', default='roofline_us')
    p.add_argument('--label', default='roof')
    p.add_argument('--target-efficiency', type=float)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    rows = json.loads(args.rows.read_text())['rows']
    cases = json.loads(args.cases.read_text()) if args.cases else None
    cases = cases.get('cases') if isinstance(cases, dict) else cases
    if cases is not None:
        rows = with_manifest(rows, cases)
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())['rows']
        result = compare_cases(rows, with_manifest(candidate, cases) if cases is not None else candidate, args.roof_key,args.allow_condition_mismatch)
    else:
        result = {'/'.join(k): v for k, v in fold_cases(rows, roof_key=args.roof_key, target_efficiency=args.target_efficiency).items()}
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    print(text, end='')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
