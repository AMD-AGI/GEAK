#!/usr/bin/env python3
"""Phase 3.0: the 单侧 (isolated) gate for kernel-fusion candidates.

Deterministic validator of the per-candidate microbench VERDICTS produced by the
fusion_unit_validator role. It does NOT run kernels and claims no perf of its own —
it VALIDATES that each verdict is trustworthy (well-formed + provenance-consistent
with the Phase 2.1 candidate it claims to test) and derives the gate:

  unit_side_status ∈ {pass, fail, blocked}

  * pass    — parity==pass AND isolated_speedup > 1+margin AND (for a collective) the
              fused path actually engaged. This fusion is eligible for apply-back.
  * fail    — the microbench ran but the fusion is not a win (parity failed, or no
              speedup). Do NOT apply back.
  * blocked — a collective whose fused path did NOT engage at this shape (size-guard
              fallback to split). Not a fail — it matches the Top-K non-actionable
              verdict; there is simply nothing to apply at this shape.

A verdict that is malformed, references an unknown candidate, tests a DIFFERENT shape
than the candidate captured, or names a fused_fn that is not the candidate's existing
API is an ERROR (untrustworthy → the harness FAILS so it is re-run) — never silently a
pass. This is the anti-cheat that keeps 单侧 honest, mirroring the provenance checks in
fusion_candidate_harness.py.

Usage:
  python3 fusion_unitside_harness.py --candidates fusion_candidates.json \
      --verdicts <dir-of-*.json | combined.json> \
      --out-md FUSION_UNITSIDE.md --out-json fusion_unitside.json \
      [--min-speedup 1.0]
"""
import argparse
import glob
import json
import os
import sys


REQUIRED_VERDICT_FIELDS = (
    "candidate_id", "parity", "isolated_speedup", "tested_shape", "fused_fn")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _load_verdicts(path):
    """A directory of *.json (one verdict each) OR a combined json ({verdicts:[...]}
    or a bare list)."""
    if os.path.isdir(path):
        out = []
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            out.append(_load(fp))
        return out
    data = _load(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("verdicts") or [data]
    return []


def _norm(text):
    """Normalize an API/fn name for a robust contains-match: lowercase, keep only
    alnum + underscore runs (drops the ' (--flag)' suffixes, '::', 'aiter ' prefix)."""
    text = str(text or "").lower()
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch == "_") else " ")
    return [tok for tok in "".join(keep).split() if tok]


def _fn_matches_api(fused_fn, existing_apis):
    """The verdict's fused_fn must be one of the candidate's declared existing APIs
    (the microbench cannot claim a win for a kernel the candidate never cited)."""
    fn_toks = set(_norm(fused_fn))
    if not fn_toks:
        return False
    for api in existing_apis or []:
        api_toks = set(_norm(api.get("name")))
        if not api_toks:
            continue
        # the fused_fn's distinctive tokens (drop the generic 'aiter') are a subset of
        # the API name's tokens, or vice-versa — either direction is a match.
        core = fn_toks - {"aiter"}
        acore = api_toks - {"aiter"}
        if core and (core <= api_toks or acore <= fn_toks):
            return True
    return False


def _candidate_shapes(candidate):
    """The set of captured member input-dim rows (as tuples) the candidate legitimately
    operates on. A verdict must have tested ONE of these."""
    shapes = set()
    for member in candidate.get("members", []) or []:
        dims = ((member.get("shape") or {}).get("input_dims")) or []
        for row in dims:
            if isinstance(row, list) and row:
                shapes.add(tuple(int(x) for x in row))
    return shapes


def _tested_shape_tuple(tested_shape):
    """Accept [tokens, hidden] or [[..],[..]]; return the primary 2-D row as a tuple,
    or None if unparseable."""
    if not isinstance(tested_shape, list) or not tested_shape:
        return None
    first = tested_shape[0]
    row = first if isinstance(first, list) else tested_shape
    try:
        return tuple(int(x) for x in row)
    except (TypeError, ValueError):
        return None


def validate(candidates_path, verdicts_path, min_speedup=1.0):
    payload = _load(candidates_path)
    candidates = {c["candidate_id"]: c
                  for c in payload.get("candidates", [])}
    verdicts = _load_verdicts(verdicts_path)

    errors = []
    results = []
    for index, verdict in enumerate(verdicts):
        vp = "verdict[%d]" % index
        cid = verdict.get("candidate_id")
        # 1. well-formed
        missing = [f for f in REQUIRED_VERDICT_FIELDS if verdict.get(f) is None]
        if missing:
            errors.append("%s missing fields %s" % (vp, missing))
            continue
        vp = "verdict[%s]" % cid
        # 2. known candidate (provenance)
        candidate = candidates.get(cid)
        if candidate is None:
            errors.append("%s references unknown candidate_id" % vp)
            continue
        family = str(candidate.get("family") or "")
        # 3. fused_fn is one of the candidate's declared APIs (anti-cheat)
        if not _fn_matches_api(verdict.get("fused_fn"),
                               candidate.get("existing_apis")):
            errors.append(
                "%s fused_fn '%s' is not one of the candidate's existing_apis %s"
                % (vp, verdict.get("fused_fn"),
                   [a.get("name") for a in candidate.get("existing_apis", [])]))
            continue
        # 4. shape provenance — the microbench must have tested the shape the candidate
        #    actually ran on (a pass on a different shape is meaningless). Two cases:
        #    (a) members carry exact captured input_dims (e.g. prefill kernel_exact):
        #        tested_shape MUST be one of them (strict).
        #    (b) members carry no dims (e.g. decode runtime_probe_wrapper): fall back to
        #        the selected_bucket token count — tested_shape's leading (token) dim
        #        MUST equal batch_size (decode) / input_tokens (prefill). This still
        #        stops a pass faked on a different token count.
        tested = _tested_shape_tuple(verdict.get("tested_shape"))
        cand_shapes = _candidate_shapes(candidate)
        if tested is None:
            errors.append("%s tested_shape %s is unparseable"
                          % (vp, verdict.get("tested_shape")))
            continue
        if cand_shapes:
            if tested not in cand_shapes:
                errors.append(
                    "%s tested_shape %s is not among the candidate's captured member "
                    "shapes %s (cannot trust a 单侧 pass on a different shape)"
                    % (vp, verdict.get("tested_shape"), sorted(cand_shapes)))
                continue
        else:
            bucket = candidate.get("selected_bucket") or {}
            expect_tok = (bucket.get("batch_size")
                          if candidate.get("phase") == "decode"
                          else bucket.get("input_tokens"))
            if expect_tok and int(tested[0]) != int(expect_tok):
                errors.append(
                    "%s tested_shape leading dim %d != selected_bucket token count %d "
                    "(cannot trust a 单侧 pass on a different token count)"
                    % (vp, tested[0], int(expect_tok)))
                continue
        # ---- verdict is TRUSTWORTHY; derive the gate from its content ------------
        try:
            speedup = float(verdict.get("isolated_speedup"))
        except (TypeError, ValueError):
            errors.append("%s isolated_speedup is not a number" % vp)
            continue
        parity = str(verdict.get("parity")).lower()
        is_collective = family.startswith("collective")
        engaged = verdict.get("engaged")
        status, reason = "pass", ""
        if is_collective and engaged is False:
            status = "blocked"
            reason = ("fused collective path did not engage at this shape "
                      "(size-guard fallback to split) — nothing to apply here")
        elif parity != "pass":
            status = "fail"
            reason = "parity != pass (fused output diverges from the split reference)"
        elif speedup <= min_speedup:
            status = "fail"
            reason = ("isolated_speedup %.3f <= %.3f — not a win, do not apply back"
                      % (speedup, min_speedup))
        else:
            reason = ("parity pass, isolated_speedup %.3fx%s"
                      % (speedup, ", engaged" if is_collective else ""))
        results.append({
            "candidate_id": cid,
            "family": family,
            "phase": candidate.get("phase"),
            "tier_hint": candidate.get("implementation_class"),
            "unit_side_status": status,
            "reason": reason,
            "parity": parity,
            "isolated_speedup": round(speedup, 4),
            "ref_ms": verdict.get("ref_ms"),
            "cand_ms": verdict.get("cand_ms"),
            "engaged": engaged,
            "tested_shape": verdict.get("tested_shape"),
            "fused_fn": verdict.get("fused_fn"),
            "tp": verdict.get("tp"),
            "tol": verdict.get("tol"),
        })

    counts = {"pass": 0, "fail": 0, "blocked": 0}
    for r in results:
        counts[r["unit_side_status"]] = counts.get(r["unit_side_status"], 0) + 1
    result = {
        "schema_version": 1,
        "phase": "unitside_gate",
        "status": "pass" if not errors else "fail",
        "min_speedup": min_speedup,
        "candidates_json": os.path.abspath(candidates_path),
        "errors": errors,
        "counts": counts,
        "verdict_count": len(verdicts),
        "results": results,
    }
    return result


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


def render_markdown(result):
    lines = ["# Kernel Fusion 单侧 Gate (Phase 3.0)", ""]
    lines.append(
        "隔离验证每个 fusion 的 **fused kernel vs split 参考**：正确性(parity) + "
        "isolated speedup。**`pass` 才可进 apply-back**；`blocked` = 该 shape 下 "
        "fused 路径未生效(size-guard 回退)，非失败；`fail` = 无收益/不正确。")
    lines.append("")
    c = result["counts"]
    lines.append("总计 %d 条 verdict：pass %d / fail %d / blocked %d；harness status=**%s**（错误 %d）。"
                 % (result["verdict_count"], c.get("pass", 0), c.get("fail", 0),
                    c.get("blocked", 0), result["status"], len(result["errors"])))
    lines.append("")
    lines.append("| 候选 | 阶段 | family | 单侧结论 | parity | isolated speedup | engaged | tested shape | fused fn | 说明 |")
    lines.append("|---|:--:|---|:--:|:--:|---:|:--:|---|---|---|")
    for r in result["results"]:
        sp = ("%.3fx" % r["isolated_speedup"]) if r["isolated_speedup"] is not None else "-"
        lines.append("| `%s` | %s | %s | **%s** | %s | %s | %s | %s | `%s` | %s |" % (
            _esc(r["candidate_id"]), _esc(r["phase"]), _esc(r["family"]),
            r["unit_side_status"], _esc(r["parity"]), sp, _esc(r["engaged"]),
            _esc(r["tested_shape"]), _esc(r["fused_fn"]), _esc(r["reason"])))
    lines.append("")
    if result["errors"]:
        lines.append("## 不可信 verdict（harness 错误，需按报错重跑，不得当作 pass）")
        lines.append("")
        for e in result["errors"]:
            lines.append("- %s" % e)
        lines.append("")
    lines.append(
        "说明：本 gate 只校验 microbench verdict 的**可信度**（字段完整 + 候选存在 + "
        "tested_shape 属于候选抓到的 member shape + fused_fn 属于候选 existing_apis + "
        "collective 是否真生效）并据此判 pass/fail/blocked；它不跑 kernel、不产生自己的 "
        "perf 数字。isolated speedup / parity 由 fusion_unit_validator 的隔离 microbench 实测。")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(candidates_path, verdicts_path, out_md, out_json, min_speedup=1.0):
    result = validate(candidates_path, verdicts_path, min_speedup)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verdicts", required=True,
                        help="dir of per-candidate *.json verdicts OR a combined json")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="isolated_speedup must exceed this for a pass (default 1.0)")
    args = parser.parse_args()
    result = run(args.candidates, args.verdicts, args.out_md, args.out_json,
                 args.min_speedup)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "errors": len(result["errors"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
