"""Deterministic AVO lineage writer (the exact, error-prone parts of the commit gate).

The avo_workflow.js JS loop makes the commit *decision* (passesCommitGate). This
script performs the *precise data operations* an LLM should not hand-do:
  - compute the geomean speedup across per-case results,
  - build the rich, greppable commit message,
  - atomically read/modify/write lineage.json,
  - append attempts.jsonl / an evolution-log entry.

The role (lineage_engineer.md) still runs the git side-effects (apply/commit/tag)
in shell; this script owns the numbers + JSON so they are deterministic.

Subcommands
-----------
seed    --state DIR --per-case JSON [--baseline-geomean-ms F]
        Write the initial lineage.json with the v0 baseline node.

commit  --state DIR --version-n N --per-case JSON --strategy S
        [--rationale R] [--parent avo-vK] [--step I] [--ir-file PATH]
        Append version vN to lineage.json (best -> vN), append attempts.jsonl +
        evolution_log/step_*.json, and PRINT the commit message to stdout (so the
        role can pass it to `git commit -m` / `git tag -m`). Exits non-zero if the
        candidate's geomean does not exceed the current best (defense in depth;
        the JS gate should already guarantee this). Optional **--ir-file** merges
        parsed IR JSON into `score.ir` (ir_feedback audit + resume `best_ir`).

attempt --state DIR --step I [--worker W] [--strategy S] [--status ST]
        [--self-speedup F] [--verified F] [--patch-hash H] [--rationale R]
        [--committed false]
        Append ONE attempt (committed or not) to attempts.jsonl — the full search
        trajectory (every attempt incl. failures). Called by the variation engineer
        on every exit path so rejected/failed directions are durable (read by the
        supervisor + cmd_resume). Best-effort: always exits 0.

resume  --state DIR
        Print JSON for a resumed AVO run (best version, baseline, supervisor state).
        Includes **prior_steps** (max evolve step index) and **best_ir** (from the
        best lineage node's `score.ir`, or `{ok:false}`). See commit `--ir-file`.

geomean --per-case JSON
        Print the geomean speedup of a per-case list (utility / debugging).

`--per-case JSON` is a JSON array of {name, speedup[, baseline_ms, optimized_ms]}.
It may be a literal string or a path to a .json file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

# Top-level evolve dirs from avo_workflow.js: variation_XXXX[...], escalate_XXXX[...]
_STEP_DIR_RE = re.compile(r"^(?:variation|escalate)_(\d{4})")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _load_per_case(val: str) -> list[dict]:
    """Accept a JSON literal or a path to a .json file."""
    p = Path(val)
    text = p.read_text(encoding="utf-8") if p.exists() else val
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("per-case must be a JSON array")
    return data


def _case_weight(c: dict) -> float | None:
    for key in ("weight", "calls", "n_calls"):
        w = c.get(key)
        if isinstance(w, (int, float)) and w > 0:
            return float(w)
    return None


def _included_cases(per_case: list[dict]) -> list[dict]:
    return [c for c in per_case if isinstance(c, dict) and c.get("included") is not False]


def _geomean(per_case: list[dict]) -> float:
    """Score of per-case speedups: skip ``included: false``.

    When every included case that has a positive speedup also carries a weight
    (``weight`` / ``calls`` / ``n_calls``) plus ``baseline_ms`` / ``optimized_ms``,
    use the time-weighted ratio of sums. Otherwise fall back to an unweighted
    geometric mean of positive speedups.
    """
    cases = _included_cases(per_case)
    scored = []
    weighted = []
    for c in cases:
        su = c.get("speedup")
        if not isinstance(su, (int, float)) or su <= 0:
            continue
        scored.append(float(su))
        w = _case_weight(c)
        b, o = c.get("baseline_ms"), c.get("optimized_ms")
        if (w is not None
                and isinstance(b, (int, float)) and b > 0
                and isinstance(o, (int, float)) and o > 0):
            weighted.append((w, float(b), float(o)))
    if weighted and len(weighted) == len(scored):
        base = sum(w * b for w, b, _ in weighted)
        opt = sum(w * o for w, _, o in weighted)
        return (base / opt) if opt > 0 else 0.0
    if not scored:
        return 0.0
    return math.exp(sum(math.log(s) for s in scored) / len(scored))


def _geomean_of(per_case: list[dict], key: str) -> float | None:
    """Geometric mean of a per-case latency field (optimized_ms / baseline_ms)."""
    vals = [float(c[key]) for c in _included_cases(per_case)
            if isinstance(c.get(key), (int, float)) and c[key] > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Roofline model (mirror of lib/roofline.mjs — keep the two in sync).
#
# Turns per-case telemetry {flops, bytes, latency} into a hardware-anchored signal:
# what % of the MI300X roofline the kernel reaches and which ceiling binds it. The
# JS loop makes the steering/stop decision; this writer stores the numbers in the
# lineage so the history is auditable (and resumable). All fields degrade to None
# when telemetry is absent, so a kernel without FLOP/byte counts is unaffected.
# ---------------------------------------------------------------------------

# FLOP/s by dtype + HBM B/s. Source: kernel_knowledge/hardware/cdna3_mi300/peak_tables.md
_PEAKS = {
    "gfx942": {
        "label": "MI300X",
        "compute": {
            "fp64": 81.7e12, "fp64_matrix": 163.4e12, "fp32": 163.4e12, "tf32": 653.7e12,
            "fp16": 1307.4e12, "bf16": 1307.4e12, "fp8": 2614.9e12, "int8": 2614.9e12,
        },
        "hbm_bw": 5.325e12,
    },
    "gfx942_mi325": {
        "label": "MI325X",
        "compute": {
            "fp64": 81.7e12, "fp64_matrix": 163.4e12, "fp32": 163.4e12, "tf32": 653.7e12,
            "fp16": 1307.4e12, "bf16": 1307.4e12, "fp8": 2614.9e12, "int8": 2614.9e12,
        },
        "hbm_bw": 6.0e12,
    },
    # CDNA4 / MI355X (gfx950). Source: perf_knowledge/hardware/cdna4_mi350/peak_tables.md
    "gfx950": {
        "label": "MI355X",
        "compute": {
            "fp64": 78.6e12, "fp32": 157.3e12,
            "fp16": 2.5e15, "bf16": 2.5e15,
            "fp8": 5.0e15, "int8": 5.0e15,
            "fp6": 10.0e15, "fp4": 10.0e15,
            "mxfp8": 5.0e15, "mxfp6": 10.0e15, "mxfp4": 10.0e15,
        },
        "hbm_bw": 8.0e12,
    },
}
_PEAKS["gfx950_mi355"] = _PEAKS["gfx950"]
_TF = 1e12
_GB = 1e9


def _hw(name: str) -> dict:
    return _PEAKS.get(name) or _PEAKS["gfx942"]


def _peak_compute(dtype: str, hw: str) -> float:
    c = _hw(hw)["compute"]
    return c.get(str(dtype or "fp16").lower(), c["fp16"])


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def _roofline_case(c: dict, latency_key: str, hw: str,
                   basement: float = 0.35, near: float = 0.80,
                   frac_tol: float = 0.05) -> dict | None:
    """Roofline analysis for one case, or None if telemetry is insufficient.

    Mirrors lib/roofline.mjs: a fraction_of_roofline meaningfully above 1 (achieved
    beating the hardware ceiling) is physically impossible and signals untrustworthy
    flops/bytes; such a case is returned flagged ``telemetry_invalid`` with a null
    fraction/regime so it is excluded from the geomean/regime/early-stop signal.
    """
    latency_ms = _num(c.get(latency_key))
    flops = _num(c.get("flops"))
    # A2: prefer rocprof-MEASURED HBM bytes over the analytic estimate (mirror of
    # lib/roofline.mjs) — analytic bytes can mis-classify a cache-friendly kernel.
    bytes_ = _num(c.get("measured_bytes")) or _num(c.get("bytes"))
    if latency_ms is None or flops is None or bytes_ is None:
        return None
    dtype = c.get("dtype") or "fp16"
    pc = _peak_compute(dtype, hw)
    bw = _hw(hw)["hbm_bw"]
    latency_s = latency_ms / 1000.0
    ai = flops / bytes_
    achieved = flops / latency_s
    achieved_gbps = bytes_ / latency_s / _GB
    roofline = min(pc, ai * bw)
    fraction = achieved / roofline if roofline > 0 else None
    mem_frac = achieved_gbps / (bw / _GB)
    cmp_frac = achieved / pc
    if fraction is None or not math.isfinite(fraction) or fraction > 1 + frac_tol:
        return {
            "name": c.get("name"), "dtype": dtype,
            "ai": round(ai, 4), "achieved_tflops": round(achieved / _TF, 4),
            "achieved_gbps": round(achieved_gbps, 4),
            "roofline_tflops": round(roofline / _TF, 4),
            "fraction_of_roofline": None, "regime": "unknown", "near_roofline": False,
            "potential_speedup": None,
            "telemetry_invalid": bool(fraction is not None and math.isfinite(fraction)
                                      and fraction > 1 + frac_tol),
        }
    if max(mem_frac, cmp_frac) < basement:
        regime = "basement"
    else:
        regime = "compute-bound" if ai >= pc / bw else "memory-bound"
    return {
        "name": c.get("name"), "dtype": dtype,
        "bytes_source": "measured" if _num(c.get("measured_bytes")) else "analytic",
        "ai": round(ai, 4), "achieved_tflops": round(achieved / _TF, 4),
        "achieved_gbps": round(achieved_gbps, 4),
        "roofline_tflops": round(roofline / _TF, 4),
        "fraction_of_roofline": round(fraction, 6),
        "regime": regime, "near_roofline": bool(fraction >= near),
        "potential_speedup": round(roofline / achieved, 4) if achieved > 0 else None,
        "telemetry_invalid": False,
    }


def _roofline_summary(per_case: list[dict], latency_key: str, hw: str = "gfx942") -> dict:
    analyses = [a for a in (_roofline_case(c, latency_key, hw) for c in per_case) if a]
    valid = [a for a in analyses if a.get("fraction_of_roofline") is not None]
    n_invalid = sum(1 for a in analyses if a.get("telemetry_invalid"))
    if not valid:
        return {"ok": False, "geomean_fraction": None, "dominant_regime": "unknown",
                "worst_case": None, "per_shape": {}, "n_valid": 0, "n_invalid": n_invalid,
                "n_total": len(per_case)}
    fracs = [a["fraction_of_roofline"] for a in valid]
    geo = math.exp(sum(math.log(f) for f in fracs) / len(fracs)) if fracs else None
    worst = max(valid, key=lambda a: a.get("potential_speedup") or 0.0)
    per_shape = {a["name"]: {"fraction_of_roofline": a["fraction_of_roofline"],
                             "regime": a["regime"], "ai": a["ai"],
                             "achieved_tflops": a["achieved_tflops"]} for a in valid if a["name"]}
    return {
        "ok": True, "geomean_fraction": round(geo, 6) if geo is not None else None,
        "dominant_regime": worst["regime"], "worst_case": worst,
        "per_shape": per_shape, "n_valid": len(valid), "n_invalid": n_invalid,
        "n_total": len(per_case),
    }


def _per_shape(per_case: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in per_case:
        name = str(c.get("name", "")).strip()
        su = c.get("speedup")
        if name and isinstance(su, (int, float)):
            out[name] = round(float(su), 6)
    return out


def _write_heartbeat(state: Path, **fields) -> None:
    """Merge progress fields into avo_state/heartbeat.json (liveness / progress UI).

    Best-effort: a heartbeat write must never break seed/commit. Existing fields are
    preserved unless overwritten, so partial updates (e.g. step-only) are fine.
    """
    try:
        state.mkdir(parents=True, exist_ok=True)
        p = state / "heartbeat.json"
        cur = {}
        if p.exists():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cur = {}
        cur.update({k: v for k, v in fields.items() if v is not None})
        cur["ts"] = time.time()
        cur["iso"] = _now_iso()
        p.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def _read_lineage(state: Path) -> dict:
    p = state / "lineage.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"best_id": None, "active_best_id": None, "committed": []}


def _write_lineage(state: Path, data: dict) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / "lineage.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _best_speedup(data: dict) -> float:
    best = 1.0
    for n in data.get("committed", []):
        su = (n.get("score") or {}).get("speedup")
        if isinstance(su, (int, float)) and su > best:
            best = float(su)
    return best


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_seed(args) -> int:
    state = Path(args.state)
    per_case = _load_per_case(args.per_case) if args.per_case else []
    hw = getattr(args, "hw", None) or "gfx942"
    score = {
        "speedup": 1.0,
        "latency_ms": args.baseline_geomean_ms,
        "verified": True,
        "per_shape": {c.get("name", ""): 1.0 for c in per_case if c.get("name")},
    }
    # Baseline roofline (from baseline_ms telemetry, if the COMMANDMENT emits
    # flops/bytes): records where the ORIGINAL kernel sits vs the ceiling, so the
    # whole lineage's roofline progress is auditable from v0 onward.
    rl = _roofline_summary(per_case, latency_key="baseline_ms", hw=hw)
    if rl.get("ok"):
        score["roofline"] = {"geomean_fraction": rl["geomean_fraction"],
                             "regime": rl["dominant_regime"], "per_shape": rl["per_shape"]}
    data = {
        "best_id": "v0",
        "active_best_id": "v0",
        "hw": hw,
        "committed": [
            {
                "id": "v0", "parent_id": None, "patch": None, "git_ref": "avo-v0",
                "strategy": "baseline",
                "score": score,
                "committed_at": _now_iso(),
            }
        ],
    }
    _write_lineage(state, data)
    # Persist the absolute baseline so a later resume can restore it (lineage.json
    # only stores per-shape speedups, not the baseline ms).
    (state).mkdir(parents=True, exist_ok=True)
    (state / "baseline.json").write_text(json.dumps({
        "per_case": per_case, "geomean_ms": args.baseline_geomean_ms,
    }, indent=2), encoding="utf-8")
    _write_heartbeat(state, phase="seed", step=0, commits=0,
                     best_version="avo-v0", best_speedup=1.0)
    print("seeded avo-v0")
    return 0


def _attempt_signal(state: Path, max_items: int = 8) -> list[str]:
    """Distill recurring NON-committed strategies from attempts.jsonl into insights.

    So a resumed run (and the supervisor) sees which strategies have been tried and
    keep failing — even when no supervisor intervention fired — and avoids
    re-treading them. A strategy that was later committed is never reported as dead.
    """
    p = state / "attempts.jsonl"
    if not p.exists():
        return []
    tried: dict[str, int] = {}
    committed_strats: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = (d.get("strategy") or "").strip()
        if not s:
            continue
        if d.get("committed"):
            committed_strats.add(s)
        else:
            tried[s] = tried.get(s, 0) + 1
    out = []
    for s, n in sorted(tried.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        if s in committed_strats or n < 2:
            continue
        out.append(f"[attempts] '{s}' tried {n}x without a commit — change angle or avoid.")
    return out[:max_items]


def _supervisor_state(state: Path) -> dict:
    """Reconstruct failed-directions / insights / last direction from the audit log."""
    failed: list[str] = []
    insights: list[str] = []
    direction = {"name": "", "expected": "", "prompt": ""}
    log = state / "supervisor_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = (json.loads(line) or {}).get("directive") or {}
            except json.JSONDecodeError:
                continue
            for f in d.get("mark_failed") or []:
                if f and f not in failed:
                    failed.append(str(f))
            if d.get("diagnosis"):
                insights.append(f"[supervisor] {d['diagnosis']}")
            nds = d.get("new_directions") or []
            nd = next((x for x in nds if x.get("name") and x["name"] not in failed), None)
            if nd:
                direction = {"name": nd.get("name", ""), "expected": nd.get("expected", ""),
                             "prompt": nd.get("prompt", "")}
    return {"failed": failed, "insights": insights[-20:], "direction": direction}


def _max_completed_evolve_step(eval_dir: Path) -> int:
    """Largest step index N from variation_NNNN* / escalate_NNNN* directories.

    Team mode creates multiple dirs per logical step (e.g. variation_0060_0..2);
    counting directories would inflate ``prior_steps`` (e.g. 60 -> 174). The
    orchestrator uses ``padStart(4,'0')`` for NNNN — we parse the first four digits
    after the prefix and take the max.
    """
    if not eval_dir.is_dir():
        return 0
    max_s = 0
    for pat in ("variation_*", "escalate_*"):
        for p in eval_dir.glob(pat):
            if not p.is_dir():
                continue
            m = _STEP_DIR_RE.match(p.name)
            if m:
                max_s = max(max_s, int(m.group(1), 10))
    return max_s


def _load_optional_ir_file(path: str | None) -> dict | None:
    """Load IR JSON from a path (avo_workflow / variation_engineer §6b). Returns None if missing/invalid."""
    if not path or not str(path).strip():
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def cmd_resume(args) -> int:
    """Reconstruct the in-memory loop state from durable files for a resumed run."""
    state = Path(args.state)
    data = _read_lineage(state)
    nodes = data.get("committed", [])
    if not nodes:
        print(json.dumps({"resume": False}))
        return 0
    committed = [
        {"id": n.get("id"), "ref": n.get("git_ref"),
         "parent_id": n.get("parent_id"),
         "speedup": (n.get("score") or {}).get("speedup", 1.0), "strategy": n.get("strategy") or "",
         "regime": ((n.get("score") or {}).get("roofline") or {}).get("regime"),
         "per_case": (n.get("score") or {}).get("per_case", [])}
        for n in nodes
    ]
    best = max(committed, key=lambda c: c.get("speedup") or 0.0)
    best_node = next((n for n in nodes if n.get("id") == best["id"]), {})
    best_roofline = ((best_node.get("score") or {}).get("roofline") or {})
    base = {}
    bp = state / "baseline.json"
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))
    sup = _supervisor_state(state)
    # Failure memory survives a resume from TWO durable sources: supervisor
    # interventions (named dead directions) AND the full attempt ledger (recurring
    # strategies that never committed) — so even a run where no supervisor fired
    # does not re-tread dead strategies on resume.
    insights = (sup["insights"] + _attempt_signal(state))[-20:]
    eval_dir = state.parent
    prior_steps = _max_completed_evolve_step(eval_dir)
    best_ir = (best_node.get("score") or {}).get("ir")
    if not isinstance(best_ir, dict):
        best_ir = {"ok": False}
    print(json.dumps({
        "resume": True,
        "best_version": best["ref"], "best_id": best["id"],
        "cumulative": best["speedup"], "commits": max(0, len(committed) - 1),
        "roofline_fraction": best_roofline.get("geomean_fraction"),
        "roofline_regime": best_roofline.get("regime"),
        "committed": committed,
        "baseline_per_case": base.get("per_case", []),
        "baseline_geomean_ms": base.get("geomean_ms"),
        "failed": sup["failed"], "insights": insights, "direction": sup["direction"],
        "prior_steps": prior_steps,
        "best_ir": best_ir,
    }, default=str))
    return 0


def _compose_message(*, version_n: int, geomean: float, strategy: str, rationale: str,
                     parent: str, step: int, per_shape: dict[str, float],
                     latency_ms: float | None = None, baseline_latency_ms: float | None = None,
                     expected: str = "") -> str:
    git_ref = f"avo-v{version_n}"
    lat_s = f"{latency_ms:.4f}ms" if isinstance(latency_ms, (int, float)) else None
    # Subject (scannable in `git log --oneline`): avo-vN | <speedup> | <latency> | <strategy>
    subject = f"avo: {git_ref} | {geomean:.3f}x"
    if lat_s:
        subject += f" | {lat_s}"
    if strategy:
        subject += f" | {' '.join(strategy.split())}"
    if len(subject) > 100:
        subject = subject[:97] + "..."

    body = [f"speedup:  {geomean:.4f}x vs baseline (avo-v0)"]
    if isinstance(latency_ms, (int, float)):
        if isinstance(baseline_latency_ms, (int, float)) and baseline_latency_ms > 0:
            body.append(f"latency:  {latency_ms:.4f}ms (baseline {baseline_latency_ms:.4f}ms)")
        else:
            body.append(f"latency:  {latency_ms:.4f}ms")
    body.append(f"strategy: {strategy or '(unspecified)'}")
    if expected and expected.strip():
        body.append(f"  expected: {' '.join(expected.split())[:400]}")
    body.append(f"parent:   {parent or '(none)'}")
    body.append(f"step:     {step}")
    if per_shape:
        body.append("per-shape: " + "  ".join(f"{k}={v:.3f}x" for k, v in sorted(per_shape.items())))
    if rationale and rationale.strip():
        body.append("")
        body.append("rationale:")
        clip = rationale.strip()[:2000]
        body.extend("  " + ln for ln in clip.splitlines())
    return subject + "\n\n" + "\n".join(body) + "\n"


def cmd_commit(args) -> int:
    state = Path(args.state)
    per_case = _load_per_case(args.per_case)
    # The authoritative verified geomean is the one the deterministic commit gate
    # (avo_workflow.js -> commit_gate.mjs) already accepted: `commitVer.verified_geomean`,
    # which is what `cumulative` is advanced from. When the orchestrator passes it via
    # --verified-geomean we record THAT number, and only recompute the per-case
    # score as a fallback. Recomputing here can disagree with the gate when
    # per_case mixes a dominant improving case with a tiny regressing one (e.g. a
    # few-call decode_bs1 alongside a 1024-call decode_bs64): the unweighted mean would
    # spuriously refuse a candidate the gate already passed. per_case is still used
    # below for per_shape / latency / roofline display.
    verified_override = getattr(args, "verified_geomean", None)
    if isinstance(verified_override, (int, float)) and verified_override > 0:
        geomean = float(verified_override)
    else:
        geomean = _geomean(per_case)
    if geomean <= 0:
        print("error: no positive per-case speedups; nothing to commit", file=sys.stderr)
        return 2

    data = _read_lineage(state)
    best = _best_speedup(data)
    if geomean < best:  # defense in depth (JS gate should already ensure this)
        print(f"error: candidate geomean {geomean:.4f} < best {best:.4f}; refusing to commit", file=sys.stderr)
        return 3

    version_id = f"v{args.version_n}"
    git_ref = f"avo-{version_id}"
    parent = args.parent or (data.get("best_id") and f"avo-{data['best_id']}") or "avo-v0"
    parent_id = parent.replace("avo-", "") if parent else None
    per_shape = _per_shape(per_case)
    latency_ms = _geomean_of(per_case, "optimized_ms")          # candidate latency (geomean across cases)
    baseline_latency_ms = _geomean_of(per_case, "baseline_ms")  # baseline latency for context

    msg = _compose_message(
        version_n=args.version_n, geomean=geomean, strategy=args.strategy or "",
        rationale=args.rationale or "", parent=parent, step=args.step, per_shape=per_shape,
        latency_ms=latency_ms, baseline_latency_ms=baseline_latency_ms, expected=args.expected or "",
    )

    score = {"speedup": round(geomean, 6), "verified": True,
             "per_shape": per_shape, "per_case": per_case}
    if isinstance(latency_ms, (int, float)):
        score["latency_ms"] = round(latency_ms, 6)
    # Candidate roofline (from optimized_ms telemetry). Stored on the node so the
    # loop's start-point archive can bucket by regime and the report can show how
    # close the lineage got to the hardware ceiling. None when no flops/bytes given.
    hw = data.get("hw") or getattr(args, "hw", None) or "gfx942"
    rl = _roofline_summary(per_case, latency_key="optimized_ms", hw=hw)
    if rl.get("ok"):
        score["roofline"] = {"geomean_fraction": rl["geomean_fraction"],
                             "regime": rl["dominant_regime"], "per_shape": rl["per_shape"]}
    tag_ir = _load_optional_ir_file(getattr(args, "ir_file", None))
    if tag_ir is not None:
        score["ir"] = tag_ir
    node = {
        "id": version_id, "parent_id": parent_id, "patch": f"patches/{version_id}.patch",
        "git_ref": git_ref, "strategy": args.strategy or None,
        "expected": (args.expected or None), "score": score,
        "committed_at": _now_iso(),
    }
    data.setdefault("committed", []).append(node)
    data["best_id"] = version_id
    data["active_best_id"] = version_id
    _write_lineage(state, data)

    # attempts.jsonl (committed=true) + evolution log entry
    with open(state / "attempts.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "step_index": args.step, "version": version_id, "strategy": args.strategy,
            "verified_geomean": round(geomean, 6), "committed": True, "ts": time.time(),
        }) + "\n")
    elog = state / "evolution_log"
    elog.mkdir(parents=True, exist_ok=True)
    (elog / f"step_{args.step:04d}.json").write_text(json.dumps({
        "step_index": args.step, "strategy": args.strategy, "committed": True,
        "verified_speedup": round(geomean, 6), "per_shape": per_shape,
        "rationale": args.rationale or "",
    }, indent=2), encoding="utf-8")

    _write_heartbeat(state, phase="commit", step=args.step, commits=len(data["committed"]) - 1,
                     best_version=git_ref, best_speedup=round(geomean, 6),
                     best_latency_ms=(round(latency_ms, 6) if isinstance(latency_ms, (int, float)) else None))

    # The git side-effects (apply/commit/tag) are done by the role using THIS message.
    sys.stdout.write(msg)
    return 0


def cmd_heartbeat(args) -> int:
    """Update the progress heartbeat (called by the variation engineer per step)."""
    _write_heartbeat(
        Path(args.state), phase=args.phase or "variation",
        step=args.step, best_version=args.best or None,
        best_speedup=(args.speedup if args.speedup is not None else None),
        commits=(args.commits if args.commits is not None else None),
        note=args.note or None,
    )
    return 0


def cmd_attempt(args) -> int:
    """Append ONE attempt (usually NOT committed) to attempts.jsonl — the full
    search trajectory (L2 memory in avo_memory_design.md: "every attempt incl.
    failures").

    Called by the variation engineer at the end of EVERY step (operator=single and
    each operator=team worker), on every exit path, regardless of outcome — so
    rejected / failed / no-change directions are durable and visible to the
    supervisor (`supervisor.md` reads attempts.jsonl) and to a resumed run
    (`cmd_resume` → `_attempt_signal`). Committed candidates are additionally
    recorded with committed=true by `cmd_commit`; this records the attempt itself.

    Best-effort by contract: a logging failure must NEVER break the variation step,
    so any error is reported to stderr and the command still exits 0.
    """
    state = Path(args.state)
    committed = str(args.committed).strip().lower() in ("1", "true", "yes")
    try:
        state.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "step_index": args.step,
            "strategy": (args.strategy or None),
            "status": (args.status or None),
            "self_speedup": (round(args.self_speedup, 6) if args.self_speedup is not None else None),
            "verified_geomean": (round(args.verified, 6) if args.verified is not None else None),
            "committed": committed,
            "patch_hash": (args.patch_hash or None),
            "rationale": ((args.rationale or "").strip()[:500] or None),
        }
        if args.worker is not None:
            rec["worker"] = args.worker
        with open(state / "attempts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        sys.stderr.write(f"[avo_lineage attempt] non-fatal: {e}\n")
    return 0


def cmd_geomean(args) -> int:
    print(f"{_geomean(_load_per_case(args.per_case)):.6f}")
    return 0


def cmd_roofline(args) -> int:
    """Print the roofline summary JSON for a per-case telemetry list (utility/debug).

    Each case needs {flops, bytes, dtype?} plus a latency field (--latency-key,
    default optimized_ms). Cases missing telemetry are skipped; ok=false if none
    have it (relative-only mode).
    """
    per_case = _load_per_case(args.per_case)
    summary = _roofline_summary(per_case, latency_key=args.latency_key, hw=args.hw)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_context(args) -> int:
    """Print a bounded "prior implementations" block for the step prompt (§3.2).

    Surfaces the top-k committed versions (by speedup), excluding v0 and the
    current best, each with its verified speedup + per-shape + a truncated patch —
    so the agent can compare multiple prior implementations in one step instead of
    walking git by hand.
    """
    state = Path(args.state)
    p = state / "lineage.json"
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    best_id = data.get("best_id")
    pool = [
        n for n in data.get("committed", [])
        if n.get("id") != "v0" and n.get("id") != best_id and (n.get("score") or {}).get("speedup")
    ]
    pool.sort(key=lambda n: (n.get("score") or {}).get("speedup", 0.0), reverse=True)
    pool = pool[: max(0, args.k)]
    if not pool:
        return 0

    out = ["## Prior lineage implementations (compare — NOT your current base)"]
    out.append("Other committed versions in P_t. Compare approaches / per-shape behavior to "
               "decide your next move. Full source: `git -C <avo_repo> show avo-v{id}`.")
    for n in pool:
        score = n.get("score") or {}
        ps = score.get("per_shape") or {}
        ps_s = "  ".join(f"{k}={v:.3f}x" for k, v in sorted(ps.items())) if ps else ""
        head = f"### {n.get('git_ref') or n.get('id')} — {score.get('speedup', 0):.4f}x"
        if n.get("strategy"):
            head += f" via {n['strategy']}"
        if ps_s:
            head += f"  | per-shape: {ps_s}"
        out.append(head)
        diff = ""
        patch_rel = n.get("patch")
        if patch_rel:
            pf = state / patch_rel if not Path(patch_rel).is_absolute() else Path(patch_rel)
            if pf.exists():
                try:
                    diff = pf.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    diff = ""
        if len(diff) > args.max_diff:
            diff = diff[: args.max_diff] + "\n... [diff truncated; use `git show` for full] ...\n"
        out.append(f"```diff\n{diff}\n```" if diff else "(patch unavailable; use `git show`)")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic AVO lineage writer.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed")
    s.add_argument("--state", required=True)
    s.add_argument("--per-case", default="")
    s.add_argument("--baseline-geomean-ms", type=float, default=None)
    s.add_argument("--hw", default="gfx942")
    s.set_defaults(func=cmd_seed)

    c = sub.add_parser("commit")
    c.add_argument("--state", required=True)
    c.add_argument("--version-n", type=int, required=True)
    c.add_argument("--per-case", required=True)
    c.add_argument("--strategy", default="")
    c.add_argument("--expected", default="")
    c.add_argument("--rationale", default="")
    c.add_argument("--parent", default="")
    c.add_argument("--step", type=int, default=0)
    c.add_argument("--hw", default="gfx942")
    c.add_argument("--ir-file", default=None, help="Optional JSON path: IR vector for score.ir (ir_feedback / ir_probe parse output)")
    c.add_argument("--verified-geomean", type=float, default=None,
                   help="Authoritative verified geomean from the commit gate (commitVer.verified_geomean). "
                        "When set, recorded as the score instead of recomputing the unweighted per-case geomean.")
    c.set_defaults(func=cmd_commit)

    g = sub.add_parser("geomean")
    g.add_argument("--per-case", required=True)
    g.set_defaults(func=cmd_geomean)

    rf = sub.add_parser("roofline")
    rf.add_argument("--per-case", required=True)
    rf.add_argument("--latency-key", default="optimized_ms")
    rf.add_argument("--hw", default="gfx942")
    rf.set_defaults(func=cmd_roofline)

    x = sub.add_parser("context")
    x.add_argument("--state", required=True)
    x.add_argument("--k", type=int, default=3)
    x.add_argument("--max-diff", type=int, default=1500)
    x.set_defaults(func=cmd_context)

    r = sub.add_parser("resume")
    r.add_argument("--state", required=True)
    r.set_defaults(func=cmd_resume)

    h = sub.add_parser("heartbeat")
    h.add_argument("--state", required=True)
    h.add_argument("--step", type=int, default=None)
    h.add_argument("--phase", default="variation")
    h.add_argument("--best", default="")
    h.add_argument("--speedup", type=float, default=None)
    h.add_argument("--commits", type=int, default=None)
    h.add_argument("--note", default="")
    h.set_defaults(func=cmd_heartbeat)

    at = sub.add_parser("attempt")
    at.add_argument("--state", required=True)
    at.add_argument("--step", type=int, default=0)
    at.add_argument("--worker", type=int, default=None)
    at.add_argument("--strategy", default="")
    at.add_argument("--status", default="")
    at.add_argument("--self-speedup", type=float, default=None)
    at.add_argument("--verified", type=float, default=None)
    at.add_argument("--patch-hash", default="")
    at.add_argument("--rationale", default="")
    at.add_argument("--committed", default="false")
    at.set_defaults(func=cmd_attempt)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
