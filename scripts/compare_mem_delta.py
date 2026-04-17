"""Compare mem=off vs mem=on for the 3 L3 targets across canonical snapshots.

Walks /data/sapmajum/triton_runs/canonical_snapshots/*/fused_* and reads
each kernel's geak_summary.json (written by GEAK postprocess) plus the
raw log, producing a per-kernel table:

   kernel            | mem=off geomean | mem=on geomean | speedup(on/off) | delta
"""

import json
import re
import sys
from pathlib import Path

SNAP_DIR = Path("/data/sapmajum/triton_runs/canonical_snapshots")
LOG_DIR = Path("/data/sapmajum/triton_runs")


def _geomean_from_summary(summary_path: Path) -> tuple[float | None, float | None]:
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    base = data.get("baseline_latency_ms") or data.get("full_benchmark_baseline_ms")
    best = data.get("best_latency_ms") or data.get("full_benchmark_best_ms")
    speedup = data.get("verified_speedup") or data.get("best_speedup")
    try:
        return float(base) if base else None, float(speedup) if speedup else None
    except (TypeError, ValueError):
        return None, None


def _geomean_from_log(log_path: Path) -> tuple[float | None, float | None]:
    """Fallback: parse 'geomean: X.XXX ms -> Y.YYY ms' from log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None, None
    m = re.search(r"Full benchmark geomean:\s*([0-9.]+)\s*ms\s*->\s*([0-9.]+)\s*ms", text)
    if not m:
        return None, None
    base = float(m.group(1))
    best = float(m.group(2))
    speedup = base / best if best else None
    return base, speedup


def collect() -> dict:
    targets = ["fused_rms_fp8", "fused_qkv_rope", "gemm_a16wfp4"]
    results: dict[str, dict] = {t: {"off": [], "on": []} for t in targets}
    for snap in sorted(SNAP_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        name = snap.name
        for target in targets:
            if target not in name:
                continue
            if "memoff" in name:
                mode = "off"
            elif "memon" in name:
                mode = "on"
            else:
                continue
            # Find geak_summary.json
            sum_paths = list(snap.rglob("geak_summary.json")) + list(snap.rglob("*/geak_summary.json"))
            base, speedup = None, None
            for sp in sum_paths:
                base, speedup = _geomean_from_summary(sp)
                if speedup:
                    break
            # Fallback to log parsing
            if speedup is None:
                for log in LOG_DIR.glob(f"{target}*.log"):
                    b, s = _geomean_from_log(log)
                    if s:
                        base, speedup = b, s
                        break
            if speedup:
                results[target][mode].append(
                    {"snapshot": name, "baseline_ms": base, "speedup": speedup}
                )
    return results


def main() -> int:
    results = collect()
    print()
    print(
        f"{'Kernel':<22} | {'Mode':<5} | {'Runs':<5} | {'Best speedup':<13} | {'Best baseline (ms)':<20} | {'Delta (on/off)':<15}"
    )
    print("-" * 110)
    for target, modes in results.items():
        for mode, runs in modes.items():
            if not runs:
                continue
            best = max(runs, key=lambda r: r["speedup"])
            print(
                f"{target:<22} | {mode:<5} | {len(runs):<5} | {best['speedup']:<13.4f} | "
                f"{best['baseline_ms'] if best['baseline_ms'] else 'n/a':<20} | "
            )
        off_runs = modes["off"]
        on_runs = modes["on"]
        if off_runs and on_runs:
            best_off = max(r["speedup"] for r in off_runs)
            best_on = max(r["speedup"] for r in on_runs)
            delta = (best_on / best_off - 1) * 100
            print(f"  --> {target} mem-delta: {delta:+.1f}% ({best_off:.3f}x -> {best_on:.3f}x)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
