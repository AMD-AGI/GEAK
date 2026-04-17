"""Inject a cross-session-memory seed from standalone geak logs.

When a `geak -t ...` run produces a verified speedup but its /outputs was
ephemeral (container recreated), the winning patch body is lost but the
log still contains enough metadata (kernel name, strategy name, verified
speedup, latencies) to build a useful seed that survives and transfers.

This script parses such logs and appends a seed entry to
`src/minisweagent/memory/cross_session/knowledge_base.json`.

Usage (CLI):
  python scripts/inject_offline_seed.py \
      --log /data/sapmajum/triton_runs/fast_rms_layernorm_20260416_203545.log \
      --kernel-category normalization \
      --bottleneck-type memory \
      --kernel-url triton2triton/geak_eval/L2/fast_rms_layernorm
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_KB = Path(__file__).resolve().parents[1] / "src" / "minisweagent" / "memory" / "cross_session" / "knowledge_base.json"


_RE_VERIFIED = re.compile(
    r"Verified speedup:\s*(?P<speedup>[0-9.]+)x\s*\((?P<base>[0-9.]+)\s*ms\s*->\s*(?P<opt>[0-9.]+)\s*ms\)"
)
_RE_FROM_ROUND = re.compile(r"speedup\s+(?P<speedup>[0-9.]+)x\s+from\s+round\s+(?P<round>\d+)")
_RE_STRATEGY = re.compile(r"\(\s*(?P<strategy>[a-z0-9\-]+)\s*\)")
_RE_KERNEL_PATH = re.compile(r"kernel_url[:=]\s*(?P<url>\S+kernel\.py)")
_RE_KERNEL_NAME = re.compile(r"kernel_name[:=]\s*(?P<name>\w+)")


def parse_log(log_path: Path) -> dict:
    """Extract best verified speedup + strategy name + latencies from a
    geak_agent log. Returns a dict with keys suitable for seed injection.
    """
    text = log_path.read_text(errors="replace")

    verified = list(_RE_VERIFIED.finditer(text))
    if not verified:
        raise ValueError(f"no 'Verified speedup:' lines in {log_path}")

    best = max(verified, key=lambda m: float(m.group("speedup")))
    best_speedup = float(best.group("speedup"))
    baseline_ms = float(best.group("base"))
    best_ms = float(best.group("opt"))

    # Locate the nearest 'speedup X.XXx from round N' marker so we know which round
    best_round = None
    for m in _RE_FROM_ROUND.finditer(text):
        if float(m.group("speedup")) == best_speedup:
            best_round = int(m.group("round"))
            break

    strategies = sorted(
        {s.group("strategy") for s in _RE_STRATEGY.finditer(text) if "-" in s.group("strategy")},
        key=len,
        reverse=True,
    )
    best_strategy = strategies[0] if strategies else "unknown-strategy"

    kernel_url_match = _RE_KERNEL_PATH.search(text)
    kernel_name_match = _RE_KERNEL_NAME.search(text)

    return {
        "best_speedup": best_speedup,
        "baseline_latency_ms": baseline_ms,
        "best_latency_ms": best_ms,
        "best_round": best_round,
        "best_strategy": best_strategy,
        "kernel_url": kernel_url_match.group("url") if kernel_url_match else "",
        "kernel_name": kernel_name_match.group("name") if kernel_name_match else log_path.stem.split("_")[0],
        "strategies_seen": strategies[:10],
    }


def build_seed_record(
    parsed: dict,
    *,
    kernel_name: str,
    kernel_category: str,
    bottleneck_type: str,
    kernel_url: str,
    kernel_language: str = "triton",
    key_insight: str = "",
    extra_strategies: list[dict] | None = None,
) -> dict:
    now_utc = datetime.now(timezone.utc).isoformat()
    record_id = f"{kernel_name}_synthetic_v{int(datetime.now(timezone.utc).timestamp())}"

    if not key_insight:
        key_insight = (
            f"Verified {parsed['best_speedup']:.2f}x speedup via '{parsed['best_strategy']}' "
            f"strategy. Baseline {parsed['baseline_latency_ms']:.4f}ms -> "
            f"{parsed['best_latency_ms']:.4f}ms."
        )

    strategy_entries = []
    # Primary winning strategy
    strategy_entries.append(
        {
            "round": f"round_{parsed['best_round'] or 1}",
            "task": parsed["best_strategy"],
            "patch": "patch_0",
            "speedup": parsed["best_speedup"],
            "after_code": "",
            "before_code": "",
            "success": True,
            "note": "Patch body not recoverable (container /outputs ephemeral); strategy name + verified speedup preserved.",
        }
    )
    if extra_strategies:
        strategy_entries.extend(extra_strategies)

    return {
        "record_id": record_id,
        "timestamp": now_utc,
        "kernel_name": kernel_name,
        "kernel_category": kernel_category,
        "kernel_language": kernel_language,
        "kernel_url": kernel_url or parsed.get("kernel_url", ""),
        "bottleneck_type": bottleneck_type,
        "baseline_latency_ms": parsed["baseline_latency_ms"],
        "top_kernels": [],
        "hardware": "MI355X",
        "profiling_metrics": {},
        "best_speedup": parsed["best_speedup"],
        "best_latency_ms": parsed["best_latency_ms"],
        "success": True,
        "best_strategy": parsed["best_strategy"],
        "best_change_category": "algorithmic",
        "key_insight": key_insight,
        "trajectory_sketch": (
            f"R{parsed['best_round'] or 1}:{parsed['best_strategy']},={parsed['best_speedup']:.3f}x "
            f"(best: {parsed['best_speedup']:.3f}x)"
        ),
        "patch_content": "",
        "code_changes_summary": (
            f"## Verified Final Selection\n"
            f"- Best task: {parsed['best_strategy']}\n"
            f"- Verified FULL_BENCHMARK speedup: {parsed['best_speedup']:.4f}x\n"
            f"- Full benchmark geomean: {parsed['baseline_latency_ms']:.4f} ms -> {parsed['best_latency_ms']:.4f} ms"
        ),
        "profiling_insight": f"Baseline latency: {parsed['baseline_latency_ms']:.6f}ms (geomean).",
        "original_kernel_code": "",
        "baseline_benchmark": "",
        "kernel_structure": f"Triton kernel, {kernel_category} category",
        "round_insights": [
            f"Round {parsed['best_round'] or 1}: {parsed['best_speedup']:.3f}x verified, "
            f"task={parsed['best_strategy']}, "
            f"{parsed['baseline_latency_ms']:.4f}ms->{parsed['best_latency_ms']:.4f}ms."
        ],
        "strategies": strategy_entries,
        "verified_speedup_source": "standalone_geak_run_log",
        "source_logs": [],  # filled by caller
    }


def append_to_kb(kb_path: Path, record: dict) -> None:
    data = json.loads(kb_path.read_text())
    data.setdefault("experiences", []).append(record)
    data["experience_count"] = len(data["experiences"])
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    kb_path.write_text(json.dumps(data, indent=2))
    print(f"[inject] appended '{record['kernel_name']}' ({record['best_speedup']:.3f}x) -> {kb_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path, help="geak_agent.log path")
    ap.add_argument("--kernel-name", help="override parsed kernel name")
    ap.add_argument("--kernel-category", required=True)
    ap.add_argument("--bottleneck-type", required=True)
    ap.add_argument("--kernel-url", default="", help="relative kernel URL (e.g. triton2triton/geak_eval/L2/fast_rms_layernorm)")
    ap.add_argument("--kernel-language", default="triton")
    ap.add_argument("--key-insight", default="")
    ap.add_argument("--kb", default=str(DEFAULT_KB), type=Path)
    args = ap.parse_args()

    parsed = parse_log(args.log)
    kernel_name = args.kernel_name or parsed["kernel_name"]
    record = build_seed_record(
        parsed,
        kernel_name=kernel_name,
        kernel_category=args.kernel_category,
        bottleneck_type=args.bottleneck_type,
        kernel_url=args.kernel_url,
        kernel_language=args.kernel_language,
        key_insight=args.key_insight,
    )
    record["source_logs"] = [str(args.log)]
    append_to_kb(args.kb, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
