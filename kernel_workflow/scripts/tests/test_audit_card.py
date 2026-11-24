"""The routing card quotes the same stop/continue constants hw_budget reads (thresholds.json), routes
only tools that exist, stays short, and carries no absolute steering lines (no GPU)."""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CARD = REPO / "kernel_workflow/scripts/kernel_tools/CARD.md"
KT = REPO / "kernel_workflow/scripts/kernel_tools"
TH = json.loads((REPO / "perf_knowledge/hardware/data/thresholds.json").read_text())["hw_budget"]


def _fmt(x):
    return f"{x:g}"


def test_card_and_help_quote_the_thresholds_constants():
    card = CARD.read_text()
    lo, hi = _fmt(TH["little_room_below"]), _fmt(TH["large_room_from"])
    assert f"< {lo}x" in card and f"≥ {hi}x" in card
    help_text = subprocess.run([sys.executable, str(KT / "hw_budget.py"), "--help"],
                               capture_output=True, text=True).stdout
    assert f"{lo}x" in help_text and f"{hi}x" in help_text
    for stale in ("1.1x", "3x+", "4x = rounds"):
        assert stale not in card and stale not in help_text, stale


def test_card_routes_only_existing_tools_and_is_short():
    card = CARD.read_text()
    assert len(card.encode()) < 3800
    for tool in set(re.findall(r"\$KT/([\w.]+\.(?:py|sh|md))", card)):
        assert (KT / tool).is_file(), tool
    for gone in ("calc_perf", "kernel_breakdown", "roofline.sh", "rocprof_compute_probe", "profiler_deps"):
        assert gone not in card, gone


def test_card_has_no_absolute_steering():
    card = CARD.read_text()
    for line in ("do not recall", "Run every profile_kernel.sh", "within ~3%", "run them on every candidate",
                 "Do not recall"):
        assert line not in card, line
