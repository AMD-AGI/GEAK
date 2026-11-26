"""The kernel-tools routing lives in the Gluon fact store as `kind: tool` cards (it replaced
kernel_tools/CARD.md): every tool the kernel-tools prompt block names has a card, the cards route
only tools that exist, hw_budget --help quotes the same stop/continue constants it reads
(thresholds.json), and no card carries an absolute steering line (no GPU)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow/scripts/kernel_tools"
PACK = REPO / "perf_knowledge/expert_skills/skills/gluon_authoring"
FACTS = PACK / "facts.jsonl"
TH = json.loads((REPO / "perf_knowledge/hardware/data/thresholds.json").read_text())["hw_budget"]
# The tools kernel_lane.js kernelToolsBlock names by file; each needs a card to be found by q.py.
BLOCK_TOOLS = ("dump_ir.sh", "probe.py", "asm_loop_audit.py", "hw_budget.py", "rocprofv3_safe.sh",
               "parse_pmc.py", "amd_isa.py", "extract_sku.py", "ab_bench.py")


def _fmt(x):
    return f"{x:g}"


def _tool_cards():
    if not FACTS.is_file():
        pytest.fail(f"{FACTS} missing: the kernel-tools routing is the fact store's kind:tool cards")
    cards = [json.loads(line) for line in FACTS.read_text().splitlines() if line.strip()]
    tools = [c for c in cards if c.get("kind") == "tool"]
    assert tools, "no kind:tool cards in facts.jsonl"
    return tools


def _text(card):
    return json.dumps(card, ensure_ascii=False)


def test_every_tool_the_prompt_block_names_has_a_card():
    blob = "\n".join(_text(c) for c in _tool_cards())
    for tool in BLOCK_TOOLS:
        assert tool in blob, f"no kind:tool card names {tool}"


def test_tool_cards_route_only_existing_tools():
    for card in _tool_cards():
        text = _text(card)
        for tool in set(re.findall(r"(?:\$KT/|kernel_tools/)([\w.-]+\.(?:py|sh))", text)):
            assert (KT / tool).is_file(), (card["id"], tool)
        for tool in set(re.findall(r"(?<![\w/])scripts/([\w.-]+\.(?:py|sh))", text)):
            assert (PACK / "scripts" / tool).is_file() or (KT / tool).is_file(), (card["id"], tool)
        for gone in ("calc_perf", "kernel_breakdown", "roofline.sh", "rocprof_compute_probe",
                     "CARD.md", "isa_tools.md", "profiling-dependencies.md"):
            assert gone not in text, (card["id"], gone)


def test_hw_budget_help_quotes_the_thresholds_constants():
    lo, hi = _fmt(TH["little_room_below"]), _fmt(TH["large_room_from"])
    help_text = subprocess.run([sys.executable, str(KT / "hw_budget.py"), "--help"],
                               capture_output=True, text=True).stdout
    assert f"{lo}x" in help_text and f"{hi}x" in help_text
    for stale in ("1.1x", "3x+", "4x = rounds"):
        assert stale not in help_text, stale


def test_tool_cards_have_no_absolute_steering():
    for card in _tool_cards():
        text = _text(card)
        for line in ("do not recall", "Run every profile_kernel.sh", "within ~3%",
                     "run them on every candidate", "Do not recall"):
            assert line not in text, (card["id"], line)
