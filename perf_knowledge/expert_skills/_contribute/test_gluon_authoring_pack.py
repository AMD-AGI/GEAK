from pathlib import Path
import json
import re
import subprocess
import sys

import pytest
import yaml


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
EXPERT_SKILLS_ROOT = SKILLS_ROOT.parent
GEAK_ROOT = EXPERT_SKILLS_ROOT.parents[1]
PACK = SKILLS_ROOT / "gluon_authoring"
SCRIPTS = PACK / "scripts"
STAGE_CARDS = sorted((PACK / "runtime" / "stages").glob("*.json"))
SKILL_MD_MAX_LINES = 1400
# Folded into skill.md and references/method/* by the v2 restructure; nothing may route to them.
SUPERSEDED = [
    "tile-programming-gluon.md",
    "references/method-reference.md",
    "references/phases.md",
    "references/phases",
    "references/entry-modes.md",
    "references/experiment-records.md",
    "references/gluon-negative-patterns.md",
    "references/platform-known-issues.md",
    "references/benchmark-hygiene.md",
    "scripts/profile_kernel.sh",
]


def run(args, cwd=PACK, **kw):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=600, **kw)


def body_outside_sources(text):
    """skill.md minus its `## Sources` section (provenance names the upstream files it folded in)."""
    out, skip = [], False
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            skip = len(m.group(1)) == 2 and m.group(2).strip().lower().startswith("sources")
        if not skip:
            out.append(line)
    return "\n".join(out)


def test_static_validation_passes():
    result = run([sys.executable, str(EXPERT_SKILLS_ROOT / "_contribute" / "validate_skill.py"),
                  "gluon_authoring", "--static"], cwd=GEAK_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_pack_reference_resolves():
    result = run([sys.executable, str(SCRIPTS / "check_pack_refs.py"), "--pack", str(PACK)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_pack_refs_selftest_catches_breakage():
    result = run([sys.executable, str(SCRIPTS / "check_pack_refs.py"), "--selftest"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "selftest: PASS" in result.stdout


def test_term_index_resolves():
    result = run([sys.executable, str(SCRIPTS / "check_term_index.py"), "--pack", str(PACK)])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rel", SUPERSEDED)
def test_superseded_sources_are_gone(rel):
    assert not (PACK / rel).exists(), f"{rel} was folded into skill.md / references/method and must not ship"


def test_toolctl_starts_a_fresh_run_at_entry(tmp_path):
    result = run([sys.executable, str(SCRIPTS / "toolctl.py"), "stage",
                  "--work", str(tmp_path), "--role", "deep", "--json"])
    assert result.returncode == 0, result.stdout + result.stderr
    stage = json.loads(result.stdout)
    assert stage["current"] == "entry"
    assert stage["reference"].startswith("references/method/entry.md#")


def test_stage_cards_agree_with_the_skill_index():
    index = json.loads((PACK / "runtime" / "skill-index.json").read_text())
    by_id = {s["id"]: s["reference"] for s in index["stage_graph"]["stages"]}
    assert len(STAGE_CARDS) == len(by_id) == 6
    for card_path in STAGE_CARDS:
        card = json.loads(card_path.read_text())
        assert card["stage_spec"]["reference"] == by_id[card["stage"]], card_path.name


@pytest.mark.parametrize("card_path", STAGE_CARDS, ids=lambda p: p.stem)
def test_stage_reference_heading_is_queryable(card_path, tmp_path):
    """The heading each stage card addresses is served by `toolctl context query`, exact text."""
    card = json.loads(card_path.read_text())
    ref = card["stage_spec"]["reference"]
    rel, _, heading = ref.partition("#")
    assert rel.startswith("references/method/"), ref
    assert heading, ref
    acquire = run([sys.executable, str(SCRIPTS / "toolctl.py"), "context", "acquire",
                   "--work", str(tmp_path), "--role", "deep", "--artifact", f"stage_ref=pack:/{rel}",
                   "--json"])
    assert acquire.returncode == 0, acquire.stdout + acquire.stderr
    query = run([sys.executable, str(SCRIPTS / "toolctl.py"), "context", "query",
                 "--work", str(tmp_path), "--role", "deep", "--artifact", "stage_ref",
                 "--markdown-heading", heading, "--max-bytes", "2000", "--json"])
    assert query.returncode == 0, query.stdout + query.stderr
    result = json.loads(query.stdout)["result"]
    assert re.match(r"^#{1,6} " + re.escape(heading), result), result[:200]


def test_standing_references_exist():
    facts = json.loads((SCRIPTS / "pack_facts.json").read_text())
    refs = dict(facts["standing_references"])
    for card_path in STAGE_CARDS:
        refs.update(json.loads(card_path.read_text()).get("standing_references") or {})
    for name, rel in refs.items():
        assert (PACK / rel).is_file(), f"standing reference {name} -> {rel} does not exist"


def test_skill_md_is_the_entry_and_routes_nowhere_superseded():
    text = (PACK / "skill.md").read_text()
    assert len(text.splitlines()) <= SKILL_MD_MAX_LINES
    body = body_outside_sources(text)
    # a bare pack-relative name; `e2e_workflow/.../SKILL.md` or GEAK's own profiler path is fine
    for stale in ("SKILL.md", "tile-programming-gluon.md", "references/method-reference.md",
                  "references/phases/", "scripts/profile_kernel.sh"):
        hit = re.search(r"(?<![\w/])" + re.escape(stale), body)
        assert hit is None, f"skill.md routes to superseded {stale!r}: ...{body[hit.start()-60:hit.end()+20]!r}"
    for stage in ("entry", "transcribe", "recover", "profile", "climb", "close"):
        assert f"references/method/{stage}.md" in text


def test_no_agents_and_the_method_maps_onto_existing_geak_roles():
    # GEAK consumes expert skills through its existing kernel_workflow roles; this skill adds none.
    assert not (PACK / "agents").exists()
    text = (PACK / "skill.md").read_text()
    assert "## Roles in GEAK" in text
    for role in ("deep_engineer", "deep_explore", "verify_engineer", "benchmark_engineer", "profile_engineer"):
        assert role in text, role
    roles_dir = PACK.parents[3] / "kernel_workflow" / "roles"
    for role in ("deep_engineer", "verify_engineer", "benchmark_engineer", "profile_engineer", "tech_lead", "director"):
        assert (roles_dir / f"{role}.md").is_file(), role
