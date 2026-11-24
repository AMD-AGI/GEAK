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
    # the removed standalone workflow's stage cards, records and gates: GEAK's rounds replace them
    "runtime",
    "references/method/entry.md",
    "references/method/climb.md",
    "references/method/close.md",
    "references/method/records.md",
    "references/method/orchestration.md",
    "references/method/budget.md",
    "references/method/index.md",
    "scripts/toolctl.py",
    "scripts/round_record.py",
    "scripts/close_audit.py",
    "scripts/parity_gate.py",
    "scripts/champion_gate.py",
    "scripts/run_state.py",
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










def test_skill_md_is_the_entry_and_routes_nowhere_superseded():
    text = (PACK / "skill.md").read_text()
    assert len(text.splitlines()) <= SKILL_MD_MAX_LINES
    body = body_outside_sources(text)
    # a bare pack-relative name; `e2e_workflow/.../SKILL.md` or GEAK's own profiler path is fine
    for stale in ("SKILL.md", "tile-programming-gluon.md", "references/method-reference.md",
                  "references/phases/", "scripts/profile_kernel.sh"):
        hit = re.search(r"(?<![\w/])" + re.escape(stale), body)
        assert hit is None, f"skill.md routes to superseded {stale!r}: ...{body[hit.start()-60:hit.end()+20]!r}"
    for page in ("transcribe", "recover"):
        assert f"references/method/{page}.md" in text


def test_no_agents_and_the_method_maps_onto_existing_geak_roles():
    # GEAK consumes expert skills through its existing kernel_workflow roles; this skill adds none.
    assert not (PACK / "agents").exists()
    assert not (EXPERT_SKILLS_ROOT / "expert_workflows" / "gluon_authoring").exists()
    text = (PACK / "skill.md").read_text()
    assert "## Roles in GEAK" in text
    for role in ("deep_engineer", "deep_explore", "tech_lead", "verify_engineer", "benchmark_engineer",
                 "profile_engineer", "director"):
        assert f"`{role}`" in text, role
    roles_dir = GEAK_ROOT / "kernel_workflow" / "roles"
    for role in ("deep_engineer", "verify_engineer", "benchmark_engineer", "profile_engineer", "tech_lead", "director"):
        assert (roles_dir / f"{role}.md").is_file(), role
    assert (roles_dir / "_fragments" / "expert_skills.md").is_file()


def test_injected_mode_entry_is_the_playbook():
    # use_expert_skills injects matched, validated skills into kernel_workflow roles; their entry is the
    # index's playbook_file, which for this pack is playbook.md (mechanics only), not skill.md.
    playbook = PACK / "playbook.md"
    assert playbook.is_file()
    text = (PACK / "skill.md").read_text()
    front = yaml.safe_load(text.split("\n---\n", 1)[0].removeprefix("---\n"))
    assert front["playbook_file"] == "playbook.md"
    assert "use_expert_skills" in text and "playbook.md" in text
    pb_front = yaml.safe_load(playbook.read_text().split("\n---\n", 1)[0].removeprefix("---\n"))
    assert pb_front["id"] == "gluon_authoring"
    index = yaml.safe_load((EXPERT_SKILLS_ROOT / "index.yaml").read_text())
    entry = next(s for s in index["skills"] if s["id"] == "gluon_authoring")
    assert entry["playbook_file"] == "skills/gluon_authoring/playbook.md"
    assert (EXPERT_SKILLS_ROOT / entry["playbook_file"]) == playbook


def test_markdown_links_use_real_relative_paths_and_exact_anchors(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('pack_refs_strict', SCRIPTS / 'check_pack_refs.py')
    checker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker_module)
    pack = tmp_path / 'pack'
    (pack / 'references').mkdir(parents=True)
    (pack / 'target.md').write_text('# Complete heading\n')
    citing = pack / 'references' / 'page.md'
    citing.write_text('[wrong path](target.md)\n[wrong anchor](../target.md#complete)\n'
                      '## Sources\n[broken source](deleted.md)\n')
    checker = checker_module.Checker(pack, ['*'])  # allowlist must not hide real broken links
    checker.scan_markdown_links(citing)
    assert len(checker.problems) == 3
    citing.write_text('[valid](../target.md#complete-heading)\n`array[i][j]`\n')
    checker = checker_module.Checker(pack, [])
    checker.scan_markdown_links(citing)
    assert not checker.problems


def test_locus_maps_and_forwards_frozen_calibration(tmp_path):
    host = tmp_path / 'host'; host.mkdir()
    env = dict(__import__('os').environ, TILE_KERNEL_HOST_WORKDIR=str(host),
               TILE_KERNEL_CONTAINER_WORKDIR='/mounted', ROOFLINE_MIBENCH_CACHE=str(host / 'peak.csv'),
               GEAK_HARDWARE_CONTEXT=str(host / 'hardware_context.json'))
    env.pop('LOCUS_ENV', None)
    result = run(['bash', '-c', 'source "$1"; _locus_env_args', 'test', str(PACK.parents[3] / 'kernel_workflow/scripts/kernel_tools/locus.sh')], env=env)
    assert result.returncode == 0, result.stderr
    assert 'ROOFLINE_MIBENCH_CACHE=/mounted/peak.csv\x00' in result.stdout
    assert 'GEAK_HARDWARE_CONTEXT=/mounted/hardware_context.json\x00' in result.stdout
