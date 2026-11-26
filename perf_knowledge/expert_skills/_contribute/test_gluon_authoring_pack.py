"""gluon_authoring is a reference fact store: facts.jsonl (one card per line) + a generated INDEX.md,
read through scripts/q.py and gated by scripts/lint_facts.py. The prose trees it replaced
(playbook.md, reference.md, references/, the scripts' README/USAGE, kernel_tools' CARD/README/
isa_tools/profiling-dependencies) must be gone, and nothing in the pack may route to them."""
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
KT = GEAK_ROOT / "kernel_workflow" / "scripts" / "kernel_tools"
FACTS = PACK / "facts.jsonl"
INDEX = PACK / "INDEX.md"
Q = SCRIPTS / "q.py"
LINT = SCRIPTS / "lint_facts.py"
# Kept ONLY because default-path KB pages (perf_knowledge/hardware/rdna*/{arch,pitfalls}.md) link it
# and test_rdna4_policy.js / test_gfx1151_knowledge_docs.js assert it; the pack itself never routes
# to it. Changing those KB pages would change the use_expert_skills=off default.
DEFAULT_PATH_KEEP = {"references/pitfalls/platform-known-issues.md"}

REMOVED_PACK = [
    "playbook.md", "reference.md", "scripts/README.md", "scripts/USAGE.md",
    "scripts/check_pack_refs.py", "scripts/check_pack_refs.allow", "scripts/gen_section_map.py",
    "scripts/check_term_index.py", "agents",
    # superseded by earlier restructures; must stay gone
    "tile-programming-gluon.md", "runtime", "scripts/profile_kernel.sh", "scripts/toolctl.py",
    "scripts/round_record.py", "scripts/close_audit.py", "scripts/parity_gate.py",
    "scripts/champion_gate.py", "scripts/run_state.py",
]
REMOVED_KT = ["CARD.md", "README.md", "isa_tools.md", "profiling-dependencies.md"]
# A pointer to any of these is a route to a removed file.
ROUTE_PATTERNS = [
    r"(?<![\w-])playbook\.md", r"(?<![\w./-])reference\.md", r"(?<![\w-])references/",
    r"USAGE\.md", r"scripts/README\.md", r"kernel_tools/README", r"CARD\.md", r"isa_tools\.md",
    r"profiling-dependencies\.md", r"check_pack_refs", r"check_term_index", r"gen_section_map",
    r"term-index\.md", r"[Ss]ection map",
]
# PROVENANCE.md names what was removed, by design.
ROUTE_EXEMPT = {"PROVENANCE.md"} | DEFAULT_PATH_KEEP
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".txt", ""}


def run(args, cwd=PACK, **kw):
    return subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True,
                          timeout=600, **kw)


def front_matter(path):
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert m, f"{path} has no YAML front matter"
    return yaml.safe_load(m.group(1)), m.group(2)


def cards():
    assert FACTS.is_file(), "facts.jsonl missing"
    return [json.loads(line) for line in FACTS.read_text().splitlines() if line.strip()]


def test_fact_store_files_exist():
    for p in (FACTS, INDEX, Q, LINT, PACK / "skill.md", PACK / "PROVENANCE.md"):
        assert p.is_file(), p
    ids = [c["id"] for c in cards()]
    assert ids and len(ids) == len(set(ids))


def test_lint_facts_pack_is_clean():
    for cwd in (PACK, GEAK_ROOT):
        r = run([sys.executable, LINT, "--pack"], cwd=cwd)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_q_finds_a_pasted_error_string_listed_in_index():
    index = INDEX.read_text()
    errors = [(c["id"], k) for c in cards() if c.get("kind") == "error"
              for k in (c.get("keys") or []) if len(k) >= 12 and k in index]
    assert errors, "no kind:error card has a verbatim key listed in INDEX.md"
    for cid, key in errors[:3]:
        pasted = f"Traceback (most recent call last):\n  ...\n{key}\n"
        r = run([sys.executable, Q, pasted])
        assert r.returncode == 0, r.stderr
        assert cid in r.stdout, (cid, key, r.stdout[:1500])


def test_q_lists_tool_cards():
    r = run([sys.executable, Q, "--kind", "tool"], cwd=GEAK_ROOT)
    assert r.returncode == 0, r.stderr
    tools = [c["id"] for c in cards() if c.get("kind") == "tool"]
    assert tools and all(t in r.stdout for t in tools[:5]), r.stdout[:1500]


def _q_pointers():
    """Every `q.py [--kind K] <query>` pointer the scripts, kernel tools and wiring print or cite."""
    pat = re.compile(r'q\.py(?: --kind ([a-z]+))?(?: (\\?"[^"\\<]+\\?"|[A-Za-z_][\w-]*))?')
    roots = [SCRIPTS, KT]
    extra = [GEAK_ROOT / "kernel_workflow" / "kernel_lane.js",
             GEAK_ROOT / "kernel_workflow" / "roles" / "_fragments" / "expert_skills.md"]
    files = [p for r in roots for p in r.iterdir() if p.is_file() and p.suffix in {".py", ".sh"}] + extra
    found = set()
    for p in files:
        if p.name in {"q.py", "lint_facts.py"}:
            continue
        for kind, query in pat.findall(p.read_text()):
            query = query.replace('\\"', '"').strip('"')
            if query in {"", "and"}:
                continue
            found.add((kind, query))
    return sorted(found)


@pytest.mark.parametrize("kind,query", _q_pointers())
def test_every_q_pointer_in_the_tools_finds_a_card(kind, query):
    ids = {c["id"] for c in cards()}
    args = [sys.executable, Q, query] + (["--kind", kind] if kind else [])
    r = run(args, cwd=GEAK_ROOT)
    assert r.returncode == 0 and any(i in r.stdout for i in ids), (args[2:], r.stdout[:800], r.stderr[-400:])


@pytest.mark.parametrize("rel", REMOVED_PACK)
def test_removed_pack_files_are_gone(rel):
    assert not (PACK / rel).exists(), f"{rel} was replaced by the fact store and must not ship"


@pytest.mark.parametrize("name", REMOVED_KT)
def test_removed_kernel_tools_docs_are_gone(name):
    assert not (KT / name).exists(), f"kernel_tools/{name} was replaced by kind:tool cards"


def test_references_tree_holds_only_the_default_path_keep():
    left = {str(p.relative_to(PACK)) for p in (PACK / "references").rglob("*") if p.is_file()} \
        if (PACK / "references").exists() else set()
    assert left <= DEFAULT_PATH_KEEP, sorted(left - DEFAULT_PATH_KEEP)


def _pack_text_files():
    for p in sorted(PACK.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix not in TEXT_SUFFIXES:
            continue
        rel = str(p.relative_to(PACK))
        if rel in ROUTE_EXEMPT:
            continue
        yield rel, p


def test_nothing_in_the_pack_routes_to_a_removed_path():
    hits = []
    for rel, p in _pack_text_files():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for pat in ROUTE_PATTERNS:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}: {pat}")
    assert not hits, "\n".join(hits[:40])


def test_kernel_tools_and_wiring_route_nowhere_removed():
    files = [p for p in KT.iterdir() if p.is_file() and p.suffix in {".py", ".sh"}]
    files += [GEAK_ROOT / "kernel_workflow" / "kernel_lane.js",
              GEAK_ROOT / "kernel_workflow" / "kernel_workflow.js",
              GEAK_ROOT / "kernel_workflow" / "README.md",
              GEAK_ROOT / "kernel_workflow" / "roles" / "_fragments" / "expert_skills.md",
              GEAK_ROOT / "e2e_workflow" / "roles" / "_fragments" / "expert_skills.md"]
    pats = [r"CARD\.md", r"isa_tools\.md", r"profiling-dependencies\.md", r"kernel_tools/README",
            r"gluon_authoring/playbook", r"references/method/", r"references/hardware/[\w-]+\.md",
            r"check_term_index", r"check_pack_refs", r"gen_section_map"]
    hits = [f"{f.relative_to(GEAK_ROOT)}: {pat}" for f in files for pat in pats
            if re.search(pat, f.read_text())]
    assert not hits, "\n".join(hits)


def test_skill_md_is_a_reference_fact_store_and_index_agrees():
    fm, body = front_matter(PACK / "skill.md")
    assert fm["id"] == "gluon_authoring" and fm["role"] == "reference_fact_store"
    assert not fm.get("playbook_file")
    assert (fm.get("validation") or {}).get("status") == "draft"  # reference store: no recipe validation claimed
    for sec in ("When to use", "Evidence and scope", "How to query", "Sources"):
        assert f"## {sec}" in body, sec
    assert "## Procedure" not in body
    assert "scripts/q.py" in body and "overrides" in body
    index = yaml.safe_load((EXPERT_SKILLS_ROOT / "index.yaml").read_text())
    entry = next(s for s in index["skills"] if s["id"] == "gluon_authoring")
    assert str(entry["revision"]) == str(fm["revision"])
    assert entry["playbook_file"] == "skills/gluon_authoring/skill.md"
    assert entry["validation_file"] == "skills/gluon_authoring/skill.md"
    assert entry["validation_status"] == "draft" and entry["auto_apply"] is False and entry["scope"] == "kernel"


def test_static_validation_passes():
    result = run([sys.executable, EXPERT_SKILLS_ROOT / "_contribute" / "validate_skill.py",
                  "gluon_authoring", "--static"], cwd=GEAK_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_holds_a_fact_store_to_its_own_sections(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validate_skill_under_test", EXPERT_SKILLS_ROOT / "_contribute" / "validate_skill.py")
    vs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vs)
    fm, body = front_matter(PACK / "skill.md")
    skill = tmp_path / "skill.md"
    skill.write_text("x")
    errs = vs.static_check(str(skill), dict(fm, playbook_file="playbook.md"), body)
    assert any("missing facts.jsonl" in e for e in errs), errs
    assert any("no playbook_file" in e for e in errs), errs
    assert any("## How to query" in e for e in vs.static_check(str(PACK / "skill.md"), fm, "## When to use\nx\n"))


def test_no_agents_and_the_geak_roles_and_fragment_exist():
    assert not (PACK / "agents").exists()
    assert not (EXPERT_SKILLS_ROOT / "expert_workflows" / "gluon_authoring").exists()
    roles_dir = GEAK_ROOT / "kernel_workflow" / "roles"
    for role in ("deep_engineer", "verify_engineer", "benchmark_engineer", "profile_engineer",
                 "tech_lead", "director"):
        assert (roles_dir / f"{role}.md").is_file(), role
    frag = (roles_dir / "_fragments" / "expert_skills.md").read_text()
    assert "reference_fact_store" in frag and "q.py" in frag and "overrides[]" in frag


def test_locus_maps_and_forwards_frozen_calibration(tmp_path):
    host = tmp_path / 'host'; host.mkdir()
    env = dict(__import__('os').environ, TILE_KERNEL_HOST_WORKDIR=str(host),
               TILE_KERNEL_CONTAINER_WORKDIR='/mounted', ROOFLINE_MIBENCH_CACHE=str(host / 'peak.csv'),
               GEAK_HARDWARE_CONTEXT=str(host / 'hardware_context.json'))
    env.pop('LOCUS_ENV', None)
    result = run(['bash', '-c', 'source "$1"; _locus_env_args', 'test', str(KT / 'locus.sh')], env=env)
    assert result.returncode == 0, result.stderr
    assert 'ROOFLINE_MIBENCH_CACHE=/mounted/peak.csv\x00' in result.stdout
    assert 'GEAK_HARDWARE_CONTEXT=/mounted/hardware_context.json\x00' in result.stdout
