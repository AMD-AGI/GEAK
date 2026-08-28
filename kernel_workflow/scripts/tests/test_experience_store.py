"""Tests for the local experience store (kernel_workflow/scripts/experience_store.py).

No GPU, no network, no repo state: every case builds its own kb root in tmp_path. The store is the
read/write half of warm start that decides which historical patch a lane spends an on-box verify on,
so what is pinned here is the SELECTION, not the plumbing:
  - identity: the same kernel under three different dir layouts is ONE page (read and write agree);
  - curation: a retired entry is never offered, one rank per direction, near-ties are not read;
  - honesty: a speedup carries the bench it was measured on, and cross-bench ranks say so;
  - the loop: re-writing code the store already holds is a reproduction, not a new entry.
"""

import json
import os
import subprocess
import sys

import pytest

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experience_store.py")
yaml = pytest.importorskip("yaml")


# --------------------------------------------------------------------------- helpers
def run(*args):
    """Invoke the store as the lane does — over the CLI — and parse its single-line JSON."""
    p = subprocess.run([sys.executable, STORE] + [str(a) for a in args],
                       capture_output=True, text=True)
    assert p.returncode == 0, f"store must never fail the caller: {p.stderr}"
    return json.loads(p.stdout)


def patch_text(path="source/k.py", old="BLOCK = 64", new="BLOCK = 128", at=1):
    return (f"diff --git a/{path} b/{path}\nindex 111..222 100644\n"
            f"--- a/{path}\n+++ b/{path}\n@@ -{at},3 +{at},3 @@\n ctx\n-{old}\n+{new}\n")


def write_entry(root, exp_id, *, kernel="fused_moe_kernel", lang="triton", gfx="gfx950",
                kclass="triton", speedup=2.0, direction="tile-retune", retired=False,
                bench="b:aaa", reproductions=1, lifecycle="candidate", patch=None):
    """Seed one entry directly on disk, the way the imported backlog looks."""
    d = os.path.join(root, gfx, kclass, f"{kernel}__{lang}__{gfx}", exp_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "layer": "artifact", "lifecycle": lifecycle, "gfx": gfx, "kernel_class": kclass,
        "kernel_name": kernel, "language": lang, "direction": direction,
        "reproductions": reproductions,
        "metric": {"speedup": speedup, "gpu_arch": gfx, "bench_key": bench, "metric_kind": "geomean"},
        "strategy": f"strategy for {exp_id}",
    }
    if retired:
        meta["retained"] = False
        meta["retired_reason"] = f"duplicate_direction:{direction}"
    with open(os.path.join(d, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f)
    with open(os.path.join(d, "patch.diff"), "w") as f:
        f.write(patch if patch is not None else patch_text(new=f"BLOCK = {exp_id}"))
    return d


def resolve(root, refs, kernel="fused_moe_kernel", lang="triton", gfx="gfx950", *extra):
    return run("resolve", "--root", root, "--kernel-name", kernel, "--language", lang,
               "--gfx", gfx, "--refs-dir", refs, *extra)


# --------------------------------------------------------------------------- identity
@pytest.mark.parametrize("name", [
    "fused_moe_kernel",              # standalone lane: the kernel dir
    "fused_moe_kernel_task",         # e2e head path: the EXTRACTED task dir
    "triton_fused_moe_kernel.py",    # a producer that carries the language in the filename
    "/abs/path/to/fused_moe_kernel", # a lane that hands over a path
])
def test_one_kernel_is_one_page_across_layouts(tmp_path, name):
    """If these forked into separate pages, an e2e head run could never find the history of the
    kernel it is optimizing."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_aaaaaa", speedup=3.0)
    out = resolve(root, refs, kernel=name)
    assert out["read_reason"] == "read", out
    assert out["slug"] == "fused_moe_kernel__triton__gfx950"
    assert out["match_tier"] in ("exact", "normalized")
    assert len(out["candidates"]) == 1


def test_write_and_read_derive_the_same_page(tmp_path):
    """The docstring's contract: read and write MUST canonicalize identically."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    w = run("write", "--root", root, "--kernel-name", "triton_demo_kernel.py", "--language", "triton",
            "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "2.5", "--patch", str(p),
            "--direction", "vectorize")
    assert w["written"] and w["slug"] == "demo_kernel__triton__gfx950"
    out = resolve(root, refs, kernel="demo_kernel_task")     # the e2e head path's name for it
    assert out["match_tier"] == "exact" and len(out["candidates"]) == 1


def test_op_kind_reaches_the_kernel_page_only_when_unambiguous(tmp_path):
    """e2e names a head by op_kind (`fused_moe`); the page is `fused_moe_kernel`. Fuzzy bridges that
    — but never guesses between two pages that are equally close."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_aaaaaa")
    write_entry(root, "20260101_000000_bbbbbb", kernel="fused_moe_kernel_gptq_awq")
    out = resolve(root, refs, kernel="fused_moe")
    assert out["match_tier"] == "fuzzy" and out["slug"] == "fused_moe_kernel__triton__gfx950", \
        "the CLOSEST containing page wins, not just any page sharing the prefix"

    # two pages the query fits equally well: no basis to choose, so choose neither
    write_entry(root, "20260101_000000_cccccc", kernel="fused_moe_alpha")
    write_entry(root, "20260101_000000_dddddd", kernel="fused_moe_bravo")
    amb = resolve(root, refs, kernel="fused_moe")
    assert amb["read_reason"] == "ambiguous_kernel_page" and not amb["candidates"]
    assert len(amb["ambiguous_pages"]) == 2

    # --match exact opts out of all of it: only a page whose slug is literally the requested one.
    strict = resolve(root, refs, "fused_moe", "triton", "gfx950", "--match", "exact")
    assert strict["read_reason"] == "kernel_page_not_found" and not strict["candidates"]


def test_wrong_language_reports_the_page_it_did_find(tmp_path):
    """A hip kernel optimized by a lane defaulted to triton must not look like an empty store —
    that is the difference between 'no history' and 'this lane was invoked wrong'."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_aaaaaa", kernel="wvSplitK", lang="hip", kclass="hip")
    out = resolve(root, refs, kernel="wvsplitk", lang="triton")
    assert out["read_reason"] == "no_page_for_language"
    assert out["other_language_pages"] == ["wvSplitK__hip__gfx950"]
    assert not out["candidates"]


# --------------------------------------------------------------------------- curation
def test_retired_entries_are_never_offered(tmp_path):
    """`retained: false` is the curation's verdict that a better entry of the same idea is already
    served. Ranking by raw speedup would spend verifies re-testing what was rejected."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_hi", speedup=50.0, direction="fp8-bitcast", retired=True)
    write_entry(root, "20260101_000001_lo", speedup=4.0, direction="fp8-bitcast")
    out = resolve(root, refs)
    assert [c["speedup"] for c in out["candidates"]] == [4.0]
    assert out["filtered"]["retired"] == 1

    audit = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--include-retired")
    assert audit["candidates"][0]["speedup"] == 50.0


def test_one_rank_per_direction_runners_up_ride_along(tmp_path):
    """Three impls of one idea verify or fail together, at one full measurement each. The store
    offers the idea once and keeps the rest reachable as alternates."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=9.0, direction="fp8-bitcast")
    write_entry(root, "20260101_000001_b", speedup=8.0, direction="fp8-bitcast")
    write_entry(root, "20260101_000002_c", speedup=7.0, direction="fp8-bitcast")
    write_entry(root, "20260101_000003_d", speedup=2.0, direction="tile-retune")
    out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--top-n", "3")
    assert [c["direction"] for c in out["candidates"]] == ["fp8-bitcast", "tile-retune"]
    assert [a["speedup"] for a in out["candidates"][0]["alternates"]] == [8.0, 7.0]
    assert out["filtered"]["same_direction_collapsed"] == 2


def test_undirected_entries_are_each_their_own_direction(tmp_path):
    """An entry written without a direction label must not collapse with every other such entry."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=9.0, direction="")
    write_entry(root, "20260101_000001_b", speedup=8.0, direction="")
    out = resolve(root, refs)
    assert len(out["candidates"]) == 2


def test_near_ties_are_not_worth_a_measurement(tmp_path):
    """Reading a recorded 1.02x costs the same full verify as reading a 50x."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=1.02, direction="mask-elision")
    out = resolve(root, refs)
    assert out["read_reason"] == "below_min_speedup" and not out["candidates"]
    assert resolve(root, refs, "fused_moe_kernel", "triton", "gfx950",
                   "--min-speedup", "1.0")["candidates"][0]["speedup"] == 1.02


def test_rank_order_carries_bench_comparability(tmp_path):
    """58x on one case set and 5x on another are not ordered facts. Adoption is decided by a fresh
    measurement anyway — but the ranking must not silently claim otherwise."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=58.0, direction="fp8", bench="b:one")
    write_entry(root, "20260101_000001_b", speedup=5.0, direction="tiles", bench="b:two")
    write_entry(root, "20260101_000002_c", speedup=3.0, direction="loads", bench="b:one")
    out = resolve(root, refs)
    assert [c["comparable"] for c in out["candidates"]] == [True, False, True]


def test_reproduced_entry_outranks_a_one_off_tie(tmp_path):
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=4.0, direction="a", reproductions=1)
    write_entry(root, "20260101_000001_b", speedup=4.0, direction="b", reproductions=3)
    out = resolve(root, refs)
    assert out["candidates"][0]["exp_dir"].endswith("20260101_000001_b")


def test_a_fully_retired_page_offers_nothing(tmp_path):
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=4.0, retired=True)
    out = resolve(root, refs)
    assert out["read_reason"] == "all_retired" and not out["candidates"]


# --------------------------------------------------------------------------- the loop
def test_rewriting_known_code_counts_a_reproduction(tmp_path):
    """A warm-started run re-emits the patch it adopted as its own diff: different workspace,
    byte-different, same code. Otherwise the store keeps re-importing its own output as a fresh win."""
    root = str(tmp_path / "kb")
    seeded = write_entry(root, "20260101_000000_a", speedup=3.0, patch=patch_text())
    # same change, e2e head layout + a different hunk offset + reflowed whitespace
    p = tmp_path / "again.diff"
    p.write_text(patch_text(path="kernel_src/sglang/k.py", old="BLOCK  =  64", new="BLOCK  =  128", at=87))
    w = run("write", "--root", root, "--kernel-name", "fused_moe_kernel_task", "--language", "triton",
            "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "2.9", "--patch", str(p),
            "--direction", "tile-retune", "--eval-dir", "/tmp/run2")
    assert w["written"] is False and w["reason"] == "duplicate_impl"
    assert w["reproductions"] == 2 and w["lifecycle"] == "active"
    meta = yaml.safe_load(open(os.path.join(seeded, "meta.yaml")))
    assert meta["metric"]["speedup"] == 3.0, "the original's own measurement must not be overwritten"
    assert meta["reproductions"] == 2 and meta["lifecycle"] == "active"
    assert len(os.listdir(os.path.dirname(seeded))) == 1, "no second entry for the same code"


def test_genuinely_new_code_is_a_new_entry(tmp_path):
    root = str(tmp_path / "kb")
    seeded = write_entry(root, "20260101_000000_a", speedup=3.0, patch=patch_text())
    p = tmp_path / "new.diff"
    p.write_text(patch_text(new="BLOCK = 256"))
    w = run("write", "--root", root, "--kernel-name", "fused_moe_kernel", "--language", "triton",
            "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "3.5", "--patch", str(p),
            "--direction", "tile-retune")
    assert w["written"] is True
    assert len(os.listdir(os.path.dirname(seeded))) == 2


def test_write_records_what_curation_needs(tmp_path):
    root = str(tmp_path / "kb")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    w = run("write", "--root", root, "--kernel-name", "k", "--language", "triton", "--gfx", "gfx950",
            "--kernel-class", "triton", "--speedup", "2.0", "--patch", str(p),
            "--direction", "Vectorize Loads", "--metric-kind", "time_weighted",
            "--case-names", "c64,c2", "--parent", "/kb/old/exp")
    meta = yaml.safe_load(open(os.path.join(w["dir"], "meta.yaml")))
    assert meta["direction"] == "Vectorize Loads"
    assert meta["derived_from"] == "/kb/old/exp"
    assert meta["metric"]["case_names"] == ["c64", "c2"]
    # bench key is order-insensitive over the case set, and namespaced away from imported `b:` keys
    other = run("write", "--root", root, "--kernel-name", "k2", "--language", "triton", "--gfx",
                "gfx950", "--kernel-class", "triton", "--speedup", "2.0", "--patch", str(p),
                "--metric-kind", "time_weighted", "--case-names", "c2,c64")
    bk = yaml.safe_load(open(os.path.join(other["dir"], "meta.yaml")))["metric"]["bench_key"]
    assert bk == meta["metric"]["bench_key"] and bk.startswith("b2:")


# --------------------------------------------------------------------------- path remapping
def remap(tmp_path, patch, editable="", **kw):
    p = tmp_path / "in.diff"
    p.write_text(patch)
    args = ["remap", "--patch", str(p), "--out", str(tmp_path / "out.diff")]
    if editable:
        args += ["--editable", editable]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), v]
    return run(*args), str(tmp_path / "out.diff")


def test_a_stored_patch_is_rewritten_onto_the_head_layout(tmp_path):
    """The store's patches were won in an arena checkout (`source/triton_<kernel>.py`); an e2e head
    run edits an extracted subtree. Different prefix AND basename, so no -p<N> depth reaches the
    file — without this every warm start on the head path fails to apply."""
    out, dest = remap(tmp_path, patch_text(path="source/triton_fused_moe_kernel.py"),
                      "kernel_src/sglang/layers/moe/fused_moe_kernel.py,kernel_src/other.py")
    assert out["remapped"] is True, out
    body = open(dest).read()
    assert "--- a/kernel_src/sglang/layers/moe/fused_moe_kernel.py" in body
    assert "+++ b/kernel_src/sglang/layers/moe/fused_moe_kernel.py" in body
    assert "diff --git a/kernel_src/sglang/layers/moe/fused_moe_kernel.py " in body
    assert "source/" not in body and "-BLOCK = 64" in body, "hunks must pass through untouched"


def test_remap_scans_the_workspace_when_no_editable_set_is_known(tmp_path):
    ws = tmp_path / "ws" / "kernel_src" / "moe"
    ws.mkdir(parents=True)
    (ws / "fused_moe_kernel.py").write_text("x\n")
    out, dest = remap(tmp_path, patch_text(path="source/triton_fused_moe_kernel.py"),
                      workspace=str(tmp_path / "ws"))
    assert out["remapped"] is True
    assert "+++ b/kernel_src/moe/fused_moe_kernel.py" in open(dest).read()


def test_remap_refuses_to_guess_between_two_equally_good_homes(tmp_path):
    out, _ = remap(tmp_path, patch_text(path="source/triton_k.py"), "a/k.py,b/k.py")
    assert out["remapped"] is False and out["reason"] == "unmapped_paths"


def test_remap_is_all_or_nothing(tmp_path):
    """Applying the mapped half of a patch leaves a workspace that is neither before nor after."""
    two = (patch_text(path="source/triton_k.py") + patch_text(path="3rdparty/ck/deep.hpp"))
    out, dest = remap(tmp_path, two, "kernel_src/k.py")
    assert out["remapped"] is False and out["unmapped"] == ["3rdparty/ck/deep.hpp"]
    assert not os.path.exists(dest), "a refused remap must not leave a half-rewritten patch behind"


def test_remap_reports_when_the_paths_already_fit(tmp_path):
    out, _ = remap(tmp_path, patch_text(path="kernel_src/k.py"), "kernel_src/k.py")
    assert out["remapped"] is False and out["reason"] == "no_change_needed"


def test_a_new_file_lands_beside_the_file_the_patch_edits(tmp_path):
    """A created file has nothing to match against, so it follows its edited sibling."""
    add = ("diff --git a/source/helper.py b/source/helper.py\nnew file mode 100644\n"
           "--- /dev/null\n+++ b/source/helper.py\n@@ -0,0 +1,1 @@\n+HELPER = 1\n")
    out, dest = remap(tmp_path, patch_text(path="source/triton_k.py") + add, "kernel_src/moe/k.py")
    assert out["remapped"] is True
    assert out["mapped"]["source/helper.py"] == "kernel_src/moe/helper.py"
    assert "+++ b/kernel_src/moe/helper.py" in open(dest).read()


def test_a_new_file_stays_put_when_the_layout_already_matches(tmp_path):
    """The store's own arena layout IS this workspace's layout in the common re-run case. Nothing
    shifted, so a root-level file the patch creates must not be dragged into the sibling's dir."""
    add = ("diff --git a/profile_run.py b/profile_run.py\nnew file mode 100644\n"
           "--- /dev/null\n+++ b/profile_run.py\n@@ -0,0 +1,1 @@\n+import torch\n")
    out, _ = remap(tmp_path, patch_text(path="source/triton_k.py") + add, "source/triton_k.py")
    assert out["remapped"] is False and out["reason"] == "no_change_needed"


def test_a_non_source_file_this_workspace_lacks_is_dropped_not_refused(tmp_path):
    """A `.gitignore` hunk rode along with 4 real store patches; refusing them over it would throw
    away the kernel work. The section is dropped, the kernel edit survives, and it is reported."""
    ignore = ("diff --git a/.gitignore b/.gitignore\n--- a/.gitignore\n+++ b/.gitignore\n"
              "@@ -5,3 +5,4 @@\n *.o\n+.nfs*\n")
    out, dest = remap(tmp_path, ignore + patch_text(path="source/triton_k.py"),
                      "kernel_src/moe/k.py")
    assert out["remapped"] is True and out["dropped"] == [".gitignore"]
    body = open(dest).read()
    assert ".gitignore" not in body and "+++ b/kernel_src/moe/k.py" in body


@pytest.mark.parametrize("patch, reason", [
    ("diff --git a/x b/y\nrename from x\nrename to y\n", "rename_not_supported"),
    ("not a diff at all\n", "no_paths_in_patch"),
])
def test_remap_refuses_what_it_cannot_rewrite(tmp_path, patch, reason):
    out, _ = remap(tmp_path, patch, "kernel_src/k.py")
    assert out["remapped"] is False and out["reason"] == reason


def test_remap_never_raises(tmp_path):
    out = run("remap", "--patch", str(tmp_path / "missing.diff"), "--out", str(tmp_path / "o.diff"),
              "--editable", "k.py")
    assert out["remapped"] is False and out["reason"].startswith("unreadable_patch")


# --------------------------------------------------------------------------- degradation
@pytest.mark.parametrize("args, reason", [
    (("--gfx", "cpu"), "missing_arch"),
    (("--gfx", "gfx942"), "kernel_page_not_found"),        # arch present in the request, absent on disk
])
def test_resolve_never_raises(tmp_path, args, reason):
    """The lane calls this over Bash mid-run: a store problem must degrade to a cold start, never
    fail the run."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a")
    out = run("resolve", "--root", root, "--kernel-name", "fused_moe_kernel", "--language", "triton",
              "--refs-dir", refs, *args)
    assert out["read_reason"] == reason and out["candidates"] == []


def test_resolve_survives_a_corrupt_entry(tmp_path):
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    good = write_entry(root, "20260101_000000_a", speedup=2.0)
    bad = os.path.join(os.path.dirname(good), "20260101_000001_b")
    os.makedirs(bad)
    with open(os.path.join(bad, "meta.yaml"), "w") as f:
        f.write("{{{ not yaml")
    out = resolve(root, refs)
    assert [c["exp_dir"] for c in out["candidates"]] == [good]


def test_prose_is_mirrored_for_audit_before_any_verdict(tmp_path):
    """A rejected warm start must still be auditable after the run."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    d = write_entry(root, "20260101_000000_a", speedup=2.0)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write("# why this worked\nfolded the scale\n")
    out = resolve(root, refs)
    prose = open(out["candidates"][0]["prose_path"]).read()
    assert "folded the scale" in prose and "tile-retune" in prose
    assert "direction" in open(os.path.join(refs, "index.md")).read()


# --------------------------------------------------------------------------- content
# What an entry CARRIES, and what of it reaches the agent. The store's densest fields (`techniques`,
# the report's dead-ends) were being written and never read; these pin them to the reference.
def run_lines(*args):
    """For subcommands that stream one JSON line per record and a summary line last."""
    p = subprocess.run([sys.executable, STORE] + [str(a) for a in args],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    return lines[:-1], lines[-1]


REPORT = """# TechLead Final Report — k

## Summary

geomean 3.0x

## Round-by-round

### Round 1

r1 text

{heading}

- `MPerBlock=128`: built, correct, 27% slower.
- occupancy pinning: closed eight times.

## Stop rationale

out of budget
"""


@pytest.mark.parametrize("heading", [
    "## What didn't work",
    "## What didn't work (dead-ends)",
    "## What didn't work (dead ends — do not re-fund)",
    "## What didn’t work (confirmed dead-ends — do not re-open)",
    "### What Didn't Work (dead-ends from the ledger)",
])
def test_dead_ends_section_survives_every_heading_dialect(tmp_path, heading):
    """248 imported reports write this heading 20+ different ways; matching the full title would
    silently drop the one section that stops the next run re-funding a closed direction."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("es", STORE)
    es = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(es)
    text = REPORT.format(heading=heading)
    body = es.dead_ends_md(text)
    assert "MPerBlock=128" in body and "occupancy pinning" in body
    assert "out of budget" not in body, "the section must end at the next heading"
    hoisted = es.reorder_report(text)
    assert hoisted.index(heading) < hoisted.index("## Summary")
    assert sorted(hoisted.split()) == sorted((text + "\n---\n").split()), "nothing may be dropped"


def test_report_without_the_two_sections_is_passed_through_untouched(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("es", STORE)
    es = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(es)
    plain = "# report\n\n## Summary\n\nnothing else\n"
    assert es.reorder_report(plain) == plain
    assert es.dead_ends_md(plain) == ""


def test_techniques_and_dead_ends_reach_the_reference(tmp_path):
    """The curated techniques are the densest thing in the store; before this they were written and
    never shown to anyone."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    d = write_entry(root, "20260101_000000_a", speedup=3.0)
    meta = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    meta["techniques"] = ["bitcast operands to OCP e4m3fn", "retile BM=256"]
    with open(os.path.join(d, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(REPORT.format(heading="## What didn't work (dead-ends)"))

    out = resolve(root, refs)
    cand = out["candidates"][0]
    assert cand["techniques"] == ["bitcast operands to OCP e4m3fn", "retile BM=256"]
    prose = open(cand["prose_path"]).read()
    assert "- techniques:\n    * bitcast operands to OCP e4m3fn" in prose
    # the two load-bearing sections come before the narrative, so a tight context reads them first
    assert prose.index("What didn't work") < prose.index("## Summary")


def test_reference_omits_techniques_when_there_are_none(tmp_path):
    """An empty heading is worse than no heading: it reads as 'this patch does nothing'."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_a", speedup=3.0)
    prose = open(resolve(root, refs)["candidates"][0]["prose_path"]).read()
    assert "techniques" not in prose


def test_write_keeps_structured_dead_ends_when_the_report_supplies_them(tmp_path):
    root = str(tmp_path / "kb")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    rep = tmp_path / "report.md"
    rep.write_text(REPORT.format(heading="## What didn't work") + """
<!-- dead-ends:yaml -->
```yaml
- idea: use_buffer_ops=OFF negative control
  measured: 0.883x
  mechanism: the ambient default is load-bearing, -11.7%
- not-a-dict
- {mechanism: no idea field, measured: 1.0x}
```
""")
    w = run("write", "--root", root, "--kernel-name", "k", "--language", "triton", "--gfx", "gfx950",
            "--kernel-class", "triton", "--speedup", "2.0", "--patch", str(p), "--report", str(rep))
    meta = yaml.safe_load(open(os.path.join(w["dir"], "meta.yaml")))
    assert [d["idea"] for d in meta["dead_ends"]] == ["use_buffer_ops=OFF negative control"], \
        "malformed rows are dropped, never patched up into half-empty structure"
    assert meta["dead_ends"][0]["measured"] == "0.883x"
    assert "MPerBlock=128" in meta["dead_ends_md"]
    assert "dead-ends:yaml" not in meta["dead_ends_md"], "the machine block is not prose"


def test_write_falls_back_to_prose_when_there_is_no_structured_block(tmp_path):
    root = str(tmp_path / "kb")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    rep = tmp_path / "report.md"
    rep.write_text(REPORT.format(heading="## What didn't work (dead ends)"))
    w = run("write", "--root", root, "--kernel-name", "k", "--language", "triton", "--gfx", "gfx950",
            "--kernel-class", "triton", "--speedup", "2.0", "--patch", str(p), "--report", str(rep))
    meta = yaml.safe_load(open(os.path.join(w["dir"], "meta.yaml")))
    assert "dead_ends" not in meta, "no structure is honest; invented structure is not"
    assert "MPerBlock=128" in meta["dead_ends_md"]


# --------------------------------------------------------------------------- backfill
def test_backfill_adds_content_without_touching_curation(tmp_path):
    root = str(tmp_path / "kb")
    d = write_entry(root, "20260101_000000_a", speedup=3.0, direction="tile-retune", retired=True)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(REPORT.format(heading="## What didn't work (dead-ends)"))
    before = yaml.safe_load(open(os.path.join(d, "meta.yaml")))

    _rows, summary = run_lines("backfill-content", "--root", root)
    assert summary["changed"] == 1 and summary["applied"] is False
    assert yaml.safe_load(open(os.path.join(d, "meta.yaml"))) == before, "dry run writes nothing"

    _rows, summary = run_lines("backfill-content", "--root", root, "--apply")
    after = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    assert "MPerBlock=128" in after["dead_ends_md"]
    assert after["verified_stack"]["triton"] == "3.6.0"
    assert after["verified_stack"]["recorded_by"] == "campaign20_backfill", \
        "a recovered stack must not read as one we observed"
    assert "layer" not in after
    for k in ("retained", "retired_reason", "direction", "metric", "reproductions", "lifecycle"):
        assert after.get(k) == before.get(k), f"backfill must not touch {k}"

    _rows, summary = run_lines("backfill-content", "--root", root, "--apply")
    assert summary["changed"] == 0, "backfill must be idempotent"


def test_backfill_recomputes_the_signature_the_dedup_actually_reads(tmp_path):
    """The backlog carries `impl_signature`, a different hash under a name nothing reads, so every
    resolve re-hashed all 248 patches. Renaming is not enough — the value must be recomputed."""
    root = str(tmp_path / "kb")
    d = write_entry(root, "20260101_000000_a", speedup=3.0, patch=patch_text())
    meta = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    meta["impl_signature"] = "sha256:deadbeef"
    with open(os.path.join(d, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f)

    run_lines("backfill-content", "--root", root, "--apply")
    after = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    assert "impl_signature" not in after
    assert after["content_signature"].startswith("csha:")

    p = tmp_path / "again.diff"
    p.write_text(patch_text(path="other/layout/k.py", at=99))
    w = run("write", "--root", root, "--kernel-name", "fused_moe_kernel", "--language", "triton",
            "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "3.1", "--patch", str(p))
    assert w["reason"] == "duplicate_impl" and w["reproductions"] == 2


# --------------------------------------------------------------------------- language lookup
def test_languages_lets_the_store_pick_the_language(tmp_path):
    """A caller that guesses `triton` for a kernel filed under `hip`/`ck` reads nothing and never
    learns why; `task_type: hip2hip` cannot tell those two apart, but the pages can."""
    root = str(tmp_path / "kb")
    write_entry(root, "20260101_000000_a", kernel="wvSplitK", lang="hip", kclass="hip")
    write_entry(root, "20260101_000000_b", kernel="moe_stage1", lang="ck", kclass="ck")
    assert run("languages", "--root", root, "--gfx", "gfx950",
               "--kernel-name", "wvSplitK")["languages"] == ["hip"]
    assert run("languages", "--root", root, "--gfx", "gfx950",
               "--kernel-name", "moe_stage1")["languages"] == ["ck"]
    # the name a lane passes is not always the name on disk
    assert run("languages", "--root", root, "--gfx", "gfx950",
               "--kernel-name", "wvsplitk")["languages"] == ["hip"]
    out = run("languages", "--root", root, "--gfx", "gfx950", "--kernel-name", "nothing_here")
    assert out["languages"] == [] and out["reason"] == "no_page"


# --------------------------------------------------------------------------- remote export
# The mapping onto KernelForge's KB Store. The failure mode being guarded is silent: a wrong
# dimension is not an error, it is a cold start — the write lands at an address nobody reads.
def export(root, *extra):
    return run_lines("export-remote", "--root", root, *extra)


def stacked(root, exp_id, *, rocm="7.2", **kw):
    """An entry with a verified_stack, which is where the remote framework_version comes from."""
    d = write_entry(root, exp_id, **kw)
    meta_path = os.path.join(d, "meta.yaml")
    meta = yaml.safe_load(open(meta_path))
    meta["verified_stack"] = {"rocm": rocm} if rocm else {}
    yaml.safe_dump(meta, open(meta_path, "w"))
    return d


def exact(recs):
    """The records at the exact address only. Every measurement is published at each rung of its
    ladder, so counting raw records counts rungs; a test asking "how many measurements" wants this."""
    return [r for r in recs if r["rung"] == 0]


def test_canonical_id_is_seven_ordered_segments(tmp_path):
    """Order and content of the address. Upstream splits on ':' positionally, so a swapped pair is
    a different identity that still parses."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="moe_stage1", lang="ck", kclass="ck")
    recs, summary = export(root)
    assert summary["sessions"] == 1                # one measurement, published at both rungs
    assert recs[0]["canonical_id"] == "geak:kernel:gfx950:moe_stage1:ck:rocm:7.2"
    # and the echoed identity must reconstruct it, or a reader validating the envelope rejects it
    ident = recs[0]["knowledge"]["identity"]
    assert ":".join(["geak", "kernel", ident["gpu"], ident["kernel_name"], ident["backend"],
                     ident["framework"], ident["framework_version"]]) == \
        recs[0]["canonical_id"]


@pytest.mark.parametrize("lang,kclass,backend", [
    ("triton", "triton", "triton"), ("hip", "hip", "hip"), ("ck", "ck", "ck")])
def test_language_is_the_backend_dimension(tmp_path, lang, kclass, backend):
    """`backend` is the final implementation type upstream; `framework` stays rocm for all three
    because one container image supplies all of them."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1", lang=lang, kclass=kclass)
    recs, _ = export(root)
    ident = recs[0]["knowledge"]["identity"]
    assert (ident["backend"], ident["framework"]) == (backend, "rocm")


@pytest.mark.parametrize("raw,want", [
    ("7.2", "7.2"),            # what the recovered backlog carries
    ("7.2.0", "7.2"),          # a full release string
    ("7.2.0-98765", "7.2"),    # what /opt/rocm/.info/version actually holds on some images
    ("", "unspecified"),       # never guessed
])
def test_framework_version_is_cut_to_major_minor(tmp_path, raw, want):
    """A patch release must not split one kernel's history across two addresses, but the exact
    string still has to survive in the payload."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1", rocm=raw)
    recs, _ = export(root)
    assert recs[0]["knowledge"]["identity"]["framework_version"] == want
    if raw:
        assert recs[0]["knowledge"]["value"]["verified_stack"]["rocm"] == raw


def test_session_id_is_content_addressed_and_identity_scoped(tmp_path):
    """Re-exporting one entry must update one candidate, not add another; and two identities must
    never produce the same id, since artifacts are partitioned by session id alone."""
    root = str(tmp_path / "kb")
    same = patch_text(new="BLOCK = 256")
    stacked(root, "20260101_000000_a", kernel="k1", lang="triton", kclass="triton", patch=same)
    stacked(root, "20260101_000000_b", kernel="k1", lang="hip", kclass="hip", patch=same)
    recs, _ = export(root)
    assert len(exact(recs)) == 2
    a, b = sorted(exact(recs), key=lambda r: r["canonical_id"])
    assert a["session_id"] != b["session_id"], "same patch, different identity -> different id"
    assert export(root)[0][0]["session_id"] == recs[0]["session_id"], "not stable across runs"


def test_recorded_signature_is_the_one_the_session_id_was_built_from(tmp_path):
    """The address and the record must name the same patch. A reader dedups against its own store
    by content_signature and fetches by session id; if those two disagree it silently re-downloads
    a port it already has, or worse, treats two different patches as one."""
    root = str(tmp_path / "kb")
    d = stacked(root, "20260101_000000_a", kernel="k1")
    open(os.path.join(d, "report.md"), "w").write(REPORT)
    rec = export(root)[0][0]
    sig = rec["knowledge"]["value"]["content_signature"]
    assert sig.startswith("csha:")
    assert sig[len("csha:"):][:12] in rec["session_id"]
    # and specifically not an artifact digest, which is what a shadowed local reads as
    assert sig[len("csha:"):] not in {f["sha256"] for f in rec["files"]}


def test_identical_patches_collapse_to_the_best_measurement(tmp_path):
    """Byte-identical patches are one candidate upstream. Which measurement we publish still
    matters: the record is written with mode=replace, so sending the lower one publishes a speedup
    we have already beaten."""
    root = str(tmp_path / "kb")
    same = patch_text(new="BLOCK = 256")
    stacked(root, "20260101_000000_a", kernel="k1", speedup=1.20, patch=same)
    stacked(root, "20260101_000000_b", kernel="k1", speedup=1.90, patch=same)
    recs, summary = export(root)
    # `deduped` counts records like `emitted` does, so the one duplicate is dropped once per rung
    assert len(exact(recs)) == 1 and len(set(summary["deduped_dirs"])) == 1
    assert recs[0]["knowledge"]["speedup"] == 1.9
    assert summary["deduped_dirs"] and "20260101_000000_a" in summary["deduped_dirs"][0]


def test_retired_entries_are_not_offered_remotely(tmp_path):
    """The service ranks purely on the speedup we declare, so a retired win exported as a live
    candidate would surface in someone's top-N with no way to tell."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1", speedup=3.0, retired=True,
            direction="dead-idea")
    stacked(root, "20260101_000000_b", kernel="k1", speedup=1.5, direction="live-idea")
    recs, summary = export(root)
    assert [r["knowledge"]["value"]["direction"] for r in exact(recs)] == ["live-idea"]
    assert summary["skipped"]["retired"] == 1
    recs, _ = export(root, "--include-retired")
    assert len(exact(recs)) == 2


def test_champion_is_one_per_identity_and_must_beat_baseline(tmp_path):
    """Upstream's own gate. A candidate is always recorded; only the pointer is earned."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1", speedup=1.5, direction="d1")
    stacked(root, "20260101_000000_b", kernel="k1", speedup=2.5, direction="d2")
    stacked(root, "20260101_000000_c", kernel="k2", speedup=0.9, direction="d3")
    recs, summary = export(root)
    champs = [r for r in exact(recs) if r["champion"]]
    assert len(champs) == 1 and champs[0]["knowledge"]["speedup"] == 2.5
    # two identities, one of which earns the pointer — held at both of its rungs
    assert summary["exact_identities"] == 2 and summary["champions"] == 2
    losing = [r for r in exact(recs) if r["knowledge"]["value"]["direction"] == "d3"][0]
    assert losing["champion"] is False and losing["champion_eligible"] is False


def test_comparability_fields_travel_with_the_speedup(tmp_path):
    """get_top_sessions ranks on the bare `speedup` number and knows nothing about bench keys, so
    a reader can only filter incomparable entries out if we sent what to filter on."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1", bench="b2:zzz")
    value = export(root)[0][0]["knowledge"]["value"]
    assert value["metric"]["bench_key"] == "b2:zzz"
    assert value["metric"]["metric_kind"] == "geomean"
    assert value["direction"] and value["content_signature"]


def test_artifacts_are_referenced_not_inlined(tmp_path):
    """Patches run to 240KB here and reach an agent through a tool result. The record carries a
    path plus a digest; the bytes go up the artifact channel."""
    root = str(tmp_path / "kb")
    d = stacked(root, "20260101_000000_a", kernel="k1")
    open(os.path.join(d, "report.md"), "w").write(REPORT)
    rec = export(root)[0][0]
    assert rec["knowledge"]["value"]["artifacts"] == {"patch": "patch.diff", "report": "report.md"}
    assert {f["path"] for f in rec["files"]} == {"patch.diff", "report.md"}
    for f in rec["files"]:
        assert len(f["sha256"]) == 64 and f["size"] > 0 and f["kind"] == "rewrite"
    blob = json.dumps(rec)
    assert "BLOCK = " not in blob and "Round-by-round" not in blob


def test_verbatim_dead_ends_stay_out_of_the_record(tmp_path):
    """dead_ends_md runs to tens of KB and report.md already carries it; the structured form is
    small enough to ride along and is the half an agent can act on."""
    root = str(tmp_path / "kb")
    d = stacked(root, "20260101_000000_a", kernel="k1")
    meta_path = os.path.join(d, "meta.yaml")
    meta = yaml.safe_load(open(meta_path))
    meta["dead_ends_md"] = "- a very long verbatim section " * 200
    meta["dead_ends"] = [{"idea": "buffer_ops off", "measured": 0.883, "mechanism": "load-bearing"}]
    yaml.safe_dump(meta, open(meta_path, "w"))
    value = export(root)[0][0]["knowledge"]["value"]
    assert "dead_ends_md" not in value
    assert value["dead_ends"][0]["measured"] == 0.883


def test_export_filters_and_overrides(tmp_path):
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", kernel="k1")
    stacked(root, "20260101_000000_b", kernel="k2")
    assert len(exact(export(root, "--kernel-name", "k1")[0])) == 1
    assert export(root, "--gfx", "gfx942")[1]["emitted"] == 0
    rec = export(root, "--kernel-name", "k1", "--producer", "forge-loop", "--gpu", "MI300X")[0][0]
    # the gpu override lands in the address; the producer does not, because who published a result
    # is not part of what the result is about, and an address that moved with it would scatter one
    # kernel's history across every lane that ever wrote to it
    assert rec["canonical_id"] == "geak:kernel:mi300x:k1:triton:rocm:7.2"
    assert rec["session_id"].startswith("geak-")   # the id prefix is ours, not the producer arg


# --------------------------------------------------------------------------- the store plane
# `resolve-remote` / `write-remote` read and write the same experience through a KB Store held on
# disk in the shape the service uses. They exist to make the plane swappable, so what is pinned here
# is that the lane cannot tell the difference: same JSON shape, same curation, same gates — plus the
# two write outcomes the store adds, append vs update, which is what a key-value plane makes visible.

UPLOADER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(STORE))), "kb", "remote_upload.py")


def seed_store(tmp_path, root, *extra):
    """Build a store the way the real one is built: export the directory plane, then load it."""
    jsonl = str(tmp_path / "records.jsonl")
    run("export-remote", "--root", root, "--out", jsonl, *extra)
    store = str(tmp_path / "store")
    p = subprocess.run([sys.executable, UPLOADER, "--records", jsonl, "--local", store,
                        "--apply", "--quiet"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return store


def resolve_remote(store, refs, kernel="fused_moe_kernel", lang="triton", gfx="gfx950", *extra):
    return run("resolve-remote", "--store", store, "--kernel-name", kernel, "--language", lang,
               "--gfx", gfx, "--refs-dir", refs, *extra)


def test_both_planes_offer_the_same_candidates(tmp_path):
    """The whole premise of the store plane: one curated backlog, two ways to read it."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", speedup=4.0, direction="tile-retune")
    stacked(root, "20260101_000000_b", speedup=2.0, direction="vectorize")
    stacked(root, "20260101_000000_c", speedup=1.01, direction="unroll")   # below the floor
    store = seed_store(tmp_path, root)

    local = resolve(root, str(tmp_path / "r1"), "fused_moe_kernel", "triton", "gfx950",
                    "--min-speedup", "1.05")
    remote = resolve_remote(store, str(tmp_path / "r2"), "fused_moe_kernel", "triton", "gfx950",
                            "--min-speedup", "1.05")
    assert remote["read_reason"] == local["read_reason"] == "read"
    keys = ("rank", "speedup", "direction", "comparable", "kernel_name", "language", "gfx")
    assert [{k: c.get(k) for k in keys} for c in remote["candidates"]] == \
           [{k: c.get(k) for k in keys} for c in local["candidates"]]
    assert remote["filtered"]["below_min_speedup"] == local["filtered"]["below_min_speedup"] == 1
    assert remote["canonical_id"] == "geak:kernel:gfx950:fused_moe_kernel:triton:rocm"


def test_the_store_plane_curates_what_the_store_itself_cannot(tmp_path):
    """The store ranks on speedup alone. Direction collapse and bench comparability are ours."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", speedup=40.0, direction="tile-retune", bench="b:imported")
    stacked(root, "20260101_000000_b", speedup=6.0, direction="tile-retune", bench="b:imported")
    stacked(root, "20260101_000000_c", speedup=3.0, direction="vectorize", bench="b2:onbox")
    store = seed_store(tmp_path, root)
    out = resolve_remote(store, str(tmp_path / "refs"))
    assert [c["direction"] for c in out["candidates"]] == ["tile-retune", "vectorize"]
    assert out["filtered"]["same_direction_collapsed"] == 1
    assert out["candidates"][0]["comparable"] is True
    assert out["candidates"][1]["comparable"] is False, "a b2: measurement is not a b: one"
    assert out["candidates"][0]["alternates"], "the runner-up rides along, it is not discarded"


def test_every_offered_patch_resolves_to_real_bytes(tmp_path):
    """A candidate listed with a path that resolves to nothing is worse than not listing it."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", speedup=4.0, direction="tile-retune")
    stacked(root, "20260101_000000_b", speedup=3.0, direction="tile-retune")   # an alternate
    store = seed_store(tmp_path, root)
    out = resolve_remote(store, str(tmp_path / "refs"))
    paths = [c["patch_path"] for c in out["candidates"]]
    paths += [a["patch_path"] for c in out["candidates"] for a in (c.get("alternates") or [])]
    assert paths and all(os.path.getsize(p) > 0 for p in paths)


def test_the_store_plane_mirrors_prose_for_the_same_audit(tmp_path):
    """The report travels as an artifact, so a store-sourced reference reads like a local one."""
    root = str(tmp_path / "kb")
    d = stacked(root, "20260101_000000_a", speedup=4.0)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write("# why this worked\nfolded the scale\n")
    store = seed_store(tmp_path, root)
    refs = str(tmp_path / "refs")
    out = resolve_remote(store, refs)
    assert out["candidates"], out
    prose = open(out["candidates"][0]["prose_path"]).read()
    assert "folded the scale" in prose and "tile-retune" in prose
    assert "direction" in open(os.path.join(refs, "index.md")).read()


def test_a_cold_key_is_a_cold_start_not_a_failure(tmp_path):
    """Same vocabulary as the directory plane, so the lane's logging and gates need no branch."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", speedup=4.0)
    store = seed_store(tmp_path, root)
    out = resolve_remote(store, str(tmp_path / "refs"), "some_other_kernel")
    local = resolve(root, str(tmp_path / "r2"), "some_other_kernel")
    assert out["candidates"] == [] and out["read_reason"] == local["read_reason"]


def test_a_store_root_that_is_not_there_is_a_miss_not_an_empty_store(tmp_path):
    """A typo'd path must not read as 'no experience' and silently cold-start a warm run."""
    out = resolve_remote(str(tmp_path / "nope"), str(tmp_path / "refs"))
    assert out["candidates"] == [] and "no_such_store" in out["reason"]


def test_a_key_that_differs_only_in_stack_version_is_reported_not_silently_used(tmp_path):
    """Asking for 6.4 and being served a 7.2 result is usually right — a patch normally survives the
    hop — but it must never look like an exact hit, or a genuine version-specific regression reads as
    a kernel that simply did not reproduce."""
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_a", speedup=4.0, rocm="7.2")
    store = seed_store(tmp_path, root)
    out = resolve_remote(store, str(tmp_path / "refs"), "fused_moe_kernel", "triton", "gfx950",
                         "--framework-version", "6.4")
    # Rung 2 drops the version rather than naming a different one, so there is no per-version page
    # to point at: the read lands on the version-agnostic rung and says so in the tier.
    assert out["match_tier"] == "any_version"
    assert out["canonical_id"] == "geak:kernel:gfx950:fused_moe_kernel:triton:rocm"
    assert [c["speedup"] for c in out["candidates"]] == [4.0], "the coarse rung still serves it"


def test_a_write_records_both_planes(tmp_path):
    root = str(tmp_path / "kb")
    store = str(tmp_path / "store")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    w = run("write-remote", "--root", root, "--store", store, "--kernel-name", "fused_moe_kernel",
            "--language", "triton", "--gfx", "gfx950", "--kernel-class", "triton",
            "--speedup", "2.0", "--patch", str(p), "--direction", "tile-retune",
            "--framework-version", "7.2")
    assert w["written"] is True and os.path.isfile(os.path.join(w["dir"], "meta.yaml"))
    assert w["remote"]["written"] is True
    assert w["remote"]["canonical_id"] == "geak:kernel:gfx950:fused_moe_kernel:triton:rocm:7.2"
    assert w["remote"]["champion"] is True and w["remote"]["replaced"] is False
    out = resolve_remote(store, str(tmp_path / "refs"))
    assert [c["speedup"] for c in out["candidates"]] == [2.0]
    assert out["candidates"][0]["is_champion"] is True


def test_a_new_patch_appends_a_candidate_under_the_same_key(tmp_path):
    root, store = str(tmp_path / "kb"), str(tmp_path / "store")
    first, second = tmp_path / "a.diff", tmp_path / "b.diff"
    first.write_text(patch_text(new="BLOCK = 128"))
    second.write_text(patch_text(new="BLOCK = 256"))
    a = run("write-remote", "--root", root, "--store", store, "--kernel-name", "k", "--language",
            "triton", "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "2.0",
            "--patch", str(first), "--direction", "tile-retune", "--framework-version", "7.2")
    b = run("write-remote", "--root", root, "--store", store, "--kernel-name", "k", "--language",
            "triton", "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "3.0",
            "--patch", str(second), "--direction", "vectorize", "--framework-version", "7.2")
    assert a["remote"]["canonical_id"] == b["remote"]["canonical_id"]
    assert b["remote"]["session_id"] != a["remote"]["session_id"]
    assert b["remote"]["replaced"] is False and b["remote"]["champion"] is True
    out = resolve_remote(store, str(tmp_path / "refs"), "k")
    assert [c["speedup"] for c in out["candidates"]] == [3.0, 2.0]


def test_remeasuring_one_patch_updates_its_candidate_instead_of_adding_one(tmp_path):
    """Session ids are content-addressed, so the store cannot inflate from repeated runs."""
    root, store = str(tmp_path / "kb"), str(tmp_path / "store")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    a = run("write-remote", "--root", root, "--store", store, "--kernel-name", "k", "--language",
            "triton", "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "2.0",
            "--patch", str(p), "--direction", "tile-retune", "--framework-version", "7.2")
    # the same code re-emitted from a different workspace: shifted hunk, reflowed whitespace
    again = tmp_path / "again.diff"
    again.write_text(patch_text(path="kernel_src/sglang/k.py", old="BLOCK  =  64",
                                new="BLOCK  =  128", at=87))
    b = run("write-remote", "--root", root, "--store", store, "--kernel-name", "k", "--language",
            "triton", "--gfx", "gfx950", "--kernel-class", "triton", "--speedup", "2.1",
            "--patch", str(again), "--direction", "tile-retune", "--framework-version", "7.2")
    assert b["written"] is False and b["reason"] == "duplicate_impl"     # the directory plane's view
    assert b["remote"]["session_id"] == a["remote"]["session_id"]
    assert b["remote"]["replaced"] is True
    out = resolve_remote(store, str(tmp_path / "refs"), "k")
    assert len(out["candidates"]) == 1, "a reproduction is not a second candidate"
    # The confirmation is not lost, it is recorded on the one candidate: same code, measured twice.
    document = json.load(open(os.path.join(
        store, *b["remote"]["canonical_id"].split(":"), "sessions",
        b["remote"]["session_id"], "knowledge.json")))
    assert document["value"]["reproductions"] == 2
    assert document["speedup"] == 2.0, "the original's own measurement is what was recorded"


def test_a_store_failure_never_costs_the_measured_result(tmp_path):
    """The directory plane is the source of truth; a KB write is bookkeeping on top of it."""
    root = str(tmp_path / "kb")
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    p = tmp_path / "p.diff"
    p.write_text(patch_text())
    w = run("write-remote", "--root", root, "--store", str(blocked), "--kernel-name", "k",
            "--language", "triton", "--gfx", "gfx950", "--kernel-class", "triton",
            "--speedup", "2.0", "--patch", str(p), "--framework-version", "7.2")
    assert w["written"] is True, "the local entry must survive a broken store"
    assert w["remote"]["written"] is False and w["remote"]["reason"]


@pytest.mark.parametrize("args,reason", [
    (("resolve-remote", "--store", "/nonexistent/store"), "a store that is not there"),
    (("resolve-remote", "--store", "/tmp", "--canonical-id", "not a valid id"), "a malformed key"),
])
def test_the_store_read_never_raises(tmp_path, args, reason):
    out = run(*args, "--kernel-name", "k", "--language", "triton", "--gfx", "gfx950",
              "--refs-dir", str(tmp_path / "refs"))
    assert out["candidates"] == [], reason
    assert out["read_reason"], "a cold start still has to say why"


# --------------------------------------------------------------------------- attestation
# `reproductions` counts the same code being WRITTEN twice. This counts the stored entry being
# READ back out and put on a box — the only signal a later retire pass can act on, and the one
# thing the schema never recorded.
def attest(root, exp_dir, outcome="validated", *extra):
    return run("attest", "--exp-dir", exp_dir, "--outcome", outcome, *extra)


def test_attesting_counts_the_attempt_and_moves_no_speedup(tmp_path):
    root = str(tmp_path / "kb")
    d = write_entry(root, "20260101_000000_aaaaaa", speedup=2.0)
    out = attest(root, d, "validated", "--measured-speedup", "1.9", "--measured-by", "boxA",
                 "--note", "held on 7.2", "--apply")
    assert out["attested"] is True
    assert out["attestations"]["recalls"] == 1 and out["attestations"]["validations"] == 1
    meta = yaml.safe_load(open(os.path.join(d, "meta.yaml")))
    assert meta["metric"]["speedup"] == 2.0, "an attestation is evidence, not a re-measurement"
    assert meta["reproductions"] == 1, "not the same counter as a duplicate write"
    entry = meta["attestations"]["history"][-1]
    assert entry["measured_speedup"] == 1.9 and entry["by"] == "boxA"


def test_a_dry_attestation_writes_nothing(tmp_path):
    root = str(tmp_path / "kb")
    d = write_entry(root, "20260101_000000_aaaaaa")
    out = attest(root, d, "failed")
    assert out["attested"] is False and out["attestations"]["failures"] == 1
    assert "attestations" not in yaml.safe_load(open(os.path.join(d, "meta.yaml")))


def test_attestations_accumulate_and_raise_a_hint_the_read_surfaces(tmp_path):
    """The hint is advisory: the entry is still offered and still ranked. Retiring it is a separate
    act by a separate caller, which is what makes the counters safe to write automatically."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    d = write_entry(root, "20260101_000000_aaaaaa")
    for _ in range(2):
        attest(root, d, "not_reproduced", "--apply")
    candidate = resolve(root, refs)["candidates"][0]
    assert candidate["recalls"] == 2 and candidate["validations"] == 0
    assert "could not reproduce" in candidate["retire_hint"]
    assert "track record" in open(candidate["prose_path"]).read()


def test_a_never_tried_entry_reads_as_untried_not_as_failing(tmp_path):
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_aaaaaa")
    candidate = resolve(root, refs)["candidates"][0]
    assert candidate["recalls"] == 0 and candidate["retire_hint"] == ""


def test_attest_never_fails_the_caller(tmp_path):
    out = run("attest", "--exp-dir", str(tmp_path / "nope"), "--outcome", "validated", "--apply")
    assert out["attested"] is False and out["reason"]
    out = run("attest", "--exp-dir", str(write_entry(str(tmp_path / "kb"), "20260101_000000_a")),
              "--outcome", "validated", "--measured-speedup", "not a number", "--apply")
    assert out["attested"] is True, "unusable evidence drops; the attempt still counted"


def _seeded(tmp_path, root):
    """A store built the way seed_store builds one, plus the session id the export minted."""
    jsonl = str(tmp_path / "records.jsonl")
    run("export-remote", "--root", root, "--out", jsonl)
    store = str(tmp_path / "store")
    p = subprocess.run([sys.executable, UPLOADER, "--records", jsonl, "--local", store,
                        "--apply", "--quiet"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    records = [json.loads(line) for line in open(jsonl) if line.strip()]
    return store, records[0]["session_id"]


def test_the_remote_record_carries_the_local_ledger(tmp_path):
    """Both planes read the same page, so a count that only lands on one of them makes the two
    disagree about how much anyone has actually tried this."""
    root = str(tmp_path / "kb")
    d = stacked(root, "20260101_000000_aaaaaa")
    attest(root, d, "validated", "--apply")
    store, _ = _seeded(tmp_path, root)
    out = resolve_remote(store, str(tmp_path / "refs"), "fused_moe_kernel")
    assert out["candidates"][0]["validations"] == 1


def test_attest_remote_counts_against_the_key_addressed_record(tmp_path):
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_aaaaaa")
    store, session = _seeded(tmp_path, root)
    out = run("attest-remote", "--store", store, "--session-id", session, "--outcome", "validated",
              "--kernel-name", "fused_moe_kernel", "--language", "triton", "--gfx", "gfx950",
              "--framework-version", "7.2", "--measured-speedup", "2.2", "--apply")
    assert out["attested"] is True
    found = [r for r in out["pages"] if r["found"]]
    assert found and all(r["attestations"]["validations"] == 1 for r in found)
    assert resolve_remote(store, str(tmp_path / "refs"), "fused_moe_kernel"
                          )["candidates"][0]["validations"] == 1


def test_attest_remote_on_an_unknown_session_reports_rather_than_raises(tmp_path):
    root = str(tmp_path / "kb")
    stacked(root, "20260101_000000_aaaaaa")
    store, _ = _seeded(tmp_path, root)
    out = run("attest-remote", "--store", store, "--session-id", "nope", "--outcome", "failed",
              "--kernel-name", "fused_moe_kernel", "--language", "triton", "--gfx", "gfx950",
              "--apply")
    assert out["attested"] is False and all(p["found"] is False for p in out["pages"])


# --------------------------------------------------------------------------- the tuning carrier
# A tuning win is a config table plus the env var that binds it, deployed into an installed package —
# there is no diff, so the store's `empty_diff` gate used to reject it and the knowledge was lost with
# the run. `carrier: tuned_artifact` is how it is filed instead. What is pinned here is the SEPARATION:
# the two carriers share one ranking and one page, and a reader is only ever served the one it asked
# for, because being offered a candidate you cannot install is worse than being offered nothing.
def tuned_files(tmp_path, name="E=8,N=1024,device_name=AMD Instinct MI355X.json", body='{"1":{}}'):
    d = tmp_path / "tuned"
    d.mkdir(exist_ok=True)
    (d / name).write_text(body)
    return str(d / name)


def write_tuned(root, artifact, *, kernel="fused_moe_kernel", speedup=1.21, **kw):
    return run("write", "--root", root, "--kernel-name", kernel, "--language", "triton",
               "--gfx", "gfx950", "--kernel-class", "tuning", "--speedup", speedup,
               "--carrier", "tuned_artifact", "--artifact", artifact,
               *[x for k, v in kw.items() for x in ("--" + k.replace("_", "-"), v)])


def test_a_tuning_win_with_no_diff_is_stored_not_rejected(tmp_path):
    """The whole point: before carriers this returned empty_diff and the tuned table was lost."""
    out = write_tuned(str(tmp_path / "kb"), tuned_files(tmp_path))
    assert out["written"] is True and out["carrier"] == "tuned_artifact"
    assert os.path.isdir(os.path.join(out["dir"], "artifact"))


def test_the_installed_name_survives_sanitization(tmp_path):
    """The stored name must satisfy the remote plane's path validator, but the RUNTIME finds a tuned
    table only under its exact shape-derived name. Losing it does not error — the table is silently
    ignored and the recall reads as a tuning loss, which is the expensive way to find this bug."""
    out = write_tuned(str(tmp_path / "kb"), tuned_files(tmp_path))
    stored = out["artifacts"][0]
    assert "=" not in stored and " " not in stored          # safe_rel_path would have raised
    meta = yaml.safe_load(open(os.path.join(out["dir"], "meta.yaml")))
    assert meta["artifact_names"][stored] == "E=8,N=1024,device_name=AMD Instinct MI355X.json"


def test_a_directory_of_tables_is_expanded(tmp_path):
    """A tuner hands back one file or the dir it filled, and which one is not the caller's problem."""
    art = tuned_files(tmp_path)
    (tmp_path / "tuned" / "second.json").write_text('{"2":{}}')
    out = write_tuned(str(tmp_path / "kb"), os.path.dirname(art))
    assert len(out["artifacts"]) == 2


def test_an_empty_artifact_set_is_refused(tmp_path):
    """Symmetric with empty_diff: an entry with nothing installable in it is not knowledge."""
    out = write_tuned(str(tmp_path / "kb"), str(tmp_path / "does_not_exist.json"))
    assert out["written"] is False and out["reason"] == "no_artifact"


def test_re_tuning_the_same_tables_is_a_reproduction(tmp_path):
    """Content addressing has to work off the artifact bytes, the way it works off patch text —
    otherwise every run of a deterministic tuner mints a new entry and the page fills with itself."""
    root, art = str(tmp_path / "kb"), tuned_files(tmp_path)
    first = write_tuned(root, art)
    again = write_tuned(root, art, speedup=1.25)
    assert again["written"] is False and again["reason"] == "duplicate_impl"
    assert again["reproductions"] == 2 and again["reproduced"] == first["exp_id"]
    assert again["lifecycle"] == "active"      # an independent second measurement promotes it


def test_a_reader_is_served_one_carrier_and_it_defaults_to_patch(tmp_path):
    """Both carriers on ONE page, and the kernel lane — which passes no --carrier at all — must not
    see the tuned entry, because `git apply` does nothing with a config table."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_entry(root, "20260101_000000_aaaaaa", speedup=1.5)      # a normal patch entry
    write_tuned(root, tuned_files(tmp_path))
    default = resolve(root, refs)
    assert [c["carrier"] for c in default["candidates"]] == ["patch"]
    assert default["filtered"]["other_carriers"] == 1

    tuned = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950",
                    "--carrier", "tuned_artifact")
    assert [c["carrier"] for c in tuned["candidates"]] == ["tuned_artifact"]


def test_the_tuned_candidate_carries_what_installing_it_needs(tmp_path):
    """A path alone is not enough: the table is inert without its env var, and stale without the
    cache invalidation. The candidate has to hand over all three or the recall cannot succeed."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_tuned(root, tuned_files(tmp_path), apply_env="SGLANG_MOE_CONFIG_DIR=/x/tuned",
                cache_invalidation="rm -rf ~/.triton/cache", tuner="benchmark_moe.py")
    c = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950",
                "--carrier", "tuned_artifact")["candidates"][0]
    assert c["apply_env"] == "SGLANG_MOE_CONFIG_DIR=/x/tuned"
    assert c["cache_invalidation"] == "rm -rf ~/.triton/cache"
    assert c["artifact_paths"] and all(os.path.isfile(p) for p in c["artifact_paths"])


def test_a_page_with_only_the_other_carrier_says_so(tmp_path):
    """read_reason has to distinguish "nothing here" from "nothing here FOR YOU" — a caller that
    cannot tell them apart will record a cold start where there is knowledge it could not use."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    write_tuned(root, tuned_files(tmp_path))
    out = resolve(root, refs)
    assert out["read_reason"] == "no_such_carrier" and out["other_carriers"] == 1


def test_the_tuned_entry_reaches_the_remote_plane_installable(tmp_path):
    """The kernel lane's export skipped anything without a patch.diff, which is every tuned entry.
    Round-trip it: the bytes must materialize AND the reader must learn what to name them."""
    root = str(tmp_path / "kb")
    write_tuned(root, tuned_files(tmp_path), apply_env="SGLANG_MOE_CONFIG_DIR=/x/tuned")
    store, _ = _seeded(tmp_path, root)
    c = resolve_remote(store, str(tmp_path / "refs"), "fused_moe_kernel", "triton", "gfx950",
                       "--carrier", "tuned_artifact", "--min-speedup", "1.0")["candidates"][0]
    assert c["carrier"] == "tuned_artifact" and c["apply_env"] == "SGLANG_MOE_CONFIG_DIR=/x/tuned"
    assert c["artifact_paths"] and all(os.path.isfile(p) for p in c["artifact_paths"])
    assert [c["artifact_names"][os.path.basename(p)] for p in c["artifact_paths"]] == \
        ["E=8,N=1024,device_name=AMD Instinct MI355X.json"]


# --------------------------------------------------------------------------- precision
# The page is keyed on arch and op, NOT on dtype: one `fused_moe` page holds the bf16 and the
# fp8_w8a8 tables side by side, ranked against each other on speedup alone. Precision could not
# JOIN the key — the remote store exposes no delete, so moving every existing entry's address
# would orphan the whole backlog — so it is recorded and filtered instead. What is pinned here is
# that the filter never costs a caller history it could have used: unstated on either side is a
# match, and a coarse dtype still reaches its own refinements.
def tuned_at(root, tmp_path, precision, *, speedup, direction, body):
    home = tmp_path / (precision or "unstated")
    home.mkdir(exist_ok=True)
    return write_tuned(root, tuned_files(home, body=body), speedup=speedup,
                       direction=direction, metric_kind="tuning_isolated",
                       **({"precision": precision} if precision else {}))


def _offers(out):
    return [(c["speedup"], c["direction"]) for c in out["candidates"]]


def test_precision_is_recorded_but_does_not_move_the_page(tmp_path):
    """The reason this is a filter and not a key dimension. If stating precision changed the
    address, every entry written before the field existed would become unreachable — on a store
    that cannot delete, that is the whole backlog stranded, permanently."""
    root = str(tmp_path / "kb")
    plain = tuned_at(root, tmp_path, "", speedup=2.0, direction="d1", body='{"1":{}}')
    typed = tuned_at(root, tmp_path, "fp8_w8a8", speedup=3.0, direction="d2", body='{"2":{}}')
    assert os.path.dirname(plain["dir"]) == os.path.dirname(typed["dir"])   # same page
    assert yaml.safe_load(open(os.path.join(typed["dir"], "meta.yaml")))["upstream"] == \
        {"precision": "fp8_w8a8"}
    # An unstated write stays byte-identical to what it was before the field existed.
    assert "upstream" not in yaml.safe_load(open(os.path.join(plain["dir"], "meta.yaml")))


def test_a_reader_that_states_no_precision_sees_what_it_always_saw(tmp_path):
    """The migration guarantee. Every caller predates this flag; none may lose a candidate by
    not yet passing it."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "fp8_w8a8", speedup=3.29, direction="tuning-aiter", body='{"1":{}}')
    tuned_at(root, tmp_path, "bf16", speedup=4.10, direction="tuning-ck", body='{"2":{}}')
    out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950",
                  "--carrier", "tuned_artifact", "--min-speedup", "1.0")
    assert _offers(out) == [(4.1, "tuning-ck"), (3.29, "tuning-aiter")]
    assert out["filtered"]["other_precisions"] == 0


def test_the_other_dtype_is_dropped_before_ranking_not_after(tmp_path):
    """Why the filter is worth having: bf16 outranks fp8 on raw speedup, so an fp8 deployment
    reading this page unfiltered spends its first verify slot on a table whose filename encodes
    a dtype its runtime never looks up. Ranking pollution, not corruption — but it costs a slot."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "fp8_w8a8", speedup=3.29, direction="tuning-aiter", body='{"1":{}}')
    tuned_at(root, tmp_path, "bf16", speedup=4.10, direction="tuning-ck", body='{"2":{}}')
    out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--carrier",
                  "tuned_artifact", "--min-speedup", "1.0", "--precision", "FP8-w8a8")
    assert _offers(out) == [(3.29, "tuning-aiter")]      # spelling folded: FP8-w8a8 == fp8_w8a8
    assert out["filtered"]["other_precisions"] == 1   # and it SAYS what it withheld


def test_a_coarse_dtype_still_reaches_its_own_refinements(tmp_path):
    """A caller that only knows it is on fp8 must still see the fp8_w8a8 entries, and vice versa:
    they are two statements about the same thing at different resolutions. Matching on the token
    boundary rather than a raw prefix is what keeps `fp8` from also swallowing `fp16`."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "fp8_w8a8", speedup=3.29, direction="tuning-aiter", body='{"1":{}}')
    tuned_at(root, tmp_path, "fp16", speedup=9.99, direction="tuning-fp16", body='{"2":{}}')
    for asked in ("fp8", "float8"):
        out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--carrier",
                      "tuned_artifact", "--min-speedup", "1.0", "--precision", asked)
        assert _offers(out) == [(3.29, "tuning-aiter")], asked


def test_the_backlog_is_never_excluded_for_saying_nothing(tmp_path):
    """Every entry recovered before this field existed states no precision. Excluding those would
    empty every page in the store — and an unlabelled entry is still a lead worth a verify slot,
    exactly as an unvalidated one is."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "", speedup=2.0, direction="tuning-legacy", body='{"1":{}}')
    out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--carrier",
                  "tuned_artifact", "--min-speedup", "1.0", "--precision", "fp8_w8a8")
    assert _offers(out) == [(2.0, "tuning-legacy")]


def test_a_page_with_only_the_wrong_dtype_says_so(tmp_path):
    """Same contract `no_such_carrier` holds: a caller has to be able to tell "nothing here" from
    "nothing here FOR YOU", or it records a cold start where there is knowledge it could not use."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "bf16", speedup=4.1, direction="tuning-ck", body='{"1":{}}')
    out = resolve(root, refs, "fused_moe_kernel", "triton", "gfx950", "--carrier",
                  "tuned_artifact", "--min-speedup", "1.0", "--precision", "int4")
    assert out["read_reason"] == "no_such_precision" and out["other_precisions"] == 1
    assert out["candidates"] == []


def test_the_filter_survives_the_round_trip_to_the_store_plane(tmp_path):
    """`upstream` has to cross the export, or the remote plane — the one the tuning role actually
    reads — filters on a field that is always empty and silently offers everything."""
    root, refs = str(tmp_path / "kb"), str(tmp_path / "refs")
    tuned_at(root, tmp_path, "fp8_w8a8", speedup=3.29, direction="tuning-aiter", body='{"1":{}}')
    tuned_at(root, tmp_path, "bf16", speedup=4.10, direction="tuning-ck", body='{"2":{}}')
    store, _ = _seeded(tmp_path, root)
    common = ("--carrier", "tuned_artifact", "--min-speedup", "1.0")
    assert _offers(resolve_remote(store, refs, "fused_moe_kernel", "triton", "gfx950", *common)) == \
        [(4.1, "tuning-ck"), (3.29, "tuning-aiter")]
    narrowed = resolve_remote(store, refs, "fused_moe_kernel", "triton", "gfx950",
                              *common, "--precision", "fp8_w8a8")
    assert _offers(narrowed) == [(3.29, "tuning-aiter")] and narrowed["filtered"]["other_precisions"] == 1
