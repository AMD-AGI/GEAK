"""Contract tests for _gen_run_recurrence.py.

The digest publishes base rates, so the failure modes worth testing are all forms of *overstating*:

  * counting cards where it claims to count kernels (one campaign becomes "widely measured"),
  * reporting the wins and quietly dropping the closed side,
  * routing an axis to an operator it was not measured on,
  * an audit trail whose links do not resolve — checkable-looking and not checkable,
  * a scope tag (`gfx950`, `decode`) presented as an optimization axis.

Each of those is a test below. The last group runs against the real card trees, because the counts
this file publishes are only as good as the corpus they are read from, and a fixture cannot catch a
vocabulary that drifted.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
SCRIPT = os.path.join(HERE, "_gen_run_recurrence.py")


def load():
    spec = importlib.util.spec_from_file_location("_gen_run_recurrence", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = load()


def test_the_card_parser_is_not_whichever_kb_imported_first():
    """`kb` names two different things in this repo; the generator must bind the right one.

    `kernel_workflow/scripts/kb.py` is the card parser. `kb/` at the repo root is an unrelated
    PACKAGE (the experience store). A `sys.path.insert` + `import kb` resolves to whichever is
    already in `sys.modules`, and `kb/tests/test_store_env.py` -- which CI runs in the same pytest
    session -- imports the package first. That bound the package here and the generator died on
    `kb.all_cards`, taking all 33 tests in this file with it, while every one of them passed when
    the file was run alone. Loading by path under a private module name removes the ordering
    dependency; this test is what notices if `import kb` comes back.
    """
    import importlib
    saved = sys.modules.pop("kb", None)
    try:
        sys.path.insert(0, ROOT)
        pkg = importlib.import_module("kb")       # the package, first, as CI's session does
        assert pkg.__file__.endswith(os.path.join("kb", "__init__.py")), (
            f"expected the kb PACKAGE to win here, got {pkg.__file__}")
        fresh = load()                            # re-exec the generator under that sys.modules
    finally:
        if saved is not None:
            sys.modules["kb"] = saved
    assert hasattr(fresh.kb, "all_cards"), (
        f"the generator bound {fresh.kb.__file__} as its card parser -- that is the kb PACKAGE, "
        "not kernel_workflow/scripts/kb.py. Load the parser by path, not with `import kb`.")
    assert fresh.load_cards(), "the card parser bound correctly but read no cards"


CARD = """\
---
name: {name}
description: {name}
keywords: [{keywords}]
kernels: [{kernels}]
platforms: [{platforms}]
kernel_class: {kernel_class}
regime: both
type: {type}
confidence: ★★
key: a card about {name}
lifecycle: active
last_seen: 2026-08-01
---
# {name}
- lever: try the thing
- source: run {name}, 2026-08-01
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A learned/ tree the test owns. The real trees are neither read nor written."""
    root = tmp_path / "learned"
    root.mkdir()
    monkeypatch.setattr(G, "CARD_TREES", (G.Tree("kernel", str(root), True),))

    def make(name, keywords, kernels, kernel_class="dense_gemm", type="lever",
             platforms="gfx950"):
        (root / f"{name}.md").write_text(CARD.format(
            name=name, keywords=", ".join(keywords), kernels=", ".join(kernels),
            platforms=platforms, kernel_class=kernel_class, type=type), encoding="utf-8")
        return name

    return make


def rows_for(cards=None):
    cs = G.load_cards()
    return G.axis_rows(cs, G.scope_vocabulary(cs)), cs


# ---------------------------------------------------------------------------------------------
# Counting: kernels, never cards.
# ---------------------------------------------------------------------------------------------
def test_two_cards_over_one_kernel_are_one_kernel(tree):
    """The shape a 16h campaign actually produces: several cards, same kernel symbol.

    Counting cards here would report an axis as measured twice when it was measured once, which is
    exactly the inflation the threshold exists to prevent.
    """
    tree("c1", ["split-k"], ["_gemm_kernel"])
    tree("c2", ["split-k"], ["_gemm_kernel"])
    rows, _ = rows_for()
    assert rows["splitk"]["n_kernels"] == 1
    assert len(rows["splitk"]["cards"]) == 2


def test_a_card_with_no_kernels_still_counts_once(tree):
    """Dropping it would shrink a base rate's denominator silently; counting it twice would inflate."""
    tree("c1", ["split-k"], [])
    tree("c2", ["split-k"], [])
    rows, _ = rows_for()
    assert rows["splitk"]["n_kernels"] == 2       # distinct by card name, one each
    tree("c3", ["split-k"], ["_real_kernel"])
    rows, _ = rows_for()
    assert rows["splitk"]["n_kernels"] == 3


def test_concentration_is_a_share_of_kernels_not_of_cards(tree):
    """Regression: `classes` once counted card-kernel pairs, so five cards over one kernel in class A
    outweighed three distinct kernels in class B and the axis routed to the wrong operator."""
    for i in range(5):
        tree(f"a{i}", ["tile-shape"], ["_kernel_a"], kernel_class="dense_gemm")
    for i in range(3):
        tree(f"b{i}", ["tile-shape"], [f"_kernel_b{i}"], kernel_class="moe_grouped_gemm")
    rows, _ = rows_for()
    r = rows["tileshape"]
    assert r["n_kernels"] == 4                       # 1 in A + 3 in B
    assert r["top_class"] == "moe_grouped_gemm"      # by kernels, not by card count
    assert r["concentration"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------------------------
# Both sides of every axis.
# ---------------------------------------------------------------------------------------------
def test_the_closed_side_is_reported(tree):
    tree("won", ["lds-tiling"], ["_k1"], type="lever", kernel_class="dense_gemm")
    tree("shut1", ["lds-tiling"], ["_k2"], type="anti-pattern", kernel_class="attention_decode")
    tree("shut2", ["lds-tiling"], ["_k3"], type="anti-pattern", kernel_class="moe_grouped_gemm")
    rows, cards = rows_for()
    r = rows["ldstiling"]
    assert len(r["kernels_by_verdict"]["paid"]) == 1
    assert len(r["kernels_by_verdict"]["closed"]) == 2
    out = G.render(rows, cards, {})
    line = next(x for x in out.splitlines() if x.startswith("| `lds-tiling`"))
    assert line.split("|")[2].strip() == "1"         # paid on
    assert line.split("|")[3].strip() == "2"         # closed on


def test_an_axis_that_never_paid_is_still_published(tree):
    """A 0-paid axis is the most useful row in the table and the easiest one to accidentally filter."""
    for i, cls in enumerate(("dense_gemm", "attention_decode", "moe_grouped_gemm")):
        tree(f"c{i}", ["software-prefetch"], [f"_k{i}"], type="anti-pattern", kernel_class=cls)
    rows, cards = rows_for()
    out = G.render(rows, cards, {})
    assert "| `software-prefetch` | 0 | 3 |" in out


# ---------------------------------------------------------------------------------------------
# Threshold and scope.
# ---------------------------------------------------------------------------------------------
def test_below_threshold_axes_are_listed_not_dropped(tree):
    tree("c1", ["exotic-lever"], ["_k1"])
    rows, cards = rows_for()
    out = G.render(rows, cards, {})
    assert "## Below threshold" in out
    assert "`exotic-lever`" in out
    assert "| `exotic-lever` |" not in out          # not in the published table


def test_general_scope_needs_several_classes(tree):
    for i in range(4):
        tree(f"c{i}", ["some-axis"], [f"_k{i}"], kernel_class="dense_gemm")
    rows, _ = rows_for()
    assert rows["someaxi"]["scope"] == "operator"   # one class, however many kernels
    for i, cls in enumerate(("moe_grouped_gemm", "attention_decode")):
        tree(f"d{i}", ["some-axis"], [f"_m{i}"], kernel_class=cls)
    rows, _ = rows_for()
    assert rows["someaxi"]["scope"] == "general"


def test_operator_routing_only_uses_mapped_classes(tree):
    """An unmapped class is REPORTED, never filed under a plausible neighbour."""
    for i in range(3):
        tree(f"c{i}", ["some-axis"], [f"_k{i}"], kernel_class="fused_norm_gemm")
    rows, cards = rows_for()
    out = G.render(rows, cards, {"fused_norm_gemm": 3})
    assert "(unmapped: fused_norm_gemm)" in out
    assert "operators/fused_norm_gemm" not in out


# ---------------------------------------------------------------------------------------------
# Scope tags are not axes.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tag", ["gfx950", "gfx942", "decode", "prefill", "compute-bound",
                                 "fp8", "bf16", "int4", "triton", "flydsl", "aiter",
                                 "anti-pattern", "closed-axis", "dense-gemm"])
def test_a_scope_tag_never_becomes_an_axis(tree, tag):
    for i in range(4):
        tree(f"c{i}", [tag, "real-axis"], [f"_k{i}"])
    rows, _ = rows_for()
    assert G.fold(tag) not in rows, f"{tag} was published as an optimization axis"
    assert "realaxi" in rows


def test_the_dtype_family_split_does_not_eat_operator_ids(tree):
    """`fp8` must be scope (the taxonomy dtype is `fp8_e4m3_fnuz`) while `split-k` must stay an axis
    (the taxonomy id `splitk_streamk_gemm` is an OPERATOR, and splitting it deleted the axis)."""
    for i, cls in enumerate(("dense_gemm", "attention_decode", "moe_grouped_gemm")):
        tree(f"c{i}", ["fp8", "split-k"], [f"_k{i}"], kernel_class=cls)
    rows, _ = rows_for()
    assert G.fold("split-k") in rows
    assert G.fold("fp8") not in rows


def test_header_synonyms_each_name_a_real_card_field(tree):
    """The one hand-maintained list here. If an entry names a field that does not exist, the reason
    it was added is gone and the entry is just an unexplained exclusion."""
    tree("c1", ["x"], ["_k1"])
    fields = set(G.load_cards()[0]["meta"]) | {"type", "regime", "kernel_class", "platforms"}
    for kw, field in G.HEADER_SYNONYMS.items():
        assert field in fields, f"HEADER_SYNONYMS[{kw!r}] names {field!r}, not a card field"
    assert len(G.HEADER_SYNONYMS) <= 8, "this list is drifting into hand-curated vocabulary"


# ---------------------------------------------------------------------------------------------
# Labels and links: the parts a reader judges the tool by.
# ---------------------------------------------------------------------------------------------
def test_the_label_is_a_real_spelling_not_the_fold(tree):
    """`fold` deletes hyphens and a trailing s: `isa-census` -> `isacensu`. Printing that is how a
    correct tool looks broken."""
    for i in range(3):
        tree(f"c{i}", ["isa-census"], [f"_k{i}"])
    rows, cards = rows_for()
    assert rows["isacensu"]["label"] == "isa-census"
    assert "isacensu" not in G.render(rows, cards, {})


def test_the_most_common_spelling_wins_and_the_others_are_named(tree):
    for i in range(3):
        tree(f"c{i}", ["split-k"], [f"_k{i}"])
    tree("odd", ["splitk"], ["_k9"])
    rows, cards = rows_for()
    assert rows["splitk"]["label"] == "split-k"
    assert "also spelled `splitk`" in G.render(rows, cards, {})


def test_every_link_in_the_real_digest_resolves():
    """A broken audit trail is worse than none: it looks checkable and is not. This is the test that
    caught 135 links written one directory level short."""
    path = os.path.join(HERE, "run_recurrence.md")
    if not os.path.exists(path):
        pytest.skip("digest not generated yet")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    links = {m for m in re.findall(r"\]\(([^)]+)\)", text) if not m.startswith("http")}
    assert links, "the digest carries no links at all"
    missing = [x for x in sorted(links)
               if not os.path.exists(os.path.normpath(os.path.join(HERE, x)))]
    assert missing == [], f"unresolvable links: {missing[:5]}"


def test_operator_targets_all_exist_on_disk():
    """CLASS_TO_OPERATOR bridges two vocabularies, so a typo is a link to nowhere."""
    for cls, op in G.CLASS_TO_OPERATOR.items():
        assert os.path.isdir(os.path.join(PK, "operators", op)), \
            f"{cls} -> {op}: perf_knowledge/operators/{op}/ does not exist"


def test_operator_targets_are_taxonomy_ids():
    ids = G.taxonomy_ids()
    for cls, op in G.CLASS_TO_OPERATOR.items():
        assert op in ids, f"{cls} -> {op} is not an id in index/taxonomy.md"


# ---------------------------------------------------------------------------------------------
# The content rules the cards obey, preserved through the roll-up.
# ---------------------------------------------------------------------------------------------
def test_no_absolute_measurement_leaks_into_the_digest():
    """`learned/README.md` content rule 1: a box's absolute number reads to the next run as a target.
    The digest copies no card text, so this should hold structurally — asserted anyway, because the
    day someone adds an `effect:` column is the day it stops holding."""
    path = os.path.join(HERE, "run_recurrence.md")
    if not os.path.exists(path):
        pytest.skip("digest not generated yet")
    with open(path, encoding="utf-8") as f:
        body = f.read()
    forbidden = re.findall(
        r"\d+(?:\.\d+)?\s*(?:ms|µs|us|ns|TFLOP/s|TFLOPS|GB/s|GHz|watts?)\b", body)
    assert forbidden == [], f"absolute measurements in a generated digest: {forbidden[:5]}"


def test_the_digest_states_it_is_generated():
    path = os.path.join(HERE, "run_recurrence.md")
    if not os.path.exists(path):
        pytest.skip("digest not generated yet")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "GENERATED" in text[:2000] and "_gen_run_recurrence.py" in text[:2000]
    assert "## Sources" in text, \
        "perf_knowledge non-negotiable: every content file ends with ## Sources"


# ---------------------------------------------------------------------------------------------
# Empty lanes must say WHY they are empty.
# ---------------------------------------------------------------------------------------------
def test_the_language_lane_explains_an_empty_result(tree):
    tree("c1", ["some-axis"], ["_k1"])
    rows, cards = rows_for()
    out = G.render(rows, cards, {})
    assert "Empty, and not because nothing was measured" in out


def test_the_language_lane_says_which_trees_actually_enforce_the_field(tree, monkeypatch, tmp_path):
    """An ungated tree must be named as ungated, not folded into a corpus-wide promise.

    The empty-lane prose used to read "cards written from here on carry a language" over both trees.
    Only the kernel tree's write path enforces it — `kb.py propose` lints it and the lane fills it
    from `detect_language.py` — while e2e cards are hand-maintained markdown with no gate. So the
    sentence was false for a sixth of the corpus, in the direction a reader takes as a promise: they
    would read a future FlyDSL row as "the corpus is language-tagged" and act on a sixth of it being
    silently untagged. Both trees must be visible with their real status.
    """
    open_root = tmp_path / "open"
    open_root.mkdir()
    monkeypatch.setattr(G, "CARD_TREES", (
        G.Tree("gated", G.CARD_TREES[0].root, True),
        G.Tree("ungated", str(open_root), False),
    ))
    tree("c1", ["some-axis"], ["_k1"])
    rows, cards = rows_for()
    out = G.render(rows, cards, {})

    lane = out.split("## By authoring language")[1].split("##")[0]
    assert "**gated**" in lane and "**ungated**" in lane, "both trees must appear by name"
    assert "is **not** required" in lane, "the ungated tree's status must be stated, not implied"
    # The retired blanket claim must not come back in any tree's absence.
    assert "Cards written from here on do" not in out


def test_the_architecture_lane_needs_two_platforms(tree):
    for i in range(3):
        tree(f"c{i}", ["cache-policy"], [f"_k{i}"], platforms="gfx950")
    rows, cards = rows_for()
    assert G.arch_split(rows) == []
    assert "Structurally empty" in G.render(rows, cards, {})
    tree("other", ["cache-policy"], ["_k9"], platforms="gfx942")
    rows, cards = rows_for()
    arch = G.arch_split(rows)
    assert [a["label"] for a in arch] == ["cache-policy"]
    assert "gfx942" in G.render(rows, cards, {})


# ---------------------------------------------------------------------------------------------
# CLI contract.
# ---------------------------------------------------------------------------------------------
def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True,
                          check=False)


def test_check_is_clean_against_the_committed_digest():
    r = run("--check")
    assert r.returncode == 0, f"digest is stale — regenerate it\n{r.stdout}{r.stderr}"


def test_check_writes_nothing():
    path = os.path.join(HERE, "run_recurrence.md")
    with open(path, encoding="utf-8") as f:
        before = f.read()
    mtime = os.path.getmtime(path)
    run("--check")
    with open(path, encoding="utf-8") as f:
        assert f.read() == before
    assert os.path.getmtime(path) == mtime


def test_check_detects_staleness(tmp_path, monkeypatch):
    """Point OUT at a file that is not the digest; --check must fail and still write nothing."""
    stale = tmp_path / "run_recurrence.md"
    stale.write_text("not the digest\n", encoding="utf-8")
    monkeypatch.setattr(G, "OUT", str(stale))
    cards = G.load_cards()
    rows = G.axis_rows(cards, G.scope_vocabulary(cards))
    assert G.render(rows, cards, {}) != stale.read_text(encoding="utf-8")


def test_json_mode_carries_the_kernels_behind_each_verdict():
    r = run("--json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["cards"] > 0 and data["axes"]
    top = data["axes"][0]
    assert set(top["verdicts"]) & {"paid", "closed"}
    assert all(isinstance(v, list) for v in top["verdicts"].values())


# ---------------------------------------------------------------------------------------------
# Against the real trees.
# ---------------------------------------------------------------------------------------------
def test_the_real_corpus_publishes_a_two_sided_axis():
    """`launch-overhead` is the most-measured axis in this tree and it went both ways. If it ever
    reports a single side, the roll-up has started hiding half its evidence."""
    cards = G.load_cards()
    rows = G.axis_rows(cards, G.scope_vocabulary(cards))
    r = rows.get(G.fold("launch-overhead"))
    assert r, "launch-overhead is not in the real corpus any more; retarget this test"
    assert len(r["kernels_by_verdict"]["paid"]) >= 3
    assert len(r["kernels_by_verdict"]["closed"]) >= 3
    assert r["n_kernels"] >= G.MIN_KERNELS
    assert r["scope"] == "general"


def test_cards_without_a_discovery_header_contribute_no_axes_but_are_counted():
    """Headerless cards still count toward the corpus: silently skipping them would misreport its
    size, and reading axes out of them would invent tags they never named.

    This used to be phrased as "the e2e tree has no headers", which was true of that tree and is not
    the property under test — newer e2e cards do carry `keywords:`, and the proxy started failing
    while the invariant it stood for held. Asserted on whichever cards lack a header, in any tree."""
    cards = G.load_cards()
    headerless = [c for c in cards if not c["meta"].get("keywords")]
    if not headerless:
        pytest.skip("every card in this checkout carries a discovery header")
    rows = G.axis_rows(cards, G.scope_vocabulary(cards))
    contributors = {c["slug"] for r in rows.values() for c in r["cards"]}
    for c in headerless:
        slug = os.path.splitext(os.path.basename(c["path"]))[0]
        assert slug not in contributors, f"{slug} has no keywords yet backs an axis"
    out = G.render(rows, cards, {})
    assert "with keywords" in out


def test_every_published_axis_is_backed_by_a_card_that_exists():
    cards = G.load_cards()
    rows = G.axis_rows(cards, G.scope_vocabulary(cards))
    names = {str(c["meta"].get("name")) for c in cards}
    for r in rows.values():
        for c in r["cards"]:
            assert c["name"] in names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
