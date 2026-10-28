"""Contract tests for _gen_version_map.py.

The generator's value depends on two properties that are easy to lose silently:

  * the cross-version diff must report where a symbol is DEFINED, not where it is forwarded, or a
    re-exported name reads as living in every module that touches it;
  * a replacement note must stop being published the moment the scan disagrees with it, because a
    stale porting instruction ("X is gone, use Y") is worse than no instruction.

Both are checked here against synthetic package trees, so the tests do not depend on any particular
FlyDSL version being unpacked on the box. The last test does read the checked-in version_map.md, to
catch a doc regenerated from notes that no longer validate.
"""
import importlib.util
import os
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def load_gen():
    path = os.path.join(HERE, "_gen_version_map.py")
    spec = importlib.util.spec_from_file_location("flydsl_gen_version_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = load_gen()


def make_tree(root, version, modules):
    """Write a minimal flydsl package: {relative module path: source}."""
    pkg = os.path.join(root, "flydsl")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(f'__version__ = "{version}"\n')
    for rel, src in modules.items():
        path = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(src))
    return root


# --------------------------------------------------------------------------------------------------
# Definition vs re-export, and the opacity flag
# --------------------------------------------------------------------------------------------------

def test_reexport_is_not_reported_as_a_definition(tmp_path):
    root = make_tree(tmp_path / "a", "1.0.0", {
        "expr/__init__.py": "from .arith import add\n",
        "expr/arith.py": "def add(x, y):\n    return x + y\n",
    })
    tree = GEN.scan_tree(str(root))
    defs_init, bound_init, _ = tree["flydsl.expr"]
    defs_arith, _, _ = tree["flydsl.expr.arith"]

    assert "add" in defs_arith, "the defining module must claim the symbol"
    assert "add" not in defs_init, "a forwarding module must not claim it"
    assert "add" in bound_init, "but the forwarded name is still importable from there"


def test_private_and_constant_names_are_importable_but_not_definitions(tmp_path):
    root = make_tree(tmp_path / "a", "1.0.0", {
        "expr/typing.py": "T = object()\n\ndef _to_raw(v):\n    return v\n",
    })
    defs, bound, opaque = GEN.scan_tree(str(root))["flydsl.expr.typing"]

    assert defs == set(), "a constant and a private helper are not public definitions"
    assert {"T", "_to_raw"} <= bound, "but `from ... import T` and `_to_raw` both resolve"
    assert not opaque


@pytest.mark.parametrize("src", [
    "from .arith import *\n",
    "def __getattr__(name):\n    raise AttributeError(name)\n",
])
def test_undecidable_surface_is_flagged_opaque(tmp_path, src):
    root = make_tree(tmp_path / "a", "1.0.0", {"expr/__init__.py": src, "expr/arith.py": "x = 1\n"})
    _defs, _bound, opaque = GEN.scan_tree(str(root))["flydsl.expr"]
    assert opaque, "a star-import or __getattr__ means the module surface cannot be decided statically"


# --------------------------------------------------------------------------------------------------
# Cross-version classification
# --------------------------------------------------------------------------------------------------

def build_pair(tmp_path):
    old = make_tree(tmp_path / "old", "0.1.0", {
        "expr/buffer_ops.py": "def make_rsrc():\n    pass\n\ndef load():\n    pass\n",
        "expr/arith.py": "def keep():\n    pass\n",
    })
    new = make_tree(tmp_path / "new", "0.2.0", {
        "expr/arith.py": "def keep():\n    pass\n\ndef load():\n    pass\n",
        "expr/fresh.py": "def brand_new():\n    pass\n",
    })
    trees = {"0.1.0": GEN.scan_tree(str(old)), "0.2.0": GEN.scan_tree(str(new))}
    return trees, GEN.symbol_index(trees)


def test_classify_splits_gone_moved_and_new(tmp_path):
    _trees, idx = build_pair(tmp_path)
    removed, added, moved = GEN.classify(idx, "0.1.0", "0.2.0")

    assert [n for n, _ in removed] == ["make_rsrc"]
    assert [n for n, _ in added] == ["brand_new"]
    assert [(n, a, b) for n, a, b in moved] == [
        ("load", ["flydsl.expr.buffer_ops"], ["flydsl.expr.arith"])
    ]
    assert "keep" not in {n for n, *_ in removed + added} | {n for n, *_ in moved}


def test_version_ordering_is_numeric_not_lexicographic():
    assert sorted(["0.10.0", "0.2.0", "0.3.0"], key=GEN.version_key) == ["0.2.0", "0.3.0", "0.10.0"]


# --------------------------------------------------------------------------------------------------
# Replacement notes: a note that no longer matches the scan must be dropped AND reported
# --------------------------------------------------------------------------------------------------

def write_notes(tmp_path, body):
    path = tmp_path / "notes.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def load(tmp_path, body):
    _trees, idx = build_pair(tmp_path)
    return GEN.load_notes(write_notes(tmp_path, body), idx, {"0.1.0", "0.2.0"})


def test_valid_note_is_kept(tmp_path):
    kept, errors = load(tmp_path, """
        replacements:
          - from: "0.1.0"
            to: "0.2.0"
            topic: descriptor construction
            gone: [make_rsrc]
            use: [brand_new]
            note: verified against the scan
        """)
    assert errors == []
    assert [n["topic"] for n in kept] == ["descriptor construction"]


def test_note_claiming_a_live_symbol_is_gone_is_rejected(tmp_path):
    kept, errors = load(tmp_path, """
        replacements:
          - from: "0.1.0"
            to: "0.2.0"
            topic: stale removal claim
            gone: [keep]
            use: [brand_new]
            note: "keep still exists in 0.2.0, so this note must not be published"
        """)
    assert kept == []
    assert len(errors) == 1 and "listed as gone" in errors[0]


def test_note_pointing_at_a_nonexistent_replacement_is_rejected(tmp_path):
    kept, errors = load(tmp_path, """
        replacements:
          - from: "0.1.0"
            to: "0.2.0"
            topic: dangling replacement
            gone: [make_rsrc]
            use: [never_existed]
            note: the replacement named here is not in 0.2.0
        """)
    assert kept == []
    assert len(errors) == 1 and "no such symbol" in errors[0]


def test_note_for_an_unscanned_version_pair_is_rejected(tmp_path):
    kept, errors = load(tmp_path, """
        replacements:
          - from: "9.9.9"
            to: "10.0.0"
            topic: unscanned pair
            gone: [make_rsrc]
            use: [brand_new]
            note: neither version was scanned
        """)
    assert kept == []
    assert len(errors) == 1 and "not a scanned version pair" in errors[0]


def test_missing_notes_file_is_not_an_error(tmp_path):
    _trees, idx = build_pair(tmp_path)
    kept, errors = GEN.load_notes(str(tmp_path / "absent.yaml"), idx, {"0.1.0", "0.2.0"})
    assert (kept, errors) == ([], [])


# --------------------------------------------------------------------------------------------------
# The checked-in artifacts
# --------------------------------------------------------------------------------------------------

def test_checked_in_map_publishes_every_note_topic():
    """Guards against committing a map generated while its notes were failing validation."""
    yaml = pytest.importorskip("yaml")
    notes_path = os.path.join(HERE, "version_map_notes.yaml")
    map_path = os.path.join(HERE, "version_map.md")
    if not (os.path.isfile(notes_path) and os.path.isfile(map_path)):
        pytest.skip("version map artifacts not present")

    doc = yaml.safe_load(GEN.read_text(notes_path)) or {}
    rendered = GEN.read_text(map_path)
    for entry in doc.get("replacements") or []:
        assert entry["topic"] in rendered, (
            f"note {entry['topic']!r} is absent from version_map.md — regenerate it, and if the note "
            f"was rejected, fix the note rather than shipping the map without it"
        )


def test_checked_in_map_records_what_it_was_generated_from():
    map_path = os.path.join(HERE, "version_map.md")
    if not os.path.isfile(map_path):
        pytest.skip("version_map.md not present")
    text = GEN.read_text(map_path)
    assert "## Sources" in text, "perf_knowledge requires every content file to end with ## Sources"
    assert "_gen_version_map.py" in text, "the map must name the generator that produced it"
