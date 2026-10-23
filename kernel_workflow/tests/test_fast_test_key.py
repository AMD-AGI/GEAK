"""Deterministic fast-test cache-key contract."""

import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "fast_test_key.py"
SPEC = importlib.util.spec_from_file_location("fast_test_key", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _tree(root: Path, value: str = "x = 1\n") -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text(value)
    (root / "pkg" / "ignored.txt").write_text("not part of frozen digest")
    return root


def test_tree_digest_is_root_path_independent(tmp_path):
    left = _tree(tmp_path / "left")
    right = _tree(tmp_path / "different-name")
    assert MODULE._tree_digest(left) == MODULE._tree_digest(right)


def test_tree_digest_changes_with_python_content(tmp_path):
    left = _tree(tmp_path / "left")
    right = _tree(tmp_path / "right", "x = 2\n")
    assert MODULE._tree_digest(left) != MODULE._tree_digest(right)


def test_json_hash_is_key_order_independent(tmp_path):
    frozen = _tree(tmp_path / "frozen")
    bench = tmp_path / "bench.py"
    bench.write_text("print('bench')\n")
    one = MODULE.compute_key(frozen, bench, '{"b":2,"a":1}', '["g"]')
    two = MODULE.compute_key(frozen, bench, '{"a":1,"b":2}', '["g"]')
    assert one == two
    assert one["algorithm"] == "geak-fast-test-key-v1"
