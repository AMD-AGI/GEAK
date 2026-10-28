"""Hermetic tests for the early FlyDSL API compatibility gate."""

import importlib.util
import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_flydsl_compat", HERE / "check_flydsl_compat.py",
)
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)


def fake_package():
    return {
        "available": True,
        "version": "0.test",
        "module": "/tmp/flydsl/__init__.py",
        "provider": "standalone_flydsl",
    }


def test_collects_imports_and_module_attribute_uses(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import flydsl.expr as fx\n"
        "from flydsl.expr import arith\n"
        "x = fx.thread_idx\n"
        "y = arith.addi(x, x)\n"
    )
    imports, attributes, errors = COMPAT.collect_references([source])
    assert errors == []
    assert {(row["module"], row["name"]) for row in imports} == {
        ("flydsl.expr", None), ("flydsl.expr", "arith"),
    }
    assert {(row["module"], row["name"]) for row in attributes} == {
        ("flydsl.expr", "thread_idx"),
        ("flydsl.expr", "arith.addi"),
    }


def test_active_surface_passes_and_missing_attribute_fails(tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text("import flydsl.expr as fx\nx = fx.thread_idx\n")
    expr = types.SimpleNamespace(thread_idx=object())

    def importer(name):
        if name in {"flydsl", "flydsl.expr"}:
            return expr
        raise ImportError(name)

    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    monkeypatch.setattr(COMPAT.importlib, "import_module", importer)
    assert COMPAT.check([source])["compatible"] is True

    source.write_text("import flydsl.expr as fx\nx = fx.removed_api\n")
    result = COMPAT.check([source])
    assert result["compatible"] is False
    assert result["reason"] == "missing_api"
    assert result["missing"][0]["name"] == "removed_api"


def test_dotted_import_and_from_import_attributes_are_resolved(tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import flydsl.expr\n"
        "from flydsl.expr import arith\n"
        "x = flydsl.expr.thread_idx\n"
        "y = arith.addi(x, x)\n"
    )
    arith = types.SimpleNamespace(addi=lambda a, b: a)
    expr = types.SimpleNamespace(thread_idx=object(), arith=arith)
    flydsl = types.SimpleNamespace(expr=expr)

    def importer(name):
        return {
            "flydsl": flydsl,
            "flydsl.expr": expr,
            "flydsl.expr.arith": arith,
        }[name]

    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    monkeypatch.setattr(COMPAT.importlib, "import_module", importer)
    assert COMPAT.check([source])["compatible"] is True

    source.write_text(
        "from flydsl.expr import arith\n"
        "x = arith.definitely_removed_api()\n"
    )
    result = COMPAT.check([source])
    assert result["compatible"] is False
    assert result["missing"][0]["missing_component"] == "definitely_removed_api"


def test_attributes_of_a_property_value_are_not_reported_missing(tmp_path, monkeypatch):
    """`ir.Context.current` is a property: `__exit__` belongs to the Context it returns at run
    time, not to the property object. The AITER seams that run correctly use exactly this."""
    source = tmp_path / "kernel.py"
    source.write_text(
        "from flydsl._mlir import ir\n"
        "ir.Context.current.__exit__(None, None, None)\n"
    )

    class Context:
        @property
        def current(self):
            return None

    ir = types.SimpleNamespace(Context=Context)
    mlir = types.SimpleNamespace(ir=ir)

    def importer(name):
        if name in {"flydsl", "flydsl._mlir"}:
            return mlir
        raise ImportError(name)

    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    monkeypatch.setattr(COMPAT.importlib, "import_module", importer)
    assert COMPAT.check([source])["compatible"] is True

    source.write_text("from flydsl._mlir import ir\nir.Removed.current\n")
    result = COMPAT.check([source])
    assert result["compatible"] is False
    assert result["missing"][0]["missing_component"] == "Removed"


def test_a_packaged_flydsl_kernel_counts_as_a_flydsl_import(tmp_path, monkeypatch):
    """The author role reuses aiter's production FlyDSL hgemm through its Python entry point, and
    detect_language already reads that import as FlyDSL; the gate must check it, not reject it."""
    source = tmp_path / "kernel.py"
    source.write_text(
        "from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm\n"
        "y = flydsl_hgemm(1, 2)\n"
    )
    gemm_kernels = types.SimpleNamespace(flydsl_hgemm=lambda a, b: a)

    def importer(name):
        if name == "aiter.ops.flydsl.gemm_kernels":
            return gemm_kernels
        raise ImportError(name)

    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    monkeypatch.setattr(COMPAT.importlib, "import_module", importer)
    result = COMPAT.check([source])
    assert result["compatible"] is True
    assert result["imports_checked"] == 1

    source.write_text("from aiter.ops.flydsl.gemm_kernels import flydsl_removed\n")
    result = COMPAT.check([source])
    assert result["compatible"] is False
    assert result["missing"][0]["name"] == "flydsl_removed"

    source.write_text("from aiter.ops.triton.gemm import gemm_a16w16\n")
    assert COMPAT.check([source])["reason"] == "no_flydsl_import"


def test_no_flydsl_import_and_parse_error_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n")
    assert COMPAT.check([plain])["reason"] == "no_flydsl_import"

    broken = tmp_path / "broken.py"
    broken.write_text("def nope(:\n")
    result = COMPAT.check([broken])
    assert result["compatible"] is False
    assert result["reason"] == "parse_error"
    assert result["parse_errors"]


def test_task_unittest_cannot_shadow_stdlib_during_package_probe(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "unittest.py").write_text("raise RuntimeError('shadowed')\n")
    old = list(sys.path)
    sys.path.insert(0, str(task))
    try:
        with COMPAT.safe_import_path():
            assert str(task) not in sys.path
        assert str(task) in sys.path
    finally:
        sys.path[:] = old


def test_json_mode_transports_a_failed_gate_without_nonzero_exit(tmp_path, monkeypatch, capsys):
    source = tmp_path / "kernel.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(COMPAT, "check", lambda paths: {
        "compatible": False,
        "flydsl": fake_package(),
        "missing": [],
        "reason": "no_flydsl_import",
    })
    assert COMPAT.main([str(source), "--json"]) == 0
    assert '"compatible": false' in capsys.readouterr().out


def test_package_only_reports_availability_without_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(COMPAT, "package_info", fake_package)
    assert COMPAT.main(["--package-only", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"compatible": true' in output
    assert '"reason": "package_available"' in output


def test_a_guarded_optional_import_is_not_a_compatibility_claim(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "try:\n"
        "    from flydsl.maybe import helper\n"
        "except ImportError:\n"
        "    def helper():\n"
        "        return None\n"
        "from flydsl.expr import arith\n"
        "z = helper()\n"
    )
    imports, _attributes, errors = COMPAT.collect_references([source])
    assert errors == []
    # Only the unguarded import is a requirement; the guarded one has a working fallback.
    assert {(row["module"], row["name"]) for row in imports} == {("flydsl.expr", "arith")}


def test_guard_forms_that_do_not_swallow_an_import_error_still_count(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "try:\n"
        "    from flydsl.a import one\n"
        "except ValueError:\n"
        "    one = None\n"
        "try:\n"
        "    from flydsl.b import two\n"
        "except (OSError, ModuleNotFoundError):\n"
        "    two = None\n"
        "try:\n"
        "    from flydsl.c import three\n"
        "except:\n"
        "    three = None\n"
    )
    imports, _attributes, _errors = COMPAT.collect_references([source])
    # `except ValueError` does not catch a failed import, so flydsl.a remains a requirement;
    # ModuleNotFoundError in a tuple and a bare except both do.
    assert {(row["module"], row["name"]) for row in imports} == {("flydsl.a", "one")}


def test_task_declared_stack_is_pinned_ahead_of_the_ambient_path(tmp_path):
    task = tmp_path / "task"
    (task / "kernel_src").mkdir(parents=True)
    stack = tmp_path / "pinned_stack"
    stack.mkdir()
    (task / "geak_pythonpath.txt").write_text(
        f"# the tree this kernel was written against\n{stack}\n{tmp_path / 'absent'}\n"
    )
    source = task / "kernel_src" / "kernel.py"
    source.write_text("from flydsl.expr import arith\n")
    old = list(sys.path)
    try:
        pinned = COMPAT.pin_task_stack([source])
        # Declared from a parent of the scanned file, and a directory that does not exist is dropped
        # rather than pinned -- a stale line must not shadow a real one.
        assert pinned == [str(stack)]
        assert sys.path[0] == str(stack)
    finally:
        sys.path[:] = old


def test_the_nearest_stack_declaration_wins_and_stacks_are_not_merged(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "outer" / "inner"
    inner.mkdir(parents=True)
    far, near = tmp_path / "far", tmp_path / "near"
    far.mkdir()
    near.mkdir()
    (outer / "geak_pythonpath.txt").write_text(f"{far}\n")
    (inner / ".geak_pythonpath").write_text(f"{near}\n")
    source = inner / "kernel.py"
    source.write_text("from flydsl.expr import arith\n")
    old = list(sys.path)
    try:
        assert COMPAT.pin_task_stack([source]) == [str(near)]
    finally:
        sys.path[:] = old
