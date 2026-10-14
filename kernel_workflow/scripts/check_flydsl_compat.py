#!/usr/bin/env python3
"""Statically check a candidate's FlyDSL imports and referenced module attributes.

This is an early compatibility gate, not a compiler substitute. It catches recipes copied across
FlyDSL versions when a module or public symbol is absent before a GPU compile is funded. Dynamic
values and call semantics still require the immutable correctness test.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path


def python_files(paths):
    seen = set()
    for raw in paths:
        path = Path(raw)
        candidates = path.rglob("*.py") if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if candidate.suffix == ".py" and resolved not in seen:
                seen.add(resolved)
                yield candidate


def is_flydsl_module(name):
    # Same scope as detect_language's FlyDSL markers: FlyDSL itself, and kernels written in it that
    # a package ships under a `flydsl` subpackage (aiter.ops.flydsl.*), which the author role reuses.
    return "flydsl" in name.split(".")


def guarded_imports(tree):
    """Import nodes the source itself declares optional, i.e. wrapped in `try/except ImportError`.

    Such an import is NOT a compatibility claim -- the author already wrote what happens when the
    module is absent, and the except branch is reached, not an exception. Reporting it as a missing
    API fails a kernel for a module it never required. Measured: a fused-MoE seam guarding
    `from aiter.ops.flydsl.utils import is_flydsl_available` with a `def is_flydsl_available():
    return False` fallback was failed by this gate on a stack where that module does not exist,
    while the kernel imported, compiled and ran.

    A handler counts as a guard when it catches ImportError (or its subclass ModuleNotFoundError),
    Exception, BaseException, or is bare -- the same set that actually swallows a failed import.
    """
    guard_names = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

    def catches_import_error(handler):
        if handler.type is None:
            return True
        exceptions = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        for exception in exceptions:
            # `except ImportError` and `except builtins.ImportError` both reach here.
            name = exception.attr if isinstance(exception, ast.Attribute) else getattr(
                exception, "id", None)
            if name in guard_names:
                return True
        return False

    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not any(map(catches_import_error, node.handlers)):
            continue
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(child))
    return guarded


def collect_references(paths):
    imports = []
    attributes = []
    parse_errors = []
    for path in python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue
        aliases = {}
        # A guarded import binds nothing we can check: the name may come from the except branch's
        # fallback, so its attributes are skipped along with the import itself.
        optional = guarded_imports(tree)
        for node in ast.walk(tree):
            if id(node) in optional:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if is_flydsl_module(alias.name):
                        local = alias.asname or alias.name.split(".")[0]
                        # `import flydsl.expr` binds the top-level name `flydsl`; an explicit alias
                        # (`as fx`) binds the dotted module itself.
                        aliases[local] = (
                            alias.name if alias.asname else alias.name.split(".")[0],
                            None,
                        )
                        imports.append({
                            "file": str(path), "line": node.lineno, "module": alias.name,
                            "name": None,
                        })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if is_flydsl_module(module):
                    for alias in node.names:
                        imports.append({
                            "file": str(path), "line": node.lineno, "module": module,
                            "name": alias.name,
                        })
                        aliases[alias.asname or alias.name] = (module, alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = []
            root = node
            while isinstance(root, ast.Attribute):
                parts.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in aliases:
                continue
            module, imported_name = aliases[root.id]
            path_parts = ([imported_name] if imported_name else []) + list(reversed(parts))
            if path_parts:
                attributes.append({
                    "file": str(path), "line": node.lineno, "module": module,
                    "name": ".".join(path_parts), "path": path_parts,
                })
    return imports, attributes, parse_errors


# Both spellings are accepted, nearest-first within a directory. The visible one is listed first
# because a workspace is usually populated with `cp -r task/* workspace/`, which silently drops
# dotfiles -- and a stack declaration that does not survive the copy is worse than none, since the
# gate then answers on the ambient stack without saying so.
STACK_FILES = ("geak_pythonpath.txt", ".geak_pythonpath")


def pin_task_stack(paths):
    """Resolve imports against the stack the TASK pins, not the box's ambient one.

    A task dir may declare the exact aiter/FlyDSL tree its kernel was written against by dropping a
    `.geak_pythonpath` next to its oracle: one absolute directory per line, `#` comments allowed,
    highest priority first. Every such directory is prepended to `sys.path` here, before any
    resolution happens.

    Without this the gate silently answers a different question than the benchmark does. Measured:
    a GLM-5.2 MXFP4 fused-MoE task whose oracle pins aiter_main + flydsl 0.3.4.1 was failed by this
    gate as `missing_api` (no `buffer_ops`, no `mxfp4_kname`, ...) purely because the box's ambient
    PYTHONPATH pointed at an older aiter that predates those modules. The kernel imported and ran
    fine. A gate that rejects a working kernel on the wrong stack is worse than no gate: it spends
    the whole budget on nothing and reports a false incompatibility.

    Returns the directories actually pinned, so the caller can report what it resolved against.
    """
    roots, pinned = [], []
    for raw in paths or [Path.cwd()]:
        path = Path(raw).resolve()
        roots.extend([path] if path.is_dir() else [path.parent])
    for root in roots:
        found = False
        for directory in [root, *root.parents]:
            for name in STACK_FILES:
                stack = directory / name
                if not stack.is_file():
                    continue
                try:
                    lines = stack.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for line in lines:
                    entry = line.split("#", 1)[0].strip()
                    if entry and Path(entry).is_dir() and entry not in pinned:
                        pinned.append(entry)
                found = True
            if found:
                break  # nearest declaration wins; do not merge stacks up the tree
    for entry in reversed(pinned):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return pinned


@contextlib.contextmanager
def safe_import_path():
    """Exclude task roots whose `unittest.py` shadows Python's stdlib package."""
    old = list(sys.path)
    cwd = Path.cwd()
    filtered = []
    for raw in old:
        candidate = Path(raw).resolve() if raw else cwd.resolve()
        if (candidate / "unittest.py").is_file():
            continue
        filtered.append(raw)
    sys.path[:] = filtered
    try:
        yield
    finally:
        sys.path[:] = old


def package_info():
    try:
        with safe_import_path():
            module = importlib.import_module("flydsl")
    except Exception as exc:
        return {
            "available": False, "version": "not_captured", "module": "not_captured",
            "provider": "not_captured", "error": repr(exc),
        }
    try:
        version = importlib.metadata.version("flydsl")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "not_captured"))
    module_path = str(getattr(module, "__file__", "not_captured"))
    provider = "aiter_vendored_flydsl" if "/aiter/" in module_path.replace("\\", "/").lower() \
        else "standalone_flydsl"
    return {
        "available": True, "version": version, "module": module_path, "provider": provider,
    }


def yields_runtime_value(value):
    return (not inspect.ismodule(value) and not inspect.isclass(value)
            and hasattr(type(value), "__get__"))


def resolve_reference(reference):
    module_name = reference["module"]
    path = reference.get("path")
    if path is None:
        name = reference.get("name")
        path = [] if not name or name == "*" else [name]
    try:
        with safe_import_path():
            module = importlib.import_module(module_name)
    except Exception as exc:
        return {**reference, "reason": "module_import_failed", "detail": repr(exc)}
    if not path:
        return None
    current = module
    module_prefix = module_name
    for index, name in enumerate(path):
        try:
            # Avoid executing descriptors such as FlyDSL's T.f32, which requires an active MLIR
            # Context merely to evaluate. This gate checks surface presence, not call semantics.
            current = inspect.getattr_static(current, name)
            module_prefix = f"{module_prefix}.{name}"
            if index + 1 < len(path) and yields_runtime_value(current):
                # A property such as MLIR's `ir.Context.current` produces its object only when
                # evaluated; the remaining components name attributes of that object, which a
                # static lookup cannot see. Present as far as it can be checked.
                return None
            continue
        except AttributeError:
            pass
        try:
            inspect.getattr_static(type(current), "__getattr__")
            # Dynamic DSL proxy objects (thread_idx.x, block_idx.y, etc.) intentionally do not
            # publish every legal child as a static Python attribute.
            return None
        except AttributeError:
            pass
        # `from pkg import child_module` and lazy package attributes may require importing the
        # progressively longer module path.
        candidate_module = f"{module_name}." + ".".join(path[:index + 1])
        try:
            with safe_import_path():
                current = importlib.import_module(candidate_module)
            module_prefix = candidate_module
            continue
        except Exception:
            return {
                **reference, "reason": "name_absent", "missing_component": name,
                "resolved_prefix": module_prefix,
            }
    return None


def check(paths):
    package = package_info()
    imports, attributes, parse_errors = collect_references(paths)
    missing = []
    if package["available"]:
        for reference in [*imports, *attributes]:
            failure = resolve_reference(reference)
            if failure and failure not in missing:
                missing.append(failure)
    elif imports or attributes:
        missing.append({"reason": "flydsl_unavailable", "detail": package.get("error")})
    compatible = bool(imports) and package["available"] and not parse_errors and not missing
    return {
        "schema_version": 1,
        "compatible": compatible,
        "flydsl": package,
        "files_scanned": len(list(python_files(paths))),
        "imports_checked": len(imports),
        "attributes_checked": len(attributes),
        "missing": missing,
        "parse_errors": parse_errors,
        "reason": "compatible" if compatible else (
            "parse_error" if parse_errors else
            "no_flydsl_import" if not imports else
            "flydsl_unavailable" if not package["available"] else
            "missing_api"
        ),
    }


def check_package_only():
    package = package_info()
    return {
        "schema_version": 1,
        "compatible": package["available"],
        "flydsl": package,
        "files_scanned": 0,
        "imports_checked": 0,
        "attributes_checked": 0,
        "missing": [] if package["available"] else [{
            "reason": "flydsl_unavailable", "detail": package.get("error"),
        }],
        "parse_errors": [],
        "reason": "package_available" if package["available"] else "flydsl_unavailable",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.package_only and not args.paths:
        parser.error("paths are required unless --package-only is set")
    pinned = pin_task_stack(args.paths)
    result = check_package_only() if args.package_only else check(args.paths)
    result["stack_pinned"] = pinned
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{'PASS' if result['compatible'] else 'FAIL'} "
            f"flydsl={result['flydsl']['version']} reason={result['reason']}"
        )
        for missing in result["missing"]:
            print(f"  {missing}")
    # Structured workflow callers branch on `compatible`; keep JSON available even for a failed
    # gate instead of turning the transport itself into an agent/tool error.
    return 0 if args.json or result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
