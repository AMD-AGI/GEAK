"""Execute deferred installers after relocation and checked composition, without GPU dependencies."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import overlay_setup as ov
import e2e_store


def _overlay(root, *, installer=False, legacy=False):
    root.mkdir()
    manifest = {"modules": [], "rebinds": [], "captures": [], "markers": []}
    if installer:
        package = root / "sample_installer"
        package.mkdir()
        (package / "__init__.py").write_text('''import sys, importlib.abc, importlib.util
from pathlib import Path

def install():
    sys.fixture_installed = True
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "incoming_target":
                return importlib.util.spec_from_file_location(name, Path(__file__).with_name("payload.py"))
    sys.meta_path.insert(0, Finder())
''')
        (package / "payload.py").write_text("VALUE = 42\n")
        (package / "kernel.hip").write_text("// opaque packaged kernel source\n")
        manifest["installers"] = [{"id": "sample", "module": "sample_installer", "callable": "install",
                                   "required": True, "targets": ["incoming_target:VALUE"],
                                   "kernels": ["sample_kernel"]}]
    else:
        (root / "_patched").mkdir()
        (root / "_patched/base.py").write_text("VALUE = 41\n")
        manifest["modules"] = [{"module": "base_target", "file": "_patched/base.py"}]
    (root / "_overlay_manifest.json").write_text(json.dumps(manifest))
    (root / "sitecustomize.py").write_text(ov.LEGACY_SITECUSTOMIZE if legacy else ov.SITECUSTOMIZE)
    return root


def _child(overlay, code, *, script_name=None):
    argv = [sys.executable, "-c", code]
    if script_name:
        script = overlay.parent / script_name
        script.write_text(code)
        argv = [sys.executable, str(script)]
    return subprocess.run(argv, env=dict(os.environ, PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1"),
                          cwd=overlay.parent, capture_output=True, text=True)


def test_merge_preserves_old_hooks_and_runs_deferred_installer(tmp_path):
    base = _overlay(tmp_path / "base", legacy=True)
    candidate = _overlay(tmp_path / "incoming", installer=True)
    merged = Path(ov.merge_overlays(str(tmp_path / "relocated"), [str(base), str(candidate)]))
    proc = _child(merged, "import sys; assert 'incoming_target' not in sys.modules; "
                  "import base_target, incoming_target; print(base_target.VALUE, incoming_target.VALUE)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "41 42"
    assert (base / "sitecustomize.py").read_text() == ov.LEGACY_SITECUSTOMIZE
    assert (merged / "sample_installer/kernel.hip").is_file()


def test_installer_only_bundle_packs_all_package_sources(tmp_path):
    import tarfile
    overlay = _overlay(tmp_path / "source", installer=True)
    archive = e2e_store._pack_overlay(str(overlay))
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    with tarfile.open(archive) as tar:
        assert "overlay/sample_installer/kernel.hip" in tar.getnames()
        tar.extractall(materialized, filter="data")
    proc = _child(materialized / "overlay", "import incoming_target; print(incoming_target.VALUE)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "42"
    assert e2e_store._overlay_kernel_names({"final_overlay": str(materialized / "overlay")}) == {"sample_kernel"}
    assert _legacy_loadable(materialized / "overlay")
    assert not _legacy_loadable(overlay)  # packaging did not mutate the input


def test_required_installer_failure_stops_python(tmp_path):
    overlay = _overlay(tmp_path / "bad", installer=True)
    (overlay / "sample_installer/__init__.py").write_text("def install():\n    raise RuntimeError('missing seam')\n")
    proc = _child(overlay, "print('would have run stock')")
    assert proc.returncode == 70
    assert "would have run stock" not in proc.stdout
    assert "missing seam" in proc.stderr


def test_probe_process_does_not_install_deferred_hook(tmp_path):
    overlay = _overlay(tmp_path / "probe", installer=True)
    proc = _child(overlay, "import sys; print(getattr(sys, 'fixture_installed', False))", script_name="rocminfo")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "False"


def test_merge_rejects_target_conflicts_without_writing_output(tmp_path):
    base = _overlay(tmp_path / "base")
    incoming = _overlay(tmp_path / "incoming", installer=True)
    path = incoming / "_overlay_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["installers"][0]["targets"] = ["base_target:VALUE"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicting overlay target"):
        ov.merge_overlays(str(tmp_path / "output"), [str(base), str(incoming)])
    assert not (tmp_path / "output").exists()


def test_merge_rejects_custom_startup_code_instead_of_losing_it(tmp_path):
    base = _overlay(tmp_path / "base")
    (base / "sitecustomize.py").write_text("import custom_hook\n")
    with pytest.raises(ValueError, match="custom sitecustomize"):
        ov.merge_overlays(str(tmp_path / "output"), [str(base)])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("base_target", ["serving:Indexer", "serving:Indexer.forward"])
def test_merge_rejects_rebinding_an_installer_method_or_its_class(tmp_path, base_target):
    base = _overlay(tmp_path / "base")
    (base / "_overlay_manifest.json").write_text(json.dumps({"rebinds": [
        {"target": base_target, "impl_module": "other", "impl_attr": "replacement"}]}))
    incoming = _overlay(tmp_path / "incoming", installer=True)
    path = incoming / "_overlay_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["installers"][0]["targets"] = ["serving:Indexer.forward"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicting overlay target"):
        ov.merge_overlays(str(tmp_path / "output"), [str(base), str(incoming)])


def test_overlay_coverage_does_not_fetch_another_kernel_champion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    overlay = _overlay(tmp_path / "overlay", installer=True)
    monkeypatch.setattr(e2e_store, "_fetch_kernel_patch", lambda *args: pytest.fail("unrelated patch fetched"))
    monkeypatch.setattr(e2e_store, "identity_of", lambda args: {"framework": "sglang"})
    kernels = [{"name": "sample_kernel", "kernel_canonical_id": "other-kernel-kb-entry"}]
    value = {"artifacts": {"overlay": "overlay.tar.gz", "launch": "launch.sh"}}
    result = e2e_store._repro(SimpleNamespace(kernel_store="some-store"),
                              {"final_overlay": str(overlay)}, value, kernels, {}, str(tmp_path))
    assert result["complete"]
    assert result["kernels_in_overlay"] == ["sample_kernel"]
    assert not kernels[0].get("patch_missing")


def test_adding_installer_upgrades_the_old_generic_runtime(tmp_path):
    base = _overlay(tmp_path / "base", legacy=True)
    (base / "x.py").write_text("def install():\n    pass\n")
    ov.cmd_add_installer(type("Args", (), dict(overlay=str(base), base="", id="x", module="x",
                                             callable="install", target=["a:b"], kernel=["k"]))())
    assert (base / "sitecustomize.py").read_text() == ov.SITECUSTOMIZE
    assert json.loads((base / "_overlay_manifest.json").read_text())["installers"][0]["kernels"] == ["k"]


def _legacy_loadable(root):
    """The pre-installers Hyperloom loader contract, kept here without an HL dependency."""
    manifest = json.loads((root / "_overlay_manifest.json").read_text())
    return (root / "sitecustomize.py").is_file() and bool(
        manifest.get("modules") or manifest.get("rebinds") or manifest.get("captures"))


def _legacy_digest(root):
    raw = (root / "_overlay_manifest.json").read_bytes()
    digest = hashlib.sha256(raw)
    spec = json.loads(raw)
    bodies = {entry["file"] for entry in spec.get("modules", [])}
    bodies.update(entry["impl_module"] + ".py" for entry in spec.get("rebinds", []))
    for rel in sorted(bodies):
        digest.update(rel.encode())
        digest.update(hashlib.sha256((root / rel).read_bytes()).digest())
    return digest.hexdigest()[:16]


def test_stock_replay_is_loadable_and_nested_kernel_changes_affect_legacy_digest(tmp_path):
    source = _overlay(tmp_path / "source", installer=True)
    # A real package import, including a relative import, still registers the deferred hook once.
    init = source / "sample_installer/__init__.py"
    init.write_text("from .payload import VALUE\n" + init.read_text())
    merged = Path(ov.merge_overlays(str(tmp_path / "first"), [str(source)]))
    assert _legacy_loadable(merged)
    proc = _child(merged, "import sys, sample_installer, incoming_target; "
                  "assert sample_installer.VALUE == incoming_target.VALUE == 42; "
                  "assert sys.fixture_installed")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("[overlay] installed sample") == 1
    assert "[overlay] injected module sample_installer" in proc.stderr
    again = Path(ov.merge_overlays(str(tmp_path / "again"), [str(source), str(merged)]))
    assert _legacy_digest(again) == _legacy_digest(merged)
    (source / "sample_installer/kernel.hip").write_text("// changed kernel, same __init__\n")
    changed = Path(ov.merge_overlays(str(tmp_path / "changed"), [str(source)]))
    assert _legacy_digest(changed) != _legacy_digest(merged)


@pytest.mark.parametrize("problem", ["missing", "conflicting_module", "symlink"])
def test_installer_compatibility_rejects_broken_sources_atomically(tmp_path, problem):
    source = _overlay(tmp_path / "source", installer=True)
    if problem == "missing":
        (source / "sample_installer/__init__.py").unlink()
    elif problem == "conflicting_module":
        path = source / "_overlay_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["modules"] = [{"module": "sample_installer", "file": "unrelated.py"}]
        path.write_text(json.dumps(manifest))
    else:
        (source / "sample_installer/link.py").symlink_to(source / "sample_installer/payload.py")
    with pytest.raises(ValueError):
        ov.merge_overlays(str(tmp_path / "output"), [str(source)])
    assert not (tmp_path / "output").exists()
