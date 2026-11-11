"""Deployable artifact collection for e2e knowledge records.

This module owns overlay packaging and tuning-file retention. It does not select
records, evaluate kernels, launch servers, or publish to a knowledge plane.
"""

import io
import json
import os
import re
import sys
import tarfile
import tempfile

from kb.store_local import SAFE_COMPONENT_CHARS

_ARTIFACT_KEYS = (("patch", "final_patch", "final.patch"),
                  ("launch", "final_launch_script", "launch.sh"),
                  ("report", "report_path", "report.md"),
                  ("overlay", "final_overlay", "overlay.py"))

OVERLAY_MANIFEST = "_overlay_manifest.json"
OVERLAY_TARBALL = "overlay.tar.gz"
# Build/interpreter caches. Neither survives being moved to another reader's box, and both are
# larger than the source they were derived from.
OVERLAY_SKIP_DIRS = ("__pycache__", ".torch_ext")
# Holds the TemporaryDirectory objects alive for the life of the process. The tarball has to still
# be on disk when the store uploads it, which happens long after _artifact_files() returns, and a
# TemporaryDirectory that goes out of scope deletes its tree immediately.
_PACKED = []


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)\s+import\b|import\s+([A-Za-z_]\w*))",
                        re.MULTILINE)


def _sibling_imports(path: str) -> list:
    """Read top-level import names without importing or compiling overlay code.

    Only plain absolute imports are followed; dotted and relative imports are
    outside this overlay-sibling dependency lookup."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError:
        return []
    return [a or b for a, b in _IMPORT_RE.findall(source)]


def _overlay_modules(manifest_path: str) -> list:
    """Collect overlay-root modules referenced by the manifest and sitecustomize.

    Follow sibling imports transitively, including package sources. Missing
    siblings are third-party dependencies; malformed manifests still allow the
    sitecustomize closure. Never resolve paths outside the overlay root."""
    root = os.path.dirname(manifest_path)
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    seen, out = set(), []
    queue = [str((entry or {}).get("impl_module") or "") for entry in (manifest.get("rebinds") or [])]
    queue.extend(str((entry or {}).get("module") or "") for entry in (manifest.get("installers") or []))
    queue.extend(_sibling_imports(os.path.join(root, "sitecustomize.py")))
    while queue:
        # A submodule rebind (`geak_authored.gemm_flydsl`) is addressed by its top-level package,
        # which is what sits at the overlay root and what has to be packed. An absolute or
        # separator-bearing name is not a module name at all and is refused rather than resolved
        # into somebody's filesystem.
        module = queue.pop(0).split(".")[0]
        if not module or module in seen or os.path.isabs(module) or os.sep in module:
            continue
        as_file, as_pkg = os.path.join(root, module + ".py"), os.path.join(root, module)
        if not os.path.isfile(as_file) and not os.path.isdir(as_pkg):
            continue
        seen.add(module)
        out.append(module)
        if os.path.isfile(as_file):
            queue.extend(_sibling_imports(as_file))
        else:
            for base, dirs, names in os.walk(as_pkg):
                dirs[:] = [d for d in dirs if d not in OVERLAY_SKIP_DIRS]
                for name in names:
                    if name.endswith(".py"):
                        queue.extend(_sibling_imports(os.path.join(base, name)))
    return out


def _pack_overlay(dirpath: str) -> str:
    """Package a deployable overlay under one ``overlay/`` tarball root.

    Include the manifest, sitecustomize, patched modules, rebind/installer modules
    and their sibling dependencies. Exclude benchmark evidence and build caches.
    No manifest returns an empty path. Keep the temporary directory alive until
    process exit so the caller can upload the returned file later."""
    manifest_path = os.path.join(dirpath, OVERLAY_MANIFEST)
    if not os.path.isfile(manifest_path):
        return ""
    holder = tempfile.TemporaryDirectory(prefix="e2e_overlay_")
    _PACKED.append(holder)
    out = os.path.join(holder.name, OVERLAY_TARBALL)
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    packed_manifest = None
    if isinstance(manifest, dict) and manifest.get("installers"):
        from overlay_setup import installer_module_manifest
        packed_manifest = json.dumps(installer_module_manifest(dirpath, manifest), indent=2).encode()

    def _add_tree(tar, root_dir):
        for root, dirs, names in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in OVERLAY_SKIP_DIRS]
            for name in names:
                if name.endswith(".pyc"):
                    continue
                src = os.path.join(root, name)
                tar.add(src, arcname="overlay/" + os.path.relpath(src, dirpath))

    with tarfile.open(out, "w:gz") as tar:
        for name in (OVERLAY_MANIFEST, "sitecustomize.py"):
            src = os.path.join(dirpath, name)
            if name == OVERLAY_MANIFEST and packed_manifest is not None:
                info = tarfile.TarInfo("overlay/" + name)
                info.size = len(packed_manifest)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(packed_manifest))
            elif os.path.isfile(src):
                tar.add(src, arcname="overlay/" + name)
        _add_tree(tar, os.path.join(dirpath, "_patched"))
        for module in _overlay_modules(manifest_path):
            pkg = os.path.join(dirpath, module)
            if os.path.isdir(pkg):
                _add_tree(tar, pkg)
            elif os.path.isfile(pkg + ".py"):
                tar.add(pkg + ".py", arcname="overlay/" + module + ".py")
    return out


TUNING_PREFIX = "tuning/"
# Bounds on what the tuning phase can drag into a record. Tuned tables are small (the a8w8 blockscale
# CSVs run 0.5-1 MB), so a path far outside that is a deploy bundle or a built .so that got listed by
# mistake, and a record is the wrong place to discover it. Both caps report what they dropped rather
# than truncating quietly — a record that silently carries half its lever is the failure being fixed.
TUNING_FILE_MAX = 24
TUNING_FILE_MAX_BYTES = 64 * 1024 * 1024


_UNSAFE_NAME_RE = re.compile("[^%s]" % SAFE_COMPONENT_CHARS)


def _safe_basename(path: str) -> str:
    """Return a filename accepted by the KB store's path validator.

    Preserve the original source path separately; one punctuation-heavy tuned
    filename must not prevent the entire record from being stored."""
    name = _UNSAFE_NAME_RE.sub("_", os.path.basename(path).strip()) or "artifact"
    return name.lstrip(".") or "artifact"      # ".", ".." and dotfiles are rejected the same way


def _tuning_files(result: dict, dropped: list) -> dict:
    """Collect deployable files from accepted tuning, reporting every omission.

    Data tables can live outside the source patch. Retain artifacts and live-tree
    files, bounded by file count and size, and index names to avoid collisions."""
    tuning = result.get("tuning_skillset") or {}
    if not isinstance(tuning, dict) or str(tuning.get("gate") or "") != "accepted":
        return {}
    found, seen = {}, set()
    for path in (list(tuning.get("artifacts") or []) + list(tuning.get("live_tree_files") or [])):
        path = str(path or "")
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            dropped.append("%s (missing)" % path)
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            dropped.append("%s (unreadable)" % path)
            continue
        if size > TUNING_FILE_MAX_BYTES:
            dropped.append("%s (%d bytes > cap)" % (path, size))
            continue
        if len(found) >= TUNING_FILE_MAX:
            dropped.append("%s (over %d-file cap)" % (path, TUNING_FILE_MAX))
            continue
        # Index-prefixed: two tuned tables can share a basename across trees (the deploy bundle's copy
        # and the installed one), and a bare basename would silently drop one of them.
        stored = "%s%02d_%s" % (TUNING_PREFIX, len(found), _safe_basename(path))
        found["tuning:" + path] = (stored, path)
    return found


def _rebind_tuning_artifacts(kernels, files: dict) -> None:
    """Replace local tuning paths with their stored names, preserving provenance.

    The first stored name for a source wins, matching artifact insertion order.
    Keep the untouched path in ``tuning_artifact_source`` for later diagnosis."""
    # First stored name wins, not last. Two roles can hand back the same source path (the deploy
    # bundle's copy of a table and the installed one are one file to `artifacts` + `live_tree_files`),
    # and `files` is built in insertion order, so the first entry is the one whose stored name the
    # rest of the record already refers to. Taking the last would point the kernel at a bundle name
    # that is a duplicate copy of its table under a different index prefix — reproducible, but it
    # makes two names for one artifact and a reader cannot tell which is canonical.
    by_source = {}
    for stored, src in files.values():
        by_source.setdefault(src, stored)
    for k in kernels:
        if not isinstance(k, dict):
            continue
        source = str(k.get("tuning_artifact") or "")
        stored = by_source.get(source) if source else ""
        if stored:
            k["tuning_artifact"] = stored
            k["tuning_artifact_source"] = source


def _artifact_files(a, result: dict) -> dict:
    """Map artifact roles to stored names and local files that actually exist.

    Package overlay directories, report omitted tuning files, and retain explicit
    extra files. Missing paths are never advertised as stored artifacts."""
    found = {}
    for role, field, stored in _ARTIFACT_KEYS:
        path = str(result.get(field) or "")
        if not path:
            continue
        if os.path.isfile(path):
            found[role] = (stored, path)
        elif role == "overlay" and os.path.isdir(path):
            packed = _pack_overlay(path)
            if packed:
                found[role] = (OVERLAY_TARBALL, packed)
    dropped = []
    found.update(_tuning_files(result, dropped))
    for why in dropped:
        sys.stderr.write("e2e_store: tuning artifact NOT carried into the record: %s\n" % why)
    for extra in (getattr(a, "file", None) or []):   # retract recomputes a record but takes no --file
        path = str(extra)
        if os.path.isfile(path):
            found["file:" + os.path.basename(path)] = (os.path.basename(path), path)
    return found
