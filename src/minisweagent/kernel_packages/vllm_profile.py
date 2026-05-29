"""vLLM profile: wheel-only site-packages install with shadow worktree.

Layout assumed (matches ``vllm/vllm-openai-rocm:v0.19.0`` and similar
wheel-installed images)::

    /usr/local/lib/python3.X/dist-packages/vllm/
        __init__.py
        _C.abi3.so          (~850 MB pre-compiled native extension)
        _moe_C.abi3.so
        _rocm_C.abi3.so
        cumem_allocator.abi3.so
        ... rest of the .py source tree ...

Distinguishing characteristics:

  * No ``csrc/`` directory (no source-built install possible).
  * No ``setup.py`` / ``pyproject.toml`` (wheel-installed; not editable).
  * At least one ``_C*.so`` / ``_C*.abi3.so`` binary at the package root.

For this layout we use :func:`shadow_worktree` (copy ``.py``, symlink
``.so``) and add the worktree to ``PYTHONPATH`` at harness-runtime so
``import vllm`` resolves to the agent's edited copy while still
loading baseline binaries via the symlinks.

We set ``skip_install=True`` because there is no marker file to
``pip install -e`` against — running ``ensure_worktree_installed`` on
the shadow tree would either no-op or produce confusing errors.

We also set ``VLLM_USE_PRECOMPILED=1`` defensively to prevent vLLM's
optional auto-rebuild paths from trying to compile against
non-existent sources.
"""

from __future__ import annotations

from pathlib import Path

from .profile import PackageProfile, register
from .shadow_worktree import is_shadow_worktree, shadow_worktree


def _detect_vllm(path: Path) -> bool:
    """Match either:

    * the wheel-installed source dir (e.g. ``/usr/local/.../vllm/``),
    * the post-shadow worktree (carries our marker file).
    """
    if not path.is_dir():
        return False

    # Case 1 — already-shadowed worktree (detect by marker file).
    if is_shadow_worktree(path):
        # Confirm the marker is for vllm specifically.
        try:
            import json

            from .shadow_worktree import SHADOW_MARKER

            payload = json.loads((path / SHADOW_MARKER).read_text(encoding="utf-8"))
            return payload.get("package_name", "").lower() == "vllm"
        except Exception:
            return False

    # Case 2 — original wheel-installed vllm source.
    if not (path / "__init__.py").is_file():
        return False
    if (path / "csrc").is_dir():
        # Source-built install: not the wheel-only profile.  Falls
        # through to the default git-worktree pipeline.
        return False
    # Heuristic: vllm ships a "_C" binary extension at the package root.
    has_vllm_binary = any(path.glob("_C*.so")) or any(path.glob("_C*.abi3.so"))
    if not has_vllm_binary:
        return False
    # Tighten by inspecting the package name in __init__.py.  Avoids
    # false positives on unrelated packages that happen to ship a
    # ``_C.abi3.so`` file (rare but possible).
    try:
        head = (path / "__init__.py").read_text(encoding="utf-8", errors="replace")[:1024]
        return "vllm" in head.lower()
    except Exception:
        return False


VLLM_PROFILE = PackageProfile(
    name="vllm",
    detect=_detect_vllm,
    make_worktree=shadow_worktree,
    runtime_env={
        # Belt-and-suspenders: prevent vllm's optional auto-rebuild
        # paths from kicking in (we have no source to rebuild against).
        "VLLM_USE_PRECOMPILED": "1",
    },
    skip_install=True,
)


register(VLLM_PROFILE)


__all__ = ["VLLM_PROFILE"]
