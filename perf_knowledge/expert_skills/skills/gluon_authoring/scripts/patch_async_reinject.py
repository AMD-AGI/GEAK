#!/usr/bin/env python3
"""Splice AMD's async-copy coalescing pass into `gluon_to_ttgir`, reversibly.

Companion to the skill's `patch_reinject.py`, which restores the software pipeliner. This one
restores a DIFFERENT pass that make_ttgir runs and gluon_to_ttgir does not:

    make_ttgir:                                gluon_to_ttgir:
      add_pipeline(pm, use_async_copy, ...)      (absent)
      if use_async_copy:                         (absent)
          add_coalesce_async_copy(pm, arch)

**Scope, corrected by measurement:** this is NOT what makes async copy work on gfx950. Both
`global_load_to_shared` and `buffer_load_to_shared` lower and are numerically correct on stock
`gluon_to_ttgir` when each lane makes exactly one access of a native direct-to-LDS width -- 4 B or
16 B on gfx950 (`hw_constants.json` `direct_to_lds_bit_widths: [128, 32]`), 4 B only on gfx942.
What fails without coalescing is the **non-native** pattern: 8 B and 32 B per lane, and any layout
whose per-lane contribution is a native size but split across repetitions (a `BlockedLayout`
covering `[64, 16]` on a `[32, 32]` tile makes two accesses per lane and does not lower). That is
the class a coalescing pass exists to repair, and repairing it is not free -- it lands the copy in
an arrangement the read has to bounce through. So prefer fixing the layout to match a native width;
reach for this patch when you cannot, and measure rather than assume it helped.

Armed by env only (TRITON_GLUON_ASYNC=1), so unarmed is byte-identical to stock.

    patch_async_reinject.py apply | revert | status
"""
from __future__ import annotations

import inspect
import os
import re
import sys

MARK_A = "# --- BEGIN reinject-async-coalesce ---"
MARK_B = "# --- END reinject-async-coalesce ---"

BLOCK = '''
{ind}{mark_a}
{ind}# make_ttgir runs add_coalesce_async_copy whenever async copy is on; gluon_to_ttgir
{ind}# never calls it, which leaves ttg.async_copy_global_to_local illegal at LLVM lowering.
{ind}if os.environ.get("TRITON_GLUON_ASYNC") == "1":
{ind}    amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
{ind}{mark_b}
'''


def _target():
    try:
        import triton.backends.amd.compiler as m
    except ImportError as e:
        print(f"no AMD Triton backend importable here ({e}); this command edits an "
              "installed compiler.py and has to run where that file lives",
              file=sys.stderr)
        raise SystemExit(2) from None
    return inspect.getsourcefile(m)


def _fn_body(src):
    return re.search(r"( *)def gluon_to_ttgir.*?(?=\n    @|\n    def |\nclass )", src, re.DOTALL)


def choose_anchor(body: str) -> str | None:
    """The line to splice after: the LAST `add_*` call that is not `add_warp_pipeline`.

    Coalescing only has to run before the module leaves `gluon_to_ttgir`, and ordering among
    the generic passes does not matter -- but it must land ahead of `add_warp_pipeline` where
    that exists (3.7+), which is why that one call is excluded rather than merely not chosen.
    Factored out of `apply_patch` so `--selftest` can pin it on synthetic bodies of both
    shapes: this is the piece that can be wrong without failing loudly, exactly as in the
    sibling `patch_reinject.py`.
    """
    cands = [ln for ln in body.splitlines()
             if "add_" in ln and not ln.strip().startswith("#")
             and "add_warp_pipeline" not in ln]
    return cands[-1] if cands else None


def status():
    path = _target()
    with open(path) as fh:
        src = fh.read()
    import triton
    print(f"triton {triton.__version__} -> {path}")
    print("  async-coalesce patched:", MARK_A in src)
    fn = _fn_body(src)
    if fn:
        for ln in fn.group(0).splitlines():
            if "add_" in ln and not ln.strip().startswith("#"):
                print("   ", ln.strip())
    return 0


def apply_patch():
    path = _target()
    with open(path) as fh:
        src = fh.read()
    if MARK_A in src:
        print("already patched")
        return 0
    fn = _fn_body(src)
    if not fn:
        print("ERROR: gluon_to_ttgir not found", file=sys.stderr)
        return 2
    body = fn.group(0)
    anchor = choose_anchor(body)
    if anchor is None:
        print("ERROR: no add_* call to anchor to", file=sys.stderr)
        return 2
    ind = anchor[:len(anchor) - len(anchor.lstrip())]
    blk = BLOCK.format(ind=ind, mark_a=MARK_A, mark_b=MARK_B)
    new_body = body.replace(anchor, anchor + blk.rstrip("\n"), 1)
    src2 = src.replace(body, new_body, 1)
    if not re.search(r"^import os$", src2, re.MULTILINE):
        src2 = src2.replace("\nimport ", "\nimport os\nimport ", 1)
    with open(path + ".orig_async", "w") as fh:
        fh.write(src)
    with open(path, "w") as fh:
        fh.write(src2)
    print(f"patched {path} (after {anchor.strip()})")
    return 0


def revert():
    path = _target()
    if not os.path.exists(path + ".orig_async"):
        print("no backup; nothing to revert")
        return 1
    with open(path + ".orig_async") as fh:
        orig = fh.read()
    with open(path, "w") as fh:
        fh.write(orig)
    os.remove(path + ".orig_async")
    pyc = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pyc):
        for f in os.listdir(pyc):
            os.remove(os.path.join(pyc, f))
    print(f"reverted {path}")
    return 0


def _compiles(blk: str) -> bool:
    import textwrap
    try:
        compile(textwrap.dedent(f"def _f(options, pm, amd, passes):\n{blk}"), "<blk>", "exec")
        return True
    except SyntaxError:
        return False


def selftest():
    """Offline, no triton: pin the splice point and the arming, the two things here that can
    be wrong while every command still exits 0."""
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("patch_async_reinject selftest")
    body_37 = (
        "    def gluon_to_ttgir(src, metadata, options):\n"
        "        pm = ir.pass_manager(mod.context)\n"
        "        amd.passes.ttgpuir.add_combine_tensor_select_and_if(pm)\n"
        "        amd.passes.ttgpuir.add_warp_pipeline(pm, options.num_stages)\n"
        "        amd.passes.ttgpuir.add_allocate_warp_groups(pm)\n")
    body_36 = (
        "    def gluon_to_ttgir(src, metadata, options):\n"
        "        pm = ir.pass_manager(mod.context)\n"
        "        amd.passes.ttgpuir.add_combine_tensor_select_and_if(pm)\n")
    a37 = choose_anchor(body_37)
    ck("3.7+ anchors on a call that is NOT add_warp_pipeline",
       a37 is not None and "add_warp_pipeline" not in a37, repr(a37))
    ck("3.7+ picks the last eligible call, so coalescing runs after the rest",
       "add_allocate_warp_groups" in (a37 or ""), repr(a37))
    a36 = choose_anchor(body_36)
    ck("3.6 (no warp pipeline) falls back to the last add_* call",
       "add_combine_tensor_select_and_if" in (a36 or ""), repr(a36))
    ck("a body with no add_* call is refused rather than guessed",
       choose_anchor("    def gluon_to_ttgir(src, metadata, options):\n        return mod\n") is None)
    ck("a commented-out add_* call is not an anchor",
       choose_anchor("        # amd.passes.ttgpuir.add_foo(pm)\n") is None)

    ind = a37[:len(a37) - len(a37.lstrip())]
    blk = BLOCK.format(ind=ind, mark_a=MARK_A, mark_b=MARK_B)
    ck("the emitted block is syntactically valid python at that indent", _compiles(blk))
    ck("both markers survive formatting", MARK_A in blk and MARK_B in blk)
    ck("armed by env only, so unarmed is byte-identical to stock",
       'os.environ.get("TRITON_GLUON_ASYNC") == "1"' in blk)
    ck("the pass is called with the arch, which it requires",
       "add_coalesce_async_copy(pm, options.arch)" in blk)

    # The `import os` injection must not fire where the module already imports os, or the
    # patched file grows a duplicate import every time someone re-applies it.
    ck("an existing `import os` is detected and not duplicated",
       re.search(r"^import os$", "import os\nimport re\n", re.MULTILINE) is not None)
    ck("...and a module without it is detected as needing one",
       re.search(r"^import os$", "import re\nimport sys\n", re.MULTILINE) is None)

    # The marker is what makes `apply` idempotent and `status` truthful.
    ck("MARK_A is distinct from the sibling patcher's marker, so the two do not collide",
       "async" in MARK_A and MARK_A != "# --- BEGIN reinject-swp ---")
    print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
          + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd in ("--selftest", "selftest"):
        sys.exit(selftest())
    sys.exit({"apply": apply_patch, "revert": revert, "status": status}[cmd]())
