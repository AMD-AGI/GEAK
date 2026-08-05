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
    import triton.backends.amd.compiler as m
    return inspect.getsourcefile(m)


def _fn_body(src):
    return re.search(r"( *)def gluon_to_ttgir.*?(?=\n    @|\n    def |\nclass )", src, re.DOTALL)


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
    # Anchor on the last add_* call: the coalescing pass has to see the async ops before
    # the module leaves gluon_to_ttgir, and ordering among the generic passes does not matter.
    cands = [ln for ln in body.splitlines()
             if "add_" in ln and not ln.strip().startswith("#")
             and "add_warp_pipeline" not in ln]
    if not cands:
        print("ERROR: no add_* call to anchor to", file=sys.stderr)
        return 2
    anchor = cands[-1]
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"apply": apply_patch, "revert": revert, "status": status}[cmd]())
