#!/usr/bin/env python3
"""Splice plain's TTGIR software pipeliner into `gluon_to_ttgir`, reversibly.

Why a patch and not a monkeypatch: `gluon_to_ttgir` builds its own pass manager
inline, so there is no seam to wrap -- the pass list only exists inside that
function body. Editing the installed file is what the procedure actually asks an
author to do, and gating the inserted block on an env var means splice-ON and
splice-OFF are the SAME binary, which is the only way the IR diff between them
means anything.

The splice point is version-dependent, and this is measured rather than assumed:
  3.6.0            `gluon_to_ttgir` ends after add_combine_tensor_select_and_if
                   and has NO add_warp_pipeline (absent from libtriton too)
  3.7.0/3.7.1/3.8  insert before add_warp_pipeline

Usage:  patch_reinject.py apply | revert | status
Then:   TRITON_GLUON_SWP=<num_stages>  to arm it at runtime (unset/1 = off).
"""
import inspect
import os
import re
import sys

MARK_A = "# --- BEGIN reinject-swp (gluon-pipeline-0805) ---"
MARK_B = "# --- END reinject-swp (gluon-pipeline-0805) ---"

BLOCK = '''
{ind}{mark_a}
{ind}# Plain's make_ttgir runs add_schedule_loops + add_pipeline unconditionally;
{ind}# gluon_to_ttgir never calls them on any upstream version. Armed by env only,
{ind}# so the unarmed path is byte-identical to stock.
{ind}_swp = int(os.environ.get("TRITON_GLUON_SWP", "0") or 0)
{ind}if _swp > 1:
{ind}    import triton.backends.amd.compiler as _sc
{ind}    _arch = options.arch
{ind}    _async = getattr(_sc, "is_async_copy_enabled", lambda a: False)(_arch)
{ind}    _pp = getattr(_sc, "is_pingpong_schedule_enabled", lambda a, b: False)(_arch, _async)
{ind}    if os.environ.get("TRITON_GLUON_SWP_NOPP") == "1":
{ind}        _pp = False
{ind}    amd.passes.ttgpuir.add_optimize_dot_operands(pm, _arch)
{ind}    amd.passes.ttgpuir.add_schedule_loops(pm, _swp)
{ind}    amd.passes.ttgpuir.add_pipeline(pm, _async, _pp)
{ind}    # Buffer conversion has to come AFTER the pipeliner, exactly as plain orders it
{ind}    # (make_ttgir: schedule_loops #15, pipeline #16, convert_to_buffer_ops #28).
{ind}    # Without this the author is forced to choose: write gl.load and the pipeliner
{ind}    # sees the loads but the anchor pays 64-bit pointer tensors, or write explicit
{ind}    # gl.amd.cdna3.buffer_load -- which the transcription guidance asks for -- and the
{ind}    # pipeliner cannot recognise them at all. Measured 2x2: only tt.load + an
{ind}    # annotated loop pipelines. Restoring plain's order removes the choice.
{ind}    # OPT-IN, not opt-out: arming this on an anchor whose loads are already
{ind}    # gl.amd.cdna3.buffer_load aborts the pass manager outright (measured:
{ind}    # `PassManager::run failed` on both buffer-op cells of the 2x2). So the default
{ind}    # splice is safe for any anchor, and this half is requested only by an anchor
{ind}    # written with gl.load, which is the only shape that can be pipelined anyway.
{ind}    if os.environ.get("TRITON_GLUON_SWP_BUF") == "1":
{ind}        from triton import knobs as _kn
{ind}        passes.common.add_canonicalizer(pm)
{ind}        amd.passes.ttgpuir.add_canonicalize_pointers(pm)
{ind}        passes.common.add_canonicalizer(pm)
{ind}        amd.passes.ttgpuir.add_convert_to_buffer_ops(
{ind}            pm, _arch, _kn.amd.use_buffer_atomics,
{ind}            _kn.amd.buffer_ops_analyze_small_tensor_range)
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


def status():
    path = _target()
    with open(path) as fh:
        src = fh.read()
    import triton
    print(f"triton {triton.__version__}  ->  {path}")
    print("  patched:", MARK_A in src)
    fn = re.search(r"def gluon_to_ttgir.*?(?=\n    @|\n    def |\nclass )", src, re.DOTALL)
    if fn:
        for ln in fn.group(0).splitlines():
            if "add_" in ln and not ln.strip().startswith("#"):
                print("   ", ln.strip())
    return 0


def choose_anchor(body: str):
    """Pick the splice point. Before add_warp_pipeline where it exists (3.7+), else after
    the last add_* call in the function (3.6, which has no warp pipeline at all)."""
    for ln in body.splitlines():
        if "add_warp_pipeline" in ln:
            return ln
    cands = [ln for ln in body.splitlines()
             if "add_" in ln and not ln.strip().startswith("#")]
    return cands[-1] if cands else None


def apply_patch():
    path = _target()
    with open(path) as fh:
        src = fh.read()
    if MARK_A in src:
        print("already patched"); return 0
    fn = re.search(r"( *)def gluon_to_ttgir.*?(?=\n    @|\n    def |\nclass )", src, re.DOTALL)
    if not fn:
        print("ERROR: gluon_to_ttgir not found", file=sys.stderr); return 2
    body = fn.group(0)
    anchor = choose_anchor(body)
    if anchor is None:
        print("ERROR: no add_* call to anchor to", file=sys.stderr); return 2
    ind = anchor[:len(anchor) - len(anchor.lstrip())]
    blk = BLOCK.format(ind=ind, mark_a=MARK_A, mark_b=MARK_B)
    if anchor.strip().startswith("amd.passes.ttgpuir.add_warp_pipeline"):
        new_body = body.replace(anchor, blk.rstrip("\n") + "\n" + anchor, 1)
        where = "before add_warp_pipeline"
    else:
        new_body = body.replace(anchor, anchor + blk.rstrip("\n"), 1)
        where = f"after {anchor.strip()}"
    src2 = src.replace(body, new_body, 1)
    if "\nimport os" not in src2 and "\nimport os\n" not in src2:
        src2 = src2.replace("\nimport ", "\nimport os\nimport ", 1)
    with open(path + ".orig_swp", "w") as fh:
        fh.write(src)
    with open(path, "w") as fh:
        fh.write(src2)
    print(f"patched {path}  ({where})")
    return 0


def revert():
    path = _target()
    if not os.path.exists(path + ".orig_swp"):
        print("no backup; nothing to revert"); return 1
    with open(path + ".orig_swp") as fh:
        backup = fh.read()
    with open(path, "w") as fh:
        fh.write(backup)
    os.remove(path + ".orig_swp")
    for d in (os.path.join(os.path.dirname(path), "__pycache__"),):
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
    print(f"reverted {path}")
    return 0


def selftest():
    """Offline: the version-dependent splice point is the one thing here that can be wrong
    without failing loudly, so pin it on synthetic bodies of both shapes."""
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("patch_reinject selftest")
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
    ck("3.7+ splices before add_warp_pipeline", "add_warp_pipeline" in (a37 or ""), repr(a37))
    a36 = choose_anchor(body_36)
    ck("3.6 (no warp pipeline) falls back to the last add_* call",
       "add_combine_tensor_select_and_if" in (a36 or ""), repr(a36))
    ck("a body with no add_* call is refused rather than guessed",
       choose_anchor("    def gluon_to_ttgir(src, metadata, options):\n        return mod\n") is None)
    ind = a37[:len(a37) - len(a37.lstrip())]
    blk = BLOCK.format(ind=ind, mark_a=MARK_A, mark_b=MARK_B)
    ck("the emitted block is syntactically valid python at that indent",
       _compiles(blk, ind))
    ck("both markers survive formatting", MARK_A in blk and MARK_B in blk)
    ck("the block is armed by env only, so unarmed is byte-identical to stock",
       'os.environ.get("TRITON_GLUON_SWP"' in blk and "if _swp > 1:" in blk)
    ck("buffer conversion is opt-in, not part of the default splice",
       'os.environ.get("TRITON_GLUON_SWP_BUF") == "1"' in blk)
    print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
          + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
    return 1 if fails else 0


def _compiles(blk: str, ind: str) -> bool:
    import textwrap
    try:
        compile(textwrap.dedent(f"def _f(options, pm, amd, passes):\n{blk}"), "<blk>", "exec")
        return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd in ("--selftest", "selftest"):
        sys.exit(selftest())
    sys.exit({"apply": apply_patch, "revert": revert, "status": status}[cmd]())
