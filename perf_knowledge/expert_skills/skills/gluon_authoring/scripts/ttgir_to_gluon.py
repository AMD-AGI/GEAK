#!/usr/bin/env python3
"""TTGIR -> Gluon recovery library (tile-programming-triton-gluon).

Pure-text parser + emitter that turns the compiler-inferred layouts (and,
optionally, the auto-pipeline) in a plain-Triton ``.ttgir`` into explicit Gluon
source. This is the automation of the manual recovery map documented in
``references/tile-programming/layout-recipes.md ## TTGIR -> Gluon recovery map``.

Scope / boundaries (see references/tile-programming/compiler-contract.md):
- Recovers LAYOUTS (coalesce/MMA/shared/dot-operand) -- they are explicit TTGIR
  attributes, so the mapping is deterministic and 1:1.
- Recovers the async double-buffer PIPELINE STRUCTURE (opt-in) -- it is physically
  present in the post-pipeliner ``.ttgir`` (``local_alloc`` multi-buffer +
  ``async_copy``/``commit``/``wait``) and Gluon can express it (see pipeline.md).
- Does NOT recover register allocation / spills: that happens in LLVM AFTER
  ``make_ttgir`` and is not present in TTGIR at all. Pressure is governed by the
  recovered layouts + slicing; never inferred here.

No GPU / triton import is required for parsing -- it operates on the ``.ttgir``
text emitted by ``scripts/dump_ir.sh``. Run ``--selftest`` to validate the
parser against bundled samples.
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Low-level value parsing for the MLIR attribute grammar
# --------------------------------------------------------------------------- #


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split ``s`` on ``sep`` ignoring separators nested inside [] <> () {}."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    openers, closers = "[<({", "])})>"  # note: '}' and ')' both close
    closers = "])}>"
    for ch in s:
        if ch in openers:
            depth += 1
        elif ch in closers:
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_value(v: str):
    """Parse an MLIR attribute value into int / bool / nested list / ref-string."""
    v = v.strip()
    if v == "true":
        return True
    if v == "false":
        return False
    if v.startswith("["):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(it) for it in _split_top_level(inner, ",")]
    if v.startswith("#"):
        return v  # layout reference token, e.g. "#mma"
    try:
        return int(v)
    except ValueError:
        return v


def _parse_dict_body(body: str) -> dict:
    """Parse ``key = value, key = value`` (brackets respected)."""
    out: dict = {}
    for kv in _split_top_level(body, ","):
        if "=" not in kv:
            continue
        k, _, val = kv.partition("=")
        out[k.strip()] = _parse_value(val.strip())
    return out


def _pylist(x) -> str:
    """Render a parsed nested list back to Python literal source."""
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, list):
        return "[" + ", ".join(_pylist(e) for e in x) + "]"
    return str(x)


# --------------------------------------------------------------------------- #
# Layout records
# --------------------------------------------------------------------------- #


@dataclass
class Layout:
    name: str          # ttgir name without '#', e.g. "blocked", "mma"
    kind: str          # blocked|amd_mfma|swizzled_shared|padded_shared|linear|dot_op|slice
    attrs: dict
    shape: Optional[list[int]] = None   # resolved from a use site (linear/padded)
    var: str = ""                       # python identifier in the emitted factory

    def to_gluon_expr(self, ref_to_var: dict[str, str]) -> str:
        a = self.attrs
        if self.kind == "blocked":
            return (f"gl.BlockedLayout({_pylist(a['sizePerThread'])}, "
                    f"{_pylist(a['threadsPerWarp'])}, {_pylist(a['warpsPerCTA'])}, "
                    f"{_pylist(a['order'])})")
        if self.kind == "amd_mfma":
            return (f"gl.amd.AMDMFMALayout(version={a['version']}, "
                    f"instr_shape={_pylist(a['instrShape'])}, "
                    f"transposed={_pylist(a.get('isTransposed', False))}, "
                    f"warps_per_cta={_pylist(a['warpsPerCTA'])})")
        if self.kind == "swizzled_shared":
            return (f"gl.SwizzledSharedLayout({a['vec']}, {a['perPhase']}, "
                    f"{a['maxPhase']}, order={_pylist(a['order'])})")
        if self.kind == "padded_shared":
            shape = self.shape if self.shape is not None else "None  # TODO: shape"
            return (f"gl.PaddedSharedLayout({_pylist(a['interval_padding_pairs'])}, "
                    f"{_pylist(a['offset'])}, {_pylist(a.get('block', []))}, "
                    f"{_pylist(shape)})")
        if self.kind == "linear":
            shape = self.shape if self.shape is not None else "None  # TODO: shape"
            return ("gl.DistributedLinearLayout("
                    f"reg_bases={_pylist(a['register'])}, "
                    f"lane_bases={_pylist(a['lane'])}, "
                    f"warp_bases={_pylist(a['warp'])}, "
                    f"block_bases={_pylist(a.get('block', []))}, "
                    f"shape={_pylist(shape)})")
        if self.kind == "dot_op":
            parent = ref_to_var.get(a["parent"], a["parent"].lstrip("#"))
            return (f"gl.DotOperandLayout(operand_index={a['opIdx']}, "
                    f"parent={parent}, k_width={a['kWidth']})")
        if self.kind == "slice":
            parent = ref_to_var.get(a["parent"], a["parent"].lstrip("#"))
            return f"gl.SliceLayout({a['dim']}, {parent})"
        # A layout the TTGIR uses that gluon.language cannot NAME. The load-bearing case is
        # `amd_rotating_shared` (the plain backend's rotating operand-staging layout): gluon.language
        # exposes no rotating-shared constructor, so this is a LANGUAGE-SURFACE gap, not an unfinished
        # converter -- do not read it as a tool TODO to fill. A faithful transcription of a kernel the
        # plain backend lowered with such a layout is not fully expressible in Gluon today. (See
        # references/gluon-negative-patterns.md.) Other unknown kinds land here too.
        return (f"None  # UNSUPPORTED: layout kind {self.kind!r} has no gluon.language constructor "
                f"(language-surface gap, not a tool TODO)")


# --------------------------------------------------------------------------- #
# Parsing the .ttgir
# --------------------------------------------------------------------------- #

# Named preamble defs: ``#name = #ttg.<kind><...>``
_NAMED_RE = re.compile(r"^#(?P<name>\w+)\s*=\s*#ttg\.(?P<kind>\w+)<(?P<rest>.*)>\s*$")
# Inline dot_op anywhere in the body.
_DOTOP_RE = re.compile(r"#ttg\.dot_op<\{(?P<body>[^}]*)\}>")
# Inline slice anywhere in the body.
_SLICE_RE = re.compile(r"#ttg\.slice<\{(?P<body>[^}]*)\}>")
# padded_shared prefix: ``[512:+16]`` or ``[512:+16, 1024:+8]``
_PAD_PAIR_RE = re.compile(r"(\d+)\s*:\s*\+\s*(\d+)")


def _ident(name: str) -> str:
    """A safe python identifier for an emitted layout variable."""
    return name if name.isidentifier() else f"_{name}"


def _parse_named_attr(kind: str, rest: str) -> dict:
    """Parse the ``<...>`` body of a named layout def into an attrs dict."""
    rest = rest.strip()
    if kind == "padded_shared":
        # form: [512:+16] {offset = [...], block = [...]}
        m = re.match(r"\[(?P<pairs>[^\]]*)\]\s*\{(?P<body>.*)\}$", rest, re.S)
        pairs = [[int(i), int(p)] for i, p in _PAD_PAIR_RE.findall(m.group("pairs"))]
        attrs = _parse_dict_body(m.group("body"))
        attrs["interval_padding_pairs"] = pairs
        return attrs
    # generic form: {k = v, ...}
    body = rest
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    return _parse_dict_body(body)


def _shape_from_bases(*basis_lists: list) -> Optional[list[int]]:
    """Derive the per-CTA tile shape from GF(2) bases.

    For a Triton-emitted linear/padded layout the extent along each dim is a
    power of two equal to ``2 * max_stride`` (1 if no basis touches the dim).
    This is exact for these layouts and needs no use-site scan.
    """
    rank = None
    for bl in basis_lists:
        if bl:
            rank = len(bl[0])
            break
    if not rank:
        return None
    maxs = [0] * rank
    for bl in basis_lists:
        for b in bl:
            for i, v in enumerate(b):
                maxs[i] = max(maxs[i], v)
    return [2 * m if m > 0 else 1 for m in maxs]


def parse_layouts(ttgir: str) -> list[Layout]:
    """Parse all recoverable layouts from a ``.ttgir`` string, in stable order."""
    layouts: list[Layout] = []
    seen_named: set[str] = set()

    # 1) named preamble defs
    for line in ttgir.splitlines():
        m = _NAMED_RE.match(line.strip())
        if not m:
            continue
        kind, name = m.group("kind"), m.group("name")
        if kind == "shared_memory":
            continue  # #smem marker, no Gluon analogue
        attrs = _parse_named_attr(kind, m.group("rest"))
        lay = Layout(name=name, kind=kind, attrs=attrs, var=_ident(name))
        if kind == "linear":
            lay.shape = _shape_from_bases(attrs.get("register", []), attrs.get("lane", []),
                                          attrs.get("warp", []), attrs.get("block", []))
        elif kind == "padded_shared":
            lay.shape = _shape_from_bases(attrs.get("offset", []))
        layouts.append(lay)
        seen_named.add(name)

    # 2) inline dot_op (dedup by signature) -> dot_op_<opIdx>
    seen_dot: set[tuple] = set()
    for m in _DOTOP_RE.finditer(ttgir):
        a = _parse_dict_body(m.group("body"))
        key = (a["opIdx"], a["parent"], a["kWidth"])
        if key in seen_dot:
            continue
        seen_dot.add(key)
        layouts.append(Layout(name=f"dot_op_{a['opIdx']}", kind="dot_op", attrs=a,
                              var=f"dot_op_{a['opIdx']}"))

    return layouts


_NUM_WARPS_RE = re.compile(r'["\']?ttg\.num[-_]warps["\']?\s*=\s*(\d+)')
_NUM_STAGES_RE = re.compile(r'num[-_]stages\s*=?\s*(\d+)')


def parse_schedule_targets(ttgir: str) -> dict:
    """Recover the plain config that the pipeline + slicing layers should target.

    ``num_warps`` (module attr) sets the occupancy/slicing target; ``num_stages``
    (if still present) sets the starting pipeline depth. ``None`` when absent (e.g.
    ``num_stages`` is often consumed by the time the post-pipeliner TTGIR is dumped
    -- in that case read the buffer count via ``parse_pipeline``).
    """
    nw = _NUM_WARPS_RE.search(ttgir)
    ns = _NUM_STAGES_RE.search(ttgir)
    return {
        "num_warps": int(nw.group(1)) if nw else None,
        "num_stages": int(ns.group(1)) if ns else None,
    }


def emit_layout_factory(layouts: list[Layout], source: str = "") -> str:
    """Emit Gluon ``gl.constexpr`` layout definitions for the recovered layouts."""
    ref_to_var = {f"#{lay.name}": lay.var for lay in layouts if lay.kind != "dot_op"}
    lines = [
        "# Auto-recovered Gluon layouts from the plain-Triton TTGIR.",
        f"# source: {source}" if source else "# source: <ttgir>",
        "# Generated by scripts/ttgir_to_gluon.py -- review before use.",
        "from triton.experimental.gluon import language as gl",
        "",
    ]
    for lay in layouts:
        expr = lay.to_gluon_expr(ref_to_var)
        lines.append(f"{lay.var}: gl.constexpr = {expr}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Pipeline-structure recovery (P4, opt-in)
# --------------------------------------------------------------------------- #
#
# The software pipeliner (lib/Dialect/TritonGPU/Transforms/Pipeliner) materializes
# the double buffer directly in the post-make_ttgir TTGIR:
#   ttg.local_alloc -> !ttg.memdesc<NBUFFERSxTILE..., #shared, #smem, mutable>
#   ttg.memdesc_index %smem[%idx]
#   amdg.buffer_load_to_local / ttg.async_copy_global_to_local  [mask = ...]
#   ttg.async_commit_group ; ttg.async_wait {num = N}
#   ttg.local_load -> tensor<..., #ttg.dot_op<{opIdx, parent, kWidth}>>
# Gluon can express all of these (gl.allocate_shared_memory([nBuffers, ...]),
# .index(), gl.amd.cdna4.async_copy.{buffer_load_to_shared,commit_group,wait_group},
# .load(dot_op)), so the staging is mechanically recoverable. The kernel-specific
# addressing (base ptrs / offsets / mask) is NOT in the layout/pipeline structure;
# it comes from the algorithm skeleton, so it is left as TODO here.
#
# Methodology: the faithful layouts-only anchor stays the attribution baseline
# (built first, in transcribe); the pipeline layer then REPRODUCES this recovered
# structure as its starting point and IMPROVES on it (deeper buffering / operand
# prefetch / manual interleave). So this emitter is the standard pipeline-layer
# start (recover_gluon.py --with-pipeline), not a discouraged opt-in -- just keep it
# out of the transcription step so gains stay attributable.
# See references/tile-programming/pipeline.md ## Auto-recovering the pipeline structure.


@dataclass
class PipelinePlan:
    detected: bool = False
    n_buffers: int = 1
    wait_num: int = 0
    prologue_commits: int = 0
    masked: bool = False
    copy_op: str = ""                              # buffer_load_to_local | async_copy_global_to_local
    operand_tiles: list = field(default_factory=list)   # [[256, 64], [64, 256]]
    shared_vars: list = field(default_factory=list)     # ["shared", "shared1"]
    dot_op_vars: list = field(default_factory=list)     # ["dot_op_0", "dot_op_1"]
    notes: list = field(default_factory=list)


_LOCAL_ALLOC_RE = re.compile(
    r"ttg\.local_alloc\s*:\s*\(\)\s*->\s*!ttg\.memdesc<([0-9x]+)x[A-Za-z]\w*,\s*#(\w+)")
_ASYNC_WAIT_RE = re.compile(r"ttg\.async_wait\s*\{num\s*=\s*(\d+)")


def parse_pipeline(ttgir: str) -> PipelinePlan:
    """Recover the async double-buffer pipeline structure from a post-pipeliner .ttgir."""
    plan = PipelinePlan()
    bufs = []
    for dims_s, shared_name in _LOCAL_ALLOC_RE.findall(ttgir):
        dims = [int(d) for d in dims_s.split("x")]
        if len(dims) >= 3:
            nb, tile = dims[0], dims[1:]
        else:
            nb, tile = 1, dims
        bufs.append((nb, tile, shared_name))
    if not bufs:
        return plan

    plan.n_buffers = max(nb for nb, _, _ in bufs)
    plan.operand_tiles = [tile for _, tile, _ in bufs]
    plan.shared_vars = [s for _, _, s in bufs]

    if "amdg.buffer_load_to_local" in ttgir:
        plan.copy_op = "buffer_load_to_local"
    elif "async_copy_global_to_local" in ttgir:
        plan.copy_op = "async_copy_global_to_local"

    for_pos = ttgir.find("scf.for")
    head = ttgir[:for_pos] if for_pos >= 0 else ttgir
    loop = ttgir[for_pos:] if for_pos >= 0 else ttgir
    plan.prologue_commits = head.count("ttg.async_commit_group")
    loop_waits = _ASYNC_WAIT_RE.findall(loop)
    all_waits = _ASYNC_WAIT_RE.findall(ttgir)
    plan.wait_num = int(loop_waits[0]) if loop_waits else (int(all_waits[-1]) if all_waits else 0)
    plan.masked = bool(re.search(r"(?:buffer_load_to_local|async_copy_global_to_local)[^\n]*mask\s*=", loop))
    plan.dot_op_vars = [f"dot_op_{i}" for i in sorted({int(x) for x in re.findall(r"dot_op<\{opIdx = (\d+)", ttgir)})]

    plan.detected = bool(plan.copy_op) and ("ttg.async_commit_group" in ttgir)
    if plan.n_buffers <= 1:
        plan.notes.append("single-buffer (no double buffering); emit a 1-stage prefetch scaffold")
    if plan.n_buffers > 2:
        plan.notes.append(f"nBuffers={plan.n_buffers}: re-check l_idx / wait_group(N) per pipeline.md")
    return plan


def emit_pipeline_skeleton(plan: PipelinePlan, layouts: Optional[list[Layout]] = None,
                           source: str = "") -> str:
    """Emit a Gluon prologue/loop/epilogue scaffold matching the recovered pipeline."""
    if not plan.detected:
        return ("# No async pipeline detected in the TTGIR (single-stage / no async_copy).\n"
                "# Nothing to recover; keep the simple per-iteration load+dot.\n")

    nb = plan.n_buffers
    wait = plan.wait_num
    a_tile = plan.operand_tiles[0] if plan.operand_tiles else ["BLOCK_M", "BLOCK_K"]
    b_tile = plan.operand_tiles[1] if len(plan.operand_tiles) > 1 else ["BLOCK_K", "BLOCK_N"]
    sA = plan.shared_vars[0] if plan.shared_vars else "sharedA"
    sB = plan.shared_vars[1] if len(plan.shared_vars) > 1 else "sharedB"
    dA = plan.dot_op_vars[0] if plan.dot_op_vars else "dot_op_0"
    dB = plan.dot_op_vars[1] if len(plan.dot_op_vars) > 1 else "dot_op_1"
    cp = "gl.amd.cdna4.async_copy"
    maskA = ", mask=a_mask" if plan.masked else ""
    maskB = ", mask=b_mask" if plan.masked else ""
    mcmt = "  # TODO: last-iter mask guard" if plan.masked else ""

    L = [
        f"# Auto-recovered async pipeline scaffold (nBuffers={nb}, wait_group({wait}), "
        f"copy={plan.copy_op}).",
        f"# source: {source or '<ttgir>'}",
        "# Mechanical staging is filled; replace every TODO with the kernel's base/offset/mask",
        "# expressions (they come from the algorithm skeleton, not the pipeline structure).",
        "# Buffer indices are compile-time literals so the scheduler can prove overwrite-safety",
        "# (see references/tile-programming/pipeline.md ## Hand-built buffering rules).",
    ]
    for n in plan.notes:
        L.append(f"# NOTE: {n}")
    L += [
        f"nBuffers: gl.constexpr = {nb}",
        f"smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, {a_tile[0]}, {a_tile[1]}], {sA})",
        f"smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, {b_tile[0]}, {b_tile[1]}], {sB})",
        "",
        "# --- prologue: issue the first nBuffers prefetches (literal buffer index) ---",
    ]
    for i in range(nb):
        L += [
            f"{cp}.buffer_load_to_shared(smemA.index({i}), a_base, a_offsets)  # TODO: a_base for k={i}",
            f"{cp}.buffer_load_to_shared(smemB.index({i}), b_base, b_offsets)  # TODO: b_base for k={i}",
            f"{cp}.commit_group()",
            "# TODO: a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk",
        ]
    L += [
        "",
        f"{cp}.wait_group({wait})",
        f"a = smemA.index(0).load({dA})",
        f"b = smemB.index(0).load({dB})",
        "",
        "for k in range(0, iterMax - (nBuffers - 1)):",
        "    g_idx = k % nBuffers          # buffer being (re)filled",
        "    l_idx = (k + 1) % nBuffers    # buffer consumed this iter; verify for nBuffers>2",
        "    acc = gl.amd.cdna3.mfma(a, b, acc)",
        f"    {cp}.wait_group({wait})",
        f"    {cp}.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets{maskA}){mcmt}",
        f"    {cp}.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets{maskB}){mcmt}",
        f"    {cp}.commit_group()",
        f"    a = smemA.index(l_idx).load({dA})",
        f"    b = smemB.index(l_idx).load({dB})",
        "    # TODO: a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk",
        "",
        "# --- epilogue: drain + final compute ---",
        "acc = gl.amd.cdna3.mfma(a, b, acc)",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI / selftest
# --------------------------------------------------------------------------- #

_SAMPLE_SWIZZLE = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
  %a = ttg.local_load %0 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  %b = ttg.local_load %2 : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
"""

_SAMPLE_PADDED = """\
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]], warp = [[1, 0], [2, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]], block = []}>
#smem = #ttg.shared_memory
  %smemA = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
  %1 = amdg.buffer_load_to_local %p[%o] into %0 : <f16>[tensor<256x64xi32, #linear>] -> <256x64xf16, #shared, #smem, mutable>
"""

_SAMPLE_PIPELINE = """\
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
  %smemA = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
  %smemB = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xf16, #shared1, #smem, mutable>
  %1 = amdg.buffer_load_to_local %a[%o] into %0 : <f16>[tensor<256x64xi32, #shared>] -> <256x64xf16, #shared, #smem, mutable>
  %4 = ttg.async_commit_group
  %6 = amdg.buffer_load_to_local %a[%o] into %5 : <f16>[tensor<256x64xi32, #shared>] -> <256x64xf16, #shared, #smem, mutable>
  %9 = ttg.async_commit_group
  %10 = ttg.async_wait {num = 1 : i32}
  %a = ttg.local_load %0 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  %b = ttg.local_load %2 : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
  %b_22:5 = scf.for %k = %c0_i32 to %11 step %c1_i32 iter_args() : i32 {
    %acc = tt.dot %a, %b, %acc : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
    %16 = amdg.buffer_load_to_local %a[%o] mask = %15 into %14 : <f16>[tensor<256x64xi32, #shared>] -> <256x64xf16, #shared, #smem, mutable>
    %20 = ttg.async_commit_group
    %21 = ttg.async_wait {num = 1 : i32}
  }
"""


def _selftest() -> int:
    failures = 0
    out = emit_layout_factory(parse_layouts(_SAMPLE_SWIZZLE), "swizzle-sample")
    for needle in [
        "gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])",
        "gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])",
        "gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])",
        "gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)",
        "gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)",
    ]:
        if needle not in out:
            print(f"SELFTEST FAIL (swizzle): missing {needle!r}")
            failures += 1

    out = emit_layout_factory(parse_layouts(_SAMPLE_PADDED), "padded-sample")
    for needle in [
        "gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]]",
        "shape=[256, 64])",
        "[], [256, 64])",  # PaddedSharedLayout block bases + derived shape
    ]:
        if needle not in out:
            print(f"SELFTEST FAIL (padded): missing {needle!r}")
            failures += 1

    plan = parse_pipeline(_SAMPLE_PIPELINE)
    for cond, msg in [
        (plan.detected, "detected"),
        (plan.n_buffers == 2, f"n_buffers==2 (got {plan.n_buffers})"),
        (plan.wait_num == 1, f"wait_num==1 (got {plan.wait_num})"),
        (plan.masked is True, "masked"),
        (plan.copy_op == "buffer_load_to_local", f"copy_op (got {plan.copy_op!r})"),
        (plan.operand_tiles == [[256, 64], [64, 256]], f"operand_tiles (got {plan.operand_tiles})"),
        (plan.prologue_commits == 2, f"prologue_commits==2 (got {plan.prologue_commits})"),
    ]:
        if not cond:
            print(f"SELFTEST FAIL (pipeline-parse): {msg}")
            failures += 1
    scaffold = emit_pipeline_skeleton(plan, source="pipeline-sample")
    for needle in [
        "nBuffers: gl.constexpr = 2",
        "gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, 256, 64], shared)",
        "gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, 64, 256], shared1)",
        "gl.amd.cdna4.async_copy.wait_group(1)",
        "smemA.index(0).load(dot_op_0)",
        "gl.amd.cdna3.mfma(a, b, acc)",
        "mask=a_mask",
    ]:
        if needle not in scaffold:
            print(f"SELFTEST FAIL (pipeline-emit): missing {needle!r}")
            failures += 1
    # The emitted scaffold (and layout factory) must be syntactically valid Python.
    for label, src in [("layout-factory", emit_layout_factory(parse_layouts(_SAMPLE_SWIZZLE))),
                       ("pipeline-scaffold", scaffold)]:
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"SELFTEST FAIL ({label}-syntax): {e}")
            failures += 1

    print("SELFTEST PASS" if not failures else f"SELFTEST FAIL ({failures})")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ttgir", nargs="?", help="path to a plain-Triton .ttgir")
    ap.add_argument("--pipeline", action="store_true",
                    help="also emit the recovered async double-buffer scaffold (opt-in)")
    ap.add_argument("--selftest", action="store_true", help="run bundled parser self-tests")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if not a.ttgir:
        ap.error("provide a .ttgir path or --selftest")
    with open(a.ttgir) as f:
        text = f.read()
    layouts = parse_layouts(text)
    print(emit_layout_factory(layouts, a.ttgir))
    if a.pipeline:
        print()
        print(emit_pipeline_skeleton(parse_pipeline(text), layouts, a.ttgir))


if __name__ == "__main__":
    main()
