#!/usr/bin/env python3
"""TTGIR <-> Gluon layout bridge, driven by the compiler instead of by a regex.

WHY THIS EXISTS, next to ttgir_to_gluon.py
------------------------------------------
`ttgir_to_gluon.py` reads the `.ttgir` as TEXT and carries a hand-written mapping
from each `#ttg.<kind>` spelling to a `gl.*` constructor. That mapping is the thing
that rots: it has to track every attribute rename (`isTransposed` vs `isTranspose`),
every generation change (`warpsPerCTA` vs `ctaLayout`), and every field that gets
added upstream. When it falls behind, it does not fail -- it emits a layout that is
missing a field, and the anchor is then silently not the champion.

Upstream already owns that mapping, completely, in C++:
`layoutToGluon()` (`python/src/gluon_ir.cc`) converts any TTGIR layout attribute
into the corresponding `gluon.language` object, field for field, and raises on a
kind it does not handle. It is reachable from Python today, with no rebuild, as
`GluonOpBuilder.get_gluon_layout_from_tensor / _from_memdesc`.

So this tool does no layout parsing at all:

    .ttgir --[ir.parse_mlir_module]--> MLIR module   (the compiler parses it)
           --[walk + get_result]-----> Value
           --[layoutToGluon]---------> gl.BlockedLayout(...) / gl.amd.AMDMFMALayout(...)

Three consequences worth stating, because they are the reason to switch:

  1. No mapping to maintain, and no way to be silently behind upstream. A layout
     kind with no Gluon constructor surfaces as an explicit UNRECOVERABLE row
     naming the kind -- never as a plausible-looking wrong constructor.
  2. Shapes come for free. The encoding arrives attached to its use site
     (`tensor<128x64xf16, #ttg.blocked<...>>`), so `linear` / `padded_shared` no
     longer need a shape guessed from GF(2) bases.
  3. Equivalence can be SEMANTIC. `to_linear_layout()` normalises any layout to its
     LinearLayout, so `verify` compares normal forms. Counting attribute
     occurrences -- which is what a text diff ends up doing -- compares the plain
     pipeliner's unroll factor and can never pass against a pre-pipeline anchor.

MODES
-----
  recover   .ttgir -> Gluon layout constants (+ JSON facts), with per-use-site
            provenance and a round-trip proof for every layout.
  verify    plain.ttgir vs anchor.ttgir, compared as LinearLayout normal forms.
            Multiplicity is REPORTED and never gates.
  view      render a layout as an ASCII per-lane table (`get_layout_view`), to
            choose between candidate shared layouts by their access pattern.
  --selftest  offline, no GPU, no Triton required for the pure parts.

REQUIREMENTS
------------
Needs the installed Triton (`import triton`), because the whole point is to use the
compiler's own attribute parser and layout classes. No GPU, no kernel launch, no
compile -- so it runs wherever `import triton` works, including inside the
container while the host has no torch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 0. context / backend plumbing
# --------------------------------------------------------------------------- #

# Ops whose result or operand types carry a layout we care about, and the role hint
# that op position implies. The hint is a HINT: the authoritative facts are always
# (op, index, shape, layout), which are printed next to it. A/B disambiguation for
# dot operands is authoritative (it is `opIdx` inside the DotOperandLayout itself);
# for loads and smem it is positional and marked as such.
# The AMD ops are in the `amdg` dialect, NOT `amdgpu`: a gfx942 dump prints
# `amdg.buffer_load`. Getting this wrong does not error, it just silently classifies
# every global load as "other" and the emitted role names become useless.
_DOT_OPS = ("tt.dot", "tt.dot_scaled", "ttng.warp_group_dot", "amdg.mfma", "amdg.wmma")
_GLOBAL_LOAD_OPS = ("tt.load", "tt.descriptor_load", "amdg.buffer_load")
_GLOBAL_STORE_OPS = ("tt.store", "tt.descriptor_store", "amdg.buffer_store")
_SMEM_OPS = ("ttg.local_alloc", "ttg.local_load", "ttg.local_store", "ttg.memdesc_index",
             "ttg.memdesc_subview", "ttg.memdesc_trans", "ttg.async_copy_global_to_local",
             "amdg.buffer_load_to_local")
_CVT_OPS = ("ttg.convert_layout",)
# Index / address math. Their layouts are the SliceLayouts an explicit Gluon kernel
# has to pass to `gl.arange`, and forgetting to is the single most expensive
# transcription mistake there is -- the operands then default to a scalar blocked
# layout and every load is uncoalesced. Worth naming rather than lumping in "other".
_INDEX_OPS = ("tt.make_range", "tt.expand_dims", "tt.splat", "tt.broadcast")
# Pure relayout / reshape ops. Their operands must be COLLECTED, not just their results:
# the backward A/B walk propagates a letter from a labelled result onto its source, and a
# `tt.trans` whose operand was never recorded is a hole the chain cannot cross. That hole
# is why a B side reaching its dot operand through reshape-permute came out as generic
# TO_SMEM / SMEM / FROM_SMEM.
_RELAY_OPS = ("tt.trans", "tt.reshape", "tt.join", "tt.split", "tt.cat",
              "ttg.memdesc_trans", "ttg.memdesc_reinterpret")


def _backend_key(arch: str) -> tuple:
    """(backend_key, driver_name, target_value, default_warp_size) from an arch string.

    Kept free of any triton import so the offline half of --selftest runs on a box that
    has no Triton at all -- which is the box this pack's CI validates on.
    """
    a = arch.strip().lower()
    if a.startswith("gfx"):
        return "amd", "hip", a, 64
    cap = a[2:] if a.startswith("sm") else a
    if not cap.isdigit():
        raise SystemExit(f"[ttgir_bridge] unrecognised --arch {arch!r}; "
                         f"expected gfx<NNN> or sm<NN>")
    return "nvidia", "cuda", int(cap), 32


def _resolve_target(arch: str, warp_size: int | None = None):
    """(backend_key, GPUTarget) for an arch string like 'gfx942' or 'sm90'/'90'."""
    from triton.backends.compiler import GPUTarget
    key, driver, value, default_ws = _backend_key(arch)
    return key, GPUTarget(driver, value, warp_size or default_ws)


_CAPS: dict | None = None

# Each capability is reported the instant it is confirmed, and the FATAL candidate goes
# LAST. That ordering is the whole design of this probe: the child dies on the shared
# normalisation attempt, so anything not already flushed is lost. Putting all the checks
# in one try/except (or running the risky one first) reports every capability as absent
# on the one build where it matters, which is how a 3.6 box ends up comparing its
# DISTRIBUTED layouts as text too -- weaker than it needs to be, for no reason.
_CAP_PROBE = """\
import sys
from triton._C.libtriton import ir, gluon_ir
import triton.experimental.gluon.language as gl


def say(c):
    sys.stdout.write("CAP:" + c + "\\n")
    sys.stdout.flush()


if hasattr(gluon_ir, "get_layout_view"):
    say("layout_view")
ctx = ir.context()
ir.load_dialects(ctx)
b = gluon_ir.GluonOpBuilder(ctx)
try:
    lay = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    b.to_linear_layout(lay._to_ir(b), [64, 64])
    say("distributed_norm")
except Exception:
    pass
# LAST: aborts the interpreter on Triton 3.6.
try:
    lay = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
    b.to_linear_layout(lay._to_ir(b), [64, 64])
    say("shared_norm")
except Exception:
    pass
"""


def capabilities() -> dict:
    """What this Triton build can do, probed in a CHILD process.

    Probed rather than version-gated, and out-of-process rather than in a try/except,
    because the capability that actually varies fails by ABORTING. On Triton 3.6
    `to_linear_layout` wraps its result in `LinearEncodingAttr` unconditionally, and
    that attribute requires input dim 0 to be `register`; hand it a SHARED layout
    (input dim `offset`) and it trips an MLIR assertion and kills the interpreter.
    3.7 added the `SharedEncodingTrait` branch that wraps in
    `SharedLinearEncodingAttr` instead. There is no exception to catch, so the probe
    runs somewhere it is allowed to die.

    Set `TTGIR_BRIDGE_CAPS=shared_norm,distributed_norm,layout_view` to skip the
    subprocess (or to force a set) if the ~0.4s is unwelcome in a loop.
    """
    global _CAPS
    if _CAPS is not None:
        return _CAPS
    env = os.environ.get("TTGIR_BRIDGE_CAPS")
    if env is not None:
        names = {n.strip() for n in env.split(",") if n.strip()}
        _CAPS = {"shared_norm": "shared_norm" in names,
                 "distributed_norm": "distributed_norm" in names,
                 "layout_view": "layout_view" in names, "probed": "env"}
        return _CAPS
    import subprocess
    names: set = set()
    how = "subprocess"
    try:
        # A non-zero / signal exit is EXPECTED on the builds this probe exists for; what
        # matters is the lines that made it out before the child died, so returncode is
        # deliberately not checked.
        r = subprocess.run([sys.executable, "-c", _CAP_PROBE], capture_output=True,
                           text=True, timeout=180, check=False)
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("CAP:"):
                names.add(ln[4:].strip())
    except Exception:  # noqa: BLE001
        how = "probe-failed"
    _CAPS = {"shared_norm": "shared_norm" in names,
             "distributed_norm": "distributed_norm" in names,
             "layout_view": "layout_view" in names, "probed": how}
    return _CAPS


def make_context(arch: str, warp_size: int | None = None):
    """A loaded MLIRContext plus a GluonOpBuilder on it.

    The backend dialects are NOT optional: an AMD `.ttgir` contains `amdgpu.*` ops,
    and without `HIPBackend.load_dialects` the parse fails with
    "Dialect `amdgpu' not found for custom op" -- which reads like a corrupt dump.
    """
    from triton._C.libtriton import gluon_ir, ir
    from triton.backends import backends
    key, tgt = _resolve_target(arch, warp_size)
    if key not in backends:
        raise SystemExit(f"[ttgir_bridge] backend {key!r} is not installed "
                         f"(have: {sorted(backends)})")
    ctx = ir.context()
    ir.load_dialects(ctx)
    backends[key].compiler(tgt).load_dialects(ctx)
    return ctx, gluon_ir.GluonOpBuilder(ctx)


# --------------------------------------------------------------------------- #
# 1. records
# --------------------------------------------------------------------------- #

_TENSOR_RE = re.compile(r"^tensor<(?P<dims>[0-9x]*)(?P<elem>[a-zA-Z_][\w.]*)\s*,\s*(?P<enc>.*)>$")
_MEMDESC_RE = re.compile(r"^!ttg\.memdesc<(?P<dims>[0-9x]*)(?P<elem>[a-zA-Z_][\w.]*)\s*,\s*(?P<rest>.*)>$")
_KIND_RE = re.compile(r"#(?:ttg|ttng|gluon|amdgpu)\.(?P<kind>\w+)")
_NUM_WARPS_RE = re.compile(r'"ttg\.num-warps"\s*=\s*(\d+)\s*:')
_THREADS_PER_WARP_RE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*(\d+)\s*:')


def _split_type(type_str: str) -> dict | None:
    """Split a printed tensor/memdesc type into dims, element type and encoding text.

    Returns None for a type that carries no layout (scalars, pointers, i1 masks in
    default encoding, ...). Deliberately string-based: this is the ONE place text is
    read, and it only splits at the top level -- it never interprets a layout body.
    """
    s = type_str.strip()
    m = _TENSOR_RE.match(s)
    if m:
        space = "reg"
    else:
        m = _MEMDESC_RE.match(s)
        if not m:
            return None
        space = "smem"
    dims = [int(d) for d in m.group("dims").split("x") if d]
    alloc = None
    if space == "reg":
        enc = m.group("enc").strip()
    else:
        # memdesc prints as <shape x elem, #layout, #space, mutable[, allocShape]>.
        # The layout is the first comma-separated field; splitting at the FIRST
        # top-level comma is wrong because a layout body contains commas, so scan.
        rest = m.group("rest")
        enc = _first_top_level_field(rest)
        if "shared_memory" not in rest and "tensor_memory" in rest:
            space = "tmem"
        # A SUBVIEW carries a trailing allocShape (`..., mutable, 256x128`), and the
        # shared layout's LinearLayout is sized by the ALLOCATION, not by the view. Pass
        # the view shape and `toLinearLayout` trips `assert(size == llSize)` -- a process
        # abort, so it has to be got right rather than caught.
        tail = _top_level_fields(rest)[-1] if rest else ""
        if re.fullmatch(r"\d+(x\d+)*", tail):
            alloc = [int(d) for d in tail.split("x")]
    if not enc.startswith("#"):
        return None
    km = _KIND_RE.match(enc)
    return {"space": space, "shape": dims, "dtype": m.group("elem"),
            "enc": enc, "kind": km.group("kind") if km else "?",
            "alloc_shape": alloc}


def _top_level_fields(s: str) -> list:
    """Comma-separated fields of ``s``, respecting <>, {}, [] and () nesting.

    Needed because a layout body is full of commas: naive splitting turns one field into
    six and the trailing allocShape can no longer be told from a basis vector.
    """
    out, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "<{[(":
            depth += 1
        elif ch in ">}])":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


def _first_top_level_field(s: str) -> str:
    """First comma-separated field of ``s``, respecting nesting."""
    return _top_level_fields(s)[0]


@dataclass
class Site:
    """One (op instance, position) that carries a layout."""
    op: str
    kind: str                 # "result" | "operand"
    index: int
    shape: list
    dtype: str
    space: str                # reg | smem | tmem
    enc_kind: str             # blocked | amd_mfma | swizzled_shared | dot_op | ...
    enc_text: str             # the encoding as the compiler printed it
    group: int = -1           # op-instance id; positions of ONE op share it
    alloc_shape: list | None = None   # memdesc allocShape when this is a SUBVIEW
    layout: Any = None        # the gl.* object from upstream layoutToGluon
    error: str | None = None
    role: str = ""
    roundtrip: str = "not-checked"   # EXACT | DIFFERS | n/a
    src: str = ""             # source variable name + file:line, from the MLIR location

    def as_json(self) -> dict:
        return {"op": self.op, "position": f"{self.kind}[{self.index}]",
                "shape": self.shape, "dtype": self.dtype, "space": self.space,
                "enc_kind": self.enc_kind, "role_hint": self.role, "src": self.src,
                "layout": repr(self.layout) if self.layout is not None else None,
                "roundtrip": self.roundtrip, "error": self.error,
                "enc_text": self.enc_text}


_FUNC_RE = re.compile(r"^\s*tt\.func\s+(public|private)?\s*@([\w$.]+)", re.MULTILINE)
_NS_ATTR_RE = re.compile(r"tt\.num_stages\s*=\s*(\d+)")
_ITER_ARGS_RE = re.compile(r"iter_args\(([^)]*)\)")


def _pipeline_facts(text: str) -> dict:
    """Is plain's loop software-pipelined, read from the dump rather than from the source.

    This is the single most useful thing the tool can say before anyone authors, and it
    used to say nothing. The residual of a faithful transcription is dominated by the lost
    software pipeline -- Gluon has no auto pipeliner -- so whether plain was pipelined
    decides what ratio to EXPECT, and therefore whether a measured shortfall is a debt or
    a defect.

    Read from the dump, never from the source: a kernel whose file contains
    `num_stages=1` can perfectly well dispatch a different branch at `num_stages=2`. That
    exact mistake set a wrong expectation for a control kernel in this trial -- the
    in-file 1 belonged to a branch that never runs.

    Signals, both literal in the TTGIR: the `tt.num_stages` attribute, and the count of
    `scf.for` loop-carried `iter_args` (the pipeliner adds carries for its in-flight
    buffers, so 0 vs several is a second, independent read).
    """
    ns = [int(m.group(1)) for m in _NS_ATTR_RE.finditer(text)]
    carries = [len([x for x in m.group(1).split(",") if x.strip()])
               for m in _ITER_ARGS_RE.finditer(text)]
    return {"num_stages": ns, "max_num_stages": max(ns) if ns else None,
            "loops": text.count("scf.for"), "iter_args": carries,
            "max_iter_args": max(carries) if carries else 0}


@dataclass
class Recovery:
    path: str
    arch: str
    num_warps: int | None
    threads_per_warp: int | None
    sites: list = field(default_factory=list)
    warp_check: str = "not-checked"
    funcs: list = field(default_factory=list)   # tt.func names in this module
    funcs_public: list = field(default_factory=list)   # ... of which these are `public`
    buffer_facts: dict = field(default_factory=dict)   # compiled form of amdg.buffer_* ops
    pipeline: dict = field(default_factory=dict)   # was plain's loop pipelined

    @property
    def ok_sites(self) -> list:
        return [s for s in self.sites if s.layout is not None]

    @property
    def failed_sites(self) -> list:
        return [s for s in self.sites if s.layout is None]


# --------------------------------------------------------------------------- #
# 2. recover
# --------------------------------------------------------------------------- #


def _layout_from_value(builder, value, space: str):
    """Upstream layoutToGluon, via the only two entry points Python has.

    MUST dispatch on the space first. `get_gluon_layout_from_tensor` does
    `dyn_cast<RankedTensorType>(...)` and then calls `.getEncoding()` on the result
    WITHOUT checking the cast (gluon_ir.cc, `get_gluon_layout_from_tensor`), so
    handing it a memdesc SEGFAULTS the interpreter rather than raising. Same the
    other way round. A wrong guess here is not a caught exception, it is a dead
    process with no traceback, so the type is decided before the call and never by
    try/except.
    """
    if space == "reg":
        return builder.get_gluon_layout_from_tensor(value)
    return builder.get_gluon_layout_from_memdesc(value)


# Role priority: when one layout serves several use sites, it is named after the most
# informative one. Naming it after whichever site the walk reached first is what makes
# an emitted file full of ARITH_CONSTANT / ARITH_CONSTANT_2 -- technically true and
# useless to whoever has to wire it in.
_ROLE_RANK = {
    "A_DOT_OPERAND": 100, "B_DOT_OPERAND": 100,
    "MMA": 95,
    "A_SMEM": 90, "B_SMEM": 90, "SMEM": 85,
    "A_LOAD": 82, "B_LOAD": 82,
    "A_FROM_SMEM": 80, "B_FROM_SMEM": 80,
    "TO_SMEM": 78, "FROM_SMEM": 78, "GLOBAL_LOAD": 75,
    "A_PRE_DOT": 74, "B_PRE_DOT": 74,
    "EPILOGUE_STORE": 70,
    "A_INDEX": 50, "B_INDEX": 50, "INDEX": 45,
    "CVT_DST": 40, "CVT_SRC": 35,
}

_MMA_KINDS = ("amd_mfma", "amd_wmma", "nvidia_mma")

_BUF_LOAD = re.compile(r"amdg\.buffer_load\s+([^:{\n]*)([^\n]*)")


def _buffer_facts(text: str) -> dict:
    """What a Gluon author must copy from plain's COMPILED buffer ops, not from its source.

    Two transcriptions lost 1.2% and ~2% to the same thing: `tl.load(..., other=0.0)` in
    the source compiles to a buffer_load with a mask and NO `other` operand, because buffer
    OOB returns zero on CDNA -- but the faithful-looking `gl.amd.cdna3.buffer_load(...,
    other=0.0)` emits the third operand and pays a v_cndmask per register. Transcribing the
    SOURCE faithfully is what produces the regression; transcribing the DUMP does not. The
    dump says which, and nothing was reporting it.
    """
    n_load = n_other = n_contig = 0
    for m in _BUF_LOAD.finditer(text):
        n_load += 1
        # `%base[%offsets], %mask` is two top-level operands after the base; a third
        # (`, %cst`) is the `other` value.
        if len(re.findall(r"%[\w$.]+", m.group(1))) >= 4:
            n_other += 1
        if "contiguity" in m.group(2):
            n_contig += 1
    return {"loads": n_load, "with_other": n_other, "with_contiguity": n_contig}


_LOC_NAME = re.compile(r'"([^"]+)"\(')
_LOC_LINE = re.compile(r'"([^"]*?)([^"/]+\.py)":(\d+)')


def _src_of(value: Any) -> str:
    """`SOURCE_NAME @ file.py:LINE` from an MLIR location, or "" if there is none.

    This is the fix for the single most-reported defect of the role names: five separate
    transcriptions had to open the raw TTGIR to learn which of Q/K/V a layout named
    `GLOBAL_LOAD` actually belonged to, because the role table ranks by op kind and three
    global loads that all feed a local_alloc are indistinguishable by rank alone. The
    compiler already carries the answer -- `loc("Q"("fwd_prefill.py":895:0))` names the
    source variable AND the line -- and it is reachable from the operand Value. A name the
    author wrote beats any role taxonomy this tool could invent.
    """
    try:
        raw = str(value.get_loc())
    except Exception:  # noqa: BLE001  -- no location info is normal, not an error
        return ""
    name = _LOC_NAME.search(raw)
    line = _LOC_LINE.search(raw)
    out = name.group(1) if name else ""
    if line:
        out += f" @ {line.group(2)}:{line.group(3)}"
    return out.strip()

# Ops with no `gluon.language` equivalent. Hand-maintained and ADVISORY: there is no
# upstream API to query builtin coverage the way layoutToGluon covers layouts.
#
# Used in two places, and the second is why it lives at module scope. `recover` prints it
# so a 100%-recovered kernel does not read as transcribable when it is not. `verify` uses
# it as reconciliation rule R3: a layout that is MISSING from the anchor because the
# LANGUAGE cannot express the op that produced it is not a transcription error, and
# calling it FAIL sends the author hunting for a mistake that is not theirs. Two trial
# kernels were graded FAIL on bit-exact, measured anchors for exactly this reason.
NO_GLUON_OP = {
    "amdg.in_thread_transpose": "no Gluon builtin; the transposed staging it produces "
                                "has to be re-expressed (often as an LDS round trip)",
    "amdg.rotating_shared": "no Gluon rotating-shared constructor",
}


def _base_role(op: str, kind: str, index: int, enc_kind: str, layout: Any,
               space: str = "reg") -> str:
    """Role of one use site, from the op, the position, and the memory SPACE.

    The space is not optional. A `ttg.local_store` has two operands -- a register
    tensor and a memdesc -- and classifying both as "shared memory" because the OP is
    a shared-memory op mislabels the staged value's register layout, which is then
    never promoted to A_LOAD/B_LOAD by the attribution pass.
    """
    if enc_kind == "dot_op" and layout is not None:
        idx = getattr(layout, "operand_index", None)
        if idx in (0, 1):
            return f"{'AB'[idx]}_DOT_OPERAND"
    # An MMA layout is named for what it IS, not for where it happens to be seen. It
    # shows up on the accumulator constant, on every arith op that touches the
    # accumulator, and on the epilogue store's operand; calling it EPILOGUE_STORE
    # because that site outranks the others describes the last thing it does.
    if enc_kind in _MMA_KINDS:
        return "MMA"
    if op in _DOT_OPS and kind == "operand":
        return {0: "A_DOT_OPERAND", 1: "B_DOT_OPERAND"}.get(index, "DOT_OPERAND")
    if op in _GLOBAL_LOAD_OPS:
        return "GLOBAL_LOAD"
    if op in _GLOBAL_STORE_OPS:
        return "EPILOGUE_STORE"
    if op in _SMEM_OPS:
        if space == "smem":
            return "SMEM"
        # The register side of a shared-memory op: the value being staged in, or the
        # value loaded back out. Kept distinct from SMEM so attribution can promote it.
        return "TO_SMEM" if op in ("ttg.local_store", "ttg.async_copy_global_to_local",
                                   "amdg.buffer_load_to_local") else "FROM_SMEM"
    if op in _CVT_OPS:
        return "CVT_SRC" if kind == "operand" else "CVT_DST"
    if op in _INDEX_OPS:
        return "INDEX"
    return op.replace(".", "_").upper()


def _attribute_operands(sites: list) -> None:
    """Tie the shared-memory and global-load layouts to operand A or B, authoritatively.

    `opIdx` on a DotOperandLayout is the only place the A/B question is answered by the
    IR rather than guessed, so the attribution starts there and walks BACKWARDS along
    the ops that produced it:

        amdg.buffer_load -> ttg.local_store -> memdesc -> ttg.local_load -> dot_op(opIdx)
              A_LOAD          (same value)     A_SMEM         (same memdesc)

      * `ttg.local_load`: operand[0] is the staging buffer, the result carries the
        dot_op layout with its opIdx  =>  that buffer is A's or B's.
      * `ttg.local_store`: operand[1] is the buffer (now labelled), operand[0] is the
        register value that fed it  =>  that value's layout is A's or B's global load.
      * `ttg.convert_layout` whose result is a dot_op: its source is the pre-dot
        register layout for that operand (the no-LDS path).

    Sites are matched by (encoding text, shape) rather than by SSA identity because
    pybind's `Value` exposes no `__eq__`/`__hash__`. Two different tensors with the
    same encoding AND the same shape are the same layout by definition, so the label
    lands on the right constant either way; what can happen is that A and B share one
    layout, and then they legitimately share one constant (reported as serving both).
    """
    by_op: dict = {}
    for s in sites:
        by_op.setdefault(s.group, []).append(s)

    smem_side: dict = {}   # enc_text -> 'A' | 'B'
    reg_side: dict = {}    # (enc_text, shape) -> 'A' | 'B'

    for group in by_op.values():
        op = group[0].op
        if op == "ttg.local_load":
            res = next((s for s in group if s.kind == "result"), None)
            mem = next((s for s in group if s.kind == "operand" and s.space == "smem"), None)
            letter = _letter(res)
            if letter is None and res is not None:
                letter = reg_side.get((res.enc_text, tuple(res.shape)))
            if letter and mem is not None:
                smem_side.setdefault(mem.enc_text, letter)
        elif op in _CVT_OPS:
            res = next((s for s in group if s.kind == "result"), None)
            src = next((s for s in group if s.kind == "operand"), None)
            letter = _letter(res)
            if letter and src is not None:
                reg_side.setdefault((src.enc_text, tuple(src.shape)), letter)

    # Follow the pure-relayout ops too. The backward walk used to stop at `tt.trans` /
    # reshape / permute, so a B side that reaches its dot operand through a
    # reshape-permute chain (rather than straight out of LDS) never got labelled and came
    # out as generic TO_SMEM / SMEM / FROM_SMEM. The runbook's checklist then says nothing
    # about what to use for B, and that was the one place the tool sent an agent back to
    # the raw TTGIR. Iterated to a fixed point because a chain can be several ops long.
    RELAY = _RELAY_OPS + _INDEX_OPS + ("ttg.memdesc_index",)
    for _ in range(8):
        grew = False
        for group in by_op.values():
            if group[0].op != "ttg.local_load":
                continue
            res = next((x for x in group if x.kind == "result"), None)
            mem = next((x for x in group if x.kind == "operand" and x.space != "reg"), None)
            lt = _letter(res) or (reg_side.get((res.enc_text, tuple(res.shape)))
                                  if res is not None else None)
            if lt and mem is not None and mem.enc_text not in smem_side:
                smem_side[mem.enc_text] = lt
                grew = True
        for group in by_op.values():
            if group[0].op not in RELAY + _CVT_OPS:
                continue
            res = [s for s in group if s.kind == "result"]
            ops = [s for s in group if s.kind == "operand"]
            for dst in res:
                letter = (_letter(dst)
                          or reg_side.get((dst.enc_text, tuple(dst.shape)))
                          or (smem_side.get(dst.enc_text) if dst.space != "reg" else None))
                if not letter:
                    continue
                for src in ops:
                    key = ((src.enc_text, tuple(src.shape)) if src.space == "reg"
                           else src.enc_text)
                    tgt = reg_side if src.space == "reg" else smem_side
                    if key not in tgt:
                        tgt[key] = letter
                        grew = True
        if not grew:
            break

    for group in by_op.values():
        if group[0].op != "ttg.local_store":
            continue
        mem = next((s for s in group if s.kind == "operand" and s.space == "smem"), None)
        val = next((s for s in group if s.kind == "operand" and s.space == "reg"), None)
        if mem is None or val is None:
            continue
        letter = smem_side.get(mem.enc_text)
        if letter:
            reg_side.setdefault((val.enc_text, tuple(val.shape)), letter)

    for s in sites:
        if s.space != "reg":
            letter = smem_side.get(s.enc_text)
            if letter and s.role in ("SMEM", "TTG_LOCAL_ALLOC", "TTG_MEMDESC_INDEX",
                                     "TTG_MEMDESC_TRANS", "TTG_MEMDESC_REINTERPRET"):
                s.role = f"{letter}_SMEM"
            continue
        letter = reg_side.get((s.enc_text, tuple(s.shape)))
        if not letter:
            continue
        if s.role in ("GLOBAL_LOAD", "TO_SMEM"):
            s.role = f"{letter}_LOAD"
        elif s.role == "FROM_SMEM":
            s.role = f"{letter}_FROM_SMEM"
        elif s.role in ("INDEX", "CVT_SRC"):
            s.role = f"{letter}_INDEX" if s.role == "INDEX" else f"{letter}_PRE_DOT"
        elif s.role == s.op.replace(".", "_").upper():
            s.role = f"{letter}_PRE_DOT"


def _letter(site) -> str | None:
    if site is None or site.layout is None or site.enc_kind != "dot_op":
        return None
    idx = getattr(site.layout, "operand_index", None)
    return "AB"[idx] if idx in (0, 1) else None


def _roundtrip(builder, site: Site) -> str:
    """Rebuild the MLIR type from the recovered object and compare printed encodings.

    This is not circular. It tests that ``layoutToGluon`` followed by ``_to_ir`` is
    the identity -- i.e. that the Python object carries every field the attribute
    had. A converter that drops a field round-trips to a DIFFERENT string, and this
    is the only cheap way to see that from Python (Attribute itself has no
    ``__str__`` binding; Type does, so the attribute is printed through a type).
    """
    if site.layout is None:
        return "n/a"
    # Same rank guard as _shape_for_layout: a rank mismatch here reaches an LLVM
    # assert, which is a process abort rather than an exception.
    shape = _shape_for_layout(site.layout, _sizing_shape(site))
    if shape is None:
        return f"n/a (rank {_layout_rank(site.layout)} layout at shape {site.shape})"
    try:
        attr = site.layout._to_ir(builder)
        elem = _element_ty(builder, site.dtype)
        if elem is None:
            return "n/a (element type not constructible here)"
        if site.space == "reg":
            ty = builder.get_distributed_ty(elem, shape, attr)
        else:
            ty = builder.get_shared_mem_desc_ty(elem, shape, attr, shape)
        got = _split_type(str(ty))
        if got is None:
            return "DIFFERS (re-print not parseable)"
        return "EXACT" if _norm(got["enc"]) == _norm(site.enc_text) else "DIFFERS"
    except Exception as e:  # noqa: BLE001 - a round-trip failure is a reportable fact
        return f"DIFFERS ({type(e).__name__}: {e})"[:120]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_ELEM_GETTERS = {
    "f16": "get_half_ty", "bf16": "get_bf16_ty", "f32": "get_float_ty",
    "f64": "get_double_ty", "i32": "get_int32_ty", "i64": "get_int64_ty",
    "i16": "get_int16_ty", "i8": "get_int8_ty", "i1": "get_int1_ty",
    "f8E4M3FNUZ": "get_fp8e4b8_ty", "f8E5M2FNUZ": "get_fp8e5b16_ty",
    "f8E4M3FN": "get_fp8e4nv_ty", "f8E5M2": "get_fp8e5_ty",
}


def _element_ty(builder, dtype: str):
    """Best-effort element type. Only used for the round-trip re-print.

    The element type does not affect a distributed layout, so when a dtype has no
    getter on this build we substitute f16 and say so, rather than skipping the
    check. For a SHARED layout the element WIDTH can matter (NVMMAShared carries
    elementBitWidth), so a substitution is reported instead of hidden.
    """
    name = _ELEM_GETTERS.get(dtype)
    if name and hasattr(builder, name):
        return getattr(builder, name)()
    return builder.get_half_ty() if hasattr(builder, "get_half_ty") else None


def recover(path: str, arch: str, warp_size: int | None = None,
            roundtrip: bool = True) -> Recovery:
    from triton._C.libtriton import ir
    ctx, builder = make_context(arch, warp_size)
    if not os.path.exists(path):
        raise SystemExit(f"[ttgir_bridge] no such file: {path}")
    try:
        mod = ir.parse_mlir_module(path, ctx)
    except RuntimeError as e:
        # MLIR prints the real diagnostic to stderr just above this; the Python
        # traceback adds nothing and hides it. The two causes seen in practice are
        # both actionable, so name them rather than re-raising.
        raise SystemExit(
            f"[ttgir_bridge] MLIR refused to parse {path}\n"
            f"  {e}\n"
            f"  The diagnostic printed above this line is the real reason. Three seen in\n"
            f"  practice, all actionable:\n"
            f"    * 'undefined symbol alias id' -- the dump was hand-edited and its `#loc`\n"
            f"      (or layout) aliases were stripped. Use the ORIGINAL dump from\n"
            f"      dump_ir.sh, not a cleaned copy; this tool needs a parseable module and\n"
            f"      gains nothing from the cleaning.\n"
            f"    * \"custom op 'ttg.barrier' is unknown\" (or any other unknown op) -- the\n"
            f"      dump was produced by a NEWER Triton than the one running this tool.\n"
            f"      `ttg.barrier` in particular does not exist before 3.7, so a clean 3.6\n"
            f"      cannot read a 3.7+ dump. Run the tool under a Triton at least as new as\n"
            f"      the compiler that wrote the dump.\n"
            f"    * \"Dialect `amdg' not found\" -- --arch names the wrong backend, so the\n"
            f"      AMD ops have no dialect. Pass the arch the dump was compiled for.") from None
    text = mod.str()

    nw = _NUM_WARPS_RE.search(text)
    tpw = _THREADS_PER_WARP_RE.search(text)
    rec = Recovery(path=path, arch=arch,
                   num_warps=int(nw.group(1)) if nw else None,
                   threads_per_warp=int(tpw.group(1)) if tpw else None,
                   funcs=[m.group(2) for m in _FUNC_RE.finditer(text)],
                   funcs_public=[m.group(2) for m in _FUNC_RE.finditer(text)
                                 if m.group(1) == "public"],
                   buffer_facts=_buffer_facts(text),
                   pipeline=_pipeline_facts(text))

    ops: list = []
    mod.walk(ops.append)

    # Collected per OP INSTANCE, without deduplication. Deduplicating here -- which is
    # the obvious thing to do, since many ops share a layout -- collapses A's
    # `ttg.local_load` and B's into one record and destroys the operand attribution
    # below, which needs each op's positions to stay together. Dedup happens at
    # report/emit time instead, where it is a presentation choice.
    layout_cache: dict = {}
    for gid, op in enumerate(ops):
        try:
            name = op.get_name()
        except Exception:  # noqa: BLE001
            name = "<unknown>"
        positions = [("result", i, op.get_result(i)) for i in range(op.get_num_results())]
        # Operands matter for the ops whose LAYOUT CHOICE lives on the input side: a
        # dot's operands, a store's value, a local_load's memdesc. Collecting every
        # operand of every op would re-report each result once per consumer.
        if name in (_DOT_OPS + _GLOBAL_STORE_OPS + _SMEM_OPS + _CVT_OPS
                    + _RELAY_OPS + _INDEX_OPS):
            positions += [("operand", i, op.get_operand(i))
                          for i in range(op.get_num_operands())]
        for kind, idx, val in positions:
            info = _split_type(str(val.get_type()))
            if info is None:
                continue
            site = Site(op=name, kind=kind, index=idx, shape=info["shape"],
                        dtype=info["dtype"], space=info["space"],
                        enc_kind=info["kind"], enc_text=info["enc"], group=gid,
                        alloc_shape=info.get("alloc_shape"), src=_src_of(val))
            ck = (info["enc"], info["space"])
            if ck in layout_cache:
                site.layout, site.error, site.roundtrip = layout_cache[ck]
            else:
                try:
                    site.layout = _layout_from_value(builder, val, info["space"])
                except Exception as e:  # noqa: BLE001
                    # An UNRECOVERABLE layout is a first-class result, not a warning to
                    # bury: it means this kernel is not fully transcribable on this
                    # build, and the transcription must stop rather than substitute.
                    site.error = f"{type(e).__name__}: {e}"
                if roundtrip:
                    site.roundtrip = _roundtrip(builder, site)
                layout_cache[ck] = (site.layout, site.error, site.roundtrip)
            site.role = _base_role(name, kind, idx, info["kind"], site.layout, info["space"])
            rec.sites.append(site)

    _attribute_operands(rec.sites)
    rec.warp_check = _check_num_warps(rec)
    return rec


def _group_by_layout(rec: Recovery) -> dict:
    """Distinct layout -> its use sites, with the best-ranked role first.

    Keyed on the layout's repr rather than on the encoding text so that two spellings
    of one layout (which upstream normalises) collapse into one constant.
    """
    out: dict = {}
    for s in rec.ok_sites:
        out.setdefault(repr(s.layout), []).append(s)
    for sites in out.values():
        sites.sort(key=lambda s: -_ROLE_RANK.get(s.role, 0))
    return out


def _check_num_warps(rec: Recovery) -> str:
    """Cross-check the module's num_warps against every layout's warps_per_cta.

    The recovered layouts HARD-CODE the warp distribution. If a later round changes
    `num_warps` at the launch site, the transcribed kernel does not get slower, it
    gets WRONG (or fails to parse) -- the config knob and the layout constants are
    two sources of truth for one fact. Checking it here makes that mechanical
    instead of a warning somebody has to remember.
    """
    if rec.num_warps is None:
        return "SKIP (module carries no ttg.num-warps)"
    bad = []
    for s in rec.ok_sites:
        wpc = _warps_per_cta(s.layout)
        if wpc is None:
            continue
        prod = 1
        for v in wpc:
            prod *= v
        if prod != rec.num_warps:
            bad.append(f"{s.role}/{s.op} warps_per_cta={wpc} -> {prod} != {rec.num_warps}")
    if bad:
        return "FAIL: " + "; ".join(bad[:4])
    return f"PASS (num_warps={rec.num_warps} agrees with every warps_per_cta)"


def _warps_per_cta(layout: Any) -> list | None:
    """warps_per_cta of a distributed layout, following Slice/DotOperand to a parent.

    A SliceLayout's parent keeps the full rank, so the product is still the warp
    count; that is why the parent is followed rather than the child inspected.
    """
    seen = 0
    while layout is not None and seen < 8:
        wpc = getattr(layout, "warps_per_cta", None)
        if wpc:
            return list(wpc)
        nxt = getattr(layout, "parent", None)
        if nxt is None:
            return None
        layout, seen = nxt, seen + 1
    return None


# --------------------------------------------------------------------------- #
# 3. emit
# --------------------------------------------------------------------------- #

_HEADER = '''\
# ===========================================================================
# Recovered Gluon layouts -- {src}
# recovered-from: {src}
#
# Produced by scripts/ttgir_bridge.py recover (arch={arch}, triton={tver}).
# The Triton version is stamped because it changes what this tool can PARSE (3.6 cannot
# read a dump containing ttg.barrier) and how `verify` compares shared layouts (3.6 falls
# back to canonical text). The recovered file is the artefact people keep; without the
# stamp there is no way to tell later which build produced it.
# Every constant below came from the COMPILER's own layoutToGluon(), not from a
# text parse, and each one carries a round-trip proof that re-printing it yields
# the identical MLIR attribute.
#
# READ THIS BEFORE USING THE FILE:
#   * Declaring a layout is not applying it. These constants do nothing until they
#     are passed as `layout=` on the tensor that plays that role -- `gl.arange(...,
#     layout=gl.SliceLayout(d, X_LOAD))` for index tensors, `gl.allocate_shared_
#     memory(..., X_SMEM)` for staging, `gl.convert_layout(x, A_DOT_OPERAND)` before
#     the dot. An anchor that declares them and leaves the body on AutoLayout
#     compiles, is bit-exact, and is several times slower.
#   * NUM_WARPS below is pinned BY these layouts. Changing it at the launch site
#     without re-recovering is a correctness bug, not a slow config.
#   * Role names are HINTS derived from the use site. The authoritative facts are in
#     the provenance comment on each line (op, position, shape).
# ===========================================================================
# gl.amd is reachable from gl, so the AMD layout classes need no second import.
import triton.experimental.gluon.language as gl

# Assignment form, not `NAME: gl.constexpr = N`. Triton rejects the annotated form for a
# kernel-visible global scalar, and the runbook says to use NUM_WARPS at the launch site
# unchanged -- so the annotated spelling made the emitted file unusable as written.
# Layout constants below keep the annotation; only bare ints trip it.
NUM_WARPS = {num_warps}
THREADS_PER_WARP = {threads_per_warp}
'''


def _py_ident(role: str, used: set) -> str:
    base = re.sub(r"\W+", "_", role).strip("_").upper() or "LAYOUT"
    name, n = base, 1
    while name in used:
        n += 1
        name = f"{base}_{n}"
    used.add(name)
    return name


def assign_names(rec: Recovery) -> list:
    """[(symbol, layout_repr, sites)] — THE single naming rule.

    Both `recover`'s emitted file and `view` resolve through this. They used to resolve
    independently: emission named one constant per distinct LAYOUT (best-ranked role
    wins, ties get a `_2` suffix), while `view` looked up the first SITE whose role
    matched. On a kernel where one layout serves both a load and a store, those two rules
    disagree — `EPILOGUE_STORE` named a rank-2 scale store in the emitted file while
    `view --role EPILOGUE_STORE` printed a rank-3 data store. Anyone who inspected with
    one and wired the other got a rank-mismatched layout, which is the exact failure the
    role names exist to prevent. One rule, one answer.
    """
    used: set = set()
    out = []
    for key, sites in _group_by_layout(rec).items():
        out.append((_py_ident(sites[0].role, used), key, sites))
    return out


def _gl_expr(layout: Any) -> str:
    """`repr()` of a frozen dataclass, re-spelled as a `gl.*` constructor call.

    The layout classes are frozen dataclasses whose repr is already
    `ClassName(field=value, ...)`, so this only has to qualify the class names --
    which is why there is no per-kind emission table here, and therefore nothing
    to fall behind upstream.
    """
    txt = repr(layout)
    amd = ("AMDMFMALayout", "AMDWMMALayout")
    def qual(m):
        cls = m.group(1)
        return ("gl.amd." if cls in amd else "gl.") + cls + "("
    return re.sub(r"\b([A-Z]\w*Layout)\(", qual, txt)


def emit_layouts(rec: Recovery) -> str:
    try:
        import triton as _t
        _tver = _t.__version__
    except Exception:  # noqa: BLE001
        _tver = "unknown"
    out = [_HEADER.format(src=rec.path, arch=rec.arch, tver=_tver,
                          num_warps=rec.num_warps, threads_per_warp=rec.threads_per_warp)]
    out.append(f"# num_warps cross-check: {rec.warp_check}\n")

    # One constant per DISTINCT layout, named after its best-ranked role, with every
    # use site listed. One constant per SITE would duplicate the same layout under
    # several names, and the transcriber could then no longer see that two roles share
    # a layout -- which is exactly the fact that decides whether a convert_layout is
    # needed between them.
    names: dict = {}
    for name, key, sites in assign_names(rec):
        names[key] = name
        # Sorted by rank AND THEN BY NAME. Rank alone leaves equal-ranked roles in set
        # iteration order, which made the emitted file differ between two runs on the
        # same input -- and the runbook uses byte-identical output across Triton versions
        # as its evidence that a constant is the compiler's rather than a fork's. A
        # comment that reorders on its own defeats that check on a single machine.
        roles = sorted({s.role for s in sites}, key=lambda r: (-_ROLE_RANK.get(r, 0), r))
        out.append(f"# {name}: {sites[0].enc_kind}  [round-trip {sites[0].roundtrip}]"
                   + (f"  also serves: {', '.join(roles[1:])}" if len(roles) > 1 else ""))
        for prov, n in _provenance(sites):
            out.append(f"#   {prov}" + (f"   x{n}" if n > 1 else ""))
        out.append(f"{name}: gl.constexpr = {_gl_expr(sites[0].layout)}")
        out.append("")

    if rec.failed_sites:
        out.append("# " + "-" * 74)
        out.append("# UNRECOVERABLE -- no gluon.language constructor on this build.")
        out.append("# The kernel is NOT fully transcribable as written; do not proceed with a")
        out.append("# substitute. Each row names the TTGIR kind so the gap can be looked up.")
        for s in rec.failed_sites:
            out.append(f"#   {s.enc_kind:<24} {s.op} {s.kind}[{s.index}] shape={s.shape}")
            out.append(f"#     {s.error}")
            out.append(f"#     {s.enc_text[:140]}")
        out.append("")

    # Suffixes are allocation-ordered, NOT tile-ordered: `MMA_3` can belong with `A_LOAD_2`.
    # Assuming `X_n` pairs with `Y_n` wires the wrong MFMA family to the wrong dot operand
    # AND STILL COMPILES. So the pairing is stated rather than left to be inferred.
    fam = {}
    for name, key, sites in assign_names(rec):
        lay = sites[0].layout
        root, seen_ = lay, 0
        while root is not None and seen_ < 8:
            if type(root).__name__ in ("AMDMFMALayout", "AMDWMMALayout",
                                       "NVMMADistributedLayout"):
                break
            root, seen_ = getattr(root, "parent", None), seen_ + 1
        else:
            root = None
        tile = "x".join(str(d) for d in sites[0].shape)
        fam.setdefault(repr(root) if root is not None else "(no MFMA parent)",
                       []).append(f"{name}[{tile}]")
    if len(fam) > 1 or any(len(v) > 1 for v in fam.values()):
        out.append("# " + "-" * 74)
        out.append("# PAIRING -- which constants belong together. The numeric suffixes are")
        out.append("# allocation-ordered, not tile-ordered, so do NOT assume X_n goes with Y_n.")
        for root, names_ in fam.items():
            out.append(f"#   MFMA family {root[:88]}")
            out.append(f"#     {', '.join(sorted(names_))}")
        out.append("")

    # Per-DOT grouping. The MFMA-family table above collapses to a single bucket whenever
    # the compiler reuses one #mma for every dot -- on a 14-dot backward kernel all 23
    # layouts landed under one heading, and the question the author actually had ("is this
    # A_SMEM the qk dot, the dv dot or the dq dot?") had no answer in it. The dot instances
    # and their operands' source names are already parsed; this is just printing them.
    dots = {}
    for s in rec.sites:
        if s.op in _DOT_OPS:
            dots.setdefault(s.group, {})[f"{s.kind}{s.index}"] = s
    if dots:
        out.append("# " + "-" * 74)
        out.append(f"# DOTS -- {len(dots)} tt.dot instance(s), each with its operands' source")
        out.append("# names. Use this, not the suffix numbering, to decide which constant goes")
        out.append("# on which tensor.")
        def _d(pos: dict, p: str) -> str:
            s = pos.get(p)
            if s is None:
                return "?"
            return (s.src.split(" @ ")[0] or "?") + f"{s.shape}"

        for i, (_gid, pos) in enumerate(sorted(dots.items()), 1):
            out.append(f"#   dot #{i}: A={_d(pos, 'operand0')}  B={_d(pos, 'operand1')}  "
                       f"-> {_d(pos, 'result0')}")
        out.append("")

    out.append("ROLES = {")
    for key, name in names.items():
        out.append(f"    {name!r}: {name},")
    out.append("}")
    return "\n".join(out) + "\n"


_INTERESTING_OPS = frozenset(_DOT_OPS + _GLOBAL_LOAD_OPS + _GLOBAL_STORE_OPS
                             + _SMEM_OPS + _CVT_OPS + _INDEX_OPS)


def _provenance(sites: list, limit: int = 8) -> list:
    """Distinct (op, position, shape) rows with a count, most informative first.

    Capped, because an MMA layout propagates through every `arith` op that touches the
    accumulator and listing all thirty of them buries the four lines that say where the
    layout is actually decided (the dot, the loads, the stores, the staging).
    """
    rows: dict = {}
    for s in sites:
        # The source name is part of the key, not a decoration: it is the whole reason two
        # otherwise identical `amdg.buffer_load result[0] shape=[128,128]` rows are worth
        # printing separately -- one is Q and one is K.
        k = (f"{s.op} {s.kind}[{s.index}]  shape={s.shape} {s.dtype} ({s.space})"
             + (f"  <- {s.src}" if s.src else ""))
        rows[k] = rows.get(k, 0) + 1
    ordered = sorted(rows.items(),
                     key=lambda kv: (kv[0].split(" ")[0] not in _INTERESTING_OPS, -kv[1]))
    out = ordered[:limit]
    hidden = sum(n for _, n in ordered[limit:])
    if hidden:
        out.append(((f"... and {hidden} further site(s) in {len(ordered) - limit} "
                     f"other op(s) (the layout propagates; see --json for the full list)"), 1))
    return out


def report(rec: Recovery, verbose: bool = False) -> str:
    lines = [f"TTGIR -> Gluon recovery: {rec.path}",
             (f"  arch={rec.arch}  num_warps={rec.num_warps}  "
              f"threads_per_warp={rec.threads_per_warp}"),
             f"  num_warps cross-check: {rec.warp_check}",
             (f"  sites carrying a layout: {len(rec.sites)}  "
              f"recovered: {len(rec.ok_sites)}  UNRECOVERABLE: {len(rec.failed_sites)}")]
    rt = {}
    for s in rec.ok_sites:
        rt[s.roundtrip.split(" ")[0]] = rt.get(s.roundtrip.split(" ")[0], 0) + 1
    lines.append("  round-trip: " + ", ".join(f"{k}={v}" for k, v in sorted(rt.items())))
    pf = rec.pipeline or {}
    mns, mia = pf.get("max_num_stages"), pf.get("max_iter_args", 0)
    if pf.get("loops"):
        lines.append("")
        lines.append(f"  PIPELINE (read from THIS dump, not from the source): "
                     f"tt.num_stages={pf.get('num_stages') or 'absent'}, "
                     f"{pf['loops']} scf.for, max iter_args={mia}")
        if (mns or 1) > 1 or mia >= 2:
            lines.append("  => plain IS software-pipelined. Gluon has no auto pipeliner, so a")
            lines.append("     FAITHFUL anchor sits below plain BY CONSTRUCTION -- that gap is a")
            lines.append("     debt you knowingly take on, not a defect. Measure `plain at")
            lines.append("     num_stages=1` as the control: if the anchor lands there, the whole")
            lines.append("     residual is the pipeline and no layout work will move it.")
        else:
            lines.append("  => plain is NOT pipelined (num_stages=1, no loop carries). There is no")
            lines.append("     pipeline to lose, so a faithful anchor should land at ~1.00 and")
            lines.append("     anything materially below that is a transcription DEFECT, not a debt.")
            lines.append("     Check the ASM load-width histogram before believing any clock.")
        lines.append("     Do not infer num_stages from the source: a file containing 1 can")
        lines.append("     dispatch a branch compiled at 2.")
    lines.append("")
    by_layout = _group_by_layout(rec)
    if len(by_layout) == 1:
        # The role ranking is dot-shaped. On a kernel with ONE layout it picks a name from
        # whichever site outranks the rest -- reported as `GLOBAL_LOAD` on a kernel where
        # the same layout also governs every store operand, which reads as if loads and
        # stores had been distinguished. Say the useful thing instead.
        lines.append("  1 distinct layout: SINGLE-LAYOUT KERNEL -- apply it to every tensor.")
        lines.append("  The role name below is picked from the highest-ranked site and does NOT")
        lines.append("  mean loads and stores differ here; read the provenance list, not the name.")
    else:
        lines.append(f"  {len(by_layout)} distinct layouts:")
    for key, sites in sorted(by_layout.items(), key=lambda kv: -_ROLE_RANK.get(kv[1][0].role, 0)):
        s0 = sites[0]
        lines.append(f"    {s0.role:<16} {s0.enc_kind:<17} {len(sites):>3} sites  "
                     f"{s0.roundtrip}")
        lines.append(f"      {key[:150]}")
        if verbose:
            # On a single-layout kernel the site list IS the whole report, so the usual cap
            # removes the only content there is -- it truncated 21 of 41 sites on a kernel
            # whose entire actionable output was "here is where this one layout goes".
            for prov, n in _provenance(sites, limit=10 ** 6 if len(by_layout) == 1 else 8):
                lines.append(f"        {prov}" + (f"   x{n}" if n > 1 else ""))
    for s in rec.failed_sites:
        lines.append(f"    [UNRECOVERABLE] {s.enc_kind}  {s.op} {s.kind}[{s.index}] "
                     f"shape={s.shape}")
        lines.append(f"      {s.error}")
    # LDS staging that exists only BELOW TTGIR. `local_alloc` with no `local_store` is the
    # backend materialising a `convert_layout` scratch buffer during lowering: there is no
    # shared layout in the IR for it, so no amount of recovery will surface one, and the
    # transcriber has to author the staging AND derive its layout by hand. One kernel in
    # the trial spent most of its round discovering this -- the tool is complete with
    # respect to TTGIR and structurally blind to anything that only exists under it, and
    # saying so costs two lines.
    # `ttg.local_alloc %value` -- the INITIALISED form -- writes LDS itself, so a module can
    # legitimately have allocs and zero stores while its shared layouts are right there in
    # the IR. The first version of this NOTE counted stores only and then concluded "there
    # is NO shared layout in the IR for it and none can be recovered", on a kernel where two
    # shared layouts had just been recovered round-trip EXACT. A correct count with a wrong
    # conclusion is worse than silence: it pushed the author onto hand-deriving GF(2) bases
    # that were already emitted. Only the operand-less form means backend-created staging.
    alloc_groups = {s.group for s in rec.sites if s.op == "ttg.local_alloc"}
    bare_alloc = sum(1 for g in alloc_groups
                     if not any(s.group == g and s.kind == "operand" for s in rec.sites))
    n_alloc = len(alloc_groups)
    n_store = sum(1 for s in rec.sites if s.op == "ttg.local_store")
    if bare_alloc and not n_store:
        lines.append("")
        lines.append(f"  NOTE: {bare_alloc} of {n_alloc} ttg.local_alloc have NO initialiser and")
        lines.append("  there are ZERO ttg.local_store. For those, plain never")
        lines.append("  writes LDS at the TTGIR level here -- the backend materialises the staging")
        lines.append("  while lowering a convert_layout. So there is NO shared layout in the IR for")
        lines.append("  it and none can be recovered: expect to author that staging yourself and to")
        lines.append("  DERIVE its layout (gl.SharedLinearLayout bases) by hand. Recovery being")
        lines.append("  complete here means complete w.r.t. TTGIR, not w.r.t. what plain does.")

    subv = [s for s in rec.sites if s.space != "reg" and s.alloc_shape
            and list(s.alloc_shape) != list(s.shape)]
    if subv:
        lines.append("")
        lines.append(f"  NOTE: {len(subv)} shared SUBVIEW site(s) (view shape != allocShape, e.g. "
                     f"{subv[0].shape} of {subv[0].alloc_shape}). Layouts here are sized by the")
        lines.append("  ALLOCATION; this tool evaluates them at allocShape for exactly that reason.")

    # A ladder rung is a PUBLIC func. Gating on the total count both fires falsely and
    # misses the real hazard: a `noinline=True` scalar helper makes a 2-func module whose
    # private callee carries no layouts at all (the NOTE then tells you to go dump a rung
    # that does not exist), while a genuine four-way ladder compiles to four separate
    # single-func modules and the NOTE never fires on any of them. The visibility keyword
    # is right there in the dump.
    priv = [f for f in rec.funcs if f not in set(rec.funcs_public)]
    if len(rec.funcs_public) > 1:
        lines.append("")
        lines.append(f"  NOTE: this module defines {len(rec.funcs_public)} PUBLIC tt.func:")
        for fn_ in rec.funcs_public[:6]:
            # Keep the TAIL of a mangled specialization name: that is where the tile
            # constants live, and the tile is the whole reason the rungs differ.
            shown = fn_ if len(fn_) <= 76 else fn_[:26] + " ... " + fn_[-45:]
            lines.append(f"    {shown}")
        if len(rec.funcs_public) > 6:
            lines.append(f"    ... and {len(rec.funcs_public) - 6} more")
        lines.append("  A champion can be a LADDER -- one launcher dispatching several")
        lines.append("  specializations at runtime, each with its own tile and its own recovered")
        lines.append("  layouts. This dump is ONE of them. Recovering it and wiring every rung")
        lines.append("  from these constants silently gives the wrong blocked layouts to the other")
        lines.append("  rungs, and the kernel still compiles and is still bit-exact. Dump and")
        lines.append("  recover each rung separately, to SEPARATE --out files.")

    if priv:
        lines.append("")
        lines.append(f"  NOTE: {len(priv)} PRIVATE tt.func -- inlining did not happen here. "
                     f"Not ladder rungs;")
        lines.append(f"  they carry no tile of their own: {', '.join(priv[:4])}")

    # Ops with no `gluon.language` equivalent. This list is hand-maintained and ADVISORY --
    # there is no upstream API to query builtin coverage the way layoutToGluon covers
    # layouts. It exists because `recover` audits layouts and not ops, so a kernel can read
    # as 100% recovered and still not be transcribable: one trial kernel saw
    # `amdg.in_thread_transpose` reported as a SUCCESSFUL row and only found out at authoring.
    hit_ops = sorted({s.op for s in rec.sites if s.op in NO_GLUON_OP})
    if hit_ops:
        lines.append("")
        lines.append("  NOTE: op(s) present here have NO gluon.language equivalent, so a layout")
        lines.append("  count of 100% does NOT mean this body is transcribable (advisory list,")
        lines.append("  hand-maintained -- there is no API to query builtin coverage):")
        for o in hit_ops:
            lines.append(f"    {o}: {NO_GLUON_OP[o]}")

    # LDS footprint. One trial kernel's entire 1.69x residual was the anchor's shared
    # allocation crossing the 64 KiB/CU divisor (40960 vs plain's 32768 -> 2 WG/CU became 1)
    # while every layout verified. Every number needed for that is already parsed here.
    smem = {}
    for s in rec.sites:
        if s.space == "smem" and s.op == "ttg.local_alloc" and s.kind == "result":
            shp = s.alloc_shape or s.shape
            n = 1
            for d in shp:
                n *= d
            smem[s.group] = n
    if smem:
        tot = sum(smem.values())
        lines.append("")
        lines.append(f"  LDS: {len(smem)} allocation(s), {tot} element(s) total across them "
                     f"(x element size = bytes).")
        lines.append("  Compare the ANCHOR's total against this one: an anchor that crosses the")
        lines.append("  65536 B/CU divisor loses a workgroup per CU with every layout still")
        lines.append("  verifying, and `verify` cannot see allocation size.")

    buf = sorted({s.op for s in rec.sites if s.op.startswith("amdg.buffer")})
    if buf:
        lines.append("")
        lines.append(f"  NOTE: this dump uses {', '.join(buf)}. `gluon_to_ttgir` does NOT run")
        lines.append("  `add_convert_to_buffer_ops` (that pass is gated inside make_ttgir), so a")
        lines.append("  plain `gl.load(ptr_tensor)` lowers to `tt.load` on 64-bit pointer tensors")
        lines.append("  instead -- more registers, lower occupancy, narrower loads. Write these")
        lines.append("  accesses as explicit `gl.amd.cdna3.buffer_load/buffer_store` with i32")
        lines.append("  offsets to reproduce what plain lowered to.")
        bf = rec.buffer_facts or {}
        if bf.get("loads"):
            bare = bf["loads"] - bf["with_other"]
            lines.append(f"  COMPILED FORM of the {bf['loads']} buffer_load site(s): "
                         f"{bare} carry a mask and NO `other`,")
            lines.append(f"  {bf['with_other']} carry `other`, {bf['with_contiguity']} carry a "
                         f"`contiguity` attribute.")
            if bare:
                lines.append("  Copy that. `tl.load(..., other=0.0)` in the SOURCE compiles to no")
                lines.append("  `other` operand (buffer OOB returns zero on CDNA), but passing")
                lines.append("  other= in Gluon emits it and costs a v_cndmask per register --")
                lines.append("  measured at 1.2-2% on two kernels. Transcribe the DUMP, not the source.")
            if bf["with_contiguity"]:
                lines.append("  `contiguity` is NOT reachable from gl.amd.cdna3.buffer_load "
                             "(no parameter for it).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. verify -- semantic normal forms
# --------------------------------------------------------------------------- #


def _layout_rank(layout: Any) -> int | None:
    """Rank of a layout, from `.rank` when it exists and from its fields when it does not.

    `DistributedLayout` defines `rank`; `SharedLayout` does NOT -- neither the base class
    nor `SwizzledSharedLayout` has it. Since the rank is what keeps the calls below on
    the safe side of an LLVM assert, it is derived from whichever field pins the rank
    rather than left as None (which would silently skip every shared layout, i.e. skip
    the checks on exactly the layouts this pack spends its rounds on).
    """
    try:
        r = getattr(layout, "rank", None)
        if isinstance(r, int):
            return r
    except Exception:  # noqa: BLE001 - the base class raises NotImplementedError
        return None
    for attr in ("order", "warps_per_cta", "size_per_thread", "threads_per_warp"):
        v = getattr(layout, attr, None)
        if v:
            return len(v)
    for attr in ("offset_bases", "reg_bases", "lane_bases"):
        v = getattr(layout, attr, None)
        if v and v[0]:
            return len(v[0])
    v = getattr(layout, "shape", None)
    if v:
        return len(v)
    # A SliceLayout removes one dim from its parent, and its own `rank` property covers
    # that -- this is only the fallback for a hand-built stub in the selftest.
    parent = getattr(layout, "parent", None)
    if parent is not None:
        pr = _layout_rank(parent)
        return None if pr is None else max(pr - 1, 1)
    return None


def _shape_for_layout(layout: Any, shape: list) -> list | None:
    """The sub-shape ``to_linear_layout`` may be called with, or None if it may not.

    This guard is NOT defensive style, it is a hard requirement. ``toLinearLayout``
    indexes the shape by the layout's rank behind an ``assert`` in LLVM's
    ``ArrayRef::operator[]``, so a rank mismatch ABORTS the process -- there is no
    exception to catch and no traceback. The case that hits it in practice is the one
    the whole pack is about: a multi-buffered staging allocation prints as
    ``memdesc<2x128x64xf16, #ttg.swizzled_shared<...>>``, a rank-3 shape carrying a
    rank-2 layout. The layout describes ONE buffer, so the trailing dims are the
    correct sub-shape, and the leading dims are the buffer count.
    """
    rank = _layout_rank(layout)
    if rank is None:
        return None
    if len(shape) == rank:
        return list(shape)
    if len(shape) > rank:
        return list(shape[-rank:])
    return None


def _canonical_enc(builder, layout: Any, shape: list, space: str) -> str | None:
    """The layout as the COMPILER prints it, obtained by building a type around it.

    The fallback equivalence basis where the LinearLayout normal form is unavailable.
    Weaker than the normal form -- two different spellings of one layout compare
    unequal -- but sound in the direction that matters: equal text means equal layout.
    Available on every build, because printing goes through `Type.__str__`.
    """
    sub = _shape_for_layout(layout, shape)
    if sub is None or any(d <= 0 for d in sub):
        return None
    try:
        attr = layout._to_ir(builder)
        elem = _element_ty(builder, "f16")
        ty = (builder.get_distributed_ty(elem, sub, attr) if space == "reg"
              else builder.get_shared_mem_desc_ty(elem, sub, attr, sub))
        got = _split_type(str(ty))
        return None if got is None else "TEXT:" + _norm(got["enc"])
    except Exception:  # noqa: BLE001
        return None


def _shape_for_layout_or(site: Site) -> list:
    """Comparison shape for a site whose layout could NOT be recovered.

    Such a site has no Gluon object and therefore no normal form, so reconciliation can
    only match it on shape. Uses the same allocation-vs-view rule as everything else so an
    excluded subview lines up with the anchor's replacement.
    """
    return _sizing_shape(site)


def _sizing_shape(site: Site) -> list:
    """The shape a layout must be evaluated at for THIS use site.

    For a shared SUBVIEW that is the allocation shape, not the view shape: the layout's
    LinearLayout is sized by the allocation, and evaluating it at the view shape trips
    `assert(size == llSize)` inside `SharedLinearEncodingAttr::toLinearLayout` -- a
    process abort with no traceback, and it takes the whole verdict with it including the
    distributed half. A subview is the only way to reproduce plain's LDS overlap, and
    `shared_linear` is a layout this tool itself emits, so the tool could emit a construct
    its own `verify` could not ingest.
    """
    if site.space != "reg" and site.alloc_shape:
        return list(site.alloc_shape)
    return list(site.shape)


def _normal_form(builder, layout: Any, shape: list, space: str = "reg") -> str | None:
    """Comparable canonical form of a layout at a shape.

    Prefers the LinearLayout normal form: two layouts are the same iff they map
    (register, lane, warp, block) to the same tensor coordinates. That is the right
    equivalence relation -- comparing printed attributes makes two spellings of one
    layout look different, and comparing attribute COUNTS makes an unrolled loop look
    like a missing layout.

    Falls back to canonical attribute text for the (space, build) combinations where
    normalisation is not available -- on Triton 3.6 that is every shared layout, where
    calling it would abort the process (see `capabilities`). The prefixes keep the two
    bases from ever comparing equal to each other by accident.
    """
    caps = capabilities()
    can_norm = caps["shared_norm"] if space != "reg" else caps["distributed_norm"]
    if not can_norm:
        return _canonical_enc(builder, layout, shape, space)
    sub = _shape_for_layout(layout, shape)
    if sub is None or any(d <= 0 for d in sub):
        return None
    try:
        attr = layout._to_ir(builder)
        return "NORM:" + repr(builder.to_linear_layout(attr, sub))
    except Exception:  # noqa: BLE001
        return _canonical_enc(builder, layout, shape, space)


def _basis_note(n_shared: int | None = None) -> str:
    """One line naming which equivalence basis each space is compared on, and why.

    Printed in every verdict: a reader who does not know that the shared layouts were
    compared as TEXT on this build cannot judge how much a PASS is worth.

    `n_shared` is the number of shared layouts actually compared. On a kernel with none,
    the < 3.7 caveat applies to nothing and printing it invites discounting a verdict that
    rests on exactly the same normal form 3.7 would have used -- which is what happened to
    a dot-free kernel whose 3.6.0 PASS was fully normalised.
    """
    caps = capabilities()
    parts = []
    for space, key in (("distributed", "distributed_norm"), ("shared", "shared_norm")):
        if space == "shared" and n_shared == 0:
            parts.append("shared=n/a (this comparison has no shared layouts)")
            continue
        parts.append(f"{space}={'LinearLayout normal form' if caps[key] else 'canonical text'}")
    note = ", ".join(parts)
    if n_shared == 0:
        return note + f"  (caps probed via {caps['probed']})"
    if not (caps["shared_norm"] and caps["distributed_norm"]):
        note += ("  [this build cannot normalise every space -- Triton < 3.7 aborts on"
                 " to_linear_layout of a shared layout, so those fall back to text,"
                 " which is sound but stricter: two spellings of one layout read as"
                 " different]")
    return note + f"  (caps probed via {caps['probed']})"


def _mma_signature(rec: Recovery) -> set:
    """(warps_per_cta, instr_shape) of every MMA layout in a dump.

    The MMA layout is the one that pins the whole recovered family together, so it is
    the cheapest single fingerprint of "same config".
    """
    out = set()
    for s in rec.ok_sites:
        lay = s.layout
        for _ in range(8):
            if lay is None:
                break
            if type(lay).__name__ in ("AMDMFMALayout", "AMDWMMALayout", "NVMMADistributedLayout"):
                out.add((tuple(getattr(lay, "warps_per_cta", ()) or ()),
                         tuple(getattr(lay, "instr_shape", ()) or ())))
                break
            lay = getattr(lay, "parent", None)
    return out


def _config_mismatch(plain: Recovery, anchor: Recovery) -> list:
    """Reasons the two dumps are not the same config, or [] if they are."""
    out = []
    if (plain.num_warps is not None and anchor.num_warps is not None
            and plain.num_warps != anchor.num_warps):
        out.append(f"num_warps: plain={plain.num_warps} anchor={anchor.num_warps}")
    if (plain.threads_per_warp is not None and anchor.threads_per_warp is not None
            and plain.threads_per_warp != anchor.threads_per_warp):
        out.append(f"threads_per_warp: plain={plain.threads_per_warp} "
                   f"anchor={anchor.threads_per_warp}")
    pm, am = _mma_signature(plain), _mma_signature(anchor)
    if pm and am and pm != am:
        out.append(f"MMA family: plain={sorted(pm)} anchor={sorted(am)}")
    return out


def verify(plain_path: str, anchor_path: str, arch: str,
           warp_size: int | None = None) -> tuple[str, str, dict]:
    """Returns (status, report, data). status is PASS | RECONCILED | FAIL | NOT_COMPARABLE.

    Four states, not two, each with its own exit code (0 / 0 / 1 / 3), because each asks
    for a different next action. A config mismatch is not a failed transcription, and a
    difference with a structural cause is not one either -- reporting either as FAIL sends
    the reader off to fix layouts that are correct.
    """
    plain = recover(plain_path, arch, warp_size, roundtrip=False)
    anchor = recover(anchor_path, arch, warp_size, roundtrip=False)
    return _verify_recs(plain, anchor, arch, warp_size,
                        plain_path=plain_path, anchor_path=anchor_path)


def _verify_recs(plain: Recovery, anchor: Recovery, arch: str,
                 warp_size: int | None = None,
                 plain_path: str = "<plain>", anchor_path: str = "<anchor>"):
    """The comparison itself, on already-recovered modules.

    Split out from `verify` so the four verdict states can be tested on hand-built pairs.
    Reconciliation logic that is only reachable through two real .ttgir files is logic that
    only gets exercised when a kernel happens to trip it.
    """
    _, builder = make_context(arch, warp_size)

    def norms(rec: Recovery) -> dict:
        """(sub-shape, normal form) -> sites.

        Keyed on the sub-shape the layout actually describes, not on the printed shape,
        so plain's single-buffered `memdesc<128x64>` and an anchor's double-buffered
        `memdesc<2x128x64>` compare as the same layout -- which they are. The buffer
        COUNT is a pipeline fact, not a layout fact, and conflating the two is what
        makes a correct transcription look like a layout mismatch.
        """
        out: dict = {}
        for s in rec.ok_sites:
            sz = _sizing_shape(s)
            nf = _normal_form(builder, s.layout, sz, s.space)
            if nf is None:
                continue
            sub = _shape_for_layout(s.layout, sz)
            out.setdefault((tuple(sub), nf), []).append(s)
        return out

    # Config precheck FIRST. A layout diff between two dumps taken at different
    # configs is not a transcription verdict -- every layout differs because
    # `warps_per_cta` and the tile shapes are baked into all of them, and the report
    # is then fifteen rows of noise that look like fifteen bugs. This is the
    # `[CONFIG]` entry condition, checked mechanically instead of remembered.
    cfg = _config_mismatch(plain, anchor)
    if cfg:
        return "NOT_COMPARABLE", "\n".join(
            ["LAYOUT EQUIVALENCE: NOT COMPARABLE -- CONFIG MISMATCH",
             f"  plain  = {plain_path}",
             f"  anchor = {anchor_path}"]
            + [f"  {c}" for c in cfg]
            + ["",
               "  The two dumps are not the same config, so a layout diff says nothing about",
               "  the transcription. Re-dump the anchor at the champion's PINNED config, or",
               "  re-dump plain at the anchor's -- but do not read the diff below either way.",
               "  A transcribed Gluon kernel cannot follow plain to a different tile: the",
               "  recovered layouts hard-code warps_per_cta, so a tile change invalidates the",
               "  whole recovered family and the anchor every later number is measured against."]), \
            {"status": "NOT_COMPARABLE", "config_mismatch": cfg}

    p, a = norms(plain), norms(anchor)
    missing = [k for k in p if k not in a]
    extra = [k for k in a if k not in p]

    # --- Reconciliation. Two differences are STRUCTURAL rather than transcription errors,
    # and calling either one FAIL sends the reader off to fix layouts that are correct.
    #
    # (R1) The anchor supplies a layout at a shape where PLAIN carried one the tool could
    #      not read (UNRECOVERABLE) and therefore excluded. The anchor is then penalised
    #      for correctly reproducing the very thing the comparison dropped, and no correct
    #      transcription of such a kernel can ever PASS. Matched on shape, because the
    #      excluded side has no normal form by construction -- `layoutToGluon` threw before
    #      one could be computed -- so this is a PROBABLE reconciliation and is labelled as
    #      one, never silently folded into a PASS.
    #
    # (R2) A faithful non-pipelined anchor against a PIPELINED plain. Plain's IR carries
    #      the union of a K-sliced loop body and a peeled full-K epilogue, so it holds dot
    #      layouts at K-shapes the anchor's single decomposition never produces. The tell
    #      is a MISSING row whose normal form the anchor does have at a DIFFERENT shape,
    #      with EXTRA empty: the layout family is recovered, only the loop transformation
    #      differs. EXTRA being empty is what separates this from "the compiler chose a
    #      layout because you did not", and the report never used to say so.
    excl_shapes = {tuple(_shape_for_layout_or(s)) for s in plain.failed_sites}
    # R2 keys on the layout CONSTRUCTOR, not on the normal form. A normal form is tied to
    # the shape it was evaluated at, so "the same normal form at another shape" is a
    # contradiction and that rule could never have fired. What actually happens is the
    # identical `DotOperandLayout(...)` appearing at K=16 in plain's pipelined loop body
    # and at K=64 in its peeled epilogue, while the anchor has one decomposition: same
    # constructor, different extents.
    a_ctors = {repr(a[k][0].layout) for k in a}
    rec_r1 = [k for k in extra if tuple(k[0]) in excl_shapes]
    rec_r2 = [k for k in missing if repr(p[k][0].layout) in a_ctors] if not extra else []
    # (R3) A MISSING layout whose sites belong to an op the LANGUAGE cannot express. The
    #      anchor did not fail to reproduce it -- Gluon has no builtin to reproduce it
    #      with, so no correct transcription of this body can ever produce that row. Two
    #      trial kernels were graded FAIL on anchors that were bit-exact AND faster than
    #      plain, on this row alone, which is the verdict telling the author to go find a
    #      mistake that does not exist. Kept visible and named, never folded into a PASS.
    rec_r3 = [k for k in missing
              if k not in rec_r2 and any(s.op in NO_GLUON_OP for s in p[k])]
    unexplained_missing = [k for k in missing if k not in rec_r2 and k not in rec_r3]
    unexplained_extra = [k for k in extra if k not in rec_r1]

    if not missing and not extra:
        status = "PASS"
    elif not unexplained_missing and not unexplained_extra:
        status = "RECONCILED"
    else:
        status = "FAIL"

    _n_shared = sum(1 for k in set(p) | set(a)
                    if (p.get(k) or a.get(k))[0].space != "reg")
    lines = ["LAYOUT EQUIVALENCE: " + status,
             f"  plain  = {plain_path}",
             f"  anchor = {anchor_path}",
             (f"  config: num_warps={plain.num_warps} "
              f"threads_per_warp={plain.threads_per_warp} (matched)"),
             f"  basis: {_basis_note(_n_shared)}",
             f"  {len(p)} distinct (shape, normal-form) in plain, {len(a)} in anchor"]
    if status == "RECONCILED":
        lines.append("  Every difference is accounted for by a STRUCTURAL cause, not by a")
        lines.append("  transcription error. See RECONCILED below; treat this as a pass on the")
        lines.append("  layout question and read the cause, which is a real fact about the anchor.")

    def _rows(keys, side, sign):
        for k in keys:
            s = side[k][0]
            lines.append(f"    {sign} {s.role:<18} {s.enc_kind:<18} shape={list(k[0])}  "
                         f"({len(side[k])} sites)")
            lines.append(f"        {repr(s.layout)[:140]}")

    if unexplained_missing:
        lines.append("  MISSING -- in plain, not reproduced by the anchor:")
        _rows(unexplained_missing, p, "-")
    if unexplained_extra:
        lines.append("  EXTRA -- introduced by the anchor, absent in plain:")
        _rows(unexplained_extra, a, "+")
        if not unexplained_missing:
            lines.append("      (EXTRA with an EMPTY missing list does NOT match the "
                         "'the compiler chose a layout because you did not' signature --")
            lines.append("       that one shows up as MISSING+EXTRA at the same shape. Look for a "
                         "copy plain materialises that the anchor folded.)")
    if rec_r1 or rec_r2 or rec_r3:
        lines.append("  RECONCILED -- differences with a structural cause:")
        for k in rec_r3:
            s = p[k][0]
            ops = sorted({x.op for x in p[k] if x.op in NO_GLUON_OP})
            lines.append(f"    ~ {s.role:<18} {s.enc_kind:<18} shape={list(k[0])}  MISSING, "
                         f"produced by an op Gluon cannot express")
            for o in ops:
                lines.append(f"        {o}: {NO_GLUON_OP[o]}")
            lines.append("        no correct transcription of this body can produce this row; "
                         "verify cannot")
            lines.append("        tell you the substitute is FREE -- only the ISA can.")
        for k in rec_r1:
            s = a[k][0]
            lines.append(f"    ~ {s.role:<18} {s.enc_kind:<18} shape={list(k[0])}  EXTRA at a "
                         f"shape where plain carried an UNRECOVERABLE layout")
            lines.append("        probable, matched on shape only: the excluded side has no "
                         "normal form to compare (layoutToGluon threw)")
        for k in rec_r2:
            s = p[k][0]
            lines.append(f"    ~ {s.role:<18} {s.enc_kind:<18} shape={list(k[0])}  MISSING, but "
                         f"the anchor has this SAME layout at another shape")
            lines.append("        i.e. the layout family is recovered and only the loop "
                         "decomposition differs (pipelined plain vs single-dot anchor)")

    lines.append("")
    lines.append("  MULTIPLICITY (informational -- never gates):")
    lines.append("    A plain loop that the AMD pipeliner unrolled prints each layout once")
    lines.append("    per unrolled body, so plain:anchor ratios above 1 are EXPECTED and are")
    lines.append("    a read on the unroll factor, not on a missing layout.")
    for k in sorted(set(p) & set(a), key=lambda k: -len(p[k])):
        s = p[k][0]
        lines.append(f"    {s.role:<18} plain x{len(p[k]):<4} anchor x{len(a[k]):<4} "
                     f"ratio {len(p[k]) / max(len(a[k]), 1):.2f}")

    for rec, tag in ((plain, "plain"), (anchor, "anchor")):
        if rec.failed_sites:
            lines.append(f"  NOTE: {len(rec.failed_sites)} UNRECOVERABLE layout(s) in {tag} "
                         f"were excluded from the comparison:")
            for s in rec.failed_sites:
                lines.append(f"    {s.enc_kind} at {s.op} {s.kind}[{s.index}]")
        if rec.warp_check.startswith("FAIL"):
            lines.append(f"  NOTE: {tag} num_warps cross-check {rec.warp_check}")

    def _pack(keys, side):
        out = []
        for k in keys:
            s = side[k][0]
            out.append({"role": s.role, "enc_kind": s.enc_kind, "shape": list(k[0]),
                        "sites": len(side[k]), "layout": repr(s.layout),
                        "normal_form": k[1]})
        return out

    # Machine-readable arrays alongside the prose. Both diagnoses this trial produced had
    # to be regexed out of rendered English, which is not a report, it is a screenshot.
    data = {
        "status": status,
        "counts": {"plain": len(p), "anchor": len(a),
                   "missing": len(unexplained_missing), "extra": len(unexplained_extra),
                   "reconciled": len(rec_r1) + len(rec_r2) + len(rec_r3)},
        "missing": _pack(unexplained_missing, p),
        "extra": _pack(unexplained_extra, a),
        "reconciled": ([dict(_pack([k], a)[0], reason="extra_at_unrecoverable_shape")
                        for k in rec_r1]
                       + [dict(_pack([k], p)[0], reason="same_normal_form_other_shape")
                          for k in rec_r2]
                       + [dict(_pack([k], p)[0], reason="op_has_no_gluon_equivalent",
                               ops=sorted({x.op for x in p[k] if x.op in NO_GLUON_OP}))
                          for k in rec_r3]),
        "multiplicity": [{"role": p[k][0].role, "shape": list(k[0]),
                          "plain": len(p[k]), "anchor": len(a[k]),
                          "ratio": round(len(p[k]) / max(len(a[k]), 1), 3)}
                         for k in sorted(set(p) & set(a), key=lambda k: -len(p[k]))],
        "unrecoverable": {tag: [{"enc_kind": s.enc_kind, "op": s.op,
                                 "position": f"{s.kind}[{s.index}]", "shape": s.shape}
                                for s in r.failed_sites]
                          for r, tag in ((plain, "plain"), (anchor, "anchor"))},
        "num_warps_check": {"plain": plain.warp_check, "anchor": anchor.warp_check},
        "basis": _basis_note(_n_shared),
    }
    return status, "\n".join(lines), data


# --------------------------------------------------------------------------- #
# 5. view
# --------------------------------------------------------------------------- #


def _clip(text: str, max_rows: int, max_cols: int) -> str:
    """Trim a layout view to something a terminal and an agent can both read.

    A wide memory tile is not a small object: one role of a `[1, 32, 128]` tile renders
    as 32 rows of ~4000 characters, 37 KB for a single call. Emitting that unannounced
    makes `view` the subcommand nobody dares run on a memory-bound kernel -- which is
    exactly the kernel whose `ds_read`/`ds_write` argument it exists to settle.
    """
    # 0 means "no limit", and that rule lives HERE rather than in the caller. Splitting
    # it across caller and callee is how `view` and `recover` came to disagree about a
    # role name in the first place.
    max_rows = max_rows or 10 ** 9
    max_cols = max_cols or 10 ** 9
    rows = text.splitlines()
    out, clipped_cols = [], 0
    for r in rows[:max_rows]:
        if len(r) > max_cols:
            clipped_cols += 1
            r = r[:max_cols] + f"  ... +{len(r) - max_cols} chars"
        out.append(r)
    notes = []
    if len(rows) > max_rows:
        notes.append(f"{len(rows) - max_rows} of {len(rows)} rows elided")
    if clipped_cols:
        notes.append(f"{clipped_cols} row(s) truncated in width")
    if notes:
        out.append(f"  [{'; '.join(notes)} -- raise --max-rows / --max-cols, or 0 for all]")
    return "\n".join(out)


def view(rec: Recovery, role: str, hardware: bool,
         max_rows: int = 24, max_cols: int = 200) -> tuple[bool, str]:
    """Render one or more recovered layouts. Returns (found, text).

    `role` matches the EMITTED SYMBOL first (unique by construction) and the raw role
    second. When a bare role is ambiguous -- several distinct layouts share it -- every
    candidate is shown with its symbol, rather than one being picked silently.
    """
    from triton._C.libtriton import gluon_ir
    named = assign_names(rec)
    want = role.strip().upper()
    exact = [(n, k, ss) for n, k, ss in named if n == want]
    if exact:
        chosen, ambiguous = exact, False
    else:
        chosen = [(n, k, ss) for n, k, ss in named
                  if any(s.role.upper() == want for s in ss)]
        ambiguous = len(chosen) > 1
    if not chosen:
        avail = ", ".join(n for n, _k, _s in named)
        return False, (f"no layout for {role!r}.\nAvailable symbols: {avail}\n"
                       f"(pass a symbol from that list, or a raw role name)")

    out = []
    if ambiguous:
        out.append(f"NOTE: role {role!r} is carried by {len(chosen)} DISTINCT layouts; "
                   f"all are shown. Wire the one whose symbol you actually use.")
    for name, _key, sites in chosen:
        s = sites[0]
        others = sorted({x.role for x in sites} - {s.role})
        out.append("")
        out.append(f"{name}  ({s.enc_kind})  shape={s.shape} {s.dtype} ({s.space})"
                   + (f"  also serves: {', '.join(others)}" if others else ""))
        out.append(f"  {s.layout!r}")
        if not capabilities()["layout_view"]:
            out.append("  [no view: this Triton has no gluon_ir.get_layout_view "
                       "(added in 3.7). recover/verify still work; only `view` needs it]")
            continue
        # The SAME rank guard `_normal_form` and `_roundtrip` already use, and it was
        # missing here alone. `get_layout_view` indexes the shape by the layout's rank
        # behind an LLVM assert, so a rank-3 use site carrying a rank-2 shared layout --
        # every multi-buffered staging allocation -- ABORTS the interpreter. Not
        # catchable: the `except` below never runs. Three separate agents hit it on
        # `--role A_SMEM`, i.e. on the single use case this subcommand is documented for.
        sub = _shape_for_layout(s.layout, _sizing_shape(s))
        if sub is None:
            out.append(f"  [no view: rank {_layout_rank(s.layout)} layout at shape "
                       f"{s.shape}; no sub-shape it can be rendered at]")
            continue
        if sub != list(s.shape):
            out.append(f"  [rendered at {sub}, the sub-shape this layout describes; the "
                       f"use site is {s.shape} (leading dims are the buffer count)]")
        try:
            raw = gluon_ir.get_layout_view(s.layout, sub, hardware)
            out.append(_clip(raw, max_rows, max_cols))
        except Exception as e:  # noqa: BLE001
            out.append(f"  [no view: {type(e).__name__}: {e}]")
    return True, "\n".join(out)


# --------------------------------------------------------------------------- #
# 6. selftest
# --------------------------------------------------------------------------- #

_SELFTEST_TTGIR = '''\
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %a = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    %b = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked1>
    %sa = ttg.local_alloc %a : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %la = ttg.local_load %sa : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %lb = ttg.convert_layout %b : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %d = tt.dot %la, %lb, %cst : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
'''


def _selftest() -> int:
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("ttgir_bridge selftest")

    # --- pure string layer: no Triton needed, so a broken install still tests this.
    t = _split_type("tensor<128x64xf16, #ttg.blocked<{sizePerThread = [8, 1], order = [0, 1]}>>")
    ck("_split_type tensor: shape", t and t["shape"] == [128, 64], repr(t))
    ck("_split_type tensor: dtype", t and t["dtype"] == "f16", repr(t))
    ck("_split_type tensor: kind", t and t["kind"] == "blocked", repr(t))
    m = _split_type("!ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, order = [1, 0]}>, "
                    "#ttg.shared_memory, mutable>")
    ck("_split_type memdesc: space", m and m["space"] == "smem", repr(m))
    ck("_split_type memdesc: layout only (commas inside the body do not split)",
       m and m["enc"].endswith("}>") and "shared_memory" not in m["enc"], repr(m))
    ck("_split_type memdesc: no allocShape when the view IS the allocation",
       m and m["alloc_shape"] is None, repr(m))
    sv = _split_type("!ttg.memdesc<128x128xi8, #ttg.swizzled_shared<{vec = 4, order = [1, 0]}>, "
                     "#ttg.shared_memory, mutable, 256x128>")
    ck("_split_type memdesc: SUBVIEW allocShape is captured",
       sv and sv["alloc_shape"] == [256, 128] and sv["shape"] == [128, 128], repr(sv))
    # A shared layout is sized by the ALLOCATION. Evaluating it at the view shape aborts
    # the process inside SharedLinearEncodingAttr::toLinearLayout, so this must be a
    # computed decision and not a caught exception.
    _sv_site = Site(op="ttg.memdesc_index", kind="result", index=0, shape=[128, 128],
                    dtype="i8", space="smem", enc_kind="swizzled_shared", enc_text="#x",
                    alloc_shape=[256, 128])
    ck("_sizing_shape uses allocShape for a shared subview",
       _sizing_shape(_sv_site) == [256, 128])
    ck("_sizing_shape uses the view shape when there is no allocShape",
       _sizing_shape(Site(op="ttg.local_alloc", kind="result", index=0, shape=[64, 64],
                          dtype="i8", space="smem", enc_kind="swizzled_shared",
                          enc_text="#x")) == [64, 64])
    ck("_sizing_shape never rewrites a register site",
       _sizing_shape(Site(op="tt.load", kind="result", index=0, shape=[64, 64], dtype="f16",
                          space="reg", enc_kind="blocked", enc_text="#x",
                          alloc_shape=[2, 64, 64])) == [64, 64])
    ck("_split_type rejects a layout-free type", _split_type("!tt.ptr<f16>") is None)
    ck("_first_top_level_field respects nesting",
       _first_top_level_field("#ttg.x<{a = [1, 2], b = 3}>, #ttg.shared_memory")
       == "#ttg.x<{a = [1, 2], b = 3}>")
    ck("_norm collapses whitespace", _norm("a  =\n [1,  2]") == "a = [1, 2]")

    # The compiled form of a buffer_load. `other` costs a v_cndmask per register and the
    # SOURCE's other=0.0 does not appear in the dump, so this distinction is worth 1-2%.
    _bf = _buffer_facts(
        'x = amdg.buffer_load %V[%off], %mask {contiguity = 8 : i32} : tensor<64xf16>\n'
        'y = amdg.buffer_load %Q[%o2], %m2, %cst : tensor<64xf16>\n')
    ck("_buffer_facts counts loads", _bf["loads"] == 2, str(_bf))
    ck("_buffer_facts separates the `other` form from the mask-only form",
       _bf["with_other"] == 1, str(_bf))
    ck("_buffer_facts sees the contiguity attribute", _bf["with_contiguity"] == 1, str(_bf))

    # Source name + line from an MLIR location. This is what replaced guessing which of
    # Q/K/V a layout named GLOBAL_LOAD belonged to.
    class _FakeVal:
        def __init__(self, loc):
            self._loc = loc

        def get_loc(self):
            return self._loc

    ck("_src_of reads the source variable name and line",
       _src_of(_FakeVal('loc("Q"("/a/b/fwd_prefill.py":895:0))')) == "Q @ fwd_prefill.py:895",
       _src_of(_FakeVal('loc("Q"("/a/b/fwd_prefill.py":895:0))')))
    ck("_src_of is empty when there is no location", _src_of(object()) == "")

    # tt.func visibility: a ladder rung is PUBLIC. Gating on the total count both fires
    # falsely on a noinline callee and misses a real ladder (each rung its own module).
    _vis = "  tt.func public @rung_a() {\n  tt.func private @__helper__i32() {\n"
    ck("_FUNC_RE captures visibility",
       [(m.group(1), m.group(2)) for m in _FUNC_RE.finditer(_vis)]
       == [("public", "rung_a"), ("private", "__helper__i32")])

    # roles
    ck("_base_role reads A/B from the dot_op's own opIdx",
       _base_role("ttg.local_load", "result", 0, "dot_op",
                  type("L", (), {"operand_index": 1})()) == "B_DOT_OPERAND")
    ck("_base_role global load", _base_role("tt.load", "result", 0, "blocked", None) == "GLOBAL_LOAD")
    ck("_base_role names the AMD dialect correctly (amdg, not amdgpu)",
       _base_role("amdg.buffer_load", "result", 0, "blocked", None) == "GLOBAL_LOAD")
    ck("_base_role names an mma layout for what it is, not for its last use site",
       _base_role("arith.constant", "result", 0, "amd_mfma", None) == "MMA"
       and _base_role("amdg.buffer_store", "operand", 0, "amd_mfma", None) == "MMA")
    ck("_base_role names index math", _base_role("tt.make_range", "result", 0, "slice", None) == "INDEX")
    ck("_base_role splits a shared-memory op by SPACE, not by op name",
       _base_role("ttg.local_store", "operand", 1, "swizzled_shared", None, "smem") == "SMEM"
       and _base_role("ttg.local_store", "operand", 0, "blocked", None, "reg") == "TO_SMEM")

    # operand attribution: load -> local_store -> memdesc -> local_load -> dot_op(opIdx)
    def _s(op, kind, index, enc, space="reg", shape=(64, 64), enc_kind="blocked",
           layout=None, group=0):
        st = Site(op=op, kind=kind, index=index, shape=list(shape), dtype="f16", space=space,
                  enc_kind=enc_kind, enc_text=enc, group=group)
        st.layout = layout if layout is not None else object()
        st.role = _base_role(op, kind, index, enc_kind, layout, space)
        return st

    dop_b = type("L", (), {"operand_index": 1})()
    chain = [
        _s("amdg.buffer_load", "result", 0, "#B_REG", group=0),
        _s("ttg.local_store", "operand", 0, "#B_REG", group=1),
        _s("ttg.local_store", "operand", 1, "#B_SH", space="smem", group=1),
        _s("ttg.local_alloc", "result", 0, "#B_SH", space="smem", group=2),
        _s("ttg.local_load", "operand", 0, "#B_SH", space="smem", group=3),
        _s("ttg.local_load", "result", 0, "#B_DOP", enc_kind="dot_op", layout=dop_b, group=3),
    ]
    _attribute_operands(chain)
    roles = {(s.op, s.kind, s.index): s.role for s in chain}
    ck("attribution: local_load's opIdx labels the staging buffer",
       roles[("ttg.local_alloc", "result", 0)] == "B_SMEM", str(roles))
    ck("attribution: the buffer then labels the global load that fed it",
       roles[("amdg.buffer_load", "result", 0)] == "B_LOAD", str(roles))
    ck("attribution: an unrelated layout keeps its base role",
       _base_role("tt.splat", "result", 0, "blocked", None) == "INDEX")

    # The B side of a preshuffled GEMM reaches its dot operand through a reshape/trans
    # chain and reads back into a linear layout, not into dot-operand form. Requiring a
    # dot_op result at the local_load left the whole chain generic (TO_SMEM / SMEM /
    # FROM_SMEM / TT_RESHAPE), and the runbook's checklist then says nothing about B.
    dop_a = type("L", (), {"operand_index": 0})()
    chain2 = [
        _s("amdg.buffer_load", "result", 0, "#BREG", group=10),
        _s("ttg.local_store", "operand", 0, "#BREG", group=11),
        _s("ttg.local_store", "operand", 1, "#BSH", space="smem", group=11),
        _s("ttg.local_alloc", "result", 0, "#BSH", space="smem", group=12),
        _s("ttg.local_load", "operand", 0, "#BSH", space="smem", group=13),
        _s("ttg.local_load", "result", 0, "#BLIN", enc_kind="linear", group=13),
        _s("tt.trans", "operand", 0, "#BLIN", enc_kind="linear", group=14),
        _s("tt.trans", "result", 0, "#BTR", enc_kind="linear", group=14),
        _s("tt.reshape", "operand", 0, "#BTR", enc_kind="linear", group=15),
        _s("tt.reshape", "result", 0, "#BDOP", enc_kind="dot_op", layout=dop_a, group=15),
    ]
    _attribute_operands(chain2)
    r2 = {(s.op, s.kind, s.index): s.role for s in chain2}
    ck("attribution: a letter crosses a reshape/trans chain to the staging buffer",
       r2[("ttg.local_alloc", "result", 0)] == "A_SMEM", str(r2))
    ck("attribution: and reaches the global load that fed it",
       r2[("amdg.buffer_load", "result", 0)] == "A_LOAD", str(r2))
    ck("attribution: the read-back layout is named for its operand, not generically",
       r2[("ttg.local_load", "result", 0)] == "A_FROM_SMEM", str(r2))
    ck("attribution: intermediate relayouts say where they are going",
       r2[("tt.trans", "result", 0)] == "A_PRE_DOT", str(r2))
    ck("_ROLE_RANK prefers a dot operand over a constant",
       _ROLE_RANK["A_DOT_OPERAND"] > _ROLE_RANK.get("ARITH_CONSTANT", 0))

    # _warps_per_cta follows parents
    leaf = type("L", (), {"warps_per_cta": [2, 2], "parent": None})()
    slice_ = type("S", (), {"parent": leaf})()
    ck("_warps_per_cta follows parent chain", _warps_per_cta(slice_) == [2, 2])
    ck("_warps_per_cta returns None when absent",
       _warps_per_cta(type("X", (), {"parent": None})()) is None)

    # _gl_expr qualification
    ck("_gl_expr qualifies gl.*",
       _gl_expr(type("BlockedLayout", (), {"__repr__": lambda s: "BlockedLayout(size_per_thread=[1])"})())
       == "gl.BlockedLayout(size_per_thread=[1])")
    ck("_gl_expr qualifies gl.amd.* for MFMA",
       _gl_expr(type("AMDMFMALayout", (), {"__repr__": lambda s: "AMDMFMALayout(version=3)"})())
       == "gl.amd.AMDMFMALayout(version=3)")
    ck("_gl_expr qualifies a nested parent",
       "gl.amd.AMDMFMALayout" in _gl_expr(
           type("DotOperandLayout", (),
                {"__repr__": lambda s: "DotOperandLayout(operand_index=0, "
                                       "parent=AMDMFMALayout(version=3))"})()))

    # Pipeline facts, read from the dump. Whether plain was pipelined decides what ratio
    # to EXPECT, and reading it off the SOURCE set a wrong expectation for a control kernel
    # in this trial -- its file said num_stages=1 while the dispatched branch ran at 2.
    pf2 = _pipeline_facts('scf.for %i = %c0 to %n step %c1 iter_args(%a = %x, %b = %y) '
                          '{ } {tt.num_stages = 2 : i32}')
    ck("_pipeline_facts reads tt.num_stages from the dump", pf2["max_num_stages"] == 2, str(pf2))
    ck("_pipeline_facts counts loop carries", pf2["max_iter_args"] == 2, str(pf2))
    pf1 = _pipeline_facts('scf.for %i = %c0 to %n step %c1 { } {tt.num_stages = 1 : i32}')
    ck("_pipeline_facts sees a NON-pipelined loop",
       pf1["max_num_stages"] == 1 and pf1["max_iter_args"] == 0, str(pf1))
    ck("_pipeline_facts on a loop-free module is empty",
       _pipeline_facts("tt.func @k() { tt.return }")["loops"] == 0)

    ck("_backend_key amd", _backend_key("gfx942") == ("amd", "hip", "gfx942", 64))
    ck("_backend_key nvidia via sm90", _backend_key("sm90") == ("nvidia", "cuda", 90, 32))
    ck("_backend_key nvidia via bare capability", _backend_key("100")[0] == "nvidia")
    try:
        _backend_key("hopper")
        ck("_backend_key refuses an unrecognised arch", False)
    except SystemExit:
        ck("_backend_key refuses an unrecognised arch", True)

    # The rank guard. A multi-buffered staging allocation is a rank-3 shape carrying a
    # rank-2 layout, and calling to_linear_layout on that mismatch trips an assert in
    # LLVM's ArrayRef -- a process ABORT, not an exception. So this is checked, not caught.
    ck("_shape_for_layout trims a leading buffer dim",
       _shape_for_layout(type("L", (), {"rank": 2})(), [2, 128, 64]) == [128, 64])
    ck("_shape_for_layout passes a matching rank through",
       _shape_for_layout(type("L", (), {"rank": 2})(), [128, 64]) == [128, 64])
    ck("_shape_for_layout refuses a shape below the layout rank",
       _shape_for_layout(type("L", (), {"rank": 3})(), [128, 64]) is None)
    ck("_shape_for_layout refuses a layout with no rank",
       _shape_for_layout(object(), [128, 64]) is None)

    # A config mismatch is its own state: reporting it as FAIL sends the reader off to
    # fix layouts that were never wrong.
    def _rec(nw, tpw=64, wpc=(2, 2)):
        r = Recovery(path="x", arch="gfx942", num_warps=nw, threads_per_warp=tpw)
        lay = type("AMDMFMALayout", (), {"warps_per_cta": list(wpc),
                                         "instr_shape": [16, 16, 16], "parent": None})()
        r.sites = [Site(op="tt.dot", kind="result", index=0, shape=[64, 64], dtype="f32",
                        space="reg", enc_kind="amd_mfma", enc_text="#x", layout=lay)]
        return r
    ck("_config_mismatch: matching configs are comparable",
       _config_mismatch(_rec(4), _rec(4)) == [])
    ck("_config_mismatch: num_warps difference is named",
       any("num_warps: plain=4 anchor=8" in c for c in _config_mismatch(_rec(4), _rec(8))))
    ck("_config_mismatch: an MMA warps_per_cta difference is named",
       any("MMA family" in c for c in _config_mismatch(_rec(4), _rec(4, wpc=(4, 1)))))
    ck("_config_mismatch: threads_per_warp difference is named",
       any("threads_per_warp" in c for c in _config_mismatch(_rec(4), _rec(4, tpw=32))))

    # --- live layer: only if Triton imports. A box without it still passes above.
    try:
        import triton  # noqa: F401
        have_triton = True
    except Exception as e:  # noqa: BLE001
        have_triton = False
        print(f"  skip live layer: triton not importable ({type(e).__name__})")

    if have_triton:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "selftest.ttgir")
            with open(p, "w") as f:
                f.write(_SELFTEST_TTGIR)
            try:
                rec = recover(p, "gfx942")
            except SystemExit as e:
                print(f"  skip live layer: {e}")
                rec = None
            if rec is not None:
                ck("live: module attrs read", rec.num_warps == 4 and rec.threads_per_warp == 64,
                   f"{rec.num_warps}/{rec.threads_per_warp}")
                ck("live: every site recovered (no UNRECOVERABLE)",
                   not rec.failed_sites, str([s.enc_kind for s in rec.failed_sites]))
                kinds = {s.enc_kind for s in rec.ok_sites}
                ck("live: found blocked + amd_mfma + swizzled_shared + dot_op",
                   {"blocked", "amd_mfma", "swizzled_shared", "dot_op"} <= kinds, str(sorted(kinds)))
                ck("live: every layout round-trips EXACT",
                   all(s.roundtrip == "EXACT" for s in rec.ok_sites),
                   str([(s.enc_kind, s.roundtrip) for s in rec.ok_sites
                        if s.roundtrip != "EXACT"][:3]))
                ck("live: num_warps cross-check passes", rec.warp_check.startswith("PASS"),
                   rec.warp_check)
                roles = {s.role for s in rec.ok_sites}
                ck("live: A/B dot operands labelled from opIdx",
                   {"A_DOT_OPERAND", "B_DOT_OPERAND"} <= roles, str(sorted(roles)))
                src = emit_layouts(rec)
                ck("live: emitted file is valid python", _compiles(src))
                # Assignment form, not annotated: Triton rejects `NAME: gl.constexpr = N`
                # for a kernel-visible global scalar, which made the file the runbook
                # tells you to use at the launch site unusable as written.
                ck("live: NUM_WARPS is emitted in assignment form",
                   "\nNUM_WARPS = 4" in src and "NUM_WARPS: gl.constexpr" not in src)
                ck("live: layout constants keep the constexpr annotation",
                   ": gl.constexpr = gl." in src)
                # The rank guard that was missing from view(), on the exact shape class
                # that aborts: a rank-2 shared layout at a rank-3 (multi-buffered) site.
                sm = [s for s in rec.ok_sites if s.space == "smem"]
                if sm:
                    fake = Site(op="ttg.local_alloc", kind="result", index=0,
                                shape=[2] + list(sm[0].shape), dtype=sm[0].dtype,
                                space="smem", enc_kind=sm[0].enc_kind,
                                enc_text=sm[0].enc_text, layout=sm[0].layout,
                                role="A_SMEM")
                    r2 = Recovery(path=rec.path, arch=rec.arch, num_warps=rec.num_warps,
                                  threads_per_warp=rec.threads_per_warp)
                    r2.sites = [fake]
                    ok2, t2 = view(r2, "A_SMEM", False, 4, 80)
                    # The guard must hold on every build; the "rendered at" note only
                    # appears where there is something to render. Asserting the note
                    # unconditionally failed on 3.6, which has no get_layout_view at all --
                    # i.e. the test would have been red on the one build whose degradation
                    # the tool goes out of its way to handle.
                    ck("live: view survives a rank-3 site with a rank-2 shared layout",
                       ok2 and ("rendered at" in t2 if capabilities()["layout_view"]
                                else "no view" in t2), t2[:140])
                ck("live: emitted file warns that declaring != applying",
                   "Declaring a layout is not applying it" in src)
                st, rep, vdata = verify(p, p, "gfx942")
                ck("live: verify(x, x) is PASS", st == "PASS", rep.splitlines()[0])
                ck("live: verify reports multiplicity as informational",
                   "never gates" in rep)
                ck("live: verify returns machine-readable arrays, not just prose",
                   all(kk in vdata for kk in ("missing", "extra", "reconciled",
                                              "multiplicity", "counts", "unrecoverable")),
                   str(sorted(vdata)))
                ck("live: a PASS has empty missing/extra arrays",
                   vdata["missing"] == [] and vdata["extra"] == [])

                # RECONCILED, both rules, on hand-built Recovery pairs so the structural
                # cause is isolated from any real kernel's noise.
                def _mk(sites, failed=()):
                    r = Recovery(path="x", arch="gfx942", num_warps=4, threads_per_warp=64)
                    r.sites = list(sites) + list(failed)
                    return r

                base = [s for s in rec.ok_sites if s.space == "reg"][:1]
                if base:
                    b0 = base[0]

                    def _site(shape, layout, role="X", err=None):
                        return Site(op="tt.dot", kind="result", index=0, shape=list(shape),
                                    dtype=b0.dtype, space="reg", enc_kind=b0.enc_kind,
                                    enc_text=b0.enc_text, layout=None if err else layout,
                                    error=err, role=role)

                    # R2: same normal form present at another shape, EXTRA empty.
                    pl = _mk([_site(b0.shape, b0.layout), _site([s // 2 for s in b0.shape],
                                                                b0.layout)])
                    an = _mk([_site(b0.shape, b0.layout)])
                    st_r2, _rep_r2, d_r2 = _verify_recs(pl, an, "gfx942")
                    ck("live: RECONCILED when a MISSING form exists at another shape",
                       st_r2 == "RECONCILED", f"{st_r2}: {d_r2['counts']}")
                    ck("live: that reconciliation names its reason",
                       any(x["reason"] == "same_normal_form_other_shape"
                           for x in d_r2["reconciled"]))

                    # R1: anchor supplies a layout where plain's was UNRECOVERABLE.
                    pl2 = _mk([], failed=[_site(b0.shape, None, err="ValueError: unhandled")])
                    an2 = _mk([_site(b0.shape, b0.layout)])
                    st_r1, _r, d_r1 = _verify_recs(pl2, an2, "gfx942")
                    ck("live: RECONCILED when EXTRA lands on an UNRECOVERABLE shape",
                       st_r1 == "RECONCILED", f"{st_r1}: {d_r1['counts']}")
                    ck("live: that reconciliation is labelled probable, not proven",
                       "probable" in _r)

                    # R3: MISSING because the op has no Gluon builtin. Two trial kernels
                    # were graded FAIL on bit-exact anchors for exactly this row.
                    def _op_site(shape, layout, op):
                        return Site(op=op, kind="result", index=0, shape=list(shape),
                                    dtype=b0.dtype, space="reg", enc_kind=b0.enc_kind,
                                    enc_text=b0.enc_text, layout=layout, error=None,
                                    role="AMDG_IN_THREAD_TRANSPOSE")

                    # The R3 site must carry a layout the anchor does NOT have, otherwise
                    # R2 ("same constructor at another shape") claims the row first and
                    # this case never exercises R3.
                    no_op = min(NO_GLUON_OP)
                    distinct = [s for s in rec.ok_sites
                                if s.space == "reg" and repr(s.layout) != repr(b0.layout)]
                    r3_layout = distinct[0].layout if distinct else b0.layout
                    pl4 = _mk([_site(b0.shape, b0.layout),
                               _op_site([s * 2 for s in b0.shape], r3_layout, no_op)])
                    an4 = _mk([_site(b0.shape, b0.layout)])
                    st4, rep4, d4 = _verify_recs(pl4, an4, "gfx942")
                    ck("live: RECONCILED when MISSING is owned by an op Gluon lacks",
                       st4 == "RECONCILED", f"{st4}: {d4['counts']}")
                    ck("live: R3 names the offending op, not a generic cause",
                       any(x["reason"] == "op_has_no_gluon_equivalent" and no_op in x.get("ops", [])
                           for x in d4["reconciled"]), str(d4["reconciled"])[:200])
                    ck("live: R3 does not silently promote to PASS",
                       "RECONCILED" in rep4 and no_op in rep4)

                    # A genuine difference must still FAIL.
                    other = [s for s in rec.ok_sites
                             if s.space == "reg" and repr(s.layout) != repr(b0.layout)]
                    if other:
                        pl3 = _mk([_site(b0.shape, b0.layout)])
                        an3 = _mk([_site(b0.shape, other[0].layout)])
                        st3, _r3, _d3 = _verify_recs(pl3, an3, "gfx942")
                        ck("live: a real layout difference still FAILs", st3 == "FAIL", st3)
                ck("live: MMA signature is extracted for the config precheck",
                   _mma_signature(rec) == {((2, 2), (16, 16, 16))}, str(_mma_signature(rec)))

                # --- the four defects the first trial kernel found. Each is registered
                # here because each was invisible to every check that existed before it.
                #
                # (a) recover and view must resolve a name to the SAME layout. They used
                # to resolve independently and disagreed on a kernel where one layout
                # served both a load and a store, handing over a rank-mismatched layout.
                named = assign_names(rec)
                ck("live: symbols are unique", len({n for n, _k, _s in named}) == len(named))
                for nm, key, _ss in named:
                    found, txt = view(rec, nm, False, 4, 80)
                    if not found or key[:60] not in txt:
                        ck(f"live: view({nm}) resolves to the emitted layout", False,
                           txt.splitlines()[0] if txt else "")
                        break
                else:
                    ck("live: view resolves every emitted symbol to ITS layout", True)
                ck("live: view reports not-found instead of guessing",
                   view(rec, "NO_SUCH_ROLE", False)[0] is False)

                # (b) byte-reproducibility. The runbook uses identical output across
                # Triton versions as its evidence that a constant is the compiler's; an
                # emitted file that reorders its own comments defeats that on one machine.
                ck("live: emit_layouts is byte-reproducible",
                   emit_layouts(rec) == emit_layouts(recover(p, "gfx942")))

                # (c) the view must not dump unbounded text.
                big = _clip("\n".join("x" * 5000 for _ in range(200)), 4, 80)
                ck("live: _clip bounds rows and columns",
                   len(big.splitlines()) == 5 and max(len(l) for l in big.splitlines()) < 130,
                   f"{len(big.splitlines())} rows")
                ck("live: _clip says what it elided", "elided" in big and "truncated" in big)
                ck("live: _clip(0) means no limit",
                   len(_clip("a\nb\nc", 0, 0).splitlines()) == 3)
                # The capability probe has to survive a build where the thing it probes
                # ABORTS. If this line is reached at all on Triton 3.6, it survived.
                caps = capabilities()
                ck("live: capability probe returned without killing us",
                   caps["probed"] in ("subprocess", "env"), str(caps))
                ck("live: distributed normalisation is available on every build",
                   caps["distributed_norm"], str(caps))
                ck("live: the verdict names its equivalence basis",
                   "basis:" in rep and ("normal form" in rep or "canonical text" in rep))
                _, b2 = make_context("gfx942")
                shared = [s for s in rec.ok_sites if s.space == "smem"]
                ck("live: a shared layout yields SOME comparable form",
                   bool(shared) and _normal_form(b2, shared[0].layout, shared[0].shape,
                                                 "smem") is not None,
                   f"caps={caps}")

    print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
          + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
    return 1 if fails else 0


def _compiles(src: str) -> bool:
    try:
        compile(src, "<emitted>", "exec")
        return True
    except SyntaxError:
        return False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="offline smoke; no GPU")
    sub = ap.add_subparsers(dest="mode")

    r = sub.add_parser("recover", help="ttgir -> Gluon layout constants + JSON facts")
    r.add_argument("--ttgir", required=True)
    r.add_argument("--arch", default="gfx942")
    r.add_argument("--warp-size", type=int)
    r.add_argument("--out", help="write the layout module here (default: stdout report only)")
    r.add_argument("--json", dest="json_out", help="write machine-readable facts here")
    r.add_argument("-v", "--verbose", action="store_true", help="list every use site")
    r.add_argument("--force", action="store_true",
                   help="overwrite an --out recovered from a different dump")
    r.add_argument("--allow-unrecoverable", action="store_true",
                   help="exit 0 even when a layout has no Gluon constructor")

    v = sub.add_parser("verify", help="plain vs anchor, as LinearLayout normal forms")
    v.add_argument("--plain", required=True)
    v.add_argument("--anchor", required=True)
    v.add_argument("--arch", default="gfx942")
    v.add_argument("--warp-size", type=int)
    v.add_argument("--json", dest="json_out", help="write the verdict here")

    w = sub.add_parser("view", help="ASCII per-lane view of one recovered layout")
    w.add_argument("--ttgir", required=True)
    w.add_argument("--role", required=True)
    w.add_argument("--arch", default="gfx942")
    w.add_argument("--warp-size", type=int)
    w.add_argument("--hardware", action="store_true", help="hardware view instead of tensor view")
    w.add_argument("--max-rows", type=int, default=24, help="0 = no limit")
    w.add_argument("--max-cols", type=int, default=200, help="0 = no limit")

    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.mode is None:
        ap.print_help()
        raise SystemExit(2)

    if a.mode == "recover":
        rec = recover(a.ttgir, a.arch, a.warp_size)
        print(report(rec, a.verbose))
        if a.out:
            # Refuse to overwrite a file recovered from a DIFFERENT dump. A ladder champion
            # has several bodies, and `recover --out one/path` per body silently leaves only
            # the last -- which is how a shipped layouts.json ended up describing one rung
            # while being read as if it described all three.
            if os.path.exists(a.out) and not a.force:
                prev = ""
                try:
                    with open(a.out, errors="replace") as _f:
                        for ln in _f:
                            if ln.startswith("# recovered-from:"):
                                prev = ln.split(":", 1)[1].strip()
                                break
                except OSError:
                    # Unreadable or absent previous file -> nothing to guard against, and
                    # refusing to write because the GUARD failed would be worse than the
                    # overwrite it exists to prevent.
                    prev = ""
                if prev and os.path.abspath(prev) != os.path.abspath(a.ttgir):
                    raise SystemExit(
                        f"[ttgir_bridge] refusing to overwrite {a.out}\n"
                        f"  it was recovered from a DIFFERENT dump: {prev}\n"
                        f"  now recovering:                        {a.ttgir}\n"
                        f"  A ladder champion has one body per rung and each needs its own file --"
                        f" writing them to one path leaves only the last, and the survivor then"
                        f" reads as if it described every rung. Use a per-rung --out, or --force"
                        f" if you really mean to replace it.")
            with open(a.out, "w") as f:
                f.write(emit_layouts(rec))
            print(f"\nwrote layout module -> {a.out}")
        if a.json_out:
            with open(a.json_out, "w") as f:
                json.dump({"source": rec.path, "arch": rec.arch,
                           "num_warps": rec.num_warps,
                           "threads_per_warp": rec.threads_per_warp,
                           "num_warps_check": rec.warp_check,
                           "sites": [s.as_json() for s in rec.sites]}, f, indent=2)
            print(f"wrote facts -> {a.json_out}")
        # Distinct codes, because the two failures call for opposite responses. An
        # UNRECOVERABLE layout is a LANGUAGE gap: the recovery is sound, part of the kernel
        # just is not expressible, and a caller may legitimately proceed on the rest. A
        # num_warps cross-check FAIL means the dump contradicts itself, so nothing in it is
        # trustworthy and a caller must stop. One exit code for both made a script unable to
        # tell "partly transcribable" from "throw this dump away".
        bad = rec.failed_sites and not a.allow_unrecoverable
        if rec.warp_check.startswith("FAIL"):
            print("\n[ttgir_bridge] exit 4: num_warps cross-check FAILED -- the dump is "
                  "internally inconsistent, so NO layout in it is trustworthy.")
            raise SystemExit(4)
        raise SystemExit(1 if bad else 0)

    if a.mode == "verify":
        status, rep, data = verify(a.plain, a.anchor, a.arch, a.warp_size)
        print(rep)
        if a.json_out:
            with open(a.json_out, "w") as f:
                json.dump(dict(data, plain=a.plain, anchor=a.anchor, arch=a.arch,
                               report=rep), f, indent=2)
            print(f"\nwrote verdict -> {a.json_out}")
        raise SystemExit({"PASS": 0, "RECONCILED": 0, "FAIL": 1, "NOT_COMPARABLE": 3}[status])

    if a.mode == "view":
        rec = recover(a.ttgir, a.arch, a.warp_size, roundtrip=False)
        found, text = view(rec, a.role, a.hardware, a.max_rows, a.max_cols)
        print(text)
        # A missing role used to print a friendly message and exit 0, so no script could
        # tell a rendered layout from a typo.
        raise SystemExit(0 if found else 2)


if __name__ == "__main__":
    main()
