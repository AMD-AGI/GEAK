#!/usr/bin/env python3
"""Extract source evidence from AITER's six GEMM implementations, with file:line provenance.

    # from the repo root, with an aiter checkout to read
    python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /sgl-workspace/aiter --emit
    python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /sgl-workspace/aiter --json
    python3 perf_knowledge/corpus/_render_facts.py --emit             # evidence YAML -> source index
    python3 perf_knowledge/corpus/_render_decisions.py --emit         # cards + evidence -> advice

WHAT PROBLEM THIS SOLVES
GEAK has to author FlyDSL, and the FlyDSL corpus is thin: 33 files against Triton's 318. The
knowledge needed to close that gap is already on disk — the same operators are implemented six ways
in AITER, and every one of those files is a record of somebody's decision about tiling, LDS staging,
layout and which MFMA to issue. What was missing is a way to LOOK ONE UP. `kernel_families.md` is
the hand-written attempt and it shows the limits: file-level pointers, no line numbers, and claims
like "default tile 128x128x64" that no longer have anything checking them.

So this walks the implementations and emits one record per source observation, each carrying a
content-bound identity, `file:line` and the source lines themselves.

WHAT IT DOES NOT DO — and this is the whole safety argument
It records what the source SAYS. It never says why a choice is faster, never ranks two
implementations, and never turns a constant into a recommendation. Those are claims, claims need
measurement, and nothing here measured anything: this is a citation index over code that already
exists. The docs that DO make claims (`languages/flydsl/*.md`) become checkable against it, which is
the actual win — a reader can go from "FlyDSL picks a different MFMA on gfx950" to the four lines
that do it.

The evidence excerpt is copied verbatim for the same reason. A paraphrase of a tile derivation is a
new artifact that can be wrong; the lines themselves can only be stale, and staleness is detectable
because every record carries the commit it was read at.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
EVIDENCE = os.path.join(HERE, "evidence")

# The GEMM family, as the pilot. Each entry is (language, subtree, glob) — the subtree is where that
# language's implementations live in aiter, and languages are kept apart because "which language is
# this" must not be inferred here: `kernel_workflow/scripts/detect_language.py` exists for source
# whose language is unknown, and a corpus keyed on a guess is worse than a smaller correct one.
IMPLS = [
    # `gemm_tune/…_common.py` is where the A8 (fp8/int8) preshuffle family states its search space —
    # per-arch tile catalogs, the lds_stage x cshuffle x async x waves_per_eu sweep and the register
    # cap on waves_per_eu. Without it the corpus could cite that family's kernel but not the set of
    # configurations AITER is willing to build for it. `utils.py` carries the per-arch LDS budget
    # every tiling estimate is compared against.
    ("flydsl", "aiter/ops/flydsl", ("kernels/splitk_hgemm.py", "kernels/small_m_hgemm.py",
                                   "kernels/preshuffle_gemm.py",
                                   "kernels/mfma_preshuffle_pipeline.py",
                                   "kernels/moe_gemm_2stage.py",
                                   "kernels/mixed_moe_gemm_2stage.py",
                                   "kernels/hgemm_dispatch.py", "kernels/mfma_epilogues.py",
                                   "kernels/layout_utils.py", "gemm_kernels.py",
                                   "gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py",
                                   "utils.py", "test_flydsl_splitk_hgemm.py")),
    ("triton", "aiter/ops/triton/_triton_kernels/gemm", ("basic/*.py", "batched/*.py", "fused/*.py")),
    # The GEMM kernels above call `remap_xcd` / `pid_grid` but do not define them; the definitions —
    # which are the actual workgroup-to-XCD and grouped-M rasterization — live in a shared helper.
    ("triton", "aiter/ops/triton/utils/_triton", ("pid_preprocessing.py",)),
    # How a shape reaches a shipped Triton config: a per-(N, K) file if one exists, else the generic
    # file, then M_LEQ/M_GEQ buckets, then `any`. An untuned shape gets the generic `any` entry at
    # every M, which is where a frozen Triton baseline for such a task actually sits.
    ("triton", "aiter/ops/triton/utils", ("gemm_config_utils.py",)),
    ("gluon", "aiter/ops/triton", ("_gluon_kernels/*/moe/*gemm*.py", "_gluon_kernels/*/gemm/*.py",
                                   "gluon/*gemm*.py")),
    # HIP and CK both live under `csrc/` and both are `.cu`/`.cuh`, so the split is by DIRECTORY,
    # not by extension: `ck_*` subtrees are CK template instantiations, `py_itfs_cu/` and `kernels/`
    # are hand-written HIP. Getting this backwards is the mis-filing `detect_language.py` guards
    # against on the source side — a CK instance read as HIP teaches a FlyDSL author to look for a
    # hand-written loop where there is a template parameter list.
    # Hand-written HIP GEMM in aiter is the `opus_gemm` tree. The kernels themselves are the per-arch
    # `include/gfx*/opus_gemm_pipeline_*.cuh` and `..._traits_*.cuh` — the top-level `.cu` is only a
    # dispatcher — and the codegen scripts hold the tile tables the instances are stamped from. All
    # three layers are wanted: the pipeline says how, the traits say with what, the codegen says which
    # combinations were thought worth building.
    # `kernels/custom_kernels.cu` is AITER's hand-written skinny GEMM/GEMV family (`wvSplitK*`,
    # `HGEMV_WFPerRow`) — the default bf16/fp16 route for M <= 16 — and `include/opus/opus.hpp` is
    # the single-header HIP micro-DSL (buffer load/store, MFMA wrappers, layouts) AITER's HIP kernels
    # are written in. The `opus_gemm/` tree lives on AITER branches newer than the pinned commit, so
    # its patterns report as `missing` until an AITER bump brings it in.
    ("hip", "csrc", ("py_itfs_cu/gemm_common.cu", "include/gemm_common.h",
                     "include/gemm_dispatch_utils.h", "kernels/custom_kernels.cu",
                     "include/opus/opus.hpp", "opus_gemm/*.cu",
                     "opus_gemm/include/*.cuh", "opus_gemm/include/*.h",
                     "opus_gemm/include/gfx*/opus_gemm_pipeline_*.cuh",
                     "opus_gemm/include/gfx*/opus_gemm_traits_*.cuh",
                     "opus_gemm/codegen/gen_instances_*.py", "opus_gemm/opus_gemm_common.py")),
    ("ck", "csrc", ("ck_gemm*/*.cu", "ck_gemm*/include/*.cuh", "ck_gemm*/include/*.h",
                    "ck_batched_gemm*/*.cu", "ck_batched_gemm*/include/*.cuh",
                    "cktile_gemm*/**/*.cu", "cktile_gemm*/**/*.cuh",
                    "ck_tile_gemm_moe_2stages/**/*.cu", "ck_tile_gemm_moe_2stages/**/*.cuh",
                    "ck_deepgemm/**/*.cu", "ck_deepgemm/**/*.cuh")),
    # Hand-written assembly is not readable from `hsa/**/*.co` by a text pass, and the binary itself
    # is not where the decisions are stated anyway. What IS readable is the launcher: `asm_gemm_*.cu`
    # names the `.co` to load, the block/grid it is launched with, and the argument block layout the
    # assembly expects. That is the asm contract, and it is the part a FlyDSL author has to match.
    # The instructions inside the binary are a separate, deeper layer. They require a future
    # disassembly evidence pass; this source pass does not pretend that file:line can describe a
    # binary instruction.
    ("asm", "csrc/py_itfs_cu", ("asm_gemm_*.cu", "asm_flatmm_*.cu",
                                "asm_a8w8_blockscale_bpreshuffle.cu")),
    # The binaries publish no source, but each `hsa/<gfx>/<kind>/*.csv` states which kernels exist
    # for that architecture: tile, B-preshuffle and split-K capability. One record per row, per gfx.
    ("asm", "hsa", ("gfx*/*gemm*/*.csv",)),
]

# AITER's runtime dispatch — which backend a shape is routed to by default. Dispatch is library code,
# not a kernel language, so each routing statement is filed under the language of the backend it
# routes TO; library targets (`hipblaslt`, `torch`) are not languages and are not recorded.
ROUTING_FILES = ("aiter/tuned_gemm.py", "aiter/ops/gemm_op_a8w8.py", "aiter/ops/gemm_op_a16w16.py",
                 "aiter/ops/gemm_op_a4w4.py")
ROUTING_RULES = (
    r"\[\s*['\"]libtype['\"]\s*\]\s*=\s*['\"](\w+)['\"]",
    r"\blibtype['\"]?\s*\]?\s*==\s*['\"](\w+)['\"]",
    r"\breturn\s+(\w+)\s*\(",
    r"\bdef\s+(is_\w+_shape)\s*\(",
)
# Matched against `_`-separated name tokens, first hit wins: `blockscale_ck` is CK, `block` is not.
ROUTE_TARGETS = (("cktile", "ck"), ("skinny", "hip"), ("flydsl", "flydsl"), ("triton", "triton"),
                 ("gluon", "gluon"), ("asm", "asm"), ("ck", "ck"))

# Per-shape selected knob sets that AITER ships for its Triton GEMMs: one JSON per
# `{gfx}-{KERNEL}-{VARIANT}[-N=..-K=..]`, each mapping an M bucket to a concrete config. The source
# tree preserves neither the benchmark archive nor rejected alternatives, so these are useful seeds,
# not measured winners.
TUNED_CONFIG_DIR = "aiter/ops/triton/configs/gemm"


# ---------------------------------------------------------------------------------------------
# Fact rules. Each is a category plus a matcher over a stripped-comment view of the file.
#
# Regex rather than a per-language parser, deliberately and with its limit stated: six languages
# means six grammars, and a Python AST pass over Triton would still not read a `.cu`. What matters
# for a citation index is that a hit points at the right LINE, not that the construct is fully
# parsed — the reader opens the file. Where structure is genuinely needed (Triton's autotune config
# lists) there is an AST pass below, applied only to the .py files where it is valid.
#
# Comments are stripped before matching. `detect_language.py` learned this the expensive way: five
# plainly-HIP files were classified CK on the strength of `ck_tile::` mentions that appeared only in
# commented-out code. A corpus that cites a commented-out tile size is worse than one that misses it.
# ---------------------------------------------------------------------------------------------
RULES = [
    # --- which matrix instruction, and under what condition -----------------------------------
    ("mfma_intrinsic", r"\brocdl\.(mfma_[a-z0-9_]+)\s*\("),
    ("mfma_intrinsic", r"\b(v_mfma_[a-z0-9_]+)\b"),
    ("mfma_intrinsic", r"\b(?:tl|ttgl)\.dot\w*\s*\("),
    ("mfma_shape", r"\b(?:WMMA|MFMA|INSTR)_(?:M|N|K)\s*=\s*(\d+)"),
    ("mfma_shape", r"\bmatrix_instr_nonkdim\s*[=:]\s*(\d+)"),
    ("mfma_shape", r"\binstr_shape\s*[=:]\s*[\(\[]([^)\]]*)[\)\]]"),
    ("mfma_shape", r"\bm16n16k(\d+)\b"),

    # --- tiling ---------------------------------------------------------------------------------
    ("tile_shape", r"\b(BLOCK_SIZE_[MNK]|BLOCK_[MNK]|TILE_[MNK]|tile_[mnk])\s*[=:]\s*([0-9]+)"),
    ("tile_shape", r"\b(BM|BN|BK)\s*=\s*([0-9]+)"),
    ("waves", r"\b(BLOCK_[MNK]_WARPS|num_warps|NUM_WARPS|WARP_SIZE|waves_per_eu)\s*[=:]\s*([0-9]+)"),
    ("split_k", r"\b(SPLIT_K|split_k|SPLITK|NUM_KSPLIT)\s*[=:]\s*([0-9]+)"),
    ("split_k", r"\b(IS_SPLIT_K|IS_SLICE_K)\s*=\s*"),
    ("persistent", r"\b(persistent|PERSISTENT|NUM_XCDS|num_xcds)\w*\s*[=:]\s*"),

    # --- the shipped search space, as opposed to one point in it -----------------------------------
    # The rules above capture a tile VALUE: `TILE_M = 128` in some file. That answers "what did this
    # kernel use", which is the wrong question for somebody choosing a configuration — they need the
    # set the library is willing to compile, and the derivation that narrows it for a shape. AITER
    # states both, as module-level option tuples and as the `*_options(...)` helpers that filter them.
    # Missing this was a real gap and not a subtle one: a reader of the corpus could learn that the
    # default tile_m is 128 without learning that tile_m 16 exists, which is precisely the knob a
    # skinny-M shape needs. A single value read as the whole space is worse than no value.
    # Anchored to column 0: a module-level constant is the library's declared space, an indented
    # ALL-CAPS tuple is a local intermediate inside somebody's kernel body. The first is a fact about
    # what can be configured; the second is arithmetic, and mixing them makes the space look larger
    # and vaguer than it is.
    ("config_space", r"^([A-Z][A-Z0-9_]{2,})\s*=\s*[\(\[]"),
    ("config_space", r"\bdef\s+(\w*_options|iter_\w+_configs?|\w*_registry_variants)\s*\("),
    # Caps and floors: the boundary of the space, which is a different fact from its members. Written
    # as `HGEMM_MAX_SPLIT_K`, `MAX_LDS_BYTES`, `SMALL_M_KERNEL_MAX` — prefix optional, suffix optional.
    ("config_limit", r"\b((?:[A-Z][A-Z0-9_]*_)?(?:MAX|MIN)(?:_[A-Z0-9_]+)?)\s*=\s*([0-9]+)"),
    # What makes a configuration legal at all: LDS-budget estimators and the validators that reject a
    # tiling before it reaches the compiler. An optimizer that cannot tell "slow" from "will not
    # build" spends its budget on the second.
    ("config_validity", r"\bdef\s+(\w*(?:validate|_check_)\w*)\s*\("),
    ("config_validity", r"\bdef\s+(\w*estimate\w*|\w*lds_bytes\w*)\s*\("),
    # Which kernel FAMILY a shape is routed to. Named constants rather than a comment, and the
    # families do not share a config surface — a knob that exists in one is absent from the other.
    ("kernel_family", r"\b(KERNEL_FAMILY_[A-Z0-9_]+)\s*=\s*['\"](\w+)['\"]"),
    # The comparison sites only — those are the routing decision. The dozen places that merely pass
    # `kernel_family=` through say the same thing a dozen times.
    ("kernel_family", r"\bkernel_family\s+in\s+[\(\[]|\bkernel_family\s*==\s*(\w+)"),
    # Architecture gates. Whether a code path exists on the box you are on outranks every tuning fact
    # about it, and the corpus was silent on this: the small-M family raises outright on gfx942, and
    # nothing in the extracted evidence said so. A card that recommends a path the target architecture
    # refuses to compile does not cost a little accuracy, it costs a whole round. Match the comparison
    # sites against an arch name, not every mention of one.
    ("arch_gate", r"(?:gpu_arch|GPU_ARCH|arch|target_gfx|get_gfx\(\)|get_rocm_arch\(\))\s*[=!]=\s*['\"](gfx[0-9a-z]+)['\"]"),
    ("arch_gate", r"['\"](gfx[0-9a-z]+)['\"]\s*(?:not\s+)?in\s+|\bin\s+\{?\(?\s*['\"](gfx[0-9a-z]+)['\"]"),

    # --- LDS ------------------------------------------------------------------------------------
    ("lds_stage", r"\b(STAGES|lds_stage|LDS_STAGE|num_stages|NUM_STAGES)\s*[=:]\s*([0-9]+)"),
    ("lds_swizzle", r"\bdef\s+(swizzle_\w+)\s*\("),
    ("lds_swizzle", r"\b(swizzle_xor\w*|xor_swizzle|XOR_SWIZZLE)\b"),
    # `padded_m = 16` is the dispatcher's M-bucket granularity for a tuned-config lookup, not an LDS
    # row pad; it used to be the only "lds_pad" CK and HIP had, which read as CK padding its LDS.
    # It is recorded under `dispatch_padding` below instead.
    ("lds_pad", r"\b(?!padded_[mnk]\b)(\w*(?:pad|PAD)\w*)\s*[=:]\s*([0-9]+)"),
    ("lds_alloc", r"\b(?:fx\.)?(?:alloc_shared|shared_alloc|allocate_shared|smem_alloc)\s*\("),
    ("lds_alloc", r"\b__shared__\b"),

    # --- global memory / async copy ---------------------------------------------------------------
    ("async_copy", r"\b(ASYNC_COPY|use_async_copy|async_copy)\s*[=:]\s*"),
    ("access_width", r"\b(DMA_BYTES|LDG_VEC_SIZE|DTYPE_BYTES|VEC_SIZE)\s*=\s*([0-9]+)"),
    ("cache_policy", r"\b(?:cache_modifier|CACHE_MODIFIER)\s*[=:]\s*['\"]?(\.?[a-z]+)"),
    ("cache_policy", r"\b(?:nt|sc0|sc1|nontemporal|GLC|SLC)\s*=\s*True"),

    # --- layout ------------------------------------------------------------------------------------
    ("layout_preshuffle", r"\b(b_preshuffle|preshuffle|BPRESHUFFLE|shuffle_weight)\w*\s*[=:(]"),
    ("layout_transpose", r"\b(?:trans_?[ab]|TRANS_[AB]|transpose_[ab])\s*[=:]\s*"),
    ("layout_epilogue", r"\b(use_cshuffle_epilog|cshuffle|CShuffle|c_shuffle)\w*\s*[=:(]"),

    # --- scheduling ---------------------------------------------------------------------------------
    ("scheduling", r"\brocdl\.(sched_\w+|s_setprio|iglp_opt)\s*\("),
    ("scheduling", r"\b(?:tl|ttgl)\.(?:async_wait|barrier|debug_barrier)\s*\("),

    # --- the asm contract, as stated by the launcher ------------------------------------------------
    # A hand-written assembly kernel publishes no source. What it publishes is the shape of the call:
    # the object to load, the workgroup it assumes, the grid formula its block-id decode expects, and
    # the byte layout of its argument block. A FlyDSL author reimplementing one of these has to match
    # all four, and every one of them is written down here rather than in the binary.
    ("asm_object", r"\b(co_name|CO_NAME|hsaco|\w+\.co)\b"),
    ("asm_launch", r"^\s*(?:\w[\w:<>*&\s]*?\s)?blockSize[XYZ]?\s*=\s*([0-9]+)\s*;"),
    # Anchored to the start of a statement, and the right-hand side must name something. Both
    # restrictions come from real misses: unanchored, `gdx=` matched inside a
    # `printf("gdx=%d, gdy=%d")` format string; and without the letter requirement the `int gdx = 0;`
    # declarations were reported alongside the real formula, three records of `0` per launcher saying
    # nothing. A grid formula that mentions no dimension is not a grid formula.
    ("asm_launch", r"^\s*(?:int\s+)?gd[xyz]\s*=\s*([^;\"]*[A-Za-z_][^;\"]*);"),
    ("asm_launch", r"\bhipModuleLaunchKernel\s*\("),
    ("asm_argblock", r"struct\s+__attribute__\(\(packed\)\)\s+(\w+)"),
    ("asm_tile", r"\bSUB[MNK]\s*=\s*(.+?);"),
    ("asm_tile", r"\bcfg\.(tile_[mnk]|splitK)\b"),

    # --- decisions encoded in a name --------------------------------------------------------------
    # aiter's hand-written HIP GEMM stamps its instances from a naming scheme that carries the whole
    # configuration: `opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128` is block tile, then the warp
    # grid, then the MFMA shape, then the scale granularity. Matching the name is not a shortcut for
    # reading the template — it is the only place the four are stated together, and it makes the set
    # of combinations somebody chose to build enumerable.
    ("instance_name", r"\b(opus_gemm_\d+x\d+x\d+(?:x\d+)?_\d+x\d+_\d+x\d+x\d+(?:_[0-9x]+)?)\b"),
    # The pipeline variants are the HIP-side design axes, named: persistent vs not, mono-tile,
    # flat-mm, split-k, and the 4GB-safe addressing sibling that exists because a 32-bit offset
    # overflows on large tensors.
    ("pipeline_variant", r"opus_gemm_(?:pipeline|traits)_(\w+?)_gfx\d+"),
    ("pipeline_variant", r"\b(is_4g_safe|_4g_safe)\b"),
    # CK states its configuration as a positional template argument list, and the corpus does not try
    # to decode it — 40-odd unnamed positions are exactly the kind of thing a regex would get subtly
    # wrong. What it records instead is the instance template chosen and the two arguments CK does
    # name, the loop scheduler and the pipeline version, because those are the decisions with an
    # analogue an author can act on: interwave vs intrawave is a scheduling choice a FlyDSL kernel
    # also has to make by hand.
    ("ck_instance", r"\b(DeviceGemm\w*|DeviceBatchedGemm\w*|GemmPipelineAgBgCr\w*)\s*<?"),
    ("ck_pipeline", r"BlockGemmPipeline(?:Scheduler|Version)::(\w+)"),
    ("ck_pipeline", r"\b(GemmSpecialization::\w+)"),

    # --- the tunable surface, as declared ---------------------------------------------------------
    # Which knobs a kernel even has. In Triton this is the `tl.constexpr` parameter list, and it is
    # worth its own category because it answers a different question from a tile VALUE: the values
    # for these live in the shipped JSON, so the source can only tell you what is adjustable. An
    # author porting to FlyDSL needs the list before the numbers.
    ("tunable_param", r"^\s*(\w+)\s*:\s*tl\.constexpr\s*,?\s*$"),

    # --- added with the A8 / epilogue / split-K / XCD cards -----------------------------------------
    # Every rule below is additive: none of them can match a construct an earlier rule already
    # records, so adding them leaves every existing `src_…` identity — and therefore every card that
    # cites one — unchanged.
    #
    # An MFMA chosen as a function object rather than called in place. The A8 kernel selects
    # `mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8` per dtype and calls `mfma_fn(...)` later, so the
    # call-site rule above saw only the scaled-MFMA branch and none of the fp8/int8/f16/bf16 choices.
    ("mfma_intrinsic", r"\brocdl\.(mfma_[a-z0-9_]+)\b(?!\s*\()"),
    ("mfma_intrinsic", r"\bgetattr\(\s*rocdl\s*,\s*['\"](mfma_[a-z0-9_]+)['\"]"),
    # Arch gates spelled through `get_hip_arch()` or a prefix test. The A8 family gates its gfx950
    # preload table and its FP4 path this way, and the `==` rule above cannot see either.
    ("arch_gate", r"\bget_hip_arch\(\)\)?\s*[=!]=\s*['\"](gfx[0-9a-z]+)['\"]"),
    ("arch_gate", r"\.startswith\(\s*['\"](gfx[0-9a-z]+)['\"]\s*\)"),
    # The declared space, in the two spellings the upper-case rule misses: private module constants
    # (`_LDS_STAGES = (1, 2)`, `_TILE_PRELOAD_TABLE = {`) and lower-case tile catalogs / per-arch
    # kernel lists (`_base_tiles_lds2_common = [`, `kernels_list_942 = …`, `default_kernels_dict_950`).
    ("config_space", r"^(_[A-Z][A-Z0-9_]{2,})\s*=\s*[\(\[\{]"),
    ("config_space", r"^(\w*tiles\w*)\s*=\s*\["),
    ("config_space", r"^(\w*kernels_(?:list|dict)\w*)\s*="),
    # Legality stated as the message of the exception that enforces it. This is where FlyDSL writes
    # most of its contract — `tile_k_bytes must be divisible by 64`, `tile_n must be divisible by
    # (CShuffleNLane*EVec)` — and a checker you cannot cite is a constraint an optimizer rediscovers
    # by compiling. Anchored on the `raise`, so the excerpt shows the condition above it.
    ("config_validity", r"\braise\s+(?:ValueError|NotImplementedError|RuntimeError|AssertionError)"
                        r"\(\s*f?[\"']([^\"'\n]*(?:must be|must equal|requires|only support|"
                        r"not supported|not yet supported|exceed|divisible|fixes)[^\"'\n]*)"),
    # How split-K partials are combined: the module-level state constants and the helpers that rotate
    # the per-stream signal state. `SPLIT_K\s*[=:]` above only sees the split factor itself.
    ("split_k", r"^(SPLIT_K_[A-Z0-9_]+)\s*(?::[^=\n]+)?="),
    ("split_k", r"\bdef\s+(_?\w*split_k_(?:signal|semaphore)\w*)\s*\("),
    # The kernel side of that combine: the helpers that initialise the output and wait on it, and the
    # `if const_expr(IS_SPLIT_K):` blocks that hold it. The host constants above say a combine
    # exists; these lines are how it is written, and `end_line` carries each block's extent.
    ("split_k", r"\bdef\s+(zero_c|\w*split_k_barrier\w*)\s*\("),
    ("split_k", r"^[ \t]*if\s+const_expr\(\s*(IS_SPLIT_K)\s*\)\s*:"),
    # Accumulating into the output with atomics. Split-K partials and MoE top-k scatter both combine
    # this way, and it fixes two things a store does not: the output must hold its initial value
    # before the first add, and the rounding order is whatever order the workgroups arrive in.
    ("atomic_combine", r"\bllvm\.AtomicRMWOp\(\s*llvm\.AtomicBinOp\.(\w+)"),
    ("atomic_combine", r"\btl\.atomic_add\("),
    # Workgroup-to-tile/XCD mapping: Triton's shared `remap_xcd` / `pid_grid` helpers and the FlyDSL
    # MoE kernel's `xcd_swizzle` remap. Kept apart from `persistent`, which is a different decision.
    ("grid_mapping", r"\bdef\s+(remap_xcd\w*|pid_grid\w*)\s*\("),
    ("grid_mapping", r"\b(xcd_swizzle)\s*(?::\s*int\s*)?=|\b(xcd_swizzle)\s*>\s*0"),
    ("grid_mapping", r"\b(\w*per_xcd\w*)\s*="),
    # Which scale granularity a quantized kernel consumes. A per-token/per-channel kernel and a
    # 128x128 block-scale kernel are different math contracts, and that is decided here, not by dtype.
    # `*_repr` names are Triton's kernel-name serialisers, which carry the granularity in the kernel
    # name and consume no scale.
    ("scale_operand", r"\b(?!\w*_repr\b)(\w*(?:per_token|per_tensor|per_channel|block_?scale)\w*)"
                      r"\s*=(?!=)"),
    # Where the scale meets the accumulator, which is the granularity written as arithmetic. Inside
    # the K loop (`acc += dot(a, b) * a_scale[:, None] * …`) every K block's partial is scaled before
    # it joins the fp32 accumulator: the 128x128 block-scale contract. After the loop
    # (`acc *= a_scale[:, None] * b_scale[None, :]`) one multiply finishes a per-token/per-channel
    # accumulator.
    ("scale_in_k_loop", r"(\w+)\s*\+=\s*\(?\s*(?:tl\.dot\([^)]*\)|\w+)\s*\*\s*(\w*scale\w*)"
                        r"\[\s*(?::\s*,\s*None|None\s*,\s*:)\s*\]"),
    ("scale_after_k_loop", r"(\w+)\s*\*=\s*(\w*scale\w*)\[\s*:\s*,\s*None\s*\]\s*\*\s*"
                           r"(\w*scale\w*)\[\s*None\s*,\s*:\s*\]"),
    # Gluon states its LDS layout as a constructor, which is the same bank-conflict question the
    # FlyDSL XOR16 helper answers by hand.
    ("lds_swizzle", r"\b(?:gl|ttgl)\.(SwizzledSharedLayout|PaddedSharedLayout)\s*\("),
    # The dispatcher's M granularity for tuned-config lookups (see the `lds_pad` note above).
    ("dispatch_padding", r"\b(padded_[mnk])\s*[=:]\s*([0-9]+)"),
    # The lookup itself: which shipped config a shape is handed.
    ("config_lookup", r"\bdef\s+(get_gemm_config)\s*\("),
    # Hand-written HIP: MFMA builtins, the declared workgroup size, the skinny family's tiling knobs
    # (one record per kernel template) and the size test that picks a skinny variant.
    ("mfma_intrinsic", r"\b__builtin_amdgcn_(mfma_[a-z0-9_]+)\s*\("),
    ("waves", r"__launch_bounds__\s*\(\s*([^)]+?)\s*\)"),
    ("tunable_param", r"template\s*<([^<>]*\bYTILE\b[^<>]*)>"),
    ("kernel_family", r"\bif\s*\(\s*\(?\s*(K_in\s*\*\s*N_in\s*<=\s*[^)&]+?)\s*\)"),
    # Sizing the grid against the CU count. AITER's ASM launchers pick tile and split by the number
    # of rounds `ceil(workgroups / num_cu)` and the idle CUs in the last one; the HIP skinny launcher
    # sizes waves per group with `mindiv(rows, CuCount * YTILE, …)`. It is the grid-fill rule written
    # as code, and the same on every architecture because the CU count is read at run time.
    ("grid_fill", r"\b(__wvPrGrp)\s*=\s*(mindiv\([^;]*\))"),
    ("grid_fill", r"\b(\w*round\w*)\s*=\s*\(\s*(\w+)\s*\+\s*(num_cu|cu_num|CuCount)\s*-\s*1\s*\)"
                  r"\s*/\s*(?:num_cu|cu_num|CuCount)"),
    ("grid_fill", r"\b(max_splitk)\s*=\s*([^;]*num_cu[^;]*);"),
    # Partial sums reduced across the lanes of a wave: how a skinny kernel splits K inside a wave.
    ("lane_reduction", r"\b(__shfl(?:_xor|_down|_up)?)\s*\("),
    # Which lane holds which MFMA fragment element: the HGEMM's A/B load and C store index math, the
    # row epilogue's `lane_div_16 * 4` row offset and the A8 kernel's 4x16 lane layout. Scaling an
    # accumulator per row or per column, or writing it back, needs exactly this mapping.
    ("fragment_layout", r"\b((?:ldmatrix|stmatrix)_[abc]_\w+_idx)\s*=\s*([^\n]+)"),
    ("fragment_layout", r"\b(lane_div_16_mul4)\s*=\s*([^\n]+)"),
    ("fragment_layout", r"\b(layout_lane\d+)\s*=\s*(fx\.make_layout\([^\n]*)"),
    ("layout_epilogue", r"\bdef\s+(default_epilog|mfma_epilog)\s*\("),
    # The A8/MoE data-movement helpers, named by width (16/8/4-byte XOR16 LDS stores, dwordx4 global
    # copies) and the preshuffled layouts they read.
    ("access_width", r"\bdef\s+((?:lds_(?:store|load)|buffer_copy|load_b)_\w+)\s*\("),
    ("layout_preshuffle", r"\bdef\s+(make_preshuffle_\w+_layout)\s*\("),
    # Host contract: the launch grid (which block index is which tile, which is the K split), the
    # compiled-kernel cache and the kernel-name grammar a tuned row is parsed with.
    ("grid_mapping", r"\.launch\(\s*grid\s*=\s*\(([^)]*)\)"),
    ("jit_cache", r"@functools\.lru_cache\(\s*(?:maxsize\s*=\s*)?(\w*)\s*\)\s*\n\s*def\s+(\w+)"),
    ("instance_name", r"^(_\w*KERNEL_RE)\s*=\s*re\.compile\("),
    # The gate an implementation's own test applies against its reference, including the cases it
    # relaxes — which is where the implementation says its numerics drift.
    ("parity_tolerance", r"^(DEFAULT_(?:ATOL|RTOL|PASS_PCT))\s*=\s*([0-9.eE+-]+)"),
    ("parity_tolerance", r"[\"'](pass_pct|max_delta_limit)[\"']\s*:\s*([0-9.eE+-]+)"),
    # opus's buffer access and wait builtins — the HIP spelling of the loads FlyDSL issues.
    ("access_width", r"\b__builtin_amdgcn_raw_buffer_(load|store)_(\w+)\s*\("),
    ("scheduling", r"\b__builtin_amdgcn_(s_waitcnt\w*|s_barrier|sched_group_barrier|sched_barrier|"
                   r"s_setprio)\s*\("),
]

# Which gfx a decision is conditional on. Captured separately from the rules because it applies to
# ALL of them: a tile size inside `if arch == "gfx942"` is a different fact from the same number in
# a shared path, and "only true on one arch" is precisely what the user of this corpus needs told.
ARCH_GATE = re.compile(
    r"""(?x)
    (?: GPU_ARCH | arch | ARCH | gpu_arch | target | get_rocm_arch\(\) )
    \s* (?: == | \sin\s | \.startswith ) \s*
    [\(\[]? \s* ['"]? (gfx\d+[a-z]*) """)
ARCH_ANY = re.compile(r"\b(gfx\d{3,4}[a-z]?)\b")


def strip_comments(text, python):
    """Blank out comment bodies, preserving line count and column positions.

    Line numbers are the product here, so nothing may shift: a record whose excerpt points three
    lines off is worse than no record, because the reader trusts it enough not to double-check.
    """
    if python:
        return re.sub(r"#[^\n]*", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.DOTALL)


def arch_context(lines, idx):
    """The gfx this line sits under, by walking back through enclosing conditions.

    Indentation-based for Python and brace-agnostic for C++, so it is a hint and labelled as one:
    `arch_scope` is reported, never used to filter. An over-eager arch attribution would quietly
    narrow a fact that is actually general, and the reader has the line number to check.
    """
    for j in range(idx, max(-1, idx - 40), -1):
        m = ARCH_GATE.search(lines[j])
        if m:
            return m.group(1)
    return ""


def excerpt(lines, idx, span=2):
    lo, hi = max(0, idx - span), min(len(lines), idx + span + 1)
    return [f"{n + 1}: {lines[n].rstrip()}" for n in range(lo, hi) if lines[n].strip()]


def source_evidence_id(record):
    """Content-bound identity used by decision cards instead of a loose `file:line` pointer.

    `file:line` is useful provenance but not an identity: after an upstream refresh the same line can
    contain a different construct and a card would still appear grounded. Including the match and
    excerpt means such a change invalidates the reference and forces the card to be reviewed.
    """
    fields = {
        key: record.get(key)
        for key in ("category", "language", "file", "line", "match", "captured",
                    "arch_scope", "evidence")
    }
    canonical = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "src_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def statement_ends(raw):
    """Map each line that starts a Python statement to the last line of that statement.

    A record is one line plus a two-line excerpt, which is enough for a constant and not for a
    mechanism: a combine written as a 50-line `if` block, or a helper `def`, needs its extent for a
    reader to know where to stop. The AST gives it exactly. Only `.py` has a parser here; a file that
    does not parse simply gets no extents.
    """
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return {}
    ends = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and getattr(node, "end_lineno", None):
            ends[node.lineno] = max(ends.get(node.lineno, 0), node.end_lineno)
    return ends


INVENTORY_KEYS = {
    "tile_m": "tile_m", "tilem": "tile_m", "tile_n": "tile_n", "tilen": "tile_n",
    "splitk": "split_k_capable", "bpreshuffle": "bpreshuffle", "pf": "prefetch", "tn": "tn",
    "subk": "sub_k", "bias": "bias",
}
GFX_DIR = re.compile(r"(?:^|/)(gfx[0-9]+[a-z]?)/")


def inventory_records(raw, rel, language):
    """One record per kernel row of an ASM inventory CSV (`hsa/<gfx>/<kind>/*.csv`).

    Read by header, not by position: the kinds disagree on column order and spelling (`tileM`,
    `tile_m`, `tile_M`). `splitK` is a capability flag, not a split factor — the launchers accept a
    split above 1 only for a kernel whose flag is 1 — so it is recorded as `split_k_capable`. The
    architecture is the directory the CSV sits in.
    """
    lines = raw.splitlines()
    if not lines:
        return []
    header = [cell.strip() for cell in lines[0].split(",")]
    gfx = GFX_DIR.search(rel)
    out = []
    for idx in range(1, len(lines)):
        cells = [cell.strip() for cell in lines[idx].split(",")]
        if len(cells) != len(header):
            continue
        captured = [f"{INVENTORY_KEYS[key.lower()]}={value}"
                    for key, value in zip(header, cells) if key.lower() in INVENTORY_KEYS]
        record = {
            "category": "asm_tile",
            "language": language,
            "file": rel,
            "line": idx + 1,
            "match": lines[idx].strip(),
            "captured": captured,
            "arch_scope": gfx.group(1) if gfx else "",
            "evidence": [f"1: {lines[0].rstrip()}", f"{idx + 1}: {lines[idx].rstrip()}"],
        }
        record["evidence_id"] = source_evidence_id(record)
        out.append(record)
    return out


def route_language(name):
    tokens = name.lower().split("_")
    for token, language in ROUTE_TARGETS:
        if token in tokens:
            return language
    return None


def routing_records(path, rel):
    """Routing statements in AITER's dispatch code, each filed under the backend it routes to.

    The excerpt reaches five lines up, because the condition that selects a route sits above the
    statement that names it. No architecture is inferred: a route in the `else` of `gfx == "gfx942"`
    is the other architectures', and a nearest-literal guess would file it under gfx942. The excerpt
    shows the condition; the card that cites it states the architecture.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    view = strip_comments(raw, python=True)
    rlines = raw.splitlines()
    ends = statement_ends(raw)
    out = []
    for pattern in ROUTING_RULES:
        for m in re.finditer(pattern, view, re.MULTILINE):
            language = route_language(m.group(1))
            if language is None:
                continue
            idx = view.count("\n", 0, m.start())
            record = {"category": "kernel_family", "language": language, "file": rel,
                      "line": idx + 1}
            if ends.get(idx + 1, 0) > idx + 1:
                record["end_line"] = ends[idx + 1]
            record.update({
                "match": m.group(0).strip(),
                "captured": [m.group(1)],
                "arch_scope": "",
                "evidence": [f"{n + 1}: {rlines[n].rstrip()}"
                             for n in range(max(0, idx - 5), min(len(rlines), idx + 2))
                             if rlines[n].strip()],
            })
            record["evidence_id"] = source_evidence_id(record)
            out.append(record)
    return list({record["evidence_id"]: record for record in out}.values()), raw


def scan_file(path, rel, language):
    """Every rule hit in one file, as source-evidence records.

    A hit answers only "what is written here?". It is not yet a development decision: deciding
    whether to copy, benchmark or reject the pattern requires conditions and evidence strength, which
    live in `decisions/gemm.yaml`.

    `end_line` is location, not identity: it is left out of `source_evidence_id`, so an edit deep
    inside a long function does not invalidate every card that cites the function's first line.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if path.endswith(".csv"):
        return inventory_records(raw, rel, language), raw, raw.splitlines()
    python = path.endswith(".py")
    view = strip_comments(raw, python=python)
    vlines, rlines = view.splitlines(), raw.splitlines()
    ends = statement_ends(raw) if python else {}
    out = []
    for category, pattern in RULES:
        # MULTILINE so a rule can anchor to a line, which is what a declaration-per-line construct
        # like Triton's `NAME: tl.constexpr,` parameter list needs. No rule uses `$` to mean
        # end-of-file, so the wider meaning of the anchors costs nothing here.
        for m in re.finditer(pattern, view, re.MULTILINE):
            idx = view.count("\n", 0, m.start())
            if idx >= len(rlines):
                continue
            groups = [g for g in m.groups() if g] if m.groups() else []
            record = {
                "category": category,
                "language": language,
                "file": rel,
                "line": idx + 1,
            }
            if ends.get(idx + 1, 0) > idx + 1:
                record["end_line"] = ends[idx + 1]
            record.update({
                "match": m.group(0).strip(),
                "captured": groups,
                "arch_scope": arch_context(vlines, idx),
                "evidence": excerpt(rlines, idx),
            })
            record["evidence_id"] = source_evidence_id(record)
            out.append(record)
    # Overlapping rules occasionally describe the exact same source construct (notably asm launcher
    # aliases). One piece of evidence must have one identity; keeping duplicates inflates coverage and
    # makes an ID ambiguous to a decision card.
    unique = {}
    for record in out:
        unique.setdefault(record["evidence_id"], record)
    return list(unique.values()), raw, vlines


def autotune_configs(path, rel):
    """Triton's `@triton.autotune(configs=[Config({...}, num_warps=..), ...])`, via AST.

    The one place a parser earns its keep: the tuned search space is a literal in the decorator, and
    it is the single most reusable thing in a Triton kernel — a FlyDSL author starting from scratch
    wants the tile shapes somebody already swept, not a guess. Regex cannot read a nested dict
    reliably, and a half-read config list would be an invented search space.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not name.endswith("Config"):
            continue
        knobs = {}
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                for k, v in zip(arg.keys, arg.values):
                    try:
                        knobs[ast.literal_eval(k)] = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        pass
        for kw in node.keywords:
            try:
                knobs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                pass
        if knobs:
            out.append({"file": rel, "line": node.lineno, "knobs": knobs})
    return out


TUNED_SHAPE_SEG = re.compile(r"^(?P<dim>[MNK])=(?P<val>\d+)$")


def parse_tuned_name(name):
    """Split `gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json` into gfx, tags, shape.

    A splitter rather than a grammar, on purpose. The names carry more forms than a regex should
    claim to know — `FUSED-GEMM-AFP4WFP4-A16W16` has two dtype tags, `FF-A16W16-fused` mixes case,
    `GEMM-A16W16-ATOMIC` and `-gated` are epilogue markers — and a pattern that insisted on
    kernel-then-variant silently dropped 239 of 257 files into an empty-field bucket. Splitting on
    `-` and keeping the middle segments as an ordered tag list records what the name says without
    inventing a taxonomy the filenames do not actually promise.
    """
    stem = name[:-len(".json")] if name.endswith(".json") else name
    segs = stem.split("-")
    if not segs or not segs[0].startswith("gfx"):
        return None
    gfx, shape, tags = segs[0], {}, []
    for seg in segs[1:]:
        m = TUNED_SHAPE_SEG.match(seg)
        if m:
            shape[m.group("dim").lower()] = int(m.group("val"))
        else:
            tags.append(seg)
    return {"gfx": gfx, "tags": tags, "shape": shape}


def tuned_configs(aiter):
    """The shipped per-shape knob sets, as one record per (file, M bucket).

    Flattened rather than summarised. The obvious compression — "here are the tile shapes AITER
    uses" — throws away the two things that make the table worth having: which gfx it was tuned on,
    and which M bucket it applies to. A knob set is only meaningful with both, and a corpus that
    dropped them would read as a global recommendation, which is exactly what it is not.
    """
    base = os.path.join(aiter, TUNED_CONFIG_DIR)
    if not os.path.isdir(base):
        return [], [{"language": "triton", "subtree": TUNED_CONFIG_DIR,
                     "why": "no such directory (no shipped tuned configs to read)"}]
    out, unparsed = [], []
    for name in sorted(os.listdir(base)):
        if not name.endswith(".json"):
            continue
        meta = parse_tuned_name(name)
        if meta is None:
            unparsed.append(name)
            continue
        try:
            with open(os.path.join(base, name), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            unparsed.append(name)
            continue
        if not isinstance(data, dict):
            unparsed.append(name)
            continue
        for bucket, knobs in data.items():
            if not isinstance(knobs, dict):
                continue
            out.append({"file": os.path.join(TUNED_CONFIG_DIR, name),
                        "gfx": meta["gfx"], "tags": meta["tags"], "shape": meta["shape"],
                        "m_bucket": str(bucket),
                        "knobs": {k: v for k, v in sorted(knobs.items())}})
    gap = []
    if unparsed:
        gap.append({"language": "triton", "subtree": TUNED_CONFIG_DIR,
                    "why": f"{len(unparsed)} config filename(s) not in `gfx…-TAG…[-N=..]` form, "
                           f"skipped rather than guessed: {', '.join(sorted(unparsed)[:4])}"})
    return out, gap


def summarize_tuned(rows):
    """Group the shipped knob sets by (gfx, tags, M bucket) and report each knob's value spread.

    Grouped rather than copied. 1742 verbatim rows is a re-encoding of JSON that already exists in
    aiter — a second copy with a shelf life, and 34k lines nobody diffs. The grouping is not a
    space trick though: it surfaces the distinction the raw rows bury. Within one bucket some knobs
    take a single value across every shipped shape config (`BLOCK_SIZE_K` is 128 in all nine gfx942
    A8W8_BLOCKSCALE M<=32 files, `num_stages` always 2) while others move with the shape. The first
    kind is a useful candidate to seed; the second is a dimension they still have to vary. Both are
    read off the counts, not asserted — `unanimous` is literally "one distinct value".

    Exact per-shape rows are deliberately not reproduced: an author who needs N=4096,K=7168 should
    open the source JSONs, whose exact paths are retained on every group below.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["gfx"], " ".join(r["tags"]), r["m_bucket"])].append(r)
    out = []
    for (gfx, tags, bucket), members in sorted(groups.items()):
        spread = collections.defaultdict(collections.Counter)
        for mem in members:
            for k, v in mem["knobs"].items():
                spread[k][json.dumps(v)] += 1
        fixed, varies = {}, {}
        for k, counter in sorted(spread.items()):
            if len(counter) == 1:
                fixed[k] = json.loads(next(iter(counter)))
            else:
                # `value: count`, not `valueXcount` — `.cgx23` reads as a cache modifier named
                # `.cgx23` rather than `.cg` seen 23 times, and `32x20` as a tile shape.
                varies[k] = ", ".join(f"{json.loads(val)}: {n}" for val, n in counter.most_common())
        rec = {
            "config_id": "cfg_" + hashlib.sha256(
                json.dumps([gfx, tags, bucket], separators=(",", ":")).encode()
            ).hexdigest()[:16],
            "gfx": gfx,
            "tags": tags,
            "m_bucket": bucket,
            "shape_configs": len(members),
            "source_files": sorted({member["file"] for member in members if member.get("file")}),
        }
        if len(members) == 1:
            # One shape config means there is no cross-shape agreement to report, so the field is not
            # called agreement. `same_across_configs` over a single row would read as an invariant
            # backed by evidence that does not exist.
            rec["knobs"] = fixed
        else:
            # Every shipped shape config in this bucket carries this value. Useful seed evidence, not
            # proof of a constraint — the source does not preserve alternatives or benchmark results.
            rec["same_across_configs"] = fixed
            rec["varies_by_shape"] = varies
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# AITER's tuned GEMM databases: `aiter/configs/**/*_tuned_*gemm*.csv`
# ---------------------------------------------------------------------------------------------
# The Triton JSON above records one backend's selected knobs. The CSV databases record something the
# JSON cannot: for each shipped (gfx, CU count, M, N, K) AITER raced its backends — asm, ck, cktile,
# flydsl, triton — and kept one. That is the only place in AITER where "which implementation for
# which shape regime" is written down, and it is where the FlyDSL configurations AITER actually ships
# live. What is kept is the selection; what is dropped on read is how long anything took.
TUNED_DB_DIRS = ("aiter/configs", "aiter/configs/model_configs")
TUNED_DB_KIND = re.compile(
    r"(?P<kind>(?:bf16|a8w8|a4w4)(?:_blockscale)?(?:_bpreshuffle)?_tuned_(?:batched_)?gemm)")
# Timings from a box this corpus cannot name. The always-on layer carries no performance numbers, so
# these columns never enter a record.
TUNED_DB_TIMING = ("bw", "errRatio", "err_ratio", "tflops", "us")
TUNED_DB_CONTEXT = ("bias", "bpreshuffle", "dtype", "outdtype", "q_dtype_w", "scaleAB")
M_BUCKETS = ((16, "M<=16"), (64, "M17-64"), (256, "M65-256"), (1024, "M257-1024"),
             (4096, "M1025-4096"), (None, "M>4096"))
M_BUCKET_ORDER = {label: i for i, (_, label) in enumerate(M_BUCKETS)}
# AITER's two serialisations of a FlyDSL configuration: `gemm_kernels.flydsl_kernel_name` for the
# split-K/small-M HGEMM, and the A8 tune `kernelInstance.name` (upstream spells it `bpreshuflle`).
FLYDSL_HGEMM_NAME = re.compile(
    r"^flydsl_gemm(?P<stage>\d+)_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_split_k(?P<split_k>\d+)_"
    r"block_m_warp(?P<block_m_warps>\d+)_block_n_warp(?P<block_n_warps>\d+)_"
    r"async_copy(?P<async_copy>True|False)_b_to_lds(?P<b_to_lds>True|False)_"
    r"b_preshuffle(?P<b_preshuffle>True|False)_c_to_lds(?P<c_to_lds>True|False)"
    r"(?P<small_m>_small_m(?:_nr(?P<n_tile_repeat>\d+))?(?:_pn(?P<persistent_n_tiles>\d+))?"
    r"(?:_wpe(?P<waves_per_eu>\d+))?(?:_ur(?P<b_to_lds_unroll>\d+))?)?"
    r"_(?P<target_gfx>gfx[0-9a-z]+)$")
FLYDSL_A8_NAME = re.compile(
    r"^flydsl_bpreshu(?:ffle|flle)_(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<a_dtype>[A-Z0-9]+)_(?P<w_dtype>[A-Z0-9]+)_(?P<out_dtype>[A-Z0-9]+)_"
    r"(?P<lds_stage>\d+)x(?P<use_cshuffle_epilog>\d+)x(?P<use_async_copy>\d+)x(?P<waves_per_eu>\d+)_"
    r"(?P<scheduler>\w+)$")


def m_bucket(m):
    for bound, label in M_BUCKETS:
        if bound is None or m <= bound:
            return label
    return M_BUCKETS[-1][1]


def decode_flydsl_kernel(name):
    """`(family, dtype, knobs)` encoded in a shipped FlyDSL kernel name, or None.

    The names are AITER's own serialisation of the configuration, so this reads a configuration
    rather than guessing one. `tile` and `warps` stay single knobs: their parts vary together, and
    three independent spreads for tile_m/n/k would advertise combinations nobody selected.
    """
    m = FLYDSL_HGEMM_NAME.match(name or "")
    if m:
        g = m.groupdict()
        knobs = {
            "tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}",
            "split_k": int(g["split_k"]),
            "warps": f"{g['block_m_warps']}x{g['block_n_warps']}",
            "b_to_lds": g["b_to_lds"] == "True",
            "b_preshuffle": g["b_preshuffle"] == "True",
            "async_copy": g["async_copy"] == "True",
            "c_to_lds": g["c_to_lds"] == "True",
            "stages": int(g["stage"]),
        }
        for key in ("n_tile_repeat", "persistent_n_tiles", "waves_per_eu", "b_to_lds_unroll"):
            if g.get(key):
                knobs[key] = int(g[key])
        family = "small_m" if g["small_m"] else "hgemm"
        return family, f"a{g['a_dtype']}_w{g['w_dtype']}_{g['out_dtype']}", knobs
    m = FLYDSL_A8_NAME.match(name or "")
    if m:
        g = m.groupdict()
        knobs = {
            "tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}",
            "lds_stage": int(g["lds_stage"]),
            "use_cshuffle_epilog": int(g["use_cshuffle_epilog"]),
            "use_async_copy": int(g["use_async_copy"]),
            "waves_per_eu": int(g["waves_per_eu"]),
            "scheduler": g["scheduler"].lower(),
        }
        dtype = f"a{g['a_dtype'].lower()}_w{g['w_dtype'].lower()}_{g['out_dtype'].lower()}"
        return "a8_preshuffle", dtype, knobs
    return None


def tuned_db_rows(aiter):
    """Rows of AITER's tuned GEMM databases, without their timing columns.

    Kept: what makes a row comparable (gfx, CU count, M/N/K, the dtype/bias/preshuffle key) and what
    was selected (backend, kernel name). A database without a backend column is inventoried and
    reported as a gap rather than attributed to a backend by guessing from kernel-name prefixes.
    """
    rows, inventory, gaps = [], [], []
    for sub in TUNED_DB_DIRS:
        base = os.path.join(aiter, sub)
        if not os.path.isdir(base):
            gaps.append({"language": "aiter_tuned_db", "subtree": sub, "why": "no such directory"})
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith(".csv") or "untuned" in name or "fmoe" in name:
                continue
            kind = TUNED_DB_KIND.search(name)
            if not kind:
                continue
            rel = os.path.join(sub, name)
            with open(os.path.join(base, name), encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                columns = list(reader.fieldnames or [])
                file_rows = list(reader)
            has_backend = "libtype" in columns
            inventory.append({
                "file": rel, "db_kind": kind.group("kind"), "rows": len(file_rows),
                "backend_column": has_backend,
                "gfx": sorted({r["gfx"] for r in file_rows if r.get("gfx")}),
            })
            if file_rows and not has_backend:
                gaps.append({"language": "aiter_tuned_db", "subtree": rel,
                             "why": "no `libtype` column: the selected backend is not recorded, so "
                                    "these rows are inventoried but not counted as a selection"})
            for r in file_rows:
                try:
                    m, n, k, cu = int(r["M"]), int(r["N"]), int(r["K"]), int(r["cu_num"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({
                    "file": rel, "db_kind": kind.group("kind"), "gfx": r.get("gfx") or "",
                    "cu_num": cu, "m": m, "n": n, "k": k,
                    "libtype": (r.get("libtype") or None) if has_backend else None,
                    "kernel_name": r.get("kernelName") or "",
                    "context": {c: r[c] for c in TUNED_DB_CONTEXT if c in r},
                })
    return rows, inventory, gaps


def _short_id(prefix, parts):
    return prefix + hashlib.sha256(
        json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]


def _value_text(value):
    return "true" if value is True else "false" if value is False else str(value)


def summarize_flydsl_tuned(rows):
    """FlyDSL selections grouped by (gfx, CUs, database, family, dtype, M bucket).

    Same contract as `summarize_tuned`: a knob every selected row in the group shares is a seed, a
    knob that moved with the shape is one to vary. Nothing here says a selection beat anything by a
    margin — the database keeps the winner, not the race.
    """
    groups = collections.defaultdict(list)
    unparsed = collections.Counter()
    for r in rows:
        if r["libtype"] != "flydsl":
            continue
        decoded = decode_flydsl_kernel(r["kernel_name"])
        if decoded is None:
            unparsed[r["file"]] += 1
            continue
        family, dtype, knobs = decoded
        groups[(r["gfx"], r["cu_num"], r["db_kind"], family, dtype, m_bucket(r["m"]))].append(
            (r, knobs))
    out = []
    for key in sorted(groups, key=lambda t: (t[0], t[1], t[2], t[3], t[4], M_BUCKET_ORDER[t[5]])):
        gfx, cu, kind, family, dtype, bucket = key
        members = groups[key]
        spread = collections.defaultdict(collections.Counter)
        for _, knobs in members:
            for knob, value in knobs.items():
                spread[knob][json.dumps(value)] += 1
        fixed, varies = {}, {}
        for knob, counter in sorted(spread.items()):
            if len(counter) == 1 and sum(counter.values()) == len(members):
                fixed[knob] = json.loads(next(iter(counter)))
            else:
                varies[knob] = ", ".join(f"{_value_text(json.loads(v))}: {n}"
                                         for v, n in counter.most_common())
        rec = {
            "config_id": _short_id("cfg_", ["aiter_tuned_db", gfx, cu, kind, family, dtype, bucket]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "family": family, "dtype": dtype,
            "m_bucket": bucket, "shape_configs": len(members),
            "source_files": sorted({r["file"] for r, _ in members}),
        }
        if len(members) == 1:
            rec["knobs"] = fixed
        else:
            rec["same_across_configs"] = fixed
            rec["varies_by_shape"] = varies
        out.append(rec)
    gaps = [{"language": "aiter_tuned_db", "subtree": path,
             "why": f"{n} flydsl kernelName(s) in neither shipped naming form; skipped, not guessed"}
            for path, n in sorted(unparsed.items())]
    return out, gaps


def summarize_backend_selection(rows):
    """How many shipped rows AITER's tuner assigned to each backend, per (gfx, CUs, database, M).

    A count of selections, not of speedups. A shape shipped in two model databases counts twice,
    and the competing set, tuner box and toolchain are not recorded — the counts say where each
    backend was chosen, never by how much.
    """
    groups = collections.defaultdict(collections.Counter)
    files = collections.defaultdict(set)
    for r in rows:
        if not r["libtype"]:
            continue
        key = (r["gfx"], r["cu_num"], r["db_kind"], m_bucket(r["m"]))
        groups[key][r["libtype"]] += 1
        files[key].add(r["file"])
    out = []
    for key in sorted(groups, key=lambda t: (t[0], t[1], t[2], M_BUCKET_ORDER[t[3]])):
        gfx, cu, kind, bucket = key
        out.append({
            "selection_id": _short_id("sel_", ["aiter_tuned_db", gfx, cu, kind, bucket]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "m_bucket": bucket,
            "rows": sum(groups[key].values()),
            "selected_backend_rows": dict(sorted(groups[key].items())),
            "source_files": sorted(files[key]),
        })
    return out


def aiter_commit(aiter):
    r = subprocess.run(["git", "-C", aiter, "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def aiter_origin(aiter):
    """The repository the source evidence came from, as a remote URL where one exists.

    Not the local path. A reader who wants to check a citation needs to know WHICH aiter, and the
    honest way to extract from a dirty working copy is a throwaway clean worktree — which made the
    recorded path `/tmp/aiter-clean`, a directory that no longer exists and never meant anything to
    anyone else. The remote plus the commit is the pair that actually identifies the source.
    """
    r = subprocess.run(["git", "-C", aiter, "remote", "get-url", "origin"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or f"(no remote; read from {os.path.abspath(aiter)})"


def dirty_paths(aiter):
    """Repo-relative paths with uncommitted changes, as a set.

    Needed per-file, not as a boolean. The boolean this replaced (`aiter_tree_dirty`) recorded the
    problem and did nothing with it: 56 records pointed into a locally-modified kernel while the
    provenance named a commit that did not contain those edits, so `file:line` did not resolve for
    anyone but the person who ran the extractor. That is the corpus's one promise, broken, with a
    green test suite — the test asserted a commit was PRESENT, never that it IDENTIFIED what was read.
    """
    r = subprocess.run(["git", "-C", aiter, "status", "--porcelain"],
                       capture_output=True, text=True, check=False)
    out = set()
    for line in r.stdout.splitlines():
        if len(line) > 3:
            # `XY path` — and for a rename, `XY old -> new`; the new name is what was read.
            out.add(line[3:].split(" -> ")[-1].strip().strip('"'))
    return out


def collect(aiter):
    """Walk the six implementation sets. Missing subtrees are reported, never silently empty."""
    import glob as globmod
    evidence, configs, seen, missing = [], [], [], []
    for language, subtree, patterns in IMPLS:
        base = os.path.join(aiter, subtree)
        if not os.path.isdir(base):
            missing.append({"language": language, "subtree": subtree, "why": "no such directory"})
            continue
        files, empty_patterns = [], []
        for pat in patterns:
            hits = [f for f in sorted(globmod.glob(os.path.join(base, pat), recursive=True))
                    if "__pycache__" not in f and os.path.isfile(f)]
            if not hits:
                empty_patterns.append(pat)
            files += hits
        if patterns and not files:
            missing.append({"language": language, "subtree": subtree,
                            "why": f"no file matched {list(patterns)}"})
            continue
        # Reported per pattern as well. A language with one matching file used to hide every other
        # pattern that matched nothing: HIP kept `gemm_common.cu`, so the whole `opus_gemm/` tree
        # it was declared to read could vanish upstream without a line in `missing`.
        for pat in empty_patterns:
            missing.append({"language": language, "subtree": subtree,
                            "why": f"declared pattern matched no file: {pat}"})
        for path in files:
            rel = os.path.relpath(path, aiter)
            got, raw, _ = scan_file(path, rel, language)
            evidence += got
            if language in ("triton", "gluon") and path.endswith(".py"):
                configs += [dict(c, language=language) for c in autotune_configs(path, rel)]
            seen.append({"language": language, "file": rel, "lines": raw.count("\n") + 1,
                         "evidence_records": len(got)})
    for rel in ROUTING_FILES:
        path = os.path.join(aiter, rel)
        if not os.path.isfile(path):
            missing.append({"language": "routing", "subtree": os.path.dirname(rel),
                            "why": f"declared dispatch file not found: {rel}"})
            continue
        got, raw = routing_records(path, rel)
        evidence += got
        seen.append({"language": "routing", "file": rel, "lines": raw.count("\n") + 1,
                     "evidence_records": len(got)})
    tuned, tuned_gap = tuned_configs(aiter)
    # An empty autotune list is a finding, not a failed pass, so it is stated rather than left as a
    # zero a reader would take for a bug. AITER's GEMM Triton kernels carry no `@triton.autotune` at
    # all: the search space lives in the shipped JSON instead. Worth saying out loud, because an
    # author who greps for autotune in a GEMM kernel and finds nothing will otherwise conclude the
    # kernel is untuned, when in fact it is the most heavily swept code in the tree.
    if not configs:
        missing.append({"language": "triton", "subtree": "aiter/ops/triton/_triton_kernels/gemm",
                        "why": "no @triton.autotune in the GEMM family — tuning is shipped as "
                               f"per-shape JSON under {TUNED_CONFIG_DIR} "
                               f"(see evidence/gemm_tuned_configs.yaml, {len(tuned)} rows)"})
    return evidence, configs, tuned, seen, missing + tuned_gap


# ---------------------------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------------------------
def yaml_quote(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Newlines escaped, not passed through. A CK template head like `DeviceGemmMX_Xdl_CShuffleV3\n<`
    # matches across a line break, and writing that raw put a second physical line inside a quoted
    # scalar — which PyYAML folds back correctly, but which breaks the one-record-per-line property
    # the whole hand-rolled format depends on, and makes `git diff` attribute the change to the
    # wrong record. Tabs go too, for the same "a scalar occupies exactly one line" reason.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n").replace("\t", "\\t")
    return '"' + s + '"'


def emit_yaml(payload, path):
    """Hand-rolled, because a corpus file has to be readable in a diff and PyYAML's dump is not.

    Only two shapes appear (a scalar map and a list of scalar maps), which is little enough to write
    out directly and keeps this tool importable on a box without PyYAML — the same reason `kb.py`
    carries its own front-matter parser.
    """
    L = ["# GENERATED by perf_knowledge/corpus/_extract_impl_facts.py — do not hand-edit.",
         "# These are SOURCE-EVIDENCE records: what a file states, not what an author should choose.",
         "# Actionable choices live in decisions/gemm.yaml and are rendered to gemm_decisions.md.", ""]
    L.append("provenance:")
    for k, v in payload["provenance"].items():
        if isinstance(v, list):
            L.append(f"  {k}: [{', '.join(yaml_quote(x) for x in v)}]")
        else:
            L.append(f"  {k}: {yaml_quote(v)}")
    L.append("")
    for section in ("coverage", "inventory", "missing", "autotune_configs", "tuned_configs",
                    "backend_selection", "source_evidence"):
        if section not in payload:
            continue
        rows = payload[section]
        L.append(f"{section}:" if rows else f"{section}: []")
        for row in rows:
            first = True
            for k, v in row.items():
                lead = "  - " if first else "    "
                first = False
                if isinstance(v, list) and v and isinstance(v[0], str):
                    L.append(f"{lead}{k}:")
                    L += [f"      - {yaml_quote(x)}" for x in v]
                elif isinstance(v, dict):
                    L.append(f"{lead}{k}:" if v else f"{lead}{k}: {{}}")
                    L += [f"      {kk}: {yaml_quote(vv)}" for kk, vv in sorted(v.items())]
                elif isinstance(v, list):
                    L.append(f"{lead}{k}: [{', '.join(yaml_quote(x) for x in v)}]")
                else:
                    L.append(f"{lead}{k}: {yaml_quote(v)}")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aiter", help="an aiter checkout to read (required for --emit / --json)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true", help="write evidence/gemm_source.yaml")
    mode.add_argument("--json", action="store_true", help="dump to stdout instead")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="extract even when files that produce evidence have uncommitted changes; "
                         "the affected records are marked `unreproducible: true`")
    a = ap.parse_args()

    if not a.aiter:
        ap.error("--aiter is required: this tool reads a source tree that is not part of this repo")
    if not os.path.isdir(a.aiter):
        print(f"no such aiter tree: {a.aiter}", file=sys.stderr)
        return 2

    evidence, configs, tuned, seen, missing = collect(a.aiter)
    db_rows, db_inventory, db_gaps = tuned_db_rows(a.aiter)
    commit = aiter_commit(a.aiter)
    for record in evidence:
        record.setdefault("evidence_id", source_evidence_id(record))

    # Which files that CONTRIBUTED evidence are locally modified. Scoped to contributors rather than
    # the whole tree because that is the actual condition: an unrelated edit elsewhere in aiter cannot
    # make a citation unresolvable, and refusing on it would be superstition rather than a check.
    # The shipped JSON and CSV tables contribute too: a locally re-tuned database would otherwise be
    # published under a commit that never contained it.
    contributors = ({f["file"] for f in evidence} | {r["file"] for r in tuned}
                    | {r["file"] for r in db_rows})
    unreproducible = sorted(contributors & dirty_paths(a.aiter))
    if unreproducible and not a.allow_dirty:
        print(f"REFUSING: {len(unreproducible)} file(s) that produced evidence have uncommitted changes, "
              f"so their `file:line` would not resolve at {commit[:12]} for anyone else:",
              file=sys.stderr)
        for p in unreproducible[:10]:
            print(f"  {p}", file=sys.stderr)
        print("\nA citation index whose citations do not resolve is not a weaker index, it is a\n"
              "different document that looks like one. Either commit those changes, or extract from a\n"
              "clean checkout:\n"
              f"  git -C {a.aiter} worktree add /tmp/aiter-clean {commit[:12]}\n"
              "Pass --allow-dirty to record the evidence anyway; the affected records are then marked\n"
              "and the committed corpus is not allowed to contain them.", file=sys.stderr)
        return 3

    prov = {
        # The commit is the expiry date. Without it a stale fact and a current one are the same
        # document, which is how `kernel_families.md` came to state a default tile nothing checks.
        "aiter_commit": commit,
        # Empty means every citation below resolves at that commit for anyone. Non-empty names exactly
        # which ones do not, which is the part a boolean could not say.
        "aiter_dirty_sources": unreproducible,
        "aiter_origin": aiter_origin(a.aiter),
        "extractor": "perf_knowledge/corpus/_extract_impl_facts.py",
        "operator_family": "gemm",
    }
    if unreproducible:
        dirt = set(unreproducible)
        for f in evidence:
            if f["file"] in dirt:
                f["unreproducible"] = True
    payload = {
        "provenance": dict(prov, evidence_records=len(evidence), autotune_configs=len(configs)),
        "coverage": seen, "missing": missing, "autotune_configs": configs,
        "source_evidence": evidence,
    }
    # The shipped config table changes on a different clock from the source (a reselection rewrites all
    # of it; a kernel edit touches a few lines), so it gets its own file and its own diff.
    grouped = summarize_tuned(tuned)
    tuned_payload = {
        "provenance": dict(prov, records=len(grouped), rows_summarised=len(tuned),
                           source=TUNED_CONFIG_DIR),
        "tuned_configs": grouped,
    }
    flydsl_groups, flydsl_gaps = summarize_flydsl_tuned(db_rows)
    selection = summarize_backend_selection(db_rows)
    db_payload = {
        "provenance": dict(prov, records=len(flydsl_groups), selection_groups=len(selection),
                           rows_read=len(db_rows), source=", ".join(TUNED_DB_DIRS),
                           timing_columns_dropped=list(TUNED_DB_TIMING)),
        "inventory": db_inventory,
        "missing": db_gaps + flydsl_gaps,
        "tuned_configs": flydsl_groups,
        "backend_selection": selection,
    }
    if a.json:
        print(json.dumps({"evidence": payload, "tuned": tuned_payload, "tuned_db": db_payload},
                         ensure_ascii=False, indent=2))
        return 0

    os.makedirs(EVIDENCE, exist_ok=True)
    out = os.path.join(EVIDENCE, "gemm_source.yaml")
    tuned_out = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
    db_out = os.path.join(EVIDENCE, "gemm_csv_tuned_configs.yaml")
    emit_yaml(payload, out)
    emit_yaml(tuned_payload, tuned_out)
    emit_yaml(db_payload, db_out)
    by_lang = collections.Counter(f["language"] for f in evidence)
    print(f"OK: {len(evidence)} source-evidence records from {len(seen)} files "
          f"({', '.join(f'{k}={v}' for k, v in sorted(by_lang.items()))}), "
          f"{len(configs)} autotune -> {os.path.relpath(out, ROOT)}")
    print(f"OK: {len(tuned)} shipped tuned rows in {len(grouped)} (gfx, tags, M) groups "
          f"-> {os.path.relpath(tuned_out, ROOT)}")
    print(f"OK: {len(db_rows)} tuned-DB rows from {len(db_inventory)} CSVs -> {len(flydsl_groups)} "
          f"FlyDSL groups, {len(selection)} backend-selection groups "
          f"-> {os.path.relpath(db_out, ROOT)}")
    for m in missing:
        print(f"  gap: {m['language']} — {m['why']} ({m['subtree']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
