# Reserve: fork-only compiler mechanisms (LOWEST PRIORITY — not part of the method)

**None of the fork names on this page is read by upstream Triton 3.8.0** (no hit for any of them in the
installed 3.8.0 tree, `libtriton.so` included; the one upstream capability is in §0). The names below come
from a private vendor fork (LLIR-scheduler plugin, RA-hint / AGPR knobs, `amdgcnas` post-assembly peephole); all are absent
from `python/triton/knobs.py` and `include/triton/Tools/Sys/GetEnv.h` at the `v3.8.0` tag. An env var
no installed code reads is **tolerated and inert**: nothing errors, nothing changes, and the null
result reads as "this technique does not work on my kernel". Setting them changes nothing, and their
absence is not a ceiling. A note naming `LLIRSchedule.cpp`, `libLlirSched.so` or `triton/tools/amdgcnas`
describes a fork build.

## 0. Read this first: one capability here is upstream after all

Matrix/VALU co-execution scheduling ships upstream as the `coexec` scheduler strategy — **gfx1250 only**
(no route on gfx950 / gfx942). Gate and acceptance: `compiler-contract.md ## What upstream 3.8.0
actually gives you`.

## 4. Names that were checked and have no referent

No trace in the installed `v3.8.0` tree (re-checked by grep), nor, per the original check, in upstream `main`
or the fork tags: `TRITON_GLUON_SWP_PIPELINE`,
`TRITON_GLUON_COOP_LDS`, `TRITON_GLUON_PINGPONG`, `TRITON_ENABLE_ATTN_SCHED`, `AttnSchedule.cpp`,
"cooperative-LDS staging". The capabilities they gesture at: multi-buffered LDS staging by hand;
wave-level ping-pong via `gl.amd.warp_pipeline_stage` (or `TRITON_HIP_USE_BLOCK_PINGPONG` on plain).

**Not to be confused with the pack's own `TRITON_GLUON_SWP`, `TRITON_GLUON_SWP_BUF`,
`TRITON_GLUON_SWP_NOPP`** (defined and read by `scripts/patch_reinject.py`; arm the lowest-rung
pipeliner re-injection, `pipeline.md`). They are not Triton knobs, so they are outside
`CACHE_INVALIDATING_ENV_VARS`: set `TRITON_ALWAYS_COMPILE=1` when toggling them, or you A/B a cached
binary against itself.

## 6. Environment-variable index

Every left-column name is **fork-only; absent in upstream 3.8.0, inert if set** (installed-tree grep) — a
null result from setting one is not a performance ceiling.

| fork-only variable | mechanism | upstream route |
| --- | --- | --- |
| `TRITON_ENABLE_LLIR_SCHED` (superseded on the fork too) | LLIR scheduler, in-tree | `gl.amd.warp_pipeline_stage` + hand region balancing (`llir-codesign.md`); `coexec` on gfx1250 only |
| `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE` | target-machine retention for LLVM plugins | none (plugin loading on the stock wheel was not demonstrated here) |
| `TRITON_ENABLE_AMDGPU_RA_HINTS` (superseded), `TRITON_FORCE_MFMA_AGPR` | AGPR accumulator pinning | `llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]` — 1 wave/SIMD only; at 2+ waves `"256"` exceeds the per-wave budget and the binary fails at launch (`llvm-fn-attrs.md` (AGPR-budget section)); GEMM-class (`compiler-contract.md ## AGPR hint workload gate: accumulator read-cadence`). `amdgpu-mfma-vgpr-form` is an `llc` option, inert as an `llvm_fn_attrs` entry |
| `TRITON_ENABLE_AMDGCN_AS` (superseded), `TRITON_AMDGCNAS_PLUGIN` | post-assembly LICM / scalar-into-MFMA-gap peephole | **none**; reduce loop-invariant address math and in-loop scalar work at source |

**Genuinely upstream (present in the installed 3.8.0 tree).** `TRITON_CACHE_DIR`,
`TRITON_ALWAYS_COMPILE`, `TRITON_F32_DEFAULT`, `TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`,
`DISABLE_LLVM_OPT` (comma-separated LLVM flag list, so `DISABLE_LLVM_OPT=disable-machine-sink` is a
real control), `TRITON_PLUGIN_PATHS`, `LLVM_PASS_PLUGIN_PATH`, `TRITON_EXT_ENABLED` (a CMake option,
default OFF), `TRITON_HIP_USE_ASYNC_COPY`, `TRITON_HIP_USE_BLOCK_PINGPONG`,
`TRITON_HIP_USE_EXPERT_SCHEDULING`, `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`, `TRITON_DUMP_MIR`,
`TRITON_SWAP_MIR`, `TRITON_SWAP_MIR_ENABLE_MISCHED`. Not read by 3.8.0: `TRITON_HIP_USE_COEXEC_SCHEDULER`.
