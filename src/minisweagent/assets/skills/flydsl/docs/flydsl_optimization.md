---
layer: "flydsl"
category: "optimization"
tags: ["flydsl", "optimization", "kernel", "amd", "gpu", "mfma", "lds"]
last_updated: 2026-06-01
---

# FlyDSL Kernel Optimization Workflow

Parameter tuning alone yields marginal gains. **Prioritize structural optimizations in early patches**, then fall back to tuning in later patches.

## Step 1: Read Before You Optimize

1. Read the **full kernel source** — all `@flyc.kernel` functions, their algorithm, data flow, and loop structure
2. Read the **`@flyc.jit` host wrapper** — count how many kernels are launched per call and what data each receives. Multiple kernels sharing data = fusion opportunity
3. Read **any imported helper modules** (e.g. `flydsl.utils`, `flydsl.expr`) — they contain reusable building blocks and may reveal optimization opportunities
4. Read the **test harness** — know what shapes, dtypes, and modes are benchmarked

## Step 2: Classify the Bottleneck

- Check the **target GPU arch** via `get_hip_arch()` — LDS size, MFMA variants, and wavefront width are arch-dependent (e.g. gfx942: 64 KB LDS, 304 CUs, wavefront 64)
- **Memory-bound** → reduce data movement (fusion, LDS caching, vectorization)
- **Compute-bound** → improve instruction throughput (MFMA selection, software pipelining)
- **Latency-bound** (small shapes) → reduce kernel launch count (fusion)

## Step 3: Optimize — High Impact First

### Tier 1: Structural (highest impact)

- **Kernel fusion**: if the `@flyc.jit` wrapper launches 2+ kernels that share input data, merge them into a single `@flyc.kernel`. Eliminates launch overhead and redundant HBM reads
- **Fast-path relaxation**: look for overly-restrictive conditions guarding optimized code paths (e.g. disabled branches, alignment checks stricter than necessary). Relaxing these lets more shapes use the fast path
- **Loop restructuring**: if the kernel uses constexpr loop unrolling that causes code bloat for large iteration counts, convert to `scf.for` with loop-carried state to reduce binary size and register pressure
- **Redundant work elimination**: identify repeated loads, recomputed indices, or overlapping branches, and hoist/cache them
- **Algorithm replacement**: if the current algorithm has unnecessary passes over data, restructure to reduce pass count (e.g. online softmax vs two-pass, fused attention vs separate Q×K then softmax then ×V)

### Tier 2: Memory hierarchy (medium impact)

- **LDS utilization**: if the kernel reads the same global data multiple times across threads, stage through LDS for reuse. Use `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`
- **Vectorized access**: use the widest vector loads/stores (`vec(8, ...)`, `vec(4, ...)`) that match the element type for maximum HBM bandwidth
- **Overlap loads with compute**: move global loads earlier so they complete while ALU/MFMA work is in progress. Use scheduler barriers (`sched_barrier`) to control interleaving
- **Pre-load across passes**: if the kernel makes multiple passes, load data needed in later passes during earlier ones to avoid redundant HBM reads
- **Data layout / coalescing**: ensure memory access patterns are coalesced; restructure loop ordering if needed
- **Register pressure management**: balance between keeping data in registers vs spilling to LDS

### Tier 3: Compute (medium impact)

- **MFMA instruction selection**: use the most efficient variant available on the target arch (via `flydsl.expr.rocdl`)
- **Software pipelining**: overlap LDS reads/writes with MFMA compute using ping-pong buffers and scheduler barriers
- **Scheduler tuning**: match `sched_mfma` group counts to actual MFMA ops per iteration; verify `sched_dswr`/`sched_dsrd` timing
- **Loop unrolling**: unroll inner loops to expose ILP; merge loops that iterate over the same range

### Tier 4: Parameter tuning (low impact)

- Block size, tile dimensions, unroll factors, `known_block_size` hints

## Modification Rules

- **For kernel fusion**: you MAY create new `@flyc.kernel` functions, remove old ones, and modify the `@flyc.jit` wrapper to launch the fused kernel
- **For all other optimizations**: modify code inside `@flyc.kernel` functions and their kernel-internal helper functions
- The `@flyc.jit` function's **external signature** (parameters as called by the test harness) must remain unchanged
- Do NOT modify: build system, compilation flags, test harness, or benchmark framework

## Correctness Constraints

Always verify after each patch — violations often cause silent corruption:
- **LDS limit**: check `get_hip_arch()` for the arch-specific limit (e.g. 64 KB on gfx942) — exceeding it silently corrupts results
- **Tile divisibility**: `tile_k_bytes % 64 == 0`; `tile_m * tile_k * elem_bytes` divisible by thread count
- **FP8 (E4M3FNUZ)**: value `0x80` is NaN — sanitize loads with byte AND `0x7F`
- **f32→f16 truncation**: clamp to ±65504 first to avoid Inf
- **Alignment**: ensure tile/vector dimensions satisfy any alignment requirements in the kernel's memory access patterns

## Step 4: Validate

1. Run correctness tests first — never sacrifice correctness for speed
2. Confirm speedup across **all** tested shapes, not just one
3. If speedup is marginal, move to the next structural strategy rather than re-tuning the same approach

## Key FlyDSL APIs

- Device kernel: `@flyc.kernel` | Host launcher: `@flyc.jit`
- Intrinsics: `flydsl.expr.rocdl` — MFMA, exp2, rcp, sched_barrier, sched_mfma
- Shared memory: `SmemAllocator` / `SmemPtr` from `flydsl.utils.smem_allocator`
- Types: `T.f16`, `T.bf16`, `T.f32`, `T.i32`, `T.vec(...)` from `flydsl.expr.typing`
- Buffer ops: `fx.rocdl.make_buffer_tensor`, `fx.make_copy_atom`
