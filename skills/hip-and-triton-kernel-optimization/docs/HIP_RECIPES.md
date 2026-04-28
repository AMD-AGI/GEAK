# HIP_RECIPES.md — per-signal HIP optimization patterns

Companion to [SKILL.md](../SKILL.md). Nine recipes, each linked to a
specific profiler signal from the
[signal-to-decision table](../SKILL.md#signal-to-decision-table). All
intrinsics target gfx950 (CDNA-4); most also work on gfx942 (CDNA-3).

Each recipe is structured:

- **Trigger** — the profiler signal and threshold.
- **Change** — the code-level pattern.
- **Expected impact** — qualitative.
- **Gotchas** — what bites.

## 1. Transposed LDS reads

**Trigger.** `LDSBankConflict / MemUnitStalled > 0.05` from PMC group
`b`. The roofline target is < 0.005 (any LDS conflict is sub-roofline);
a baseline HIP kernel loading scattered KV through LDS is often
0.05–0.15.

**Change.** Replace scattered `ds_read` patterns with the gfx950
transposed reads, and XOR-swizzle the LDS layout so each read hits all
32 banks.

```cpp
// BF16 PV path: 16 lanes × 4B per read, transposed so adjacent
// lanes hit adjacent banks.
auto v = __builtin_amdgcn_ds_read_b64_tr_b16(
    (const __local uint64_t*)(lds_base + swizzled_offset));

// FP8 QK path
auto k = __builtin_amdgcn_ds_read_b64_tr_b8(
    (const __local uint64_t*)(lds_base + swizzled_offset));
```

XOR-swizzle pattern that gives all-32-bank coverage for a 16-lane
transpose:

```cpp
const int lane = threadIdx.x & 0x3F;
const int slot = ...;  // your KV-tile slot index
const uint32_t swizzled_offset =
    base_offset ^ ((lane >> 3) << 4) ^ ((slot << 5) & 0x3FF);
```

**Expected impact.** Closes 30–60 % of the gap to roofline when LDS is
the dominant stall. Empirically this single change can move
`LDSBankConflict / MemUnitStalled` from ~0.12 → ~0.025 at low context.

**Gotchas.**

- The transposed-read intrinsics return packed 64-bit values. Cast to
  the right type before feeding to MFMA — use `__builtin_bit_cast` or
  reinterpret through a union, not C-style casts.
- Add a 4-byte stride pad on the PV LDS slot to avoid conflict with
  concurrent QK writes in a double-buffered layout.
- Swizzle the **write** side too. A read swizzle without a matching
  write swizzle just trades one conflict pattern for another.

## 2. Native FP8 MFMA + caller-side Q quantization

**Trigger.** PC-sample VMEM > 10 % combined with kernel doing
`fp8 → bf16` upcasts on load. The upcast burns VMEM bandwidth.

**Change.** Use native FP8 MFMA opcodes; require the caller to pre-
quantize Q to FP8.

```cpp
// QK matmul, FP8 inputs, fp32 accumulator, K=32 per opcode
auto qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
    q_fp8_packed, k_fp8_packed, qk_acc, /*cbsz=*/0,
    /*abid=*/0, /*blgp=*/0);
```

Caller side (host or fused upstream kernel):

```python
q_fp8 = q.to(torch.float8_e4m3fn)  # match the kernel's e4m3 path
# Pass q_fp8 into the kernel directly; do NOT dequantize on load.
```

**Expected impact.** Halves QK datapath bandwidth and eliminates the
FP8 → BF16 upcast VALU work. In the MLA project this drove the
v9k → v9l transition.

**Gotchas.**

- aiter's a8w8 MLA stores FP8 in a specific layout; check stride and
  page format match the production format before claiming a win.
- FP8 MFMA accuracy is fine for inference (cosine > 0.999 against fp32
  reference is typical) but always validate per
  [SKILL.md#accuracy-validation](../SKILL.md#accuracy-validation).
- e4m3 vs. e5m2: aiter and most attention paths use e4m3
  (`torch.float8_e4m3fn`). e5m2 has more dynamic range but lower
  precision; only use it if the workload demands it.

## 3. K=128 scaled FP8 MFMA

**Trigger.** `SQ_INSTS_MFMA` per wave < 50 % of theoretical peak
(MFMA pipe is starved, not saturated). Confirmed by Tier 3 PC
sampling showing MFMA share < 20 %.

**Change.** Use the K=128 scaled FP8 MFMA, which does 4× the work per
issue compared to the K=32 variant.

```cpp
auto acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_fp8_packed,    // 128 K elements per row
    b_fp8_packed,
    acc,
    scale_a, scale_b,    // dynamic scale factors
    /*opsel*/ 0, /*cbsz*/ 0, /*blgp*/ 0);
```

**Expected impact.** Doubles MFMA work-per-issue when MFMA was the
issue-rate bottleneck. Combined with hand-scheduling (recipe 6) this
is what unblocks the path from "MFMA pipe under-fed" to the
"wait-counter-bound" regime that any roofline-tracking attention
kernel ends up in (see [PROFILING.md](PROFILING.md) Tier 3 worked
example).

**Gotchas.**

- The scaled variant takes a per-block scale factor (`f8f6f4` = 8-bit
  A operand, 6-bit shared exponent, 4-bit B operand encoding). For
  uniform-scale workloads, set `scale_a = scale_b = 1.0`; for true
  block-scaling use the dispatcher's per-block scales.
- K=128 means 128-element K-loop tiles. Adjust `BLOCK_K`, LDS layout,
  and stride math accordingly. A common bug is using K=32 LDS strides
  with K=128 MFMA.

## 4. Persistent grid (CU-sized + atomic dispenser)

**Trigger.** PC sampling shows > 30 % of samples in the persistent
prologue region (Q-tile load, accumulator init, m_i/l_i seed) — i.e.
per-launch setup is not amortized across tiles.

**Change.** Promote the grid to CU-sized (~304 WGs on MI355X). Each
WG pulls work-tile tuples `(split, batch, head_group)` from a global
atomic counter and processes them sequentially. Peel the prologue work
to run only on the first tile per WG.

```cpp
__device__ int g_work_counter;  // initialized to 0 host-side per launch

__global__ void persistent_kernel(...) {
    // Per-WG persistent prologue: runs once.
    init_q_tile(q_lds, ...);
    init_softmax_state(...);

    while (true) {
        const int tile_id = atomicAdd(&g_work_counter, 1);
        if (tile_id >= TOTAL_TILES) return;

        const int split  = tile_id / (BATCH * HG);
        const int batch  = (tile_id / HG) % BATCH;
        const int hg     = tile_id % HG;

        process_tile(split, batch, hg, ...);
    }
}
```

**Expected impact.** Closes the gap that comes from per-launch
prologue work. This is the dominant source of remaining gap to the
roofline at long context lengths after recipes 1–3 land.

**Gotchas.**

- Grid size must equal CU count (or a multiple); too small and you
  underutilize the GPU, too large and you pay for the over-subscribed
  WGs to drain.
- The atomic counter must be reset per dispatch. Easiest pattern:
  zero it from the launcher before each `kernel<<<...>>>` call.
- Work-tile tuple ordering matters for cache locality. Order by
  `split` outer, `batch` middle, `head_group` inner — keeps the same
  KV-page hot in L2 across consecutive tiles.

## 5. Register-resident Q + softmax state

**Trigger.** VALU per wave > 3× algorithmic minimum in PMC group `c`
(count the VALU ops your algorithm actually requires; everything above
is redundant data movement). Most often this means Q tiles and softmax
accumulators (`m_i`, `l_i`) are being re-loaded from LDS every
K-iteration.

**Change.** Accept 1 WG/CU occupancy. Budget VGPRs aggressively:

- Full Q tile in registers (e.g. NHEAD × LK / 64 lanes per WG).
- `m_i` / `l_i` in registers per row.
- Output accumulator in registers (no LDS spill).

Use `__launch_bounds__` to lock occupancy:

```cpp
__launch_bounds__(NUM_THREADS, 1)   // 1 WG/CU
__global__ void kernel(...) { ... }
```

**Expected impact.** A 1 WG/CU register-budgeted layout (full Q tile +
softmax state in registers, ~512 VGPRs / ~160 KB LDS) typically lands
at 2× the throughput of a 2 WG/CU shared-LDS layout (~124 VGPRs /
~64 KB LDS) for decode-shaped attention, because the LDS round-trips
dominate at low context.

**Gotchas.**

- Register spilling. If `nvcc -Xptxas -v`-equivalent (`-Rpass-analysis=
  kernel-resource-usage` for hipcc) reports `VGPRs Spill: > 0`, you
  have lost the trade. Reduce tile size or refactor.
- Some compilers emit a register allocation pattern that artificially
  inflates VGPR count by 8–16 for callee-saved registers in
  device-side function calls. Inline aggressively
  (`__device__ __forceinline__`).
- 1 WG/CU is fragile under low-batch decode (b=1) — there is no
  second WG to hide latency. Recipe 6 (hand-scheduling) is mandatory
  to make this profitable.

## 6. Hand-scheduled inner loop

**Trigger.** PC-sample `s_waitcnt` > 25 %. The kernel is waiting on
synchronization, not arithmetic. The compiler scheduler is not
interleaving load and MFMA properly.

**Change.** Take control of the issue order and the wait-count
placement. Two key intrinsics:

```cpp
// Force a specific wait-count value (typically used to drop a single
// vmcnt or lgkmcnt without waiting on the others).
__builtin_amdgcn_s_waitcnt(0);                // wait on all
__builtin_amdgcn_s_waitcnt(0x0FFF);           // vmcnt=15, lgkmcnt=0

// Inform the scheduler about an issue-class barrier within a group.
// Mask: 0x08=MFMA, 0x02=VALU, 0x400=EXP, ...
// Count, group_id are integer hints to the LLVM scheduler.
__builtin_amdgcn_sched_group_barrier(0x08, 1, 0);  // 1 MFMA in group 0
__builtin_amdgcn_sched_group_barrier(0x02, 5, 0);  // 5 VALU in group 0
```

Pattern that works on gfx950:

```cpp
// Inner K-loop body
load_next_kv_tile_to_lds(...);                      // VMEM issue
__builtin_amdgcn_sched_group_barrier(0x02, 4, 0);   // 4 VALU
qk_mfma_step_0();                                   // MFMA
__builtin_amdgcn_sched_group_barrier(0x08, 1, 0);   // 1 MFMA
ds_read_next_q_chunk();                             // LDS
__builtin_amdgcn_sched_group_barrier(0x02, 4, 0);   // 4 VALU
qk_mfma_step_1();                                   // MFMA
// ... repeat 4-8 times per K-tile, then one waitcnt at PV entry ...
__builtin_amdgcn_s_waitcnt(0);                      // single drain
pv_mfma_phase(...);
```

**Expected impact.** This is the recipe that takes a kernel from
"recipes 1–5 applied, MFMA pipe fed but waitcnt-stalled" to
"roofline-tracking". For attention decode, expect the kernel to land
within ~10 % of the binding ceiling once this recipe is in.

**Gotchas.**

- Sched-group barriers are **hints** to the LLVM scheduler, not hard
  constraints. The scheduler can still re-order across them in some
  cases. Always validate with Tier 3 PC sampling that
  `s_waitcnt` share dropped.
- Manual `s_waitcnt(0)` over-syncs; only use it where you actually
  need all in-flight memory ops complete. Use the partial-count form
  (e.g. `vmcnt(N)`) when you only need VMEM drained.
- Match HipKittens' patterns where applicable — their
  `sched_barrier_pairs<Pairs, VALU_CNT, Group>` template encapsulates
  the most useful interleave (1 MFMA + N VALU per pair). See
  [HipKittens kernels/attn/gqa](https://github.com/HazyResearch/HipKittens/blob/main/kernels/attn/gqa/kernel.cpp).

## 7. K-split + dedicated reduce kernel

**Trigger.** Long-context decode at low batch — a single CU cannot
finish a long K-loop fast enough to keep the GPU busy.

**Change.** Split the K-dimension across multiple WGs and follow with
a dedicated reduce kernel that combines partial accumulators.

```cpp
// pick_k_splits is the dispatcher's contract with the reduce kernel
static int pick_k_splits(int ctx, int batch) {
    const int HG = NHEAD / BLOCK_H;
    const int BLOCK_N = 32;
    const int TOTAL_SLOTS = 512;       // CU count target
    const int K_MAX_SUPPORTED = 32;    // reduce dispatcher max
    int k_fill = std::max(1, TOTAL_SLOTS / std::max(1, batch * HG));
    int k_ctx  = std::max(1, ctx / BLOCK_N);
    int k = std::min(k_fill, k_ctx);
    int k_pow2 = 1;
    while (k_pow2 < k) k_pow2 <<= 1;
    return std::min(k_pow2, K_MAX_SUPPORTED);
}
```

**Expected impact.** Critical for low-batch decode. Without K-split,
b=1 decode at ctx=9000 leaves 95 % of the GPU idle.

**Gotchas.**

- The reduce dispatcher typically supports a fixed set of split
  factors (e.g. `{1, 2, 4, 8, 16, 32}`). `pick_k_splits` MUST cap and
  round to a supported value, or the reduce kernel will silently
  produce garbage.
- Per-split partial accumulators need fp32 storage even if the final
  output is bf16. Don't use bf16 partials — softmax-rescale ordering
  drift accumulates fast across splits.
- LSE (log-sum-exp) per partial must be tracked and combined in the
  reduce step. The reduce kernel is doing online-softmax merge across
  partials.

## 8. Reduce-kernel tuning

**Trigger.** Tier 1 timeline shows reduce kernel > 20 % of total
launch chain. (Or you've already done recipes 1–7 and the reduce is
now the longest stage.)

**Change.** Three patterns that compound:

1. **Vector loads** — read partial accumulators with
   `buffer_load_dwordx4` instead of per-element `global_load`.
2. **Wider workgroup** — 256 threads if your reduce was 128. The
   reduce is memory-bound; more threads = more outstanding loads.
3. **Branchless accumulation** — replace `if (k < K_SPLITS)` with a
   masked add that always issues. The branch was killing warp coherence.
4. **Persistent done-counter** — use an atomic counter incremented by
   stage-1 to signal "all my partials are ready"; reduce kernel
   spins on it instead of the host syncing.

```cpp
__device__ int g_done_count;     // initialized 0 per launch

// stage 1 finishes
atomicAdd(&g_done_count, 1);

// reduce kernel waits
while (atomicAdd(&g_done_count, 0) < EXPECTED) { /* spin */ }
```

**Expected impact.** Recipes 1–4 in this section together typically
trim a non-vectorized reduce by 30–50 %. For attention decode, that
moves the reduce stage from "Tier 1 dominant" back to a fraction of
the partial-kernel time.

**Gotchas.**

- Spin-waiting on a done-counter requires careful WG ordering. If
  reduce starts before stage-1 launches, it deadlocks on `< EXPECTED`.
  Easiest fix: launch reduce after stage-1 with a stream dependency,
  not a global sync.
- Don't fuse reduce into stage-1 unless you've validated the fused
  kernel does not destabilize the MFMA accumulator schedule. The MLA
  project tried this and saw register spilling that net-regressed.

## 9. `buffer_load_dwordx4` with hand-built v-descriptor

**Trigger.** PMC group `d` shows L2 hit % is high (e.g. > 70 %) but
`FetchSize` is more than 1.2× the algorithmic minimum bytes
(working-set × tile-passes). The L2 is catching redundant fetches that
the algorithm does not require.

**Change.** Replace `global_load` with `buffer_load_dwordx4` against
a v-descriptor (BD resource record) built once in SGPRs.

```cpp
// Build the BD once in the persistent prologue.
struct alignas(16) bd_t { uint32_t base_lo, base_hi, num_records, config; };
__device__ bd_t make_bd(const void* base, uint32_t bytes) {
    return bd_t{
        (uint32_t)((uintptr_t)base),
        (uint32_t)((uintptr_t)base >> 32),
        bytes,
        0x00020000u  // dword-x4 stride, default config
    };
}

// Use it for KV loads in the inner loop.
auto x = __builtin_amdgcn_raw_buffer_load_b128(
    *reinterpret_cast<int4*>(&bd),
    /*voffset=*/byte_offset,
    /*soffset=*/0,
    /*aux=*/0);
```

**Expected impact.** Eliminates per-lane-predicated redundant fetches
that `global_load` issues for OOB-protection. Empirically this trims
`FetchSize` by 20–30 % toward the algorithmic minimum, with the bigger
gains at long context where the OOB-tail is a smaller fraction of the
total.

**Gotchas.**

- The BD resource record format is gfx-version-specific. The
  `0x00020000u` config above is gfx950; check
  `Vega ISA Reference Guide` or `CDNA-3 ISA` for gfx942.
- Free bounds-protection: the BD has a `num_records` field; loads
  past it return zeros automatically. This handles short-ctx tails
  without explicit branches.
- Only use this for *predictable* access patterns (KV cache loads,
  fixed strides). For irregular access, `global_load` is fine.

## Recipe order (when applying multiple)

If a kernel exhibits multiple signals, apply in this order:

1. **Recipe 1 (LDS transposed reads)** — fixes LDS bank conflicts first.
2. **Recipe 2 (native FP8 MFMA)** — fixes datapath bandwidth.
3. **Recipe 3 (K=128 scaled MFMA)** — feeds the MFMA pipe.
4. **Recipe 9 (`buffer_load_dwordx4`)** — fixes HBM redundancy.
5. **Recipe 7 (K-split)** — fixes low-batch decode underutilization.
6. **Recipe 5 (register-resident state)** — only AFTER 1–4 land.
7. **Recipe 6 (hand-scheduled inner loop)** — only after 5 lands and
   `s_waitcnt` is the remaining dominant signal.
8. **Recipe 4 (persistent grid)** — last; pulls the per-launch
   prologue out, but only matters once everything else is tuned.
9. **Recipe 8 (reduce-kernel tuning)** — independent of the partial
   kernel; do whenever Tier 1 says the reduce is > 20 % of total.

Re-profile after each recipe lands — the targeted signal should move
materially. If it does not, revert and reconsider; the diagnosis was
wrong.
