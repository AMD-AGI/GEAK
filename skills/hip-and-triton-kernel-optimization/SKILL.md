---
name: hip-and-triton-kernel-optimization
description: >-
  Profiler-driven optimization playbook for AMD CDNA-3 / CDNA-4 GPU kernels
  (MI300X, MI325X, MI355X — gfx942 / gfx950) covering attention, GEMM, and
  MoE workloads. Use when optimizing a HIP or Triton kernel on AMD ROCm;
  when the user mentions rocprofv3, PMC counters, MFMA utilization, LDS
  bank conflicts, s_waitcnt stalls, or the "Triton ceiling"; when
  computing a roofline target for a kernel; when validating accuracy with
  run_perftest / aiter or against an fp32 torch reference; or when
  scoping a HipKittens port. Encodes the three-tier rocprofv3
  methodology, a roofline-anchored signal-to-decision table, nine HIP
  optimization recipes, and the build / accuracy / benchmarking patterns.
---

# AMD Kernel Optimization

Profiler-driven optimization for AMD CDNA-3 / CDNA-4 kernels. Methodology
generalizes to attention, GEMM, and MoE. The skill assumes the kernel
under optimization has already been selected (by an upstream workload
analysis or the user); the goal here is to take that one kernel and
drive it toward its roofline.

## Roofline reference

The performance bar is the **hardware roofline**, not an external
reference kernel. A roofline is always available (you have the spec
sheet) whereas vendor or hand-tuned kernels often are not. Compute the
binding ceiling once per kernel:

```
peak_compute_TFLOPS = arch_peak_for_MFMA_opcode    # opcode-specific
peak_HBM_TBs        = arch_HBM_bandwidth           # device-specific
arith_intensity     = total_ops / total_bytes_HBM_traffic
binding_ceiling     = min(peak_compute_TFLOPS,
                          peak_HBM_TBs * arith_intensity)
target_runtime_us   = total_ops / binding_ceiling
```

Use the MFMA peak that matches your opcode (K=32 fp8/bf16, K=128 scaled
fp8, etc.); the ISA reference lists per-opcode throughput. Reference
HBM bandwidth: MI300X / MI325X ≈ 5.3 TB/s, MI355X ≈ 8.0 TB/s.

From the binding ceiling derive the per-counter reference points used
throughout this skill:

| Counter | Roofline-derived reference |
|---|---|
| Kernel runtime | `total_ops / binding_ceiling` |
| MFMA utilization | `(SQ_INSTS_MFMA × mfma_cycles) / GRBM_GUI_ACTIVE` ≥ 50 % |
| FetchSize | Algorithmic minimum bytes (working-set × tile-passes) |
| `LDSBankConflict / MemUnitStalled` | < 0.005 (any LDS conflict is sub-roofline) |
| PC-sample `s_waitcnt` share | < 15 % |
| VALU per wave | Algorithmic minimum (count the VALU ops the algorithm requires; everything above is shuffling) |
| Persistent prologue PC share | < 30 % of total samples |

If you happen to have a tuned reference kernel (vendor ASM, vendor HIP,
a previous best variant), use it as a *sanity check* on the roofline
computation, never as the bar. The bar is the roofline.

## Three-tier profiling methodology

Each tier answers a different question. Run them in order; do not skip
to a tier until the previous one is conclusive.

### Tier 1 — Perfetto timeline (`rocprofv3 --sys-trace`)

Question: **where is time going across kernels in one step?**

Output: a `.pftrace` file plus per-kernel CSVs (`*_kernel_stats.csv`).
Decision lever: identify the dominant stage (partial vs. reduce, gemm
vs. epilogue, all-reduce vs. compute) so you do not waste effort
optimizing a stage that is already fast.

Compare each stage's measured runtime to its **roofline-derived target
runtime** (compute the binding ceiling for each stage independently —
they often have different arithmetic intensities). Optimize stages that
are sub-roofline; leave stages that already track their roofline alone.
A common mistake is rewriting a stage that is already at its ceiling
because it is the largest absolute contributor.

### Tier 2 — PMC counter sweep (`rocprofv3 --pmc`)

Question: **which subsystem is stalling?**

gfx950 cannot fit a useful counter set in one HW pass, so shard into
four groups and run each separately:

| Group | Counters | Tells you |
|---|---|---|
| a | `SQ_WAVES GRBM_GUI_ACTIVE VALUBusy SALUBusy` | Wave count, kernel duration, ALU occupancy |
| b | `LDSBankConflict MemUnitStalled FetchSize WriteSize` | LDS pressure, HBM traffic |
| c | `SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_WAVES` | Instruction mix per wave |
| d | `SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR TCC_HIT_sum TCC_MISS_sum` | VMEM mix, L2 hit rate |

Always sweep across ≥ 3 contexts (e.g. ctx ∈ {1000, 4000, 9000}) so you
can tell short-context from long-context bottlenecks apart. See
[PROFILING.md](docs/PROFILING.md) for full invocation.

### Tier 3 — PC sampling (`rocprofv3 --pc-sampling-method host_trap`)

Question: **which instruction is the kernel sitting on?**

Use 50 µs interval (`--pc-sampling-interval 50us`). Annotate the
disassembly and bucket samples by instruction class.

Key insight that recurs across decode-shaped attention and short-K
GEMMs: even at roofline, **MFMA share is often well under 10 %** while
`s_waitcnt` is over 25 %. The kernel is **wait-counter-bound, not
arithmetic-bound**, so adding MFMA throughput cannot help — only
cutting staging traffic and tightening the load/MFMA interleave can.
Read the histogram before changing the kernel.

### Tier 4 — ATT thread trace (often unavailable)

Full per-CU per-wave-slot trace via `rocprofv3 --att`. Requires
`librocprof-trace-decoder.so`, which is not always installed in the
container. If absent, substitute per-dispatch `SQ_INSTS_*` PMC counters
(Tier 2 group c) for the per-wave instruction histogram. Do not block
optimization on Tier 4 access.

## Signal-to-decision table

This is the centerpiece. Map each profiler signal directly to the
kernel change it justifies; do not change code without a signal.

| Signal | Threshold | Action | Recipe |
|---|---|---|---|
| `LDSBankConflict / MemUnitStalled` | > 0.05 (roofline target < 0.005) | Swap LDS layout for transposed reads | [HIP_RECIPES.md#1](docs/HIP_RECIPES.md) — `ds_read_b64_tr_b16/_b8` |
| VALU per wave | > 3× algorithmic minimum | Redundant data shuffling — promote state to register-resident | [HIP_RECIPES.md#5](docs/HIP_RECIPES.md) |
| MFMA utilization | < 50 % of opcode peak | MFMA pipe starved — switch to larger MFMA opcode | [HIP_RECIPES.md#3](docs/HIP_RECIPES.md) — K=128 scaled FP8 |
| PC-sample `s_waitcnt` | > 25 % | Wait-counter-bound — do **not** add MFMA, cut staging traffic | [HIP_RECIPES.md#6](docs/HIP_RECIPES.md) — hand-scheduled inner loop |
| FetchSize ÷ algorithmic-minimum | > 1.2 with high L2 hit rate | Redundant fetches under per-lane predication | [HIP_RECIPES.md#9](docs/HIP_RECIPES.md) — `buffer_load_dwordx4` |
| Reduce kernel share of total | > 20 % | Rewrite reduce stage | [HIP_RECIPES.md#8](docs/HIP_RECIPES.md) — vector loads, persistent done-counter |
| Persistent prologue PC samples | > 30 % of total | Per-launch prologue work not amortized across tiles | [HIP_RECIPES.md#4](docs/HIP_RECIPES.md) — persistent grid |

Order matters: address LDS pressure before tuning MFMA, address VMEM
redundancy before tuning the reduce.

## The Triton ceiling pattern

When PC sampling on a Triton kernel shows **25–30 % `s_waitcnt`** and
**MFMA utilization at 15–20 % of opcode peak** (well below the 50 %
target derived from the roofline), you have hit the AMD Triton
scheduling ceiling. Do not keep tuning Triton.

Root cause: AMD Triton's MLIR → LLVM → gfx950 path uses a generic
instruction scheduler that does not co-issue MFMA with VALU/memory ops,
even though the hardware can issue them in the same cycle when they
target different functional units. LLVM treats them as sequential, so
the MFMA pipe sits idle waiting for loads.

Action: pivot to HIP with `__builtin_amdgcn_mfma_*` intrinsics and
hand-schedule the load/MFMA interleave (see
[HIP_RECIPES.md#6](docs/HIP_RECIPES.md)). Expected recovery: 70–90 % of
the gap to the roofline in 1–2 engineering-weeks. Do not wait on a
multi-quarter upstream LLVM scheduler fix unless you have nothing else
to ship.

## HIP optimization recipes (summary)

Each recipe maps to a specific signal and is detailed in
[HIP_RECIPES.md](docs/HIP_RECIPES.md):

1. **Transposed LDS reads** — `ds_read_b64_tr_b16/_b8` with XOR-swizzled
   offsets. Triggered by `LDSBankConflict / MemUnitStalled > 0.05`.
2. **Native FP8 MFMA + caller-side Q quantization** — `mfma_f32_16x16x32_fp8_fp8`.
   Triggered by VMEM share > 10 % when the kernel is upcasting FP8 → BF16.
3. **K=128 scaled FP8 MFMA** — `mfma_scale_f32_16x16x128_f8f6f4`.
   Triggered by MFMA pipe starvation (recipe is bigger MFMA, not more).
4. **Persistent grid** — CU-sized grid with atomic work-tile dispenser.
   Triggered by PC samples in the persistent-prologue region.
5. **Register-resident Q + softmax state** — accept 1 WG/CU, budget VGPRs.
   Triggered by VALU/wave > 3× algorithmic minimum and high LDS round-trips.
6. **Hand-scheduled inner loop** — `__builtin_amdgcn_s_waitcnt` +
   `sched_group_barrier`. Triggered by `s_waitcnt > 25 %`.
7. **K-split + dedicated reduce kernel** — split-K factor capped at the
   reduce dispatcher's max, power-of-2 rounded.
8. **Reduce-kernel tuning** — vector loads, wider workgroup, branchless
   accumulation, persistent done-counter.
9. **`buffer_load_dwordx4` with hand-built v-descriptor** —
   `__builtin_amdgcn_raw_buffer_load_b128`. Triggered by FetchSize >
   1.2× algorithmic minimum despite high L2 hit rate.

## Accuracy validation

Always validate against an **fp32 torch reference**, not just
kernel-vs-kernel. A kernel-vs-kernel cosine of 0.99 tells you the two
kernels agree; it does not tell you either is correct.

### Reference recipe

Mirror `test_triton_mla.py`'s `ref_masked_attention` for FP8 attention:

```python
def fp32_ref(q_fp8, kv_fp8, sm_scale):
    # Dequantize to fp32. Use q_scale = kv_scale = 1.0 to match the
    # kernel's a8w8 production format.
    q = q_fp8.to(torch.float32)
    kv = kv_fp8.to(torch.float32)
    attn = torch.einsum("bhd,bckd->bhck", q, kv) * sm_scale
    m = attn.max(-1, keepdim=True).values
    w = torch.exp(attn - m)
    w = w / w.sum(-1, keepdim=True)
    v = kv[..., :v_head_dim]
    return torch.einsum("bhck,bckd->bhd", w, v)
```

For non-attention kernels, the same principle applies: dequantize all
low-precision inputs to fp32 and run the reference math in fp32.

### Metrics

Report all of: cosine, max_abs, mean_abs, RMSE, rel_RMSE.

```python
def stats(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    diff = a - b
    return dict(
        cos=float((a @ b) / (a.norm() * b.norm() + 1e-30)),
        max_abs=float(diff.abs().max()),
        mean_abs=float(diff.abs().mean()),
        rmse=float((diff ** 2).mean().sqrt()),
        rel_rmse=float((diff ** 2).mean().sqrt() / (a.abs().mean() + 1e-30)),
    )
```

### Cell sweep

Test at minimum: `(b=1, ctx=low)`, `(b=4, ctx=low)`, `(b=4, ctx=mid)`,
`(b=4, ctx=high)`. Both small and large batch, both low and high
context. Single-cell accuracy is misleading because rounding noise
typically grows with context length.

### Bar

Cosine > 0.99 against the fp32 reference. This matches aiter's
`cal_diff < 3e-2` for FP8 paths.

### Bonus pattern

If a tuned reference kernel (vendor or in-house) is available, also
report `kernel-vs-fp32-ref` for **both** your kernel **and** the
reference, side by side. You may discover your kernel is more accurate
than the reference; when that happens the speedup is a free win, not a
precision compromise — and that should be a slide in the deck.

## Benchmark harness

Use `aiter.test_common.run_perftest`:

- torch.profiler device-time over 101 iterations after warmup
- argument rotation (different inputs per iter) to thrash L2
- no CUDA graphs (these hide launch overhead in a way that does not
  match production dispatch)

```python
from aiter.test_common import run_perftest

us, _ = run_perftest(my_kernel, *args, num_iters=101, num_warmup=20)
```

**Do not** measure with one-shot subprocess per ctx (a process restart
per data point). That methodology produces noisy curves dominated by
JIT and dispatch overhead, not kernel time. Same kernel under
`run_perftest` typically lands at 30–50 % of the runtime that one-shot
subprocess methodology reports — the difference is overhead, not the
kernel.

Apples-to-apples requirements:

- Same harness for every kernel under test
- Same dims (NHEAD, page_size, qseqlen, batch, dtype) across kernels
- Same warmup / iteration count
- Note in plots whether CUDA graphs are used or not

## Build pattern (HIP extensions)

`torch.utils.cpp_extension.load` with per-variant `build_directory`:

```python
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"  # MUST be set before importing torch

import torch
from torch.utils.cpp_extension import load

ext = load(
    name=f"my_kernel_{variant}_ext",
    sources=[f"{variant}_kernel.cu", f"{variant}_launcher.cpp"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "--offload-arch=gfx950", "-std=c++17",
        "-munsafe-fp-atomics", "-ffast-math", "-fno-math-errno",
        "-fomit-frame-pointer",
        "-mllvm", "-amdgpu-coerce-illegal-types=1",
    ],
    build_directory=f"build_{variant}",
    is_python_module=True,
)
```

Key gotchas:

- **Set `PYTORCH_ROCM_ARCH` before importing torch.** Otherwise hipcc
  compiles for all default archs, overriding `--offload-arch=gfx950`,
  and you will get cryptic errors like "`__builtin_amdgcn_*` needs
  target feature `gfx950-insts`".
- **Per-variant `build_directory`** lets multiple variants coexist in
  one process. Each `.so` is loaded with `RTLD_LOCAL` automatically, so
  duplicate `extern "C"` symbols across variants do **not** collide.
  You do not need to rename symbols per variant.
- **`.cpp` files passed to `load` get C++ flags only.** If your launcher
  uses HIP intrinsics or `::max`/`::min`, copy or symlink it as `.cu`
  so hipcc applies CUDA flags.

## NHEAD-aware porting (across tensor-parallel degrees)

Different TP degrees give different `NHEAD` (e.g. K2/TP1 has NHEAD=128,
K2/TP4 has NHEAD=16). This changes:

- `BLOCK_H` (Q tile rows = NHEAD when GQA-ratio = NHEAD)
- `HEAD_GROUPS` (NHEAD / BLOCK_H)
- `K_SPLITS` cap (limited by reduce dispatcher's max)
- Grid shape

Common bug: NHEAD-dependent constants hardcoded in kernel body cause
silent OOB at smaller NHEAD (e.g. `constexpr int NHEAD = 128` plus
indexing assumptions that scale with NHEAD will memory-fault when the
kernel is re-instantiated at NHEAD = 16). Fix: parameterize all
NHEAD-dependent constants and re-validate accuracy after porting.

`pick_k_splits` pattern that is safe across NHEAD:

```cpp
static int pick_k_splits(int ctx, int batch) {
    const int HG = NHEAD / BLOCK_H;
    const int BLOCK_N = 32;
    const int TOTAL_SLOTS = 512;          // CU count target
    const int K_MAX_SUPPORTED = 32;       // reduce dispatcher max
    int k_fill = std::max(1, TOTAL_SLOTS / std::max(1, batch * HG));
    int k_ctx  = std::max(1, ctx / BLOCK_N);
    int k = std::min(k_fill, k_ctx);
    int k_pow2 = 1;
    while (k_pow2 < k) k_pow2 <<= 1;       // round up to power of 2
    return std::min(k_pow2, K_MAX_SUPPORTED);
}
```

## HipKittens as a parallel track

[HipKittens](https://github.com/HazyResearch/HipKittens) is Stanford
HazyResearch's tile DSL for AMD CDNA-3/4. It is registered as an AITER
backend and worth considering as a **parallel** track to a hand-coded
HIP kernel.

Validation point: on MI355X for standard GQA forward (B=16, H=64,
H_KV=8, N=2048, D=128, BF16) HipKittens lands within ~4 % of an
AITER-tuned HIP flash_attn while using roughly 6× fewer source lines.

When to consider:

- Kernel shape matches HK's existing zoo: BF16/FP8 GEMM, GQA fwd/bwd
  at head_dim 64/128, rotary, fused layernorm.
- Or a small parametric tweak of the above (different B, H, N, D within
  the same family).

When **not** to consider:

- Decode-shaped kernels (qseqlen=1) — HK's existing attention is forward-
  pass-over-full-N; the grid `(ATTN_H, ATTN_N/Q_BLOCK_SIZE/NUM_WARPS, B)`
  collapses at qseqlen=1.
- MLA — head_dim=576 = 128·4 + 64 does not tile cleanly into HK's
  128-wide subtiles.
- Paged KV — HK does not ship paged-attention.

Risk: HK's reference GQA is still 591 lines of explicitly-scheduled code
with `sched_barrier_pairs<>` templates per cluster. HK abstracts the
**primitives** (tile types, mma, sched-group barriers); you still
hand-author cluster-level scheduling. The productivity win is real but
it is 6× LOC, not "10-line MLA".

## Communicating results

For exec-facing material:

- **Story-arc plot.** Kernel evolution v1 → vN with milestone
  annotations. One line per version, distinct color, milestone text in
  a legend or table. Plot the **roofline target** as a horizontal
  reference line so the audience sees how close each variant gets.
- **Zoom plot.** When a noisy or much-slower baseline compresses the
  kernel-vs-kernel band into the bottom 10 % of the y-axis, drop the
  baseline in a separate "zoom" plot so the deltas between optimized
  variants are readable. Cap the y-axis at ~1.12× the highest sample
  in the zoomed view.
- **Accuracy slide.** Always include a slide proving the speedup is
  not a precision compromise — fp32-reference cosine numbers for the
  new kernel (and the reference, if one is available), side by side.
- **Annotation placement.** Park summary callouts in plot whitespace
  using axes-fraction coordinates (e.g.
  `xytext=(0.97, 0.06), textcoords="axes fraction"`), not over the
  data band. Use a curved arrow to point from the callout to the
  data point.

## Workflow checklist

For a new optimization task, copy this and track progress:

```
- [ ] Step 1: Compute roofline (binding ceiling + per-counter targets)
- [ ] Step 2: Tier 1 timeline — identify dominant stage
- [ ] Step 3: Tier 2 PMC sweep — identify stalled subsystem
- [ ] Step 4: Tier 3 PC sampling — identify instruction class
- [ ] Step 5: Map signals to recipes (signal-to-decision table)
- [ ] Step 6: Implement smallest signal-driven change first
- [ ] Step 7: Validate accuracy against fp32 reference (cell sweep)
- [ ] Step 8: Bench through run_perftest with consistent harness
- [ ] Step 9: Re-profile — confirm the targeted signal moved
- [ ] Step 10: Iterate from Step 5 with the next-highest signal
```

Stop when (a) measured runtime is within 10 % of the roofline target,
(b) PC sampling shows `s_waitcnt < 15 %`, or (c) every remaining signal
is at or below its roofline-derived reference.

## Additional resources

- [PROFILING.md](docs/PROFILING.md) — full rocprofv3 invocations, counter
  reference, ATT fallback details.
- [HIP_RECIPES.md](docs/HIP_RECIPES.md) — per-recipe code-level patterns
  with intrinsics, expected impact, and gotchas.
