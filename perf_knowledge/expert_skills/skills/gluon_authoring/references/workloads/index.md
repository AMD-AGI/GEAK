# Workload router — archetype -> pages

`kernel_workflow/scripts/kernel_tools/hw_budget.py --workload <archetype>` prints the row below for the
archetype you name (the `## <key>` sections are that tool's vocabulary; its selftest checks them). The general
entry index is [`../../reference.md`](../../reference.md). The archetype follows the body, not the
file name: a `..._attention_...` kernel with `q_len == 1` is a gather; a GEMM whose operand assignment is
data-dependent is `moe`. Rows are gfx950-first; gfx942 downgrades (ISA / spec, and Triton 3.8.0 for the async width): no
`ds_read_b64_tr`, no scaled MFMA, async direct-to-LDS 32-bit only (sync staging is the default there),
64 KiB LDS / 32 banks, FNUZ fp8.

Generic hardware and language facts live in the KB and are not repeated in these pages:
`perf_knowledge/hardware/cdna4_mi350/` (arch, memory, peak tables, block-scale matrix core),
`perf_knowledge/hardware/shared/wavefront_simd_vgpr_agpr.md`, `perf_knowledge/languages/gluon/`,
`perf_knowledge/languages/triton_amd/` (codegen, ISA checks, knobs) and the per-operator tuning
cards under `perf_knowledge/operators/`.

## Before any bound claim

1. **Does a roofline apply?** A warm working set inside the memory-side last level (MALL / L2) can void
   the memory floor; the tell is a measured time below it (`../hardware/roofline-models.md`).
2. **Which dtype's ridge.** Per dtype and SKU (`../hardware/amd-cdna4-skus.md`, from `sku.json`);
   `hw_budget.py --brief` recomputes it per call.
3. **Which counter caliber.** `MfmaUtil` / `SQ_VALU_MFMA_BUSY_CYCLES` **do** count scaled MFMA on gfx950 (rocprofv3 1.3.5, Triton 3.8.0); the per-format op counters split by operand format — fp8 x fp4 under `SQ_INSTS_VALU_MFMA_MOPS_F8`, fp4 x fp4 under `SQ_INSTS_VALU_MFMA_MOPS_F6F4` — so read the op counter for the formats you issue
   (`../hardware/bound-class-signals.md ## Rules (necessary-not-sufficient)`).

## Before any correctness-gated lever

Read the oracle's **type and threshold** together: which gate is stricter for a low-precision path
depends on both (`../tile-programming/low-precision.md`). `linear_attention` compounds error across
steps; on `collective` a protocol that passed every launch you ran is what a broken one looks like.

Read across the kernel boundary before blaming this kernel: **a producer that emits both a wide and a
quantized form while the interface carries only the wide one makes the consumer quantize twice** (each
kernel is correct; no single-kernel profile shows it). On gfx942 an OCP-e4m3 producer feeding an FNUZ
consumer is a **silent** numeric mismatch. A host-side dependence (`.item()`, a host-read shape or trip
count) is correct eager and **wrong under graph capture** (the value is frozen into the replay). When the
cause and the removal site sit on opposite sides of the boundary, record both sites; it is not
solvable inside this kernel alone.

## gemm

`gemm.md` (tiles by regime: small-M / mid-M / large; decode fill / split-K / reduce choice:
`gemm-small-m.md ## Split-K decision rule (small-M GEMM)`; beyond the hot loop: XCD / L2 grouping;
preshuffled / block-scaled). Overlap: the async ring of `../tile-programming/pipeline.md` (gfx942: sync
staged). In Gluon on Triton 3.8.0 / gfx950, fp8 at `[16,16,128]` / `[32,32,64]` and all fp4 go through
`cdna4.mfma_scaled`, and `cdna4.mfma` takes fp8 only at `[16,16,32]` / `[32,32,16]` (compile-checked; the
ISA also has unscaled `v_mfma_f32_16x16x128_f8f6f4` / `32x32x64_f8f6f4`, which Gluon does not expose
through `cdna4.mfma`) (`../gluon/matrix-reference.md`). Plain `tl.dot(fp8)` reaches the
K=64 scaled-family instruction through a scale-less `tt.dot_scaled`, so an fp8 decode port to Gluon can be an
instruction downgrade. Decode / paged shapes: `../pitfalls/negative-patterns.md` has a quick-reject check.

## attention

The scenario picks the section: softmax prefill (MHA/GQA) is `attention.md ## Prefill levers by signal (gfx950, Triton 3.8.0, bf16 varlen prefill)`; a DSA/MQA indexer (logits, no softmax, no PV) is not flash attention
(`attention.md ### DSA indexer (MQA logits): not flash attention`; decode: `attention.md ### DSA indexer decode (paged MQA logits): measured levers`);
sparse MLA: `attention.md ### Sparse MLA (DSA): measured levers`; decode / split-KV: `attention.md ## Decode (split-KV): a structure, not a config knob`.
Structure: `flash-attention-structural-insights.md`; low precision: `attention-lowprec.md`. Wave-level gate before
ring depth: `../tile-programming/warp-pipeline.md`. This pack positions re-injecting plain's pipeliner
as a diagnostic rather than the main route (`../gluon/pipeline/reinjection.md`).

## attention_bwd

`attention.md` (backward pass): the reduction lands on another axis and recompute is decided before the
tile; the extra operand forms a third commit chain. Register budget: `../tile-programming/slicing.md`.

## moe

`moe.md`: typing the kernel (fusion boundary, quantization form, shape band) frames the rest; tokens-per-expert is the
bucketing variable; small-M is reached per expert. Bound: the byte model is expert weight bytes actually
touched (`E_touched`), not tokens. LDS `.gather` / `.scatter` 3.7.0+, LDS atomic scatter 3.8.0+ (tagged
sources checked). The routing
(topk / sort) kernel is `## routing`, not this row.

## norm

`reduction-elementwise.md` (group-wise quantization stages). The CDNA4 hardware scale group is **32**; 128 is
a software convention, NVFP4 uses 16. Usually no ring: whether the row is register-resident decides whether
the absmax costs a second pass. No inline-asm site is documented for this row.

## reduction

`reduction-elementwise.md`. A dot-free loop has no anchor for re-injection, so overlap comes from
authored staging (register prefetch or an async ring) or from occupancy. Declare `--flops-engine valu` (no matrix rate).

## elementwise

`reduction-elementwise.md`. Bandwidth decision; fusing producer/consumer removes a round trip, while a
ring only overlaps it. Declare
`--flops-engine valu`.

## scan

Mechanism row: in the kernels covered, a scan sits inside another archetype (linear-attention decay, MoE
bucketing), which supplies the tile and bound. On Triton 3.8.0 `gl.associative_scan` takes no layout
kwarg (signature checked) and its element-wise combine cannot express a chunk-level matrix scan (`../gluon/appendix-api.md`, `linear-attention.md`).

## gather

`prologue.md` and `../gluon/layout-reference.md` (`gl.gather` is a warp-local lane exchange only when the
layouts allow it; otherwise it falls back to LDS scratch sized by the whole source). Paged index chains:
`../pitfalls/negative-patterns.md`.

## linear_attention

`linear-attention.md` (chunked GLA / GDN / delta rule, state in registers). No byte model: declare
`--tensors` and `--flops --flops-engine`.

## collective

`collective.md` (sem / scope, cooperative launch, graph replay). No model: scope the measurement to the
local work.

## prologue

`prologue.md` (QK-norm, RoPE, paged KV write, per-token quant): no loop, so the pipeline layer has nothing to
overlap; vector width and launch geometry are the main levers.

## routing

MoE metadata kernel (topk, counting sort / prefix sum, expert offsets): `moe.md` bucketing and topk
sections. Typically bound by the serial sort / scan chain, not bytes; declare `--tensors` and
`--flops --flops-engine valu`.

## When no row fits

Name the closest archetype for `hw_budget.py` and declare `--tensors` / `--flops`; route reading by the
mechanism through [`../../reference.md`](../../reference.md).
