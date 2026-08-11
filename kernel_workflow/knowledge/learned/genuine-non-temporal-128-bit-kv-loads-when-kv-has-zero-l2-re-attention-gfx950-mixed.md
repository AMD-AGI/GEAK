---
key: paged/ragged attention over a read-once bf16 KV cache on gfx950 (HIP body, frozen QKV inner loop) — cache-bypass hinting on the KV stream
type: lever
confidence: ★★
effect: 1.066x weighted geomean vs frozen baseline, all 9 cases improved, no regression: +6.5..+7.6% on the eight short-context signature cases, +2.6% on the long-context (S=512, ctx=4096) case; ISA-confirmed
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: genuine-non-temporal-128-bit-kv-loads-when-kv-has-zero-l2-re-attention-gfx950-mixed
description: Read-once paged-KV attention: one __builtin_nontemporal_load on a native 128-bit vector emits a real nt dwordx4; the shipped nt helper drops nt on gfx950.
keywords: ['non-temporal', 'nontemporal-load', 'kv-cache', 'l2-reuse', 'vectorized-load', 'memory-bound', 'attention', 'gfx950', 'isa-check']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention
regime: mixed
layer: learned
lifecycle: active
---
# genuine non-temporal 128-bit KV loads when KV has zero L2 reuse
- lever: On a KV stream that is read once with no L2 reuse, issue the load as a SINGLE __builtin_nontemporal_load over a native 128-bit vector type (e.g. an ext_vector int x4) so the compiler emits one global_load_dwordx4 ... nt.
- apply: Replace the library 16-byte nt helper in the KV load path; that helper re-vectorizes four scalar nt loads and silently drops the nt bit on this arch. L2-level change, no touch to the frozen inner body.
- verify: Disassemble and confirm the nt modifier survives on the dwordx4; then re-run the frozen-baseline isolated A/B per case plus oracle parity — a real nt lifts every case here, so a mixed sign means it did not engage.
- pitfall: speedup absent though the source says nontemporal -> the vendor multi-scalar nt helper compiles to plain loads on this arch -> emit the single 128-bit intrinsic and check the ISA rather than the source.
- caution: The SIGN of nt / evict-first depends on L2 reuse of the streamed set — also verify reuse before assuming it is uniformly favorable: on a sibling decode kernel the same bit flipped +7% at one context length and -7% at another as the streamed-set size changed.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, round 1 direction d0, 2026-08-11
