---
title: FlyDSL kernel debugging
kind: language
gens: [gfx942, gfx950]
updated: 2026-10-16
source_commit: AMD-AGI/GEAK@c0a1f937
---

# Debug a FlyDSL kernel

Use the actual reference and tolerance. Symptom size alone cannot distinguish a
layout bug, stale state or rounding difference. Keep source/configuration and
input history bound to the failing path.

| Symptom | Useful check |
|---|---|
| NaN/Inf | Uninitialized values, all-masked softmax, zero normalization sums, conversion/scale overflow |
| Zero/missing output | Output address/strides, written partition slots, missing stages; sentinel buffers can reveal unwritten regions |
| Large mismatch | Layout and MFMA operand ownership, complete partition coverage, reduction indexing |
| Small mismatch | Quantization granularity, scales, accumulation and rounding order; keep the declared tolerance |
| Hang | Loop progress, out-of-bounds access, barriers reached uniformly by participating workgroup threads |

## Numerical and partition triage

For an attention contract that assigns zero contribution to an all-masked row,
use the existing typed exponential and `arith.select` operations. Here `valid`
is the token mask, `zero`/`one` are matching typed constants, and `exp` denotes
the kernel's existing exponential:

```python
has_values = row_max > NEG_INF
active = valid & has_values
safe_max = arith.select(has_values, row_max, zero)
safe_score = arith.select(active, score, safe_max)
p = arith.select(active, exp(safe_score - safe_max), zero)
safe_sum = arith.select(running_sum > zero, running_sum, one)
output = numerator / safe_sum
```

Invalid lanes contribute zero to both numerator and sum. A valid `score=-inf`
with a finite row maximum still yields zero probability. Do not substitute this
convention for a reference that requires NaN propagation or different all-`-inf`
behavior; diagnose unexpected NaN/+Inf inputs separately.

- **Find unwritten slots:** fill an intermediate with a recognizable sentinel,
  launch the diagnostic, synchronize, then inspect only slots the kernel must write:
  `exp_sums.fill_(-999.0)`. Check strides and whether producer and reducer both use
  the local partition slot (`part_z`) rather than an absolute partition index.
- **Isolate addressing:** set valid Q/K/V entries to one. Ordinary unscaled attention
  should reproduce the uniform V value. This misses V/P permutations because every
  value is equal; follow with nonuniform data. Adjust the expected result for scales
  and masks rather than changing the numerical gate.
- **Separate main/reduce stages:** in implementations supporting `one_shot` or
  `max_context_partition_num=1`, use a case that fits one partition. Compare its
  main result before testing the full multi-partition operation. Verify that each
  required partition is covered by `grid_z` or by a loop.
- **Cross-check another implementation:** run Gluon and FlyDSL on identical inputs,
  then `torch.testing.assert_close(actual, reference, atol=task_atol, rtol=task_rtol)`.
  The task's designated reference remains authoritative.
- **Check quantization:** match per-row versus per-tensor Q scales. Trace the score
  scale `softmax_scale * q_scale * k_scale` through the actual quantized formulation,
  and check that `v_scale` is applied once. A mismatch is not automatically benign
  because its error is small.
- **Check MFMA orientation:** for `C=A@B`, A owns M×K and B owns K×N under the selected
  instruction's fragment mapping. The imported QK implementation computes a
  transposed tile with K as LHS and Q as RHS; a conventional QKᵀ mapping uses Q as
  LHS. Follow that kernel's layout, not a universal operand-order rule.

## Frontend and memory details

- Use `range_constexpr()` for compile-time Python indexing. Runtime-carried state
  uses the installed `range(..., init=...)`/scf API; keep raw SSA types and state
  positions consistent.
- If a raw GPU condition cannot be evaluated as a Python bool, use `arith.select`
  for pure values or supported structured control flow. Select does not make
  side-effecting or invalid memory operations in its operands safe.
For lower-level scf calls, unwrap carried values and preserve their order/type in
init, yield and results. Common carried types include `f32`, integer/index and
vectors; support depends on the installed frontend:

```python
def unwrap(value):
    return value.ir_value() if hasattr(value, "ir_value") else value

init_state = [unwrap(acc), unwrap(next_offset)]
```

See the [loop examples](authoring_tile_programming.md#step-4-add-control-flow)
for index bounds and `range(..., init=...)`. A compile-time Python `if` is valid;
a raw runtime boolean fails outside the frontend's supported control-flow rewrite.

Buffer offsets use elements of the selected dtype, not universally bytes:

```python
packed = buffer_ops.buffer_load(rsrc, byte_offset // 4, vec_width=4, dtype=T.i32)
```

`vector.store` requires the corresponding vector type; for a one-element vector:

```python
vec = vector.from_elements(T.vec(1, T.i32), [scalar_i32])
vector.store(vec, lds_ptr, [idx])
```
- Check cached shared-memory views for SSA dominance when moving them across
  regions; see [rewrite constraints](authoring_optimization.md).

## Cache and execution

When a compiler/runtime change appears ineffective, an isolated diagnostic can
bypass the disk cache without deleting it:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 my_kernel.py
```

This does not clear in-process or Python `lru_cache` state. Check the installed
runtime and resolved cache path before attributing a failure to cache reuse.
Before a loop, inspect `start`, `stop` and `step` on the host. A zero step or an
unsigned bound that wraps can prevent progress. Every participating workgroup
thread must reach a `gpu.barrier()` consistently; a divergent branch can deadlock.
For a hang, preserve the failed operation and use the existing owner recovery
route; do not launch overlapping retries or reset an unrelated device.

Use nonuniform valid inputs and warmed same-storage changes where state reuse is
part of the path. A passing fallback or a new-tensor check may miss stale cached
values. Reference repeatability failures remain separate from candidate defects.

## Sources

- Imported source: `AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_debug_kernel.md`
- [Tile programming](authoring_tile_programming.md)
- [AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
