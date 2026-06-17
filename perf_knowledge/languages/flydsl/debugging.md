---
title: "FlyDSL — kernel debugging (NaN / zeros / mismatch / compile / hang)"
kind: language
gens: [gfx942, gfx950]
updated: 2026-06-01
source_commit: AMD-AGI/GEAK@c0a1f937
sources:
  - AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_debug_kernel.md
---

> **Reference (how-to), not a verdict.** Ingested from the FlyDSL authoring skill as reference material.
> It tells you how to triage a FlyDSL kernel; it does not decide correctness for you — always confirm
> against the kernel's own test/oracle on-box.

# Debug a FlyDSL kernel

This reference covers correctness, stability, and hang triage on runnable FlyDSL kernels.

**Scope**: execution debugging, not performance triage. For kernel-speed work see
[`authoring_optimization.md`](authoring_optimization.md) (and [`authoring_gemm_levers.md`](authoring_gemm_levers.md)
for GEMM). Here, "tracing" refers to FlyDSL frontend tracing, not ATT collection.

## Step 0: Decide whether cache is involved

FlyDSL uses both an in-process cache and a disk cache, so stale artifacts can make a fix look ineffective. Start with the least destructive check:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 my_kernel.py
```

This bypasses disk-cache reads and writes for this process run. It does not delete existing cache files, and it does not clear any in-memory cache entries.

Only consider manually clearing cached artifacts after you have confirmed the exact cache directory (for example, `FLYDSL_RUNTIME_CACHE_DIR` may override the default) and you are OK losing them. Avoid recursive wildcard deletes.

Also clear Python-level caches if using `@functools.lru_cache`:
```python
compile_my_kernel.cache_clear()
```

## Step 1: Classify the error

| Symptom | Likely Cause | Go to |
|---|---|---|
| All NaN output | Softmax -inf/-inf, division by zero, uninitialized buffer | Section 2 |
| All zeros output | Wrong output address, uninitialized temp buffer | Section 3 |
| Partially wrong (>50% mismatch) | Wrong partition count, missing partitions, layout mismatch | Section 4 |
| Small errors (1-5% mismatch) | FP8 quantization, scale factor, off-by-one masking | Section 5 |
| Compilation error / crash | Type mismatch, scf.for state, range vs range_constexpr | Section 6 |
| GPU hang | Infinite loop, deadlock in barrier, OOB memory access | Section 7 |

## Step 2: Debugging NaN

### 2.1 Softmax NaN: -inf minus -inf

When ALL tokens in a partition are masked (out of context), `qk_max = -inf`. Then `exp(s - qk_max) = exp(-inf - (-inf)) = exp(NaN) = NaN`.

**Fix**: Guard the exp calculation:
```python
safe_diff = arith.select(qk_max > NEG_INF, diff, ZERO_F)
```

### 2.2 Division by zero in normalization

When `exp_sum = 0` (all probs zero), `1/exp_sum = inf`.

**Fix**:
```python
safe_sum = arith.select(running_sum > ZERO_F, running_sum, arith.constant(1.0, type=T.f32))
inv_sum = arith.constant(1.0, type=T.f32) / safe_sum
```

### 2.3 Host-side NaN check

Add prints in the Python launch function to check intermediate buffers:
```python
torch.cuda.synchronize()
print(f"exp_sums nan={exp_sums.isnan().sum()}, inf={exp_sums.isinf().sum()}")
print(f"max_logits nan={max_logits.isnan().sum()}, range=[{max_logits.min():.4f}, {max_logits.max():.4f}]")
print(f"temp_out nan={temporary_output.isnan().sum()}")
```

## Step 3: Debugging all-zeros output

### 3.1 Wrong output address

Check stride parameters: if `stride_out_seq` or `stride_out_part` is wrong, output writes go to incorrect locations. Print strides:
```python
print(f"out strides: {output.stride()}, temp strides: {temporary_output.stride()}")
```

### 3.2 Partition slot mismatch

For multi-partition kernels, verify the output is written to `part_z` slot (not absolute partition index). The reduce kernel reads from `part_z = 0..grid_z-1` slots.

### 3.3 exp_sums at zero / max_logits at -inf

If the main kernel doesn't write exp_sums/max_logits, the reduce kernel produces zeros. Initialize sentinel values before kernel launch:
```python
exp_sums.fill_(-999.0)  # sentinel
# ... launch kernel ...
torch.cuda.synchronize()
print(f"exp_sums[0,0,0,:4] = {exp_sums[0,0,0,:4]}")  # should NOT be -999
```

## Step 4: Debugging large mismatch (>50%)

### 4.1 Missing partitions

If `grid_z < total_partitions` and the kernel processes only ONE partition per CTA (no loop), most of the context is skipped. Verify:
```python
total_parts = math.ceil(context_len / KV_COMPUTE_BLOCK)
print(f"grid_z={grid_z}, total_parts={total_parts}")
assert grid_z == total_parts or kernel_has_multi_partition_loop
```

### 4.2 All-1s isolation test

Fill query, key_cache, value_cache with 1.0 to eliminate data-dependent bugs:
```python
query.fill_(1.0)
key_cache.fill_(1.0)
value_cache.fill_(1.0)
```
With uniform input: all softmax probs are equal, PV output = 1.0. Any deviation reveals layout/addressing bugs.

**Caveat**: All-1s test does NOT catch V/P operand misalignment (since uniform values produce correct results regardless of ordering).

### 4.3 Single-partition test

Force `max_context_partition_num=1` (one_shot mode) to bypass the reduce kernel and test the main kernel in isolation.

### 4.4 Compare against Gluon

Run both Gluon and FlyDSL on the same input and compare element-wise:
```python
torch.testing.assert_close(flydsl_output, gluon_output, atol=5e-3, rtol=5e-3)
```

## Step 5: Debugging small errors (1-5%)

### 5.1 FP8 probability requantization

FP8 PV MFMA introduces ~0.03 max error vs bf16 reference. This is inherent to the FP8 data path and NOT a bug. Expected tolerance: `atol=5e-3`.

### 5.2 Per-tensor vs per-row quantization

If the reference uses per-row Q quantization but FlyDSL uses per-tensor, expect ~1-3% mismatch. Verify quantization mode matches.

### 5.3 Scale factor mismatch

Verify `_scale = softmax_scale * q_scale * k_scale` matches the reference. Common bug: applying v_scale twice (once in prob scaling, once after PV).

## Step 6: Compilation errors

### 6.1 `range()` vs `range_constexpr()` inside @flyc.kernel

FlyDSL's AST rewriter converts ALL `range()` to `scf.for` (runtime loops). Use `range_constexpr()` for compile-time unrolled loops:
```python
# WRONG: i becomes an ArithValue, can't index Python lists
for i in range(4): result[i] = ...

# CORRECT: i is a Python int
for i in range_constexpr(4): result[i] = ...
```

### 6.2 Runtime conditional as Python bool

FlyDSL frontend tracing evaluates Python `if` at trace time. Runtime GPU values can't be used:
```python
# WRONG: "cannot evaluate dynamic 'Boolean' as Python bool during tracing"
if kv_tok < context_len:  # runtime comparison
    fx.printf(...)

# CORRECT: use arith.select for runtime conditionals
val = arith.select(kv_tok < context_len, good_val, bad_val)
```

Python `if` is fine for COMPILE-TIME decisions (e.g., `if trans_v:` where trans_v is a Python bool).

### 6.3 scf.for state packing

All loop-carried values must be raw SSA values (not Python wrappers):
```python
def _unwrap(v):
    return v.ir_value() if hasattr(v, 'ir_value') else v

init_state = [_unwrap(v) for v in [val1, val2, vec_val]]
```

Supported state types: `f32` (scalar), `f32x4` (vector), `i32`, `i64`, `index`.

### 6.4 buffer_load type mismatch

`buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=T.i32)` — the offset is in units of `dtype`. For FP8 data addressed in bytes, divide by element size:
```python
k_addr_bytes = ...  # address in FP8 elements (= bytes for FP8)
k_4xi32 = buffer_ops.buffer_load(k_rsrc, k_addr_bytes // 4, vec_width=4, dtype=T.i32)
```

### 6.5 vector.store requires vector type

LDS `vector.store` requires the value to be a vector, not scalar:
```python
# WRONG
vector.store(scalar_i32, lds_ptr, [idx])

# CORRECT
vec = vector.from_elements(T.vec(1, T.i32), [scalar_i32])
vector.store(vec, lds_ptr, [idx])
```

## Step 7: GPU hang

### 7.1 Infinite scf.for loop

If loop bounds are wrong (`stop < start` with unsigned comparison issues, or `step=0`), the GPU hangs. Verify bounds on host:
```python
print(f"loop: start={part_start}, stop={part_end}, step={cpb}")
```

### 7.2 Barrier deadlock

`gpu.barrier()` requires ALL threads in the workgroup to reach it. If some threads take a different branch (runtime `if`), the barrier deadlocks. FlyDSL doesn't support divergent barriers.

### 7.3 Recovery from GPU hang

```bash
# Check GPU state
rocm-smi
```

If the GPU shows no forward progress, stop automated actions here. Recover the device with your local ops playbook, and treat privileged reset or reboot steps as a manual action.

## Step 8: Diagnostic workflow

```
1. If compiler/runtime internals changed, rerun once with FLYDSL_RUNTIME_ENABLE_CACHE=0 to bypass disk-cache reads/writes for that process
2. Run with all-1s input → passes? Layout is OK, data issue
3. Run with single partition (one_shot) → passes? Multi-partition/reduce bug
4. Add host-side prints (tensor shapes, strides, NaN checks)
5. Compare intermediate buffers (exp_sums, max_logits, temp_out)
6. If layout bug suspected: walk one thread's addresses manually
   (tid=0: lane16id=0, rowid=0, warp_id=0)
7. For MFMA bugs: verify operand order (K is LHS, Q is RHS for QK)
```

## Step 9: Common pitfalls checklist

- [ ] If cache is suspected, reran once with `FLYDSL_RUNTIME_ENABLE_CACHE=0` to bypass disk-cache reads/writes before clearing anything manually
- [ ] `range_constexpr()` for all compile-time loops (not `range()`)
- [ ] No Python `if` on runtime GPU values
- [ ] `buffer_load` offset units match dtype (bytes/4 for i32)
- [ ] `vector.store` uses vector type (not scalar)
- [ ] `scf.for` state packed with `_unwrap()` (raw SSA values)
- [ ] Output written to correct partition slot (`part_z`, not absolute index)
- [ ] `exp_sums`/`max_logits` strides match actual tensor layout
- [ ] Softmax guards against `-inf - (-inf) = NaN`
- [ ] Division by zero guarded (`select(sum > 0, sum, 1.0)`)
- [ ] K/V address calculation matches tensor layout (4D vs 5D trans_v)
- [ ] MFMA operand order: `mfma(LHS, RHS, acc)` — LHS→M, RHS→N

## Sources
- AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_debug_kernel.md — origin (ingested as reference).
- AMD Instinct CDNA3 ISA (MFMA operand order, `gpu.barrier`/`s_waitcnt` semantics): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- FlyDSL semantics cross-refs: [`overview.md`](overview.md) · [`knobs.md`](knobs.md) · [`authoring_tile_programming.md`](authoring_tile_programming.md)
