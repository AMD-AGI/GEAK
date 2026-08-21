# TritonAMDGPUCanonicalizePointers asserts on an `scf.if` that yields a pointer

Hit while rewriting `_score_kernel` in `sglang/srt/layers/attn_residual.py`.
Triton 3.6.0, `/sgl-workspace/triton-custom`, target `hip:gfx950`.

## The rewrite that fails

The kernel reads either a bank row or the prefix row, selected by a program id
that is constant for the whole CTA. The stock kernel re-evaluates that select on
every iteration of the H loop:

```python
for h0 in tl.static_range(0, H, BLOCK_H):
    offs_h = h0 + tl.arange(0, BLOCK_H)
    if j < NVB:
        v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h)
    else:
        v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h)
```

Hoisting the select into a base pointer computed once, above the loop, is the
obvious cleanup:

```python
if j < NVB:
    base = bank_ptr + pid_t * stride_bm + j * stride_bb
else:
    base = prefix_ptr + pid_t * stride_pm
for h0 in tl.static_range(0, H, BLOCK_H):
    v = tl.load(base + h0 + tl.arange(0, BLOCK_H))
```

It does not compile:

```
python3: .../TritonAMDGPUTransforms/CanonicalizePointers.cpp:1441:
  ConvertSCFIfOp::matchAndRewrite_(...): Assertion
  `(fatPtrs.at({thenFatPtrBase, thenFatPtrOffset}) ==
    fatPtrs.at({elseFatPtrBase, elseFatPtrOffset})) &&
   "expected then fat ptr canNarrow and else fat ptr canNarrow to be equal"'
  failed.
...
error: Failures have been detected while processing an MLIR pass pipeline
note: Pipeline failed while executing [`TritonAMDGPUCanonicalizePointers`
      on 'tt.func' operation: @_score_kernel_v]
RuntimeError: PassManager::run failed
```

The AMD backend splits each pointer into a (base, offset) "fat pointer" and
tracks a `canNarrow` bit saying whether the offset fits in 32 bits. When an
`scf.if` yields pointers derived from two *different* base pointers, the pass
asserts that the two branches agree on that bit instead of handling the
disagreement. Here `prefix` is `[T, H]` and `bank` is `[T, NB, H]`, so the two
branches genuinely differ.

Note the failure mode: an assertion in a C++ pass, surfacing in Python as a bare
`RuntimeError: PassManager::run failed`. The message names neither the construct
nor the line, and the reproducer it offers to dump goes to `std::errs`. Nothing
in it points at "you returned a pointer from an `if`".

## Workaround

Keep the branch inside the loop and let it yield a *value* rather than a
pointer, exactly as the stock kernel does. That costs nothing measurable — the
predicate is uniform and loop-invariant, and the resulting code is the same two
loads under a uniform branch — so this is a constraint on how the source may be
written, not on what the hardware can do.

Alternatives, if the branch is genuinely expensive in some other kernel:

- make the row source a `tl.constexpr` flag and launch two specialized grids;
- pass a single pointer plus an integer offset chosen on the host, so no `if`
  ever yields a pointer.

## Scope

The assert fires on `scf.if` specifically. Selecting a *offset* across an if is
fine; only the pointer yield trips it. `tl.where` on pointers was not tried.
