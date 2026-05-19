# Triton — Idioms Reference

Short reference of "what Triton code looks like here" — used to detect
entry points (the `@triton.jit` kernel vs the Python-level launcher),
and to ground the harness's import target in language reality.

Lifted from `src/minisweagent/kernel_languages/triton/idioms.md` and
`src/minisweagent/kernel_languages/triton/builder_hints.md`.

---

## Entry-point detection rules

From `builder_hints.md`:

- The kernel is a `@triton.jit`-decorated function. Locate it by the
  **decorator**, not by the function name — users rename things.
- The Python-level launcher (typically named after the kernel,
  e.g. `def silu(x, y, out):` then `silu_kernel[(grid,)](...)`) is what
  the harness imports — NOT the `@triton.jit` function directly. The
  launcher sets up grid / `BLOCK_SIZE` / `num_warps` and dispatches.
- When the test file imports the kernel via
  `from package.file import kernel`, preserve that module path in the
  harness so reruns stay stable.

---

## Kernel shape

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

## Launcher shape

```python
def my_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
    return out
```

The harness imports `my_op` (the launcher), not `my_kernel` (the JIT
function). Calling `my_op(*_inputs)` from the harness mirrors how the
launched kernel is normally invoked, including the grid / launch
parameters the launcher sets up.

---

## Reference selection (correctness oracle)

- Prefer `torch` reference implementations (correctness-only) over
  homemade NumPy ones — faster to write, well-vetted.
- Use `torch.allclose(candidate, reference, atol=1e-4, rtol=1e-4)`
  unless the user test specifies tighter tolerances.

---

## Timing loop

- Warmup 5 iterations before the timed run; measure 100 iterations
  and take the **median** (not mean).
- Synchronize (`torch.cuda.synchronize()`) before and after timing.

---

## Common optimisations (context only — the harness doesn't optimise)

- **Reduction kernels:** accumulate in a register, not in LDS, until the
  final stage.
- **Matmul:** use `tl.dot(a, b, acc)` rather than explicit
  `tl.sum(a * b, axis=...)` — compiles to MFMA.
- **Masked load:** `tl.load(ptr + offs, mask=mask, other=0.0)` avoids
  out-of-bounds reads; cheaper than branching.
