# Triton — idioms reference

Short, kernel-shape-independent reference that appears near the top
of the task body so the agent has "what Triton code looks like here"
as context.

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

## Common optimisations

- **Reduction kernels**: accumulate in a register, not in LDS, until
  the final stage.
- **Matmul**: use `tl.dot(a, b, acc)` rather than explicit
  `tl.sum(a * b, axis=...)` — compiles to MFMA.
- **Masked load**: `tl.load(ptr + offs, mask=mask, other=0.0)` avoids
  out-of-bounds reads; cheaper than branching.
