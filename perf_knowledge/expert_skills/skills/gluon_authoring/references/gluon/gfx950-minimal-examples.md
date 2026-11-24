# gfx950 Minimal Gluon Examples

File-backed starting points for probes. Re-derive tile constants, layouts and dtypes for the real
operator before benchmarking.

## Runnable Smoke Copy (toolchain validation only)

Define `@gluon.jit` in a `.py` file — not `python -c`, stdin, or `exec`.

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def smoke_copy(x_ptr, y_ptr, n: gl.constexpr, BLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    other = gl.full((BLOCK,), 0.0, x_ptr.dtype.element_ty, layout=layout)
    x = gl.load(x_ptr + offs, mask=mask, other=other)
    gl.store(y_ptr + offs, x, mask=mask)


def run(device="cuda", n=1024, block=256, num_warps=1):
    assert block % (64 * num_warps) == 0          # wave64
    layout = gl.BlockedLayout(
        size_per_thread=[block // (64 * num_warps)],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )
    x = torch.arange(n, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    smoke_copy[(triton.cdiv(n, block),)](x, y, n, BLOCK=block, layout=layout, num_warps=num_warps)
    torch.testing.assert_close(x, y)
```

Proves only that `@gluon.jit` compiles, launches and feeds an output on gfx950. Passing a 1D copy is
not evidence that MFMA, async copy or a fused pattern works — probe the mechanism itself.

## CDNA4 MFMA Planning Skeleton

```python
mfma_layout = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                   transposed=True, warps_per_cta=[1, 4])
a_dot = gl.convert_layout(a, gl.DotOperandLayout(0, mfma_layout, k_width))
b_dot = gl.convert_layout(b, gl.DotOperandLayout(1, mfma_layout, k_width))
acc = gl.zeros((BLOCK_M, BLOCK_N), acc_dtype, layout=mfma_layout)
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)            # cdna3.mfma + version=3 for gfx942
acc_store = gl.convert_layout(acc, blocked_mn)
```

`k_width`, accumulator dtype (int32 for INT8), fp8/fp4 shapes and the scaled path:
`matrix-reference.md ## Matrix-Family Details`. Runnable pipelined CDNA4 examples:
`../../scripts/pipeline_examples_cdna4.py`. A `gl.load -> tl.dot -> gl.store` hybrid fails to compile on
3.8.0 (`tl.*` boundary section of `imports-and-launching.md`).

## Failure classes

Record env first (Triton tag, ROCm, PyTorch, arch). Classify a failed probe as
`environment_toolchain_blocker`, `repo_production_gate`, `backend_or_lowering_failure`,
`api_or_layout_failure`, `correctness_failure` or `performance_or_integration_no_win`. The first two
are not evidence that Gluon lacks a mechanism. Lowering-failure strings and their causes:
`../method/triage.md`.
