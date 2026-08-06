# gfx950 Minimal Gluon Examples + Feasibility Gate

File-backed starting points for probes plus the capability-level + feasibility
gate that decides whether a Gluon probe is worth building. The feasibility gate
also feeds the escalation gate (`../escalation-gate.md`). Re-derive tile
constants, layouts, and dtype choices for the real operator before benchmarking.

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

This proves only that the local `@gluon.jit` path compiles, launches, and feeds
an output on gfx950. It does not justify a larger rewrite.

## CDNA4 MFMA Planning Skeleton

Use this shape for planning, not as a copy-paste performance kernel:

```python
mfma_layout = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                   transposed=True, warps_per_cta=[1, 4])
a_layout = gl.DotOperandLayout(0, mfma_layout, k_width)
b_layout = gl.DotOperandLayout(1, mfma_layout, k_width)
a_dot = gl.convert_layout(a, a_layout)
b_dot = gl.convert_layout(b, b_layout)
acc = gl.zeros((BLOCK_M, BLOCK_N), acc_dtype, layout=mfma_layout)
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)            # cdna3.mfma + version=3 for gfx942
acc_store = gl.convert_layout(acc, blocked_mn)
```

Rules: `instr_shape` is 3D on Triton >= 3.6; direct INT8 `cdna4.mfma` uses int32
accumulator; BF16/FP16 regular MFMA uses fp32 accumulator; FP4/FP8 scaled paths
use `cdna4.mfma_scaled` only after planning scale layout and format
(`matrix-reference.md`, `../hardware/capability-matrix.md`).

## Gluon Capability Levels

Do not treat "Gluon imports" as proof that matrix or attention patterns work.
Record the minimum level the proposed mechanism needs:

| Level | Meaning | Required evidence |
| --- | --- | --- |
| 0 | `triton.experimental.gluon` imports | import path + version recorded |
| 1 | 1D `BlockedLayout` load/store executes | minimal `@gluon.jit` smoke + correctness |
| 2 | target layout objects construct | layout object creation outside JIT body |
| 3 | 2D blocked load/store executes | correctness on the needed rank/layout |
| 4 | `DotOperandLayout` + MFMA executes | matrix correctness for dtype/accumulator |
| 5 | full pattern executes (dot + reduction + dot) | measured boundary path feeds final output |

Probe the required level for the candidate mechanism. Level 0/1 is not evidence
for Level 4/5.

## Quick Feasibility Gate (is a Gluon probe worth it?)

Answer before a full Gluon mechanism probe — these feed `../escalation-gate.md`:

```text
hot_path_in_measured_boundary:
plain_triton_mechanism_already_optimal:
matrix_dtype_and_accumulator_supported (capability-matrix):
explicit_layout_can_reduce_real_traffic:
kernel_body_dominates_boundary:
plain_triton_comparator_path:
gluon_extra_mechanism:
```

Skip or sharply limit a Gluon probe when all hold:

- the kernel is launch/wrapper dominated or <50us (sub-ms table in
  `../phases/harness.md`);
- there is no matrix, reduction, scan, or layout-aware memory mechanism to target;
- the plain Triton path already lowers to the desired matrix instruction;
- the proposed Gluon path would only rewrite spelling, not reduce traffic,
  conversions, launches, or wrapper cost.

If the hot path contains `tl.dot` and the only candidate is `gl.load -> tl.dot ->
gl.store`, stop: `tl.dot` does not preserve Gluon layout for later Gluon ops.

## Probe Order + Failure Classes

1. record env/version (Triton tag, ROCm, PyTorch, target arch, import path);
2. minimal `@gluon.jit` smoke (above);
3. repo-supported probe (does the production gate accept the arch/dtype?);
4. forced-backend probe only when value justifies it (distinguish repo policy
   from real backend capability);
5. mechanism probe (one named mechanism, executes + feeds measured output).

Classify every failed probe: `environment_toolchain_blocker`,
`repo_production_gate`, `backend_or_lowering_failure`, `api_or_layout_failure`,
`correctness_failure`, `performance_or_integration_no_win`. The first two only
decide whether the next probe is allowed — they are not proof that Gluon lacks a
mechanism. Fair-comparison and no-extra-mechanism rules live in
`../gluon-negative-patterns.md`.
