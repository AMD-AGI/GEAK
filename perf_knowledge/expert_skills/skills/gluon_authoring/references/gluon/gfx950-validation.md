# gfx950 layout validation

Use this procedure in the workload's Python environment. It validates a finite
set of layouts; it does not select an optimization tactic or establish a speedup.
It is an explicit smoke tool, not a new default GEAK workflow stage.

## Flow and ownership

| Step | Interface | Required evidence |
| --- | --- | --- |
| Inspect installed compiler | `ttgir_bridge.py capabilities --json` | Actual API signatures and normalization support; version strings alone are insufficient |
| Construct candidate | Native MFMA/DotOperand/scale helpers; native padding helper when applicable | Record dtype, instruction shape, warp layout, operand index, shape and helper result |
| Check hardware facts when needed | Shared `amd_isa.py --arch gfx950` | Opcode, packing/scale rules and source caveats; no Matrix Calculator dependency on gfx950 |
| Compile and check numerics | `smoke_layout_gpu.py` under the shared GPU lock | Per-case outcome, independent CPU reference, negative control, TTGIR/LLVM/ISA and metadata |
| Inspect actual artifacts | Shared `probe.py measure`; bridge `recover` | Actual resource use and recovered layout context, not resource guesses from layout objects |
| Evaluate an optimization | Existing workload oracle, harness and profiling/TraceLens flow | Correctness first, then controlled A/A and A/B; smoke results are not performance acceptance |

## Run

From the repository root, first inspect the workload environment without GPU access:

```bash
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py capabilities --json
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 facts v_mfma_scale_f32_16x16x128_f8f6f4 --as fp4
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 tables scale.16x16x128
```

Set `GPU_ID` to a GPU allocated to this run. Output directories must be new.
If already executing inside a GEAK lease, invoke Python directly and retain the
ownership environment; do not acquire the same GPU lock twice.

```bash
bash kernel_workflow/scripts/gpu_lock.sh "${GPU_ID:?Set the allocated GPU ID}" \
  python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/smoke_layout_gpu.py \
  --suite core --out results/gfx950-layout-core

bash kernel_workflow/scripts/gpu_lock.sh "${GPU_ID:?Set the allocated GPU ID}" \
  python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/smoke_layout_gpu.py \
  --suite staging --out results/gfx950-layout-staging
```

For Docker, acquire the host lock around `docker run`, expose the corresponding
device, and propagate ownership into the container. Container ordinals can differ
from host ordinals; verify the device identity rather than copying a host HIP mask.
Use a writable output mount. The runner places its default Triton cache there.

## Interpret the results

| Suite | Scope | Limitations |
| --- | --- | --- |
| `core` | 48 native GEMMs: FP16/BF16/MXFP4 × two instruction shapes × two warp arrangements × direct/Blocked epilogues × aligned/tail shape; one missing-scale negative control; 18 trivial-swizzle copy cases | Small, finite, dyadic inputs; batch 1; not the full exponent range, quantization quality or an exhaustive numerical tolerance study |
| `staging` | 54 candidate copy cases: uint8/FP16/BF16 × A/B memory order × narrow/wide nontrivial swizzle or native padding × sync/buffer async/global async | Plain Blocked producers, fixed shapes, one parent instruction shape per dtype; padding may be absent/None; this is a survey, not a guarantee that each candidate is supported |

Each copy case checks two nonconstant input patterns bit-for-bit. Producers cover
the tile exactly with 16 bytes per lane. This isolates useful staging combinations;
it does not cover arbitrary per-thread widths or a multi-stage ring. `uint8` copy
checks byte transport; it is not by itself an FP4 matrix correctness check.
The async APIs also require coalesced destination writes and swizzling within a
warp boundary. A constructible layout is not proof that its producer satisfies
that contract; investigate failed combinations before calling them compiler bugs.

`summary.json` distinguishes `pass`, `negative_control_detected`, `unsupported`,
numerical failure, worker/compiler/runtime error and timeout. Each worker has its
own directory and log, so an LLVM abort does not erase the other cases. Successful
compiles also retain `kernel.ttgir`, `kernel.llir`, `kernel.amdgcn`, `kernel.json`.

Exit 0 means all selected cases passed (including detection of the negative
control). Exit 1 means at least one failure. Exit 3 means unsupported/incomplete
coverage with no failures. An unsupported or failed candidate is not accepted for
a workload merely because the survey finished. Inspect the recorded error before
attributing it to a layout family or a compiler defect.

For a passing case, inspect its saved artifacts:

```bash
python kernel_workflow/scripts/kernel_tools/probe.py measure \
  --dir results/gfx950-layout-core/case-000 --arch gfx950 --json results/resources.json
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py recover \
  --ttgir results/gfx950-layout-core/case-000/kernel.ttgir --arch gfx950 \
  --out results/layouts.py --json results/layouts.json
```

The smoke cannot clear a failing workload oracle. In particular, any earlier
Agent experiment with a failed input remains failed until that input is diagnosed
and rerun. For shared-memory optimization, add a full GEMM using the chosen
producer/shared/consumer combination before benchmarking it.
