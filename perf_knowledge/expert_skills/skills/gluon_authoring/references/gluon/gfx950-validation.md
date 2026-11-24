# gfx950 layout validation

`smoke_layout_gpu.py` validates a finite set of layouts (correctness only — not a speedup, not an
optimization choice). Run from the repo root in the workload's Python env.

```bash
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py capabilities --json   # no GPU
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/smoke_layout_gpu.py --compile-only --out <new-dir>   # CPU compile only
bash kernel_workflow/scripts/gpu_lock.sh "${GPU_ID:?}" \
  python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/smoke_layout_gpu.py --suite core --out <new-dir>     # or --suite staging
```

Inside a GEAK lease call Python directly (do not take the same GPU lock twice). Output dirs must be
new. In Docker, take the host lock around `docker run` and verify device identity (container
ordinals differ).

| Suite | Scope | Not covered |
| --- | --- | --- |
| `core` | 48 native GEMMs (FP16/BF16/MXFP4 x 2 instr shapes x 2 warp layouts x direct/Blocked epilogue x aligned/tail) + a missing-scale negative control + 18 trivial-swizzle copies | full exponent range, batch > 1, tolerance study |
| `staging` | 54 copy cases: uint8/FP16/BF16 x A/B order x swizzle/native padding x sync/buffer-async/global-async; Blocked producers at 16 B/lane | derived padded producers, other widths, multi-stage rings — a failure here is not proof the family is unsupported |

`summary.json` per case: `pass`, `negative_control_detected`, `unsupported`, numerical failure,
worker/compiler/runtime error, timeout; artifacts `kernel.{ttgir,llir,amdgcn,json}`. Exit 0 all pass,
1 any failure, 3 unsupported/incomplete with no failures. Inspect a passing case with
`kernel_workflow/scripts/kernel_tools/probe.py measure --dir <case> --arch gfx950` and
`ttgir_bridge.py recover --ttgir <case>/kernel.ttgir --arch gfx950`. A smoke pass does not clear a
failing workload oracle.
