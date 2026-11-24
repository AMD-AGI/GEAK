# Source-level AVO example

`flydsl-gemm-a16-w16` is a compact, self-contained task package selected from a
completed MI300X campaign (`avo_flydsl_runs/_gemm_a16_w16_kernel`). It includes
the original kernel harness plus its accepted patch, lineage, and validation
evidence under `reference/`: correctness pass and 1.18613x Director-verified
geomean with 0.42% CV. The final implementation is hybrid: c2 uses FlyDSL and
c32/c64 use retuned Triton.

## Validate the integration without launching agents

From this directory (`avo-source/examples/`):

```bash
python3 run_example.py flydsl-gemm-a16-w16 --dry-run
```

The helper resolves the portable paths in `handoff.json` and delegates to
`../avo-source-impl/run_geak_avo.py`. A real run first copies the case's `kernel/`
to the output directory and optimizes that copy. Results and campaign output are
written outside the GEAK checkout under the sibling directory
`<GEAK>-runs/avo-examples/<case>/{kernel,.avo,runs}`.
Set `AVO_EXAMPLE_RUNS_ROOT` to override that output root. If that root is on a
different top-level directory from `<GEAK>`, add `root_dir` to the case's
`handoff.json`.

## Launch a campaign

Choose an idle GPU with `--gpu-ids` (for example `--gpu-ids 6`), then omit
`--dry-run`. If `<GEAK>-runs/avo-examples/<case>/kernel` already exists, the run
reuses that copy; delete it to stage a fresh one. The included budget is a short example budget, not an exact copy
of the historical day-long run.

This FlyDSL case requires the ROCm/FlyDSL environment. Set
`AVO_FLYDSL_ENV` to the environment script before launching the example.

Historical timings are evidence from the source campaign, not guaranteed
performance on different software stacks, clocks, or GPUs.
