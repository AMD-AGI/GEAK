# Source-level AVO examples

These are compact, self-contained task packages selected from completed MI300X
campaigns. Each case includes the original kernel harness plus its selected
patch, optimized source snapshot, lineage, and validation evidence under
`reference/`.

- `flydsl-per-token-group-quant-fp8`: FlyDSL, correctness pass and 3.2439x
  verified geomean, selected from `avo_flydsl_runs/_per_token_group_quant_fp8`.
- `flydsl-gemm-a16-w16`: optional FlyDSL-capable GEMM alternative, correctness
  pass and 1.18613x Director-verified geomean with 0.42% CV, selected from
  `avo_flydsl_runs/_gemm_a16_w16_kernel`. Its final implementation is hybrid:
  c2 uses FlyDSL and c32/c64 use retuned Triton.
- `triton-w8a8-block-scaled-mm`: Triton, correctness pass and 1.237854x
  verified geomean at commit-gated checkpoint `avo-v8`, selected from
  `avo_runs/w8a8_block_scaled_mm`.

## Validate the integrations without launching agents

From this directory (`avo-source/examples/`):

```bash
python3 run_example.py \
  flydsl-per-token-group-quant-fp8 --dry-run
python3 run_example.py \
  flydsl-gemm-a16-w16 --dry-run
python3 run_example.py \
  triton-w8a8-block-scaled-mm --dry-run
```

The helper resolves the portable paths in each `handoff.json` and delegates to
`../avo-source-impl/run_geak_avo.py`. A real run first copies the case's `kernel/`
to the output directory and optimizes that copy. Results and campaign output are
written outside the GEAK checkout under the sibling directory
`<GEAK>-runs/avo-examples/<case>/{kernel,.avo,runs}`.
Set `AVO_EXAMPLE_RUNS_ROOT` to override that output root.

## Launch a campaign

Choose an idle GPU by changing `gpu_ids` in the case's `handoff.json`, then omit
`--dry-run`. The included budgets are short example budgets, not exact copies
of the historical day-long runs.

The FlyDSL case also requires the ROCm/FlyDSL environment. Set
`AVO_FLYDSL_ENV` to the environment script before launching the example.

Historical timings are evidence from the source campaigns, not guaranteed
performance on different software stacks, clocks, or GPUs.

The FlyDSL campaign completed final director validation. The W8A8 campaign
stopped during evolution; its included `avo-v8` checkpoint was commit-gated and
verified, but was not director-finalized.

## Twelve-hour, three-GPU campaigns

`run_12h_3gpu.sh` overrides the short example settings with:

- 43,200 seconds per case;
- 900 seconds reserved inside that budget for final report, Director validation,
  artifact materialization, and cleanup;
- Team mode with three directions;
- diverse multi-start with `start_k=3`;
- three ESCALATE workers;
- five verification repeats;
- profiling enabled;
- `max_steps=1000`.

Run one case:

```bash
./run_12h_3gpu.sh triton 0,1,2
./run_12h_3gpu.sh flydsl 0,1,2
./run_12h_3gpu.sh flydsl-gemm 0,1,2
```

The `flydsl` selector continues to target the existing per-token campaign.
`all` runs Triton, per-token FlyDSL, and FlyDSL GEMM sequentially.

Validate all expanded configurations without launching agents:

```bash
DRY_RUN=1 ./run_12h_3gpu.sh all 0,1,2
```

Run all three sequentially on the same GPUs:

```bash
./run_12h_3gpu.sh all 0,1,2
```

`all` takes up to 36 hours because each case receives its own 12-hour budget.
To run concurrently, start each case separately with mutually disjoint
three-GPU sets. Logs, normalized results, and campaign artifacts are written under the
external output root described above, not inside this repository.

Override portable environment discovery when needed:

```bash
CLAUDE_ACTIVATE=<activate-script> \
AVO_FLYDSL_ENV=<flydsl-environment-script> \
./run_12h_3gpu.sh flydsl 0,1,2
```
