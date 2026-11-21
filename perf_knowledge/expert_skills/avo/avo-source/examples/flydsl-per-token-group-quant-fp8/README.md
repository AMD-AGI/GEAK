# FlyDSL per-token FP8 quantization

This case was selected from the successful
`avo_flydsl_runs/_per_token_group_quant_fp8` campaign completed on MI300X.
Director validation rebuilt a clean workspace, passed correctness, and measured
a 3.2439x geomean speedup across c2, c32, and c64.

The accepted one-step patch separates the optional `aiter.ops.flydsl` import
from launcher construction. In the captured environment the optional import
failed, so the baseline silently fell back to pure Triton; the patch restored
the existing FlyDSL wide-streaming path.

- `kernel/source/`: the actual AVO baseline: a FlyDSL-capable implementation
  before the accepted import-split fix
- `kernel/source_golden/`: the independent pure-Triton correctness oracle, not
  the AVO performance baseline
- `reference/final_patch.diff`: accepted `avo-v1` change
- `reference/optimized.py`: accepted optimized source snapshot
- `reference/director_validation.json`: correctness and timing evidence
- `reference/trajectory.json`: campaign lineage summary
- `handoff.json`: short source-only example campaign

Run from `avo-source/examples/`:

```bash
python3 run_example.py \
  flydsl-per-token-group-quant-fp8 --dry-run
```

For GPU execution, set `AVO_FLYDSL_ENV` to the FlyDSL environment script,
select an idle GPU in `handoff.json`, and remove `--dry-run`. The launcher
validates this variable and forwards it as `kernel_env_script`; a missing or
nonexistent script fails before starting the campaign.
