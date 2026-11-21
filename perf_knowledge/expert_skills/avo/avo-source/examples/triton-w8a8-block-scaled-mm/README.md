# Triton W8A8 block-scaled matmul

This case was selected from
`avo_runs/w8a8_block_scaled_mm/exp/avo_kernel_src_20260717_040543`. Its best
commit-gated checkpoint, `avo-v8`, passed correctness and reached a verified
1.237854x geomean speedup on MI300X across c2, c32, and c64.

The source run stopped during evolution at step 82, so it has no final
`director_validation.json`. This example therefore preserves the verified
checkpoint evidence and does not label the campaign as director-finalized.

- `kernel/`: original editable task package and golden reference
- `reference/final_patch.diff`: cumulative `avo-v0..avo-v8` source diff
- `reference/optimized.py`: exact source at tag `avo-v8`
- `reference/checkpoint_validation.json`: concise verified result
- `reference/lineage.json`: full commit-gated score history
- `reference/baseline_timing.json`: baseline timing evidence
- `handoff.json`: short source-only example campaign

Run from `avo-source/examples/`:

```bash
python3 run_example.py \
  triton-w8a8-block-scaled-mm --dry-run
```

For GPU execution, select an idle GPU in `handoff.json` and remove
`--dry-run`.
