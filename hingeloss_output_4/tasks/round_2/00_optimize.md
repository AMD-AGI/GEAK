---
label: pytorch_to_flydsl_translation
priority: 10
agent_type: strategy_agent
kernel_path: ../../../HingeLoss/hingeloss.py
repo_root: ../../../HingeLoss
test_command: python3 /workspace/GEAK/HingeLoss/test_harness_translation.py --correctness
  && python3 /workspace/GEAK/HingeLoss/test_harness_translation.py --benchmark
harness_path: /workspace/GEAK/HingeLoss/test_harness_translation.py
commandment: ../../COMMANDMENT.md
baseline_metrics: ../../baseline_metrics.json
profiling: ../../profile.json
codebase_context: ../../CODEBASE_CONTEXT.md
benchmark_baseline: /workspace/GEAK/hingeloss_output_4/benchmark_baseline.txt
round: 2
starting_patch: ../../results/round_1/pytorch_to_flydsl_translation/patch_22.patch
pytorch_translation: true
flydsl_kernel_filename: hingeloss_flydsl.py
---

Translate the PyTorch kernel `hingeloss.py` into an equivalent FlyDSL GPU kernel and optimize it.

**FlyDSL kernel file**: Create `hingeloss_flydsl.py` in the working directory.
The test harness will load it via `--flydsl-kernel` and compare outputs against the PyTorch reference.

You MUST use FlyDSL (`import flydsl.compiler as flyc`, `import flydsl.expr as fx`). Do NOT use Triton or any other kernel language.

**Phase 1 — Correctness**: Write a FlyDSL kernel whose output matches the PyTorch reference (the correctness test must PASS).
**Phase 2 — Performance**: Once correct, optimize to outperform PyTorch's default kernels. Target speedup > 1.0x.

Use `save_and_test` after every change. Report final correctness status and speedup when done.
