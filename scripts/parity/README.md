# Parity test infrastructure

Hard evidence that the `refactor-test` branch preserves the preprocess
contract of `origin/main`.  Two layers:

## 1. Static parity audit — run any time, no LLM required

```bash
# Clone origin/main side-by-side (one-time):
mkdir -p /data/sapmajum/parity_test
git clone --depth 20 --branch main https://github.com/AMD-AGI/GEAK.git \
    /data/sapmajum/parity_test/GEAK-main

# Run the audit (pure AST + filesystem inspection):
python3 scripts/parity/static_parity_audit.py
```

The audit parses both codebases' `run_preprocessor`, contract
validators, preprocess private helpers, harness-phase layer methods,
and subagent presence.  Exit 0 when:

  - `run_preprocessor` signatures are parameter-identical
  - Every private helper the new `phases/harness.py` module imports
    still exists on both pipelines
  - The refactor's own new modules (contract validators, HarnessPhase,
    run_pipeline, subagent framework) are all present on
    `refactor-test`

Latest output: [static_parity_report.md](static_parity_report.md).

## 2. End-to-end parity test — requires LLM API keys + GPU

```bash
python3 scripts/parity/parity_test_e2e.py \
    --kernels /data/sapmajum/GEAK/examples/mla_decode/kernel.py \
              /data/sapmajum/GEAK/examples/knn/src/knn_cuda.hip \
    --run-root /tmp/parity_runs \
    --clean
```

Runs each kernel through BOTH pipelines (`refactor-test` HEAD and
`origin/main`) inside the `geak_agent` container, with
`GEAK_HARNESS_ONLY=1` so only the preprocess stages run (fast enough
to compare without paying for full optimization rounds).  Captures
harness / commandment / baseline_metrics / profile artefacts for each
pipeline and diffs them at the contract level.  Produces
`parity_report.md`.

### Environment invariants during parity tests

- `GEAK_USE_KERNEL_ANALYSIS=0` — KernelAnalysisAgent is a NEW subagent
  on `refactor-test` with no origin/main equivalent.  Off by default
  to keep apples-to-apples comparison.
- `GEAK_USE_KNOWLEDGE_BASE=0` + `GEAK_SAVE_TO_KNOWLEDGE_BASE=0` — KB
  retrieval/write is non-deterministic across runs; disable for
  parity.
- `GEAK_HARNESS_ONLY=1` — skip profiling + baseline + commandment
  steps for a fast parity read on the harness-generation stage alone.
  Omit for full-pipeline comparison.
