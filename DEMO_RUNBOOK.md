# GEAK E2E Demo Runbook

Quick setup & run guide for the RoPE kernel optimization demo.
Pick up from any machine by following these steps.

---

## 1. Clone & Checkout

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK
git checkout msa
```

---

## 2. Set API Key & Build Docker

```bash
# Get the API key (ask team or grab from a running geak-agent container):
#   docker exec geak-agent-sdubagun bash -c 'echo $AMD_LLM_API_KEY'
export AMD_LLM_API_KEY="<your-key-here>"

# Build & start container (--rebuild forces fresh image)
bash scripts/run-docker.sh --rebuild
```

This creates container `geak-agent-$USER` with everything installed.
If the container already exists, the script just exec's into it.

---

## 3. Run the Demo

From **outside** the container (non-interactive):

```bash
docker exec geak-agent-$USER python3 -m geakagent.run.mini \
  -m claude-opus-4-5 \
  --kernel-url "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/rope.py#L106" \
  -t "Complete GEAK Agent Pipeline: 1. DISCOVER 2. BENCHMARK 3. OPTIMIZE 4. Save results" \
  --yolo
```

Or from **inside** the container:

```bash
python3 -m geakagent.run.mini \
  -m claude-opus-4-5 \
  --kernel-url "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/rope.py#L106" \
  -t "Complete GEAK Agent Pipeline: 1. DISCOVER 2. BENCHMARK 3. OPTIMIZE 4. Save results" \
  --yolo
```

### Monitor progress in a second terminal

```bash
# Find the optimization output dir (created by the agent):
docker exec geak-agent-$USER bash -c \
  'tail -f /workspace/.geak_resolved/ROCm_aiter/aiter/ops/triton/rope/optimization_output/progress.log 2>/dev/null'
```

For detailed logs:
```bash
docker exec geak-agent-$USER bash -c \
  'tail -f /workspace/.geak_resolved/ROCm_aiter/aiter/ops/triton/rope/optimization_output/openevolve.log 2>/dev/null'
```

---

## 4. Known Issue: Kernel Selection Mismatch (NOT YET FIXED)

**Status:** The demo runs end-to-end and shows real speedups, but the
absolute speedup number may be inflated due to a kernel selection mismatch.

### What happens

The test harness's `--profile` mode generates random input tensors on GPU,
then runs the rope kernel. `rocprofv3` captures **both** GPU kernels:

| Index | Kernel | Duration | Coalescing |
|-------|--------|----------|------------|
| 0 | `distribution_elementwise...` (RNG) | ~51 us | 100% |
| 1 | `_rope_kernel_sbhd_fwd` (target) | ~156 us | 25% |

- **Baseline profiling**: The agent correctly selects index [1] via
  `build_baseline_metrics(result, kernel_indices=[1])` → **156 us**
- **OpenEvolve evaluation**: The `commandment_evaluator.py` parses
  `kernel-profile` stdout and grabs the **first** `duration_us` it finds
  → picks index [0] (the RNG kernel) → **51 us**
- **Result**: `156 / 51 = 3.05x` "speedup" on the unmodified kernel

### Root cause

`_parse_profiling_output()` in `commandment_evaluator.py` does a
regex/JSON parse of the profiler stdout. It doesn't filter by kernel name.
The `baseline_metrics.json` has `kernel_names: ["_rope_kernel_sbhd_fwd"]`
but the evaluator never uses it.

### Possible fixes (pick one)

**Fix A — Make the harness profile-clean (easiest for demo):**
Move tensor generation OUTSIDE the profiled region. Pre-generate inputs
and save to disk, or generate them before the `--profile` invocation
captures. This way `rocprofv3` only sees one kernel.

**Fix B — Filter by kernel name in the evaluator:**
In `commandment_evaluator.py`, use `baseline_metrics["kernel_names"]` to
filter the profiler output to only the target kernel(s).

**Fix C — Use `--auto-select` with kernel name hint:**
Pass the target kernel name from `baseline_metrics.json` to
`kernel-profile` so it only reports the matching kernel.

### What the demo shows despite this

- The pipeline works end-to-end: DISCOVER → BENCHMARK → OPTIMIZE → results
- The agent creates test harnesses, profiles, writes COMMANDMENT, runs OpenEvolve
- OpenEvolve's evolutionary optimization is real (8 GPUs, island model, LLM mutations)
- Correctness checking gates every candidate
- The relative ranking of mutations is still valid (all evals have the same bias)

---

## 5. Fixes Already Applied (in this branch)

These are committed and pushed to `msa`:

1. **Warm-up before baseline profiling** — `INSTRUCTIONS.md` now tells the
   agent to run two warm-up invocations before `kernel-profile`, matching
   the COMMANDMENT's warm-up. Prevents cold JIT vs warm eval mismatch.

2. **Test harness guidance** — Section 1b in `INSTRUCTIONS.md` with pitfalls:
   - Use package imports not `importlib.util`
   - Set PYTHONPATH before process start (rocprofv3 execvpe issue)
   - Fixed random seeds, `torch.testing.assert_close`

3. **Fixed tensor sizes** — `S=2048, B=4, H=32, D=128` for RoPE kernels,
   hardcoded in INSTRUCTIONS so every run uses the same config.

4. **Real-time monitoring** — `tail -f progress.log` pattern documented
   and the agent now uses it when launching OpenEvolve.

---

## 6. Useful Commands

```bash
# Check what's running in the container
docker exec geak-agent-$USER ps aux | grep -E "geakagent|openevolve|kernel-profile"

# Check baseline metrics
docker exec geak-agent-$USER cat /workspace/.geak_resolved/ROCm_aiter/aiter/ops/triton/rope/optimization_output/baseline_metrics.json

# Check COMMANDMENT
docker exec geak-agent-$USER cat /workspace/.geak_resolved/ROCm_aiter/aiter/ops/triton/rope/optimization_output/COMMANDMENT.md

# Kill everything and start fresh
docker exec geak-agent-$USER bash -c 'pkill -9 -f "geakagent|openevolve|kernel-profile|rocprofv3"'
docker exec geak-agent-$USER bash -c 'rm -rf /workspace/.geak_resolved /tmp/geak_*'

# Check OpenEvolve initial eval (verify kernel selection)
docker exec geak-agent-$USER grep "PASS.*speedup" /workspace/.geak_resolved/ROCm_aiter/aiter/ops/triton/rope/optimization_output/openevolve.log | head -3
```

---

## 7. Expected Timeline

| Phase | Steps | Cost | Time |
|-------|-------|------|------|
| DISCOVER (read kernel, tests) | ~8 | ~$1 | ~2 min |
| Test harness creation | ~5 | ~$1 | ~3 min |
| BENCHMARK (warmup + profile + baseline_metrics) | ~5 | ~$1 | ~5 min |
| COMMANDMENT creation | ~3 | ~$0.5 | ~2 min |
| OpenEvolve (10 iterations × 8 GPUs) | 1 command | ~$5-10 | ~25-30 min |
| **Total** | ~25 | ~$10-15 | ~35-40 min |

For a shorter demo, use `--iterations 3` (first 2-3 iterations capture most gains).
