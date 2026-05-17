# HIP — Harness Writing Deep Dive

This document concentrates the HIP-family harness pitfalls scattered
across the legacy stack:

- `src/minisweagent/run/preprocess/INSTRUCTIONS.md` section 1b language
  notes, section 1c COMMANDMENT wrapper-script rule, COMMANDMENT.md
  CRITICAL RULES.
- `src/minisweagent/run/preprocess/unit_test_agent.py`
  `_LANGUAGE_GUIDANCE["hip"]`, `["cuda"]`, `["ck"]`, `["asm"]` blocks.
- `src/minisweagent/kernel_languages/hip/builder_hints.md`.

The HIP knowledge base spans HIP, CUDA, Composable Kernel (CK), and
precompiled HSACO assembly because they all use the same C++ /
build-step toolchain.

---

## Per-kernel-type testing bullets

### HIP (C++ compiled with `hipcc`)

Lifted from `_LANGUAGE_GUIDANCE["hip"]`:

- This is a HIP kernel (C++ compiled with hipcc).
- A build step is REQUIRED before running tests.
- Use the project's build system (CMake / Makefile) or compile with
  `hipcc` directly.
- Use host-side validation (compare GPU output against CPU reference).
- Use `hipEventElapsedTime` or `torch.cuda.Event` for benchmarking.
- NEVER use `sys.path.insert(0, '/absolute/path/...')`. Rely on
  `PYTHONPATH` set by the COMMANDMENT SETUP section.

### CUDA (C++ compiled with `nvcc`)

Lifted from `_LANGUAGE_GUIDANCE["cuda"]`:

- This is a CUDA kernel (C++ compiled with nvcc).
- A build step is REQUIRED before running tests.
- Use the project's build system (CMake / Makefile) or compile with
  `nvcc` directly.
- Use host-side validation (compare GPU output against CPU reference).
- Use `cudaEventElapsedTime` or `torch.cuda.Event` for benchmarking.

### Composable Kernel (CK)

Lifted from `_LANGUAGE_GUIDANCE["ck"]`:

- This is a Composable Kernel (CK) kernel (C++ compiled with hipcc +
  CK includes).
- A build step is REQUIRED. Needs CK headers and `hipcc`.
- Template parameters (tile sizes, vector widths) are compile-time;
  test multiple configs.
- Use host-side validation against a reference GEMM / convolution.
- Use `hipEventElapsedTime` for benchmarking.
- NEVER use `sys.path.insert(0, '/absolute/path/...')`. Rely on
  `PYTHONPATH` set by the COMMANDMENT SETUP section.

### Precompiled HSACO assembly

Lifted from `_LANGUAGE_GUIDANCE["asm"]`:

- This is a precompiled HSACO assembly kernel.
- The assembly binary CANNOT be modified or recompiled.
- Test ONLY via the Python wrapper that loads and launches it.
- Use `torch.testing.assert_close` for correctness against a torch
  reference.
- Benchmark the wrapper launch, not the assembly directly.

### Section 1b language note (INSTRUCTIONS.md)

> **HIP/CUDA kernels (.cu, .hip, .cpp):** The test harness should still
> be a Python script that calls the kernel via its pybind11 binding
> (e.g., `torch.ops.aiter.my_kernel(...)`) or via ctypes. If no Python
> binding exists, create a C++ test that compiles with `hipcc` and
> outputs timing to stdout. Use `--correctness` and `--profile` flags.
>
> **Composable Kernel (CK):** CK kernels are template-heavy C++. After
> editing template parameters, rebuild with `hipcc` or `cmake`. The
> test harness should import the rebuilt module and call the kernel.
>
> **Assembly (HSACO):** HSACO binaries are precompiled. You cannot edit
> the assembly. The test harness should test the Python wrapper's
> launch config (grid dims, block dims, shared memory size).

---

## The "no `sys.path.insert` for HIP / CK" rule

The single most common HIP harness failure mode: someone writes

```python
import sys
sys.path.insert(0, "/absolute/path/to/built/module")
import my_hip_module  # noqa: E402
```

…because it worked locally. It does NOT work under the v3 pipeline,
because:

1. `rocprofv3` re-execs through `execvpe()` without inheriting the
   in-process `sys.path` mutation.
2. The COMMANDMENT SETUP section is the authoritative place to set
   `PYTHONPATH` for the whole eval process tree.

Fix: rely on `PYTHONPATH` set by the COMMANDMENT SETUP section. For
local debugging, set `PYTHONPATH` in your shell before invoking the
harness, NOT inside the harness.

---

## Wrapper-script-not-inline-env rule (section 1c)

`kernel-profile` passes the command to `rocprofv3` which uses
`execvpe`. This means:

- Inline env-var prefixes (`HIP_VISIBLE_DEVICES=1 python3 ...`) are
  interpreted as the executable name and crash with `FileNotFoundError`.
- Shell built-ins (`cd`, `source`, `export`) as the first token also
  crash.

The fix is the **wrapper script pattern** — in the COMMANDMENT `## SETUP`
section, write a small bash script that sets env + execs python3, then
call that wrapper from `## CORRECTNESS` and `## PROFILE`:

```bash
#!/bin/bash
export PYTHONPATH=/path/to/repo:$PYTHONPATH
python3 /path/to/test_harness.py "$@"
```

Use `printf` (NOT a heredoc) to write the wrapper on a single line, as
the COMMANDMENT parser splits lines into separate commands.

### Why a wrapper script is REQUIRED

The COMMANDMENT evaluator runs each command as a separate subprocess.
`export PYTHONPATH=...` in one command does NOT persist to subsequent
commands. A wrapper script solves this by setting the environment
inside the same process that runs `python3`.

---

## `hipDeviceSynchronize` ordering rules

For HIP timing loops with `hipEventElapsedTime`:

- Warm up 5 iterations before the timed run; measure 100 and take the
  **median** (not mean).
- `hipDeviceSynchronize()` before AND after each measurement window so
  the recorded interval contains only the kernel under test (no
  outstanding queued work, no later launches counted in).
- When using a pybind11 wrapper that internally calls
  `hipDeviceSynchronize`, you still need to bracket your measurement
  with explicit syncs on the host side — the wrapper's internal sync
  applies to its own stream, not the timing stream.

---

## Three HIP build shapes (very brief — full detail in `hip_build_modes.md`)

Lifted from `kernel_languages/hip/builder_hints.md`. The builder picks
one based on what is in the kernel file:

| Marker found in kernel file | Build shape |
|---|---|
| `torch.utils.cpp_extension` or `pybind11::module` | pybind11 / torch-extension |
| `Makefile` at `repo_root` + existing `./bench` binary | standalone `make` |
| Raw `__global__ void ...` without any Python binding | raw `hipcc` per-invocation |

For each shape:

- **Shape 1 (pybind11):** the harness imports the compiled module and
  invokes the Python-level function. The wrapper usually exposes a
  reference implementation (PyTorch fallback); use it for correctness.
- **Shape 2 (make + ./bench):** the harness shells out to the compiled
  binary and parses its stdout. The user test file contains either a
  CPU reference or a separate validation run; preserve that path.
- **Shape 3 (raw hipcc):** same shape as 2 but compiled per-invocation.

---

## Three-tier shape lists + sampling

Same rules as Triton — the harness must define three shape lists at the
top of the script:

- **`ALL_SHAPES`** — every unique shape from discovered test files.
- **`HARNESS_SHAPES`** (20-25) — uniformly sampled.
- **`PROFILE_SHAPES`** (5) — evenly-spaced, prevents OOM under
  `rocprofv3`.

Shape routing by CLI mode:

| Flag | Shape set |
|---|---|
| `--profile` | `PROFILE_SHAPES` (5) |
| `--benchmark` | `HARNESS_SHAPES` (20-25) |
| `--correctness` | `HARNESS_SHAPES` |
| `--full-benchmark` | `ALL_SHAPES` |

Default fallback shapes when discovery returns nothing:

- **Attention / RoPE:** `S=2048, B=4, H=32, D=128` (fp16)
- **GEMM:** `M=1024, N=1024, K=1024` (fp16)
- **Elementwise / pointwise:** at least 16M elements

---

## Dtype preservation (per-tensor execution contract)

Preserve the source benchmark/test's full execution contract:

- Keep per-tensor semantics independent: **dtype**, **device**, **layout**,
  **contiguity**, **index dtypes**, **auxiliary buffers / caches /
  scales**, and any helper-side preprocessing.
- Do NOT normalise every tensor to the main activation dtype just
  because the benchmark uses that dtype for `query`, `key`, or other
  primary activations.

---

## General harness rules (HIP-flavoured)

- Use `torch.testing.assert_close` for correctness, NOT manual
  `torch.allclose` with always-pass fallbacks.
- Fixed random seed: `torch.manual_seed(42)`.
- Keep the harness file OUTSIDE the kernel directory.
- GPU event-based timing ONLY: `hipEventElapsedTime` or
  `torch.cuda.Event(enable_timing=True)`. NEVER `time.time()` or
  wall-clock.
- WARMUP = 50, ITERATIONS resolved from `--iterations` then
  `GEAK_BENCHMARK_ITERATIONS` (default 200). Report MEDIAN per shape.

---

## GEAK_SHAPES_USED (determinism contract)

In ALL modes, after the loop, print:

```
GEAK_SHAPES_USED=<list of config indices>
```

Use the config INDEX, not the config values. Two independent runs must
choose the EXACT same cases for `--benchmark`, `--correctness`, and
`--profile`. Preserve the source file's ordered full case stream
exactly — equivalent values are NOT sufficient if a different order
causes `_pick()` to choose a different subset.
