# HIP — Build Modes Reference

The HIP harness builder picks one of three build shapes based on what is
found in the kernel file. This document concentrates the build-shape
detection logic from `src/minisweagent/kernel_languages/hip/builder_hints.md`
plus the COMMANDMENT format rules from
`src/minisweagent/run/preprocess/INSTRUCTIONS.md` that constrain how the
build step is invoked.

---

## The three build shapes

| Shape | Detection marker | Harness pattern |
|---|---|---|
| **1. pybind11 / torch-extension wrapper** | `torch.utils.cpp_extension` or `pybind11::module` visible in the kernel file | Python-callable after a `torch.utils.cpp_extension.load_inline` or `load(...)` call. The harness imports the compiled module and invokes the Python-level function. |
| **2. Standalone `make` + `./bench`** | `Makefile` at `repo_root` + existing `./bench` binary | The harness shells out to the compiled binary and parses its stdout. |
| **3. Raw `hipcc` + host-side launcher** | Raw `__global__ void ...` without any Python binding | Same shape as (2) but compiled per-invocation. |

For shape 1, the wrapper usually exposes a PyTorch reference
implementation alongside the kernel call; use that for correctness.

For shapes 2 and 3, the user test file contains either a CPU reference
or a separate validation run; preserve that path.

---

## Build-step CLI patterns

### Shape 1: torch.utils.cpp_extension (recommended when applicable)

```python
from torch.utils.cpp_extension import load
mod = load(
    name="my_kernel",
    sources=["kernel.hip"],
    extra_cuda_cflags=["--offload-arch=gfx942", "-O3"],
    verbose=False,
)
out = mod.my_op(x, y)
```

The compile happens on first import; subsequent runs reuse the cached
shared object under `~/.cache/torch_extensions/`.

### Shape 2: external make

```bash
cd ${repo_root}
make -j$(nproc) bench
./bench --correctness
```

The harness wraps this with `subprocess.run([...], check=True)` and
parses the stdout for the latency line.

### Shape 3: raw hipcc

```bash
hipcc --offload-arch=gfx942 -O3 -std=c++17 \
    -I/path/to/includes \
    kernel.hip host_launcher.cpp -o ./bench
./bench --correctness
```

`--offload-arch=gfx942` targets MI300X. For MI250 use
`--offload-arch=gfx90a`. The `host_launcher.cpp` is the C++ file the
harness writes to drive the kernel under test.

---

## COMMANDMENT SETUP rules that apply to all three build modes

From `INSTRUCTIONS.md` section 4:

1. Five section headers are recognised: `## SETUP`, `## CORRECTNESS`,
   `## PROFILE`, `## BENCHMARK`, `## FULL_BENCHMARK`. Any other `##`
   header is flagged as an error by `validate_commandment`.
2. Commands run with `cwd=${GEAK_WORK_DIR}`.
3. Use `${GEAK_WORK_DIR}/kernel.py` (or `kernel.hip`, etc.) to reference
   the candidate — OpenEvolve writes the candidate there automatically.
4. Use `${GEAK_GPU_DEVICE}` instead of hardcoded GPU IDs.
5. Do NOT set or export `HIP_VISIBLE_DEVICES` — it is ALREADY SET in
   the environment by the scheduler. Use `${GEAK_GPU_DEVICE}` if you
   need the GPU ID.
6. Include TWO warm-up runs before actual profiling (Triton JIT
   compilation + GPU power ramp). This MUST match the warm-up used
   during baseline profiling — otherwise speedup numbers will be
   inflated.

---

## Wrapper-script pattern (CRITICAL for HIP)

`kernel-profile` passes the command to `rocprofv3` which uses
`execvpe`, NOT a shell. Therefore:

- Inline env-var prefixes (`HIP_VISIBLE_DEVICES=1 python3 ...`) are
  treated as the executable name and crash with `FileNotFoundError`.
- Shell built-ins (`cd`, `source`, `export`) as the first token also
  crash.

The fix is to write a small bash wrapper in `## SETUP` (single-line
`printf`, never a heredoc), then call that wrapper from `## CORRECTNESS`
and `## PROFILE`:

```
## SETUP
printf '#!/bin/bash\nexport PYTHONPATH=%s:${PYTHONPATH}\npython3 "$@"\n' "${GEAK_WORK_DIR}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh

## CORRECTNESS
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --correctness

## PROFILE
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true
kernel-profile "${GEAK_WORK_DIR}/run.sh ${GEAK_WORK_DIR}/kernel.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
```

The wrapper script sets `PYTHONPATH` inside the same process that runs
`python3` — `export` in a separate COMMANDMENT command does NOT
persist, because each command runs as its own subprocess.

---

## Compiler-flag tips (lifted from the legacy stack)

- `--offload-arch=gfx942` — MI300X target. Use `gfx90a` for MI250.
- `-O3` for the production benchmark; `-O0 -g` only when debugging
  correctness (slower but readable disassembly).
- For CK kernels: pass `-I${CK_INSTALL_DIR}/include` so the templates
  resolve. CK is template-heavy C++; after editing template parameters
  you must rebuild.

---

## When in doubt — defer to `kernel_languages/hip/builder_hints.md`

`builder_hints.md` is the source of truth for the build-shape decision
tree. This document concentrates the relevant excerpts so the
harness-generator subagent doesn't have to read multiple files; if the
two ever drift, treat `builder_hints.md` as authoritative.
