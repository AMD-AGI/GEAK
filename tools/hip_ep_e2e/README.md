# GEAK end to end on hip-ep

GEAK's end-to-end loop for an OGA model served by hip-ep, the ONNX Runtime execution
provider for AMD GPUs, validated on Strix Halo (gfx1151) with Qwen3.8-27B int4. It
rebuilds the release's kernel library, measures TTFT and decode with an A/A of the rebuild,
profiles the prefill and decode forwards, and writes a report. The design is in
[DESIGN.md](DESIGN.md).

## Prerequisites

| Piece | Workflow argument (environment variable for the tools) |
|---|---|
| hip-ep release package (v0.5.3), unzipped | `hipep_pkg` (`HIPEP_PKG`) |
| TheRock dist the package pins (10.1.0) | `therock` (`HIPEP_THEROCK`) |
| Ubuntu 26.04 loader and runtime, for glibc 2.43 | `sysroot` (`HIPEP_SYSROOT`) |
| Python 3.14 venv with the package's OGA/ORT wheels | `py314` (`HIPEP_PY`) |
| hip-ep checkout at the release tag, and its build dir | `hipep_src`, `hipep_build` (`HIPEP_SRC`, `HIPEP_BUILD`). `build_kernels.sh` clones and configures them on first use (tag `HIPEP_TAG`, default `v0.5.3`; configure needs a Python with `lit`, `HIPEP_CMAKE_PY`, default `python3`) and resets the checkout to the tag around every build |
| An OGA export of the model | `model_path`; `-p int4 -e cuda --extra_options int4_accuracy_level=4`, provider `{"AMDGPU": {"profile": "hip"}}` |

## Run

With the Workflow tool, `scriptPath` = `hip_ep_e2e_workflow.js`:

```json
{"model_path": "/models/qwen3.8-27b-int4-oga019",
 "hipep_pkg": "/opt/hip-ep-0.5.3",
 "tools_dir": "<GEAK>/tools/hip_ep_e2e",
 "exp_root": "/work/hipep-runs",
 "therock": "/opt/therock-10.1.0",
 "sysroot": "/opt/sysroot-u2604",
 "py314": "/opt/hipep-py314/bin/python",
 "hipep_src": "/work/hip-ep-src",
 "hipep_build": "/work/hip-ep-build"}
```

Optional: `lens` (`1024,2048`), `gen` (128), `reps` (3), `rounds` (2), `aa` (`true`).
The run writes `report.md` and `workflow_return.json` into its eval dir.

Each tool also runs alone with the `HIPEP_*` variables set, e.g.
`python3 bench.py ab --model M --base A --cand B --out DIR`.

## Results on Qwen3.8-27B int4 (Radeon 8060S)

Orchestrated run of 2026-11-19 (two interleaved rounds of three repetitions per length
for the A/A, 128 tokens generated):

- **Baseline:** TTFT 2022.8 ms at 1024 prompt tokens and 4135.3 ms at 2048; decode
  70.37 and 70.67 ms/token.
- **A/A** (shipped library against the rebuild): the rebuild is 0.38% and 0.10% slower
  in TTFT, decode unchanged, logits bit-identical.
- **Profile:** the MatMulNBits WMMA GEMM is 82.0% of the prefill
  (`matmul_nbits_kernel.hip`); the dp4a GEMV is 93.0% of a decode step.

## Tests

```
python3 -m pytest tools/hip_ep_e2e/tests     # needs numpy and pyyaml
node tools/hip_ep_e2e/tests/test_workflow.js
```
