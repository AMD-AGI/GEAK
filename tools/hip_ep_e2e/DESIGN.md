# GEAK end to end on hip-ep

GEAK's end-to-end loop for a model served by **hip-ep**, the ONNX Runtime execution
provider for AMD GPUs: the model is an OGA export, measured with OGA's
`model_benchmark`. `e2e_workflow/e2e_workflow.js` drives Python serving stacks through
`sitecustomize` seams; hip-ep is a C++ runtime whose kernels live in one shared library,
`libcustom_kernels_<arch>.so`, which the EP opens at session creation. So the
runtime-facing steps have their own tools here (`hipep_env.py`, `bench.py`, `parity.py`,
`kprof.py`, `build_kernels.sh`, `make_pkg.sh`), and `hip_ep_e2e_workflow.js` runs the loop
over them: Setup, Profile, Strategize, Tuning, Validate. The validator re-measures the
final package against the release; a candidate it cannot reproduce is reported, not
returned as final.

## Runtime

| Piece | Choice | Why |
|---|---|---|
| hip-ep | the release package (v0.5.3), unzipped | Ships `model_benchmark`, OGA 0.19 and ORT 1.31, which the source build does not |
| ROCm | the TheRock dist the release pins (10.1.0) | The package links against it, and `THEROCK_DIST` is required at runtime |
| glibc | the package needs 2.43; the GEAK container has 2.35 | Its binaries run through an Ubuntu 26.04 loader in a sysroot, so agents run hip-ep in their own container |
| Kernel builds | `cmake` of the release's source tag, `ninja custom_kernels_<arch>` for the arch the package ships | On Qwen3.8-27B a rebuild gives bit-identical logits and a TTFT within 0.5% of the shipped library's |

A package runs with its own `HOME` and temp dir: hip-ep keeps compiled models in the temp
dir, keyed by the model alone, and kernel changes can leave them stale. Every candidate is
a rebuilt library, so Setup measures the release against its own rebuild (the A/A), which
is also the baseline.

## Gates

A candidate package passes when all of these hold:

- **Timing:** one metric, TTFT or decode ms/token, improves by more than its noise
  band at every timed length, and the other is no worse than its band. A metric's band
  is its largest delta in Setup's A/A, at least 1%. Each value is the median over two interleaved (ABBA) rounds; a round is one
  `model_benchmark` process of three repetitions and contributes its p50.
- **Parity:** top-1 agreement at least 99% and relative L2 at most 3e-2. hip-ep's own
  lookup-vs-online-autotune switch moves the logits by relL2 7.99e-3 on Qwen3.8-27B,
  so a 1e-2 bound would reject hip-ep's own autotune.

## Tuning lane

hip-ep picks kernel configs from autotune tables built into the kernel library, and the
shipped tables do not hold every shape a model runs (Qwen3.8-27B's gate_up shape,
N=34816, has no matmul_nbits row). Switching the same package to hip-ep's online autotune
picks extended WMMA configs, 1.14–1.26x per prefill shape, but a runtime switch is not
shippable. So `lut_refresh.py` reruns hip-ep's own sweep and `update_lut.py` for the model
on the device's own table, and rebuilds the library with every changed table:

| Op | Table picks configs for | Swept on |
|---|---|---|
| matmul_nbits | the prefill GEMM (`MatMulNBitsWMMA*`) | the model's 4-bit shapes (`extract_shapes.py`); a row changes only for a 3% win |
| gqa | flash prefill and decode attention (`gqa_flash*`) | the model's (heads, KV heads, head size), when hip-ep's sweep grid has it |

An op is tuned when its kernels hold at least 20% of the prefill or of a decode step:
on Qwen3.8-27B that is matmul_nbits (82% of the prefill; gqa is under 1%), on Qwen3-0.6B
both (gqa is 27% of a decode step).

A failed Setup step stops the run. Long steps follow `e2e_workflow.js`'s LONG
MEASUREMENTS rule.
