# GEAK end to end on hip-ep

GEAK's end-to-end loop for a model served by **hip-ep**, the ONNX Runtime execution
provider for AMD GPUs: the model is an OGA export, measured with OGA's
`model_benchmark`. The first target is Qwen3.8-27B int4 on gfx1151 (Radeon 8060S).

`e2e_workflow/e2e_workflow.js` drives Python serving stacks through `sitecustomize`
seams, a serving client and launch scripts. hip-ep is a C++ runtime; its kernels live
in one shared library, `libcustom_kernels_<arch>.so`, which the EP opens at session
creation. So the runtime-facing steps need their own tools, and this workflow runs
the loop over them.

## Runtime

| Piece | Choice | Why |
|---|---|---|
| hip-ep | the release package (v0.5.3), unzipped | Ships `model_benchmark`, OGA 0.19 and ORT 1.31, which the source build does not |
| ROCm | the TheRock dist the release pins (10.1.0) | The package links against it, and `THEROCK_DIST` is required at runtime |
| glibc | the package needs 2.43; the GEAK container has 2.35 | Its binaries run through an Ubuntu 26.04 loader in a sysroot: `ld-linux-x86-64.so.2 --library-path <sysroot>:<pkg>/lib:<therock>/lib <binary>`, so agents run hip-ep in their own container |
| Kernel builds | `cmake` of the release's source tag, `ninja custom_kernels_<arch>` | The arch is the one the package ships (v0.5.3: gfx11-generic). On Qwen3.8-27B a rebuild gives bit-identical logits and a TTFT within 0.5% of the shipped library's |

The machine paths come from `HIPEP_*` environment variables (`hipep_env.PATHS`); none
has a default. A package runs with its own `HOME` and temp dir: hip-ep keeps compiled
models in the temp dir, keyed by the model alone, and kernel changes can leave them stale.

## Tools

| Tool | Does |
|---|---|
| `hipep_env.py` | machine paths, the sysroot command line, per-package `HOME` and temp dir, the serving-GPU lock (`bench_e2e.sh`'s) |
| `bench.py` | `baseline.json`, or `ab.json`: two packages in interleaved (ABBA) rounds, medians, and the A/B gate |
| `parity.py` + `parity_dump.py` | `parity.json`: teacher-forced logits of two packages on the prefill and one-token decode paths, cut to the timed lengths; top-1 agreement and relative L2 |
| `kprof.py` | `heads.json`: rocprofv3 kernel shares of the prefill pass and the decode passes, with each kernel's source file |
| `build_kernels.sh` | the kernel library from the release tag plus an overlay of files under `lib/`, under the lock `tools/ep_task_runner.py` takes for EP builds |
| `make_pkg.sh` | a candidate package: the base package with its kernel library swapped |

A tool deletes its verdict file before it starts and writes it only on success, so a
failed rerun never leaves an earlier verdict in place.

`model_benchmark` runs with `--prompt_file`: the first L tokens of a fixed English text
(`data/parity_text.txt`). With `-l` alone it generates its prompt token by token,
untimed (about 145 s per run at 2048 tokens on 27B); `--use_random_tokens` fails
intermittently on Qwen3.8-27B ("Not enough image_grid_thw data for image tokens").

## Gates

A candidate package passes when all of these hold:

- **Timing:** one metric, TTFT or decode ms/token, improves by more than its noise
  band at every timed length, and the other is no worse than its band (1% for each
  metric). Each value is the median
  over two interleaved rounds; a round is one `model_benchmark` process of three
  repetitions and contributes its p50.
- **Parity:** top-1 agreement at least 99% and relative L2 at most 3e-2. hip-ep's own
  lookup-vs-online-autotune switch moves the logits by relL2 7.99e-3 on Qwen3.8-27B,
  so a 1e-2 bound would reject hip-ep's own autotune.

## Phases

| Phase | What happens |
|---|---|
| Setup | Check the export, package and device; rebuild the release library; record the baseline; A/A of the shipped library against the rebuild, for the rebuild's offset and a logits check |
| Profile | `kprof.py` → `heads.json` |
| Validate | The report: setup, baseline, profile |

A failed Setup step stops the run. Long steps (10–30 minutes) follow
`e2e_workflow.js`'s LONG MEASUREMENTS rule: started orphaned to init with `setsid
nohup`, waited on in bounded foreground calls, with the pid and exit code in files; a
step whose pid is alive is waited on, not started again. The validator returns the
report as text, and the workflow saves it as `report.md`.

## Testing

- `node tools/hip_ep_e2e/tests/test_workflow.js`: the workflow on stub agents (phase
  order, refusals, what reaches the tools and the report), and the long-command
  template run through bash with job control on, as the agents' shell has it.
- pytest (no GPU), `tools/hip_ep_e2e/tests/`: the `model_benchmark` parser and the
  gate, the parity comparison, the trace split, `build_kernels.sh` on a fake checkout,
  `make_pkg.sh`, `hipep_env.py`.
- On the GPU, every run starts with the A/A.
