# GEAK end to end on hip-ep

GEAK's end-to-end loop for a model served by **hip-ep**, the ONNX Runtime execution
provider for AMD GPUs: the model is an OGA export, measured with OGA's
`model_benchmark`. The first target is Qwen3.8-27B int4 on gfx1151 (Radeon 8060S).

`e2e_workflow/e2e_workflow.js` drives Python serving stacks through `sitecustomize`
seams, a serving client and launch scripts. hip-ep is a C++ runtime; its kernels live
in one shared library, `libcustom_kernels_<arch>.so`, which the EP opens at session
creation. So the runtime-facing steps need their own tools, and this workflow runs
the loop over them. Two lanes: the tuning lane refreshes hip-ep's own autotune tables
for the model, and the kernel lane runs GEAK's unchanged `kernel_workflow` on a kernel
itself.

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
| `lut_refresh.py` | the kernel library rebuilt with hip-ep's autotune tables (matmul_nbits, gqa) refreshed for this model, over an optional overlay |
| `make_task.py` + `leaf_runner.py` | a GEAK kernel task for one op: the kernel sources (editable), two of the release's kernel unit tests as frozen oracles, and the `compile` / `correctness` / `performance` runner |
| `integrate.py` | an overlay from the kernel lane's final patch; refuses a patch that changes anything but the editable files, and leaves out new files an agent left outside `src/` |

A tool deletes its verdict file before it starts and writes it only on success, so a
failed rerun never leaves an earlier verdict in place.

`model_benchmark` runs with `--prompt_file`: the first L tokens of a fixed English text
(`data/parity_text.txt`). With `-l` alone it generates its prompt token by token,
untimed (about 145 s per run at 2048 tokens on 27B); `--use_random_tokens` fails
intermittently on Qwen3.8-27B ("Not enough image_grid_thw data for image tokens").

## Gates

A candidate package passes when all of these hold:

- **Timing:** one metric, TTFT or decode ms/token, improves by more than its noise
  band at every timed length, and the other is no worse than its band. A lane's gate
  takes each metric's band from Setup's A/A (`--noise-from`): its largest delta over
  the lengths, at least 1%. Each value is the median
  over two interleaved rounds; a round is one `model_benchmark` process of three
  repetitions and contributes its p50.
- **Parity:** top-1 agreement at least 99% and relative L2 at most 3e-2. hip-ep's own
  lookup-vs-online-autotune switch moves the logits by relL2 7.99e-3 on Qwen3.8-27B,
  so a 1e-2 bound would reject hip-ep's own autotune.

A failed gate rejects the candidate. A candidate the validator cannot reproduce
against the release is reported, not returned as the final package.

## Tuning lane

hip-ep picks a matmul_nbits config per shape from a lookup table built into the
kernel library, and the shipped table does not hold every shape a model runs
(Qwen3.8-27B's gate_up shape, N=34816, has no row). Switching the same package to
hip-ep's online autotune (`HIPDNN_MATMUL_AUTOTUNE_MODE=online`, one run) picks extended
WMMA configs, 1.14–1.26x per prefill shape. A runtime switch is not shippable, so the
lane refreshes the table: `lut_refresh.py` runs hip-ep's own sweep and `update_lut.py`
on the model's prefill shapes (from hip-ep's `extract_shapes.py`), keeps rows that win
by at least 3%, and rebuilds the library with the new table.

The same holds for every op hip-ep ships an autotune table for, so the lane is per op:

| Op | Table picks configs for | Swept on |
|---|---|---|
| matmul_nbits | the prefill GEMM (`MatMulNBitsWMMA*`) | the model's 4-bit shapes, from `extract_shapes.py` |
| gqa | flash prefill and decode attention (`gqa_flash*`) | the model's (heads, KV heads, head size), from `genai_config.json`, when hip-ep's sweep grid has it |

The lane refreshes the ops whose kernels hold at least 20% of the prefill or of a
decode step, on the device's own table (hip-ep loads the one for the arch it runs
on): on Qwen3.8-27B that is matmul_nbits (82% of the prefill; gqa is under 1%), on
Qwen3-0.6B both (gqa is 27% of a decode step). One overlay carries every changed
table, so one library, A/B and parity cover them together.

## Kernel lane

`make_task.py` describes each op it can write a task for (`OPS`: the kernel source and
the headers GEAK may edit, the target and frozen functions, the autotune, the oracle
leaves); the 4-bit MatMulNBits prefill GEMM is the one described. The lane runs on it
under the same 20% rule: `kernel_workflow` in optimize mode (`target_language: hip`)
on the task, with a wall-clock cap (`kernel_hours`: no round starts after it, and the
TechLead may stop earlier). Only a patch whose isolated speedup the lane's own
Director re-measured and accepted (`validation_status: accepted`, the bar
`kernel_lane.js` sets for the experience store) is integrated: `integrate.py` makes an
overlay on top of the tuning lane's, `lut_refresh.py` re-sweeps the op's tables on the
changed kernel, and the candidate goes through the same gates, against the tuning
lane's package. The task dir is `<op>_task`, so the lane's experience-store entries
are named after the op.

The task is a different cut of matmul_nbits than `tools/gen_ep_task.py`'s (the whole
`.hip` file editable, the `gemm_fp16u2` example harness, fixed flags, preset shapes),
in the same `config.yaml` format:

- **Oracles:** the release's own unit-test leaves. matmul_nbits `Xfp16_Wu4_Yfp16`
  checks the model's shapes, M=1 guards, its coverage sweep and every extended WMMA
  entry. qmoe_ragged_mm compiles the same editable WMMA core; its test also calls the
  matmul_nbits small-M grouped entry, so it links the task's matmul_nbits object.
  Generation refuses a release where another kernel compiles the core.
- **Flags:** each source compiles with the release build's own command line (`ninja
  -t commands`): `-O3`, where the leaves' Makefiles use `-O2 -ffp-contract=fast`.
- **Shapes:** the model's, weighted by calls per forward times baseline time.
- **Baseline:** timing runs the kernel's online autotune, so a config choice alone does
  not count as a kernel win; the tuning lane owns the table.
- **Not timed:** for some K (K=6144 on Qwen3.8-27B) production pads B's row stride and
  runs a padded-B variant the leaf does not have; only the A/B and the parity bound
  see an edit's effect on it.

`leaf_runner.py` is the task's runner, as `ep_task_runner.py` is for `gen_ep_task.py`'s
tasks, with the same modes and reports; it is kept apart so those tasks do not change.
It rebuilds before `correctness` or `performance` when the sources changed since the
last `compile`.

## Phases

| Phase | What happens |
|---|---|
| Setup | Check the export, package and device; rebuild the release library; record the baseline; A/A of the shipped library against the rebuild, for the lanes' noise band and a logits check |
| Profile | `kprof.py` → `heads.json` |
| Strategize | Each lane runs for the ops whose kernels hold at least 20% of the prefill or of a decode step |
| Tuning | `lut_refresh.py` → `make_pkg.sh` → A/B and parity against the release |
| HeadKernel | `make_task.py` → baseline → `kernel_workflow` |
| Integrate | `integrate.py` → `lut_refresh.py` on the changed kernel → `make_pkg.sh` → A/B against the current package, parity against the release |
| Validate | Re-measure the final package against the release independently; the report |

A failed Setup step stops the run. Long steps (10–30 minutes) follow
`e2e_workflow.js`'s LONG MEASUREMENTS rule: started orphaned to init with `setsid
nohup`, waited on in bounded foreground calls, with the pid and exit code in files; a
step whose pid is alive is waited on, not started again. The validator returns the
report as text, and the workflow saves it as `report.md`.

## Testing

- `node tools/hip_ep_e2e/tests/test_workflow.js`: the workflow on stub agents (phase
  order, when the lanes run, how their verdicts move the current package, what the
  kernel lane is handed, refusals, what reaches the tools and the report), and the
  long-command
  template run through bash with job control on, as the agents' shell has it.
- pytest (no GPU), `tools/hip_ep_e2e/tests/`: the `model_benchmark` parser and the
  gate, the parity comparison, the trace split, `build_kernels.sh` on a fake checkout,
  `make_pkg.sh`, `hipep_env.py`, `lut_refresh.py`'s shape view and overlays, the kernel
  task's layout, runner and QMoE link, and `integrate.py`'s refusals.
- On the GPU, every run starts with the A/A.
