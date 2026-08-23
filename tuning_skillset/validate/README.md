# Validation: which claims hold, in which image

Every skill here asserts things about the platform. This directory makes those
assertions re-runnable, because "validated" written on a diagram is a claim
about the author, not about the software.

```bash
python3 validate/claims.py                          # all applicable claims
python3 validate/claims.py --skill triton           # one skill
python3 validate/claims.py --json report.json       # machine-readable
```

Run it in the container you intend to tune in. It is read-only: nothing
installs, writes a config, or starts a server.

## Three outcomes, and why N/A is not a pass

| | meaning | what to do |
| --- | --- | --- |
| `PASS` | the claim holds here, with the observed value printed | nothing |
| `FAIL` | the claim is contradicted here | **the skill is wrong** — fix the skill, not the check |
| `N/A` | the precondition is absent (tool not installed, framework not in this image) | this image cannot answer; run the other one |

Collapsing `N/A` into `PASS` is the failure this whole set is written against.
A skill that says "build hipblaslt-bench, then race solutions by index" is not
confirmed by an image that ships without the tool — it is simply unanswered
there. Only once the tool exists and the documented flags behave as written is
the claim carried.

## Results — both images, both parts

Reports: [`report_vllm.json`](report_vllm.json),
[`report_sglang.json`](report_sglang.json),
[`report_vllm_gfx950.json`](report_vllm_gfx950.json),
[`report_sglang_gfx950.json`](report_sglang_gfx950.json).

| | vllm gfx942 | sglang gfx942 | vllm gfx950 | sglang gfx950 |
| --- | --- | --- | --- | --- |
| claims | 18 | 18 | 30 | 30 |
| PASS | 15 | 14 | 24 | 25 |
| FAIL | **0** | **0** | **0** | **0** |
| N/A | 3 | 4 | 6 | 5 |

Zero contradictions anywhere. The gfx942 runs predate twelve claims added while
verifying MI355; re-running them on gfx942 is worthwhile but not yet done, so
read those two columns as "of the 18 that existed then".

**Three further claims were added after a Qwen3-8B/MI355X tuning run, and they are a different
kind of claim from the rest: each one asserts a fact that this skillset had previously stated
*wrongly*.** Every column above predates them. They cover aiter's tuned-config logging (the hit
line is gated behind `AITER_LOG_TUNED_CONFIG` while the miss line is not, so the documented
engagement grep returned zero against a working deploy), the `lru_cache` on the lookup (so hit and
miss counts measure the diversity of M rather than call frequency), and the per-op quantized
tuner's interface (`-i/-o`, split-K off by default, a different result schema than gradlib). They
exist so that documentation drift of that kind fails a check rather than costing another run a day.

Two of the three read a source tree and so are checkable outside a container: they use the
installed `aiter` when present and otherwise fall back to a checkout at `/sgl-workspace/aiter`,
`/work_aiter`, or `~/tuning_workspace/libs/aiter`.

Two claims are source-tree reads rather than device probes — CK's runtime
instance gates and hipBLASLt's gfx950 logic-tree layout. They need a
`rocm-libraries` checkout, found automatically at `../libs/rocm-libraries` or
`/ws/libs/rocm-libraries`, or pointed at explicitly:

```bash
ROCM_LIBRARIES=/path/to/rocm-libraries python3 validate/claims.py --skill ck
```

Without a checkout they report `N/A`. They are in here rather than in prose
because both underpin numbers quoted in skills (166 gates; Origami at 83% of
566 files) and both will drift with the next ROCm release.

The `N/A`s on gfx950 are all missing preconditions, not missing answers:

| claim | vllm | sglang | why |
| --- | --- | --- | --- |
| `hipblaslt-bench --algo_method index` | N/A | N/A | neither image ships the client; both need the source build |
| seven CK tuners, none takes `--indtype` | N/A | **PASS** | the tuners ship in the aiter source checkout, not the wheel |
| gradlib has no `--libtype` flag | N/A | **PASS** | same |
| `ckProfiler` op list | **PASS** | N/A | apt-installed in the working vllm container |
| arch limits readable from torch | **PASS** | N/A | torch 2.9.1 in sglang does not expose LDS — see below |
| vLLM device name / `VLLM_TUNED_CONFIG_FOLDER` | **PASS** | N/A | no vllm in the sglang image |
| the three SGLang MoE claims | N/A | **PASS** | no sglang in the vllm image |

The gfx942 breakdown, for the six claims that existed then:

| claim | vllm | sglang | why |
| --- | --- | --- | --- |
| `hipblaslt-bench --algo_method index` | N/A | N/A | neither image ships the client; both need the source build |
| `ckProfiler` op list | **PASS** | N/A | apt-installed in the working vllm container; absent from a pristine sglang one |
| `VLLM_TUNED_CONFIG_FOLDER` is read | **PASS** | N/A | no vllm in the sglang image |
| SGLang MoE dirs keyed by Triton version | N/A | **PASS** | no sglang in the vllm image |
| `SGLANG_MOE_CONFIG_DIR` honored | N/A | **PASS** | same |
| arch limits readable from torch | **PASS** | N/A | see below — a real difference, not a missing framework |

Every framework-independent claim — the Triton knob trap, `HIPOptions`, the
aiter CSV schema, the `cu_num` key, TunableOp, rocprofv3, the relative-error
gate — passes in **both**. That is the property that matters: the method does
not depend on which serving framework is installed on top of it.

## What running it in both images actually found

**1. The arch limits are not equally readable.** `torch.cuda.get_device_properties`
exposes `shared_memory_per_block` on the vllm image's torch 2.10.0 and **not**
on the sglang image's torch 2.9.1. Same GPU, same 65536 B of LDS. A tuner that
reads the LDS budget from torch gets `0` on one of the two shipped images, and
`0` is the worst possible answer — it does not raise, and an LDS-fitting prune
against a budget of zero rejects every candidate. Read it from `rocminfo` or
`hipDeviceProp` when the number has to be right in both.

This is also why the check reports `N/A` rather than `PASS` there. An earlier
version of it defaulted the attribute to `0` and printed "gfx942: 304 CU, warp
64, 0 B LDS/workgroup" next to a green PASS — a validation harness reproducing
the exact silent-zero bug the skillset exists to catch.

**2. The aiter tuned tables are keyed by `cu_num`, and most rows cannot be
reached on this part.** 83 of 9964 rows in vllm, 83 of 4416 in sglang. Four
tables are entirely unreachable, including `a8w8_tuned_gemm.csv` at 0 of 582.
The tables exist, they are shipped, and they are mostly for other parts. The
identical 83 in both images is a good sign: the reachable set is a property of
the hardware, not the image.

That number also required fixing the check. Pointed at `bf16_tuned_gemm.csv`
as the skill's example suggests, it reported "0 of 0 rows" — that file ships
header-only in both images. Vacuously true, printed as PASS, and
indistinguishable from the real finding. It now scans every populated
`cu_num`-keyed table.

**3. aiter's version and shape differ, as documented.** Wheel `0.1.13` under
`dist-packages` in vllm; source checkout
`0.1.12.post2.dev150+ga6bb49937` at `/sgl-workspace/aiter` in sglang, with the
tuner tree in place. The sglang aiter is an *older* base version despite being
a newer checkout — one more reason to resolve the tuner to the installed
version rather than to the newest release.

**4. flydsl differs across the images** — 0.1.4 in vllm, 0.1.5 in sglang. Both
import and expose a Config surface, so the skill holds, but a FlyDSL config
space authored against one is not automatically valid against the other.

## What running it on gfx950 found

**5. The `cu_num` reachability inverts.** On gfx942, 83 of 9964 rows are usable.
On gfx950 it is 21 133 of 23 729 in the vllm image and 7509 of 10 574 in sglang.
aiter's CK tables are overwhelmingly tuned for MI355 — the opposite of what the
gfx942 result suggests, and a reminder that "the shipped tables do not cover my
part" is a per-part measurement, not a general property. Three families are still
empty at `cu_num=256` in both images: dense bf16, batched GEMM, and (in vllm)
generic fused-MoE.

**6. The two frameworks name the same GPU differently.** vLLM's
`current_platform.get_device_name()` returns `AMD Instinct MI355 OAM`; sglang's
`get_device_name()` returns `AMD Instinct MI355X`. Both go into config filenames
verbatim, so a tuned MoE config produced under one framework is invisible to the
other on the same machine. Compounding it, `torch.cuda.get_device_name()` returns
the correct name in the sglang image and the **empty string** in the vllm one, so
a helper that reads the name from torch is right in one container and silently
wrong in the other.

**7. aiter's shipped fused-MoE shape list cannot run on gfx950.** Eleven of its
thirteen rows specify `torch.float8_e4m3fnuz`, the gfx942 FP8 dialect. The first
one raises `KeyError` inside `gemm_moe_tune.py` and aborts the whole run, so the
default MoE tuning workflow yields zero tuned shapes on MI355 until the input is
filtered. Loud rather than silent, which is the good version of this bug.

**8. `SGLANG_MOE_CONFIG_DIR` pointed one level too deep raises rather than
misses.** The version-fallback scan calls `os.listdir(<dir>/configs)` unguarded,
so the server fails to start instead of quietly running untuned. The claim now
checks both layouts.

**9. CK changes its own candidate pool at runtime based on a device-name
string.** `library/src/tensor_operation_instance/gpu/` contains 166 gates of the
form `get_device_name() != "gfx950"` / `== "gfx950"` — 87 removing instances on
MI355 and 79 adding MI355-only ones. The removed blocks are introduced by the
comment `// instances not working on gfx950`. `gemm_universal` alone accounts for
45 exclusions. Nothing is logged, so "ckProfiler timed 126 instances" is not a
number you can compare across parts, and a tile that won on gfx942 may simply not
be in the pool.

**10. hipBLASLt's gfx950 solution logic is analytical, not tabular.** The gfx942
logic tree splits by CU count (`gfx942_20cu` … `gfx942_228cu`, 816 files). The
gfx950 tree splits by *selection strategy* and 471 of its 566 files — 83% — live
under `Origami/`, a separate monorepo project that picks tiles from a latency
model instead of a benchmarked table. There is no Origami directory for gfx942.
Measured consequence in `../tuning-hipblaslt/` §3b: the raced winner beats the
shipped default by only −2%…+14% on MI355, and loses on one shape in four.

## What is still not covered

These checks are static and read-only. They confirm that the routes exist, that
the documented flags and keys are real, and that the traps are still traps. On
gfx950 they now also exercise the MoE config **load path** in both frameworks —
planting a file and reading the hit or miss the lookup emits — but they still do
**not** stand up a server, drive traffic, and A/B serving metrics. That is the
bottom edge of the skill map, it is labelled as such, and closing it means
running a model in each image, which is a materially bigger job than this file.
