# Coverage on gfx950 / MI355X: what is verified, what is not, and what is missing

The skill map claims eleven skills over six peer backends. This document says,
per backend, which of its claims have been **measured on gfx950** and which are
gfx942 measurements that were being quoted as general facts. It exists because
the answer to "are we there yet" was no, and the honest form of that answer is
a table rather than a paragraph.

## 0. The short answer

**Every skill and every backend is now exercised on MI355.** All seven peer
backends run, all seven aiter CK offline tuners run, both framework integration
paths have had their config load path reproduced on-hardware, the full 22-case
corpus has been re-swept on the part, and the 30-claim validator passes in both
images with zero contradictions. Nothing in the skillset is now carried across
from gfx942 without being marked as such.

The single most consequential finding is not a coverage number, it is a
measurement one: **on gfx950 the corpus's back-to-back A/B timing was reporting
largely unrelated numbers**, manufacturing three wins that do not exist and
hiding seven that do (§9). Every result in this document was re-measured with
interleaved timing after that was found.

**Coverage of the libraries is a different question, and there the answer is no.**
Three numbers, none of which is 100%:

| question | answer |
| --- | --- |
| aiter tunable ops with a corpus case | **18 of 74 (24%)**, and of the 56 uncovered only 4 are blocked by anything but effort (§13) |
| aiter CK offline tuners run on gfx950 | **7 of 7**, of which 1 crashes on its own input and 1 installs regressions |
| shipped tuned rows usable at `cu_num=256` | 21 133 of 23 729 (vllm), 7509 of 10 574 (sglang) |
| CK ops raced (of 39 GEMM-family in `ckProfiler`) | 1 measured; method covers the rest |
| tuning surfaces in `rocm-libraries` with **no skill at all** | **2** — CK `tile_engine` and hipBLASLt/TensileLite kernel *generation* (§11) |

And the coverage picture inverts depending on which mechanism you look at, which
is the single most useful thing in this document:

- **aiter CK tables (CSV, keyed on `cu_num`)** — MI355 is *well* covered for fp8
  and FP4 dense GEMM, and *empty* for dense bf16, batched GEMM, and generic
  fused-MoE (§3).
- **aiter Triton tables (per-arch JSON)** — MI355 has 168 files against gfx942's
  44, and three op families exist only for gfx950 (§3).
- **framework fused-MoE configs** — vLLM ships 2 usable files for this device,
  SGLang ships **0** (§6).

The two real tuner wins found here came from exactly the places the empty tables
predicted: batched a8w8 GEMM (6.78%) and fp8 fused MoE (**1.73x**). The
large-square-GEMM shapes everyone benchmarks returned nulls, because those are
the shapes the shipped tables already cover.

Sections 1-9 are the evidence. §7 is the ledger of what is still not measured.

Measured on job 8545, node `crsuse2-m2m-040`, 8x MI355X (gfx950, 256 CU,
163840 B LDS/workgroup), in the two images `docker_select.json` names for
`MI355`:

| framework | image | ROCm | torch |
| --- | --- | --- | --- |
| vllm | `primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix` | 7.2.2 | 2.10.0 |
| sglang | `primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix` | 7.2.0 | 2.9.1 |

Note the sglang image: `docker_select.json` maps `MI300` to the `mi30x` tag and
`MI355` to `mi35x`, and every measurement previously in `tuning-in-sglang` was
taken in the **`mi30x`** image. Those are different builds, so the earlier
numbers were not merely from another part, they were from another container.

## 1. Backend-by-backend verification status

| backend | tool reachable on gfx950? | numbers re-measured? | what changed |
| --- | --- | --- | --- |
| **hip** | yes -- `hipcc --offload-arch=gfx950` compiles, resource-usage remarks emit | yes | async-timing trap is shape-dependent here, not universal |
| **ck** | yes -- `apt install composablekernel-ckprofiler`, ~1 min, ships gfx950 instances | yes | 546 -> 1245 TFLOPS; same winning instance |
| **hipblaslt** | yes -- built `-a gfx950`, ~10 min | yes | 1231 -> 2085 solutions; race/replay gap 5.5% -> 20.1% |
| **triton** | yes | yes | +7.7% -> +4.0% on nonkdim=16; 3.75x -> 2.45x on `key=` |
| **flydsl** | yes -- 0.1.4 (vllm) / 0.1.5 (sglang) | yes, first time | had no measurements of its own before; 880 TFLOPS at 4096^3 |
| **aiter** | partly -- wheel in vllm, source in sglang | yes | gradlib tuner runs, but two documented flags do not exist |
| **aiter CK tuners** | yes -- source only, so sglang image | **7 of 7** run end to end | seven exist, not nine; one crashes on its own input, one installs regressions, two found real wins |
| **in-vllm** | yes | yes -- load path exercised | 2 usable MoE files; 5 of 7 MI35x configs unreachable by name |
| **in-sglang** | yes | yes -- load path exercised | **0** usable MoE files; override dir shadows shipped tree; one wrong path raises |

## 2. The cross-backend race, one shape, one clock

All six on 4096^3 bf16, gfx950, interleaved so that clock drift is charged
equally (see §5 -- this is not a detail):

| backend | TFLOPS | how obtained |
| --- | --- | --- |
| hipBLASLt, solutions raced | 1553 | `hipblaslt-bench --algo_method all` (but see below) |
| hipBLASLt, winner replayed | 1241 | `--algo_method index --solution_index 441281` |
| aiter tuner, hipBLASLt libtype | 1328 | `gradlib/gemm_tuner.py`, solidx 440518 |
| torch (hipBLASLt heuristic) | 1285 | `A @ B` |
| CK, best of 126 instances | 1245 | `ckProfiler gemm 2 1 1 1 0 1 ...` |
| FlyDSL | 880 | `aiter.ops.flydsl.flydsl_hgemm` |
| aiter Triton `gemm_a16w16` | 810 | default config |

Two things to read out of that. hipBLASLt wins, and the ordering matches
gfx942 (where CK's 546 lost to hipBLASLt's 633) -- so the *ranking* transfers
even though no number does. And the top two rows are the same solution
measured twice, 20% apart, which is the subject of §5.

For comparison, the gfx942 figures the skills previously quoted: CK 546,
hipBLASLt 633. The gfx950/gfx942 ratio is 2.28x for CK and 2.45x for
hipBLASLt, against a 2.09x ratio in bf16 matrix-core peak.

## 3. aiter: the op-coverage gap, counted

`tools/lib_inventory.py` AST-scans the installed aiter for public functions
that expose a tuning surface, and diffs them against the `op:` field of every
case. On the vllm image:

```
family               tunable ops  covered
gemm_basic                    17        7
gemm_batched                   6        2
gemm_fused                    11        1
gemm_ff                        4        1
gemm_grouped                   3        1
moe                            7        2
attention                     15        1
quant                          2        0
comms                          1        0
other                          8        3
TOTAL                         74       18      24.3%
```

**24%**, not 100%. It was 16.2% before this pass; the six cases added below
account for the difference. The denominator excludes config-builder helpers
that take a `config` argument but are not ops (`get_gemm_config`,
`compute_splitk_params`, the `aot/` serialisers) -- counting those gave 83 and
made the gap look worse than it is.

Where the remaining 60 sit, and why it matters differently per family:

* **attention, 15 ops, 1 covered.** Was 0, the largest untouched family. The
  corpus's two attention cases call `paged_attention_decode` and
  `paged_attention_ragged`, which the scan attributes to `other`, so the extend
  and flash families were genuinely untested. `attn_extend_prefill` (new, below)
  covers `extend_attention_fwd`. Remaining: `flash_attn_func` and
  `flash_attn_varlen_func`, the `fav3_sage` family, `pod_attention`,
  `persistent_lean_attention`, MLA-with-rope decode, and the HSTU pair. Three
  are FP4 (`fav3_sage_mxfp4_*`), so gfx950-only.
* **moe, 7 ops, 1 covered.** Was 0. The corpus's two pre-existing MoE cases go
  through **vLLM's** `fused_experts`, a different kernel with a different
  config schema read from a different directory, so they left aiter's own MoE
  entirely untested. `moe_aiter_fused` (new, below) covers `fused_moe`;
  `fused_moe_mxfp4`, `fused_moe_silu`, `fused_moe_gelu`,
  `fused_moe_mxfp4_silu`, `e2e_moe` and `routing_sigmoid_top1` remain.
* **gemm_fused, 11 ops, 1 covered.** Ten fusions, six of them FP4.
* **FP4/MXFP4 across families:** `gemm_a8wfp4`,
  `gemm_afp4wfp4_preshuffle`/`_pre_quant`/`_preshuffled_scales`,
  `batched_gemm_a16wfp4`, `batched_gemm_afp4wfp4`(`_pre_quant`),
  `fused_gemm_afp4wfp4_*` (4), `fused_fp4_bmm_rope_cat_and_cache_mla`,
  `fused_moe_mxfp4`(`_silu`), `fav3_sage_mxfp4_*`. This is the gfx950-only
  surface and it is the least covered part of the corpus, because it could not
  be exercised at all on the part the corpus was built on.

### aiter's offline CK tuners — seven, not nine

Separate from the op surface, aiter ships CK-based offline tuners in its source
tree (absent from the vllm wheel, present in the sglang checkout). An earlier
draft of this document said nine. A `find` over the tree says **seven**:

**All seven now run on gfx950**, two shapes each:

| tuner | result on gfx950 |
| --- | --- |
| `ck_batched_gemm_bf16` | works; null (0.05%, 2.11%), errRatio 0.0 |
| `ck_gemm_a4w4_blockscale` | works; null (−0.13%, 0.34%); 4993 TFLOPS FP4 |
| `ck_gemm_a8w8_blockscale` | works; null (0.45%, −0.15%) |
| `ck_gemm_a8w8_bpreshuffle` | works; **4.53% win** at M=16 |
| `ck_batched_gemm_a8w8` | works; **6.78% win** at B=32 M=1024 |
| `ck_gemm_a8w8` | works but **gate broken — writes a 0.82x regression** |
| `ck_gemm_moe_2stages_codegen` | **crashes on its own shipped input**; once filtered, 4.23% bf16 and **1.73x fp8** |

Two results are worth separating from the rest. The wins did not come from the
big dense shapes — those returned nulls, because the shipped tables already cover
them. They came from exactly the families §3 showed to have **zero `cu_num=256`
rows**: batched GEMM (6.78%) and fused MoE (1.73x on the fp8 path, 390 → 218 µs
at 818 TFLOPS). The empty-table analysis predicted where the headroom was, and
the tuners confirmed it.

**`gemm_moe_tune.py` cannot run with its default input on this part.** Eleven of
the thirteen rows in `aiter/configs/untuned_fmoe.csv` specify
`torch.float8_e4m3fnuz` — the gfx942 FP8 dialect. The first one raises
`KeyError: torch.float8_e4m3fnuz` and the exception aborts the entire run,
including shapes already enumerated; the summary reads `tune 0 shapes`. Filter
with `grep -v fnuz` first. The shipped shape list is a gfx942 artifact, which is
also why re-deriving shapes for this part is where the 1.73x came from.

Two details from that run: bf16 MoE finds **no ASM kernels at all** on gfx950
(`hsa/gfx950/fmoe_2stages/fmoe_stage1_bf16_pertoken_g1u1.csv` does not exist), so
it races 15 CK candidates against 0 ASM, while the fp8 path gets 18 ASM plus 25
CK. And the tuner accepted a candidate at `max abs delta 0.625, 4.4% of elements`
— a pass under the default `--errRatio 0.05`. Tighten it if that matters.

The two that were listed and do not exist are `csrc/gemm_a16w16/` and
`csrc/opus_gemm/`. There is **no dense bf16 CK tuner** — bf16 goes through
gradlib. Also, none of the seven has an `--indtype` flag, so the gradlib bug in
§4 does not apply to them; the earlier guess that it would was wrong.

The intended entry point is not the scripts but
`python3 aiter/utility/pretune.py {--list|all|<module>}`, which auto-detects arch
and `cu_num`, tags the results, and rebuilds the inference `.so` with the winners.
It lists eight modules for seven scripts because the two `cktile` variants share
their parent's tuner.

**Trap in `pretune.py --list`:** it prints `AITER_CONFIG_<OP>_FILE`, which is the
resolver property, not the environment variable. The variable has no `_FILE`.
Exporting the printed name changes nothing and warns about nothing:

| exported | resolves to |
| --- | --- |
| nothing | `…/configs/bf16_tuned_batched_gemm.csv` |
| `AITER_CONFIG_BF16_BATCHED_GEMM=/work/mine.csv` | `/work/mine.csv` |
| `AITER_CONFIG_BF16_BATCHED_GEMM_FILE=/work/mine.csv` | `…/configs/bf16_tuned_batched_gemm.csv` |

#### The `--compare` gate, and the hole in it

These tuners have a built-in before/after check that the rest of the ecosystem
lacks: benchmark the production op with default dispatch, tune, benchmark again,
and refuse to write anything that wins by less than `--min_improvement_pct`
(default 3%). On gfx950 it behaved correctly in three of the four tuners run:
batched bf16 at 0.05% and 2.11%, a4w4 at −0.13% and 0.34%, all skipped;
bpreshuffle skipped −0.52% and passed 4.53%.

**Correction (2026-08-17).** This section previously read: "Null results are the
normal outcome on large shapes: aiter's default CK dispatch on gfx950 is already
at or near the tuned optimum there." That was generalized from square probe
shapes near 4096³ and is false at serving shapes. On
`a8w8_blockscale_bpreshuffle`, M=15104 N=34816 K=5120, gfx950: default 4153.1 µs
/ 1297 TFLOPS against a tuned 2655.8 µs / 2028 TFLOPS through the production
wrapper — **−36%**, and **+23.88% e2e throughput** (TTFT −30.8%, TPOT −18.0%)
once the full harvested shape list was tuned. The distinguishing property is not
size but shape: tall, thin, non-square, at an M the engine actually dispatches.
The 4.53% seen here is the size to expect **from these probe shapes**, not from
the workload.

Also note `--compare`'s post-tune leg measures the production op, so on a build
whose wrapper cannot select a tuned instance (`../tuning-aiter/` §2b) a null from
it may be an artifact of the runtime rather than an absence of headroom.

`gemm_a8w8_tune.py` is the exception, and it fails in the dangerous direction.
On `M=1, N=7168, K=8192` fp8:

```
(1, 7168, 8192, torch.float8_e4m3fn)  |  14.78 |  17.97 |  0.82x |  OK
Total shapes: 2 | Updated: 2 (improved: 0, new: 2) | Skipped: 0
--- Updated (2 shapes) ---
(1, 7168, 8192, torch.float8_e4m3fn)  |  14.78 |  17.97 |   N/A  |  NEW
```

It measured a 22% slowdown, printed it, called it `OK`, reported the improvement
as `N/A`, classified the shape `NEW`, and with `--update_improved` wrote the row.
`NEW` means "no baseline to compare against", which is the one case the threshold
is allowed to skip — but the baseline is printed on the same line, and the shape
already exists in `a8w8_tuned_gemm.csv` (582 rows, including this one). The
pre-run and post-run failed to line up in the gate and both sides were treated as
baseline-less.

Specific to this tuner, not to the shared `base_tuner.py`: `gemm_a8w8_bpreshuffle`
has the same key columns including the string `q_dtype_w`, ran the same two
shapes, and gated correctly. An earlier guess that the string key column was the
cause is not supported.

The slowdown itself is real and repeatable — 0.79x, 0.82x, 0.85x, 0.86x over four
runs. The best CK instance for that shape is genuinely slower than what aiter
dispatches by default, so tuning the CK table pins the op to CK and loses. A tuner
can only pick among the candidates it owns; it cannot decline in favour of another
backend. Worth generalising: **a tuner finding its own best answer is not the same
as that answer being the best available.**

Read the Speedup column yourself and delete sub-1.00x rows before deploying.

One artifact: the first `--compare` run after a JIT build reported 27.63 µs where
the next three reported 16.98, 17.50, 17.43 — 59% high. Discard the first run.
Pre and post are also run as two blocks, all pre then all post, which is the
back-to-back pattern §5 shows to be unsafe on this part.

### Are the shipped CK tables tuned for MI355 at all?

Better than expected, and unevenly. aiter keys tuned GEMM configs on `cu_num`, so
a row tuned at 304 CU contributes nothing here. Counting rows at `cu_num=256`:

| tuned CSV | vllm image | sglang image |
| --- | --- | --- |
| `a4w4_blockscale_tuned_gemm.csv` | 1470 | 1470 |
| `a8w8_blockscale_tuned_gemm.csv` | 6630 | 5 |
| `a8w8_tuned_gemm.csv` | 556 | 556 |
| `a8w8_bpreshuffle_tuned_gemm.csv` | 481 | 481 |
| `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 58 | 58 |
| `bf16_tuned_gemm.csv` | **0** | **0** |
| `bf16_tuned_batched_gemm.csv` | **0** (26 rows, all 304) | **0** |
| `a8w8_tuned_batched_gemm.csv` | **0** (26 rows, 80/304) | **0** |
| `tuned_fmoe.csv` | **0** (622 rows, all 80) | 751 |
| plus `model_configs/` overlays | ~11 900 | ~4 100 |
| **total usable at cu_num=256** | **21 133** of 23 729 | **7 509** of 10 574 |

So "is aiter tuned for MI355" is mostly yes for the fp8 and FP4 dense paths, and
flatly no for three families:

- **dense bf16** — `bf16_tuned_gemm.csv` is empty in both images, for every
  device. Everything bf16 comes from the `model_configs/` overlays or from
  nothing. This is the shape class the corpus benchmarks most, and it explains
  why the gradlib tuner finds real headroom there.
- **batched GEMM, bf16 and a8w8** — zero rows at 256 CU in both images. The
  untuned CSVs list 26 shapes each, and the tuner works (verified above). This is
  the cheapest concrete gap to close in the whole inventory.
- **generic fused-MoE in the vllm image** — all 622 rows are `cu_num=80`. The
  sglang image has 751 rows at 256 for the same file. Same op, same aiter
  version, opposite coverage depending on which container you are in.

The two images are not interchangeable more generally: 23 729 rows against
10 574, and `a8w8_blockscale_tuned_gemm.csv` differs by three orders of magnitude
(6630 against 5). A conclusion drawn in one image does not describe the other.

### aiter's Triton config tables: gfx950 is the better-covered part

The CK story above is about CSVs keyed on `cu_num`. aiter's Triton ops use a
different mechanism — per-arch JSON in `aiter/ops/triton/configs/`, read by
`_get_config` whenever a caller passes `config=None`. Counting those files
inverts the expectation:

| op family | gfx942 | gfx950 |
| --- | --- | --- |
| `gemm/GEMM` | 24 | **127** |
| `gemm/BATCHED_GEMM` | 5 | 8 |
| `gemm/FUSED` | 1 | 5 |
| `moe/MOE` | 5 | 6 |
| `gemm/GEMM_PREQUANT` | 0 | 2 |
| `gemm/BATCHED_GEMM_PREQUANT` | 0 | 3 |
| `gemm/gluon/GEMM` | 0 | **9** |
| `LEANATTN` | 1 | **0** |
| others (`EXTEND_ATTENTION`, `MHA`, `MLA_DECODE_ROPE`, `GMM`, `FF`, `HSTU_*`, `MOE_ROUTING_*`) | 1 each | 1 each |
| **total files** | **44** | **168** |

So for the aiter Triton path, MI355 is not the neglected part — it has 3.8x the
tables, and three families (`GEMM_PREQUANT`, `BATCHED_GEMM_PREQUANT`, and the
nine Gluon GEMM configs) exist only for gfx950. Any general claim that "gfx950 is
undertuned" is wrong here and should be made per mechanism, not per part.

Two specifics that do not follow the trend:

- **`LEANATTN` has a gfx942 table and no gfx950 one.** The only op in the set
  that regressed in coverage.
- **Two gfx950 tables are byte-identical to their gfx942 twins** —
  `BATCHED_GEMM-A16W16.json` and `FUSED-GEMM-A8W8_BLOCKSCALE-A16W16.json`. Those
  are copies, not retunes. They present as coverage to anything that counts
  files, and given that no other measurement in this document transferred
  unchanged between the two parts, a config that did is a claim worth
  distrusting. Everything else shared between the arches was genuinely retuned.

## 4. Two documented aiter commands that do not work

Both found by running the workflow rather than reading it.

**`--libtype` does not exist.** `tuning-aiter` §4 documents

```bash
python3 gradlib/gradlib/gemm_tuner.py ... --libtype hipblaslt,torch,triton
```

and builds a table of the six libtypes around it. `gemm_tuner.py` has no such
flag -- not at the commit `clone_libs.sh` pins (`0ba802e2`, 2026-07-27) and not
at the one the sglang image ships (`a6bb4993`, 2026-04-29). `libtype` is a
**column in the tuned CSV** and an internal filter argument in
`chip_info.build_tune_dict`; it was never a CLI option in either version. The
race across backends does happen, and the winning libtype is reported per row
-- it is just not selectable from the command line.

**`--indtype` crashes.** `gemm_tuner.py:121` converts the CLI string to a torch
dtype in place:

```python
indtype = get_dtype(args.indtype)   # "bf16" -> torch.bfloat16
args.indtype = indtype
```

and `GemmTuner.pre_process` (`GemmTuner.py:594`) then looks the *string* up
again:

```python
self.untunedf["dtype"] = f"dtypes.{_cli_to_dtypes[args.indtype]}"
# KeyError: torch.bfloat16
```

So passing `--indtype` always raises. The two files disagree about the type of
`args.indtype`, which makes this a bug in aiter rather than in the skill, but
the skill's command line cannot run as written. **Workaround: omit
`--indtype`/`--outdtype` and put the dtype in the input CSV's `dtype` column**,
which leaves `args.indtype` as `None` and skips the conversion.

With that, the tuner runs on gfx950. 2 shapes, 120.2 s:

```
gfx950,256,4096,4096,4096,...,hipblaslt,440518,0,103.5316,...MT256x256x64_MI16x16x1...ISA950...,0.0,1327.51,972.3
gfx950,256,   1,4096,4096,...,hipblaslt,439841,0,  9.4608,...MT16x16x512_MI16x16x1...ISA950...,0.0,   3.55,3548.41
```

Against the gfx942 rows the skill quotes:

| | gfx942 | gfx950 |
| --- | --- | --- |
| 4096^3 solution index | 198969 | 440518 |
| 4096^3 macro-tile | `MT256x224x64` | `MT256x256x64` |
| 4096^3 | 563.9 TFLOPS | 1327.5 TFLOPS |
| M=1 macro-tile | `MT16x16x512` | `MT16x16x512` |
| M=1 | 2446.7 GB/s | 3548.4 GB/s |

The M=1 tile is identical across parts; the 4096^3 tile widened from 224 to
256. So "read the macro-tile, it is transferable evidence about what the
problem wants" survives, with the caveat that it transfers as a *family*, not
as a value.

## 5. The finding that changed the most results: gfx950 needs interleaved timing

This is not a per-backend detail, it is a property of the part that invalidates
any A/B measured the obvious way.

Timing A to completion and then B gives, on gfx950, spreads of **20-67%** on
kernels whose spread is ~1% when the two are interleaved. The drift is not
random: it tracks position in the run, so it is charged to whichever callable
happened to execute while the part was fast. Measured on the same kernels, same
shapes, same iteration counts, only the ordering changed:

| measurement | back-to-back spread | interleaved spread |
| --- | --- | --- |
| Triton nonkdim variants, 4096^3 | 13.8-20.7% | 1.6-5.1% |
| cross-backend race, 4096^3 | 31-48% | 1.0-1.5% |
| `moe_aiter_fused` baseline, M=1 | 67.0% | 1.0% |

The consequences were larger than the numbers suggest:

* **The corpus was not under-reporting wins so much as reporting the wrong
  ones.** `common/bench.py` now has `time_pair`, and `sweep.py` re-times the
  baseline interleaved with every candidate instead of once at the top.
  Re-running the whole corpus on gfx950 that way gives **27 REAL wins over 56
  shapes**, against 21 back-to-back — but the headline understates it, because
  three back-to-back wins turn out not to exist and seven real ones had been
  hidden. §9 has the shape-by-shape comparison. `moe_aiter_fused` at M=1 went
  from *skipped as unmeasurable* (67% spread) to a 6.9% win at 1.0% noise, and
  at M=128 from "no candidate beat the noise floor (33.2%)" to a 20.7% win at
  0.9%.
* **hipBLASLt's race number is inflated.** The 4096^3 winner reports 1553
  TFLOPS during `--algo_method all` and 1241 when replayed alone by index --
  a **20.1% drop**. `tuning-hipblaslt` §3 records the same comparison on gfx942
  as 5.5% and dismisses it as run-to-run spread. At 20% it is not dismissible:
  racing 2085 solutions back-to-back holds the clocks up, and the replay is the
  honest figure. **Deploy on the replayed number, not the raced one.**
* On gfx942 none of this was visible, which is why the harness shipped this
  way.

The rotating-buffer effect moves the other direction. At 1024^3 with one
solution, `--rotating 0` vs `512` MB:

| | gfx942 (skill) | gfx950 (measured) |
| --- | --- | --- |
| inflation from cache reuse | 43% | 6.7% |

Still worth using; no longer the dominant error on this shape. On gfx950 the
dominant error is the clock, not the cache.

## 6. The MoE config drought, and a device-name split

The single largest deployable gap, and it is not a tuning problem -- it is that
nothing is tuned yet.

**vLLM.** `current_platform.get_device_name()` returns `AMD Instinct MI355 OAM`
on this box, so lookups go to `device_name=AMD_Instinct_MI355_OAM`. Of the 316
shipped fused-MoE configs, seven are MI35x-family and **two** are reachable:

| device_name in filename | files | reachable |
| --- | --- | --- |
| `AMD_Instinct_MI355_OAM` | **2** | yes |
| `AMD_Instinct_MI355X` | 1 | no |
| `AMD_Instinct_MI350_OAM` | 3 | no |
| `AMD_Instinct_MI350X` | 1 | no |

Anyone counting files with `MI35` in the name sees seven and concludes the part
is covered. Five of them are dead weight. One of the five —
`E=160,N=192,dtype=fp8_w8a8` for `MI350_OAM` — covers an (E, N, dtype) this
device has no config for at all, so it is worth renaming and benchmarking before
tuning that shape from scratch.

The load path itself was exercised rather than inferred. Writing a config into
`VLLM_TUNED_CONFIG_FOLDER` under the name `get_config_file_name()` produces:

```
INFO  [fused_moe.py:1060] Using configuration from /tmp/…/E=128,N=1024,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json for MoE layer.
```

and writing it under the name built from `torch.cuda.get_device_name()` — which
is `''` in this image — produces the miss, naming both paths it tried:

```
WARNING [fused_moe.py:1073] Using default MoE config. Performance might be sub-optimal!
Config file not found at /tmp/…/E=128,…,device_name=AMD_Instinct_MI355_OAM,….json,
/usr/local/…/fused_moe/configs/E=128,…,device_name=AMD_Instinct_MI355_OAM,….json
```

Both locations are searched, so the env var *adds* a directory. Note for §6's
punchline that sglang's equivalent does the opposite.

**SGLang.** `sglang.srt.utils.get_device_name()` returns `AMD Instinct MI355X`
-- a *different string for the same silicon*, because torch reports
`AMD Instinct MI355X` in this image and the empty string in the vllm one. And
across all seven Triton-version directories:

| dir | files | MI355 | MI300X |
| --- | --- | --- | --- |
| `triton_3_1_0` | 127 | 0 | 7 |
| `triton_3_2_0` | 35 | 0 | 0 |
| `triton_3_3_0` | 1 | 0 | 0 |
| `triton_3_3_1` | 21 | 0 | 0 |
| `triton_3_4_0` | 33 | 0 | 0 |
| `triton_3_5_1` | 103 | 0 | 0 |
| `triton_3_6_0` (installed) | 8 | 0 | 0 |

**Zero** MI355 configs anywhere, in any version directory — the only AMD names
present are `AMD_Instinct_MI325X` (15) and `AMD_Instinct_MI300X` (7). Every
fused-MoE dispatch on MI355 under sglang runs an untuned default, confirmed from
the log rather than inferred:

```
Using default MoE kernel config. Performance might be sub-optimal! Config file not found at
.../configs/triton_3_6_0/E=256,N=256,device_name=AMD_Instinct_MI355X,dtype=fp8_w8a8,block_shape=[128, 128].json
```

All three documented outcomes were reproduced by planting files in a scratch
directory: exact hit, version fallback to `triton_3_1_0`, and default. Three
further behaviours are not in the skill and were found by testing:

- **Pointing `SGLANG_MOE_CONFIG_DIR` at the version directory raises.** The
  fallback scan calls `os.listdir(<dir>/configs)` unguarded, so the server dies
  with `FileNotFoundError: … '/work/tuned/configs'` at the first MoE layer rather
  than quietly missing.
- **The override replaces the package tree, it does not extend it.** `config_dir`
  is the root for both the exact lookup and the fallback scan, so with the
  variable set, none of the 328 shipped files are reachable. Costless on MI355,
  where none of them match anyway; on MI300X a sparse override directory turns
  the seven shipped MI300X configs into misses. vLLM's variable behaves the
  opposite way.
- **`enable_deterministic_inference` discards all of it.** `get_moe_configs`
  returns `None` before touching the filesystem. Verified with a valid exact-hit
  config in place: nothing loads, and the only log line is
  `Deterministic inference is enabled, using default MoE kernel config.`

Also worth knowing when scripting: `get_moe_configs` reads
`get_global_server_args()`, which only exists inside a scheduler process, so
checking your filename from a plain script raises
`ValueError: Global server args is not set yet!`. And the generated filename
contains a literal space in `block_shape=[128, 128]` — only `device_name` has its
spaces stripped.

The two frameworks disagreeing about the device name has a direct consequence:
a config tuned through vLLM's `benchmark_moe.py` lands under a filename sglang
will never look up, and vice versa. Same GPU, same `gfx950:sramecc+:xnack-`,
three name strings in play (`''`, `AMD Instinct MI355 OAM`,
`AMD Instinct MI355X`). Read the name from the framework you are deploying
into, never from torch, and never from the other framework.

The asymmetry in `torch.cuda.get_device_name()` is what makes this survive
review: it returns `''` in the vllm image and the correct `AMD Instinct MI355X`
in the sglang one. A helper that builds filenames from torch is right in one
container and silently wrong in the other.

## 7. Per-skill re-measurement: done and outstanding

**Re-measured on gfx950 in this pass:**

| skill | claim | gfx942 | gfx950 |
| --- | --- | --- | --- |
| tuning-ck | best instance, 4096^3 | 546.3 TFLOPS | 1245.5 |
| tuning-ck | worst instance | 139.7 | 186.9 |
| tuning-ck | best/worst spread | 7x | 6.4x |
| tuning-ck | instances timed | -- | 126 |
| tuning-ck | winning instance | `CShuffleV2<Default,256,256,256,...>` v1 | **identical** |
| tuning-hipblaslt | solutions raced | 1231 | 2085 |
| tuning-hipblaslt | winner index | 198969 / `ISA942` | 441281 / `ISA950` |
| tuning-hipblaslt | winner macro-tile | `MT256x224x64` | `MT256x256x64` |
| tuning-hipblaslt | race | 633.5 TFLOPS | 1553.0 |
| tuning-hipblaslt | replay by index | 598.8 (-5.5%) | 1241.3 (**-20.1%**) |
| tuning-hipblaslt | bracket != solution index | `[99]`/`[66]` drift | `[1910]` vs index 441281 |
| tuning-hipblaslt | rotating inflation, 1024^3 | 43% | 6.7% |
| tuning-triton | baseline, 4096^3 | 356.8 TFLOPS / 5.4% | 760.5 / 5.1% |
| tuning-triton | `nonkdim=16` | +7.7%, spread 0.8% -> REAL | +4.0%, 1.6% -> REAL |
| tuning-triton | `nonkdim=32` | +4.4%, spread 7.7% -> noise | -0.3%, 2.8% -> noise |
| tuning-triton | `key=` penalty at M=1 | 3.75x | 2.45x |
| tuning-triton | regime split gain | 34% | 44% |
| tuning-hip | `--offload-arch` compiles | gfx942 | gfx950 |
| tuning-hip | resource-usage remarks | present | present (VGPR 34, LDS 2048, occ 8) |
| tuning-hip | device constants via HIP API | 304 CU / 64 KB | 256 CU / 163840 B / warp 64 |
| tuning-hip | async-timing inflation | 4363 TFLOPS vs ~1300 peak | 2.4x, **short shapes only** |
| tuning-flydsl | 4096^3 | -- (none) | 880 TFLOPS, err 0 |
| tuning-aiter | tuner wall time, 2 shapes | 214 s | 120.2 s |

Note the tuning-hip refinement: on gfx950 the unsynchronised loop inflated the
**256^3** measurement 2.4x (0.008 vs 0.019 ms) and the **4096^3** one not at
all (7.378 vs 7.396 ms). The queue saturates once the kernel is long enough to
outlast the enqueue, so the trap bites exactly where kernels are short -- which
is decode. The gfx942 write-up implies it always inflates.

The remaining rows, which do not belong in the throughput table above:

| skill | claim | gfx942 | gfx950 |
| --- | --- | --- | --- |
| tuning-ck | CK offline tuners run | 0 of 7 | **7 of 7** |
| tuning-hipblaslt | raced winner vs library default | not measured | **−2.3% … +13.7%** over 4 shapes (§10) |
| tuning-hipblaslt | logic-tree strategy | CU-count split, no Origami | **83% Origami** (analytical) (§10) |
| tuning-in-vllm | MoE config load path | log-inspected | hit + miss both reproduced |
| tuning-in-sglang | MoE config load path | log-inspected | hit + fallback + default reproduced |
| corpus | full sweep, interleaved | n/a | **22 cases / 56 shapes / 27 REAL** (§9) |

**Not yet re-measured on gfx950:**

* Deploying and confirming engagement for the two real tuner wins found in §3
  (batched a8w8 6.78%, fp8 MoE 1.73x). They were measured with `--compare` but
  not written into the runtime tables and re-checked end to end.
* `tuning-in-vllm` / `tuning-in-sglang` end to end **under load**: shapes
  captured from a live server, tuned, deployed, A/B'd on serving metrics. The
  config load path, device naming, env-var semantics and failure modes are now
  verified; what is missing is a real model serving real traffic.
  `validate/claims_live.py` exists for this and needs a server run.
* 56 of 74 aiter tunable ops (§3, classified by blocker in §13), the FP4 fusions and the attention family
  above all.
* `tuning-ck` deploy path: ckProfiler's winner carried into aiter's CK tuner
  and confirmed engaged.
* The three families with zero shipped `cu_num=256` rows (§3): dense bf16,
  batched GEMM, and generic fused-MoE in the vllm image. The tuners exist and
  work; the tables are simply empty.

## 8. Cases added in this pass

| case | why it was missing | gfx950 result |
| --- | --- | --- |
| `moe_aiter_fused` | aiter's MoE family was 0-covered; the two existing MoE cases use vLLM's kernel | 6.9% REAL at M=1, 20.7% at M=128, null at M=2048 |
| `gemm_a16wfp4` | gfx950-only; corpus had only symmetric FP4 | **56.3% REAL at M=512, 70.7% at M=2048** |
| `attn_extend_prefill` | attention was the largest 0-covered family; both existing attention cases are paged-decode | **16.8% REAL at prefix=2048/extend=256**, null on the two large-chunk shapes |

The attention result is the regime lesson again, in a family where the corpus had
no coverage to see it. The shipped `gfx950-EXTEND_ATTENTION.json` default
(`BLOCK_M=128, BLOCK_N=64, num_warps=4, waves_per_eu=2`) is a good config for
large chunks — on `p1024e512` and `p0e2048` nothing in a 40-candidate sample beat
it, and the best sample was 17.7% and 5.0% *worse*. On the prefix-heavy shape,
which is the steady state of multi-turn serving, a much smaller tile
(`BLOCK_M=32, BLOCK_N=32, num_warps=2`) took 16.8% off. One table, one arch, two
regimes, and the shipped entry can only serve one of them.

The FP4 case is the largest uplift in the corpus, and unsurprisingly:
`_get_config` reports `is_tuned=False` for every shape tried, so there is no
shipped table for it on this part at all.

It also carries an interface trap worth repeating here. Despite the name,
`gemm_a16wfp4` does **not** compute in bf16: the wrapper takes
`prequant: Optional[bool] = True` and the body asserts

```python
assert prequant, "prequant == False is not supported yet"
```

so the activations are converted to MXFP4 before the MFMA and the bf16 path the
name implies is unimplemented. A reference built on the literal reading of the
name disagrees with the op by `err_ratio` 1.1e-01 -- large enough to read as a
broken kernel rather than a misread interface. Dequantizing both operands gives
bit-exact agreement. The op is `a16` at its interface and FP4 in its
arithmetic.

## 9. The corpus re-swept on gfx950, interleaved

Everything above was measured after §5's clock-drift finding. The corpus itself
had not been, so it was re-run end to end: 22 cases, 56 shapes, identical
kernels, identical candidate ordering, identical 40-candidate budget. The only
change is that `sweep.py` now re-times the baseline alongside each candidate
through `bench.time_pair` instead of once at the top of the run.

| | back-to-back | interleaved |
| --- | --- | --- |
| cases swept | 19 | **22** |
| shapes | 47 | **56** |
| REAL wins | 21 | **27** |
| cases with at least one win | 12 | **14** |
| median win | 14.9% | 11.0% |
| largest win | 51.7% | **71.4%** (`gemm_a16wfp4` M=2048) |
| wins ≥ 20% | 8 | 6 |

The totals are the least interesting part, because the two runs disagree in both
directions and the disagreements partly cancel. Restricting to shapes present in
both runs:

**Three back-to-back "wins" do not survive.**

| case | shape | back-to-back | interleaved |
| --- | --- | --- | --- |
| `gemm_batched_bf16` | `B4xM1024xN4096xK4096` | 32.5% REAL | best candidate **+0.1%, inside noise** |
| `gemm_mxfp4_triton` | `M512xN4096xK4096` | 13.1% REAL | null |
| `gemm_fused_mul_add` | `M2048xN2624xK6144` | 8.6% REAL | null |

**Seven wins were hidden by it**, the largest being `gemm_mxfp4_gluon`
`M8192xN5120xK2880` at 37.8% and `gemm_mxfp4_gluon` `M512xN4096xK4096` at 13.3%.

**Of the 15 shapes that won under both methods, 12 shrank and 3 grew**, several
by most of the result:

| case | shape | back-to-back | interleaved |
| --- | --- | --- | --- |
| `gemm_a8w8_blockscale` | `M2048xN2624xK6144/g128` | 14.9% | **1.4%** |
| `ff_a16w16_gated` | `M512xN5632xK2048` | 19.5% | **4.4%** |
| `gemm_batched_bf16` | `B32xM128xN2048xK2048` | 26.1% | **5.5%** |
| `ff_a16w16_gated` | `M32xN11008xK4096` | 20.2% | 12.2% |
| `gemm_a16w8_blockscale` | `M512xN4096xK4096` | 27.6% | 20.5% |
| `gemm_mxfp4_gluon` | `M16xN5120xK2880` | 11.2% | **31.8%** |
| `gemm_a16w8_blockscale` | `M16xN5120xK2880` | 10.7% | 17.6% |

Two shapes in the earlier run are not comparable because their case shape lists
changed between runs (`gemm_skinny M1xN64xK7168`, `gemm_a8w8_blockscale
M32768xN2624xK6144/g128`); they are excluded above rather than counted as
regressions.

The conclusion is stronger than "back-to-back is noisy". A noisy measurement
would scatter symmetrically around the true value and the ranking would mostly
survive. This does not: it deletes real 37.8% wins, invents 32.5% ones, and
compresses or inflates the rest by factors of three or four. On this part, a
back-to-back A/B is not a worse estimate of the right quantity — it is an
estimate of a different quantity. Any gfx950 tuning result produced by a harness
that has not been checked for this should be treated as unmeasured.

Eight cases returned no win anywhere in their 40-candidate sample:
`attn_paged_decode`, `gemm_a16w16`, `gemm_a8w8_ck`, `gemm_a8w8_int8`,
`gemm_skinny`, `moe_fp8_blockscale`, `moe_int4_w4a16`,
`quant_per_token_group_fp8`. Those are honest nulls over a small sample — the
budget covers 0.2% of `gemm_skinny`'s space and 0.0% of
`gemm_batched_bf16`'s — and every one of them is recorded as an empty result
rather than left with a stale win standing.

## 10. hipBLASLt on MI355 selects analytically, and it shows

§2's race/replay gap answered "is the race number honest" (no, by 20%). It did
not answer the question that decides whether to tune at all: **how much does the
raced winner beat the solution hipBLASLt would have picked by itself?** That
needs the default (`--algo_method heuristic`) and the winner replayed by index,
interleaved against each other. Four bf16 shapes, `-i 50 -j 10`, five alternating
rounds, medians in TFLOPS:

| shape (m·n·k) | default | raced winner | the race's implied uplift | honest gain |
| --- | --- | --- | --- | --- |
| 4096·4096·4096 | 1174 | 1238 | +33% | **+5.4%** |
| 8192·1024·8192 | 956 | 1057 | +40% | **+10.6%** |
| 2048·8192·2048 | 1212 | 1185 | +19% | **−2.3%** |
| 1024·4096·4096 | 746 | 848 | +35% | **+13.7%** |

On one shape in four the raced winner is **not better than the shipped default**.
A race always produces a winner; that is not evidence the winner is worth
deploying.

The structural reason is visible in `rocm-libraries` and is a genuine
architecture difference, not a version difference. hipBLASLt's solution choice is
driven by logic YAML under
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`:

| | gfx942 (`aquavanjaram/`) | gfx950 (`gfx950/`) |
| --- | --- | --- |
| logic files | 816 | 566 |
| top-level split | **CU count** — `gfx942`, `_20cu`, `_38cu`, `_64cu`, `_80cu`, `_152cu`, `_228cu` | **selection strategy** — `Equality`, `GridBased`, `Origami`, `Range` |
| strategies | Equality, Experimental, FreeSize, GridBased, StreamK | Equality (25), GridBased (67), **Origami (471)**, Range (3) |

`Origami` is 83% of the gfx950 logic tree and **has no counterpart in the gfx942
tree at all**. It is a separate project in the same monorepo,
`shared/origami/`, whose README describes it as "Analytical GEMM Solution
Selection": it models compute and memory latency over candidate tile sizes,
estimates occupancy and L2 hit rate, and picks a tile from that model rather than
from an exhaustively benchmarked table.

That reframes what tuning hipBLASLt means on this part:

- The MI355 default is a **prediction**, so it degrades gracefully on shapes
  nobody benchmarked — which is why it stays within ~14% of the best measured
  solution across all four shapes instead of falling off between table entries.
- The gfx942 heuristic "the shipped table is sparse, so racing usually wins big"
  does not transfer. Budget MI355 tuning time against −2%…+14%.
- There is no `gfx950_256cu` directory to look for. The per-CU-count split is
  gone because CU count is an input to the model instead of a directory key.

This section's table is measured; the explanation from the logic tree is a read
of the source and is offered as such.

## 11. The `rocm-libraries` scan: what else is tunable

§3 counted aiter's op surface. aiter is not the whole platform, so the
`rocm-libraries` monorepo was scanned too — the same checkout §10 reads. The
question is narrow: **what is tunable in there that the skillset does not
address, and does it matter for LLM serving?**

### Composable Kernel

| surface | size | skillset coverage |
| --- | --- | --- |
| `ckProfiler` ops | **71** (39 GEMM-family, 11 conv, 21 norm/pool/reduce/softmax) | method covered; **1 op measured** (`gemm`) |
| instance directories | 98 | racing method applies to all; §2b gate documented |
| `tile_engine` ops | 3 (`gemm`, `gemm_multi_d`, `gemm_preshuffle`) | **not covered at all** |
| aiter's per-op CK tuners | 7 | **7 of 7 run on gfx950** (§3) |

Two honest gaps here, of very different sizes.

The small one is op breadth: `tuning-ck` demonstrates the racing method on one
GEMM and the method transfers to the other 38 GEMM-family ops unchanged. The
conv, norm, pool and softmax ops are real tunable surface but not on the serving
path this skillset targets, and are deliberately out of scope.

The large one is `tile_engine`, and it is a gap in *kind*, not in count. Every
statement in `tuning-ck` frames CK tuning as **selecting among pre-compiled
instances**. `tile_engine` is the other thing: a JSON-declared config space —
`tile_m/n/k` as `{min, max, step}`, `warp_m/n/k` and `warp_tile_*` as value
lists — that generates and benchmarks CK instances that did not previously
exist:

```
tile_engine/ops/gemm/configs/default_config.json
tile_engine/ops/gemm/gemm_instance_builder.py
tile_engine/ops/gemm/gemm_benchmark.py
```

That is structurally the *Triton* model — author a space, generate, race — done
in CK C++. So the skill map's one-line characterisation of CK ("select a
pre-compiled instance") is accurate for `ckProfiler` and wrong for
`tile_engine`, and a shape whose best instance is not in the shipped library is
reachable only through the surface the skillset does not describe. This is the
single largest *structural* gap remaining in the skillset, and it is not
measured here.

### hipBLASLt / Tensile

| surface | size | skillset coverage |
| --- | --- | --- |
| gfx950 solution logic | 566 YAML, 83% Origami | selection covered (§10); **generation not** |
| gfx942 solution logic | 816 YAML across 7 CU-count dirs | same |
| `tensilelite/` kernel generation | full Tensile assembler + generator | **not covered** |

Same shape of gap as `tile_engine`: `tuning-hipblaslt` covers picking among
compiled solutions, and TensileLite is how new solutions get compiled in the
first place. For a serving workload this is the right priority order — solution
selection is a day's work with immediate payoff, adding a Tensile kernel is a
different kind of project — but the skillset should not imply the deeper surface
does not exist.

### Everything else in the monorepo

`rocblas`, `hipsparselt`, `miopen`, `rocfft`, `rocsolver`, `rocprim` and the rest
are out of scope on purpose: none is on the dense-GEMM/attention/MoE path that
vLLM and SGLang dispatch through on these images. `hipsparselt` (structured
sparsity) is the one plausible future entry.

## 12. The microscaled formats: where MXFP4 and MXFP8 actually are

MXFP4 and MXFP8 are the reason MI355 is a different tuning target rather than a
bigger MI300X — they have no gfx942 counterpart at all. This section is the
result of going after them specifically.

### The headline: MXFP8 is not an aiter op on these images

The obvious plan was to mirror the MXFP4 cases with MXFP8 ones. That plan does
not survive contact with the installed software. Probing both images directly:

| symbol | vLLM image | SGLang image |
| --- | --- | --- |
| `gemm_afp4wfp4`, `gemm_a16wfp4`, `gemm_a8wfp4` | present | present |
| `batched_gemm_afp4wfp4`, `fused_moe_mxfp4` | present | present |
| `dynamic_mxfp4_quant` | present | present |
| **`gemm_afp8wfp8`** (MXFP8 dense) | **absent** | **absent** |
| **`dynamic_mxfp8_quant`** | **absent** | **absent** |
| **`fp8_legacy_to_mxfp8`** | **absent** | **absent** |
| **`gfx950-A8W4.json`** (MXFP8×MXFP4 MoE table) | **absent** | **absent** |

Both images ship the same MX surface, and it is **FP4-centric**. A source scan
of `libs/aiter` finds a large MXFP8 surface — `gemm_a8w8_mxfp8`,
`fmha_fwd_mxfp8_asm`, `fused_rms_mxfp8_quant` and about a dozen more — and that
scan is misleading for two independent reasons, both of which matter when
planning work:

1. **It is a newer tree than either installed build.** The checkout is not what
   the frameworks import.
2. **Most of those MXFP8 paths target gfx1250, not gfx950.** `gemm_a8w8_mxfp8`
   and `fmha_fwd_mxfp8_asm` load HSA code objects from `hsa/gfx1250/`. They
   would not run on MI355 even if installed.

So "add MXFP8 aiter cases" is not a task that can be completed against these
images, and a corpus case claiming to cover it would be measuring nothing.
Where FP8 does appear on the gfx950 MX path it is as the *activation* side
against MXFP4 weights (`gemm_a8wfp4`), which is a real op and is now covered.

### MXFP8 does exist on MI355 — in CK and hipBLASLt

Both libraries reach the same hardware the aiter wheels do not:

| route | MXFP4 | MXFP8 | tunable? |
| --- | --- | --- | --- |
| `ckProfiler gemm_mx` | 11 instances, 3844 TFLOPS best | 5 instances, 1765 best | **yes**, 2.3–2.8× spread |
| `hipblaslt-bench --scaleA 3` | 27 solutions, 2567 best | 12 solutions, 1305 best | **no** — see below |
| aiter Triton | 6 ops | none | yes, 4 now covered |

The CK numbers are the useful discovery: `gemm_mx` is gated to `gfx95` at build
time, is present in the shipped ROCm, and is the only route to an MXFP8 GEMM on
this hardware. Instances print `ck::f8_ocp_t` and `ScaleBlockSize: 32`,
confirming the OCP dialect and the 32-element block. Details and command lines
are in `../tuning-ck/SKILL.md` §2c.

CK is also substantially faster than hipBLASLt on both formats here — 3844 vs
2567 TFLOPS on MXFP4, 1765 vs 1305 on MXFP8. Both are race/heuristic numbers
measured back-to-back and so are inflated under Rule 6b, but the gap is far
larger than that error, and it points the same way for both formats.

### hipBLASLt MX tuning is a no-op, for three separate reasons

Documented fully in `../tuning-hipblaslt/SKILL.md` §6b. In short: `--algo_method
all` is refused outright for MX types; the substitute enumeration
(`--requested_solution -1`) works and shows 5.0× (MXFP8) to 10.6× (MXFP4)
between best and worst solution; but `--print_kernel_info` emits no solution
index for MX, so no winner can be addressed or replayed. The cause is visible
in the rejection messages — MX routes through **RocRoller** (`RR_GEMM_...`), not
Tensile, and RocRoller solutions are not in the Tensile index space.

The saving grace is that it does not matter: for both formats the heuristic's
own first pick was the **fastest of all** candidates offered. This is the same
conclusion §10 reached for bf16 from the Origami analysis, reached independently
for MX.

### Four new cases, and what they cost to write

| case | shapes | REAL wins | gains |
| --- | --- | --- | --- |
| `gemm_a8wfp4` — FP8 act × MXFP4 weight | 3 | **3** | 3.5%, 14.4%, **34.7%** |
| `gemm_mxfp4_batched` — batched MXFP4 | 3 | **3** | 11.3%, 2.4%, **72.8%** |
| `moe_mxfp4` — fused MoE, MXFP4 experts | 3 | **3** | 4.6%, 2.3%, 1.5% |

Nine for nine, at budget 40 with the interleaved harness. This takes the corpus
from 22 cases / 27 REAL wins to **25 cases / 36 REAL wins**, and
`gemm_mxfp4_batched` at 72.8% is now the largest single uplift in it.

The MoE result is the interesting one, because it is the opposite of what the
689-byte one-entry config table predicted. A single config covering every shape
should be leaving a great deal on the table, and instead the sweep finds 1.5–4.6%
— real, but small. The honest reading is that this op is dominated by something
other than the GEMM tiling: at these shapes the token alignment and the
scatter/reduce around the expert GEMMs are a large share of the time, and
`BLOCK_SIZE_*` cannot reach them. It is covered now, and the answer is that
there is less here than the table's thinness suggests.

Three interface faults were found in the process, all of which fail in ways that
do not name the problem, and all of which are documented in the case docstrings:

- **`_get_config` takes packed K in the batched op, logical K in the unbatched
  one.** Same name, same library, same format, different units. The knob that
  overruns is `SPLITK_BLOCK_SIZE` (`2*K`, strided by half over a `K/2` buffer),
  not `BLOCK_SIZE_K`, which agrees on most shapes. Being an out-of-bounds read
  it is undefined behaviour and the symptom is not stable: the same mistake
  produced a `Memory access fault by GPU node-5` that killed the sweep process
  at K=4096, a silent `nan` at K=256, and a plausible in-tolerance answer on a
  rerun of that call. The validator therefore checks the config arithmetic
  rather than the observed error, and finds the overrun on 4 of 4 shapes in
  both images.
- **Ops that take an output buffer return `None`,** contradicting docstrings
  that promise the tensor. True of `gemm_a8wfp4`, `batched_gemm_afp4wfp4` and
  `fused_moe_mxfp4` — assume it for any aiter op with an out parameter.
- **`batched_gemm_afp4wfp4`'s `y` is typed `Optional[...] = None`** but the body
  does `By, _, _ = y.shape`. The documented default raises.

For the MoE case there is a fourth, which is a tuning-methodology point rather
than a bug: **`BLOCK_SIZE_M` changes the input data, not just the tiling.** The
token-to-block alignment is built by `moe_align_block_size_triton` from
`BLOCK_SIZE_M`, so a sweep that computes it once and then varies that knob feeds
every candidate a token map built for a different block size. The case rebuilds
the alignment per candidate and times it.

### What is still uncovered in MX after this pass

- **MXFP6 / BF6.** CK ships `f6` and `bf6` instances; nothing else exposes them
  and no framework path reaches them.
- **CK MX from a framework.** `gemm_mx` is raceable via ckProfiler but no aiter
  or vLLM/SGLang path dispatches into those instances, so a win there is not
  currently deployable — the same gap §3 describes for CK generally.
- **MX attention.** `fav3_sage_mxfp4` exists in aiter with a hardcoded tile
  config. Not measured.
- **Swizzled MX scales.** Every case here passes `swizzle_mx_a/b = False`. The
  swizzled layout is a separate kernel path.

## 13. "Uncovered" vs "cannot be covered": the 56 remaining aiter ops, classified

A gap of 56 ops is only actionable if you know which ones are *work* and which
are *blocked*. Enumerated with `tools/_uncov.py` against the installed vllm
image and classified by what actually stops each one:

| blocker | ops | verdict |
| --- | --- | --- |
| nothing — plain work | ~40 | **untouched** |
| needs a weight-preshuffle step | ~12 | **untouched**, one shared helper unlocks all |
| harness has no backward pass | 3 | blocked by *scope*, not access |
| needs multiple ranks | 1 | blocked by *harness*, not access |

**So: untouched, not can't-touch.** Only 4 of 56 are blocked by anything other
than effort, and neither of those blockers is the hardware or the software — both
are the corpus harness.

The details, because the summary rounds off some real distinctions:

**~12 ops are one helper away.** `gemm_a16w8_blockscale_preshuffle`,
`gemm_a16wfp4_preshuffle`, `gemm_a8w8_blockscale_preshuffle`,
`gemm_afp4wfp4_preshuffle`, `gemm_afp4wfp4_preshuffled_scales`,
`gemm_afp4wfp4_preshuffled_weight_scales`, `gemm_a8w8_preshuffle`,
`gemm_a8w8_bpreshuffle_flydsl`, `fused_gemm_afp4wfp4_preshuffle_add_mul`,
`fused_gemm_afp4wfp4_preshuffle_split_cat`,
`fused_gemm_a8w8_blockscale_preshuffle_split_cat`, plus the `_pre_quant`
variants. Every one needs its weights laid out for the MFMA before the call, and
`aiter.ops.shuffle` already ships `shuffle_weight`, `shuffle_weight_NK`,
`shuffle_weight_a16w4` and `shuffle_scale_a16w4` to do it. This is the
highest-leverage remaining work in the whole gap — roughly a fifth of it behind
one piece of shared case infrastructure. It is also the family most likely to
matter, since preshuffling is what production serving actually does.

**3 are backward passes** — `flash_attn_fused_backward`,
`flash_attn_onekernel_backward`, `triton_hstu_attention_bwd`. Nothing about the
hardware blocks them. The harness is inference-shaped: it judges TFLOPS in
decode/prefill regimes and builds references by forward recomputation. Covering
these means teaching the harness a backward reference, which is a real change and
arguably a different corpus.

**1 is a collective.** `fused_pipeline_kernel` is
`reduce_scatter_rmsnorm_quant_all_gather` — it needs multiple ranks to do
anything. The box has 8 GPUs so it is physically possible; the harness is
single-device, so this needs multi-rank orchestration that does not exist.

**The rest are ordinary.** Activation variants of ops already covered
(`fused_moe_silu`, `fused_moe_gelu`, `fused_moe_mxfp4_silu`), the three
`ff_a16w16_*` shapes, grouped GEMM (`ptgmm`, `nptgmm`), the attention family
(`flash_attn_func`, `flash_attn_varlen_func`, `pod_attention`,
`persistent_lean_attention`), the Gluon paged-decode kernels, and the two
rotation quant ops. The MLA fusions
(`fused_{fp4,fp8}_bmm_rope_cat_and_cache_mla`) are the most expensive of these
because they need KV-cache plumbing rather than synthetic tensors.

### The two generation surfaces are also untouched, not unreachable

Both were listed in §11 as having no skill. Checking whether that is a wall or a
backlog item:

**CK `tile_engine`** is a codegen-plus-build system, present in the checkout, and
its `configs/default_config.json` is *already a tuning search space*: block tile
m/n/k, warp counts, `warp_tile_m/n/k`, and a `trait_config` selecting pipeline
(`compv3`, `compv4`, `mem`) and scheduler. That makes it the only surface in this
whole document where you **define the candidate set** rather than pick from a
fixed pool — everything else in `tuning-ck` and `tuning-hipblaslt` is selection
among pre-built instances. The blocker is compile time: the current design
compiles one executable per kernel configuration, so exploring a tile space is a
build, not a race. Reachable, expensive, and the most conceptually interesting
gap left.

**hipBLASLt / TensileLite** ships its generators (`TensileCreateLibrary`,
`TensileLogic`, `TensileRetuneLibrary`) in the checkout. Same shape of blocker,
larger: regenerating a Tensile library is hours of compilation. Note that
`TensileRetuneLibrary` is the interesting entry point, since retuning an existing
logic file is much cheaper than generating one — and §10 found that gfx950's
tree is 83% Origami, i.e. analytical rather than tabular, which may mean there is
less for a retune to change than on gfx942.

Neither is blocked by access, licensing, hardware or missing source. Both are
blocked by build time, which is a scheduling problem.

## 14. Coverage by *aspect*, not by op

Everything above counts ops, backends and surfaces. That is the wrong axis for
the question "what kind of tuning can this skillset not help with", because a
skillset can cover every GEMM op and still be silent about an entire dimension of
performance work. Audited by grepping the skills for each aspect:

| aspect | status |
| --- | --- |
| kernel config search (Triton/CK/hipBLASLt/aiter/FlyDSL/HIP) | covered, 6 skills |
| measurement methodology | covered, and the largest single finding |
| correctness gating | covered (`correctness_gates.md`) |
| deployment / engagement proof | covered (`engagement_verification.md`) |
| arch migration | covered (`arch_migration.md`) |
| occupancy, LDS, register/scratch pressure | covered across `tuning-core`, `tuning-hip`, `tuning-triton` |
| **clocks, power, perf determinism** | **now covered** (`clocks_and_power.md`) |
| **HIP graph capture / launch overhead** | **not covered** |
| **allocator tuning** (`PYTORCH_HIP_ALLOC_CONF`) | **not covered** |
| **runtime env knobs** (`GPU_MAX_HW_QUEUES`, NUMA) | **not covered** |
| **framework scheduler knobs** (chunked prefill, `max_num_batched_tokens`, graph batch sizes) | **not covered** |
| **collective / multi-GPU tuning** (RCCL) | **not covered** |
| **`torch.compile` / Inductor** | **not covered** |
| **bottleneck attribution** (omniperf, roofline) | **barely** — `rocprof` appears in 4 skills, `omniperf` in none |

### The one that was closed, and why it mattered

Clock and power tuning was the glaring omission, because Rule 6b — the most
consequential finding in this whole document — is a *software workaround for a
hardware problem we had never tried to fix*. The investigation is in
`../tuning-core/clocks_and_power.md`. Three results:

**The drift reproduces in isolation.** One bf16 GEMM at 4096³, 50-iteration
warmup, then twelve identical timed rounds: 1239 → 1399 TFLOPS, a 12.9%
**monotonic ramp** that is still climbing seven rounds and ~2000 GEMMs after
warmup. No A/B, no two kernels — just the part getting faster as it runs.

**The fix exists and cannot be used here.** `rocm-smi --setperfdeterminism`
is the right tool. In the container it writes through a read-only sysfs mount.

**It fails silently, which is the actual finding.** As UID 0, the call exits
**0**, prints **no** error/failure/permission text, and leaves the performance
level at `auto`. Re-measuring afterwards gives the same ramp (1194 → 1401,
17.3%). Someone who pins clocks, sees success and therefore stops interleaving
has adopted the one belief that would justify abandoning Rule 6b, while the
problem it guards against is fully intact. The only trustworthy check is
`--showperflevel`, which must read something other than `auto`; a validator claim
now enforces this.

### The ones still open, ranked by what I would do next

1. **Bottleneck attribution.** The sharpest gap conceptually: every skill answers
   "how do I tune X", none answers "is X worth tuning". `omniperf` is absent
   entirely. A roofline pass would tell you which corpus cases are
   memory-bound and therefore immune to the tile tuning we spend the budget on —
   the MoE MXFP4 result in §12, where a one-entry config table nevertheless
   yielded only 1.5-4.6%, is exactly the case that wants this explanation.
2. **Framework scheduler knobs.** Almost certainly the highest *practical* ROI on
   real serving, and entirely absent. Chunked prefill size and
   `max_num_batched_tokens` move end-to-end throughput more than most kernel
   wins, and they are free to change.
3. **Graph capture and launch overhead.** The corpus already carries an
   arch-specific launch floor (17 µs on gfx950 vs 42 on gfx942), so the machinery
   to reason about it exists; nothing uses it to evaluate graph capture, which is
   the standard remedy at decode shapes.
4. **Allocator and runtime env knobs.** Cheap to test, plausibly worth a few
   percent, currently guesswork.
5. **Collectives.** Needs the multi-rank harness that §13 already identifies as
   the blocker for `fused_pipeline_kernel`. One piece of infrastructure unlocks
   both.

### So what is the honest coverage number?

There isn't one number, because the surfaces are not commensurable. What can be
said precisely:

- **aiter ops:** 18 of 74 have a corpus case (24%). Of the 56 uncovered, 52 are
  blocked only by effort; 3 need a backward-capable harness and 1 needs multiple
  ranks (§13).
- **aiter MX (gfx950-only) ops:** 4 of 6 shipped MXFP4 ops covered; MXFP8 has
  no aiter operator to cover (§12).
- **aiter CK tuners:** 7 of 7 run on gfx950.
- **CK racing:** 1 of 39 GEMM-family ops measured; method covers the rest.
- **CK codegen (`tile_engine`):** 0 of 3 — no skill describes this surface. Not
  unreachable: the generator and its tile search space are in the checkout, and
  the blocker is per-kernel compile time (§13).
- **hipBLASLt selection:** covered and re-measured on gfx950.
- **hipBLASLt/Tensile generation:** 0 — no skill describes this surface.

"100% coverage" was never the achievable target; the achievable target is that
every surface is either covered or listed here as not covered, and that is now
true.
