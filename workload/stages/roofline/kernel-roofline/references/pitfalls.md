# Pitfalls index

Every trap this package knows about, one line each, anchored to where it is
explained. Scan this before publishing a roofline number.

## Cross-engine

| Trap | Why it bites | Where |
| --- | --- | --- |
| Comparing TL's % against rocprof's % | Different denominators: gfx950 TL/rocprof is HBM 0.856, bf16 0.573, fp8 0.628, fp4 0.502 | [combination.md §1](combination.md#1-what-this-mode-must-not-do) |
| Applying `target_eff=0.90` on top of TL's bandwidth | TL's figure is already derated; double derate inflates `attainable_speedup` ~17% | [combination.md §1](combination.md#1-what-this-mode-must-not-do) |
| Treating `x_TL%` as evidence | TL's roofs are bit-identical across legs, so its % *necessarily* tracks measured speedup. It is an identity, not a test | [combination.md §5](combination.md#5-x_tl-is-an-identity-not-a-test) |
| Reading a bound label as the measured bottleneck | It classifies the model's resource terms, nothing more | [SKILL.md](../SKILL.md), [corrections.md §4](corrections.md#4-compute-and-interpret-the-scorecard) |
| Clamping Eff > 1 to 100% | A clamped 100% reads as "saturated" — the opposite of what the row says | [rocprof.md §8](rocprof.md#8-eff--100-is-a-disproof-of-the-denominator) |

## Denominators and arch files

| Trap | Why it bites | Where |
| --- | --- | --- |
| Assuming the arch JSON is a constant | 4 MI355X variants on this cluster, 18.6% bf16 dispersion | [combination.md §7](combination.md#7-the-arch-json-is-not-a-constant) |
| Inferring provenance from the filename | The most-copied variant has no `source` field, equals the datasheet pin, and is handwritten | [`../assets/README.md`](../assets/README.md) |
| Scoring a Triton/FlyDSL fp4 kernel on `matrix_fp4` | The key holds only the max of CK (5146) and Triton (2722) — 1.89× underestimate | [combination.md §7](combination.md#7-the-arch-json-is-not-a-constant) |
| rocprof-compute < 3.6.0 on gfx95x | CDNA3 MFMA issued on a CDNA4 part; bf16/fp16/int8 peaks 2–4× low, silently | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| One-sided "empirical ≤ datasheet" sanity check | It *passes* the 614663-vs-2.5e15 bug. Need both below and within a sane fraction | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| Letting the microbench re-run per case | Drifts 17% (HBM 5326 → 6181 → 6246 on the same GPU, same day); invalidates the whole table | [rocprof.md §2](rocprof.md#2-freeze-the-denominator-across-every-comparison) |
| Treating the ≥3.6.0 gate as sufficient | It fixes bf16/int8, half-fixes fp16, and leaves **fp8 2.03× low** — which doubles every fp8 compute efficiency | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| Reading "didn't move between releases" as a control | It means the newer release did not fix that dtype, not that it was right. Check against native MFMA rates | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| `MFMAF8Flops == MFMABF16Flops` in a peak table | CDNA4 runs fp8 at 2× bf16; equality means the bench issued the bf16-K shape for both | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| Trusting the sibling guard as a cure | It is a backstop floor. Passing the gate and still reading >100% means the ceiling is still wrong | [rocprof.md §1](rocprof.md#1-version-gate-rocprof-compute--360-on-gfx95x) |
| Scoring an FP8 GEMM on the FP32 roof | Pick the peak by **MFMA dtype**, not tensor dtype. Reads ~3% and means nothing | [rocprof.md](rocprof.md#the-recipe) |
| Scoring a VALU kernel 0 on the datasheet leg | `DATASHEET_PEAKS["gfx950"]` has only `mfma_*`; drop it and report reduced ds-coverage | [rocprof.md §9](rocprof.md#9-valu-kernels-have-no-datasheet-peak) |
| Reading "< 3.6.0" as "unusable" | The counters are fine on 3.4.0; only its microbench is. Seed a `roofline.csv` of measured ceilings and the run is sound — check `MFMABF16Flops != MFMAF16Flops` and fp8 near datasheet, not half | [rocprof.md §2](rocprof.md#2-freeze-the-denominator-across-every-comparison) |
| Comparing AI against another campaign's table without checking its cache policy | AI's denominator is bytes that *reached HBM*. A `write-evict`/`read-evict` harness and a per-iteration flush give AIs 1.8–2.4x apart on the same kernel and shape, with neither being wrong | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Re-derating an already-empirical peak | That is the double-derate failure in another dress | [corrections.md §2](corrections.md#2-freeze-the-hardware-reference) |

## Counters and bytes

| Trap | Why it bites | Where |
| --- | --- | --- |
| gfx950 HBM read bytes 2× low | `FETCH_SIZE` used `TCC_BUBBLE`, which is identically 0 on this part | [rocprof.md §3](rocprof.md#3-hbm-read-bytes-read-2-low-on-gfx950) |
| Multiplying by a constant 2 to fix it | `k ∈ [1,2]`: exactly 2.000 streaming, 1.00–1.15 scattered. Overshoots on mixed loads | [rocprof.md §3](rocprof.md#3-hbm-read-bytes-read-2-low-on-gfx950) |
| Patching one copy of the bad formula | Three independent copies: analysis_configs, counter_defs.yaml, rocprofiler-sdk | [rocprof.md §3](rocprof.md#3-hbm-read-bytes-read-2-low-on-gfx950) |
| Needing a memcopy to detect it | SoL panel 2.1.24 vs L2 panel 17.1.3 differ 2× **in the same profile** | [rocprof.md §3](rocprof.md#3-hbm-read-bytes-read-2-low-on-gfx950) |
| Treating `rocprofv3` like `rocprof-compute` | `FETCH_SIZE`/`WRITE_SIZE` are in **KiB**; a container can carry one patched and one not | [rocprof.md §4](rocprof.md#4-rocprofv3-is-not-rocprof-compute) |
| Calling a constant ratio "cache residency" | A ratio constant across orders of magnitude is a **unit bug** | [rocprof.md §4](rocprof.md#4-rocprofv3-is-not-rocprof-compute) |
| A 5th TCC counter in one pass | `Request exceeds the capabilities of the hardware` | [rocprof.md §5](rocprof.md#5-at-most-4-tcc-counters-per-pass) |
| Treating a missing counter pass as zero | Unavailable ≠ 0; report coverage | [measurement.md](measurement.md), [folding.md](folding.md) |

## Running the profile

| Trap | Why it bites | Where |
| --- | --- | --- |
| Sweeping shapes in one profiled command | rocprof aggregates per kernel over the whole run — you get one blended block, no per-shape roofline | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| `while read ... done < cases.txt` around the profiler | The profiled child drains stdin, so the batch loop ends after **one** case and still prints its own "ALL DONE". Use `3< cases.txt` + `</dev/null`, and count `*_roofline.txt` against cases×arms | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Cache-warm loop vs TL's cold byte model | Fewer bytes reach HBM → higher AI. `flush_cache` each iteration to align | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Summing the flush kernel into the seam | It has its own block under its own name; read it separately | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Checking `__file__` for the arm guard | A `getattr`'d function has empty `__file__` — it passes everything. Use `__code__.co_filename`, assert the *selected* arm, and verify it *rejects* the other | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Blaming code for a moved `achieved` | Same case twice: achieved 2.17× apart, AI bit-identical. Counters stable, clocks not | [rocprof.md §7](rocprof.md#7-judge-contention-on-achieved-not-on-ai) |
| Deleting only `-b 4` to "get more" | `--roof-only` decides what is *collected*; `-b` only what is *printed*. The extra blocks come out empty | [rocprof.md §11](rocprof.md#11---roof-only-and--b-4-are-not-the-same-thing--do-not-delete-only-one) |
| Collecting `--mode full` up front | Order-of-magnitude more replays. Classify with roofline first, then open full on one kernel: 16/17 if memory-bound, 2/7 if compute-bound | [rocprof.md §11](rocprof.md#11---roof-only-and--b-4-are-not-the-same-thing--do-not-delete-only-one) |
| Re-profiling to apply a peak change | Mixes the denominator fix with a fresh noise draw; the two are inseparable in the diff. `_roofline_raw.txt` is kept for this | [rocprof.md §10](rocprof.md#10-changing-the-peak-policy-does-not-require-touching-the-gpu) |
| Re-rendering in place | Destroys the comparison the re-render exists to make; both scripts refuse | [rocprof.md §10](rocprof.md#10-changing-the-peak-policy-does-not-require-touching-the-gpu) |
| Changing two peak settings at once | The delta then means nothing. Move exactly one | [rocprof.md §10](rocprof.md#10-changing-the-peak-policy-does-not-require-touching-the-gpu) |

## Folding and reporting

| Trap | Why it bites | Where |
| --- | --- | --- |
| Averaging percentages across cases | Fold in the **time domain**: `roof_us = Σ t_k·Eff_k` | [folding.md](folding.md) |
| One denominator for a cross-axis seam | Apply each kernel's own `attainable` **before** the sum | [folding.md](folding.md) |
| Weighting by rocprof's own durations | They come from serialized dispatch, not the deployment distribution. rocprof gives the ratio, TL gives the time | [folding.md](folding.md), [combination.md §3](combination.md#3-folding-rules) |
| Averaging per-shape AIs | AI is a ratio: fold as `Σ(w·FLOPs)/Σ(w·bytes)` | [folding.md](folding.md) |
| A single bound label for a seam | Report the share of weighted device time on the compute axis | [folding.md](folding.md) |
| Blending decode and prefill weights | Normalize weights **within** a regime | [folding.md](folding.md) |
| Scoring an unscorable kernel as 0% instead of dropping it | It stays in the denominator, deflating the roof and inflating reported headroom by exactly `(1-coverage)`. Both legs default to `drop` and report the shortfall as coverage; `--missing-emp zero` only to reproduce tables published before 2026-10-12 | [folding.md](folding.md) |
| Folding two legs over different case sets | A tl-only leg covers every shape for free; a rocprof leg covers the dominant case per seam. Side by side they describe two different workloads. Re-fold the cheap leg over the covered subset — `join_legs.py` does, so a join table's `TL%` is NOT the standalone tl-only `TL%` | [folding.md](folding.md) |
| Exact kernel-name lookup in the join | rocprof truncates long mangled names — exactly the templated ATen/rocprim symbols that matter | [combination.md](combination.md#running-it) |
| Benchmark repetitions as deployment weights | They are not | [SKILL.md](../SKILL.md), [corrections.md §5](corrections.md#5-export-reviewable-results) |
| Whole-operator FLOPs ÷ one stage's time | Match modeled work to measured scope | [corrections.md §1](corrections.md#1-define-the-modeling-and-measurement-contract) |
| Quoting headroom when the byte roof is under the launch floor | A dispatched kernel costs ~2.3 us of device time before it moves a byte; a 5-kernel decode seam with a 0.13 us byte roof reads "68x available" and the real ceiling is 11.5 us. Report `max(roof, n x 2.3 us)` and say the lever is fusing nodes | [folding.md](folding.md) |
| "95% of roof" read as "nothing left" | High efficiency does not rule out doing twice the necessary work | [corrections.md §4](corrections.md#4-compute-and-interpret-the-scorecard) |
| Publishing a shape-only result as measured | `tl-only` never touches a GPU; say so | [SKILL.md](../SKILL.md) |
| A runner that can only reach `baseline_src/` | It cannot run the §6 experiment at all. `case_runner_template.py --arm candidate` | [rocprof.md §6](rocprof.md#6-one-shape-per-profiled-command) |
| Deciding which engine is wrong from one implementation | Only the two-implementation natural experiment adjudicates; measured speedup is the answer key | [combination.md §6](combination.md#6-the-only-experiment-that-adjudicates) |
| Saving `stdout or stderr` from a failed subprocess | rocprof-compute always writes hundreds of INFO lines to stdout, so the `or` **always** discards stderr — and the traceback only ever appears there. `run_roofline.py` now saves both | [rocprof.md §12](rocprof.md#12-a-finished-profile-can-still-exit-non-zero) |
| A profile that dies after the last counter pass | The PDF export needs Kaleido and takes the exit code down with it. Patch it non-fatal (`rocprof_hbm_patch.py`); do **not** `pip install kaleido>=1` — 1.x needs headless Chrome and rocprof-compute pins `kaleido==0.2.1`, so upgrading stops the launcher starting at all | [rocprof.md §12](rocprof.md#12-a-finished-profile-can-still-exit-non-zero) |
| Carrying a decode byte model into prefill | A per-row KV gather is real at 18 rows and fiction at 32 768: the measured call came out at 205–235% of its own memory roof, which disproves the bytes. Bracket it (per-row ceiling, perfect-reuse floor) instead of publishing either | [rocprof.md §8](rocprof.md#8-eff--100-is-a-disproof-of-the-denominator) |
