# MiniMax-M3-MXFP4 on MI355X — vLLM, TP=8, split-K in the Triton paged-decode kernel

**Measured win: +7.71% output throughput** (4101.59 → 4417.64 tok/s), gsm8k strict-match 0.9447 →
0.9500. Mean TPOT 15.230 → 14.095 ms (−7.45%). The win is **one Triton kernel change**: a split-K
(flash-decoding) variant of vLLM's paged-decode kernel, plus `num_warps=2`. No compilation, no config
table, no new flags.

Found 2026-08-19. **Reproduction status: measured, not yet reproduced from the artifact alone.** Both
arms were built by editing the installed file in place under `site-packages`, and the exported patch
was verified only by round-trip — applying `artifacts/001_paged_decode_splitk.patch` to the pristine
copy shipped here as `artifacts/001_pristine/chunked_prefill_paged_decode.py` reproduces the installed
file byte for byte (`8e4eaad37fd8129a8d419a0d8aab4b6c`). Nobody has yet applied the
exported patch to a clean instance and re-measured 4417.64. The arms were also **blocked rather than
interleaved** across restarts (two baseline instances, then two patched). See "Reproducing this"
for the gap and what closing it requires.

> The frozen baseline itself *has* been reproduced independently: 4051.90 tok/s on a clean node
> against the bundle's published 4050.5, agreeing to 0.04%. It is the +7.71% that is
> single-source.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 8× MI355X, `gfx950`, **256 CU** each | **yes** — `choose_segm()` sizes the split against the CU count |
| container | `vllm-openai-rocm:v0.26.0` @ `sha256:5709fafe47123becb2f5e61c32d0b97beff1a629bb40bb753c15464f69a97a18` | descriptive, but it pins everything below |
| vLLM | **0.26.0**, installed to `site-packages` with no git checkout | **yes** — the file path, the backend selection and the graph-capture behaviour are all version-specific |
| torch | 2.11.0+gitd0c8b1f | descriptive |
| ROCm / Triton | 7.2.3 / **3.6.0** | descriptive |
| aiter | in `site-packages`, **no git metadata, no commit sha recoverable** | **yes** for path selection — the MoE runs `AITER_MXFP4_MXFP4` |
| model | MiniMax-M3-MXFP4, **TP=8** | **yes** — 8 Q heads and 1 replicated KV head per rank is what makes the decode grid too small, which is the whole reason split-K helps |
| weights | **quark MXFP4 W4A4 on the MoE expert GEMMs only**; attention, indexer, router and the dense layers stay bf16 (681-entry exclude list) | **yes** — the patched kernel serves the *dense* layers, not the MXFP4 ones |
| KV cache | `--kv-cache-dtype fp8` (e4m3, uncalibrated, scales 1.0); the indexer's own KV stays bf16 | **yes** — the kernel's fp8 branch is what was tuned |
| attention | ROCM_ATTN; Triton sparse attention plus a Triton indexer at `topk_blocks=16` | **yes** |
| cudagraph | `FULL_AND_PIECEWISE`, compilation `NONE` | **yes** — decode runs entirely from graphs captured at startup |

**The single most load-bearing flag is `--block-size 128`,** and it is not obvious why. The sparse
attention backend forces it, and vLLM's C++ `paged_attention_rocm` kernel accepts only block sizes 16
or 32 on gfx9. So the C++ path is unreachable and decode falls back to the **Triton** kernel — which
is the only reason a Triton patch is on the hot path at all. On a configuration that reaches the C++
kernel this artifact is inert and silent.

## Launch configuration

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

vllm serve <MiniMax-M3-MXFP4> \
  --host 0.0.0.0 --port 43150 \
  --tensor-parallel-size 8 \
  --max-model-len 13312 \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

Those two variables are the whole env recipe for the server; the harness adds only
`BENCH_TRUST_REMOTE_CODE=1`, `HF_HUB_TRUST_REMOTE_CODE=1` and `MAGPIE_TRUST_REMOTE_CODE=1`, and the
eval adds `HF_HUB_TRUST_REMOTE_CODE=1`. `VLLM_TORCH_PROFILER_DIR` is deliberately unset.

Resolved from the engine log rather than assumed: `quantization=quark`, `dtype=torch.bfloat16`,
`kv_cache_dtype=fp8`, `max_seq_len=13312`, `speculative_config=None`, **`cudagraph_mode`
`FULL_AND_PIECEWISE`**, MoE backend `AITER_MXFP4_MXFP4`, KV pool 7,980,672 tokens / 223.04 GiB per
rank. Startup to `/health` is **~550–630 s**.

`scripts/launch_server.sh` asserts all of the above after `/health` and refuses to let you benchmark
on a mismatch. Two of its checks earn their keep: `cudagraph_mode` must be `FULL_AND_PIECEWISE`, and
the log must contain `Capturing CUDA graphs (decode, FULL)...100%`. If FULL decode capture does not
happen, the patched kernel is never baked into a graph and the change is a no-op that looks like a
null result.

## The boot-required patch: `000_enablement`

**This configuration does not start without it, and it is not an optimization.** Both
`preflight.sh` and `launch_server.sh` refuse to proceed until it is applied. It is present in the
baseline and in the patched arm alike, so it cancels out of the delta entirely — but you cannot
measure anything at all without it, which is why it ships here as `artifacts/000_enablement/`.

What fails without it: under `cudagraph_mode=FULL_AND_PIECEWISE`, vLLM captures a second set of
decode-only FULL graphs, and to capture them it hands the attention metadata builders a **fully
padded dummy batch** — `query_start_loc_cpu` all zeros, so the leading decode query length is 0.
MiniMax-M3's two builders assert it is positive:

```python
qsl_cpu = common_attn_metadata.query_start_loc_cpu
query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
decode_query_len = int(query_lens_cpu[0].item())
assert decode_query_len > 0
```

in `models/minimax_m3/common/indexer.py:317` and `.../sparse_attention.py:288`. The engine dies
before serving with `AssertionError` in `multiproc_executor` and `EngineCore failed to start`.

The patch adds a `padded_capture_batch` guard: when the leading decode query length is not positive,
fall back to the configured uniform decode query length, and relax the
`num_decode_tokens == num_decodes * decode_query_len` invariant for that padded case only. Real
batches are untouched.

**Note what it does not do.** Setting `-cc.cudagraph_mode=PIECEWISE` also boots, and is the obvious
shortcut — but it is a deliberate engine downgrade that changes what you are measuring. Fixing the
builder keeps the default capture mode. If you take the shortcut instead, every number in this entry
becomes incomparable.

```bash
./artifacts/000_enablement/apply.sh            # idempotent
./artifacts/000_enablement/apply.sh --check    # exit 0 if applied
./artifacts/000_enablement/apply.sh --revert   # restore the pristine files
```

The script applies `minimax_m3_uniform_decode_capture_guard.patch` against the vLLM package
directory, greps both target files for the `padded_capture_batch` marker to confirm it took, and saves
the untouched originals into `pristine/` on first use — which is also how `--revert` works and why
those two files ship alongside the patch.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, 8 warmups, seed 0, `random_range_ratio 1.0`,
`--ignore-eos`, against `/v1/completions` with the **InferenceX** fork of `benchmark_serving`
(reference commit `a4bb43afa7fd74c1356583ed29e51421be010f0f`).

Two workload parameters set the tuned kernel's shape, and both are structural rather than incidental:

- **Concurrency 64** gives a decode batch of 64 sequences, and with one KV head per rank the kernel
  launches a `(64, 1)` grid — **64 workgroups on 256 CUs**. The GPU is three-quarters idle during the
  kernel. That starvation *is* the opportunity; split-K exists to fill it.
- **`--block-size 128`** with sequences reaching ~9216 tokens gives ~72 KV blocks per sequence, which
  is what there is to split across.

## Baseline and noise floor

| | tok/s |
| --- | --- |
| baseline, 2 restarts × 3 warm runs (R1 4100.08, R2 4103.10) | **4101.59** |
| with patch 001, 2 restarts × 3 warm runs (R3 4418.54, R5 4416.75) | **4417.64** |
| delta | **+316.05 tok/s = +7.71%** |
| worst case: slowest patched run 4406.97 vs fastest baseline run 4112.68 | **+7.16%** |
| against the bundle's published 4050.5 reference | +9.06% |

| noise floor | spread |
| --- | --- |
| within one server process (3 warm runs) | 0.23–0.49% (baseline R1/R2) |
| **full range across 6 warm runs / 2 baseline restarts** | **0.87%** |
| restart-to-restart, difference of the two baseline instance means | 0.074% (smaller than the SEM) |

**Use 0.87%.** A source change cannot take effect without a restart, so the restart-inclusive spread
is the only floor that applies. The win is roughly 9× it, and the two arms are **disjoint by 294
tok/s — about 20× the pooled baseline σ**. Quote +7.71%; quote +7.16% if you want a number nobody can
argue with.

**Discard the first run after every restart.** Cold runs land at **2384–2508 tok/s, roughly 57% of
warm**, and the harness's own eight warmups do not absorb it because the cost is aiter CK JIT
compilation on first traffic. A cold patched run against a warm baseline run would manufacture a
**1.8×** regression; the reverse manufactures a 1.8× win. This is the largest single trap on this model.

## Deploy

```bash
# 1. the boot-required patch first
./artifacts/000_enablement/apply.sh

# 2. the optimization
VLLM=$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
cd "$(dirname "$VLLM")"
patch -p1 < artifacts/001_paged_decode_splitk.patch

# 3. cache invalidation — both of these, or the change is invisible
find "$VLLM/v1/attention/ops/__pycache__" -name 'chunked_prefill_paged_decode*' -delete
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

# 4. restart. Mandatory.
./scripts/launch_server.sh
```

No rebuild — it is pure Python and Triton JIT. Three ways this silently does nothing: a stale
`__pycache__` keeps the old dispatch; a stale Triton cache keeps the old compiled kernel even with
new Python; and a live-server drop-in changes nothing because decode replays graphs captured at
startup. All three produce a clean benchmark of the *old* kernel.

## Engagement check

```bash
grep 'split-K engaged' /tmp/vllm_server_minimax_m3.log \
  | grep -oE 'Worker_TP[0-9]+|SEGM=[0-9]+' | paste - - | sort | uniq -c
```

- **Engaged:** **24 lines — 8 ranks × 3 SEGM values**, namely `SEGM=4` at 248 sequences, `SEGM=8` at
  120, and `SEGM=16` at 56. All 24 must appear **between** the log lines
  `Capturing CUDA graphs (decode, FULL): 0%` and `Graph capturing finished`.
- **Not engaged:** zero such lines, or fewer than eight distinct ranks.

**The capture window is the whole subtlety here.** The marker fires from Python dispatch, and
steady-state decode never re-enters Python — it replays a graph. So markers appearing *inside* the
capture window are proof the split path was baked into the graphs on that rank; markers absent
*after* the benchmark are expected and mean nothing. A check that looks for activity during serving
will conclude the patch is dead while it is in fact running.

Cross-check against the Triton cache, having cleared it before launch:

```bash
ls ~/.triton/cache/*/kernel_paged_attention_2d_splitk.hsaco  | wc -l   # 3
ls ~/.triton/cache/*/reduce_paged_attention_segments.hsaco   | wc -l   # 3
ls ~/.triton/cache/*/kernel_paged_attention_2d.hsaco         | wc -l   # 1
```

That last line is the interesting one: exactly one unsplit variant survives, for batches of 256+
sequences where `choose_segm()` correctly returns 1 and the original kernel is still the right
answer.

## Accuracy gate

gsm8k 5-shot, **lm-eval 0.4.9.2**, InferenceX task variant, `--apply_chat_template`
`--fewshot_as_multiturn`, `max_tokens=9216`, `temperature=0`, 1319 problems.

| config | `exact_match,strict-match` | flexible-extract |
| --- | --- | --- |
| bundle reference, frozen config | 0.9462 ± 0.0062 | 0.9454 ± 0.0063 |
| baseline, measured here (R1) | **0.9447 ± 0.0063** | 0.9439 ± 0.0063 |
| patch 001 (R3) | **0.9500 ± 0.0060** | 0.9500 ± 0.0060 |
| patch 001 (R5) | **0.9568 ± 0.0056** | 0.9560 ± 0.0056 |

Threshold is no regression against the locally measured 0.9447. Both patched instances clear it. The
patch makes **no claim to improve accuracy** — 0.9447 → 0.9568 is about 2 stderr and the baseline was
evaluated once. Offline, the split kernel's maximum relative deviation from the unsplit kernel is
**6.5e-03**, roughly one bf16 ULP, which is the reason a numerics gate was run at all.

## The change itself

`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`. The patch adds
`kernel_paged_attention_2d_splitk` and `reduce_paged_attention_segments` beside the existing
`kernel_paged_attention_2d`, and a `choose_segm()` that decides between them:

```python
wgs = num_seqs * num_kv_heads
if wgs <= 0 or wgs >= num_cus:
    return 1
want = triton.next_power_of_2(max(1, (2 * num_cus + wgs - 1) // wgs))
return max(1, min(want, cap))          # cap = 16
```

A third grid dimension partitions each sequence's KV blocks across workgroups; each segment carries
its own running `(max, expsum)` and the reduce kernel rescales them against a common maximum. It is
flash-decoding, applied to a kernel that did not have it. At 64 sequences and one KV head on 256 CUs
this gives **SEGM=8, so 512 workgroups instead of 64**. When the grid is already big enough the
function returns 1 and the original kernel runs unchanged.

`num_warps=2` rather than the shipped 4 is not a detail — the two changes are coupled. **Unsplit at
`num_warps=2` is 0.46×, i.e. worse than half speed; split at `num_warps=2` is 5.98×.** Tuning either
alone finds nothing.

| candidate | median µs | vs baseline |
| --- | --- | --- |
| baseline, `num_warps=4` | 813.1 | 1.00× |
| **SEGM=8, `num_warps=2`** | **136.0** | **5.98×** |

**And 5.98× on the kernel became +7.71% end to end** because that kernel is ~10.9% of a decode
iteration — it serves only the 3 dense layers out of 60. The naive prediction from Amdahl was +5.4%;
the measured +7.71% is better because the benchmark is more decode-bound than the trace suggested.
The reconciliation that matters: TPOT dropped 1.135 ms over roughly 3072 decode steps, which is 3.49
s, against a measured wall-clock saving of 3.43 s. Within 2%, so the mechanism is understood rather
than merely correlated.

Selection was coordinate descent on a graph-captured offline harness with interleaved A/B, 5 rounds ×
30 replays, at the profiled shape (64 sequences × ~9096 KV tokens, 8 Q heads, 1 KV head, head dim
128, fp8 cache, 128-token blocks). The transferable rule: **`num_warps=2`, and size SEGM so that
`num_seqs × SEGM ≈ 2 × CU count`.**

## What was tried and did not work

| attempt | kernel level | end to end | verdict |
| --- | --- | --- | --- |
| prefill context tile, `BLOCK_SIZE` 32 → 128 in `prefix_prefill.py::_fwd_kernel` | **1.58×** median, max rel err 1.35e-04 | **+0.062%** | **Not shipped.** Engagement was confirmed on all 8 ranks, so this is a real kernel win that is 14× *below* the noise floor. The canonical negative on this model. |
| `waves_per_eu=2` on that same kernel | 1.45× alone, **1.12% when stacked** with the tile change | not measured | Anti-synergistic; stacking gave away a third of a worthless win. |
| sparse-decode split width, `minimax_m3_sparse_attn_decode` | best 1.06× (25.8 vs 27.4 µs) against a 0.9–1.7% spread | **not run** | Rejected offline: predicted +0.4%, half the noise floor. Correctly refused to spend a 10-minute restart on it. |
| aiter MoE tuned config table | table is inert — 617 rows, all `cu_num=80`, none fp4 | not measured | Needs a real tuning run, not a drop-in. Open lever. |
| decode MoE `MoeFlatmm` | already ~72% of roofline | ≤1.4× theoretical | Not worth a code change. |
| indexer fp8 cache | not measured | not pursued | Changes model output; needs a numerics audit first. |

**One near-miss is worth more than the negatives.** The first draft of `choose_segm()` keyed its
decision on `max_seq_len` — which, during graph capture, is frozen at a dummy value. It looked
correct offline and would have baked SEGM=1 into every captured graph, shipping a **0.0%** patch that
passed every offline test. It was caught before the restart. Anything that participates in graph
capture must be keyed only on quantities that are real at capture time.

## Reproducing this, and closing the gap

The entry is admitted with an explicit caveat, and this is what it would take to remove it:

1. Fresh container on a node with all 8 GPUs idle. Apply `000_enablement`, launch, and confirm
   `4051 ± 20 tok/s` warm — that part is already independently reproduced.
2. Apply `artifacts/001_paged_decode_splitk.patch` **from the artifact**, clear both caches, restart.
3. Verify all 24 engagement markers inside the capture window.
4. Discard the cold run, take three warm runs, and expect **≈4418 tok/s**.
5. Repeat over at least two restarts per arm, and **interleave the arms** this time
   (`base → patched → base → patched`) rather than running them in blocks, so a slow drift in node
   conditions cannot masquerade as the effect.

Then this entry can drop the caveat and state a reproduction count, like the Qwen3-8B entry does.

## When this entry stops applying

- **`--block-size` 16 or 32** routes decode to the C++ `paged_attention_rocm` kernel and the patch
  becomes **inert and silent**. This is the most likely way to be fooled.
- **Concurrency ≠ 64** changes `num_seqs`; at 256+ sequences `choose_segm()` deliberately returns 1
  and the patch does nothing by design.
- **TP ≠ 8** changes heads per rank and therefore the grid occupancy the whole change is exploiting.
- **CU count ≠ 256** mis-sizes SEGM — the code adapts, but the tuned point does not.
- **vLLM ≠ 0.26.0** may move the file, the backend selection, or the capture behaviour.
- **`cudagraph_mode` downgraded to PIECEWISE** boots and serves at a different throughput entirely.
- **Missing `000_enablement`** is the one failure that is *not* silent: the server will not start.

Still reusable when inert: the occupancy diagnosis (count the workgroups the decode kernel actually
launches against the CU count — this model launched 64 on 256), the `num_warps=2` + SEGM coupling,
the sizing rule, and the graph-capture keying hazard.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/minimax_m3_mxfp4_tuning/`. `FINDINGS.md` §1.3
has the noise floor, §2.1 the split-K sweep, §2.2 the rejected prefill work, §3.3 the Amdahl
reconciliation. `patches/001_paged_decode_splitk/RESULT.md` and `MANIFEST.json` carry the per-run
tables and the deploy contract; `patches/000_enablement/README.md` explains the boot failure in full.
`reference/local_reproduction.md` records the independent baseline reproduction at 4051.90 tok/s on
`crsuse2-m2m-031`. The offline harness is `analysis/bench_paged_decode.py`.
