# Qwen3-8B on MI355X — FP8 serving, aiter tuned-GEMM rows

**Verified win: +3.95% output throughput** (3642.1 → 3786.1 tok/s), accuracy improved
(gsm8k strict-match 0.9280 → 0.9348). The entire win is **four rows of CSV** — a data-only
drop-in config file, no compilation, no source patch.

Found and reproduced 2026-08-18 over a ~3 h agent run. Reproduced twice from the exported
artifact alone on clean instances (3790.2 and 3786.2 tok/s).

## Environment fingerprint

Diff this against your environment before deploying. **Load-bearing** fields are part of the
config lookup key or determine the shapes; a mismatch there means the rows are never found and the
artifact silently does nothing.

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | MI355X, `gfx950`, **256 CU** | **yes** — `gfx` and `cu_num` are columns in the key |
| container | `rocm/sgl-dev@sha256:95a933896aeab2a431521ece6ebe90c1db37a3aaf1e32a938d56ef7ccf6603a5` | descriptive, but it pins the two below |
| aiter | commit `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | **yes** — the `model_configs/` merge path and the schema are version-specific |
| SGLang | 0.5.17 | descriptive |
| torch | 2.9.1+rocm7.2.0 | descriptive |
| model | Qwen3-8B, TP=1, single GPU | **yes** — TP changes every N and K |
| quantization | `--quantization fp8`, `--kv-cache-dtype fp8_e4m3` | **yes** — the rows are keyed on `q_dtype_w=torch.float8_e4m3fn`; a bf16 deploy dispatches a different op entirely |
| GEMM op | `gemm_a8w8_bpreshuffle` (FP8, pre-shuffled B) | **yes** — this is which config table is read |

The tuned rows carry `gfx950,256` and `torch.float8_e4m3fn` literally. On a 304-CU part, or in
bf16, they are unreachable.

## Launch configuration

The win depends on these flags because **they are what set the M values**. Reuse the entry only if
these hold, and see "When this entry stops applying" below.

```bash
python3 -m sglang.launch_server \
  --model-path <Qwen3-8B> \
  --tp-size 1 \
  --context-length 11264 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.68 \
  --chunked-prefill-size 16384 \
  --attention-backend aiter \
  --disable-radix-cache \
  --watchdog-timeout 1800 \
  --trust-remote-code
```

Resolved: `page_size 64`, `max_prefill_tokens 16384`, full HIP-graph capture on decode,
`max_running_requests` unset.

Environment (the six aiter variables are part of the recipe, not decoration):

```bash
SGLANG_USE_AITER=1
SGLANG_USE_AITER_UNIFIED_ATTN=1
SGLANG_USE_AITER_FP8_PER_TOKEN=1
SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
SGLANG_OPT_USE_AITER_SILU_MUL=1
SGLANG_FP8_IGNORED_LAYERS=lm_head
```

**Note on `--mem-fraction-static`:** SGLang rescales it by 0.85 when aiter is enabled and
context length exceeds 8192 (`server_args.py`), so 0.68 resolves to 0.578. Both values are correct
depending on where you read them, and the difference does not affect this workload — the KV pool is
4–5× larger than it needs. Do not "fix" it.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, 8 warmups, seed 0, `random_range_ratio 1`,
measured with the **InferenceX** fork of `benchmark_serving` (its `--num-warmups` and
`--random-num-workers` flags matter; stock vllm harnesses give a different number).

Concurrency 64 is what makes decode run at **M=64**. That is not a coincidence to be tidied up
later — it is half the win.

## Baseline and noise floor

| | tok/s |
| --- | --- |
| stock, this stack (3 instances: 3629.7 / 3655.9 / 3640.8) | **3642.1** |
| with these rows (2 instances: 3773.3 / 3798.9) | **3786.1** |
| delta | **+3.95%** |

| noise floor | spread |
| --- | --- |
| repeating the benchmark inside one server process | 0.014% |
| restarting the server between runs | **0.36%** |

**Use 0.36%.** A config change cannot take effect without a restart, so restart-to-restart is the
only floor that applies (`../../tuning-core/measurement.md` Rule 3b). The +3.95% result is ~11×
that floor, and the stock and tuned instance ranges are disjoint.

For reference, the published Hyperloom config-tuned baseline for this recipe is 3636.9 tok/s;
against that number the win is +4.10%. Stock on *your* instance is the honest comparison.

## Deploy

The artifact is `artifacts/a8w8_bpreshuffle_tuned_gemm_qwen3_8b.csv` — a drop-in file, which is
also how aiter ships its own per-model configs. Nothing vendored is modified.

```bash
cp artifacts/a8w8_bpreshuffle_tuned_gemm_qwen3_8b.csv \
   /sgl-workspace/aiter/aiter/configs/model_configs/

rm -rf /tmp/aiter_configs        # MANDATORY — see below
# then start the server; a running server will not pick this up
```

Equivalently, `artifacts/001_decode_m64.patch` then `artifacts/002_prefill_m16384.patch` via
`git apply` in `/sgl-workspace/aiter` (002 depends on 001; the CSV above is just both applied).

**Two ways this silently fails, both already paid for:**

1. **`/tmp/aiter_configs` is a derived cache and is not regenerated if it already exists.**
   aiter merges `configs/*.csv` with `configs/model_configs/*.csv` into
   `/tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv` on first use. Skip the `rm -rf` and your
   rows are ignored with no message. Verify the merge took:
   `python3 -c "import pandas as pd; print(len(pd.read_csv('/tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv')))"`
   → **2556** with all four rows, 2552 without.
2. **Decode is engaged at HIP-graph capture time**, which happens once at server startup. Applying
   this to a running server has no effect. A restart is mandatory, not hygiene.

The `gate_up` winner is a FlyDSL kernel, compiled in-process at first use. It costs ~5 s of extra
startup (46 s cold vs 41 s warm) and writes no cache, so there is no stale-artifact hazard. The
2.5 GB `flydsl_cache` the *tuner* leaves behind is not needed at serve time and can be deleted.

## Engagement check

Strongest form — kernel identity, which needs no logging flag
(`../../tuning-core/engagement_verification.md` form 4). Profile the decode `gate_up` shape:

```bash
cd /sgl-workspace/aiter && python3 -c "
import torch, aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
x = torch.randn((64,4096), dtype=dtypes.bf16, device='cuda')
xq, xs = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
w = torch.randn((24576,4096), dtype=dtypes.bf16, device='cuda')
wq, ws = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
w0 = shuffle_weight(wq, layout=(16,16))
for _ in range(3): aiter.gemm_a8w8_bpreshuffle(xq, w0, xs, ws, None, dtypes.bf16)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5): aiter.gemm_a8w8_bpreshuffle(xq, w0, xs, ws, None, dtypes.bf16)
    torch.cuda.synchronize()
print([e.key for e in p.key_averages() if e.device_time_total > 0])"
```

- **Engaged:** `['kernel_gemm_0']` — the FlyDSL kernel.
- **Not engaged:** the CK symbol `_ZN2ck48kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle...`.

Secondary, flag-free check on the server log: `grep -c 'shape is M:64, N:24576' server.log`
must be **0** once engaged (it is 1 without the rows, because the *miss* line prints
unconditionally). Do **not** grep for `is tuned on cu_num` unless you also set
`AITER_LOG_TUNED_CONFIG=1` — that line is gated behind the flag and returns zero against a
perfectly working deploy.

## Accuracy gate

gsm8k 5-shot, lm-eval pinned at `b315ef3b05176acc9732bb7fdec116abe1ecc476`, InferenceX task
variant.

| config | `exact_match,strict-match` | flexible-extract |
| --- | --- | --- |
| reference recipe | 0.9280 ± 0.0071 | 0.9318 |
| with these rows | **0.9348** | 0.9401 |

Accuracy went up rather than down, which is expected: the tuner gated every winner at
`errRatio 0.0`, so these are numerically equivalent kernels and the movement is eval noise.

## What was tried and did not work

Worth more than the win itself, because it is what you do not need to repeat.

| attempt | kernel-level result | end-to-end | verdict |
| --- | --- | --- | --- |
| `down_proj` M=16384 tuned row | tuner claimed 19% (591 µs vs 728 µs) | — | **not shipped.** The tuner's `us` is warm-cache; re-timed cold it is a dead tie (727.69 vs 727.61) and slower in an eager profile. See `../../tuning-ck/` §3c |
| prefill M=8192 rows | real, −24.2% on the kernel | +0.083% | **not shipped** — inside the 0.36% restart floor |
| split-K decode variants | no kernel win | 0.0% | **not shipped** |

The general lesson, now written into `../../tuning-core/measurement.md` Rule 5: **a genuine
kernel-level win is not a shippable result.** Two of these had real, reproducible kernel
improvements and moved end-to-end throughput by less than the noise floor.

## When this entry stops applying

The rows are keyed on `(gfx, cu_num, M, N, K, q_dtype_w)`, so any change that moves an M or an N
makes them inert — silently, with no error and no log line:

- **Concurrency ≠ 64** moves the decode M off 64. The M=64 rows stop being hit.
- **`--chunked-prefill-size` ≠ 16384** moves the prefill M off 16384. The M=16384 rows stop being
  hit.
- **TP ≠ 1** changes every N and K (they are sharded). All four rows become unreachable.
- **bf16 instead of FP8** dispatches a different op with a different config table.
- **A different aiter commit** may change the merge path, the schema, or the kernel IDs. Re-check
  that `/tmp/aiter_configs` row count moves when you deploy.

In all of those cases the entry is still useful for its **shape list and method** — harvest your
own M values from a baseline run, tune those, and expect winners from the same families
(`flydsl` for the wide N shapes, `ck` for the narrow one). Do not deploy these rows unchanged.

## Provenance

Task bundle, full run log, and per-patch `MANIFEST.json` / `RESULT.md`:
`tuning_workspace/experiment_standalone/qwen_8b_tuning/` (`patches/001…`, `patches/002…`,
`FINDINGS.md`). The bundle's `reference/preflight_baseline.md` records what stock measures on this
container and the one known environment difference.
