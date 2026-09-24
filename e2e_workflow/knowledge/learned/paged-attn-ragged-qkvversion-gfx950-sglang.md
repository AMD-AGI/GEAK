---
name: aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win
key: paged decode attention (`paged_attention_ll4mi_QKV_mfma16_kernel` + `_reduce_kernel`) · gfx950/MI355X · sglang + aiter, bf16 KV, page_size=1, GQA 32:8, head_dim 128
description: On sglang's aiter attention backend the non-editable CK/HIP paged-decode head DOES have an op-level env knob after all — aiter's JIT template reads $QKV_VERSION per call and ships a faster EXPERIMENTAL kernel for head_dim=128 + bf16 KV. Bitwise-identical output, zero HBM, no overlay.
keywords: [paged attention, decode, ragged kv_indptr, QKV_VERSION, EXPERIMENTAL, aiter, cpp_itfs JIT, split-kv, partition_size, gfx950, sglang, byte-exact env win]
kernels: [paged_attention_ll4mi_QKV_mfma16_kernel, paged_attention_ll4mi_reduce_kernel, paged_attention_ragged, csrc.cpp_itfs.pa.pa_ragged]
platforms: [gfx950/MI355X, ROCm 7.2, sglang 0.5.17, aiter (cpp_itfs JIT)]
kernel_class: attention-decode
regime: decode-only, page_size=1, bf16 KV (kv-cache-dtype=auto), GQA 32:8 head_dim 128, TP=1, ISL 8192 / OSL 1024 / conc 64
type: lever
confidence: ★★★
effect: iso serving-weighted 1.086x-1.145x (geomean 1.18-1.21x) on a 73.6-75.0% head -> +6.3..9.3% e2e ceiling; BITWISE-IDENTICAL to the stock kernel on every case, both runs; e2e PENDING
confirms: 2
lifecycle: active
last_seen: 2026-08-21
---
# The "non-editable CK paged-decode head has no op-level knob" prior is WRONG on sglang+aiter

- lever: `sglang.srt.layers.attention.aiter_backend:paged_attention_ragged` →
  `aiter.ops.attention` → **`csrc/cpp_itfs/pa/pa_ragged.py`**, a *JIT-from-template* path (not a
  prebuilt `.so`). `compile()` reads **`os.getenv("QKV_VERSION", "GOLDEN")`** on EVERY call and folds it
  into the rendered `VERSION_ID` **and into the `get_default_func_name` signature hash** — so
  `QKV_VERSION=EXPERIMENTAL` selects a genuinely different, separately-cached kernel build. There is no
  stale-cache hazard and engagement needs no overlay: it is a pure **`winner_kind=env`** win.
- apply: `QKV_VERSION=EXPERIMENTAL`. Guard in the source: the experimental kernel requires
  **`head_size==128` and `kv_dtype==__hip_bfloat16`**; otherwise aiter prints
  `"EXPERIMENTAL pa_ragged kernel requires head_size=128 and kv_dtype=bf16. Fallback to original kernel"`.
  ⚠ interaction: it therefore does **not** apply once KV is fp8 — do not stack it with
  `--kv-cache-dtype fp8` and assume both.
- measured (Llama-3.1-8B-Instruct, gfx950 TP=1, immutable oracle, interleaved in-process A/B,
  `cuda_event_graph`, 7 rounds — interleaving matters, GPU 0 was shared with a foreign tenant and
  sequential runs drifted ~10%): decode_b64_s9035 **1.086x** (the serving-dominant case),
  decode_b64_ragged 1.035x, decode_b64_s2048 **1.463x**, decode_b1_s9035 1.166x. `torch.equal(golden, experimental) == True` on every case →
  **byte-exact**, so the e2e gate needs no accuracy probe.
- verify: `op_bench.py:bench_attn` takes NO timing for attn by design (`measured:false`,
  `harness_suspect:false` — expected, not a fault). Bench by importing the task's `unittest.py`
  read-only and flipping `os.environ["QKV_VERSION"]` between `ut.call(args)` invocations inside one
  process. Driver kept at `$EVAL_DIR/config/pa_qkvversion_ab.py`.
- caution: the sibling `partition_size` knob (`_AITER_PARTITION_SIZE_ROCM=256`, a module constant in
  `aiter_backend.py`, not an env) looks like the same split-KV lever as the triton backend's, and it IS
  faster (256→512 gave 1.41x on the dominant case) but it is **WRONG**: 128 and 512 both fail the oracle
  (max_rel_err 0.10–1.33, NaN at bs=1) even with `max_num_partitions` recomputed to match. The
  compiled kernel/reduce pair is valid only at 256 here. Measure parity before believing a partition
  sweep.
- caution: the head is HBM-saturated, so the **biggest byte lever is still kv-fp8** and it is a SERVER
  flag: the unittest's report-only `GEAK_KVFP8_PROBE` gave serving-weighted **1.669x**
  (b64_s9035 1.669, ragged 1.563, s2048 1.452, b1 1.234) = **~+43% e2e ceiling at a 75% head**. LOSSY →
  gsm8k accuracy gate, never byte parity. Surface it even when `ENABLE_FP8=false`.
- Tier-C seeds for this op (live impl is non-editable, so route=**author**): aiter ships real editable
  Triton paged-decode kernels to seed from — `aiter/ops/triton/attention/pa_decode.py`,
  `lean_atten_paged.py`, and the Gluon `aiter/ops/triton/gluon/pa_decode_gluon.py` (none is seam-
  compatible as-is: they take `block_tables`, not the ragged `kv_indptr/kv_page_indices` trio). A HIP
  author lane is unusually cheap here because the live kernel's own source (`pa_ragged.cuh` /
  `pa_kernels.cuh` / `pa_common.cuh`) is on disk and hipcc-JIT-compiled at runtime — no image rebuild.
  `ckProfiler` is absent from this image, so a standalone CK instance sweep is not available.
- confirm #2 (2026-08-21, Qwen3-8B gfx950 MI355X sglang 0.5.17 TP=1, DIFFERENT model, same geometry
  32q/8kv/d128, 73.62% head): reproduced independently on GPU2 with 5 interleaved rounds —
  decode_b64_s9035 **1.145x**, b64_ragged 1.069x, b64_s2048 **1.485x**,
  b1_s9035 1.184x; serving-weighted **1.145x**, geomean
  1.211x, `torch.equal` True on all 4 cases. Amdahl ceiling +9.3% e2e. `QKV_VERSION` is the ONLY
  `os.getenv` in `pa_ragged.py` (grep-verified) — there is no second env knob to stack.
- source: exp/e2e_*Llama-3.1-8B-Instruct*/ 2026-08-21 (e2e_cycle2, gfx950 MI355X TP1, 75.01% head: QKV_mfma16 +
  reduce; oracle PASS rel<=0.0041 incl. graph_replay + random-value parity; identity baseline captured
  per bucket over decode_b64_s9035 / b64_ragged / b64_s2048 / b1_s9035 — absolute timings in EVAL_DIR)
