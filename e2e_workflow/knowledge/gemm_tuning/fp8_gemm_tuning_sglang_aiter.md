---
name: fp8-gemm-tuning-sglang-aiter
description: Use when trying to optimize end-to-end SGLang performance with gemm tuning for FP8 models on AMD HIP/ROCm by replacing the default Triton GEMM backend with a tuned Composable Kernel (CK) path through aiter; this skill is the verified playbook for that entire process, using FP8 block-wise GEMM (gemm_a8w8_blockscale) as the primary worked example—GEMM shape/dispatch logging in SGLang, CK composable-kernel tuning, and AITER_CONFIG_GEMM_A8W8_BLOCKSCALE CSV integration. FP8 blockscale and bpreshuffle should also apply by switch the place for dumping gemm and the ck tool used for tuning. 
---

# FP8 block-wise GEMM tuning (SGLang + aiter)

## Overview

This workflow tunes **FP8 block-scaled GEMM** used on **HIP/AMD** when SGLang runs with **aiter** (`SGLANG_USE_AITER=1`). Stock SGLang **≥ 0.5.6** often routes block FP8 through **Triton** (`aiter.ops.triton.gemm_a8w8_blockscale`) when aiter is enabled. To use **CK** (`aiter.gemm_a8w8_blockscale`) with a tuned kernel table, you pin aiter, capture **(M, N, K)** from a representative server run by **implementing GEMM shape dump and dispatch logging in §5–§6** (trace the path in §4 first), run **aiter’s CK tuner**, point **`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`** at the produced CSV, and switch imports in **`fp8_utils.py`** so the CK symbol is used. Then rerun the same serving/benchmark pipeline and compare logs to baseline.

**Assumptions:** ROCm/HIP GPU, Python env where SGLang and aiter are importable from the workload script, and write access to the **SGLang** sources you run when adding §5 hooks.

**Backend availability (check FIRST; provision, do not silently drop).** This playbook's CK path needs the
aiter **CK tuner** (`csrc/ck_gemm_a8w8_blockscale/`), and the sibling FlyDSL lever needs aiter's
`aiter/ops/flydsl/` wrapper. If `EVAL_DIR/env_report.json` lists a lever in `absent_backends`, it is a
PROVISIONING issue, not a measured no-win — record the actionable remedy and fall back to an available
lever, never drop the head silently:
- **FlyDSL absent** (`import aiter.ops.flydsl` → `ModuleNotFoundError`, the common case): needs BOTH
  (1) the top-level `flydsl` pip package (`pip install 'flydsl>=0.1.5'`, on PyPI) AND (2) a flydsl-enabled
  `amd_aiter` build that ships `aiter/ops/flydsl/`. **`pip install flydsl` alone is INSUFFICIENT** —
  `aiter.ops.flydsl` stays a `ModuleNotFoundError` until the wrapper is present (verified empirically).
- **CK tuner / ckProfiler absent**: install/build the aiter CK tuner (pin per §2) so
  `gemm_a8w8_blockscale_tune.py` is runnable; until then the CK lever is unavailable — use the Triton or
  (if available) FlyDSL author route and flag the missing CK lever.

Similar steps could potentially apply to ck bpreshuffle gemm, etc. 

---

## 1. Resolve SGLang and aiter paths; verify versions

- From the workload script (for example a launch wrapper), read **`PYTHONPATH`**, **`VIRTUAL_ENV`**, explicit **`python -m sglang`**, or `which python3` to find which **SGLang** and **aiter** trees are used.
- **SGLang:** require **≥ 0.5.6** (block FP8 + aiter integration expectations in this workflow). Check with `pip show sglang` or `python -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))"`.
- **aiter:** require a checkout **at or after** commit `303a583c89fe392a39cad7e45d616cc43bde3278`. If the current commit is not a descendant to this commit, you must update the repo either wise issues like systemetic crash could happen. If this commit does not exsit, run git pull command to update the commit info. **THIS IS IMPORTANT**

---

## 2. Pin aiter to the required commit (if the required commit is not an ancestor of current HEAD, or it is not found in local)

Run inside the **aiter** repository root:

```bash
git pull
git checkout 303a583c89fe392a39cad7e45d616cc43bde3278
git submodule sync && git submodule update --init --recursive
# Clean JIT build artifacts before rebuild
rm -rf aiter/jit/*.so
rm -rf aiter/jit/build/*
python setup.py develop
```

Confirm: `git rev-parse HEAD` prints `303a583c89fe392a39cad7e45d616cc43bde3278` (or a descendant if you intentionally stay on newer HEAD after verifying compatibility).

---

## 3. Baseline end-to-end performance

- Run the **same** pipeline you will use after tuning: start **`launch_server`** (or equivalent), then **bench_serving** / your benchmark; capture **server log** and **benchmark output** (latency, throughput, tokens/s, etc.).
- Store logs under a timestamped directory for **before/after** comparison.

---

## 4. Understand which GEMM path SGLang uses (read code; do not assume)

- On HIP with `SGLANG_USE_AITER=1`, inspect sglang library **`python/sglang/srt/layers/quantization/fp8_utils.py`** inside the `_use_aiter` block: imports decide whether **`gemm_a8w8_blockscale`** comes from **`aiter`** (CK) or **`aiter.ops.triton.gemm_a8w8_blockscale`** (Triton).
- **Typical SGLang ≥ 0.5.6:** Triton blockscale import is active; CK import is commented. **Different SGLang revisions may differ**—always read the file you actually run.

---

## 5. GEMM shape finding and dumping support (SGLang edits)

CK tuning needs a faithful list of **(M, N, K)** (and related metadata such as dtype) from runs of **your** workload. In the **SGLang** tree that actually runs **`launch_server`**, add support for two env-driven toggles:

- **`SGLANG_DUMP_AITER_FP8_GEMM_SHAPES`:** when enabled, the server log must contain parseable GEMM shape information for downstream steps—commonly lines tagged **`[GEMM_shape_dump]`** with a **`csv_row: M,N,K,...`** suffix (or an equivalent format you document for §7–§8).
- **`SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH`:** when enabled, emit a concise log that identifies which **`w8a8_block_fp8_linear`** implementation the process is using, so you can cross-check against **§4** (CK vs Triton vs other).

**Implementation approach recommandation:** use **§4** to locate the real block-FP8 / aiter execution path in the checkout, add the smallest set of hooks that fire for the tensors that define **M, N, K**, and keep logging volume sane (for example rank 0 only, or rate-limited if hot).

**Reference implementation** (from a working HIP + aiter layout; **adapt** paths, imports, and callsites to the SGLang version you run—the snippets illustrate behavior, not the only valid layout):

```python
# fp8_utils.py — add near other FP8 helpers (needs: prod, get_bool_env_var, logger)
def log_aiter_fp8_gemm_shape_dump(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: Optional[List[int]] = None,
    layer_prefix: str = "",
) -> None:
    if not get_bool_env_var("SGLANG_DUMP_AITER_FP8_GEMM_SHAPES"):
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:
        pass
    if x.dim() < 2:
        M, K = int(x.numel()), 1
    else:
        M = int(prod(x.shape[:-1]))
        K = int(x.shape[-1])
    N = int(weight.shape[0])
    K_w = int(weight.shape[1])
    block_msg = f" block_size={block_size}" if block_size is not None else ""
    prefix_msg = f" layer.prefix={layer_prefix!r}" if layer_prefix else ""
    csv_row = f"{M},{N},{K_w},{weight.dtype}"
    logger.warning(
        "[GEMM_shape_dump] aiter FP8 block GEMM path%s%s | x.shape=%s weight.shape=%s | "
        "M=%d N=%d K=%d (aiter gemm_a8w8_blockscale / bpreshuffle tune CSV) | csv_row: %s",
        prefix_msg, block_msg, tuple(x.shape), tuple(weight.shape), M, N, K_w, csv_row,
    )
```

```python
# fp8.py — import the helper from fp8_utils, then before aiter_w8a8_block_fp8_linear(...):
if self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear:
    log_aiter_fp8_gemm_shape_dump(
        x=x[0],  # or x when not a tuple
        weight=layer.weight,
        block_size=self.quant_config.weight_block_size,
        layer_prefix=getattr(layer, "prefix", ""),
    )
```

```python
# fp8.py — Fp8LinearMethod.__init__, after self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()
if self.block_quant and get_bool_env_var("SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH"):
    _fn = self.w8a8_block_fp8_linear
    print_warning_once(
        "[Fp8LinearMethod] block_quant w8a8_block_fp8_linear="
        f"{getattr(_fn, '__qualname__', repr(_fn))} ({getattr(_fn, '__module__', '?')})"
    )
```

If your tree differs, reproduce the **same contract** (env-gated shape lines + env-gated dispatch identity) rather than copying file names blindly.

---

## 6. Enable the new env toggles; collect server log

Now that we have made gemm logging possible, add the following flags used to start SGLang:

```bash
export SGLANG_DUMP_AITER_FP8_GEMM_SHAPES=1
export SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH=1
```

Run your workload again (same scenario you will tune). **Validate** that **`server.log`** (or your log path) reflects **`SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH`** (whatever identity string your §5 hook emits) and contains **`[GEMM_shape_dump]`** with **`csv_row:`** for the hot path. If either is missing, return to §§4–5 before continuing.

---

## 7. Parse GEMM shapes from the server log for tuning

**Goal:** Turn the noisy **§6** server log into a **small, deduplicated** artifact that lists every **GEMM problem** you need to tune—at minimum **(M, N, K)**, plus **whatever else** you chose to log in **§5** if the tuner or kernels need it (activation/weight layout hints, **dtype**, block-scale geometry, **bias** presence, layer name, and so on).

**Responsibilities**

- **Parse** lines emitted under **`SGLANG_DUMP_AITER_FP8_GEMM_SHAPES`** (whatever format **§5** uses—often a stable tag plus a **`csv_row:`**-style payload, but the contract is yours as long as **§8** can read it).
- **Normalize** fields into a consistent representation for the next step (strip tqdm/progress junk on the same physical line if needed).
- **Deduplicate** so each distinct tuning key appears once; pick a key that matches how you will build the **untuned** input for **`gemm_a8w8_blockscale_tune.py`** in **§8** (for many flows that is unique **(M, N, K)**; include more columns in the key if you logged extra dimensions that affect kernel choice).
- **Write** the result to a file you own (for example a cleaned log or CSV path) and treat it as the handoff into **§8**—adjust **§8**’s CSV builder if you used a minimal format instead of full log lines.

If the artifact is empty or obviously incomplete, fix **§5–§6** before running the tuner.

---

## 8. CK tuning with aiter’s blockscale GEMM tuner

Work in the **aiter** checkout from §1–§2. The Composable Kernel entry point for this workflow typically lives under:

**`$AITER_ROOT/csrc/ck_gemm_a8w8_blockscale/`**

There you should find **`gemm_a8w8_blockscale_tune.py`** (names may vary slightly by aiter revision). **Do not rely on workspace-specific wrapper scripts**; treat the aiter tree as the source of truth.

**What should do**

1. **Read the tuner and its CLI**  
   Open **`gemm_a8w8_blockscale_tune.py`** and any helpers it imports (for example **`GemmCommonTuner`** / **`mp_tuner`** under **`aiter/utility/`**). Run **`python3 gemm_a8w8_blockscale_tune.py --help`** in that directory after setting **`PYTHONPATH`** so **`import aiter`** resolves (typically **`export PYTHONPATH="$AITER_ROOT:${PYTHONPATH}"`**). Note **input** / **output** file flags, **`--libtype`** (e.g. **`ck`**, **`cktile`**, **`both`**), **`--mp`** (worker count for **`mp_tuner`**—set this to use **all** GPUs you intend to parallelize across, not one, unless you are debugging), and whether **split-K** search (e.g. a **`-k`** flag) is optional and expensive.

2. **Build the untuned shape CSV from §7**  
   The tuner’s **`-i`** input is usually an **untuned** CSV of unique **(M, N, K)** rows (often with an **`M,N,K`** header—confirm with **`--help`** and any aiter docs). Convert the **§7** artifact into that CSV: extract the triples (and drop columns you do not need) in whatever way matches the format **§5** chose and the tuner expects.

3. **Run tuning on a GPU host**  
   **`cd`** into **`csrc/ck_gemm_a8w8_blockscale`**, ensure ROCm/PyTorch **sees every GPU** you want the tuner to use (for example leave **`HIP_VISIBLE_DEVICES`** / **`CUDA_VISIBLE_DEVICES`** unset, or set them to the full set you intend). Invoke the tuner with your untuned CSV as **`-i`** and your desired tuned CSV path as **`-o`**. **Parallelize across all of those devices:** set **`--mp`** to the **full visible device count** (or the explicit parallelism the tuner’s **`--help`** documents), not a single GPU by default—the **§3–§6** workload may have run at **TP&nbsp;< host GPU count**; tuning should still exploit **every** available accelerator to shorten wall time on large shape lists. A typical invocation shape (flags are illustrative—**confirm against `--help`**):

   ```bash
   export AITER_ROOT=/path/to/aiter
   export PYTHONPATH="${AITER_ROOT}:${PYTHONPATH:-}"
   cd "${AITER_ROOT}/csrc/ck_gemm_a8w8_blockscale"
   # ROCm: PyTorch still exposes devices via torch.cuda.device_count(); use all visible GPUs for --mp.
   NGPU="$(python3 -c 'import torch; print(torch.cuda.device_count() or 1)')"
   python3 gemm_a8w8_blockscale_tune.py -i /path/to/untuned_mnk.csv -o /path/to/a8w8_blockscale_tuned_gemm.csv --libtype both --mp "${NGPU}"
   ```

---

## 9. Switch SGLang to CK `gemm_a8w8_blockscale` (not Triton)

In **`fp8_utils.py`**, inside `if _use_aiter:`:

- **Use CK:** import `gemm_a8w8_blockscale` from **`aiter`** together with `gemm_a8w8_bpreshuffle`, `get_hip_quant`.
- **Stop using Triton blockscale:** comment out `from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale`.

Target pattern:

```python
from aiter import gemm_a8w8_blockscale, gemm_a8w8_bpreshuffle, get_hip_quant
# from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
```

(Revert or guard behind a local branch if you need to compare Triton vs CK quickly.)

### 9.1 CRITICAL — the activation-scale LAYOUT is per-kernel; match it to CK (do not assume)

This is the single most dangerous, silent failure in this whole flow. The per-token activation scale
has a **memory layout that each aiter kernel fixes**, and you MUST feed CK the layout *it* expects:

- `aiter.ops.triton.gemm_a8w8_blockscale` (Triton) → **non-transposed** scale.
- `aiter.gemm_a8w8_blockscale` (plain CK, the kernel this skill switches to) → **non-transposed** scale.
- `gemm_a8w8_blockscale_bpreshuffle` (the stock non-Triton kernel) → **transposed** scale.

Stock sglang `fp8_utils` computes `transpose_scale = not use_triton`, i.e. it produces the **transposed**
layout for the non-Triton path — correct for *bpreshuffle*, **WRONG for plain CK**. If you switch the
kernel to CK but leave that logic, every **multi-token (M≥2)** GEMM gets scrambled per-token scales →
**~19% error → garbage output / accuracy collapse**, while the tuner's `errRatio` stays 0 and an **M=1
smoke test looks perfectly fine** (transposing a 1-row scale is a no-op). This is exactly how a
"fast but wrong" config ships undetected.

**Do not hard-code the layout — DETECT it.** The correct value is aiter-version / kernel / GPU-arch
coupled, so verify it empirically every run with a tiny check: run the CK kernel in BOTH scale layouts
and compare each against the **Triton** kernel (the production-correct reference) at **M≥2**, on
identical inputs. Run this in a scratch script on one GPU (no server, seconds). The reference must be a
**single, layout-independent** result — derive it from Triton (or from an fp32 dequant of the
*non-transposed* quantization); do NOT re-derive the reference from each layout's own scale tensor, or
the wrong layout will falsely "match" itself.

```python
# scratch check — pick the activation-scale layout the CK kernel actually wants. M MUST be >=2.
import torch, aiter
from aiter import get_hip_quant
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale as triton_gemm
ck = aiter.gemm_a8w8_blockscale
qa = get_hip_quant(aiter.QuantType.per_1x128); FP8 = aiter.dtypes.fp8; FP8_MAX = 240.0  # e4m3fnuz
relerr = lambda a, b: ((a.float()-b.float()).norm()/(b.float().norm()+1e-9)).item()
M, (N, K) = 16, (5120, 5120)                      # use a LIVE (N,K) from §7; repeat per family
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
wf = w.float().reshape(N//128,128,K//128,128); s = wf.abs().amax((1,3),keepdim=True).clamp(min=1e-8)/FP8_MAX
wq = (wf/s).clamp(-FP8_MAX,FP8_MAX).to(FP8).reshape(N,K); ws = s.reshape(N//128,K//128).contiguous()
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / K**0.5
xq, xs = qa(x, quant_dtype=FP8, transpose_scale=False)
ref = triton_gemm(xq, wq, xs, ws, dtype=torch.bfloat16)            # ONE fixed, trusted reference
for flag in (False, True):
    xqi, xsi = qa(x, quant_dtype=FP8, transpose_scale=flag)
    print(f"transpose_scale={flag}:", relerr(ck(xqi, wq, xsi, ws, dtype=torch.bfloat16), ref))
```

Decision:
- Pick the `transpose_scale` whose rel-err is ~0 (≈1e-3). Set the CK path in `fp8_utils` to that value
  (and make the `input_scale is not None` branch consistent — e.g. do NOT transpose for CK if the winner
  is `transpose_scale=False`).
- If **both** layouts give large error for these shapes, CK is numerically broken here → **do not deploy
  CK; keep the Triton baseline** (a faster-but-wrong server is a regression).

For the aiter build this skill was validated against, the winner is **`transpose_scale=False`**
(non-transposed). Treat that as the expected-but-unverified default — the check above is the source of
truth, because at **M=1 both layouts look identical** (transposing a 1-row scale is a no-op), so this
must be checked at M≥2.

---

## 10. Point aiter at the tuned CSV; rerun and compare

```bash
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/path/to/ck_gemm_json_out/a8w8_blockscale_tuned_gemm.csv
```

Restart SGLang with the **same** model, TP, concurrency, and benchmark flags as the baseline. Compare **e2e latency / throughput** and server logs to the baseline from §3. You should see improved performance when the hot shapes are covered by the CSV and the CK path is active.

Example env block (from a working script): `GEMM_tuning_test4/run_sglang_test_fff.sh` (shape/dispatch exports assume the §5 patches are in the SGLang tree that script runs; tune `AITER_CONFIG_...` separately). NOTE: that reference script launches **stock Triton** (no CK env/overlay) — it is a *baseline* launch, not the CK candidate. Do not use a stock launch to "validate" the CK candidate (see §10.1).

### 10.1 MANDATORY correctness check — output parity on the ENGAGED CK server, at M≥2

Throughput and the tuner's `errRatio` are **NOT** correctness signals — a numerically-broken CK path
runs at full speed and `errRatio` stays 0. Before reporting any speedup, confirm the CK swap is
numerically sound with an **output-parity check on the exact CK-engaged server**:

1. **Same-server, same-config.** Launch ONE server with the CK switch + `AITER_CONFIG_...` engaged, and
   measure against *that* server. Never validate on a separately-launched **stock** server — it silently
   runs Triton and "passes" while the real CK candidate is broken. Confirm the CK path is actually live
   in `server.log` (CK module loaded / "is tuned on") during the check.
2. **Greedy (temp=0) output parity vs the Triton baseline**, on prompts long enough to drive
   **multi-token decode (M≥2)** — short / M=1 prompts can look fine while M≥2 is garbage. Outputs must
   match the baseline (or be coherent). This is the skill's own go/no-go: if they diverge, the CK
   integration is wrong (almost always the §9.1 scale layout) — fix the layout (or fall back to Triton)
   and re-check. Do not report a speedup that fails this.

Note: **task-level accuracy gating (e.g. gsm8k) is decided and run by the outer GEAK workflow** (its
accuracy-gate option), not prescribed here. This skill only guarantees the GEMM swap is numerically
correct (the parity check above); if the workflow enables a task-accuracy gate, it must likewise run on
the engaged CK server, not a stock stand-in.

---

## Checklist (verified)

| Step | Action |
|------|--------|
| 1 | Locate SGLang + aiter; verify SGLang ≥ 0.5.6; verify aiter ≥ commit `303a583c...` |
| 2 | If needed: `git checkout` + submodules + clean JIT + `python setup.py develop` in aiter |
| 3 | Save baseline benchmark + `server.log` |
| 4 | Read `fp8_utils.py` / `fp8.py`; trace block FP8 → aiter path (§4) |
| 5 | Implement §5: env-gated **`SGLANG_DUMP_AITER_FP8_GEMM_SHAPES`** (shape lines for §7) and **`SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH`** (which `w8a8_block_fp8_linear`); run from patched tree |
| 6 | Export `SGLANG_DUMP_AITER_FP8_GEMM_SHAPES=1` and `SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH=1`; confirm dispatch logging and `[GEMM_shape_dump]` / `csv_row` in server log (§6) |
| 7 | Parse §6 log → deduped shape artifact per §7; hand off to §8’s untuned CSV builder |
| 8 | In aiter `csrc/ck_gemm_a8w8_blockscale`: read `gemm_a8w8_blockscale_tune.py` + `--help`; build untuned **M,N,K** CSV from §7; run tuner → tuned CSV for §10 |
| 9 | Edit `fp8_utils.py` imports: CK `gemm_a8w8_blockscale` on, Triton blockscale off |
| 9.1 | **Detect the activation-scale layout** (§9.1 scratch check: CK vs Triton at M≥2, both layouts); set the CK path's `transpose_scale` to the winner (expected `False`/non-transposed); if both fail, keep Triton |
| 10 | `export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=...`; rerun; compare to baseline |
| 10.1 | **Correctness check on the ENGAGED CK server, M≥2:** greedy output parity vs Triton baseline (never throughput/`errRatio` alone). Task-accuracy gating (gsm8k) is the outer GEAK workflow's decision, not this skill's |

---

## Pitfalls

- **Wrong Python / wrong tree:** §5 edits must live in the same SGLang tree / interpreter that runs **`launch_server`**.
- **Env toggles and tree alignment:** **`SGLANG_DUMP_AITER_FP8_GEMM_SHAPES`** and **`SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH`** take effect only in the checkout where you added the §5 hooks; keep **`PYTHONPATH`** / editable installs aligned with **`launch_server`**.
- **Scale-layout mismatch (SILENT, CATASTROPHIC — see §9.1):** each aiter blockscale kernel fixes its
  own per-token activation-scale memory layout. Triton and plain CK `gemm_a8w8_blockscale` want
  **non-transposed**; `bpreshuffle` wants **transposed**. Stock sglang's `transpose_scale = not use_triton`
  produces the *transposed* (bpreshuffle) layout on the non-Triton path — **wrong for CK**. Feeding CK a
  transposed scale gives **~19% error on every M≥2 GEMM → garbage / accuracy collapse**, yet `errRatio`
  stays 0 and **M=1 smoke tests pass** (transposing a 1-row scale is a no-op). Symptom: throughput up,
  outputs degenerate (repeated tokens, gibberish) once batch/decode has ≥2 tokens. Always run the §9.1
  probe (M≥2) to pick the layout, and gate accuracy on the engaged server (§10.1). This convention is
  aiter-version/arch-coupled — verify, never assume. (This is the bug that produced a "fast but wrong"
  +77%/+124% result that passed throughput+`errRatio` but failed real generation.)
- **Validate the candidate, not a stand-in (see §10.1):** run the parity check (and any outer
  task-accuracy gate) on the SAME server that has the CK path engaged. A separately-launched stock server
  silently measures Triton and "passes" while the real CK candidate is broken — this is how a broken run
  can report a healthy accuracy number.
- **Shape coverage:** Missing **(M, N, K)** in the CSV falls back to the CK **default** kernel for those
  shapes (look for `not found tuned config ... use default` in the log). With the correct scale layout the
  default is still numerically OK (just untuned/slower); coverage is a *performance* concern, not the
  cause of gibberish — gibberish is the scale-layout bug above. Capture the live cuda-graph **decode
  buckets** (not just a few representative M) so engaged decode shapes are tuned. Use §5 **dispatch**
  logging to confirm which **`w8a8_block_fp8_linear`** implementation is active.
- **Tuner CLI drift:** Always use **`python3 gemm_a8w8_blockscale_tune.py --help`** on the checked-out aiter revision; flag names and defaults can change.
- **Skip baseline ck run:** Do not try to run ck backend fp8 benchmark without tuned csv, as it may stuck. The final goal is to correctly compare tuned ck execution with baseline (by default triton probably).
- **Version / checkout before “clever” code fixes:** Most confusing runtime errors in this flow (including aiter JIT failures such as `NameError: name 'aiter_tensor_t' is not defined`) trace back to **§1–§2 not being satisfied**—wrong **aiter** `HEAD`, stale **submodules**, or **stale JIT** (`aiter/jit/*.so`, `aiter/jit/build/*`) from another revision. **Do not** patch around that in application code (for example adding `aiter_tensor_t` shims in `aiter/jit/core.py` or SGLang). **First** realign the tree: run the full **§2** sequence (`git checkout` the required commit, `git submodule sync` / `git submodule update --init --recursive`, remove JIT artifacts as in §2, `python setup.py develop`), confirm **`git rev-parse HEAD`**, then re-run. Only treat the failure as a genuine bug to debug in source if it **still** reproduces on that pinned, clean-JIT checkout.
- **Fair speedup comparison:** Measure speedup against the **original** end-to-end workflow—the **default** block-FP8 path your stack actually uses **before** this skill’s CK switch (often **Triton** `gemm_a8w8_blockscale` when SGLang + aiter route that way; confirm with **`SGLANG_LOG_FP8_BLOCK_GEMM_DISPATCH`** / §4)—captured in the **§3** baseline with the **same** model, TP, concurrency, and benchmark flags. The **after** run is the **§9–§10** configuration: CK **`gemm_a8w8_blockscale`**, **`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`** set to the **§8** tuned CSV, and an otherwise identical pipeline. Do not attribute gains to this skill when the two runs differ in workload shape, scheduler flags, or visible GPUs unless that is explicitly part of the experiment.