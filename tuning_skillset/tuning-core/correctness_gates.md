# Correctness gates

Every accepted config must pass a correctness gate before its timing is believed. A config
that is fast because it computes the wrong answer is the easiest way to ship a regression,
and tuners will happily hand you one.

## Use a relative metric. Never absolute error.

The instinct is `(out - ref).abs().max() < tol`. It does not work, because the magnitude of
a GEMM result — and its rounding error — grows with K.

Measured on MI300X, `1024×1024×K`, random normal inputs, compared against an fp32 reference:

| shape | dtype | max_abs | rel_l2 | err_ratio | SNR (dB) |
| --- | --- | --- | --- | --- | --- |
| 1024×1024×256 | bf16 | 0.2486 | 0.001659 | 0.0000 | 55.6 |
| 1024×1024×1024 | bf16 | 0.4944 | 0.001659 | 0.0000 | 55.6 |
| 1024×1024×4096 | bf16 | 0.9951 | 0.001662 | 0.0000 | 55.6 |
| 1024×1024×16384 | bf16 | **1.9756** | 0.001662 | 0.0000 | 55.6 |
| 1024×1024×256 | fp16 | 0.0311 | 0.000208 | 0.0000 | 73.7 |
| 1024×1024×16384 | fp16 | 0.2487 | 0.000208 | 0.0000 | 73.6 |

**Read the bf16 column: `max_abs` grows 8× (0.25 → 1.98) as K grows 64×, while `rel_l2`
stays flat at 0.00166 and SNR pins at 55.6 dB.** Every one of those rows is a *correct*
GEMM. Nothing is wrong.

So an absolute threshold is unusable: pick `0.5` and you fail correct kernels at large K;
pick `2.0` and you pass genuinely broken kernels at small K. The relative metrics are
K-invariant — that is the property you need in a gate.

Note also fp16 shows ~8× lower error than bf16 at every K (73.7 dB vs 55.6 dB SNR). bf16
trades mantissa bits for exponent range. A gate must be set per dtype; a threshold tuned on
fp16 will reject every correct bf16 kernel.

## Which metric

Pick one and apply it consistently. All three are fine; they differ in what they surface.

**`err_ratio` — fraction of elements outside tolerance.** The convention across the AMD
tuning tools, gated at **`err_ratio < 0.05`**.

```python
mismatched = ~torch.isclose(out, ref, rtol=1e-2, atol=1e-2)
err_ratio  = mismatched.sum().item() / out.numel()
```

Good default: robust to a handful of outliers, catches systematic corruption. Use this
unless you have a reason not to — matching the tools' own convention means your gate agrees
with theirs.

**Relative L2 norm.** One number for whole-tensor agreement; the most sensitive to
systematic error, and cheap.

```python
rel = ((out - ref).norm() / ref.norm()).item()      # expect ~1.7e-3 for bf16 GEMM
```

**SNR in dB.** Same information, log scale, easier to compare across dtypes.

```python
snr_db = 20 * torch.log10(ref.norm() / (out - ref).norm()).item()
```

Rule of thumb from the table: bf16 GEMM lands ~55 dB, fp16 ~74 dB. A kernel that drops
tens of dB below its dtype's baseline is broken even if it hasn't produced a NaN.

## Choosing the reference

The gate is only as good as what it compares against.

- **Compute the reference in higher precision.** `a.float() @ b.float()` for a bf16 kernel.
  Comparing bf16 against bf16 hides exactly the errors you are hunting.
- **Compare against a trusted implementation of the same math**, not a different algorithm.
  A split-K kernel accumulates in a different order than a non-split-K one and will differ
  slightly — that is expected, not a failure.
- **For quantized ops, dequantize the reference the same way the kernel does.** Most false
  alarms in fp8/int8 GEMM tuning come from a mismatched scale convention, not a bad kernel.
- **Establish the baseline error first.** Run your gate against the *known-good* existing
  kernel. That number is your floor; a candidate is suspect when it is materially worse,
  not when it is merely nonzero.

## Inputs matter

Random normal inputs are the common default and are what produced the table above. Be aware:

- **All-ones or small-integer inputs hide errors** — they are exactly representable, so
  rounding differences vanish and a broken kernel can pass. Several bench tools default to
  integer init for speed; that is fine for timing, not for correctness.
- **Adversarial magnitudes surface overflow.** fp16 has a narrow range; large-magnitude
  inputs at large K can overflow to inf where bf16 would not.
- **Test the shapes you will deploy.** A kernel correct at 4096³ can be wrong at M=1 or at
  a non-multiple-of-tile size, where masking and edge handling engage.

## FP8: check the dialect before blaming the kernel

On gfx942 (MI300X), FP8 is **FNUZ**. On gfx950 (MI355X), it is **OCP**. `e4m3fnuz` and
`e4m3fn` share an identical 8-bit layout but use a different exponent bias — so
reinterpreting one as the other produces values off by a power of two, with no crash and no
NaN.

If an fp8 kernel is uniformly wrong by a constant factor, suspect the dialect before the
kernel. This is also why fp8 tuned artifacts must never be moved between the two parts.

Support **inverts**; it does not extend. Measured on both, each part computes its own dialect
and refuses the other's. What does not invert is the wording of the refusal:

| | gfx942 given OCP | gfx950 given FNUZ |
| --- | --- | --- |
| error | `HIPBLAS_STATUS_NOT_SUPPORTED` | `could not find valid hipblaslt solution` |

hipBLASLt emits that second string for an unsupported *shape* as well. So a harness that
learned to spot a dialect mismatch by matching the gfx942 text reads the gfx950 failure as
"no solution for this shape, try another" — and proceeds to tune around a numerics bug. Match
on the dtype you asked for, not on the message you got back.

The practical guard is to check what a quantizer *produced* rather than what you requested.
`aiter`'s per-token-group fp8 quantizer picks its output dtype from shipped tables that are
still FNUZ-keyed in places, so on gfx950 it can hand you FNUZ-encoded data that allocates
fine and then fails, or worse, silently feeds a downstream kernel expecting OCP. One line
comparing `out.dtype` against the arch's dialect catches it at the boundary instead of three
kernels later.

## Where the gate goes in the loop

Gate **before** you believe a timing, and gate **again** after deploying into the live path —
a config can be numerically fine in isolation and wrong once real (non-random, possibly
outlier-heavy) activations flow through it.

For end-to-end serving changes, add a task-level check on top of the numeric one. A tuned GEMM
that shifts model output is not a win regardless of what the throughput number says — but
**measure the floor before you read the result.**

"Greedy decoding produces identical output" is not true of a real server. Reduction order in
batched kernels depends on batch composition, so two runs of the *unchanged* server disagree as
well. Measured (sglang, Qwen3.5-27B fp8, 16 prompts, temperature 0, 128 new tokens):

| comparison | concurrency 8 | one request at a time |
| --- | --- | --- |
| reference vs **itself** — the control | — | 15/16 exact |
| reference vs tuned candidate | 3/16 exact | 15/16 exact, 0.972 mean common prefix |

The concurrency-8 reading of 3/16 looks exactly like a numerics failure and is not one: it is the
server's own nondeterminism. Without the control it would have rejected a genuine +23.9%
throughput win. Two rules follow:

- Issue the requests **one at a time**, so batch composition is fixed and the kernels are the only
  thing varying.
- Always run **reference against reference** first. The candidate has to match the *floor*, not
  match perfection.

Parity is a consistency check, not an accuracy check — it says the model stayed on the reference's
trajectory, not that the answers are right. Where correctness of the answer is what matters, use a
task metric with a tolerance on a seed-pinned subset (GEAK's gate is
`cand_exact_match >= ref_exact_match - 0.01` on GSM8K) and score **both** legs yourself; an
absolute published score is not a substitute for measuring your own reference.
