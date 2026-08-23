# Patches

Three patches. 1 and 2 are against **aiter** at base commit
`d9e5ef7ce08ee7045d583aed768cff41aa9210fe` (`/sgl-workspace/aiter`); 3 is against
**sglang** at `29481685462732237d80d86076d6563e1f658102` (`/sgl-workspace/sglang`). Each
file's header block carries its own measurement and the engagement check that proves it
was live. Full context in `../FINDINGS.md`.

**All three together: 1538.009 → 1985.566 tok/s, +29.05%**, measured as an interleaved
across-restart A/B run *after* all three were frozen (A B A B, four fresh servers, three
runs each, position-matched): 6/6 pairs positive, +28.91..+29.19%, arms disjoint. gsm8k
identical to the pristine baseline on **all 1319 problems**. Details in `../FINDINGS.md` §5.

| # | target | file | marginal effect | gsm8k |
|---|---|---|--:|--:|
| 1 | aiter | `0001-aiter-tuned-gemm-table-qwen3-14b-gfx950.patch` | 1536.610 → 1892.866 tok/s (**+23.18%**) | 0.9454 (bit-identical) |
| 2 | aiter | `0002-aiter-default-experimental-pa-ragged-kernel.patch` | **+3.07%** (interleaved A/B, 6/6 pairs, disjoint) | 0.9447 (1 of 1319 differs) |
| 3 | sglang | `0003-sglang-bpreshuffle-scale-from-quant-kernel.patch` | **+1.42%** (interleaved A/B, 6/6 pairs, disjoint) | 0.9454 (identical) |

Gate is ≥ 0.9328. The floor a delta has to clear is the **0.21% position-matched restart
spread** (`../FINDINGS.md` §1), not the 0.72% pooled min–max. Ambient drift on this machine
reaches **1.8% over an hour** on byte-identical code — larger than patches 2 and 3
combined — so the headline and both small marginals are interleaved rather than compared
across a wall-clock gap. Patch 1's +23.18% is the one figure that is not; at 13× the drift
it does not need to be, and the headline A/B corroborates it to 0.2%.

## Bases and how the pristine copies were obtained

Neither framework is a clean checkout — both working trees carry unrelated local
modifications — so every patch records where its base came from rather than assuming
`git diff` is meaningful:

- **aiter** is a git repo at the base commit above; patches 1 and 2 diff against
  `git show HEAD:<path>`. Patch 1 adds a new file, so it has no base. Patch 2's base — the
  untouched `csrc/cpp_itfs/pa/pa_ragged.py` — is archived at
  `base/aiter-d9e5ef7c-pa_ragged.py` so the patch can be applied and verified without the
  container.
- **sglang** is an editable install and also a git repo at the base commit above. The
  file patch 3 touches was clean at HEAD, verified with `cmp` against the untouched
  working tree *before* any edit, and archived at
  `base/sglang-29481685-fp8_utils.py`. That archived copy is the diff base and is
  checked in here so the patch can be applied and verified without the container.

## Order

Apply 1, then 2, then 3.

- Patch 1 **is** standalone and was measured that way.
- Patch 2's `+3.07%` is marginal, measured interleaved with patches 1 and 3 held
  identical in both arms, so it isolates patch 2. An earlier non-interleaved pass read
  `+3.95%`; that figure is superseded. A reviewer who considers this patch too close to an
  env-var change (see its header) can drop it; 1 and 3 are unaffected.
- Patch 3 touches sglang, not aiter, so it is independent of 1 and 2 in code — but it was
  measured with both applied, and its `+1.42%` is marginal on that stack.

```bash
cd /sgl-workspace/aiter
git apply -p1 <this-dir>/0001-aiter-tuned-gemm-table-qwen3-14b-gfx950.patch
git apply -p1 <this-dir>/0002-aiter-default-experimental-pa-ragged-kernel.patch
rm -rf /tmp/aiter_configs   # derived merge cache; stale unless dropped

cd /sgl-workspace/sglang
git apply -p1 <this-dir>/0003-sglang-bpreshuffle-scale-from-quant-kernel.patch
```

All three were verified — after the final header edits, not just when first written — to
apply cleanly to a pristine copy of their base and to reproduce the measured working tree
byte-for-byte.

To reproduce the **comparison** rather than one arm, use the interleaved driver; a single
arm measured now against a number recorded an hour ago is not a comparison on this machine:

```bash
cd <workdir>
./analysis/ab/run_ab_stack.sh analysis/ab/stack_A.sh analysis/ab/stack_B.sh stackab 2 3
python3 analysis/ab/report_ab.py stackab
```

## No `rejected/`

Nothing was measured end-to-end and discarded. The things that were considered and
dropped were ruled out before they cost a benchmark run — by trace, by source reading, or
because they were frozen configuration — and each is written up under
"Negative and no-op results" in `../FINDINGS.md` §4.3 rather than left as a dead patch
file here.
