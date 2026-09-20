# Reproduction scripts for `amd_rdna.md`

These scripts re-take **some** of the `[measured]` claims in `../amd_rdna.md` on
an `xconucstrhalo25`-class box (Radeon 8060S / gfx1151 / RDNA 3.5, ROCm 7.2.3),
inside a `rocm/vllm-dev` container. They are small and standalone on purpose:
a reader who doubts one of the covered numbers should be able to re-take it in
minutes.

**Coverage is partial.** These `[measured]` claims have **no script here** and
rest only on the original run logs -- treat them as the weakest claims in the
file and re-measure before relying on them:

| claim | where |
| --- | --- |
| pure-read bandwidth 782 / 913 / 945 GB/s (the read-only ablation) | §3 |
| tuned GEMM at 128³ / 256³ / 512³, hipBLASLt 1.6-2.1x | §5 |
| skinny GEMV, Triton ~1.8x | §5 |
| `shortk_512`, Triton 0.32x | §5 |
| the ~27 s **warm**-process profiling pass | §7 item 6 |

`bw.py` measures read-read-write, not pure read, so it does not back the first
row; `wmma_check.py` covers only 2048³, so it does not back the next three.

**Read section 8 first.** The box must be quiet -- stale processes burning a few
cores silently invalidated a whole batch of measurements once (an 18% error), and
the 2-sigma noise floor is only 0.4%, so anything smaller is not a result.

- **`bw.py`** -- Streaming read-read-write bandwidth vs working-set size. Backs the two-roofline table in section 3 (about 790 GB/s inside the 32 MB Infinity Cache, 229-233 GB/s from DRAM) and the note that the 256 GB/s paper figure is ~12% optimistic.
- **`drift.py`** -- Sustained-load drift over 4 minutes. Backs the +0.03% (no drift) claim, and its caveat that this characterises a MEMORY-bound load only.
- **`noise.py`** -- Idle run-to-run dispersion. Backs the 0.18% CV / 0.4% 2-sigma noise floor in section 8 -- the threshold below which a delta is not a result.
- **`pmc_slots.sh`** -- Sweeps one cumulative counter set 1..5 deep to find where rocprofv3 stops accepting it in a single pass. Backs section 7 item 5 (this set gets three deep; the fourth aborts with error code 38). It is a statement about that set, not about the number three. Runs pmc_work.py.
- **`pmc_work.py`** -- Trivial GPU workload used by pmc_slots.sh. Each invocation starts a FRESH python3, which is what makes the ~85 s cold-start figure in section 7 item 6 reproducible.
- **`triton_caps.py`** -- 6-point Triton capability matrix (elementwise, reduction, tl.dot fp16/bf16, autotune, atomics). Backs section 6.
- **`wmma_check.py`** -- Disassembles a tl.dot kernel and counts v_wmma / v_mfma / scalar v_fma, checks correctness, and compares against the vendor path at 2048³. Backs the claim that tl.dot emits real WMMA on this part and zero MFMA (§2/§6), and the 2048³ row of §5. **Read the retraction note in §5 before citing that row:** an earlier, unfair version of this script is what produced the since-withdrawn "Triton 1.98x faster" figure.

Two scripts referenced by the doc are already in this repo and are not duplicated
here: `perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py`
(the VGPR/occupancy table) and `kernel_workflow/scripts/gpu_lock.sh`.
