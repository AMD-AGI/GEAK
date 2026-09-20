# Reproduction scripts for `amd_rdna.md`

Each `[measured]` claim in `../amd_rdna.md` was produced by one of these on an
`xconucstrhalo25`-class box (Radeon 8060S / gfx1151 / RDNA 3.5, ROCm 7.2.3),
inside a `rocm/vllm-dev` container. They are small and standalone on purpose:
a reader who doubts a number should be able to re-take it in minutes.

**Read section 8 first.** The box must be quiet -- stale processes burning a few
cores silently invalidated a whole batch of measurements once (an 18% error), and
the 2-sigma noise floor is only 0.4%, so anything smaller is not a result.

- **`bw.py`** -- Streaming read-read-write bandwidth vs working-set size. Backs the two-roofline table in section 3 (about 790 GB/s inside the 32 MB Infinity Cache, 229-233 GB/s from DRAM) and the note that the 256 GB/s paper figure is ~12% optimistic.
- **`drift.py`** -- Sustained-load drift over 4 minutes. Backs the +0.03% (no drift) claim, and its caveat that this characterises a MEMORY-bound load only.
- **`noise.py`** -- Idle run-to-run dispersion. Backs the 0.18% CV / 0.4% 2-sigma noise floor in section 8 -- the threshold below which a delta is not a result.
- **`pmc_slots.sh`** -- Bisects how many PMC counters rocprofv3 accepts in one pass. Backs section 7 item 5 (1/2/3 succeed, 4+ abort with error code 38). Runs pmc_work.py.
- **`pmc_work.py`** -- Trivial GPU workload used by pmc_slots.sh. Each invocation starts a FRESH python3, which is what makes the ~85 s cold-start figure in section 7 item 6 reproducible.
- **`triton_caps.py`** -- 6-point Triton capability matrix (elementwise, reduction, tl.dot fp16/bf16, autotune, atomics). Backs section 6.
- **`wmma_check.py`** -- Disassembles a tl.dot kernel and counts v_wmma / v_mfma / scalar v_fma. Backs the claim that tl.dot emits real WMMA on this part and zero MFMA.

Two scripts referenced by the doc are already in this repo and are not duplicated
here: `perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py`
(the VGPR/occupancy table) and `kernel_workflow/scripts/gpu_lock.sh`.
