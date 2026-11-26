# Platform known issues: RDNA4 (gfx1201, observed on R9700)

Kept for GEAK's RDNA pages (`perf_knowledge/hardware/rdna4_gfx1201/pitfalls.md`,
`perf_knowledge/hardware/rdna35_gfx1151/pitfalls.md`), which link here. RDNA is outside the gluon_authoring
fact store's scope; its gfx950 / gfx942 (CDNA) items are cards: `python3 ../../scripts/q.py "<words>"`.
GEAK's product policy and evidence for these items are R9700-only.

## PMC availability: per-box preflight

`rocprofv3 --kernel-trace` captured dispatches on R9700, but the PMC counter set and naming on RDNA4 client
parts differ from Instinct (CDNA); CDNA names (`SQ_WAVES`, `VALUInsts`, the `MfmaUtil` / `VALUBusy` family)
may be absent or differ. The exact set depends on the device and the ROCm version.

- `rocprofv3-avail list --pmc` on the box lists the counters that exist. rocprofv3's own listing is
  `rocprofv3 -L` / `--list-avail`; older profiler generations used `--list-basic`, `--list-derived` or
  `--list-counters`, which is a CLI rename, not an image defect.
- When the discriminating counters are unavailable, other counters do not stand in for them; the evidence
  then comes from outside the profiler: an analytical roofline (`kernel_workflow/scripts/kernel_tools/hw_budget.py`),
  a static `.s` audit (`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`), a floor probe, and timing at
  the production boundary.

## `int64_strides` under a CUDA graph

For one Triton attention kernel on R9700, timed at the CUDA-graph boundary, `int64_strides=false` was slower
than `int64_strides=true`; the vectorized-load path appeared to fall back to a non-vectorized one in that
configuration. Scope: a single kernel at that boundary, not established as a general RDNA4 behaviour; an
interleaved timing of both settings on the target shape checks it.
