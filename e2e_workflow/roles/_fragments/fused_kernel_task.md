# Fused kernel for a fusion region

Read by the kernel-layer agents (`kernel_lane.js`) when `e2e_workflow.js` (its Milestone loop, or HeadKernel
for a region on a head's own kernel) hands them a FUSION REGION task (a kernel candidate naming a region of the run's fusion profile, KernelFusion
switched on): ONE fused kernel that replaces a whole kernel region of a live LLM serving stack. The lane `task` text names the region (level, the device kernels it launches today in order, the
phases it runs in, the m range, the seam, the tensors that cross its boundary, the analyst's suggested
design and whether the outputs are expected bit-exact).

The task dir was built under the fusion UT contract (`fusion_ut_contract.md` beside this file):
`kernel_src/` holds the editable copy of the seam callable and the kernels it launches, the cases are the
recorded serving shapes of every phase that calls the seam with their served weights, and the immutable
unittest checks only the region's outputs at the seam boundary against the frozen live baseline and times
one call of the whole seam per case.

What a candidate delivers:

1. The fused kernel: it replaces the region (fewer launches, usually one) behind the same seam
   signature, with the region's outputs unchanged (bit-exact when the task says so). Fewer launches and
   fewer HBM round-trips are the win: keep intermediates in registers/LDS instead of writing them back.
2. Correct and faster on the shapes the server actually runs: the unittest's cases are those shapes,
   weighted by their served share, across every phase in which the server calls the seam (prefill /
   extend chunks of every served width, decode, target verify, draft). Report the per-shape times.
3. GUARD it: inside the seam implementation in `kernel_src/`, call the fused kernel only inside the
   envelope the unit test measured it correct AND faster in (phase, row count / M range, dtypes, shapes);
   every other call -- another phase, a larger M such as a prefill chunk, a batch size you did not time,
   inputs it does not support -- goes to the stock path. State that envelope in your notes.
4. It must survive graph capture (the task text carries the deployment requirement when the server
   captures decode into a CUDA/HIP graph).
