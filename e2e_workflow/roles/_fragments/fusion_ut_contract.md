# Fusion UT contract

Read by the UT agent of a fusion region, a kernel candidate of `e2e_workflow.js` that names a region of
the run's fusion profile (KernelFusion switched on): the kernel-extract skill on an AgentX trace replay
(`kernel_extractor`, PHASE=prepare), `kernel_extractor` PHASE=extract otherwise (in the Milestone loop, or in
HeadKernel for a region on a head's own kernel).
Your task exists so that the next step can REPLACE the whole region with fewer (usually one) kernel
launches: the GEAK kernel layer (`kernel_lane.js`) optimizes the task you build, and the e2e integrator
rebinds the region's seam (`candidate_bind`) to the result on the live server.

## FUSION UT CONTRACT

The UT must not pin the current kernel structure:

1. `kernel_src/` holds an editable copy of the seam callable (the host-side function whose body launches
   the region) together with the kernels it launches; the case runner and the timer call that seam
   callable from kernel_src with the recorded inputs -- never the member kernels directly -- so a
   candidate may change how many kernels the region launches, their grid/split and their intermediates.
2. Correctness compares only the region's outputs at the seam boundary (and any declared in-place
   mutation) with the frozen live baseline; intermediates between member kernels are not checked.
3. Execution/binding proof that every member kernel fires is required for the BASELINE leg only; for the
   candidate leg require only that the kernel_src seam implementation ran, never that the original member
   kernels still launch or keep their count/split.
4. The negative control mutates the OUTPUT of the kernel_src seam implementation (e.g. perturb what the
   host function returns or writes), so it still triggers after the members are fused away.
5. Time one call of the whole seam per case (launch overhead included) -- the win of a fusion is fewer
   launches and fewer HBM round-trips.
6. The cases cover EVERY phase in which the live server calls the seam (prefill / extend chunks of every
   served width, decode, target verify, draft), each with its served shapes and weight -- not only the
   phase the region was found in: a fused kernel that wins at verify rows can lose badly at prefill
   widths. Find them from the trace digest's step kinds and the source, not only from the region's listed
   phases, and check the cases against the served distribution (`WORKLOAD_SUMMARY` /
   `PROFILE_WORKLOAD_JSON` when given, else the baseline server log, and the region's phases / m_range):
   add every served shape with real weight that the cases lack, and weight each case by its served share.

Record `"fusion_contract": true` in meta.json.

## AgentX: the kernel-extract skill (PHASE=prepare)

Explicit user scope (overrides the default selection): build exactly ONE task for the single entry of
FUSION_SCOPE, named by its `task`. The entry is a FUSION REGION: declare its device_kernels as one joint
boundary (joint_ut.phases) owned by that task, reached through the entry's seam (or the narrowest
enclosing live callable; record why). The candidate leg will rebind that callable to one fused
implementation, so the oracle is taken at the region boundary. Reuse this run's evidence
(PREVIOUS_WORKLOAD_JSON / PREVIOUS_RANKING_JSON / PREVIOUS_UT_DELIVERY and the traces under TRACE_DIR);
do not launch a new full replay. This region is your ONLY job and the time in REMAINING_MIN is all
yours: do not defer or stop early for budget. When an input is hard to reconstruct (quantized checkpoint
operands, split/partial buffers, library-op wrappers), build it the way the live path does -- load the
real tensors from MODEL_PATH, call the live producer, or bind below the wrapper -- and verify by running
it. Other regions are prepared in parallel on the other GPUs: use only GPU_ID for UT work and write only
under EVIDENCE_DIR and your task dir. Complete source extraction, UT authoring and GPU validation, keep a
failure in the delivery with its concrete blocker, pack with `agentx_kernel_preparation.mjs
--emit-prepared` and return ONLY {delivery_path, prepared} (prepared = that command's stdout VERBATIM).
Do not optimize or integrate.

## Other workloads: kernel_extractor (PHASE=extract)

Build the immutable unittest task dir for a FUSION REGION, not a single kernel. The baseline leg is the
live, unfused region reached through ONE Python seam (KERNEL.target_callable, a function whose body
launches every kernel in KERNEL.device_kernels); candidate_bind rebinds that seam, so the candidate leg
runs the fused implementation from kernel_src/. Record I/O at the seam boundary (KERNEL.inputs_outputs)
for the oracle. If the given seam does not cover the region, use the narrowest enclosing callable that
does and say so in notes; if none exists return editable:false with notes "no_rebind_seam".
