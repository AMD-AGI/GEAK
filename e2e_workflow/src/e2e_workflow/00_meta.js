export const meta = {
  name: 'e2e-workflow',

  description: 'End-to-end LLM inference-throughput optimizer for AMD Instinct MI-series GPUs (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201, vLLM only). Generic gfx1201 is not a product identity. The serving stack is pluggable via scripts/adapters/<backend>.sh (sglang + vllm + ATOM shipped on Instinct; R9700 is vLLM-only). A system layer (e2e Director / System Architect / Profiler / Config Tuner / Kernel Extractor / e2e Integrator) wraps the UNCHANGED single-kernel kernel_workflow: it preflights the env, profiles a running server, triages hot kernels by Amdahl, tunes config/backends, extracts hot editable kernels into standalone unittests, recursively optimizes them with kernel_workflow.js, overlays them back, and re-validates serving throughput. Also still optimizes a single kernel (pass-through).',
  whenToUse: 'Optimize the serving throughput of an LLM on AMD Instinct MI GPUs, or on Radeon AI PRO R9700 with vLLM. Use interface/run_e2e.py for structured auto-detection; a direct model-mode call must pass expected_gfx and expected_target before backend/knowledge policy is selected. Pass args.model_path (required) + optional args.backend (sglang|vllm|atom on Instinct, default sglang; vllm only on confirmed R9700) + args.launch_script (optional). For a single kernel, pass args.kernel_path instead and it delegates straight to the kernel layer.',

  phases: [
    { title: 'Setup', detail: 'e2e Director builds the isolated eval dir + records baseline throughput' },
    { title: 'Profile', detail: 'Profiler captures a warm trace -> standardized Top-N' },
    { title: 'Strategize', detail: 'System Architect routes kernels by Amdahl (config vs kernel vs host)' },
    { title: 'ConfigSweep', detail: 'Config Tuner sweeps flags/env/backends FIRST (default ON)' },
    { title: 'TuningSkillset', detail: 'Tuning Specialist runs the VENDORED tuning skillset whole + standalone (its own pre/post A/B) BEFORE HeadKernel, so its share of the gain is attributable' },
    { title: 'HeadKernel', detail: 'highest-%GPU ops (GEMM/attn): extract_op -> backend bake-off (incl. FlyDSL) + aiter-DB/author tune -> e2e gate' },
    { title: 'Milestone', detail: 'loop over editable kernels ABOVE milestone_min_pct% GPU (default 5): plan -> extract -> recursive kernel optimize -> overlay -> e2e gate -> reprofile' },
    { title: 'Finalize-gate', detail: 'finish any e2e A/B left incomplete before the final phase (still OPTIMIZATION work: budget-capped, never runs inside the final reserve)' },
    { title: 'Finalize', detail: 'e2e Integrator assembles the overlay + patch + launch bundle' },
    { title: 'Report', detail: 'System Architect writes the throughput report + grows the playbook' },
    { title: 'Validate', detail: 'e2e Director independently re-measures throughput + arbitrates' },
  ],
};

