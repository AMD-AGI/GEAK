export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) — auto-detected on-box; on R9700 use knowledge/amd_rdna4.md (wave32, WMMA). Generic gfx1201 without R9700 identity is not a calibrated target.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the experience KB by canonical id (geak:kernel:*) — the shared service when credentialed, else the on-disk store, brought level with kb_artifacts/ first — for the best curated patch per optimization direction for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
  ],
};

