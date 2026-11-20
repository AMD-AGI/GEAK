- muse_spark bf16_splitk_gemm_partials (1114-16) · harness defect: EAT has no task-level exit. The supervisor declared
  convergence at iter 14 (1.18x; the remaining 3.9% lever needs a caller-contract change), then ran 34 no-op
  "hold_keeper" iterations at ~$4 each plus a GPU official eval. Fix: the operator stops a task via a DISABLED
  file; future harness should let the supervisor end a task explicitly (e.g. an `## Exit` block in TEAM_CONFIG).
- 4 Muse kernels (1114-16) · teams declared false roofs measured with their own probes (6.0/5.8 TB/s, "13.7 µs
  fixed") → a ceiling challenger with a raw HIP stream / vendor kernels refuted each (real 7.1-7.6 TB/s) → it
  opened the hand-written HIP rewrites (compact 1.05→1.12 in one iteration; splitk ×1.276 in a deployment graph).
- gqa iter 3→4 · every candidate perturbed one design → an architect barred from editing it + an exact numerics
  emulator → 1.074 → 1.0937 → 1.1475 · $255 + $223.
- gqa iter 3 · engineers quit at 56-75 of 360-420 min ("tie") under geak-engineer v1's stop rules → deliverables
  (bound, ≥3 designs, ≥2 prototypes) kept them working 5-6 h. v2 engineer has no stop rules.
- norm iter 3, gqa iter 4 · cutting work by regime / size band beat cutting by technique: largest single steps.
- compact iter 5, splitk iter 5 · gains near noise and flipping verdicts → instrument investigators found in-process
  A/B state carry-over (~2x overstated), GPU bias 0.8-1.8%, host-socket offset → cross-over pairs, pinned probes.
- splitk iters 3/8/9, norm iter 5 · supervisor's own ship rules blocked audited gains → deployment-like probe
  decides, the eval must only show no loss; speed-neutral fixes ship on non-inferiority.
- all tasks · cheap adversarial verifier caught silent bugs each time (races, m0 hazard, concurrent-call
  corruption, 25/31 deployment threats) → keep a ship/audit role every iteration.
- gqa iter 6 · the server's defer_reduce=True path was never timed by the UT; scored gains did not reach serving
  → map the deployed path in iteration 1 (eval defect: the UT boundary).
