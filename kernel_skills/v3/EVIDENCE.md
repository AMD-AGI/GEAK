# v3: what changed from v2 and why

Source: the 1116-17 skill-v2 test on 8 Muse-Spark kernels (`v2test_1116`). Each lead had one GPU and
started from the baseline, isolated from earlier results. The input was their `## Harness changes` /
`## Harness notes` (96 logged changes) plus an operator roofline analysis of the two runs that ended early.
v2's evidence (the EAT run) is in `../v2/EVIDENCE.md` and still applies.

| v3 change | evidence |
|---|---|
| **geak-roofline role + end rule: gap closed ≥ 80% of a design-independent cap, or an independent ceiling role explains the rest** | Two v2 leads ended "converged" under v2's end rule. fused_norm stopped at 1.171× (score cap ~1.35; another run reached 1.29): it treated a 1.7 µs decode kernel as fixed and used its own HIP copy as the prefill roof. rowmajor_gemm stopped at 1.060× against its own measured floor cap of 1.245× (29% of the gap closed): it accepted a bound built from its own design's costs. After ending they idled through 200+ resumes. |
| `bin/hbm_stream.py`: best of several streams as the roof | on MI355X `torch.sum` reads 3.7 TB/s and a non-temporal HIP stream 7.08 TB/s. A weak probe makes a false roof |
| `bin/roofline.py`: cap / score / gap closed, rows sorted by remaining weighted gap | the gap decides both when to end and where to cut work |
| Round 0 runs setup, roofline, budget and deployment in parallel | 6 of 8 leads overlapped them anyway: investigators only need `orig` and the harness |
| Scale to resources; slow-eval mode with a validated fast probe | every lead had to adapt v2's ≥ 2-GPU assumptions. One official run took 20–40 min on 4 tasks, and every lead built its own fast probe. Validation by candidate-vs-candidate delta: gqa's probe agreed on baseline times but misranked candidates across bound classes. Process state (HW queue count) moved splitk by 0.02 |
| COMMON.md: detached GPU jobs, progress.md, inbox.md, heredoc reports | node / container restarts killed every role and its GPU jobs (one lost ~50 min of work). 4 runs had subagents unable to Write report files. gqa's lead relayed ~10 messages by hand |
| Skipped gates: benchmark lists them; ship re-establishes their intent | scoring skipped the mutation gate. One run found that no role was checking it; rowmajor's ship wrote equivalent mutations for the new kernels |
| Ship attacks: coverage of unreached branches, skewed-input timing, determinism, cold-start concurrency, spin-wait stress | muse_moe: a 2.4× decode regression on hot-expert routing lived for ~12 h (correctness-only attacks); determinism caught a double write. gqa: dispatch at n ≥ 9 never reached by UT correctness. muse_extend: cold-start concurrent first calls |
| Deployment investigator outputs a runnable UT-blind-spot threat suite | compact: the UT checks half the local experts. gqa / muse_extend: graph-padded tails and mixed lengths unchecked |
| Engineer step 0: measure the oracle's tolerance; design to the contract | fused_norm: c001 held itself to bit-exactness; relaxing to the contract gained another 4.5% |
| Compiler facts in the budget (attributes, waves / EU, co-issue) | muse_extend: its two largest wins were compiler-attribute facts the ablation table missed |
| fresh-arch also when < half the time is left and the gap ≥ 5% | rowmajor, fused_norm: the 2-round plateau rule fires too late in a short run |
| Option patches on the shipped node; ≥ 3% snapshots; probe code kept out of patches; editable-only diffs; mkws excludes harness by-products | splitk: the run's largest win (c003) was an option patch. gqa: c001's probe branch leaked into a ship. rowmajor: the UT's `_cand_overlay/` and `run/ut_*` got into a diff |
| Deployment-vs-score policy in the ship rule | gqa: c008 / c013 bought 1.2–1.5× on mixed-length production batches for −1.2..−2% on the score, and v2 had no rule |
| ENDED marker + reopen rule | after ENDED, generic "continue" resumes reopened runs that had nothing new (splitk 3×, ~1 M tokens) |
