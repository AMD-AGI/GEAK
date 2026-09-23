---
key: dense bf16 Linear shapes at decode on a 40-CU RDNA3.5 iGPU (gfx1151, shared LPDDR memory), vLLM on PyTorch's own dispatch, where two vendor BLAS libraries are both present
type: lever
confidence: ★★
effect: +10.68% e2e on the orchestrator's own revalidate arm and +11.31% median on an independent fresh-server 3-repeat (per-round range +10.40%..+11.45%, arms non-overlapping against a 0.4% box noise floor); task accuracy flat (GSM8K 0.94 -> 0.94). Same-family corroboration: per-shape mixing +14.3% on a warm-server ruler versus -8.2% for forcing one library globally, a 22-point spread across the same shapes.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-21
name: let-a-per-shape-table-pick-the-blas-library-instead-of-picki-dense-gemm-gfx1151-decode
description: gfx1151/RDNA vLLM dense bf16 decode: a per-shape TunableOp BLAS table is a real e2e lever (~+11%); forcing one library globally is a large regression
keywords: ['dense-gemm', 'bf16', 'gfx1151', 'rdna', 'strix-halo', 'vllm', 'tunableop', 'per-shape-tuning', 'backend-routing', 'rocblas', 'hipblaslt', 'decode', 'vendor-library', 'cold-vs-hot', 'measurement-discipline', 'tuning-artifact']
kernels: []
platforms: ['gfx1151']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
origin_kernels: []
cost: L1
verified_on: 2026-09-21
roofline: The heads this moves are not at the memory wall; the largest untouched masses on this model are, so the remaining headroom after this lever is byte reduction rather than a faster kernel. Scoring any of it needs the part's own last-level-cache roof as a third roofline, not just the DRAM one.
levers: ['host.backend-routing', 'host.host-tuning']
---
# Let a per-shape table pick the BLAS library, instead of picking one globally
- lever: On this part PyTorch can reach two vendor BLAS libraries and neither wins everywhere: the best choice flips shape by shape across the Linear heads of one model. Letting a per-shape table race them and record the winner per (M,N,K) is worth roughly +11% e2e at decode, while pinning the whole model to whichever library looks best in isolation measures -8.2%. The lever is the per-shape table itself, not any one library. Note the framework default here is the older of the two, so a plain baseline says nothing about the other one.
- apply: Enable the framework's own per-shape tuning pass, tune on the real decode shapes, then deploy the emitted table read-only via the tuning-enabled/tuning-off/filename env triple. It is a config lever with no source edit and no kernel authoring. Because the table is keyed by shape, it transfers only to the model and sequence geometry it was tuned on.
- stack: This composes with, and is separate from, rerouting the same dense-GEMM seam onto an authored/library Triton path: they act at different layers (which library PyTorch dispatches to, versus which op the framework calls). Their sum has not been measured here, so treat stacking as unverified rather than additive.
- verify: Score on a FRESH server, not a warm one. A warm-server A/B overstated this lever by about 25-30% on this part (14.3% warm reconciling to about 10.9% fresh), which is large enough to change a funding decision on its own. Confirm the deployed table path is the one the launch recipe actually points at, and re-check task accuracy, since a different library changes accumulation order.
- pitfall: the tuning pass emits TWO tables with the same filename shape: a raw intermediate recorded during the hot sweep, and the cold/rotating-tuned one under the install directory that the pipeline actually deploys -> deploying the intermediate measured +1.65% and deploying the installed one measured +11.31%, a 7x difference from a directory name -> resolve which artifact the launch recipe references before benchmarking, because both are non-empty, both parse, and the wrong one fails as a small positive rather than as an error.
hot op-level timing inverts the cold verdict on these shapes -> with one weight resident a naive candidate measures 1.7-2.6x faster than the vendor, and 0.66-0.90x once weights rotate past the last-level cache -> real decode streams far more than the cache per step, so rank on the cold measurement.
- caution: also verify, rather than assume, that this survives a three-arm greedy-token consistency gate: changing the BLAS library changes accumulation order, and a sibling stride-padding lever on this same part passed its throughput gate at a similar magnitude and then diverged on 2 of 8 controls. Also verify the emitted table path is portable before treating the recipe as deliverable - it is written as a session-scoped absolute path by default. Evidence here is one model, one workload point, one box.
- source: run e2e_gfx1151_tunableop_20260921, 2026-09-21 - orchestrator revalidate arm plus an independent operator-run fresh-server 3-repeat A/B on the same recipe; accuracy smoke re-run on both arms; supersedes four earlier conclusions that were drawn from the raw intermediate tuning table
