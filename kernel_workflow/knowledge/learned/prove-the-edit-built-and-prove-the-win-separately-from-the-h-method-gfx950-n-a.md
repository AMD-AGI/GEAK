---
key: confirming that a C++ edit actually took effect and actually won, inside a JIT-built vendor stack whose run directory is gitignored, gfx950
type: method
confidence: ★★
effect: 4 of 4 winning rounds in this run were reported as not-improved by the automatic flag, while hand-diff plus re-timing showed 1.01x to 1.27x holding on all three batch cases (2/32/64)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a
description: In a JIT-built frozen C++ vendor stack, header edits may not rebuild, gitignored run dirs hide the diff, and the auto improvement flag false-negatived wins
keywords: ['method', 'jit-rebuild', 'composable-kernel', 'verification', 'false-negative', 'frozen-baseline', 'moe', 'grouped-gemm']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-07-30
origin_kernels: ['moe_gemm_fp8_blockscale']
---
# Prove the edit built and prove the win separately from the harness flag
- lever: In a JIT-compiled vendor stack, three independent things can each make a real win invisible — the edit not rebuilding, the diff not being visible to git, and the harness verdict flag reading false — so confirm build, patch and timing by three separate means before declaring an axis dead.
- apply: Move aside both the cached shared object and the JIT build directory before re-timing; capture the patch with a plain unified diff against the canonical tree rather than git; read the per-case timings directly instead of the aggregated improvement flag.
- verify: Grep the disassembly or the emitted instance name for the property you changed; if it is unchanged, the run you just timed is the old binary and its 1.00x means nothing.
- pitfall: An edited header produced identical timings → the JIT keyed the cache on the built artifact, not the header mtime → moving both the artifact and its build directory aside restored the rebuild; separately, the diff came back empty → the run tree is gitignored → diff against the canonical checkout instead.
- caution: Also verify a not-improved verdict by hand when the expected delta is around a percent: the aggregate flag here was a false negative on every winning round in this run, so a cheap real win can be discarded as noise.
- source: run moe_gemm_fp8_blockscale-ch16h, 2026-07-30 — 16h time-budget campaign, durable gotchas recorded in the run state insights
