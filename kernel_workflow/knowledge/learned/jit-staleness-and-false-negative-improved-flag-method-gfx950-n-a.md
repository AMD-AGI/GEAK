---
key: measuring edits to a JIT-built vendor C++ extension under a driver script that reports its own improvement verdict
type: instrument
confidence: ★★
effect: 4 of 24 passes reported no improvement on rounds that had in fact won, all 4 recovered by diff-confirming and re-timing per-case; header-only edits reproduced the pre-edit timing exactly (ratio 1.000) until both the cached object and the JIT build tree were moved aside
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 24
toolchain: unknown
last_seen: 2026-08-11
name: jit-staleness-and-false-negative-improved-flag-method-gfx950-n-a
description: A JIT-compiled vendor extension can silently re-run the old binary after a header edit, and the harness improvement flag can be false on winning rounds.
keywords: ['instrument', 'jit', 'build-staleness', 'verification', 'false-negative', 'measurement-hygiene', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
---
# jit-staleness-and-false-negative-improved-flag
- lever: before believing any A/B on a JIT-built extension, invalidate BOTH the cached shared object and the JIT build directory — header mtime alone did not trigger a rebuild
- apply: move the built object and the jit build tree aside, rebuild, and confirm the change is present in the rebuilt artifact; capture the applied change with a plain diff against the canonical tree when the run directory is ignored by version control and the usual diff comes back empty
- verify: read the per-case ratios yourself rather than the driver's boolean verdict, and grep the built source for the edit — a ratio of exactly 1.000 across every case is the fingerprint of a stale binary, not of a neutral change
- pitfall: the driver's improvement flag read false while per-case re-timing showed a real win -> the flag was computed from a stale or differently-scoped comparison -> hand-confirm by diff plus re-timing before discarding a direction
- caution: seen on one campaign's toolchain; also verify whether your own build system keys on content hash rather than mtime, in which case only the verdict half of this applies
- source: run moe-blockscale-16h campaign, 2026-07-29..2026-08-11, durable gotchas recorded in run state
