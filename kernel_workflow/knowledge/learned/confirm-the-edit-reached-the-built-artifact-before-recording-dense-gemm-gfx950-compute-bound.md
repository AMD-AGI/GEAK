---
name: confirm-the-edit-reached-the-built-artifact-before-recording-dense-gemm-gfx950-compute-bound
description: Grep the built artifact for a patch marker before recording a negative: 6+ of 44 passes on a dense GEMM scored code that never reached the build
keywords: [measurement-method, control-experiment, isa-check, code-object, config-sweep, dense-gemm, compute-bound]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: compute-bound
key: telling a dead patch apart from a dead axis on a Triton fp16 dense GEMM, gfx950, over a case set spanning ~30x in duration
lifecycle: active
type: instrument
confidence: ★★
effect: Across 44 passes the harness reported no-improvement on patches that were in fact applied at least 6 times, so a negative was only trusted after grepping the built source for patch markers. A separate direction spent a round on scheduler flags that do not exist in the deployed Triton 3.6.0: codegen came out byte-identical and the direction was inert, recorded as 1.00x and marked dead for all future rounds. The small-M case is warmup/DVFS-sensitive at ~1/30 the duration of the largest case, which self-warms and is immune.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 7
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
last_seen: 2026-08-11
---
# Confirm the edit reached the built artifact before recording a negative result
- lever: Treat 'no improvement' as two distinct outcomes - the change ran and did not pay, or the change never ran - and separate them cheaply before closing an axis.
- apply: Grep the compiled/staged source for a marker you inserted with the patch; for a compiler or environment knob, diff the generated assembly against the unpatched build.
- verify: Byte-identical codegen means the knob is not wired in this toolchain and the round measured nothing; retire that direction as dead for all future rounds rather than filing its 1.00x as a measured negative against the underlying idea.
- pitfall: repeated 1.00x verdicts on a live axis -> patches that never reached the built artifact, plus flags absent from the deployed compiler -> marker grep and an assembly diff before the verdict is written.
- caution: Also verify which case the axis verdict is read off - on a case set spanning ~30x in duration, read it off the longest case and treat the shortest one's number as warmup-sensitive.
- source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
