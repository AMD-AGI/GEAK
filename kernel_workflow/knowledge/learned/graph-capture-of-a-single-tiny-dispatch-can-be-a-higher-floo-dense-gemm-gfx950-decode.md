---
key: graph capture as a launch-floor remedy for a single-dispatch bf16 decode GEMV on gfx950, timed through a Python harness
type: anti-pattern
confidence: ★★
effect: graph replay ~2.0x SLOWER than eager on every decode case (tokens=2 and 4), robust across 3 signatures x 3 correctness repeats; the eager-fallback wrapper alone still regressed to 0.88-0.96x per case, so nothing shipped
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode
description: Capturing one tiny decode dispatch into a HIP graph replays ~2x slower than eager; a Python signature-cache wrapper on the timed path also net-regresses.
keywords: ['hip-graph', 'launch-overhead', 'dispatch-floor', 'decode', 'gemv', 'anti-pattern', 'wrapper-overhead']
kernels: ['wvSplitK_hf_sml_']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: archived
---
# Graph capture of a single tiny dispatch can be a higher floor than the dispatch it replaces
- lever: graph capture is worth trying against a launch floor, but A/B the replay against eager in the same session before wiring it into the shipped wrapper
- apply: capture the single dispatch, keep an eager fallback in the same wrapper, and time both paths inside the harness call rather than around it
- verify: check the captured output still passes oracle parity - it did here, which is what makes a slow replay a timing result and not a capture bug; also A/B the wrapper itself against the raw unwrapped path
- pitfall: the signature-cache wrapper (pointer/shape/dtype tuple + closure dispatch) put Python on the timed path and the harness times everything between the events -> a 4-12% per-case wall regression even on the eager path -> baseline the raw unwrapped call
- caution: also verify how many dispatches a graph would amortize over: at one dispatch the graph-launch path was itself above the direct dispatch cost on this ROCm build
- source: run wvSplitK-ch16h, 16h per-kernel time-budget campaign, 2026-08-12
