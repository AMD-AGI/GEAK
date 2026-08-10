---
key: moe grouped gemm · gfx950 · mixed
type: method
confidence: ★★
effect: Three separate multi-config instance sweeps returned nothing across three rounds while paying one build per config; registering candidates additively in one module made a 7-config sweep fit inside a single round, and that sweep found the run's largest lever (+5.4 / +10.8 / +11.4% on the three scored batch sizes; whole-stack director-verified 1.22-1.29x per case). Incremental rebuild ~1.5 min instead of ~12.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm7.2.3 / torch2.11.0 / hip-ck-codegen
last_seen: 2026-08-08
---
# One build, N selectable instances: turn a config sweep into an env switch
- lever: For a codegen or templated kernel family, candidate instances are additive inside ONE compiled module: register N of them and pick one at runtime from an env var, so an N-config sweep costs one build and every comparison becomes a paired A/B inside a single binary.
- apply: Add the candidates to the generated dispatch table plus a getenv selector in the generated header, defaulting to the shipped config so the patch is runtime-inert. Where the knob is compile-time, prebuild both arms as two shared objects and swap them between runs with a checksum on each swap.
- verify: Run the arms interleaved A/B/A/B in one session and take medians; confirm the intended arm actually engaged from a trace fingerprint (LDS bytes per workgroup, VGPR count) or the module checksum rather than from a log line; re-measure the incoming best in the same session as a drift check.
- caution: Also verify the JIT loader is not serving a stale object - a cached shared object can be loaded with no source-freshness check, which certifies unpatched code as a win; and also verify that the two numbers being compared came from the same harness and the same session, since cross-harness offset ran 1-4% and session drift 1-3% here, larger than several claimed wins.
- source: run kernel_20_geak_0808_4h 2026-08-08
