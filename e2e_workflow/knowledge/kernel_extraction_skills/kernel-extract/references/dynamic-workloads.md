# AgentX workload measurement

Use the sections matching the current run. The [core skill](../SKILL.md) owns
activation, ranking artifacts and the weighting contract.

## Distribution and phase sampling

Measure active concurrency and server batches independently of requested
concurrency. Arrival gaps, think-time and scheduling change occupancy. Preserve
the configured replay duration, arrival schedule and session policy; inspect
occupancy for drain-down rather than silently recycling a finite session pool.
Establish how many iterations each log row represents, including logging
intervals and gaps; cross-check expanded counts against completed work.

When caching exists, distinguish nominal input length, cache hits and newly
executed tokens. Sample the observed cache regime. For attention, distinguish
query length, per-request context, aggregate occupancy and rank-local KV extent.
A changed cache capacity or hit distribution can invalidate old case weights.

When speculation exists, separate request batch, proposed/verified/accepted
tokens and padded kernel M. Derive their mapping from actual launches. Discover
cycle boundaries and omit partial endpoints; a named marker is valid only if
its cadence matches the current algorithm. Attribute subphases separately when
shapes or dispatch differ, including conditional layers and reused results.

Use the original joint distribution to select common cases and material tails,
including non-power-of-two values, chunk tails and served dispatch boundaries.
Record both call mass and baseline-time mass. Configured caps and graph capture
lists constrain possible shapes; runtime evidence establishes their mass and
served eager fallbacks. Keep unobserved envelope probes at zero observed weight.

## Turn observed shapes into trace sampling points

Start from the completed baseline's logs and measured-leg exports. Reuse matching
windows already captured from that replay. If kernel costs or dispatch boundaries
are unknown, take a short targeted capture, then refine the points using measured
time mass. Do not run a new full replay or generic full-model profile. Derive
probe request controls from observed work and validate the resulting launches:

| Phase | Choose from configs/logs | Set and verify for each probe |
|---|---|---|
| Prefill | Joint uncached-token/context distribution, effective chunk budget, common chunks and tails | Prepare the observed cached prefix; choose total tokenized ISL to produce the target uncached suffix/context; arm before the intended extend; verify actual chunk/query/KV lengths |
| Decode | Joint active-batch/per-request-context distribution, speculation and graph padding | Prepare target context and cache state; choose request concurrency to reach the observed server batch; choose OSL long enough to pass admission and capture complete steady-state cycles; arm after decode begins and verify executed M/KV |

Record phase, request ISL/OSL, concurrency, cache preparation, server flags,
capture trigger/length, expected and observed shapes, selection reason, trace
path and covered mass in `kernel_workload.json`'s sampling plan. ISL is not
necessarily uncached work; OSL controls how long decode remains active, not its
kernel M. Reuse relevant quantization, routing and speculation behavior rather
than inferring it from request lengths. State approximations and missing coverage.

Targeted probes measure identity/cost; their synthetic request counts do not
replace original replay weights. Keep their controls local to the probe and
leave canonical replay config, workload/state and fixed ISL/OSL paths unchanged.

## Trace integrity

Arm short captures at observed phase boundaries: starting before admission can
produce a prefill trace labeled decode. Check actual markers and launches;
fixed delays from another run do not establish steady state. Resolve nested
annotations by phase semantics and containment, not timestamp ordering alone.
Mixed windows require attribution by complete phase instances.

Compare expected forwards and per-family events across ranks only when the
execution should be symmetric. Use the modal clean pattern rather than the
largest count, which may include compilation/autotuning. Rank agreement alone
cannot detect a phase omitted everywhere; also check workload progress and
active-graph call counts. Validate phase-exclusive markers on both matching
and nonmatching captures before using them to reject a window.

Busy/span is diagnostic evidence, not a universal integrity threshold. Peer
wait can inflate collective duration; host stalls can lower activity despite
complete events. Discover collective families from all ranked launches and
inspect significant unclassified work. A rank with the smallest summed time
is not automatically an unbiased compute reference.

Use bounded eager diagnostics if production graphs hide events, then validate
final timing in deployment graph/compile mode. On affected backends, disable
stage profiling only when it demonstrably omits work and explicitly close the
capture. Before restarting an owned failed server, verify worker exit and
allocation release; stop health polling when the process exits. Preserve raw
captures and point-specific logs, placing retries in separate directories.

## Cost attribution

Complete observed counts with per-case latency need no parametric model. For
sampled costs, fit verified logical families or separate dispatch branches;
retain branch identities when assigning editable shares. Do not treat a kernel
name disappearing at a dispatch boundary as zero operation cost.

Choose axes and fixed-cost terms supported by measurements. Examples include
`a + b*batch + c*batch*context` or `a + b*new_tokens + c*new_tokens*context`, but
sparse/windowed attention or other architectures may have different dependencies.
Inspect absolute and relative residuals, validate additional points and report
extrapolation. Fit objectives must match the deployment distribution rather
than letting a few large samples dominate. Normalize per call/phase instance
before comparing captures of different lengths.

For gap-based throughput logs, inferred phase occupancy requires validation
against spans or independent progress; another backend may define its counters
differently. Keep rank/stream scope consistent and preserve collective/wait and
unattributed mass. Ledger closure checks accounting, not representativeness.

Cross-check predictions with independent benchmark counters or device telemetry
where available over the same interval/ranks. Document units, sampling error
and overlap: integrated `gfx_activity` is activity time, not summed device
kernel time or non-collective compute. Missing telemetry is a stated validation
limit. Production duty and critical-path assumptions are needed for wall-time
projections; eager profiler span and a scalar utilization factor cannot by
themselves establish an e2e gain.
