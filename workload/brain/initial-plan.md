# Choosing the first experiments

Client concurrency suggests where to look. Use Profile or existing evidence to locate
the cost in prefill, decode or verify, including active requests **per worker**, token
rows per step and the requests driving each objective. If those observations are
missing, start with the configured route and let its first report refine the plan.

| Observation on the serving path | Candidate to compare |
| --- | --- |
| Narrow steps repeatedly pay launch, synchronization or intermediate-memory costs. | Test a persistent whole-layer implementation (monokernel) when the served shapes, layout and numerics qualify. Compare with ordinary execution at fixed admission. Check target and draft coverage and the KV space the route consumes. |
| Wider batches expose substantial operator work. | Rank attention/KV reads, dense or expert GEMMs, routing and collectives by exposed cost at the observed shapes and ranks. Qualify independent UTs and search them in parallel on available disjoint devices. Where adjacent operations exchange an expensive intermediate or wait on a collective, compare regional fusion against the best compatible separate operators. |
| Queueing, cache eviction/recomputation or a lagging rank limits progress. | Profile the limiting stage. Compare a scoped change to scheduling, KV allocation or work partition before assigning more kernel time. Check first-token wait alongside output throughput and interactivity. |

If Profile comes first, say what it would dispatch: name the independent tasks that
could run concurrently, which devices each needs, and which unfused candidate a
fusion would have to beat. A shared multi-device comparison gets its own window.
For mixed traffic, re-rank the paths still serving slow requests and the phase
limiting throughput after each result.
