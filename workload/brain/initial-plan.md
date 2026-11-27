# Choosing the first experiments

Client concurrency suggests where to look. Use Profile or existing evidence to locate
the cost in prefill, decode or verify, including active requests **per worker**, token
rows per step and the requests driving each objective. If those observations are
missing, start with the configured route and let its first report refine the plan.

| Observation on the serving path | Candidate to compare |
| --- | --- |
| Narrow steps repeatedly pay launch, synchronization or intermediate-memory costs. | Test a persistent whole-layer implementation (monokernel) when shapes, layout and numerics qualify. Compare with ordinary execution at fixed admission. Check target and draft dispatch and KV cost. If the phases switch independently, isolate one when a combined result leaves the choice unclear. |
| Wider batches expose substantial operator work. | Rank attention/KV reads, dense or expert GEMMs, routing and collectives by exposed cost at the observed shapes and ranks. Qualify independent UTs and search them in parallel on available disjoint devices. |
| Queueing, cache pressure or a lagging rank limits progress. | Locate the wait or recomputation. For a shared pool, sum the participating ranks' reservations against its capacity. Compare one scheduling, cache or partition change and measure first-token wait alongside the declared serving objectives. |

Across widths, look for costly intermediates, collective boundaries or repeated work
on shared inputs. Check which calls use the same data and how long it stays valid.
Compare reuse or regional fusion with the best compatible separate operators.

If Profile comes first, say what it would dispatch: name the independent tasks that
could run concurrently, which devices each needs, and which unfused candidate a
fusion would have to beat. A shared multi-device comparison gets its own window.
For mixed traffic, re-rank the paths still serving slow requests and the phase
limiting throughput after each result.
