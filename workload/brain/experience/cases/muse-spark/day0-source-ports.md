# Source ports and slow-request coverage

Origin: `muse-spark`. One native 20-minute MI355X C8 screen per variant on the
same node. Each serving image retained the original SGLang layers, topology,
draft, external concurrency and synthetic-acceptance workload.

The original service logged one or two active requests in 880/1,134 decode samples.
A joint router/pooler backport for small batches improved output by 2.32% and
full-response P90 by 2.34%. A larger target-only backport served verify at one or
two requests. It raised output by 27.77% while full-window P90 fell by 1.85%.
The original draft and wider-batch target calls still used the ordinary forward.

The larger port finished 501 requests versus 386. All shared outputs had identical
lengths and their median paired token interval fell by 30.12%. Among its slowest
decile, 44/51 streaming intervals contained a complete second with a decode sample.
All 44 overlapped an active batch above two. Although 94.16% of all decode samples
were at or below four, 27 of those 44 intervals had no sample at or below four.
Concurrent requests can observe the same sample. Admission at fixed code and faster
ordinary execution at wider batches are competing interventions.

A prior 32-row diagnostic exceeded the implementation's numerical bounds on 112 of
1,488 layer checks. The 28 target/common compute sources match this port byte for
byte. The earlier run used a different surrounding runtime. An advertised width
still needs numerical qualification on the proposed stack. [Native evidence and source comparison](../../raw/agentx-muse-day0-ports-20261126.json).

An older six-kernel screen on the same node and protocol, on another day, shared
447 requests with matching output lengths. Relative to that screen, median token
interval fell 32.22% for the 38 requests observed only at B1/B2 in both windows,
and rose 22.67% for the 85 observed only above B2. For B5–B8, the older ordinary
attention source reduced weighted operator time by factors of 2.66 and 2.68 in two
cold tests. This prioritizes a paired test of narrow fusion plus that fallback
at fixed admission. The combination's serving effect is open. [Paired requests and cold receipts](../../raw/agentx-muse-day0-ports-20261126.json).
