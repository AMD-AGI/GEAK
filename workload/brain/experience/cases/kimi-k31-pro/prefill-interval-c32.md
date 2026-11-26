# Prefill interval at C32

Origin: `kimi-k31-pro`. A held-out planner proposed testing prefill cadence.
Two consecutive 20-minute SGLang screens used one node with eight MI355X,
TP8/EP1, DCP8, admission 32 and six verify tokens. Both used the same original
image, workload, client and effective host pools.

Shortening the prefill interval from 16 to 8 raised total output throughput
from 836.70 to 850.98 tok/s and reduced P90 TTFT from 7.51 to 3.60 seconds.
Full-response P90 interactivity fell from 34.75 to 32.42 tok/s/user. Sampled
decodes with queued work fell from 52.55% to 30.35%. Keep the reference for a
joint-gain objective. A workload prioritizing first-token latency could choose
differently. All 1,251 matched requests had identical output lengths.

The published host-pool configuration failed during local startup. Its eight
ranks requested about 1,526 GB against a shared 1,477 GB GTT limit. Reducing
only the recurrent-state host ratio let both arms allocate 1,227 GB. The pair
measures the interval under this local capacity. It used synthetic acceptance
and had one window per arm. [Native receipts](../../raw/agentx-kimi-c32-initial-plan-20261126.json).
