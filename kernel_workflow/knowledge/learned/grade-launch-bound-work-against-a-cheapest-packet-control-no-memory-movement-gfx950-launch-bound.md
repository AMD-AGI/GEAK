---
key: memory movement · gfx950 · launch-bound
type: method
confidence: ★★
effect: Called the graded outcome 3/3 times: an empty kernel measured 18.8 us vs 16.1 us for the real one before the host path was collapsed and 6.50-6.55 vs 6.51-6.56 us after, after which a 21% cut in GPU time and a 2.06 us cut in host enqueue each moved the metric by exactly 0.000 ms. Converged case spread ~1.5% across a 32x range in batch size, and the derived ceiling (2.463x) sat 2.8% above the shipped 2.39x.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm7.2 / triton3.6.0 / torch2.11
last_seen: 2026-08-08
---
# Grade launch-bound work against a cheapest-packet control, not the baseline
- lever: On a harness that records a timing-event pair around every iteration, the reported number is a difference of two GPU timestamps, so it behaves like max(host_enqueue, gpu_time) plus a fixed floor rather than their sum. Measure that floor early with an empty kernel and with the cheapest legal packet, and read baseline/floor as the ceiling for any single-dispatch implementation.
- apply: Time three controls through the same launcher the candidate uses: an empty kernel with the same arg count, an already-satisfied stream-wait (a packet that touches no memory and launches no wave), and a deep-vs-drained queue variant. Their spread separates the immutable event/barrier pair from the marginal cost of one work packet, measured here at ~4.6 us and ~1.9-2.15 us.
- verify: Score a candidate by how far it beats the cheapest-packet control rather than the baseline; when the real kernel and an empty one are indistinguishable and case spread has collapsed across a wide range in batch size, the remaining terms are under the floor and further host or body tuning will report 0.000 ms.
- caution: Also verify off-harness before writing such work off: the floor belongs to the timing rig, not the device, so a lever reading 0.000 ms in isolation can still be real end to end - about 9 us per call of removed host time here had no event pair to hide behind in serving.
- source: run kernel_20_geak_0808_4h 2026-08-08
