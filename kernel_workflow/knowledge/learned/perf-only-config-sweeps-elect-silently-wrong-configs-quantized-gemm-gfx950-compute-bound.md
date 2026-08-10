---
key: quantized gemm · gfx950 · compute-bound
type: method
confidence: ★★
effect: a correctness gate on the sweep caught 3 tile configs and 3 launch-knob configs that ran fast and computed wrong output, among them the single fastest config measured in the entire run (+2.6% over the one shipped); a k tile wider than the quantization group passed both smaller cases and failed only on the largest (max_rel 4.37 vs 0.00), while k tiles narrower than the group failed on every case (max_rel 1.18 and 1.69)
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# Perf-only config sweeps elect silently wrong configs
- lever: In a group/block-scaled quantized GEMM the k tile is coupled to the quantization group through the scale index, so BLOCK_K is not a free tuning axis; add a per-config golden comparison to whatever sweep picks the config, because the configs that break that coupling are the fastest rows the sweep will see and time alone ranks them first.
- apply: Have the sweep compute max_rel against the golden for every row before it ranks on time, run it on the largest shape in the set, and extend the gate to every knob the sweep touches — a warp count and a loop-unroll setting each produced fast, silently wrong binaries here, not only the tile.
- verify: Check the scale-index arithmetic maps exactly one scale per quantization group per k step for the elected config (a k tile wider than the group applies one scale to two groups; a narrower one splits one group across two accumulator updates), and confirm with max_rel — cosine similarity read 1.000000 on candidates with millions of violating elements.
- caution: Also verify on the largest shape available rather than a quick small one, since the widest-k-tile violation was clean on both smaller cases and only the largest exposed it; and re-confirm the elected row on the official runner, because an in-process sweep reusing the same tensors over-reports cache-sensitive knobs.
- source: run kernel_20_geak_0808_4h 2026-08-08
