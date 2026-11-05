# tl-only leg -- GLM-5.2 0927, baseline vs candidate

engine: TraceLens PerfModel (predictor) | arch `MI355X.bw6857_bf1449.json` md5 `c045ecc958e17ca6062de4970ddd3316` | peak_bw 6857 GB/s
source: These are benchmark derived peak flops and bw

- baseline run 2026-10-12 04:00:37 -- contention: node was quiet enough
- candidate run 2026-10-12 04:01:18 -- contention: node was quiet enough

## Not covered by this leg

- **indexer_preamble** (7 case(s)) -- five-kernel chain (Layernorm2dFwd + rope cache store + 2x hadamard + act_quant + indexer_k_quant_and_cache) at m<=42 rows. TraceLens has no model for the chain, and at this size the seam is dominated by the ~2.3 us per-node in-graph dispatch floor rather than by its bytes -- a byte roofline would be a confident number about the wrong quantity. Use the rocprof leg (per-kernel) for this one.

## Per case

`roof_us` is the same for both arms on purpose (it is a function of the shape). `pct` is roof/measured; higher is closer to the hardware limit.

| seam | case | w | bound | roof us | base us | cand us | speedup | base % | cand % |
|---|---|---|---|---|---|---|---|---|---|
| add_rmsnorm_quant | dec_bs2_m12_nores ⬓ | 0.022 | memory | 0.04 | 4.3 | 4.0 | **1.088x** | 1.0% | 1.1% |
| add_rmsnorm_quant | dec_bs2_m12_res ⬓ | 0.230 | memory | 0.09 | 8.9 | 8.5 | **1.048x** | 1.0% | 1.0% |
| add_rmsnorm_quant | dec_bs3_m18_nores ⬓ | 0.060 | memory | 0.07 | 4.3 | 4.0 | **1.082x** | 1.5% | 1.7% |
| add_rmsnorm_quant | dec_bs3_m18_res ⬓ | 0.622 | memory | 0.13 | 9.0 | 8.5 | **1.060x** | 1.5% | 1.5% |
| add_rmsnorm_quant | pre_m10129_nores | 0.031 | memory | 36.30 | 44.2 | 40.8 | **1.084x** | 82.1% | 89.0% |
| add_rmsnorm_quant | pre_m10129_res | 0.031 | memory | 72.61 | 293.4 | 290.2 | **1.011x** | 24.7% | 25.0% |
| add_rmsnorm_quant | pre_m1137_nores | 0.002 | memory | 4.08 | 8.0 | 8.1 | **0.990x** | 50.9% | 50.3% |
| add_rmsnorm_quant | pre_m1137_res | 0.002 | memory | 8.15 | 30.9 | 31.5 | **0.983x** | 26.3% | 25.9% |
| dsa_indexer_decode | dec_bs2_ctx318k | 0.167 | memory | 14.49 | 83.3 | 19.6 | **4.257x** | 17.4% | 74.1% |
| dsa_indexer_decode | dec_bs2_ctx418k | 0.166 | memory | 19.05 | 110.5 | 23.9 | **4.629x** | 17.2% | 79.8% |
| dsa_indexer_decode | dec_bs3_ctx104k | 0.167 | memory | 7.13 | 46.7 | 11.6 | **4.040x** | 15.2% | 61.6% |
| dsa_indexer_decode | dec_bs3_ctx173k | 0.167 | memory | 11.87 | 76.4 | 15.9 | **4.797x** | 15.5% | 74.6% |
| dsa_indexer_decode | dec_bs3_ctx217k | 0.167 | memory | 14.84 | 92.9 | 19.7 | **4.713x** | 16.0% | 75.2% |
| dsa_indexer_decode | dec_bs3_ctx261k | 0.167 | memory | 17.88 | 112.6 | 22.0 | **5.115x** | 15.9% | 81.3% |
| dsa_indexer_prefill | pre_m10129_kv40k | 0.167 | compute | 743.04 | 1575.4 | 884.2 | **1.782x** | 47.2% | 84.0% |
| dsa_indexer_prefill | pre_m1137_kv470k | 0.167 | compute | 1378.58 | 4282.1 | 1640.0 | **2.611x** | 32.2% | 84.1% |
| dsa_indexer_prefill | pre_m1591_kv329k | 0.167 | compute | 1346.28 | 4378.2 | 1576.4 | **2.777x** | 30.7% | 85.4% |
| dsa_indexer_prefill | pre_m2457_kv201k | 0.167 | compute | 1243.05 | 3837.9 | 1458.6 | **2.631x** | 32.4% | 85.2% |
| dsa_indexer_prefill | pre_m3480_kv113k | 0.167 | compute | 983.74 | 2783.6 | 1165.2 | **2.389x** | 35.3% | 84.4% |
| dsa_indexer_prefill | pre_m626_kv857k | 0.166 | compute | 1385.13 | 6688.3 | 1650.4 | **4.052x** | 20.7% | 83.9% |
| dsa_indexer_prefill_score_topk | pre_ch_r1383_ctx387k | 0.322 | compute | 1398.54 | 5294.0 | 1980.0 | **2.674x** | 26.4% | 70.6% |
| dsa_indexer_prefill_score_topk | pre_ch_r14082_ctx38k | 0.131 | compute | 1399.26 | 4855.0 | 1997.2 | **2.431x** | 28.8% | 70.1% |
| dsa_indexer_prefill_score_topk | pre_ch_r2934_ctx182k | 0.156 | compute | 1399.02 | 5980.4 | 2000.0 | **2.990x** | 23.4% | 70.0% |
| dsa_indexer_prefill_score_topk | pre_ch_r5406_ctx99k | 0.092 | compute | 1399.13 | 4645.2 | 2031.0 | **2.287x** | 30.1% | 68.9% |
| dsa_indexer_prefill_score_topk | pre_ch_r684_ctx784k | 0.166 | compute | 1398.27 | 7349.8 | 1981.7 | **3.709x** | 19.0% | 70.6% |
| dsa_indexer_prefill_score_topk | pre_un_r1784_ctx185k | 0.045 | compute | 860.65 | 3228.3 | 1235.0 | **2.614x** | 26.7% | 69.7% |
| dsa_indexer_prefill_score_topk | pre_un_r2639_ctx97k | 0.088 | compute | 669.70 | 2631.3 | 978.0 | **2.691x** | 25.5% | 68.5% |
| dsa_indexer_score_topk | dec_bs2_ctx318k | 0.167 | memory | 16.73 | 153.1 | 34.7 | **4.416x** | 10.9% | 48.3% |
| dsa_indexer_score_topk | dec_bs2_ctx418k | 0.166 | memory | 22.00 | 197.2 | 40.8 | **4.838x** | 11.2% | 54.0% |
| dsa_indexer_score_topk | dec_bs3_ctx104k | 0.167 | memory | 8.24 | 80.1 | 23.2 | **3.450x** | 10.3% | 35.5% |
| dsa_indexer_score_topk | dec_bs3_ctx173k | 0.167 | memory | 13.72 | 121.8 | 30.0 | **4.057x** | 11.3% | 45.7% |
| dsa_indexer_score_topk | dec_bs3_ctx217k | 0.167 | memory | 17.14 | 145.9 | 35.2 | **4.150x** | 11.8% | 48.8% |
| dsa_indexer_score_topk | dec_bs3_ctx261k | 0.167 | memory | 20.66 | 176.3 | 39.9 | **4.413x** | 11.7% | 51.7% |
| dsa_indexer_topk | dec_bs2_ctx318k ⬓ | 0.167 | memory | 2.24 | 64.2 | 8.0 | **8.027x** | 3.5% | 28.1% |
| dsa_indexer_topk | dec_bs2_ctx418k | 0.166 | memory | 2.94 | 78.8 | 8.3 | **9.457x** | 3.7% | 35.3% |
| dsa_indexer_topk | dec_bs3_ctx104k ⬓ | 0.167 | memory | 1.12 | 30.3 | 7.1 | **4.264x** | 3.7% | 15.7% |
| dsa_indexer_topk | dec_bs3_ctx173k ⬓ | 0.167 | memory | 1.85 | 41.6 | 8.0 | **5.207x** | 4.4% | 23.1% |
| dsa_indexer_topk | dec_bs3_ctx217k | 0.167 | memory | 2.30 | 49.5 | 8.0 | **6.167x** | 4.7% | 28.7% |
| dsa_indexer_topk | dec_bs3_ctx261k | 0.167 | memory | 2.77 | 55.8 | 8.4 | **6.613x** | 5.0% | 32.9% |
| mxfp4_fused_moe | dec_bs1 | 0.230 | memory | 18.11 | 31.3 | 22.0 | **1.424x** | 57.8% | 82.3% |
| mxfp4_fused_moe | dec_bs2 | 0.288 | memory | 32.75 | 42.5 | 37.7 | **1.126x** | 77.1% | 86.8% |
| mxfp4_fused_moe | dec_bs3 | 0.226 | memory | 44.56 | 53.5 | 48.4 | **1.105x** | 83.3% | 92.0% |
| mxfp4_fused_moe | dec_bs4 | 0.137 | memory | 54.11 | 73.6 | 60.7 | **1.212x** | 73.5% | 89.1% |
| mxfp4_fused_moe | dec_bs5 | 0.076 | memory | 61.83 | 78.0 | 67.0 | **1.163x** | 79.3% | 92.2% |
| mxfp4_fused_moe | dec_bs6 | 0.029 | memory | 68.06 | 82.8 | 77.1 | **1.073x** | 82.2% | 88.3% |
| mxfp4_fused_moe | dec_bs7 | 0.010 | memory | 73.09 | 84.5 | 78.8 | **1.073x** | 86.5% | 92.8% |
| mxfp4_fused_moe | pre_M16955 | 0.168 | compute | 281.81 | 1465.5 | 1161.9 | **1.261x** | 19.2% | 24.3% |
| mxfp4_fused_moe | pre_M1936 | 0.167 | memory | 101.54 | 277.2 | 243.1 | **1.141x** | 36.6% | 41.8% |
| mxfp4_fused_moe | pre_M30519 | 0.170 | compute | 507.26 | 2450.1 | 2005.3 | **1.222x** | 20.7% | 25.3% |
| mxfp4_fused_moe | pre_M32768 | 0.328 | compute | 544.65 | 2584.4 | 2096.7 | **1.233x** | 21.1% | 26.0% |
| mxfp4_fused_moe | pre_M5916 | 0.168 | memory | 117.14 | 571.4 | 478.8 | **1.193x** | 20.5% | 24.5% |
| sparse_mla_prefill | pre_seq16002 ⚠ | 0.164 | memory | 5248.94 | 2388.6 | 1716.6 | **1.392x** | 219.7% | 305.8% |
| sparse_mla_prefill | pre_seq1927 ⚠ | 0.161 | memory | 632.09 | 306.9 | 226.1 | **1.357x** | 206.0% | 279.6% |
| sparse_mla_prefill | pre_seq29556 ⚠ | 0.158 | memory | 9694.89 | 4405.8 | 3579.7 | **1.231x** | 220.0% | 270.8% |
| sparse_mla_prefill | pre_seq32768 ⚠ | 0.355 | memory | 10748.48 | 4568.8 | 3106.0 | **1.471x** | 235.3% | 346.1% |
| sparse_mla_prefill | pre_seq5609 ⚠ | 0.162 | memory | 1839.85 | 868.8 | 619.5 | **1.402x** | 211.8% | 297.0% |

## Per seam, weighted in the time domain (within a regime)

| seam | regime | sum w | base us | cand us | speedup | roof us | base % | cand % |
|---|---|---|---|---|---|---|---|---|
| add_rmsnorm_quant | decode | 0.934 | 8.6 | 8.1 | **1.058x** | 0.11 | 1.3% | 1.4% |
| add_rmsnorm_quant | prefill | 0.066 | 160.6 | 157.5 | **1.020x** | 51.80 | 32.3% | 32.9% |
| dsa_indexer_decode | decode | 1.000 | 87.0 | 18.8 | **4.637x** | 14.21 | 16.3% | 75.7% |
| dsa_indexer_prefill | prefill | 1.000 | 3920.3 | 1395.4 | **2.809x** | 1179.67 | 30.1% | 84.5% |
| dsa_indexer_prefill_score_topk | prefill | 1.000 | 5298.1 | 1869.1 | **2.835x** | 1310.67 | 24.7% | 70.1% |
| dsa_indexer_score_topk | decode | 1.000 | 145.7 | 34.0 | **4.290x** | 16.41 | 11.3% | 48.3% |
| dsa_indexer_topk | decode | 1.000 | 53.3 | 8.0 | **6.683x** | 2.20 | 4.1% | 27.6% |
| mxfp4_fused_moe | decode | 0.996 | 51.0 | 43.5 | **1.172x** | 38.65 | 75.8% | 88.9% |
| mxfp4_fused_moe | prefill | 1.000 | 1652.0 | 1344.3 | **1.229x** | 348.69 | 21.1% | 25.9% |
| sparse_mla_prefill | prefill | 1.000 | 2897.9 | 2085.1 | **1.390x** | 6603.45 | 227.9% | 316.7% |

## Dispatch-floor-limited rows (⬓) -- the byte roof is not reachable

A dispatched kernel costs ~2.3 us of device time before it moves a byte, so a seam of n kernels cannot go below n x 2.3 us. Where the byte roof sits under that floor, the headroom shown in the table is a kernel-COUNT statement, not a bandwidth one, and the lever is fusing nodes rather than tuning memory.

| seam | case | kernels | byte roof us | dispatch floor us | base us | cand us | cand x over floor |
|---|---|---|---|---|---|---|---|
| add_rmsnorm_quant | dec_bs2_m12_nores | 1 | 0.04 | 2.3 | 4.3 | 4.0 | 1.7x |
| add_rmsnorm_quant | dec_bs2_m12_res | 2 | 0.09 | 4.6 | 8.9 | 8.5 | 1.9x |
| add_rmsnorm_quant | dec_bs3_m18_nores | 1 | 0.07 | 2.3 | 4.3 | 4.0 | 1.7x |
| add_rmsnorm_quant | dec_bs3_m18_res | 2 | 0.13 | 4.6 | 9.0 | 8.5 | 1.8x |
| dsa_indexer_topk | dec_bs2_ctx318k | 1 | 2.24 | 2.3 | 64.2 | 8.0 | 3.5x |
| dsa_indexer_topk | dec_bs3_ctx104k | 1 | 1.12 | 2.3 | 30.3 | 7.1 | 3.1x |
| dsa_indexer_topk | dec_bs3_ctx173k | 1 | 1.85 | 2.3 | 41.6 | 8.0 | 3.5x |

## Suspect rows (model disproved by measurement)

- **sparse_mla_prefill/pre_seq16002** -- roofline_pct=220% > 100%: the measured call is FASTER than this model's own roof, so the model's bytes/FLOPs are overcounted. Not a result -- do not quote attainable_speedup for this row. Byte bracket: the perfect-reuse floor puts the roof at 181.5 us instead.
- **sparse_mla_prefill/pre_seq1927** -- roofline_pct=206% > 100%: the measured call is FASTER than this model's own roof, so the model's bytes/FLOPs are overcounted. Not a result -- do not quote attainable_speedup for this row. Byte bracket: the perfect-reuse floor puts the roof at 21.9 us instead.
- **sparse_mla_prefill/pre_seq29556** -- roofline_pct=220% > 100%: the measured call is FASTER than this model's own roof, so the model's bytes/FLOPs are overcounted. Not a result -- do not quote attainable_speedup for this row. Byte bracket: the perfect-reuse floor puts the roof at 335.3 us instead.
- **sparse_mla_prefill/pre_seq32768** -- roofline_pct=235% > 100%: the measured call is FASTER than this model's own roof, so the model's bytes/FLOPs are overcounted. Not a result -- do not quote attainable_speedup for this row. Byte bracket: the perfect-reuse floor puts the roof at 371.7 us instead.
- **sparse_mla_prefill/pre_seq5609** -- roofline_pct=212% > 100%: the measured call is FASTER than this model's own roof, so the model's bytes/FLOPs are overcounted. Not a result -- do not quote attainable_speedup for this row. Byte bracket: the perfect-reuse floor puts the roof at 63.6 us instead.
