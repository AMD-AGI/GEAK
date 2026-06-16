# final_report — Qwen-Qwen3.5-27B 端到端吞吐优化（迭代2 · GEMM 调优赢点）

## 运行概况
- **模型 / 架构**: Qwen-Qwen3.5-27B（`hybrid_linear_attention_dense`，64 层 = 48 linear-attn/gated-delta + 16 full-attn，稠密 MLP，bf16，hidden 5120，intermediate 17408）。
- **服务栈**: sglang 0.5.11 + torch 2.9.1+rocm7.2，MI300X（gfx942）。**serving 恒为 TP=1 单卡**；`gpu_ids=0,1,2,3` 仅作优化并行池。
- **负载**: ISL/OSL/conc = 1024/1024/64（prefill 主导）。噪声带 = 0.5%（紧测量：交错 A/B + E2E_REPEATS + 分布不重叠 + 引擎生效证明）。
- **一句话结论**: 本轮**首次拿到 GEMM 调优的真实 e2e 赢点 +2.23%**（aiter per-shape DB 调优，叠加在 `--attention-backend triton` 上），累计约 **+6%**（1492.7 → ~1583.5 tok/s）。head 之后进入 milestone（可编辑核簇）时因嵌套递归进程风暴卡死（finding #8），milestone 的 e2e 叠加 integrate 未跑完、运行被手动停止——故无 Finalize/Validate 的最终合并数字。

> 与迭代1（同模型同负载，仅 `--attention-backend triton`，+4.1%；GEMM/单核因 bias 失配与 Amdahl 均未过闸）对照：本轮的关键差异是 **bug#3 修复**——GEMM 调优改用 `AITER_TUNE_GEMM=1` 实捕获（真实 `bias=False` + 全覆盖），使其**既生效又获胜**。

---

## 阶段树（Phases · 每一步优化了哪几项）

```
Phases
├── ✔ 1 Setup          baseline = 1492.7 tok/s  (TP=1, GPU0, spread 0.23%)
├── ✔ 2 Profile        Top-N: 稠密 GEMM ~79%（rank1 48.7% / rank2 17.2% / rank3 8.7% / rank4 4.4%）、
│                       gated-delta 簇 ~9%（chunk_gated_delta_h 2.9% …）
├── ✔ 3 Strategize     路由: 1 个 GEMM head (dense_gemm up/gate) + 4 个可编辑核 → milestone
├── ✔ 4 ConfigSweep
│   ├── ✔ cfg0  --attention-backend triton              e2e +X%  → 接受 (ref 基线)
│   ├── ✘ cfg1  --quantization fp8 (+kv fp8)            → 拒绝 (accuracy gate)
│   └── ✘ cfg2  --enable-torch-compile                  → 拒绝 (triton 版本不匹配)
├── ✔ 5 HeadKernel     dense_gemm up/gate (K=5120, N∈{14336,16384,34816})
│   ├── ✔✔ aiter DB 调优 (AITER_TUNE_GEMM=1 实捕获→gradlib→AITER_CONFIG_GEMM_BF16)
│   │        iso 1.029×, 引擎生效 246 hits → **e2e +2.23% (1548.9→1583.5, 5-rep 不重叠) → 接受**
│   └── ·  Triton GEMM (team_workflow 著)  iso 1.466×   (e2e 闸由 aiter env 胜出)
├── ⚠ 6 Milestone      可编辑 FLA/mamba 簇 (floor=4; 并行优化已跑, e2e 叠加 integrate 未完成)
│   ├── · chunk_fwd_kernel_o                iso 1.228×   (待 integrate)
│   ├── · _causal_conv1d_fwd_kernel         iso 1.066×   (待 integrate)
│   ├── · chunk_gated_delta_rule_fwd_h      iso 1.004×   (待 integrate)
│   └── ✗ recompute_w_u_fwd_kernel          递归卡死 ~3h → 触发 #8 进程风暴, 阻塞 milestone barrier
├── ✗ 7 Finalize       未到达 (run 手动停止)
├── ✗ 8 Report         未到达 (本报告为手工整理)
└── ✗ 9 Validate       未到达 (无最终合并验证)

图例: ✔ 接受 · ✔✔ 关键赢点 · ✘ 拒绝 · ⚠ 部分完成 · ✗ 未到达 · · 已产出未入栈
已确认入栈: phase4 --attention-backend triton + phase5 aiter GEMM 调优 (+2.23%)，累计 ~+6%。
未取得: milestone 单核 e2e 叠加数字 (因 #8 卡死)。
```

## 产物树（哪个 phase 产出哪些文件）

```
e2e_Qwen-Qwen3.5-27B_20260607_193315.../
├── baseline/bench_summary.json                  # [P1] TRUE baseline 1492.7 (TP=1)
├── profile/round_0|round_config/profile_topN.*  # [P2] Top-N 细分 (GEMM ~79%)
├── strategy.md                                  # [P3] Amdahl 路由
├── config/
│   ├── sweep_results.json + cfg0/ cfg2/         # [P4] triton✔ / fp8✘ / torch.compile✘
│   ├── capture.log + captured_untuned_gemm.csv  # [P5] AITER_TUNE_GEMM=1 实捕获 (bias=False 真实 shape)
│   ├── hot_untuned_gemm.csv (78 桶, bucket-reduce)
│   └── Qwen-Qwen3.5-27B_bf16_tuned_gemm.csv     # [P5] aiter 调优产物 (78 shape; 仅本环境有效, 勿外用)
├── kernels/
│   ├── dense_gemm_aiter_tuned_gemm_task/        # [P5] head GEMM op unittest + opbench_result.json
│   ├── chunk_fwd_kernel_o_task/ … (4 个可编辑核 task)  # [P6]
│   └── _exp/team_*                              # [P5/6] 递归 team_workflow (head Triton著 1.466× + 3 核已优化)
├── overlay/cand_dense_gemm_aiter_tuned_gemm/    # [P5] head GEMM e2e A/B (ref/cand 两块, 2-launch)
│   ├── ref/bench_runs.jsonl                      #      med 1548.9 (triton attn)
│   └── cand/bench_runs.jsonl + server.log        #      med 1583.5, 246 'is tuned on cu_num' 命中
├── logs/  integrate_dense_gemm_aiter_tuned_gemm.log, opbench_dense_gemm.log, capture_*, cfg_* …
└── (无 final/ · architect_report · director_e2e_validation —— run 在 milestone 阶段停止)
```

---

## 关键数据

**Baseline**（TP=1, 3 次）: 1492.7 tok/s（spread 0.23%；TTFT 3626 ms, TPOT 39.25 ms）。

**Head GEMM 调优 e2e 闸**（fast 2-launch 交错 A/B, 各 5 次）:
| | 各次 tok/s | 中位 | min/max |
|---|---|---|---|
| ref（triton attn） | 1509.4 / 1554.7 / 1544.9 / 1554.4 / 1548.9 | **1548.9** | 1509.4 / 1554.7 |
| cand（triton attn + aiter GEMM 调优） | 1586.2 / 1583.5 / 1581.1 / 1573.4 / 1586.4 | **1583.5** | **1573.4** / 1586.4 |

→ **Δ = +2.23%，分布不重叠**（cand_min 1573.4 > ref_max 1554.7），引擎生效 **246 次命中** → **接受** ✅。
（aiter `tuned_gemm` 在线路径用上了调好的 hipBLASLt 解；同 bf16 数学，parity 安全。）

**单核隔离加速**（已优化，待 e2e 叠加）: chunk_fwd_kernel_o **1.228×**、_causal_conv1d 1.066×、chunk_gated_delta_h 1.004×、head Triton 著 1.466×。

---

## 汇总表（所有尝试）
| 杠杆 | 隔离 | e2e | 判定 | 备注 |
|---|---|---|---|---|
| `--attention-backend triton` | — | 接受（ref 基线） | ✔ | 配置赢点 |
| `--quantization fp8` | — | — | ✘ | accuracy gate 未过 |
| `--enable-torch-compile` | — | — | ✘ | triton 版本不匹配 |
| **aiter GEMM DB 调优** | 1.029× | **+2.23%（不重叠, 246 命中）** | **✔✔ 接受** | **GEMM 赢点（bug#3 修复后生效）** |
| Triton GEMM（team_workflow 著） | 1.466× | — | · | e2e 闸由 aiter env 胜出 |
| chunk_fwd_kernel_o | 1.228× | 待测 | · | milestone 未完成 |
| _causal_conv1d_fwd_kernel | 1.066× | 待测 | · | milestone 未完成 |
| chunk_gated_delta_rule_fwd_h | 1.004× | 待测 | · | milestone 未完成 |
| recompute_w_u_fwd_kernel | 卡死 | — | ✗ | 递归 hang → #8 |

---

## 结论 / 注意事项 / 下一步
- **GEMM 调优在本模型/栈上是真实有效的杠杆**：bias 正确的全覆盖 aiter 调优 = **+2.23% e2e**（叠加 triton attn 后累计 ~+6%）。这**修正了"GEMM 调优无收益"的旧结论**（旧结论源于 bias 失配的部分调优）。
- **box 漂移 → 只信同会话 A/B**；所有 e2e 用紧测量（交错 + 不重叠 + 引擎证明）。
- **finding #8（效率/稳定性）**：head 著核 + milestone 并行递归各自拉起 ROCm/aiter init，会 fork 出数百个 `rocm_agent_enumerator` 把主机 CPU 打满 → 近停滞、且污染 e2e 计时。**有界运行应**：bucket-reduce 收紧调优 shape 数、调小 `head_author_max`/`kernel_budget`、串行化重型嵌套调优。
- **下一步**：在不 fork-storm 的配置下重跑 milestone，取单核簇 **叠加在 +2.23% GEMM 赢点之上** 的 e2e 合并数字（预计 gated-delta 簇 ~0.4–0.8%，是否过 0.5% 由合并闸定）。
