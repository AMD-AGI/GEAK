# 实现线 — Qwen-Qwen3.5-27B 端到端吞吐优化全时间线

## 运行概况
- **模型 / 架构**: Qwen-Qwen3.5-27B（`Qwen3_5ForConditionalGeneration`），架构类 `hybrid_linear_attention_dense`。
  64 层 = 48 层 linear-attention（mamba/gated-delta 风格）+ 16 层 full-attention（full_attention_interval=4），
  稠密 MLP（无 MoE）；hidden 5120、intermediate 17408、head_dim 256、24 q heads / 4 kv heads、vocab 248320；dtype bf16。
- **服务栈**: sglang 0.5.11，torch 2.9.1+rocm7.2.0，tp_size=1。
- **可用后端**: aiter / triton / hipblaslt。**缺失**: hipblaslt-bench（无离线 GEMM CLI 调优）、ckProfiler（无 CK 实例扫描）。
- **负载**: ISL/OSL/conc = 1024/1024/64（**prefill 主导**）。
- **GPU**: AMD Instinct MI300X（gfx942），baseline 用 GPU0；噪声带 = 0.5%。
- **时间**: 2026-06-07。
- **最终结论一句话**: 唯一被接受的优化是配置档 `--attention-backend triton`（+4.44% e2e，1485.432 -> 1551.4 tok/s）；
  所有可编辑 Triton FLA/mamba 内核虽有真实隔离加速且引擎确认生效，但在 ~81% GPU 为稠密 GEMM 的 prefill régime 下
  均因 Amdahl 落在 0.5% 噪声带内而判 NULL，无内核源码补丁进入最终栈。

---

## 阶段树（Phases · 每一步优化了哪几项）

```
Phases
├── ✔ 1 Setup          baseline = 1485.4 tok/s  (TP=1, GPU0, spread 0.44%)
├── ✔ 2 Profile        Top-N: 稠密 GEMM ~81% / gated-delta 簇 ~9% / act_and_mul 2.2%
├── ✔ 3 Strategize     路由: 1 个 GEMM head (h0 up/gate) + 4 个可编辑核
├── ✔ 4 ConfigSweep
│   ├── ✔ cfg0  --attention-backend triton       e2e +4.15%   → 接受 (唯一进入最终栈)
│   └── ✘ cfg1  --chunked-prefill-size 8192       e2e −0.42%   → 拒绝 (带内)
├── ✔ 5 HeadKernel     dense_gemm up/gate (K=5120, N∈{14336,16384,34816})
│   ├── ✘ aiter DB 调优                 iso 1.032×  → 0 生效 (bias=True 合成 vs live bias=False 失配)
│   └── ✘ Triton GEMM (team_workflow 著) iso 0.99×  → 打不过 hipBLASLt
├── ✔ 6 Milestone      可编辑 FLA/mamba 簇 (并行优化 → 串行 e2e 闸, floor=4)
│   ├── ✘ chunk_gated_delta_rule_fwd_kernel_h   iso 1.18× → e2e +0.17%        (2.95%gpu, 上限<0.5%)
│   ├── ✘ chunk_fwd_kernel_o                    iso 1.14× → e2e −0.03%        (1.98%gpu)
│   ├── ✘ _causal_conv1d_fwd_kernel             iso 1.10× → e2e +0.29% STACK  (1.26%gpu, parity 12/12)
│   └── ✘ recompute_w_u_fwd_kernel              iso 0.99× (无隔离加速)
├── ✔ 7 Finalize       bundle: final_launch.sh + final_patch.diff(空)  (仅 triton flag, 无 overlay)
├── ✔ 8 Report         architect_report.md + 实现线.md
└── ✔ 9 Validate       1549.2 tok/s, +4.1%, accepted, parity pass

图例: ✔ 完成/接受 · ✘ 拒绝(带内/无加速/不生效) · STACK 可叠加但单独未过 0.5% 闸
结论: 仅 phase 4 的 --attention-backend triton 进入最终栈; head GEMM 与 4 个单核均未过 e2e 闸。
```

## 目录结构（产物 tree · 哪个 phase 产出哪些文件）

```
e2e_Qwen-Qwen3.5-27B_20260607_080209.../
├── env_report.{md,json}                         # 预检/能力报告
├── baseline/bench_summary.json                  # [P1] TRUE baseline 1485.4
├── profile/round_0|round_config/profile_topN.*  # [P2/5] Top-N 细分
├── strategy.md                                  # [P3] Amdahl 路由
├── config/
│   ├── sweep_results.json                        # [P4] cfg0/cfg1 扫描
│   ├── cfg0/  cfg1/                              # [P4] 各配置 bench
│   ├── hot_untuned_gemm.csv                      # [P5] AITER_TUNE_GEMM 捕获的真实 shape
│   └── Qwen-Qwen3.5-27B_bf16_tuned_gemm.csv      # [P5] aiter GEMM 调优产物 (bias=True 合成→失配)
├── kernels/
│   ├── h0_cijk_upgate_gemm_task/                 # [P5] head GEMM op unittest + opbench_result.json
│   ├── chunk_gated_delta_rule_fwd_kernel_h_task/ # [P6] 可编辑核 task (含 reference_io.pt + unittest.py)
│   ├── chunk_fwd_kernel_o_task/                  # [P6]
│   ├── _causal_conv1d_fwd_kernel_task/           # [P6]
│   ├── recompute_w_u_fwd_kernel_task/            # [P6]
│   └── _exp/team_*                               # [P5/6] 递归 team_workflow 优化 (head Triton著 + 4 核)
├── overlay/cand_*                                # [P5/6] 各候选 e2e A/B (ref/cand 两块)
├── final/
│   ├── final_launch.sh                           # [P7] 启动优化 server + 测速 (带 --attention-backend triton)
│   ├── final_patch.diff                          # [P7] 空 (纯 flag 赢点, 无源码 patch)
│   └── overlay/                                  # [P7] 空 (无接受的 kernel overlay)
├── architect_report.md  /  实现线.md             # [P8] 报告
├── director_e2e_validation.json                 # [P9] 官方验证 1549.2 / +4.1% / accepted
└── logs/                                         # 各阶段日志 (capture/cfg/opbench/integrate/validation)
```

## Baseline 阶段

**吞吐**（`baseline/bench_summary.json`，3 次 repeat）:
- 中位 **1485.432 tok/s**，spread 0.44%
- 各次: 1485.432 / 1479.825 / 1486.405 tok/s
- TTFT 中位 3598.067 ms，TPOT 中位 39.523 ms

**Profile 细分**（`profile/round_0/profile_topN.md`，torch-trace；总 GPU 时间 5051.78 ms / 11113 launches / 64 distinct kernels）:

| # | kernel | class | backend | 可编辑 | calls | 总 ms | %gpu | shape |
|--|--------|-------|---------|------|-------|----------|------|--------|
| 1 | Cijk...MT256x192x64 | library_gemm | hipblaslt | N | 336 | 2466.834 | **48.8** | [[16040,5120],[5120,14336/16384...]] |
| 2 | Cijk...MT256x192x64 | library_gemm | hipblaslt | N | 256 | 870.846 | **17.2** | [[15360,17408],[17408,5120]]... |
| 3 | Cijk...MT256x224x64 | library_gemm | hipblaslt | N | 128 | 434.096 | **8.6** | [[16040,17408],[17408,5120]]... |
| 4 | Cijk...MT224x320x64 | library_gemm | hipblaslt | N | 48 | 224.633 | **4.5** | [[16369,5120],[5120,16384]] |
| 5 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | fused_custom | triton | Y | 192 | 145.073 | 2.9 | [1,1024,16/48,128] |
| 6 | act_and_mul_kernel (sgl_hip) | elementwise_overhead | torch_native | Y | 256 | 108.330 | 2.1 | [[1024,17408],[1024,34816]] |
| 7 | chunk_fwd_kernel_o | fused_custom | triton | Y | 192 | 98.353 | 1.9 | [1,1024,16/48,128] |
| 8 | recompute_w_u_fwd_kernel | fused_custom | triton | Y | 192 | 82.337 | 1.6 | [1,1024,16/48,128] |
| 9 | elementwise_kernel_manual_unroll | elementwise_overhead | torch_native | Y | 1344 | 72.167 | 1.4 | — |
| 10 | _causal_conv1d_fwd_kernel | fused_custom | triton | Y | 192 | 65.523 | 1.3 | — |
| 11 | aiter add_rmsnorm_quant_kernel | fused_custom | aiter | Y | 512 | 65.044 | 1.3 | [[1024,5120]...] |
| 12 | ck_tile FmhaBatchPrefillWithPagedKVCache | library_attn | ck | N | 64 | 54.624 | 1.1 | paged-attn |
| 14 | Cijk...MT160x256x64 | library_gemm | hipblaslt | N | 64 | 43.350 | 0.9 | [[1024,5120],[5120,34816]] |
| 15 | chunk_gated_delta_rule_fwd_kkt_solve_kernel | fused_custom | triton | Y | 192 | 42.098 | 0.8 | [1,1024,16/48,128] |
| 17 | _layer_norm_fwd_1pass_kernel | reduction_norm | triton | Y | 192 | 30.738 | 0.6 | [[49152,128]...] |

**关键判读**: 稠密 hipBLASLt GEMM（rank 1-4）合计约 **79%**，全部 library_gemm 约 **81%** GPU 时间 —— 这是唯一具备超过噪声带 e2e 余量的杠杆（Amdahl 优先级）。可编辑 Triton FLA/mamba 簇（gated-delta、conv1d、norm）单个均 ≤3% GPU。baseline 注意力走 CK paged-attention（rank 12，1.1%）。

---

## 时间线逐阶段

### 阶段 A — ConfigSweep（配置快路径，最先做，无源码编辑）
来源: `config/sweep_results.json`，e2e 每档 5 repeats，闸 0.5%。

**尝试 1 — `--attention-backend triton`（cfg0，叠加于 baseline）**
- e2e: **1547.027 tok/s**，spread 0.8%；vs baseline **+4.15%**（远超 0.5% 闸）。
- 引擎生效: ✅ server.log 显示 `attention_backend='triton'`、`linear_attn_backend='triton'`、`mamba_backend='triton'`。
- Parity: PASS —— 贪婪 temp=0 定种子，5 prompts 中 3 个逐字节相同，2 个仅在深度重复尾部分歧但答案一致（Paris；17x23=391），属良性 bf16 平局，无质量回退。
- 副作用: 将 16 层 full-attention 与 linear-attn/mamba 路径转为可编辑 Triton 内核，为下游内核轨道开辟编辑面。
- **决策: 接受。** 当前最佳配置 = `--attention-backend triton`。

**尝试 2 — `--chunked-prefill-size 8192`（cfg1，叠加于 cfg0）**
- e2e: 1540.519 tok/s，spread 0.92%；vs cfg0 **-0.42%**（落在 0.5% 噪声带内且略负）。
- 引擎生效: ✅ `chunked_prefill_size=8192`；但 cuda-graph 默认已开（`disable_cuda_graph=False`，`cuda_graph_max_bs=512`），微小 elementwise/index/cat 已被 graph 覆盖，无 decode-dispatch 余量。
- **决策: 拒绝。** prefill 主导场景缩小 prefill chunk 无 decode 交错收益，反而略伤 prefill GEMM batching。回退，配置维持 cfg0。

**ConfigSweep 小结**: 接受 `--attention-backend triton`，env 空。best = 1547.027 tok/s，vs baseline 1.0415x。

### 阶段 B — 重新 Profile（针对 triton-attention 配置）
来源: `profile/round_config/profile_topN.md`（总 GPU 5058.44 ms / 14504 launches / 62 kernels）。
- 瓶颈**未移动**: 稠密 hipBLASLt GEMM rank 1-4 = 78.8% gpu，全部 library_gemm ~81%（与 baseline 几乎一致）。
- 变化: baseline 的 CK paged-attention（library_attn，1.1%）被替换为可编辑 Triton `_fwd_kernel`（rank 12，1.05%，64 calls = 16 full-attn 层 × prefill）；gated-delta Triton 内核在 baseline 已是 Triton，%gpu 基本不变（2.9%->3.0%）。
- 净结论: 稠密 GEMM 仍是 head/config 轨道的首要杠杆。

### 阶段 C — HeadKernel（稠密 GEMM head 轨道）
**尝试 — aiter tuned_gemm CSV（up/gate GEMM，K=5120，N ∈ {14336,16384,34816}）**
来源: `kernels/h0_cijk_upgate_gemm_task/`、`overlay/cand_cijk_upgate_gemm/`、`config/*tuned_gemm.csv`、tune.log。
- 隔离效果: ~1.032x（离线）。
- e2e: 0%（probe 1545.5 tok/s ≈ 当前接受 1547.027，统计同一）。
- 引擎生效: ❌ **未真正消费 CSV**。需 `SGLANG_USE_AITER=1` 才会走 tgemm.mm（否则 UnquantizedLinearMethod 走 F.linear/hipBLASLt 默认）。即便开了 aiter，warm server.log 中 `is tuned on cu_num`=0、`not found tuned config ... using default config`=258。
- 根因: CSV 以**合成 bias=True 的 prefill M-bucket {16040,16369,1024} × N{14336,16384,34816}** 调优，但**实测 up/gate GEMM 是 bias=False、真实运行 M-bucket（122/96/88/80...）**；aiter 查找键含 bias，逐次 miss → 退回默认 torch/hipBLASLt 解。
- **决策: 拒绝（在引擎生效闸前，TunableOp 教训）。** 一个无法接到 live 路径的真实隔离加速属预期结果，非 e2e 测量。
- 重调修复: 用 **live 捕获的真实 shape（bias=False、实际 conc=64 的 M-bucket）**重生成 CSV，集成前先验证 `is tuned on cu_num > 0`。

### 阶段 D — Milestone 1（可编辑 Triton FLA/mamba 内核簇，达到 MIN_KERNEL_TASKS 地板）
来源: `HISTORY.ledger`、`overlay/*/`、`kernels/*_task/`、`insight_log.md`。
所有方向 provenance OK（reference_io.pt sha256 与 meta.json 一致，unittest.py 未篡改），引擎均经 overlay banner 证明生效（4 hits/worker + `[OVERLAY_ENGAGED]`），紧凑同会话交错 A/B。

**尝试 1 — chunk_gated_delta_rule_fwd_kernel_h（k0）**
- 隔离: **1.1811x**；e2e: **+0.171%**；%gpu 2.95%。
- A/B（GPU0，pin port，5 clean REF + 6 clean CAND，经 3 次 grpc-port-flake 重试）: ref_med 1551.196 [1546.84, 1557.49]，cand_med 1553.847 [1550.40, 1566.40]。
- 闸: delta +0.171% < 0.5% **且**分布重叠（cand_min 1550.40 < ref_max 1557.49）→ FAIL。
- **决策: 拒绝（Amdahl）。** 2.95% GPU × 1.1811x 的 e2e 上限 ~0.45%，本就低于带。

**尝试 2 — chunk_fwd_kernel_o（k1）**
- 隔离: **1.1405x**；e2e: **-0.035%**；%gpu 1.98%。
- A/B（GPU1，pin port 31337，5 ref + 5 cand）: ref_med 1551.585 [1546.918, 1554.663]，cand_med 1551.037 [1548.667, 1558.197]。
- 闸: delta -0.035% < 带 **且**分布重叠（cand_min 1548.667 < ref_max 1554.663）→ FAIL。
- **决策: 拒绝（Amdahl）。** e2e 上限 ~1.98%×(1-1/1.14)=~0.24% < 带。

**尝试 3 — _causal_conv1d_fwd_kernel（k2）**
- 隔离: **1.1049x**；e2e: **+0.292%**；%gpu 1.26%。
- 引擎: ✅ banner `injected ...mamba.causal_conv1d_triton`（4 hits/process）；REF 无 overlay 对照确认。winner_kind=patch（add-module 单子模块注入）。
- Parity: **PASS —— 12/12 贪婪 prompts 逐字节相同**。
- A/B（GPU3，pin port 31537，5 repeats/leg）: REF med 1536.27 [1530.88, 1541.72]，CAND med 1540.75 [1531.03, 1541.17]。
- 闸: delta_med +0.292%（非负，cand_med ≥ ref_med）但 < 0.5% 带，分布重叠（cand_min 1531.03 < ref_max 1541.72）→ 非强接受。
- **决策: 拒绝 / STACK 携带前行。** e2e 上限 ~1.26%×(1-1/1.10)=~0.11% < 带；无回退、parity 净，可与同簇内核复合 —— 进 Director 的合并-vs-真 baseline 闸时携带捕获复合收益。

**尝试 4 — recompute_w_u_fwd_kernel（k3）**
- 隔离: **0.9938x**（无隔离加速）。
- **决策: 拒绝。** 无加速，无需 e2e 测量。

**共同模式**: 真实且经验证的隔离加速无法在 e2e 浮现，因每个内核在 prefill régime（~80% GPU 为稠密 GEMM）只占很小比例。属预期 Amdahl 结果，非集成 bug。快速判据: `pct_gpu × (1 - 1/iso) < 0.5%` ⇒ 只能 carry-forward，不能单独过闸。

### 阶段 E — Final 复测（接受配置 `--attention-backend triton`，无 overlay）
来源: `final/bench_out/bench_summary.json`（5 repeats）、`final/final_patch.diff`（无源码补丁）。
- 吞吐: 中位 **1551.4 tok/s**，spread 0.33%。
- 各次: 1553.739 / 1551.400 / 1550.808 / 1549.324 / 1554.405 tok/s。
- TTFT 中位 3564.864 ms，TPOT 中位 37.616 ms。
- final_patch.diff: 无任何内核源码补丁；唯一优化是 launch 标志 `--attention-backend triton`（写在 final_launch.sh 的 EXTRA_SERVER_ARGS）。

---

## 汇总表（所有尝试）

| 杠杆 | 隔离加速 | e2e 效果 | 判定 | 根因 |
|---|---|---|---|---|
| `--attention-backend triton`（cfg0） | — | **+4.15%**（1485.4→1547.0），过闸 | **接受** | 唯一过闸、引擎生效、parity 净；并开放可编辑 Triton 编辑面 |
| `--chunked-prefill-size 8192`（cfg1） | — | -0.42%（带内，略负） | 拒绝 | cuda-graph 默认已开，prefill 主导无 decode 交错余量 |
| aiter tuned_gemm CSV（up/gate GEMM） | 1.032x | 0% | 拒绝 | 引擎生效闸：bias=True/合成 M-bucket 与 live bias=False/真实 M-bucket 键失配，逐次 miss 退默认 |
| chunk_gated_delta_rule_fwd_kernel_h | 1.1811x | +0.171%（带内，分布重叠） | 拒绝 | Amdahl：2.95% gpu，上限 ~0.45% < 带 |
| chunk_fwd_kernel_o | 1.1405x | -0.035%（带内，分布重叠） | 拒绝 | Amdahl：1.98% gpu，上限 ~0.24% < 带 |
| _causal_conv1d_fwd_kernel | 1.1049x | +0.292%（带内，非负，parity 12/12） | 拒绝 / STACK | Amdahl：1.26% gpu，上限 ~0.11% < 带 |
| recompute_w_u_fwd_kernel | 0.9938x | — | 拒绝 | 无隔离加速 |

---

## 最终交付
- **接受配置**: `--attention-backend triton`，env 空，无内核源码补丁、无 overlay。
- **吞吐**: baseline 1485.432 → final **1551.4 tok/s**，**1.0444x（+4.44%）**。
- Milestone（接受改动）数: 1；budget 用 4/4，达 MIN_KERNEL_TASKS 地板。

## 测量注意事项（box 漂移 → 只信同会话 A/B）
- e2e 中位密集聚集（~1551 tok/s），机器 box 跨会话漂移；只信**同会话交错 A/B**，闸需 **delta_med > 0.5% 带 且 分布不重叠（cand_min > ref_max）**，仅靠正中位会被重叠分布否决。
- 噪声带 0.5%；每档 ≥5 repeats，server 保持 WARM，不把启动计入计时窗。
- Parity 必查（贪婪 temp=0 定种子），更快但更错的 server 是回退。
- 基础设施: sglang `grpc_port = port + 10000` 且拒绝 >65535 → 始终 pin 低端 PORT（用过 31337/31537），并预留 grpc-port-flake 重试预算（见过最多 3 次）。

## 下一步可探方向
1. **稠密 hipBLASLt GEMM（~81% GPU）是唯一仍有超过带 e2e 余量的杠杆** —— 78% GEMM 上一个 1.15x ≈ +10% e2e。
2. **aiter tuned_gemm 重调**: 用 live 捕获 shape（bias=False、真实 conc=64 M-bucket）重生成 CSV，开 `SGLANG_USE_AITER=1`，集成前验证 `is tuned on cu_num > 0`。
3. **自写 Triton 稠密 GEMM**（hipblaslt-bench 与 ckProfiler 均缺失，离线 CLI 调优不可用）于 head op 任务，让 e2e 闸择优。
4. 可编辑 Triton FLA/mamba 簇作为单独过闸轨道已耗尽；将 conv1d（非负、parity 净）携带进合并-vs-真 baseline 闸捕获复合收益。
5. 一旦达到地板且最佳剩余候选无法过带，按 Amdahl 停止规则考虑停止内核轨道。
