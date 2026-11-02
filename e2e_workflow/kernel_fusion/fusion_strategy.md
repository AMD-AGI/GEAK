你正在帮助设计 GEAK 的 Kernel Fusion 能力。

目标不是简单判断“哪些 kernel 能融合”，而是针对给定推理 workload，综合以下信息，选择最合适的 fusion pattern：

1. workload 特征；
2. execution trace；
3. operator / tensor dependency；
4. kernel duration 和 launch overhead；
5. memory traffic 和 intermediate tensor；
6. VGPR / LDS / occupancy 等资源约束；
7. concurrency、batch size、token 数以及并行度。

不要简单追求更大的 fusion scope。

核心原则是：

**使用能够消除当前主要性能瓶颈的最小 fusion range，并通过实际性能数据验证最终选择。**



## 一、Fusion Level 定义

当前将 Kernel Fusion 分为四级：

### L1 — Element-wise Fusion

融合多个轻量 element-wise kernel。

典型场景：

- add + mul
- activation + element-wise transform
- normalization 后的轻量处理链
- 多个连续的 1～5 us 小 kernel

优先考虑 L1 的条件：

- kernel 本身计算量很小；
- launch overhead 占比较高；
- kernel 之间存在简单直接的 producer-consumer dependency；
- 中间 tensor 会反复写入和读取 HBM；
- fusion 后不会明显增加 resource pressure。

主要目标：

- 减少 kernel launch；
- 减少 intermediate HBM traffic。

---

### L2 — Operator-wise Fusion

以一个主要 operator 为中心，将它与前后相关的轻量操作进行融合。

典型场景：

- GEMM + bias + activation
- GEMM + quantization
- Attention + local pre/post processing
- normalization + projection

优先考虑 L2 的条件：

- 一个主要 operator 占据该 region 的大部分计算时间；
- 前后存在若干较轻的 kernel；
- intermediate tensor 较大；
- 可以通过 epilogue / prologue fusion 消除 HBM round-trip；
- fusion 不会明显影响主 operator 的最优 tile、VGPR/LDS 使用和 occupancy。

主要目标：

- 保持主 operator 高性能；
- 同时消除周边小 kernel 和中间内存访问。

---

### L3 — Stage-wise Fusion / MegaKernel

将属于同一个功能 stage 的多个 operator 融合。

典型场景：

- dispatch + expert GEMM + combine
- router + metadata preparation + dispatch
- attention 内部多个连续 stage
- communication + compute overlap

优先考虑 L3 的条件：

- 一个 stage 内包含多个互相依赖的 kernel；
- 没有单个 operator 完全主导总耗时；
- 存在明显 producer-consumer pipeline；
- 可以减少多个 kernel launch / synchronization；
- communication 和 compute 存在 overlap 机会；
- 有足够的 token / batch / concurrency 支撑并行执行。

当前经验：

**Mega / MegaMoE 类 kernel 在较高 concurrency 场景下通常收益更明显。**

原因包括：

- token 数更多；
- 每个 expert 的 effective batch size 更大；
- expert GEMM 更容易达到较高 compute utilization；
- communication / compute 更容易 overlap。

注意：

高 concurrency 只是一个 prior，不是硬规则。

---

### L4 — Layer-wise Fusion / MonoKernel

将 decoder layer 中的大部分甚至全部 stage 融合到一个 persistent / monolithic kernel 中。

典型场景：

- 整个 decoder layer 被拆成大量短 kernel；
- Attention、Router、MoE 等 stage 通过 persistent workgroup pipeline 执行。

优先考虑 L4 的条件：

- concurrency / batch size 较低；
- workload 更偏 latency-sensitive / high-interactivity；
- layer 内存在大量短 kernel；
- kernel launch / synchronization overhead 占比较高；
- stage 之间存在较强的串行 dependency；
- 可以通过 tile-level / stage-level pipeline overlap；
- 可以减少 intermediate HBM traffic；
- fusion 后 VGPR/LDS/occupancy degradation 可接受。

当前经验：

**MonoKernel 在低 concurrency、高交互场景下通常收益更明显。**

但不要简单使用：

低 concurrency → MonoKernel

还需要判断：

- 是否存在大量短 kernel；
- launch / sync overhead 是否真的显著；
- GEMM / Attention 是否已经占据绝大多数时间；
- whole-layer fusion 是否会导致严重 occupancy loss；
- 不同 stage 的 tile / resource requirement 是否冲突。

---

## 二、Fusion Selection 原则

对每个 candidate fusion region，评估：

### Potential Gain

- kernel launch overhead saved
- synchronization overhead saved
- intermediate HBM traffic saved
- producer-consumer overlap
- communication-compute overlap
- cache / LDS reuse opportunity
- pipeline parallelism

### Potential Cost

- fused kernel 内部额外 synchronization
- VGPR pressure
- LDS pressure
- occupancy loss
- loss of optimal tile configuration
- stage load imbalance
- code specialization complexity
- maintainability / implementation cost

可以抽象为：

Fusion Benefit =
Launch Saved
+ Sync Saved
+ HBM Saved
+ Overlap Gain
+ Locality Gain
- Resource Cost
- Occupancy Loss
- Tile Compromise
- Load Imbalance

优先选择 expected net gain 最大的 fusion pattern。

【新增】Benefit 必须**量化**：
- 每一项都要以 trace 里实测的 kernel 耗时为依据，算出每次调用和每个 step 能省多少 µs；
- 再用 workload_summary 的权重，换算成"占整段 device 时间的百分比"；
- 给出理论上限和预计值两个数，并写明从上限打折到预计值的理由。

---

## 三、分析流程

### Step 0 — 探索模型和部署【新增】

没弄清楚模型结构，就很难判断哪些 kernel 能 fuse、fuse 之后能否保持正确，所以要先做这一步。结果写到 `out/MODEL_NOTES.md`：

- 读 `model/config.json`，记下关键结构参数：
  - 层数、隐藏维度；
  - attention 类型；
  - MoE 配置：专家数、top-k、shared expert；
  - 量化方式；
  - 有没有草稿层或 MTP 层；
  - 其他特殊结构。
- 在框架源码里找到这个模型的实现：
  - 用 config 里的 `architectures` 定位到模型类；
  - 梳理一层的 forward 路径，看 attention、MLP/MoE、norm、残差，以及其他特有模块分别调用了哪些函数和算子；
  - 确认不同层之间有没有结构差异。
- 结合 `deploy/` 下的启动参数和环境变量，弄清楚：
  - 并行方式和通信点：在哪里做 all-reduce 或 all-gather，用的是哪种实现；
  - 投机解码的方式，会产生哪几类 step，每类 step 的行数和 batch 之间是什么关系；
  - prefill 怎么切 chunk，decode 是否走 CUDA/HIP graph；
  - 各个算子库开关分别把哪些算子替换成了哪种实现。
- 把模型结构和 trace 里实际出现的 kernel 对上，形成一张"模块/函数 → kernel"的对应表。后面每一步都基于这张表。

### Step 1 — Workload Characterization

首先分析 workload：

- concurrency
- batch size
- token 数
- Prefill / Decode
- TP / DP / EP / CP 配置
- average kernel duration
- kernel launches per layer
- launch gap
- synchronization gap
- HBM traffic
- communication cost
- compute utilization
- occupancy

将 workload 粗略分类为：

- Low concurrency / latency-sensitive
- Medium concurrency
- High concurrency / throughput-oriented

但不要根据 concurrency 单独决定 fusion level。

【新增】要用 workload_summary 里的分布来描述 workload：batch size 和 token 数的分布、各 phase 的 device 时间占比，以及每类 step 的 kernel 数和耗时。注意区分两部分：每层重复执行的部分，和每个 step 只执行一次的部分。

---

### Step 2 — 恢复 execution region

根据 trace 和 source/model 信息，将 kernel 分成逻辑区域，例如：

- normalization
- attention
- KV/cache
- router
- dispatch
- MoE
- quantization
- communication
- residual/update
- MTP / speculative decoding

注意：

**trace 上时间相邻，不代表一定可以融合。**

必须确认：

- tensor dependency；
- multi-consumer；
- side effect；
- KV cache mutation；
- collective communication；
- shape compatibility；
- precision requirement。

【新增】还要确认 dispatch 条件：
- size guard，以及不同 M 走的是不是不同分支；
- 是否在 graph capture 里执行。

另外要注意，region 不只存在于单层内部。下面三类也要恢复出来：
- **跨层边界**：上一层的末尾接下一层的开头；
- **层外、每个 step 只执行一次的部分**：metadata、索引、拷贝等胶水 kernel；
- **投机解码的草稿路径**。

---

### Step 3 — 生成多个 Fusion Candidate

对于每个热点 region，不要只生成一种方案。

需要同时考虑：

- L1 candidate
- L2 candidate
- L3 candidate
- L4 candidate

例如：

某一 layer：

A → B → C → D → E

可以生成：

- L1：B + C
- L2：A + B + C
- L3：A + B + C + D
- L4：整个 layer

然后比较不同 fusion scope。

不要因为 L4 可行，就直接选择 L4。

【新增】"热点"要按**加权后的占比**来看，不能只看耗时最大的几个 kernel。下面这些单个 kernel 很短、但次数很多的位置也要逐一检查：
- 连续的小 kernel；
- 一个 GEMM 或 attention 前后紧挨着的 elementwise、norm、quant、cast；
- 读同一个输入的多个 GEMM；
- 通信和它前后的计算；
- 跨层边界；
- 每个 step 都要跑的胶水；
- 草稿路径；
- 多余的 copy、fill、类型转换。

---

### Step 4 — Candidate Evaluation

对每个 candidate 输出：

- Fusion Level
- 涉及哪些 kernel / operator
- 为什么可以 fuse
- 当前主要 bottleneck
- 预计主要收益
- 预计主要风险
- VGPR / LDS / occupancy 影响
- tile / wave layout 是否需要 compromise
- producer-consumer overlap 机会
- communication overlap 机会
- 为什么这个 level 优于更小范围
- 为什么不继续扩大 fusion scope

【新增】每个 candidate 还要补充以下内容：
- 出现次数：每层、每个 step 各出现几次，覆盖哪些层；
- 适用的 shape 和 M 范围；
- launch 数从几个变成几个；
- 对应的 sglang 或算子库代码位置，写成"模块.函数"；
- 量化的收益：每个 step 省多少 µs，加权后的 e2e 百分比，分上限和预计两档；
- 实现难度，按 1–5 打分；
- 能否和原实现做到逐位一致；
- 和其他 candidate 的关系：是互斥（抢同一段 kernel），还是可以叠加。

---

### Step 5 — Ranking

按照 expected performance benefit 对 candidate 排序。

最终输出：

1. 推荐 fusion pattern
2. fallback fusion pattern
3. 明确不建议 fusion 的 region
4. 还缺哪些信息才能做最终判断

【新增】排出 **Top-20**。处理好互斥和叠加关系之后，给出推荐组合的总预计收益，注意不要重复计算。

---

## 四、当前 Fusion Heuristic

下面这些只能作为 prior，不是硬规则：

### Low concurrency

优先检查：

L4 MonoKernel

但前提是：

- kernel 数量多；
- kernel duration 短；
- launch/sync overhead 明显；
- dependency chain 强；
- resource pressure 可控。

---

### High concurrency

优先检查：

L3 Mega / MegaMoE

特别关注：

- token 数；
- expert batch size；
- GEMM utilization；
- dispatch/combine overhead；
- communication / compute overlap。

---

### Medium concurrency

这是目前需要重点探索的区域。

不要预设最佳 fusion level。

需要同时比较：

- L2
- L3
- partial L4

并通过 benchmark 决定。

---

### Many tiny kernels

优先：

L1 / L2

重点减少：

- launch overhead
- HBM round-trip

---

### Heavy GEMM / Attention dominates

优先保护：

- operator optimal tile
- occupancy
- MFMA efficiency

避免为了 fusion 牺牲主 operator 性能。

---

### Communication-heavy MoE

优先考虑：

L3

并评估：

- dispatch
- expert GEMM
- combine
- all-to-all

之间的 overlap。

---

## 五、Claude Code 的角色

Claude Code 不应该只是一个：

Fusion Level Classifier

而应该是：

**Fusion Search Agent**

它需要：

1. 从 trace 找 fusion opportunity；
2. 生成不同 fusion level 的候选；
3. 根据 source / graph 验证 dependency；
4. 给出 fusion 实现建议；
5. 根据 benchmark 结果继续扩大或缩小 fusion scope。

推荐闭环：

Trace
→ Fusion Candidate Generation
→ Dependency Validation
→ Kernel Implementation
→ Microbenchmark
→ E2E Benchmark
→ Feed Result Back
→ Update Fusion Pattern

最终目标是：

**通过 workload-aware + measurement-driven 的方式找到当前场景下最优 fusion pattern，而不是预设某个 fusion level 一定最好。**

---

## 六、输出格式

所有输出都写到 `out/` 下【新增】：
- `out/MODEL_NOTES.md`：Step 0 的结果；
- `out/REPORT.md`：用中文写，按下面的结构；
- `out/fusion_candidates.json`：全部 candidate，字段见本节末尾；
- 分析脚本放在 `out/scripts/` 下。

### Workload Characterization

- concurrency:
- batch/token:
- Prefill/Decode:
- parallelism:
- dominant bottleneck:
- average kernel duration:
- launch overhead:
- memory traffic:
- communication:
- resource utilization:

### Candidate Fusion Patterns

| Rank | Level | Fusion Region | Kernels（trace 原名） | Launch 变化 | Expected Gain（加权 e2e %，上限 / 预计） | Main Risk | Conflicts | Confidence |
|---|---|---|---|---|---|---|---|---|

### Recommended Fusion Pattern

明确给出：

- fusion boundary
- fusion level
- 选择原因
- 预计解决的 bottleneck

### Why Not Other Levels

分别说明：

- 为什么不选更小 fusion scope；
- 为什么不选更大 fusion scope。

### Required Validation

列出实施前需要确认：

- dependency
- tensor shape
- VGPR/LDS
- occupancy
- tile configuration
- communication
- correctness
- E2E performance

【新增】`out/fusion_candidates.json` 是一个数组，每个元素的字段如下：
`id, rank, level(L1-L4), name, phase, region, kernels[], launches_before, launches_after, occurrences, shape_range, code_location, seam, saved_us_per_step, e2e_gain_pct_upper, e2e_gain_pct_expected, difficulty(1-5), bitwise_parity, risk, conflicts_with[], why_not_smaller, why_not_larger, evidence, notes, recommended(bool)`

【新增】第二阶段会直接用 `recommended=true` 的那组 candidate 来写 kernel，并起服务做 A/B，所以这组 candidate 必须**两两不互斥、可以同时生效**，并且**只能是 L1 / L2**（见【分析原则】第 4 条）。每个 candidate 的 `seam` 字段写替换点：在哪个模块的哪个函数入口把原来的调用换成 fused kernel，要具体到能写 monkeypatch overlay 的程度。

【新增】限制：
- 只能读本目录和 /sgl-workspace，不要访问其他路径；
- 不要修改源码目录；
- 时间预算：约 2 小时。
 
【分析原则】

底线（必须满足，不满足的不能进推荐）：
1. 有证据：kernel 归属哪段代码、seam 在哪、"不可实现"、"已经融合到上限"等判断，都要有可复查的依据（trace 调用栈、实际读过的源码位置、在当前环境实际跑过的命令和输出）。拿不出证据的判断，不能作为推荐或放弃的理由。
2. 以实际服务为准：权重、形状（M 的分布）、走哪个平台分支、调用哪个 kernel，都以本次部署实际运行的情况为准，不以默认值、其他平台或函数定义处为准。
3. 以目标指标为准：按对本次目标指标的预计收益排序；不影响目标指标的候选往后排。
4. 实现范围：L3/L4 照实排序，但标注「暂不支持」（candidate 的 implementable=false）；交给实现的推荐只能是 L1/L2，且彼此不冲突。报告里单列一节「本阶段实现推荐（L1 / L2）」。
5. 覆盖完整：交付一张「边界清单」，按目标指标对应的 phase 的实际 trace 序列（不能用其他 phase 的序列代替；不同层走不同路径时每种路径都要列）列出每层重复的 kernel 序列和每个 step 头尾只跑一次的 kernel 序列，对其中每个相邻边界写明「列成了哪个候选」或「不融合的原因」。清单里没覆盖到的边界视为没有分析，推荐之前必须补齐。

方法（提炼）：
6. 先穷举再取舍：每个相邻边界的两个融合方向都列成候选，比较之后再选，不能只看到一个方向就停下。每个推荐都给一个范围更小的 fallback。
7. 算清收益和代价：融合的收益来自减少 launch 和少一次显存往返（CUDA/HIP graph replay 只省掉 CPU 侧的 launch 开销，GPU 侧每个 kernel 的固定开销和 kernel 之间的显存往返仍然存在，所以在 graph 里跑的区域，减少 kernel 数同样有效）；代价来自丢掉并行度（如 split-K/KV）、需要替换重 kernel（GEMM、库算子）、需要跨模块接入。比较时两边都要在实际服务的形状下估算。收益用 trace 实测来算：融合后去掉的 kernel 的实测时长（含每个 kernel 的固定开销）× 每个 step 的出现次数 × 该 phase 的权重；每层重复的路径会乘上层数，通常是收益最大的地方，推荐要先保证把这些大头找全、排在前面。
8. 障碍不等于放弃：遇到"需要全局同步""已经是融合 kernel""跨了多个模块"时，先找绕过的办法（冗余计算、找共同的上层 seam、吸收进已有融合 kernel 的 prologue/epilogue），都不行才判为不可实现，并写明原因。
