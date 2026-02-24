# GEAK-v3

[English](README.md) | 中文

GEAK 是一个基于 mini-SWE-agent 构建的 AI 驱动 GPU 内核自动优化框架。

它实现了系统化、基于性能剖析的可扩展 GPU 内核优化——从单内核调优（v1/v2）演进到仓库级别的自主优化（v3）。

**v3 同时集成了 AMD AI DevTool（MCP），提供混合知识库检索能力**，将 AMD/NVIDIA GPU 知识库直接融入智能体的上下文中。

## 目录

- [演进：从内核级到仓库级自动化](#演进从内核级到仓库级自动化)
- [核心架构](#核心架构)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [使用](#使用)
  - [配置](#配置)
  - [输出产物](#输出产物)
- [MCP 集成（AMD AI DevTool）](#mcp-集成amd-ai-devtool)
- [功能特性](#功能特性)
  - [单元测试发现](#单元测试发现)
  - [内置系统工具](#内置系统工具)
  - [最优 Patch 选择](#最优-patch-选择)
- [知识库](#知识库)
- [项目结构](#项目结构)
- [总结](#总结)

---

## 演进：从内核级到仓库级自动化

### GEAK v1 / v2 — 单内核优化

早期版本专注于通过迭代式 patch 生成和性能验证来优化单个 GPU 内核。

它们证明了基于 LLM 的智能体可以：

- 分析内核结构
- 提出优化策略
- 生成能提升性能的 patch

### GEAK v3 — 自主仓库级优化

GEAK v3 将系统升级为全生命周期的 GPU 优化框架。

在仓库级别自动执行：

- 🔍 测试发现与生成
- 📊 基线性能测量
- 🧠 基于性能剖析的瓶颈诊断
- 🎯 策略规划与执行
- ✅ Patch 验证与回归测试
- 🔁 多轮迭代改进

系统形成闭环优化引擎，以最少的人工干预实现持续的性能提升。

---

## 核心架构

### 端到端优化引擎

GEAK 运行一个全自主的优化循环：

**测试检测 → 基线测量 → 性能剖析 → 策略规划 → （Patch 生成 → 验证）× N → 最优内核**

每个优化步骤都经过：

- 正确性验证
- 性能测量
- 版本跟踪

### 工具增强的智能层

GEAK v3 引入了结构化的工具生态系统：

- **性能剖析**：量化识别瓶颈（内存带宽、占用率、寄存器压力、执行停顿）
- **优化策略管理**：跟踪已探索的优化技术，标记成功/失败策略，优先处理高影响方向
- **版本与 Patch 管理**：自动 diff 跟踪、基准历史、回归检测、最优 patch 选择
- **MCP RAG 检索**：优化过程中按需检索 AMD/NVIDIA GPU 知识

### 并行探索与扩展

GEAK v3 支持并行优化智能体。并行扩展可以：

- 提高优化上限
- 增强探索鲁棒性
- 减少对单一优化轨迹的依赖

---

## 快速开始

### 安装

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
git switch -c dev origin/dev
pip install -e .

# 如需使用 MCP RAG 功能，额外安装 langchain 依赖
pip install -e '.[langchain]'

# 设置 LLM API 密钥
export AMD_LLM_API_KEY="YOUR_KEY"
```

### 使用

#### 交互式 REPL（mini-swe-agent 模式）

```bash
# REPL 交互界面
mini

# 直接指定任务
mini -t "修复 main.py 中的 bug"

# 自动执行模式（跳过确认）
mini --yolo

# 启用 MCP 知识库检索
mini --mcp
```

#### 基础单 agent GPU 内核优化

添加 `--yolo` 可端到端运行，无需交互确认。

```bash
mini --config geak.yaml \
  --task "优化 src/kernel.cpp 中的内核" \
  --yolo
```

#### 并行优化（多 agent + 最优 patch 选择）

- 每个 agent 在独立的 git 工作区中运行
- Patch 和测试结果分别保存
- 所有运行完成后，GEAK 根据指定指标自动选择最优 patch

```bash
mini --config geak.yaml \
  --num-parallel 4 \
  --repo /path/to/kernel/repo \
  --task "优化 block_reduce 内核" \
  --gpu-ids 0,1,2,3 \
  --metric "提取 Bandwidth（GB/s），越高越好" \
  --yolo
```

**参数说明：**

- `--num-parallel`：优化 agent 数量
- `--repo`：当 `--num-parallel > 1` 时必需（每个 agent 使用独立的 git worktree）
- `--gpu-ids`：逗号分隔的 GPU ID
- `--metric`：自然语言描述，用于从测试日志中提取/比较指标
- `--yolo`：端到端运行，无需交互确认

### 配置

`mini` 按层级加载配置：

1. 基础配置：`mini.yaml`
2. 模板：`mini_kernel_strategy_list.yaml`（默认）
3. 用户覆盖：`--config geak.yaml`（**最终覆盖**）

所有配置文件位于 `src/minisweagent/config/`，通过 `mini -c <配置名>` 指定使用。

#### 智能体配置

| 文件 | 用途 | 模型 | 模式 | 说明 |
|------|------|------|------|------|
| `mini.yaml` | `mini` 命令默认配置 | AMD LLM 网关 claude-opus-4.5 | yolo | 日常使用的主配置，temperature=0.0，输出截断 20000 字符，timeout 3600s |
| `default.yaml` | DefaultAgent 基础配置 | 不绑定具体模型 | confirm | 通用基础配置，temperature=0.0，输出截断 10000 字符（头尾各 5000） |
| `mini_no_temp.yaml` | 无 temperature 版本 | 不绑定具体模型 | confirm | 和 default.yaml 基本一致，但不设 temperature，cost_limit=3 |
| `mini_reverse_kl.yaml` | GPU kernel 优化分析 | AMD LLM 网关 claude-opus-4.5 | confirm | 专用于分析仓库的 kernel 优化历史并生成报告，prompt 较长 |
| `github_issue.yaml` | 自动解决 GitHub Issue | 不绑定具体模型 | — | 运行在 Docker 容器中（python:3.11，工作目录 /testbed） |

#### RAG 配置

文件：`rag_config.yaml`，控制 RAG 检索管道参数：

| 配置项 | 说明 |
|--------|------|
| `retrieval.embed_top_k` / `bm25_top_k` | Embedding / BM25 检索候选数 |
| `retrieval.enable_bm25` | 是否启用 BM25 双路召回 |
| `retrieval.mcp_top_k` | 最终返回结果数 |
| `reranker.enable_reranker` | 是否启用精排 |
| `fusion.semantic_weight` / `bm25_weight` | Embedding 和 BM25 的融合权重 |
| `summary.enable_rag_subagent` | 是否启用 LLM 总结 |
| `debug.verbose` | 是否打印 MCP 工具详细日志 |

### 输出产物

GEAK 保存 patch 和测试日志，结果可复现。

- **默认输出目录**：`optimization_logs/`
- **自动生成运行目录**：`optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`
- **并行运行**：子目录 `parallel_0/`、`parallel_1/`...

典型目录结构（并行运行）：

```bash
optimization_logs/<kernel>_<timestamp>/
├── parallel_0/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── agent_0.log
├── parallel_1/
│   └── ...
├── best_results.json
└── select_agent.log
```

---

## MCP 集成（AMD AI DevTool）

集成 AMD AI DevTool，提供基于知识库的混合检索能力（BGE Embedding + BM25 + 重排序）。内置 AMD GPU 和 NVIDIA GPU 知识库。

### 1. 预下载 ROCm 库源码（推荐）

agent 运行时可能需要参考 ROCm 库的源码，建议提前 clone 到本地，避免运行时下载大仓库导致超时：

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

### 2. 构建语义索引（首次使用必须）

```bash
# 对 knowledge-base/ 下所有文档构建索引
python scripts/build_index.py --force
```

索引默认输出到 `~/.cache/amd-ai-devtool/semantic-index/`：

- `index.faiss` + `index.pkl` — FAISS 语义搜索索引
- `bm25_index.pkl` — BM25 关键词搜索索引

以下情况需要重建索引：添加/修改知识库文档、更改索引逻辑。

### 3. 测试检索

```bash
python scripts/test_embedding_search.py      # 测试 FAISS 语义搜索
python scripts/test_hybrid_retrieval.py      # 测试混合检索（Embedding + BM25 + Reranker）
python scripts/test_rrf_fusion.py            # 测试 RRF 融合算法
```

### 4. 启用 MCP

```bash
mini --mcp        # 启用 MCP
mini --mcp -d     # 启用 MCP + 调试输出
```

在智能体中通过 `@amd:查询内容` 调用检索。

### 5. RAG 检索架构

```
Semantic + BM25 → RRF 融合去重 → BGE Reranker 精排 → Top K
```

- **Embedding**: BAAI/bge-large-en-v1.5（语义召回）
- **BM25**: 关键词召回
- **Fusion**: RRF (Reciprocal Rank Fusion) 融合去重
- **Reranker**: BAAI/bge-reranker-large（精排）

配置文件：`src/minisweagent/config/rag_config.yaml`

---

## 功能特性

### 单元测试发现

传入 `--create-test`，或**不提供** `--test-command` 时，GEAK 会运行 **UnitTestAgent** 尝试发现或创建测试：

```bash
mini --config geak.yaml \
  --repo /path/to/kernel/repo \
  --create-test \
  --task "优化 device_batch_memcpy 内核"
```

### 内置系统工具

| 工具 | 用途 | 关键输出 |
| --- | --- | --- |
| `profiling` | 性能剖析，识别瓶颈 | rocprofiler-compute 摘要 |
| `strategy_manager` | 跟踪优化策略 | `.optimization_strategies.md` |
| `test_perf` | 保存 patch 并运行 test_command | `patch_N.patch`、`patch_N_test.txt` |

### 最优 Patch 选择

并行运行完成后，GEAK 运行选择智能体，读取所有测试日志，提取指标，输出 `best_results.json` + `select_agent.log`。

---

## 知识库

### 目录结构

```
knowledge-base/
├── amd-knowledge-base/
│   ├── layer-1-hardware/         # 硬件架构
│   ├── layer-2-compute-stack/    # 计算栈（HIP、ROCm）
│   ├── layer-3-libraries/        # 库（rocBLAS、MIOpen 等）
│   ├── layer-4-frameworks/       # 框架（PyTorch、TensorFlow）
│   ├── layer-5-llm/              # LLM 相关
│   ├── layer-6-extended/         # 扩展知识
│   └── best-practices/           # 最佳实践
├── nvidia-knowledge-base/        # 同样层级结构
├── comparisons/                  # 跨平台对比文档
└── INDEX.md
```

### 添加新文档

1. **位置**：文件放在对应分类子目录下（如 `layer-6-extended/optimize-guides/*.md`）
2. **格式**：所有 `.md` 文件必须包含 YAML frontmatter：
   ```yaml
   ---
   tags: ["category1", "category2"]   # 必需
   priority: "L1-important"           # 必需
   source_url: "https://..."          # 必需
   rocm_version: "6.0+"              # 必需
   last_updated: 2026-01-14           # 必需
   ---
   ```
3. **文件名**：英文，反映内容（如 `bf16-vector-load-store.md`）
4. **质量**：800-1200 字，每个文档至少 2 个语法正确的代码示例
5. **添加后必须重建索引**：`python scripts/build_index.py --force`

---

## 项目结构

```
src/minisweagent/
├── agents/                    # 智能体实现
│   ├── default.py             #   核心智能体
│   ├── interactive.py         #   人机交互智能体
│   ├── parallel_agent.py      #   并行多 agent
│   ├── strategy_interactive.py#   策略引导智能体
│   └── unit_test_agent.py     #   单元测试发现智能体
├── models/                    # LLM 模型接口
│   ├── amd_llm.py             #   AMD LLM 网关（路由器）
│   ├── amd_base.py            #   AMD 基础模型
│   ├── amd_claude.py          #   通过 AMD 网关调用 Claude
│   └── litellm_model.py       #   LiteLLM（多供应商）
├── mcp_integration/           # MCP（AMD AI DevTool）集成
│   ├── mcp_environment.py     #   MCP 环境封装
│   ├── langchain_retrieval.py #   混合检索（Embedding + BM25）
│   └── prompts.py             #   MCP 专用提示词
├── tools/                     # 工具实现
│   ├── tools.json             #   工具 Schema 定义
│   ├── tools_runtime.py       #   工具运行时
│   ├── editor_tool.py         #   文件编辑器
│   ├── profiling_tools.py     #   GPU 性能剖析
│   └── strategy_manager.py    #   策略管理器
├── config/                    # YAML 配置文件
│   ├── mini.yaml
│   ├── default.yaml
│   ├── github_issue.yaml
│   └── rag_config.yaml
└── run/                       # 入口脚本
    ├── mini.py                #   主 CLI（mini 命令）
    └── utils/
```

其他顶层目录：
- `scripts/` — 索引构建与检索测试脚本
- `knowledge-base/` — RAG 知识库（AMD / NVIDIA）
- `examples/` — HIP 内核示例与子智能体示例

---

## 总结

GEAK v3 实现了仓库级别的可复现、可度量、可扩展的 GPU 内核优化。集成了：

- **性能剖析** + **策略管理** + **并行探索** 实现自主优化
- **MCP RAG 检索** 利用 AMD/NVIDIA 知识库辅助决策

欢迎贡献、实验和反馈。
