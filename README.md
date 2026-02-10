# Mini SWE Agent

基于 LLM 驱动 Bash 命令的极简 AI 编码智能体，核心代码约 100 行。

## 安装

```bash
pip install -e .

# 如需使用 MCP RAG 功能，额外安装 langchain 依赖
pip install -e '.[langchain]'
```

## 使用

```bash
# REPL 交互界面
mini

# 直接指定任务
mini -t "修复 main.py 中的 bug"

# 自动执行模式（跳过确认）
mini --yolo

# 启用 MCP
mini --mcp
```

## MCP 集成

集成 AMD AI DevTool，提供基于知识库的混合检索能力（BGE Embedding + BM25 + 重排序）。内置 AMD GPU 和 NVIDIA GPU 知识库。

### 1. 预下载 ROCm 库源码（推荐）

agent 运行时可能需要参考 ROCm 库的源码，建议提前 clone 到本地，避免 agent 运行时下载大仓库导致超时中断：

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

### 2. 构建语义索引（首次使用必须）

使用 MCP 前需要先构建知识库索引，否则检索功能无法工作：

```bash
# 基本用法：对 knowledge-base/ 下所有文档构建索引
# 强制重建（覆盖已有索引）
python scripts/build_index.py --force

```

索引默认输出到 `~/.cache/amd-ai-devtool/semantic-index/`，构建产物：

- `index.faiss` + `index.pkl` — FAISS 语义搜索索引
- `bm25_index.pkl` — BM25 关键词搜索索引

以下情况需要重建索引：

1. 添加/修改知识库文档
2. 更改分块或索引逻辑
3. 修复元数据解析 bug

### 3. 测试检索

构建索引后可运行测试脚本验证：

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

配置文件：`src/minisweagent/config/rag_config.yaml`，可调整检索参数、是否启用 BM25 双路召回、重排序、LLM 总结等。

## 项目结构

```
src/minisweagent/
├── __init__.py                # 版本号、协议定义、全局配置
├── agents/                    # 智能体实现
│   ├── default.py             #   核心智能体（~100 行）
│   ├── interactive.py         #   人机交互智能体
│   └── interactive_textual.py #   Textual TUI 智能体
├── models/                    # LLM 模型接口
│   ├── litellm_model.py       #   LiteLLM（支持大多数模型）
│   ├── anthropic_model.py     #   Anthropic
│   ├── amd_llm.py             #   AMD LLM 网关
│   ├── openrouter_model.py    #   OpenRouter
│   └── portkey_model.py       #   Portkey
├── environments/              # 执行环境
│   ├── local.py               #   本地 subprocess
│   ├── docker.py              #   Docker/Podman
│   └── singularity.py         #   Singularity/Apptainer
├── config/                    # YAML 配置文件
│   ├── mini.yaml              #   mini 命令默认配置
│   ├── default.yaml           #   DefaultAgent 默认配置
│   ├── github_issue.yaml      #   GitHub Issue 解决配置
│   └── rag_config.yaml        #   RAG 检索配置
├── run/                       # 入口脚本
│   ├── mini.py                #   主 CLI（mini 命令）
│   ├── hello_world.py         #   简单示例
│   ├── github_issue.py        #   GitHub Issue 自动解决
│   └── inspector.py           #   轨迹浏览器
├── mcp_integration/           # MCP（AMD AI DevTool）集成
│   ├── mcp_environment.py     #   MCP 环境封装
│   ├── langchain_retrieval.py #   混合检索（Embedding + BM25）
│   └── prompts.py             #   MCP 专用提示词
└── utils/                     # 工具函数
    ├── log.py                 #   日志
    └── subagent.py            #   子智能体
```

其他顶层目录：

- `scripts/` — 辅助脚本
- `knowledge-base/` — RAG 知识库（AMD / NVIDIA）

## 配置文件

所有配置文件位于 `src/minisweagent/config/`，通过 `mini -c <配置名>` 指定使用。

### 智能体配置

| 文件 | 用途 | 模型 | 模式 | 说明 |
|------|------|------|------|------|
| `mini.yaml` | `mini` 命令默认配置 | AMD LLM 网关 claude-opus-4.5 | yolo | 日常使用的主配置，temperature=0.0，输出截断 20000 字符，timeout 3600s |
| `default.yaml` | DefaultAgent 基础配置 | 不绑定具体模型 | confirm | 通用基础配置，temperature=0.0，输出截断 10000 字符（头尾各 5000） |
| `mini_no_temp.yaml` | 无 temperature 版本 | 不绑定具体模型 | confirm | 和 default.yaml 基本一致，但不设 temperature，cost_limit=3 |
| `mini_reverse_kl.yaml` | GPU kernel 优化分析 | AMD LLM 网关 claude-opus-4.5 | confirm | 专用于分析仓库的 kernel 优化历史并生成报告，prompt 较长 |
| `github_issue.yaml` | 自动解决 GitHub Issue | 不绑定具体模型 | — | 运行在 Docker 容器中（python:3.11，工作目录 /testbed） |

### RAG 配置

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
