# GEAK · 可切换 code-agent 后端(适配 qcoder)

> 在 Cursor 里打开本文件 → 点右上角预览(`Cmd/Ctrl + Shift + V`),下面的图会直接渲染。

---

## 1. 架构总览

```mermaid
flowchart TD
    subgraph GEAK["GEAK 核心流程（一字未改）"]
        JS["kernel_workflow.js / e2e_workflow.js<br/>roles · knowledge · scripts"]
    end

    JS --> ENTRY["入口 run_e2e.py<br/>看开关 GEAK_AGENT_BACKEND"]

    ENTRY -->|"不设（默认）"| NATIVE["原生 Claude Code · Workflow 工具<br/>行为和以前完全一样"]
    ENTRY -->|"= claude / qwen"| RT["新增：独立 Node 运行时<br/>run_workflow.mjs"]

    RT --> IMPL["运行时自己实现<br/>agent() · parallel() · workflow()<br/>★ 并行 &amp; 套娃都在这里 ★"]
    IMPL --> SPAWN["每调一次 agent()：<br/>抢并发名额 → 起一个一次性进程 → 收结果"]

    SPAWN --> GEN["backends/generic.mjs + registry.json<br/>claude / qwen / codex / kimi（加一个=加配置）"]

    classDef keep fill:#0f2f18,stroke:#3fb950,color:#e6edf3;
    classDef native fill:#0d2440,stroke:#58a6ff,color:#e6edf3;
    classDef new fill:#241033,stroke:#a371f7,color:#e6edf3;
    classDef switch fill:#2b230a,stroke:#d29922,color:#e6edf3;

    class JS keep;
    class ENTRY switch;
    class NATIVE native;
    class RT,IMPL,SPAWN,GEN new;
```

**关键点**:并行和套娃由**运行时在进程级完成**,qcoder 只当"跑单活的进程"——
所以"qcoder 不能并行 / 不能嵌套"被彻底绕开。

---

## 2. 适配 qcoder:每个坑 → 对策 → 为什么

```mermaid
flowchart LR
    P1["坑1：不认识流程语言<br/>agent/parallel/workflow<br/>是 Claude Code 私有"] --> S1["写最小流程引擎<br/>run_workflow.mjs"]
    P2["坑2：并行靠不住<br/>+ 不能套娃"] --> S2["并行/套娃放进运行时<br/>进程级实现"]
    P3["坑3：不保证规整 JSON"] --> S3["schema.mjs<br/>抽取+校验+重试"]
    P4["坑4：命令行调用方式不同<br/>qwen -p --yolo · OpenAI env"] --> S4["backends/qwen.mjs<br/>唯一纯为 qcoder 写的文件"]
    P5["坑5：入口要能选 qcoder"] --> S5["run_e2e.py 加开关<br/>GEAK_AGENT_BACKEND"]

    classDef pain fill:#3a1113,stroke:#f85149,color:#e6edf3;
    classDef fix fill:#241033,stroke:#a371f7,color:#e6edf3;
    class P1,P2,P3,P4,P5 pain;
    class S1,S2,S3,S4,S5 fix;
```

| # | 为什么这么改 |
|---|---|
| 1 | 硬门槛:文件先跑得起来,后面才谈得上 |
| 2 | **方案关键**:不依赖 qcoder 的并行 → 你担心的问题直接绕开 |
| 3 | 补上缺的格式保证(**最大风险点**,得真跑才知道听不听话) |
| 4 | qcoder 特有细节关在一处;换别的 agent 只加同类小文件 |
| 5 | 一个环境变量切换,且**默认行为完全不变** |

---

## 3. 文件清单

```mermaid
flowchart TD
    ROOT["改动"] --> NEW["新增 interface/runtime/"]
    ROOT --> EDIT["改动（3 个）"]
    ROOT --> KEEP["没动"]

    NEW --> N1["run_workflow.mjs<br/>流程引擎+并行+套娃+配置选择+指标"]
    NEW --> N2["config.mjs + registry.json<br/>agent×model 配置解析 + prompt 消毒"]
    NEW --> N3["backends/base.mjs + generic.mjs<br/>通用后端(加 CLI=加配置)"]
    NEW --> N4["schema.mjs<br/>结构化输出模拟"]
    NEW --> N5["experiment.mjs<br/>(agent×model) 对照实验 runner"]
    NEW --> N6["selftest.mjs · 自测 36 项"]

    EDIT --> E1["run_e2e.py · 加 profile/agent/model 选择"]
    EDIT --> E2["README.md / run_e2e.md · 说明"]

    KEEP --> K1["*_workflow.js · roles<br/>knowledge · scripts<br/>核心资产一字未改"]

    classDef new fill:#241033,stroke:#a371f7,color:#e6edf3;
    classDef edit fill:#0d2440,stroke:#58a6ff,color:#e6edf3;
    classDef keep fill:#0f2f18,stroke:#3fb950,color:#e6edf3;
    class NEW,N1,N2,N3,N4,N5,N6 new;
    class EDIT,E1,E2 edit;
    class KEEP,K1 keep;
```

---

## 4. 改造前 vs 改造后

| 维度 | 改造前 | 改造后 |
|---|---|---|
| 能用的 agent | 只有 Claude Code(焊死) | claude / qcoder **可切换** |
| 换 agent 成本 | 做不到 | **改一个环境变量** |
| 并行/套娃 谁负责 | Claude Code | **运行时**(不依赖 agent) |
| GEAK 核心流程 | — | **一字未改** |
| 默认行为 | Claude Code | **不变**(不设开关=老路) |

---

## 5. 用法

```bash
# 用 qcoder 跑 e2e（不设 = 原生 Claude，行为不变）
export GEAK_AGENT_PROFILE=qwen         # 或 GEAK_AGENT_BACKEND=codex / GEAK_MODEL=...
npm i -g @qwen-code/qwen-code          # 需 Node ≥ 18 + 可达端点
python interface/run_e2e.py handoff.json result.json

# 单内核直接用运行时
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --profile qwen --args '{...}'

# (agent × model) 对照实验
node interface/runtime/experiment.mjs --script kernel_workflow/kernel_workflow.js \
  --agents claude,qwen,codex --models default --repeats 3 --args '{...}'

# 逻辑自测（无需 GPU/网络，36 项）
node interface/runtime/selftest.mjs
```

> 一句话:没有改 GEAK 去迁就任何 CLI,而是**补齐它们缺的确定性编排能力**,再用配置把它们接进来。
> 详见 `DEV_SUMMARY.md`。
