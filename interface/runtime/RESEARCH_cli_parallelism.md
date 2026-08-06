# 调研:qcoder / Codex / Kimi CLI 的并行 subagent 能力

> 调研日期:2026-07-30 · 目的:确认换 code-agent 后端时,能否依赖各 CLI 自带的并行/嵌套能力
> 结论先行:**三家都"会"并行 subagent,但都是「模型驱动、不可复现、嵌套受限」,没有一个具备 Claude Code
> `Workflow` 那种「代码可编程」的确定性编排原语。因此 GEAK 的确定性并行/嵌套必须由外部 runtime 提供,
> 不能依赖 CLI 本身。**

---

## 对比表

| CLI | 有并行 subagent? | 怎么触发 | 嵌套 | Claude 式 Workflow 编程原语 |
|---|---|---|---|---|
| **qcoder (qwen-code)** | ✅ SubAgents / Agent Teams | **模型驱动**(LLM 自行 fork);Agent Team = 持久化 teammate 互发消息 | **fork 不能嵌套**(运行时强制报错) | ❌ 无(宣传有 "dynamic workflows",但无 parallel/pipeline 代码原语) |
| **Codex CLI (OpenAI)** | ✅ 2026-03-16 上线,**最多 6 并发**;角色 explorer/worker/default | **模型/提示驱动**:"不会自动开,只有你明确要求时才开" | 共享 workspace,靠 git worktree 隔离;无确定性嵌套 | ❌ 无。确定性规模化编排,OpenAI 让你用**外部编排器 Symphony** |
| **Kimi CLI (Moonshot)** | ✅ fixed + dynamic subagent,经 Task 工具并行派发 | **模型驱动**(主 agent 用 Task 工具决定) | 各 subagent 独立 runtime;嵌套无明确保证 | ❌ 无(模型级 Agent Swarm 可达 100 个,但那是模型自主,非代码编排) |

## 关键区别(对 GEAK 才是重点)

- **它们的并行 = 模型驱动**:自然语言让 LLM 临时决定开几个 —— 不确定、不可复现。
- **GEAK 需要的 = 确定性代码编排**:第几轮开哪些角色、每个 patch 独立验证、e2e 递归调 kernel,全写死在 JS,可复现。
- **没有一个** CLI 有 `parallel()/pipeline()/workflow()` 这类编程原语。
- 嵌套:qcoder 明确不行;codex/kimi 不保证。而 e2e 必须递归嵌套。

## 最有力佐证

OpenAI 自己在需要确定性/规模化编排时,不是用 codex 内置 subagent,而是推出**外部编排器 Symphony** 驱动 codex ——
这恰恰就是本项目的架构:**把确定性编排放在 CLI 外面(`interface/runtime/`),CLI 只跑单个任务。**

## 对本项目设计的影响

1. "能不能并行"不是问题 —— 三家都能,但会的是"错的那种"(模型驱动 / 不可复现 / 嵌套受限)。
2. **不能依赖** CLI 的并行/嵌套,否则丢掉 GEAK 的确定性与 e2e 递归。
3. 正确做法:runtime 做确定性并行/嵌套(起 N 个独立 CLI 进程),CLI 当"跑单活的进程",与各家 subagent 机制完全解耦(我们不碰)。

## 来源

- Codex Subagents — OpenAI Developers: https://developers.openai.com/codex/subagents
- Codex 多-agent 编排讨论 #3898: https://github.com/openai/codex/discussions/3898
- Codex Gets Subagents (Rick Hightower, Medium): https://medium.com/@richardhightower/codex-gets-subagents-the-parallel-ai-coding-pattern-is-now-industry-standard-how-does-it-stack-35bd217ef11f
- Kimi CLI Subagent System — DeepWiki: https://deepwiki.com/MoonshotAI/kimi-cli/5.3-multi-agent-coordination
- Kimi Code CLI — GitHub: https://github.com/MoonshotAI/kimi-code
- Qwen Code — GitHub: https://github.com/QwenLM/qwen-code
- Qwen Code Subagents 文档: https://qwenlm.github.io/qwen-code-docs/en/users/features/sub-agents/
