# GEAK × 多 code-agent 后端:兼容性深挖 + 待调研清单

> 方法:通读 `kernel_workflow/` + `e2e_workflow/` 的 roles / knowledge / *.js,找出对 Claude Code 的隐性依赖。
> 结论先行:**兼容面比预想的小且干净** —— 角色都是单 agent、只用 Read/Write/Bash、不联网、不自己 spawn 子 agent、
> 不引用 MCP、无硬编码模型。真正的风险集中在 3 处:结构化输出、各 CLI 的 headless 行为、沙箱/权限。

---

## 零、已实测验证(2026-07-31,SaFE 网关 + qcoder + 我的 runtime)

**qcoder(qwen-code)整条链已跑通**(登录节点,无 GPU 的纯 LLM 验证):
`run_workflow.mjs → registry/generic 后端 → qwen -p → SaFE 网关(OpenAI 端点)→ claude-opus-4-8 → JSON 抽取 → schema 校验`,
`ok=true, schema_failures=0`。profile `qwen-opus48` 亦通。

已解决的真实配置项(已固化进 registry.json / env 脚本):
- **R1 结构化输出**:claude-opus-4-8 简单 schema 首测即过,0 失败(复杂 schema 仍需在真任务上看)。
- **R3 权限**:qwen 非交互需 `~/.qwen/settings.json` = `{"security":{"auth":{"selectedType":"openai"}}}`;`--yolo` 无 sandbox 的告警用 `QWEN_CODE_SUPPRESS_YOLO_WARNING=1` 静音(已入 registry qwen.env)。
- **R6 认证**:网关有 OpenAI 兼容端点 `…/llm-proxy/v1`,SaFE `ak-` key(= 环境里的 `ANTHROPIC_API_KEY`)对 OpenAI/Anthropic 两种端点通用;TLS 需 `NODE_EXTRA_CA_CERTS=/shared_nfs/hyperloom/ca/amd-ca-combined.pem`。可用模型:claude-opus-4-8/opus-5、gpt-5.4/5.6 等。

**工具位置(共享盘,容器可直接用)**:node v22 `/shared_nfs/hongtaom/tools/node`;qcoder `/shared_nfs/hongtaom/tools/npm-global/bin/qwen`(v0.21.2);一键环境 `. /shared_nfs/hongtaom/tools/env.safe-gateway.sh`。

**待实测(需 GPU + ROCm 容器)**:R2/R4(真任务里 headless 输出完整性、单命令超时)、knn 端到端(hipcc 编译 + GPU 实测)、codex/kimi 的 bring-up、parity 对比。

## 一、已确认「不构成障碍」(实测 grep)

| 检查项 | 结果 |
|---|---|
| 角色是否自己 spawn 子 agent(Task/subagent) | ❌ 不会。编排全在 JS,角色是单 agent → **每个 CLI 只需支持单 agent 工具调用,不用它的 subagent 功能** |
| 用到的工具 | 只有 **Read / Write / Bash**(42/25/24 次);"edit" 都是散文词,非 Edit 工具 |
| 联网工具(WebFetch/WebSearch) | ❌ 不依赖 |
| MCP 工具 | ❌ 不用 |
| 硬编码模型(Opus/Claude/GPT) | ❌ 无(仅出现被优化的目标模型如 gpt-oss-120b) |
| Claude 专有词(ultracode/effort/background task) | ❌ roles/knowledge 里没有(只在 run_e2e.py 原生路径) |
| 预算(budget/token) | 轮数+时间在 JS 里控制,与 CLI 无关;无 token 记账 |

→ 意味着:**换任意 CLI,只要它能"非交互跑一个带 Read/Write/Bash 的任务到结束并打印结果"就够了。**

## 二、待调研 / 待兼容(每个 CLI 都要逐项确认)

| # | 风险项 | 为什么重要 | 每 CLI 要查什么 |
|---|---|---|---|
| R1 | **结构化输出可靠性** | 几乎**所有** agent() 调用都带 schema,靠返回规整 JSON。Claude 有强制 StructuredOutput 工具,其它没有 | 有没有**原生 JSON/schema 输出模式**(有就用,远比"请输出json"稳);没有则靠 `schema.mjs` 抽取+重试,需实测失败率 |
| R2 | **headless 一次性模式 + 输出格式** | runtime 把 prompt 喂进去、从 stdout 取最终结果 | 确切命令(`qwen -p` / `codex exec` / `kimi ?`);能否跑完整 agentic 循环并退出;最终文本是否干净打印到 stdout;有无"只输出最终消息"选项 |
| R3 | **自动批准 + 沙箱** | 角色要写 exp 目录(常在 cwd 外)、跑 rocprof/hipcc/git、可能联网 pip | 自动批准全部工具的 flag;**关闭/放宽沙箱**(codex 默认沙箱禁 cwd 外写和联网,需 `--dangerously-bypass-approvals-and-sandbox` 或 workspace-write+network) |
| R4 | **单条命令超时** | 一次 build/bench/rocprof 可能几分钟~小时 | CLI 对单个 Bash 命令有没有内置超时;能否调大/取消 |
| R5 | **模型上下文窗口** | 最大单 prompt = role(最大 63KB kernel_extractor)+ knowledge + 源码,约 ≥16K token 起 | 所选模型 ctx 是否够(Qwen3-Coder 256K/codex 大/Kimi 256K 一般都够,小模型要注意) |
| R6 | **provider 认证 / 端点** | 要指到具体模型 | OpenAI 兼容 base_url/key 的 env 名(做"同模型不同 CLI"对照时都指到同一网关) |
| R7 | **cwd / 绝对路径语义** | 角色用绝对路径做 FS 活 | CLI 是否尊重 cwd、能否操作 cwd 外的绝对路径(与 R3 沙箱相关) |

## 三、Claude 专有措辞 —— 建议 runtime 层"消毒"(不改 roles/.js)

角色 prompt 和 `roleAgent()` 基座里有几处写死了 Claude 语义,对别的 CLI 是误导:

- `"a StructuredOutput tool is forced"`(`kernel_workflow.js` 的 roleAgent 基座 + `director.md` 等多个 role):
  非 Claude 后端没有这个工具。`schema.mjs` 已追加正确指令,但这句仍在。
  **建议**:runtime 在 backend≠claude 时,对最终 prompt 做一次替换(把这句换成"以单个 ```json 代码块返回"),纯字符串处理,不动源文件。
- `"Invoke the Workflow tool"`:仅在 `run_e2e.py` 的原生 build_prompt 里,runtime 路径不走它 → 无需处理。

## 四、建议的 bring-up 顺序(每个新 CLI 都照做)

1. `<cli> --help` → 定死 R2(headless 命令)、R3(自动批准+沙箱 flag)、R6(认证 env),写进 `backends/<cli>.mjs`。
2. 跑 `selftest.mjs`(假后端)确认 runtime 本身 OK(已具备)。
3. **冒烟单任务**:用该 CLI 跑一个最小 schema agent(比如 director:setup),重点看 R1(能不能拿到合法 JSON)。
4. 端到端:`kernel_workflow` @ `examples/tasks/knn`,统计 R1 失败率、R4 有没有被超时截断。
5. 与 claude 后端做 parity 对比(见 RESEARCH/OVERVIEW)。

## 五、优先级

- **最高**:R1(结构化输出)+ R3(沙箱/权限)—— 这两个不解决,任何 CLI 都跑不完一轮。
- **高**:R2(headless 输出格式)、R4(单命令超时)。
- **中**:R5/R6/R7 —— 一般够用,配置层能覆盖。

> 关联文档:`RESEARCH_cli_parallelism.md`(并行能力调研)、`OVERVIEW.md`(架构)。
