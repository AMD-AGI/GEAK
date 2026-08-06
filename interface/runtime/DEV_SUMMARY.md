# GEAK 可切换 code-agent 后端 —— 开发总结

> 目标:让 GEAK 的两个 workflow 能换用不同 code-agent CLI(claude / qwen(qcoder) / codex / kimi),
> 后端可配置、可 pin,并**内建 (agent × model) 对照实验能力**。核心 workflow / roles / knowledge / scripts
> **一字未改**。日期:2026-07-30。

---

## 1. 为什么这么做(背景)

GEAK 的编排逻辑写在 `kernel_workflow.js` / `e2e_workflow.js`,依赖 Claude Code `Workflow` 工具私有的全局原语
(`agent/parallel/pipeline/phase/log/workflow/args/budget`),别的 CLI 跑不了。调研(见
`RESEARCH_cli_parallelism.md`)确认:qcoder/codex/kimi 都有 subagent 并行,但**都是模型驱动、不可复现、嵌套受限**,
没有一个具备确定性代码编排。因此唯一干净的做法是**把编排层外置成一个独立 Node runtime**,自己实现并行/嵌套,
把每次 `agent()` 落到一个**一次性 CLI 子进程**。这样并行与嵌套由 runtime 负责,CLI 支不支持都无所谓
(与 OpenAI 自研外部编排器 Symphony 同思路)。

## 2. 架构

```
run_e2e.py ──(GEAK_AGENT_PROFILE/-BACKEND/-MODEL)──► run_workflow.mjs(独立 runtime)
                                                          │  读 registry.json 解析 (agent, model)
                     加载 e2e/kernel .js(export 剥离 + AsyncFunction),注入 Workflow 全局
                                                          │
        脚本每调一次 agent() ─► prompt 消毒 ─► 抢并发名额 ─► generic 后端起一个一次性 CLI 进程 ─► 收结果
                    parallel()=barrier 并发   pipeline()=逐项无 barrier   workflow()=一层嵌套
                                                          │
                                        schema:抽 JSON + 校验 + 重试 + 降级 null
```

- **两轴解耦**:`agent`(CLI 怎么驱动)与 `model`(端点)在 `registry.json` 里分开;`profile` = 命名的 (agent, model) 组合。
- **并行/嵌套在 runtime**:信号量 `min(16, CPU-2)`;`workflow()` 一层嵌套;总 agent 上限 1000。
- **结构化输出**:`schema.mjs` 模拟 Claude 强制 StructuredOutput(注入 json 契约 + 抽取 + 轻校验 + 重试)。
- **prompt 消毒**:非 claude 后端时,把 roles 里 "a StructuredOutput tool is forced" 等 Claude 专有措辞替换掉
  (不改 roles/.js,runtime 层字符串处理)。

## 3. 文件清单

| 文件 | 作用 |
|---|---|
| `interface/runtime/run_workflow.mjs` | 运行时核心:Workflow 全局 + 并发 + 嵌套 + 配置选择 + 指标 + CLI。导出 `createRuntime`/`selectBackend` 供测 |
| `interface/runtime/config.mjs` | 配置解析:`loadRegistry` / `resolveSelection` / `buildInvocation` / `neutralizeForBackend`(纯函数) |
| `interface/runtime/registry.json` | agent(claude/qwen/codex/kimi)× model × profile 配置 |
| `interface/runtime/backends/base.mjs` | 子进程 spawn(prompt 走 stdin 或 arg) |
| `interface/runtime/backends/generic.mjs` | 通用后端:从 registry 配方驱动任意 CLI(加 agent = 加配置,零代码) |
| `interface/runtime/schema.mjs` | 结构化输出模拟 |
| `interface/runtime/experiment.mjs` | (agent × model) 对照实验 runner:扫矩阵 × 重复,出对比表 |
| `interface/runtime/selftest.mjs` | 假后端自测(无需 CLI/网络/GPU),36 项检查 |
| `interface/run_e2e.py` | 改:`GEAK_AGENT_PROFILE/-BACKEND/-MODEL` 时走 runtime,否则原生 Claude(默认不变) |

**未改**:`*_workflow.js`、`roles/`、`knowledge/`、`scripts/`。

## 4. 怎么用

### 选后端跑 GEAK(生产)
```bash
# 环境变量选择(优先级:命令行 flag > 环境变量 > registry.default_profile)
export GEAK_AGENT_PROFILE=qwen           # 或 GEAK_AGENT_BACKEND=codex / GEAK_MODEL=...
python interface/run_e2e.py handoff.json result.json     # e2e 自动走 runtime

# 单内核直接用 runtime
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js \
  --profile qwen \
  --args '{"kernel_path":"/abs/knn","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

### 加一个新 CLI(codex/kimi 已内置;加别的)
在 `registry.json` 的 `agents` 里加一段配置(bin、args、approve、model_flag、prompt: stdin|arg、base_url_env)。
零代码。行为特别怪的 CLI 可放 `backends/<name>.mjs` 覆盖(逃生舱)。

### 对照实验(内建)
```bash
node interface/runtime/experiment.mjs \
  --script ../../kernel_workflow/kernel_workflow.js \
  --args '{"kernel_path":"/abs/knn","workflow_dir":"/abs/kernel_workflow","budget":6}' \
  --agents claude,qwen,codex --models default --repeats 3 --out ./exp_compare
# 产出:results.jsonl + summary.md/csv(每组合:加速比均值、成功率、墙钟、agent数、schema失败数)
```
指标**不含 token/成本**(按你的要求);task/budget/gpu 在 `--args` 里对所有组合固定,只变你 sweep 的轴。

## 5. 已验证(用 Cursor 自带的 node v20 实跑)

- ✅ `selftest.mjs` **36/36 通过**:并行 barrier、pipeline 逐项+抛错落 null、信号量并发上限、schema 抽取/校验/重试、
  `workflow()` 一层嵌套 + 越级报错、config 解析优先级、buildInvocation(stdin/arg 两种投递 + model flag + base_url 路由)、
  prompt 消毒、真实 registry.json 加载解析。
- ✅ **CLI 全链路**(假后端):参数解析 → `selectBackend` → 逃生舱后端 → 并行+schema → `WORKFLOW_RESULT`/`WORKFLOW_METRICS` + result/metrics 文件。
- ✅ **真实 workflow 文件编译通过**:52KB kernel、194KB e2e 都能被 runtime 加载器(export 剥离 + AsyncFunction)成功编译。
- ✅ `run_e2e.py` `py_compile` 通过;dry-run 对 profile/agent/model/native 四种选择路由正确。

## 6. 尚未验证(需真实 CLI + GPU,见 `COMPAT_findings.md`)

真实 CLI 的运行时行为无法在本机(无 node 交互 CLI、无 GPU)验证,到 compute 节点后按优先级实测:
- **R1 结构化输出可靠性**(最高):各 CLI 拿到合法 JSON 的成功率;有原生 JSON/schema 模式的优先用。
- **R3 沙箱/权限**(最高):codex 默认沙箱禁 cwd 外写/联网,需 `--dangerously-bypass-approvals-and-sandbox`(已入 registry,待实测)。
- **R2 headless 输出格式**、**R4 单命令超时**(高)。

bring-up 顺序:`<cli> --help` 校准 registry → `selftest.mjs` → 单任务冒烟看 R1 → kernel_workflow@knn 端到端 → 与 claude parity 对比。

## 7. 相关文档

- `OVERVIEW.md` —— 架构可视化(Mermaid,Cursor 可渲染)
- `RESEARCH_cli_parallelism.md` —— 三 CLI 并行能力调研 + 来源
- `COMPAT_findings.md` —— 兼容性深挖 + R1–R7 待调研清单
