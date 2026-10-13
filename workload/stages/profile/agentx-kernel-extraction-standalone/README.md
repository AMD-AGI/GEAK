# AgentX kernel extraction — 独立 skill

**整个文件夹交给 Claude Code（CC）或 Codex，直接读取 `SKILL.md` 即可使用。**
不需要 GEAK 仓库、安装补丁、workflow runner 或角色系统。agent 自己执行各阶段；
选 kernel、捕获输入、构建 UT、检查交付所需的通用辅助工具都在包内。

解压后直接给 agent 这段话，替换路径和输入：

```text
请读取 /path/to/agentx-kernel-extraction-standalone/SKILL.md，按这个 skill 执行。

输出目录：/path/to/new/extraction-evidence
AgentX baseline 日志、summary 和 benchmark exports：<实际路径>
workload / 模型 / 有效 serving 配置：<实际路径或说明>
已有原始 trace：<实际路径；没有则说明>
当前 serving 环境：<container/image、backend、源码和已接受的 overlay>
GPU 资源：<实际分配的设备、容器编号映射、TP 和独占安排>

按各 phase Top 5 且 overall GPU 时间占比 >3% 抽取实际 kernel 源码，
交付可执行 standalone UT、正确性/绑定/graph/两腿计时和源码证据。
复用已完成 baseline，缺少证据时只做有界 capture。
不做 kernel 优化，不重测整模型吞吐。缺少必要输入请先指出并继续可独立完成的工作。
```

无需把文件安装到 agent 的特定目录；直接读上述绝对路径即可。若团队已有 skills
发现机制，也可以把**整个文件夹**放进其 skills 目录，保留相对路径。

## 文件与环境

| 文件 | 用途 |
|---|---|
| [SKILL.md](SKILL.md) | agent 的入口和五阶段工作流程 |
| `references/` | workload、capture、源码、联合 indexer、collective、UT 和交付细则 |
| `scripts/` | selector、trace/capture、overlay、测试 harness、GPU 锁、本地交付检查 |
| `tests/` | 可离线执行的 CPU 回归；不需要模型或 GPU |
| [VALIDATION.md](VALIDATION.md) | 此独立版的测试范围和限制 |
| `PROVENANCE.json`、`LICENSE.md` | 工具来源与许可证 |

CPU 工具需要 Python 3.10+、Node.js 18+，无 npm/pip 安装步骤。
实际 GPU 抽取仍需要同事已有的 PyTorch/serving backend、模型、GPU 和对应编译器；
它们是被分析的运行环境，不由 skill 下载或安装。GPU 锁工具针对 Linux/AMD ROCm；
跨容器锁共享、设备编号和其他 backend 的适配见 [setup.md](references/setup.md)。

在任意目录检查本包：

```bash
python3 /path/to/agentx-kernel-extraction-standalone/scripts/check_bundle.py
python3 /path/to/agentx-kernel-extraction-standalone/scripts/check_bundle.py --self-test
```

检查不会启动 GPU 工作、AgentX 或 serving。`GEAK_*` 等少量遗留内部变量名仅用于
已随包复制的工具兼容配置；没有外部 GEAK import、目录或服务依赖。

完整 GPU 抽取结果必须在实际环境执行后才能宣称通过；本包的 CPU 检查不代替该验收。
