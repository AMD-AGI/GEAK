# 目标：我想你基于 team_workflow 将他从单 kernel 优化拓展成一个 e2e 优化的 workflow -> team_workflow_e2e

背景：
- 这个是 team_workflow 的地址：/wekafs/zihao/2026/geak_cc/PerfSkills/workflows
- team_workflow_e2e 放到 /wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team 下
- team_worflow 是一个 kernel 优化的 workflow，你需要仔细深入的阅读它的流程，在他的基础上去做扩展，得到 team_workflow_e2e
- team_workflow_e2e 输入是一个 llm，你来优化它的 sglang 或者 vllm 的推理速度。当然他要要兼容 team_workflow 也就是去优化单个 kernel
- team_workflow_e2e 也是分级别分角色的去优化

team_workflow_e2e 优化流程: 
- 根据输入的 prompt ，获得启动脚本 (示例:/wekafs/zihao/2026/geak_cc/PerfSkills/bench_qwen35_27b_sglang.sh)
- 根据脚本获得 baseline 吞吐，还有 baseline 的 profile (示例:/wekafs/zihao/2026/geak_cc/PerfSkills/bench_qwen35_27b_sglang.sh)，profile 之后要接入一个工具，把 trace 解析成一个 per-kernel 耗时 Top-N 的文本摘要，这一步骤的结果要规范。
- 通过修改下不同 kernel 的 backend 还有启动脚本的配置，来提升吞吐，backend 可以从启动脚本，环境变量还有原始代码三个维度去修改（这一个步骤可选, 目前默认关闭）
- 根据 kernel 耗时，去优化kernel，这一步骤进一步拆解：
  - 针对 kernel， 获得一个 unittest，作为这个 kernel 优化用的 harness。这个 unittest 要做好代码隔离，放到 exp 目录下，不要修改源码，unittest 格式要规范，包括能编译（这一步可选，有些kernel没有这一步），跑通，正确，以及能测试 speedup。
  - 对于同一个 kernel 任务，尝试去找不同 backend 的实现，比如 triton, hip, ck, asm 等等，这里要整理成经验，不同 kernel 有不同的合适的backend，然后更具经验去找到合适 backend 做下速度对比，测试什么的都用 unittest 来做，所以你的 unittest 要足够通用。优化过程不能去修改 unittest，防止作弊
  - 根据你找到的合适的 kernel backend，一个或者多个，去做 kernel 的优化，这个步骤和现在的 team_workflow 类似
  - 将优化好的 kernel，放回到原始的代码里，去做 e2e 验证
- 得到优化后的代码版本，包含完整的 patch，还有启动测速脚本

你先看看上述 ppl 有什么问题，给出至少 10 个 ppl 优化的建议，目标是让 workflow 通用，效果更好。给我的建议都用选择题的形式



