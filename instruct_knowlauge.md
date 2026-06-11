/wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team 这是一个 优化 llm 推理的 workflow

我希望你来提升它的性能，具体是通过增加更过的专业的和 kernel 相关 knowlage，我希望你从网页上取收集并且去整理，越详细越完善越好

整理的结果整理到 /wekafs/zihao/2026/geak_cc/PerfSkills/kernel_knowledge 这个目录下

整理要从几个维度，这里只列了几个，每一类你要做扩充，只收集 amd 能用
1. 语言类型：tirton, hip, CK, asm 等等（和amd 相关）
2. 算子库：aiter, vllm, sglang, hipblas 等等
3. 具体算子： Gemm, group_gemm, moe, fp4/8/16, attention, sparse attention, linear attention, mla, deepseek_seek_v4 的 attention
4. 优化策略：算法，Gemm tuning

整理的结果要完善，通过这个 knowledge 能够指导 agent 去写出超高质量的代码
你从网上去收集结果，包括文档，白皮书，repo 的 pr 等等，你至少再列举 5 个来源
执行时你可以使用多一个 subagent 同时去收集

你先给出一个方案，我看看没问题之后在执行