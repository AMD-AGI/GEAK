# 实现计划：team_workflow（dynamic workflow 版 kernel 优化编排）

## 目标
把现有 markdown 驱动的 `team` skill 重构成**确定性 JS Workflow 脚本**编排的 dynamic workflow，在 knn 上 geomean 比旧 skill 的 36.5x 再快 >50%（≈55x）。

## 产出目录
`/wekafs/zihao/2026/geak_cc/PerfSkills/workflows/`（用户确认；不改任何现有文件，scripts/knowledge 为新建副本）

## 已确认的架构增强
- **A 工程师按专长分型**：algorithm / memory / compute / host_runtime 四类 persona，每类只加载相关 knowledge，上下文聚焦。
- **B Host/Runtime 设为一等角色**：专攻 launch 开销地板、dispatch 合并/fusion、CUDA graph、host 旁路 —— knn 破 55x 的关键杠杆，e2e 也复用。
- **C 跨轮洞察黑板 + 假设台账**：每轮 TechLead 蒸馏「学到了什么」注入下一轮 engineer；每个方向=带预期收益的假设，verify 后记实际 vs 预期，指导 re-plan。
- **E 整合者 Integrator**（升级版 merge）：可手动把冲突好点子重写成连贯最佳实现，不只 git apply 叠加。
- **H Director 仲裁打回**：最终验收 flag 时（verified≪claimed 等）有权打回 TechLead 做一轮纠正。
- 不采纳：D 红队、F beam search、G 先提案后去重（方向由 TechLead 统一正交规划，去重在规划时完成）。
- 保留**独立复测 verify_engineer**（核心可靠性，非 D）。每轮单赢家贪心提交。

## 目录结构
```
workflows/
├── README.md
├── team_workflow.js              # 确定性 Workflow 脚本
├── roles/
│   ├── director.md               # setup + 最终独立验证 + 仲裁打回(H)
│   ├── tech_lead.md              # 分析/roadmap + 每轮 re-plan(含洞察黑板/假设台账 C, 多样性检查) + 整合指导 + 终报
│   ├── engineer.md               # 优化 worker，按 specialty 分型(A)，含 host_runtime(B) + self_monitoring
│   ├── benchmark_engineer.md     # harness / COMMANDMENT / baseline
│   ├── profile_engineer.md       # profile + 瓶颈分类
│   ├── verify_engineer.md        # 独立复测(信任来源)
│   └── integrator.md             # 合并/重写多个胜出 patch(E)
├── knowledge/
│   ├── optimization_strategies.md
│   ├── hip_optimization.md
│   ├── triton_optimization.md
│   ├── wrapper_optimization.md   # host_runtime persona 主用
│   ├── profiling_guide.md
│   ├── amd_mi300x.md
│   ├── self_monitoring.md
│   └── geomean_levers.md         # 新增：launch 地板/最慢用例/dispatch 合并/fusion/CUDA graph
└── scripts/
    ├── gpu_lock.sh               # 搬运
    └── profile_kernel.sh         # 搬运
```

## 角色 → workflow 映射
- **Director** = 脚本编排逻辑（budget 循环、fan-out）+ setup agent + 最终验证/仲裁 agent。
- **TechLead** = agent：①分析+roadmap ②每轮用 JSON schema 返回「方向列表+数量+是否停」，方向带 specialty/category/focus_files 保证正交；维护洞察黑板与假设台账。
- **Engineer(specialist)** = 并行 agent，按 specialty 优化；另有 benchmark/profile/verify/integrator 专职。

## team_workflow.js 流程
`args = {kernel_path(必填), budget=6, gpu_ids="0", task, eval_dir, apply_to_original=false}`；`WORKFLOW_DIR` 常量指向本目录。脚本不碰 FS，全部由 agent 执行。

1. **Setup**(director agent)：agent 用 `date` 生成时间戳，建 `exp/team_<name>_<ts>/<name>/`，复制 kernel→workspace+baseline，workspace `git init` 提交 baseline。→`{eval_dir, workspace, kernel_name, source_files}`
2. **Analyze+Roadmap**(tech_lead)：analysis.json / codebase_context.md / roadmap.md。→`{kernel_type, kernel_file, modifiable_files, bottleneck_guess, roadmap_summary}`
3. **Benchmark setup**(benchmark_engineer)：复用/新建 task_runner，写 COMMANDMENT.md，跑 3 次校验稳定性，记 baseline。→`{correctness_cmd, benchmark_cmd, profile_cmd, baseline_per_case[], baseline_geomean_ms, reliable}`
4. **Baseline profile**(profile_engineer)：profile_kernel.sh → baseline_metrics.json / profiling_summary.md。→`{bottleneck, key_metrics, top_opportunities}`
5. **优化循环**(JS while: dispatched<budget && noImprove<2)：
   - a. **Plan round**(tech_lead)：传 history(洞察黑板+假设台账)+profile+remaining，返回 `{stop, directions:[{id,title,specialty,prompt,focus_files}]}`，脚本 clamp 到 remaining，round-robin 分配 gpu_id。
   - b/c. **pipeline(directions, optimize, verify)**：specialist engineer 在 canonical 私有副本优化产 patch → verify_engineer 干净副本 apply+复测产**绝对** per_case 延迟。
   - d. **integrate**(≥2 verified 时, 不占 budget)：integrator 合并/重写 → 复测。
   - e. 选 winner（含 integrate），绝对口径 speedup=baseline_geomean_ms/候选_geomean_ms；若 winner>cumulative*1.05：commit agent 应用并提交进 canonical，写 current_best.diff；cumulative=winner_speedup；noImprove=0；否则 noImprove++。
   - f. 改进则 **re-profile**(profile_engineer)。
   - dispatched += directions.length；TechLead 更新洞察黑板/台账，记 history。
6. **Final report**(tech_lead)：tech_lead_report.md（逐轮逐 engineer + 最终 per-case 表 + geomean/arith）；写 final_patch.diff（baseline→HEAD 累计 diff）。
7. **Director validation + 仲裁**(director agent)：从**原始 kernel_path** 建 validation_workspace，apply final_patch，独立复测 → director_validation.json；flag 时可打回 TechLead 一轮纠正；按 apply_to_original 决定回写。

脚本返回 `{eval_dir, final_geomean, final_arithmetic, validation_status, report_path}`。

## budget 语义
只统计「优化方向 engineer」；benchmark/profile/verify/integrate/commit/validate 不计入。脚本硬卡 `directions ≤ budget-dispatched`。TechLead 可提前 `stop`。

## 通用性
脚本永不 if kernel 类型 / 单 kernel vs e2e；一切走 benchmark 阶段发现的 COMMANDMENT。vLLM/SGLash 只是 COMMANDMENT 内容不同（启服务/吞吐基准/输出对齐），编排不变。

## 验证步骤（建好后，完整跑证明 >50%）
1. 确认 knn 例子 HEAD 为 pristine 原版，把 tracked 文件 `git checkout` 回 HEAD 作公平 baseline；清 build。
2. `rocm-smi` 选空闲 GPU。
3. Workflow scriptPath 调用，args={kernel_path=knn 例子, budget=6, gpu_ids=<空闲>, apply_to_original=false}。
4. 读 director_validation.json，确认 geomean > 54.8x。未达标则改进 knowledge/roles/roadmap 后重跑。
