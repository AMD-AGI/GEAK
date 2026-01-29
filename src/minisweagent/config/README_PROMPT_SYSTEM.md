# Prompt Management System

## 概述

本项目采用模块化的 prompt 管理体系，通过 Jinja2 条件语句实现动态 prompt 组合。

## 架构

### 核心文件

1. **mini_unified.yaml** - 统一的模板配置文件
   - 使用 Jinja2 的 `{% if save_patch %}` 条件语句
   - 根据 `save_patch` 参数动态插入不同的 prompt 片段
   - 适用于大多数 GPU kernel 优化场景

2. **prompt_fragments/** - Prompt 片段目录
   - `save_patch.j2` - save_patch 相关的 prompt 片段
   - `baseline.j2` - 基础 workflow 的 prompt 片段

3. **mini_kernel_strategy_list.yaml** - 策略管理模式的模板
   - 包含 optool 命令相关的说明
   - 用于需要策略列表管理的场景

## 工作流程

### 1. 模板选择（mini.py）

```python
if enable_strategies and not save_patch:
    template_name = "mini_kernel_strategy_list.yaml"
else:
    template_name = "mini_unified.yaml"
```

### 2. 参数传递

在 `ParallelAgent.__init__` 中：
```python
self.extra_template_vars['save_patch'] = self.config.save_patch
```

这确保了 `save_patch` 参数可以在模板渲染时使用。

### 3. 模板渲染

`DefaultAgent.render_template` 方法会：
1. 合并配置参数：`asdict(self.config)`
2. 合并环境变量：`self.env.get_template_vars()`
3. 合并模型变量：`self.model.get_template_vars()`
4. 合并额外变量：`self.extra_template_vars`
5. 使用 Jinja2 渲染模板

### 4. 条件渲染

在 `mini_unified.yaml` 中：
```jinja
{% if save_patch -%}
Phase 3: Baseline Establishment
- **ALWAYS use the TEST_BASELINE_PERFORMANCE command...**
Phase 5: Controlled Experimentation
- **ALWAYS use SAVE_PATCH_AND_TEST to test performance...**
{%- else -%}
Phase 3: Baseline Establishment
- Run existing benchmarks or create simple performance tests...
Phase 5: Implementation and Testing
- Implement optimizations incrementally...
{%- endif %}
```

## 使用示例

### 启用 save_patch 模式

```bash
mini --save-patch --test-command "python benchmark.py" --task "optimize kernel.cu"
```

模板将自动包含：
- `TEST_BASELINE_PERFORMANCE` 命令说明
- `SAVE_PATCH_AND_TEST` 命令说明
- 详细的优化策略列表

### 不启用 save_patch 模式

```bash
mini --task "optimize kernel.cu"
```

模板将使用简化的 workflow 说明，不包含 patch 相关的命令。

## 优势

1. **模块化**：prompt 片段独立管理，易于维护
2. **灵活性**：根据运行时参数动态调整 prompt 内容
3. **可扩展**：可以轻松添加新的条件片段
4. **避免重复**：不需要维护多个几乎相同的完整模板文件
5. **类型安全**：通过配置类（如 `ParallelAgentConfig`）管理参数

## 扩展指南

### 添加新的 prompt 片段

1. 在 `prompt_fragments/` 目录创建新的 `.j2` 文件
2. 在 `mini_unified.yaml` 中使用条件语句引用
3. 在相应的配置类中添加控制参数
4. 在 agent 的 `__init__` 方法中设置 `extra_template_vars`

示例：
```python
# 在 AgentConfig 中添加参数
@dataclass
class MyAgentConfig(AgentConfig):
    enable_feature_x: bool = False

# 在 Agent 中传递参数
class MyAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_template_vars['enable_feature_x'] = self.config.enable_feature_x
```

```jinja
# 在模板中使用
{% if enable_feature_x -%}
Feature X specific instructions...
{%- endif %}
```

## 已弃用的文件

- `mini_system_prompt.yaml` - 可以保留作为参考，但不再是主要模板
