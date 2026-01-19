# GEAK

**G**enerating **E**fficient **A**I-centric **K**ernels

An AI-powered agent specialized in **high-performance kernel optimization** for GPU computing. GEAK-Agent helps you analyze, optimize, and benchmark compute kernels written in HIP, Triton, and other GPU programming languages.

## ✨ Key Features

- **🚀 Kernel Optimization**: Automatically analyzes and optimizes GPU kernel code for better performance
- **📊 Strategy Management**: Maintains a prioritized list of optimization strategies with intelligent exploration
- **💡 Multi-dimensional Optimization**: Targets correctness, performance, memory efficiency, and code maintainability
- **🎯 Flexible Input**: Supports both codebase paths and text descriptions for optimization tasks
- **🔄 Iterative Refinement**: Explores different optimization approaches systematically

## 🎨 Two Ways to Use

### 1. VS Code Extension (Recommended)
Interactive development with visual strategy management and real-time feedback.

### 2. Command Line Interface
Scriptable automation for batch processing and benchmarking.

## 🏗️ Architecture

- **Minimal Core**: Built on a simple bash-based agent architecture
- **Strategy-Driven**: Uses `optool` command for managing optimization strategies
- **LLM-Powered**: Leverages large language models for intelligent decision-making
- **Modular Design**: Separates core logic, communication, and UI layers

## 🚀 Quick Start

### Option 1: VS Code Extension (Recommended)

The VS Code extension provides an interactive interface with visual strategy management, real-time chat, and sidebar controls.

#### Installation Steps

1. **Install Python dependencies** (if not already installed):

```bash
cd /path/to/geak-agent
pip install -e .
```

2. **Install Node.js** (if not already installed):

Make sure you have Node.js >= 16.x installed. Check your version:
```bash
node --version
npm --version
```

If not installed, download from [nodejs.org](https://nodejs.org/) or use a package manager:
```bash
# Ubuntu/Debian
sudo apt install nodejs npm

# macOS
brew install node

# Or use nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
```

3. **Build the extension**:

```bash
cd vscode-extension
npm install
npm run package
```

This will generate `geak-agent-X.X.X.vsix` file in the `vscode-extension` directory.

4. **Install in VS Code**:

**Method A: Via VS Code UI**
- Open VS Code
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Extensions: Install from VSIX..."
- Select the generated `.vsix` file
- Reload VS Code when prompted

**Method B: Via Command Line**
```bash
code --install-extension geak-agent-X.X.X.vsix
```

5. **Configure API Key**:
- Open VS Code Settings (`Ctrl+,` or `Cmd+,`)
- Search for "Geak Agent"
- Set your LLM API key and model name
- Default model: `claude-opus-4-20250514` (or `gpt-4o`, `claude-3-7-sonnet-20250219`)

6. **Start Using**:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
- Type "Geak Agent: Show Full Chat"
- Enter your kernel optimization task
- Select interaction mode (confirm/yolo)

#### VS Code Extension Features

- **📝 Full Chat Panel**: Interactive chat interface for task input and agent responses
- **📊 Strategy Sidebar**: 
  - View all optimization strategies with status
  - Mark strategies as high priority via checkboxes
  - Click "Set High Priority" to prioritize chosen strategies
  - Agent prioritizes `[HIGH]` strategies first
- **🎮 Agent Controls**: Start, stop, and monitor agent status
- **💬 Message History**: Complete conversation history with timestamps

### Option 2: Command Line Interface

For batch processing, benchmarking, and scriptable workflows.

#### Installation

```bash
git clone <repository-url>
cd geak-agent
pip install -e .
```

#### Basic Usage

**Interactive Mode:**
```bash
mini --config src/minisweagent/config/geak.yaml \
     --enable-strategies
```

**With Specific Task:**
```bash
mini --config src/minisweagent/config/mini_kernel.yaml \
     --enable-strategies \
     --query "Optimize the HIP kernel at /path/to/kernel.hip"
```


### Optimization Workflow

1. **Understand Task**: Analyze the kernel or description provided
2. **Analyze Current State**: Profile performance, identify bottlenecks
3. **Plan Strategies**: Generate multiple optimization approaches
4. **Prioritize**: Select high-priority strategies (user-marked or agent-chosen)
5. **Implement**: Apply optimizations systematically
6. **Benchmark**: Measure performance improvements
7. **Verify**: Ensure correctness is maintained
8. **Iterate**: Explore alternative strategies if needed


## 🔧 Configuration

GEAK-Agent uses a three-tier configuration system with clear priority order. Each tier can override settings from the previous one.

### Configuration Hierarchy (Priority: High → Low)

```
VS Code Settings (Highest)
    ↓ overrides
geak.yaml (User Config)
    ↓ overrides
Template YAML (Lowest)
```

### 1. Template YAML (Lowest Priority)

**Location**: `src/minisweagent/config/`

Contains system prompts, instance templates, and default agent behavior.

- **`mini_kernel.yaml`**: Basic kernel optimization configuration
- **`mini_kernel_strategy_list.yaml`**: With strategy list features (auto-selected when strategy mode is enabled)

**What you can configure here**:
- `system_template`: Defines the agent's role and expertise (e.g., "expert in kernel optimization")
- `instance_template`: Step-by-step workflow instructions for the agent

**Example**:
```yaml
system_template: |
  You are an expert in high-performance computing and kernel optimization...

instance_template: |
  ## Kernel Optimization Workflow
  1. Understand Task
  2. Analyze Current State
  ...
```

### 2. geak.yaml (User Config, Medium Priority)

**Location**: `~/.config/mini-swe-agent/geak.yaml` (Linux/Mac) or `%APPDATA%\mini-swe-agent\geak.yaml` (Windows)

Your personal configuration file that overrides template defaults.

**What you can configure here**:
- `model.model_name`: Your preferred LLM (e.g., `claude-opus-4.5`, `gpt-4o`)
- `model.api_key`: Your API key (or leave empty to use VS Code settings)
- `agent.mode`: Default interaction mode (`confirm`, `yolo`)
- `agent.step_limit`: Maximum steps per task (0 = unlimited)
- `agent.cost_limit`: Maximum cost in USD (0 = unlimited)
- `model.model_kwargs`: Temperature, max_tokens, reasoning effort, etc.

**Example**:
```yaml
agent:
  step_limit: 100
  cost_limit: 5.0
  mode: confirm
model:
  model_name: claude-opus-4.5
  api_key: "sk-..."  # or leave empty
  model_kwargs:
    temperature: 0.0
    max_tokens: 16000
```

### 3. VS Code Settings (Highest Priority)

**Location**: VS Code Settings UI or `.vscode/settings.json`

Per-workspace settings that override everything else. Ideal for project-specific configurations.

**How to configure**:
1. Open VS Code Settings (`Ctrl+,` or `Cmd+,`)
2. Search for "Geak Agent" or "mini-swe-agent"

**Available settings**:
- `mini-swe-agent.apiKey`: API key (overrides geak.yaml)
- `mini-swe-agent.modelName`: Model name (overrides geak.yaml)
- `mini-swe-agent.defaultMode`: Interaction mode (`confirm`, `yolo`)
- `mini-swe-agent.stepLimit`: Max steps per task
- `mini-swe-agent.costLimit`: Max cost in USD
- `mini-swe-agent.strategyMode.enabled`: Enable/disable strategy management
- `mini-swe-agent.strategyMode.filePath`: Path to `.optimization_strategies.md`

**Example** (`.vscode/settings.json`):
```json
{
  "mini-swe-agent.apiKey": "sk-ant-...",
  "mini-swe-agent.modelName": "claude-opus-4.5",
  "mini-swe-agent.defaultMode": "confirm",
  "mini-swe-agent.stepLimit": 50,
  "mini-swe-agent.strategyMode.enabled": true
}
```

### Configuration Tips

1. **For quick experiments**: Change settings in VS Code Settings (fastest, workspace-specific)
2. **For personal preferences**: Edit `geak.yaml` (applies to all projects)
3. **For prompt engineering**: Edit template YAML files (changes how the agent thinks)

**Priority Example**:
- Template sets `model_name: gpt-4`
- geak.yaml sets `model_name: claude-opus-4.5` → Uses Claude Opus 4.5
- VS Code sets `modelName: gpt-4o` → **Uses GPT-4o** (highest priority)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional optimization strategies
- Support for more kernel languages
- Improved benchmarking automation
- Enhanced UI/UX features
- Bug fixes and performance improvements

## 📄 License

See [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

Built on the foundation of [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) by the Princeton & Stanford team.

---

**Happy Optimizing! 🚀**

