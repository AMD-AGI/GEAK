# mini-swe-agent for VS Code

AI coding agent that solves GitHub issues and programming tasks directly in VS Code.

## Features

- 🤖 **AI-Powered Coding Agent**: Leverages GPT-4, Claude, and other LLMs to help with coding tasks
- 🔒 **Safe Execution**: Three modes for controlling what the agent can do:
  - **Confirm Mode**: Review and approve each command before execution
  - **YOLO Mode**: Agent executes commands automatically
  - **Human Mode**: You enter commands manually while the agent assists
- 📊 **Live Progress Tracking**: See agent's reasoning, commands, and outputs in real-time
- 💰 **Cost Tracking**: Monitor API costs as the agent works
- 🛡️ **Command Whitelisting**: Auto-approve safe commands like `ls`, `cat`, `pwd`

## Requirements

- Python 3.10 or higher
- `mini-swe-agent` Python package installed in your Python environment
- API key for your chosen LLM (OpenAI, Anthropic, etc.)

## Installation

1. Install the extension from the VS Code Marketplace (or from VSIX file)
2. Install mini-swe-agent Python package:
   ```bash
   pip install mini-swe-agent
   ```
3. Configure your LLM API key as an environment variable:
   - OpenAI: `export OPENAI_API_KEY=your-key`
   - Anthropic: `export ANTHROPIC_API_KEY=your-key`

## Quick Start

1. Open a workspace folder in VS Code
2. Open Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
3. Run `mini: Start Agent`
4. Enter your task (e.g., "Fix the bug in app.py")
5. The agent will start working and ask for approval for each command

## Usage

### Starting the Agent

- **Command Palette**: `mini: Start Agent`
- Enter your task description
- The agent will analyze the task and start working

### Interaction Modes

**Confirm Mode (Default)**
- Agent proposes commands
- You review and approve/reject each one
- Safe commands matching whitelist patterns are auto-approved

**YOLO Mode**
- Agent executes all commands automatically
- Use when you trust the agent completely
- Switch via Command Palette: `mini: Toggle YOLO Mode`

**Human Mode**
- You enter commands manually
- Agent provides guidance and suggestions
- Switch via Command Palette: `mini: Switch to Human Mode`

### Monitoring

- View agent's messages in the sidebar panel
- Track current step number and cost
- See command outputs in real-time

## Configuration

### Option 1: YAML Config File (Recommended)

Create a YAML config file to store all settings including API keys:

```yaml
# my-config.yaml
model:
  model_name: gpt-4
  api_key: "sk-your-key-here"
  model_kwargs:
    temperature: 0.0

agent:
  cost_limit: 5.0
  step_limit: 100
```

Then set the path in VS Code settings:
```json
{
  "mini-swe-agent.configPath": "my-config.yaml"
}
```

See [CONFIG_FILE_GUIDE.md](./CONFIG_FILE_GUIDE.md) for complete documentation and [example-config.yaml](./example-config.yaml) for a template.

### Option 2: VS Code Settings

Configure directly in VS Code settings (`Ctrl+,` or `Cmd+,`):

```json
{
  "mini-swe-agent.modelName": "gpt-4",
  "mini-swe-agent.costLimit": 3.0,
  "mini-swe-agent.stepLimit": 50,
  "mini-swe-agent.defaultMode": "confirm",
  "mini-swe-agent.whitelistActions": ["^ls", "^cat", "^pwd", "^echo"],
  "mini-swe-agent.pythonPath": "/path/to/python"
}
```

**Note:** With this option, you still need to set API keys as environment variables.

### Settings

- **configPath**: Path to YAML config file (e.g., `my-config.yaml`, `mini`, or absolute path)
- **modelName**: LLM model to use (default: `gpt-4`, can be set in config file)
- **costLimit**: Maximum cost in dollars (default: `3.0`)
- **stepLimit**: Maximum number of steps (default: `50`, 0 = unlimited)
- **defaultMode**: Starting mode (default: `confirm`)
- **whitelistActions**: Regex patterns for auto-approved commands
- **pythonPath**: Path to Python executable (auto-detected if empty)

**Config Priority:** Config file settings are merged with VS Code settings. VS Code settings take precedence.

## Commands

- `mini: Start Agent` - Start a new agent session
- `mini: Stop Agent` - Stop the current agent
- `mini: Toggle YOLO Mode` - Switch to YOLO mode
- `mini: Switch to Confirm Mode` - Switch to Confirm mode
- `mini: Switch to Human Mode` - Switch to Human mode

## Troubleshooting

### Agent won't start

1. Check that Python 3.10+ is installed: `python3 --version`
2. Verify mini-swe-agent is installed: `pip list | grep mini-swe-agent`
3. Check the Output panel (View → Output → mini-swe-agent) for errors

### Python not found

Set the Python path explicitly in settings:
```json
{
  "mini-swe-agent.pythonPath": "/usr/bin/python3"
}
```

### API key errors

Ensure your API key is set as an environment variable:
```bash
# For OpenAI
export OPENAI_API_KEY=sk-...

# For Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Restart VS Code after setting environment variables.

## About mini-swe-agent

mini-swe-agent is a minimal, powerful AI coding agent that:
- Resolves >70% of GitHub issues in SWE-bench verified benchmark
- Uses only bash commands (no custom tools)
- Has a simple, readable codebase (~100 lines)
- Built by the Princeton & Stanford team behind SWE-bench

Learn more: [mini-swe-agent.com](https://mini-swe-agent.com)

## License

MIT License - see LICENSE file for details

## Links

- [Documentation](https://mini-swe-agent.com)
- [GitHub Repository](https://github.com/SWE-agent/mini-swe-agent)
- [Report Issues](https://github.com/SWE-agent/mini-swe-agent/issues)

