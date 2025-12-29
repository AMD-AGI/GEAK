# Configuration File Guide

You can configure mini-swe-agent using a YAML config file, just like the standalone `mini` command. This is especially useful for:
- Storing API keys securely in a file (instead of environment variables)
- Sharing configurations across projects
- Using different configurations for different tasks

## Quick Start

### Option 1: Create a Config File in Your Project

1. Copy the example config:
   ```bash
   cp example-config.yaml my-config.yaml
   ```

2. Edit `my-config.yaml` and add your settings:
   ```yaml
   model:
     model_name: gpt-4
     api_key: "sk-your-key-here"
     model_kwargs:
       temperature: 0.0
   
   agent:
     cost_limit: 5.0
     step_limit: 100
   ```

3. Set the config path in VS Code settings:
   - Open Settings (`Ctrl+,` or `Cmd+,`)
   - Search for "mini-swe-agent"
   - Set **Config Path** to `my-config.yaml`

### Option 2: Use Built-in Configs

mini-swe-agent comes with built-in configs you can reference by name:

```json
{
  "mini-swe-agent.configPath": "mini"
}
```

Available built-in configs:
- `mini` - Default config for interactive mode
- `default` - Minimal default config
- `github_issue` - Optimized for GitHub issues

### Option 3: Use Absolute Path

```json
{
  "mini-swe-agent.configPath": "/home/user/.config/mini-swe-agent/my-config.yaml"
}
```

## Config File Structure

A config file has three main sections:

### 1. Model Configuration

```yaml
model:
  # Required: Model name
  model_name: gpt-4
  
  # Optional: API key (instead of environment variable)
  api_key: "sk-your-key-here"
  
  # Optional: Model class (auto-detected if omitted)
  model_class: litellm
  
  # Optional: Model parameters
  model_kwargs:
    temperature: 0.0
    max_tokens: 8000
```

**Supported Models:**
- OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Any model supported by litellm

### 2. Agent Configuration

```yaml
agent:
  # Mode: confirm (review each command), yolo (auto-execute), human (manual)
  mode: confirm
  
  # Cost limit in dollars (0 = unlimited)
  cost_limit: 5.0
  
  # Step limit (0 = unlimited)
  step_limit: 100
  
  # Commands that don't require confirmation (regex patterns)
  whitelist_actions:
    - "^ls"
    - "^cat"
    - "^pwd"
    - "^grep"
  
  # Whether to confirm before finishing
  confirm_exit: true
  
  # Custom prompts (optional)
  system_template: |
    You are a helpful coding assistant...
  
  instance_template: |
    Task: {{task}}
    Please solve this step by step.
```

### 3. Environment Configuration

```yaml
env:
  # Environment variables for executed commands
  env:
    PAGER: cat
    EDITOR: nano
  
  # Timeout for commands (seconds)
  timeout: 3600
```

## Priority Order

Settings are merged in this order (later overrides earlier):
1. Config file settings
2. VS Code settings
3. Command-line arguments (if applicable)

**Example:**
- Config file sets `model_name: gpt-3.5-turbo`
- VS Code settings set `mini-swe-agent.modelName: gpt-4`
- Result: Uses `gpt-4` (VS Code settings win)

## API Key Storage

### Best Practices

**Most Secure:**
```yaml
# my-config.yaml (in .gitignore!)
model:
  api_key: "sk-your-key"
```

**Also Secure:**
```bash
# Environment variable
export OPENAI_API_KEY=sk-your-key
```

**Not Recommended:**
Don't store API keys in VS Code settings (they might sync to cloud).

### Using Environment Variables in Config

You can reference environment variables:
```yaml
model:
  api_key: ${OPENAI_API_KEY}
```

## Example Configs

### Minimal Config (Cost-Conscious)

```yaml
model:
  model_name: gpt-3.5-turbo
  api_key: "sk-your-key"

agent:
  cost_limit: 1.0
  step_limit: 20
```

### Power User Config

```yaml
model:
  model_name: gpt-4
  api_key: "sk-your-key"
  model_kwargs:
    temperature: 0.0
    max_tokens: 16000

agent:
  mode: yolo  # Auto-execute trusted commands
  cost_limit: 10.0
  step_limit: 200
  whitelist_actions:
    - "^ls"
    - "^cat"
    - "^pwd"
    - "^grep"
    - "^find"
    - "^echo"
```

### Claude Config

```yaml
model:
  model_name: claude-3-5-sonnet-20241022
  api_key: "sk-ant-your-key"
  model_kwargs:
    temperature: 0.0
    max_tokens: 8000

agent:
  cost_limit: 5.0
  step_limit: 100
```

## Config File Search Paths

When you specify a config file name (not absolute path), mini-swe-agent searches:

1. Current workspace root
2. `$MSWEA_CONFIG_DIR` (if set)
3. `~/.config/mini-swe-agent/`
4. Built-in configs (in the extension)

**Example:**
```json
{
  "mini-swe-agent.configPath": "my-config"
}
```

Searches for:
- `/workspace/my-config.yaml`
- `$MSWEA_CONFIG_DIR/my-config.yaml`
- `~/.config/mini-swe-agent/my-config.yaml`
- Built-in `my-config.yaml`

## Tips

1. **Keep API keys out of git:**
   ```bash
   echo "my-config.yaml" >> .gitignore
   ```

2. **Share configs (without keys):**
   ```yaml
   # team-config.yaml (safe to commit)
   agent:
     cost_limit: 5.0
     step_limit: 100
   
   model:
     model_name: gpt-4
     # api_key: Add your own key
   ```

3. **Per-project configs:**
   - Create `.mini-swe-agent.yaml` in project root
   - Set as default: `"mini-swe-agent.configPath": ".mini-swe-agent"`

4. **Test your config:**
   - Start with low limits while testing
   - Use `gpt-3.5-turbo` for cheaper testing

## Troubleshooting

**"Config file not found"**
- Check the path in settings
- Try absolute path
- Check file has `.yaml` extension

**"API key not working"**
- Verify key format matches provider
- Check for quotes around key in YAML
- Restart VS Code after changing config

**"Settings not taking effect"**
- Remember: VS Code settings override config file
- Clear VS Code settings to use only config file

## More Information

For full config documentation, see:
- [mini-swe-agent Config Documentation](https://mini-swe-agent.com/latest/advanced/yaml_configuration/)
- Built-in config examples in `src/minisweagent/config/`

