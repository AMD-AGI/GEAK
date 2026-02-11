# Config File Support - Feature Summary

## ✅ What Was Added

The VS Code extension now supports YAML configuration files, just like the standalone `mini` command!

### Key Benefits

1. **Store API keys in files** instead of environment variables
2. **Reuse mini-swe-agent configs** - Your existing `mini.yaml` works!
3. **Share configurations** across projects and team members
4. **Full control** over model parameters, agent behavior, and environment

## 🎯 How It Works

### Quick Example

1. **Create a config file:**
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

2. **Set in VS Code settings:**
```json
{
  "mini-swe-agent.configPath": "my-config.yaml"
}
```

3. **Done!** Start the agent and it will use your config.

## 📦 What Can You Configure?

Everything that `mini.yaml` supports:

### Model Settings
- `model_name` - Which LLM to use
- `api_key` - Your API key (no more env vars needed!)
- `model_class` - Model provider (auto-detected)
- `model_kwargs` - Temperature, max_tokens, reasoning, etc.

### Agent Settings
- `mode` - confirm/yolo/human
- `cost_limit` - Maximum cost in dollars
- `step_limit` - Maximum steps
- `whitelist_actions` - Auto-approved commands
- `system_template` - Custom system prompt
- `instance_template` - Custom task prompt
- And more!

### Environment Settings
- `env` - Environment variables for commands
- `timeout` - Command timeout in seconds

## 🔄 Config Priority

Settings merge in this order (later wins):

1. **Config file** (base settings)
2. **VS Code settings** (overrides config file)

**Example:**
- Config file: `model_name: gpt-3.5-turbo`
- VS Code setting: `mini-swe-agent.modelName: gpt-4`
- **Result:** Uses `gpt-4`

## 📁 Config File Locations

You can specify:

1. **Relative path** (in workspace):
   ```json
   "mini-swe-agent.configPath": "my-config.yaml"
   ```

2. **Built-in config name**:
   ```json
   "mini-swe-agent.configPath": "mini"
   ```

3. **Absolute path**:
   ```json
   "mini-swe-agent.configPath": "/home/user/.config/mini-swe-agent/my-config.yaml"
   ```

### Search Order

For relative paths, mini-swe-agent searches:
1. Workspace root
2. `$MSWEA_CONFIG_DIR` (if set)
3. `~/.config/mini-swe-agent/`
4. Built-in configs in extension

## 🔐 Security Best Practices

### ✅ DO:
- Store API keys in config files (add to `.gitignore`)
- Use absolute paths for personal configs
- Share configs without API keys

### ❌ DON'T:
- Commit API keys to git
- Store keys in VS Code settings (might sync to cloud)
- Share config files with keys

## 📚 Documentation

Three comprehensive guides included:

1. **CONFIG_FILE_GUIDE.md** - Complete guide with examples
2. **example-config.yaml** - Template you can copy
3. **README.md** - Updated with config instructions

## 🔧 Implementation Details

### Python Backend Changes

**File:** `python/main.py`
- Added YAML config loading via `minisweagent.config.get_config_path()`
- Merges YAML config with VS Code settings
- Sends notifications about config loading

### TypeScript Changes

**Files:** `src/agentManager.ts`, `src/types.ts`
- Added `configFilePath` parameter
- Passes config path to Python backend
- Handles info/warning notifications

**File:** `package.json`
- Updated `configPath` setting description

## 🚀 Usage Examples

### Example 1: Simple Config

```yaml
model:
  model_name: gpt-3.5-turbo
  api_key: "sk-your-key"

agent:
  cost_limit: 1.0
```

### Example 2: Power User Config

```yaml
model:
  model_name: gpt-4
  api_key: "sk-your-key"
  model_kwargs:
    temperature: 0.0
    max_tokens: 16000

agent:
  mode: yolo
  cost_limit: 10.0
  step_limit: 200
  whitelist_actions:
    - "^ls"
    - "^cat"
    - "^pwd"
    - "^grep"
```

### Example 3: Claude Config

```yaml
model:
  model_name: claude-3-5-sonnet-20241022
  api_key: "sk-ant-your-key"
  model_kwargs:
    temperature: 0.0

agent:
  cost_limit: 5.0
```

### Example 4: Use Built-in Config

Just reference by name:
```json
{
  "mini-swe-agent.configPath": "mini"
}
```

This uses the built-in `mini.yaml` config!

## 🎉 Benefits Over Environment Variables

| Feature | Environment Variables | Config Files |
|---------|----------------------|--------------|
| Store API keys | ❌ Separate setup | ✅ In config |
| Share configs | ❌ Hard to share | ✅ Easy (without keys) |
| Version control | ❌ Not in repo | ✅ Can commit (without keys) |
| Multiple configs | ❌ One per shell | ✅ Switch easily |
| Full settings | ❌ Limited | ✅ Everything |
| Reuse mini configs | ❌ No | ✅ Yes! |

## 🧪 Testing

To test config file support:

1. Create `test-config.yaml`:
```yaml
model:
  model_name: gpt-3.5-turbo
  api_key: "your-key"

agent:
  cost_limit: 0.5
  step_limit: 5
```

2. Set in VS Code: `"mini-swe-agent.configPath": "test-config.yaml"`

3. Start agent - you should see: "Loaded config from /path/to/test-config.yaml"

4. Agent should use settings from the config file

## 📝 Notes

- Config files use the same format as standalone `mini` command
- All sections (`model`, `agent`, `env`) are optional
- VS Code settings can override any config file setting
- API keys in config files work the same as environment variables
- Notifications show when config is loaded or if there are issues

## 🔗 Related Files

- `python/main.py` - Config loading logic
- `src/agentManager.ts` - Config path handling
- `src/types.ts` - Type definitions
- `CONFIG_FILE_GUIDE.md` - User documentation
- `example-config.yaml` - Template
- `README.md` - Updated instructions

---

**This feature makes the VS Code extension fully compatible with mini-swe-agent's configuration system!** 🎊

