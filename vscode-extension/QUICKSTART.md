# Quick Start Guide

Get mini-swe-agent running in VS Code in 5 minutes!

## Step 1: Install Dependencies

```bash
# Install Node.js dependencies
cd vscode-extension
npm install

# Install Python dependencies
pip install mini-swe-agent
```

## Step 2: Set API Key

### Option A: Config File (Recommended)

Create a config file with your API key:

```bash
cat > my-config.yaml << 'EOF'
model:
  model_name: gpt-4
  api_key: "sk-your-openai-key-here"

agent:
  cost_limit: 5.0
  step_limit: 100
EOF
```

Then set the config path in VS Code:
1. Open Settings (`Ctrl+,`)
2. Search for "mini-swe-agent"
3. Set **Config Path** to `my-config.yaml`

**Don't forget to add to .gitignore:**
```bash
echo "my-config.yaml" >> .gitignore
```

### Option B: Environment Variable

**OpenAI (GPT-4):**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

**Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Note:** On Windows, use `set` instead of `export`:
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

## Step 3: Build the Extension

```bash
npm run compile
```

## Step 4: Run in Development Mode

1. Open the `vscode-extension` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. A new VS Code window opens with the extension loaded

## Step 5: Test the Extension

In the Extension Development Host window:

1. Open a workspace folder (e.g., your project)
2. Open Command Palette: `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "mini" and select `mini: Start Agent`
4. Enter a task, for example:
   - "Create a hello world Flask app"
   - "Fix any bugs in main.py"
   - "Add error handling to the database module"
5. Watch the agent work in the sidebar!

## Step 6: Interact with the Agent

### Approving Commands (Confirm Mode)

When the agent proposes a command:
- Click **Approve** to execute it
- Click **Reject** to skip it
- Switch to **YOLO** mode to auto-execute all commands
- Switch to **Human** mode to enter commands yourself

### Sidebar Panel

- Open via Activity Bar (left sidebar) → mini-swe-agent icon
- View agent messages, commands, and outputs
- Track steps and costs
- Switch modes with buttons

## Example Tasks to Try

**Simple tasks:**
- "Create a README.md file with project description"
- "List all Python files in the project"
- "Show me the contents of app.py"

**More complex:**
- "Add type hints to all functions in utils.py"
- "Write unit tests for the Calculator class"
- "Refactor the database connection code to use context managers"
- "Fix the bug causing the server to crash on invalid input"

## Troubleshooting

### "Python process exited"

Check the Output panel:
1. View → Output
2. Select "mini-swe-agent" from dropdown
3. Look for error messages

Common fixes:
- Restart VS Code after setting environment variables
- Verify Python path: `which python3` or `where python`
- Check mini-swe-agent is installed: `pip list | grep mini-swe-agent`

### "Module not found: minisweagent"

Install mini-swe-agent:
```bash
pip install mini-swe-agent
```

### "API key not found"

Make sure to:
1. Set the environment variable (`export OPENAI_API_KEY=...`)
2. Restart VS Code after setting it
3. Or set it in your shell profile (`.bashrc`, `.zshrc`)

### Extension not appearing

1. Check that TypeScript compiled successfully: `npm run compile`
2. Look for errors in the Debug Console (View → Debug Console)
3. Try reloading the window: `Ctrl+R` or `Cmd+R`

## Next Steps

- Read [README.md](./README.md) for full feature documentation
- Check [DEVELOPMENT.md](./DEVELOPMENT.md) for development guide
- Configure settings in VS Code settings (`Ctrl+,`)
- Try different LLM models by changing `mini-swe-agent.modelName`

## Configuration Tips

**For expensive models (GPT-4):**
```json
{
  "mini-swe-agent.costLimit": 1.0,
  "mini-swe-agent.stepLimit": 20
}
```

**For cheaper models (GPT-3.5):**
```json
{
  "mini-swe-agent.modelName": "gpt-3.5-turbo",
  "mini-swe-agent.costLimit": 5.0,
  "mini-swe-agent.stepLimit": 100
}
```

**Auto-approve safe commands:**
```json
{
  "mini-swe-agent.whitelistActions": [
    "^ls",
    "^cat",
    "^grep",
    "^find",
    "^pwd",
    "^echo"
  ]
}
```

## Have Fun!

mini-swe-agent is a powerful tool. Start with simple tasks to get comfortable, then tackle more complex challenges!

Happy coding! 🚀

