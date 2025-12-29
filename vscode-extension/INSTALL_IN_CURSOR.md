# Installing mini-swe-agent Extension in Cursor

Cursor is VS Code-compatible, so the extension works perfectly! Here's how to install it.

## Method 1: Package and Install (Recommended)

This permanently installs the extension in Cursor.

### Step 1: Package the Extension

```bash
cd /mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/vscode-extension

# Make sure it's compiled
npm run compile

# Package it
npm run package
```

This creates a file: `mini-swe-agent-vscode-0.1.0.vsix`

### Step 2: Install in Cursor

**Option A: Via Command Line**
```bash
cursor --install-extension mini-swe-agent-vscode-0.1.0.vsix
```

**Option B: Via Cursor UI**
1. Open Cursor
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type: **Extensions: Install from VSIX...**
4. Select the file: `mini-swe-agent-vscode-0.1.0.vsix`

### Step 3: Verify Installation

1. In Cursor, press `Ctrl+Shift+X` (Extensions sidebar)
2. Search for "mini-swe-agent"
3. You should see it installed!

### Step 4: Configure

1. Open Settings (`Ctrl+,`)
2. Search for "mini-swe-agent"
3. Set your config:
   ```json
   {
     "mini-swe-agent.configPath": "my-config.yaml"
   }
   ```

### Step 5: Use It!

1. Open a workspace folder
2. Press `Ctrl+Shift+P`
3. Type: **mini: Start Agent**
4. Enter your task
5. Watch the magic happen! ✨

---

## Method 2: Development Mode

For testing or development, run the extension directly:

### Step 1: Install Dependencies

```bash
cd /mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/vscode-extension
npm install
npm run compile
```

### Step 2: Open in Cursor

```bash
cursor /mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/vscode-extension
```

### Step 3: Launch Extension Development Host

1. Press `F5` (or Run → Start Debugging)
2. A new Cursor window opens with the extension loaded
3. This is a temporary session for testing

---

## Troubleshooting

### "cursor: command not found"

Install Cursor command-line tools:
1. Open Cursor
2. Press `Ctrl+Shift+P`
3. Type: **Shell Command: Install 'cursor' command in PATH**

Or use the full path:
```bash
# Linux
/usr/bin/cursor --install-extension mini-swe-agent-vscode-0.1.0.vsix

# macOS
/Applications/Cursor.app/Contents/MacOS/Cursor --install-extension mini-swe-agent-vscode-0.1.0.vsix
```

### Extension Not Showing Up

1. Restart Cursor
2. Check Extensions panel (`Ctrl+Shift+X`)
3. Look for any error messages

### "Python process not starting"

Make sure mini-swe-agent is installed:
```bash
pip install mini-swe-agent
```

### API Key Issues

Set up your config file with API key:
```yaml
# my-config.yaml
model:
  model_name: gpt-4
  api_key: "sk-your-key-here"
```

Then set in Cursor settings:
```json
{
  "mini-swe-agent.configPath": "my-config.yaml"
}
```

---

## Uninstalling

To remove the extension:

1. Open Extensions (`Ctrl+Shift+X`)
2. Find "mini-swe-agent"
3. Click the gear icon → Uninstall

Or via command line:
```bash
cursor --uninstall-extension mini-swe-agent.mini-swe-agent-vscode
```

---

## Updating the Extension

When you make changes:

1. Compile: `npm run compile`
2. Package: `npm run package`
3. Reinstall: `cursor --install-extension mini-swe-agent-vscode-0.1.0.vsix`

Cursor will ask if you want to replace the existing version - click Yes.

---

## Quick Reference

```bash
# Full installation flow
cd vscode-extension
npm install
npm run compile
npm run package
cursor --install-extension mini-swe-agent-vscode-0.1.0.vsix

# Verify
cursor
# Then: Ctrl+Shift+P → "mini: Start Agent"
```

---

## Notes

- Cursor is a VS Code fork, so all VS Code extensions work
- The extension uses the same settings system as VS Code
- All features work identically in Cursor
- You can use Cursor's AI features alongside mini-swe-agent!

Enjoy coding with mini-swe-agent in Cursor! 🎉


