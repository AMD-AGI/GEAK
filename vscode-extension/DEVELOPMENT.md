# Development Guide for mini-swe-agent VS Code Extension

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- VS Code 1.85+
- mini-swe-agent Python package

## Setup Development Environment

1. **Install dependencies**:
   ```bash
   cd vscode-extension
   npm install
   ```

2. **Install mini-swe-agent**:
   ```bash
   pip install -e ..  # Install from parent directory
   # Or: pip install mini-swe-agent
   ```

3. **Configure API keys**:
   ```bash
   export OPENAI_API_KEY=your-key
   # or
   export ANTHROPIC_API_KEY=your-key
   ```

## Development Workflow

### Running in Development Mode

1. Open the `vscode-extension` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. A new VS Code window will open with the extension loaded
4. Test the extension in this window

### Making Changes

**TypeScript changes:**
- Edit files in `src/`
- The extension will automatically recompile (if watch mode is running)
- Reload the Extension Development Host window (`Ctrl+R` or `Cmd+R`)

**Python changes:**
- Edit files in `python/`
- Restart the agent (no need to reload VS Code window)

### Watch Mode

Run TypeScript compiler in watch mode:
```bash
npm run watch
```

This will automatically recompile TypeScript files on save.

## Project Structure

```
vscode-extension/
├── src/                    # TypeScript source files
│   ├── extension.ts        # Main entry point
│   ├── agentManager.ts     # Agent lifecycle management
│   ├── pythonBridge.ts     # Python process communication
│   ├── webviewProvider.ts  # Webview UI provider
│   └── types.ts           # TypeScript type definitions
├── python/                 # Python backend
│   ├── main.py            # Python entry point
│   └── vscode_agent.py    # VSCodeAgent implementation
├── media/                  # Icons and assets
├── out/                    # Compiled JavaScript (gitignored)
├── package.json           # Extension manifest
├── tsconfig.json          # TypeScript configuration
└── README.md             # User documentation
```

## Architecture Overview

### Communication Flow

1. **VS Code Extension (TypeScript)**
   - Spawns Python process
   - Communicates via JSON-RPC over stdio
   - Manages webview UI
   - Handles user interactions

2. **Python Backend**
   - Runs mini-swe-agent
   - Sends messages to VS Code (agent thoughts, commands, outputs)
   - Blocks on user input (approval, commands)
   - Uses threading to handle async communication

3. **JSON-RPC Protocol**
   ```
   VS Code → Python: { "id": 1, "method": "agent/waitInitialize", "params": {...} }
   Python → VS Code: { "method": "agent/message", "params": {...} }
   Python → VS Code: { "id": 2, "method": "agent/requestConfirmation", "params": {...} }
   VS Code → Python: { "id": 2, "result": { "approved": true } }
   ```

### Key Classes

**TypeScript:**
- `PythonBridge`: Manages Python process and JSON-RPC communication
- `AgentManager`: Manages agent state and handles requests from Python
- `WebviewProvider`: Manages the sidebar webview UI

**Python:**
- `VSCodeBridge`: Handles JSON-RPC communication with VS Code
- `VSCodeAgent`: Extends `InteractiveAgent`, replaces terminal I/O with JSON-RPC

## Testing

### Manual Testing

1. Launch Extension Development Host (`F5`)
2. Open a test workspace
3. Run `mini: Start Agent` command
4. Test different scenarios:
   - Approve/reject commands
   - Switch between modes
   - Test cost/step limits
   - Test error handling

### Testing Different Python Environments

Set a custom Python path in settings:
```json
{
  "mini-swe-agent.pythonPath": "/path/to/specific/python"
}
```

### Debugging

**TypeScript debugging:**
- Set breakpoints in VS Code
- Press `F5` to start debugging
- Breakpoints will hit in the Extension Development Host

**Python debugging:**
- Add `import pdb; pdb.set_trace()` in Python code
- Check the Output panel (View → Output → mini-swe-agent)
- Python stderr appears in the output channel

## Building and Packaging

### Compile TypeScript

```bash
npm run compile
```

### Package Extension

```bash
npm run package
```

This creates a `.vsix` file that can be installed manually.

### Install Packaged Extension

```bash
code --install-extension mini-swe-agent-vscode-0.1.0.vsix
```

## Publishing

### Prerequisites

1. Install vsce:
   ```bash
   npm install -g @vscode/vsce
   ```

2. Create a publisher account at https://marketplace.visualstudio.com/

3. Get a Personal Access Token (PAT) from Azure DevOps

### Publish to Marketplace

```bash
vsce publish
```

Or publish a specific version:
```bash
vsce publish 0.1.1
```

## Troubleshooting Development Issues

### TypeScript compilation errors

```bash
npm run compile
```

Check for errors and fix them.

### Python process not starting

Check the Output panel (View → Output → mini-swe-agent) for errors.

Common issues:
- Python not found: Set `mini-swe-agent.pythonPath` in settings
- mini-swe-agent not installed: `pip install mini-swe-agent`
- Import errors: Check `PYTHONPATH` in `pythonBridge.ts`

### Webview not updating

- Check browser console in webview (right-click webview → Inspect)
- Verify `postMessage` calls are working
- Check state updates in `AgentManager`

### JSON-RPC communication issues

- Check Output panel for Python stderr
- Add logging to `PythonBridge.handleMessage()`
- Verify message format matches JSON-RPC 2.0 spec

## Code Style

- **TypeScript**: Follow VS Code extension conventions
- **Python**: Follow mini-swe-agent style guide (in parent project)
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Useful Resources

- [VS Code Extension API](https://code.visualstudio.com/api)
- [Webview API Guide](https://code.visualstudio.com/api/extension-guides/webview)
- [mini-swe-agent Documentation](https://mini-swe-agent.com)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

