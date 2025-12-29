import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { WebviewProvider } from './webviewProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('mini-swe-agent extension is now active');
    
    const agentManager = new AgentManager(context);
    const webviewProvider = new WebviewProvider(context, agentManager);
    
    // Register webview provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'mini-swe-agent-chat',
            webviewProvider,
            {
                webviewOptions: {
                    retainContextWhenHidden: true
                }
            }
        )
    );
    
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('mini-swe-agent.start', async () => {
            const task = await vscode.window.showInputBox({
                prompt: 'What do you want the agent to do?',
                placeHolder: 'e.g., Fix the bug in app.py, Add unit tests for the Utils class',
                validateInput: (value) => value.trim() ? null : 'Task cannot be empty'
            });
            
            if (task) {
                await agentManager.startAgent(task);
            }
        }),
        
        vscode.commands.registerCommand('mini-swe-agent.stop', () => {
            agentManager.stopAgent();
        }),
        
        vscode.commands.registerCommand('mini-swe-agent.yolo', () => {
            agentManager.setMode('yolo');
        }),
        
        vscode.commands.registerCommand('mini-swe-agent.confirm', () => {
            agentManager.setMode('confirm');
        }),
        
        vscode.commands.registerCommand('mini-swe-agent.human', () => {
            agentManager.setMode('human');
        })
    );
    
    // Show welcome message on first activation
    const hasShownWelcome = context.globalState.get('mini-swe-agent.hasShownWelcome');
    if (!hasShownWelcome) {
        vscode.window.showInformationMessage(
            'Welcome to mini-swe-agent! Use the Command Palette (Ctrl+Shift+P) and run "mini: Start Agent" to begin.',
            'Got it'
        ).then(() => {
            context.globalState.update('mini-swe-agent.hasShownWelcome', true);
        });
    }
}

export function deactivate() {
    console.log('mini-swe-agent extension is now deactivated');
}

