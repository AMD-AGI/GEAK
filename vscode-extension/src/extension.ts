import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { SidebarProvider } from './sidebarProvider';
import { PanelProvider } from './panelProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('mini-swe-agent extension is now active');
    
    const agentManager = new AgentManager(context);
    const sidebarProvider = new SidebarProvider(context, agentManager);
    const panelProvider = new PanelProvider(context, agentManager);
    
    // Register sidebar provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'mini-swe-agent-sidebar',
            sidebarProvider,
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
                // Auto-show panel when starting a task
                panelProvider.show();
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
        }),
        
        vscode.commands.registerCommand('mini-swe-agent.showPanel', (taskId?: string) => {
            panelProvider.show(taskId);
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

