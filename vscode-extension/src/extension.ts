import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { SidebarProvider } from './sidebarProvider';
import { PanelProvider } from './panelProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('mini-swe-agent extension is now active');
    
    const agentManager = new AgentManager(context);
    
    // Initialize strategy manager with workspace path
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders && workspaceFolders.length > 0) {
        const workspacePath = workspaceFolders[0].uri.fsPath;
        agentManager.initStrategyManager(workspacePath);
    }
    
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
            const state = agentManager.getState();
            
            // If panel is visible and agent is idle with no messages, let user input in panel
            if (panelProvider.isVisible() && state.status === 'idle' && 
                (!state.messages || state.messages.length === 0)) {
                // Panel is showing the task input area, just focus it
                panelProvider.show(); // This will trigger focus
                vscode.window.showInformationMessage('Please enter your task in the panel');
                return;
            }
            
            // Otherwise, use the input box
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
            'Welcome to GEAK Agent! Use the Command Palette (Ctrl+Shift+P) and run "GEAK: Start Agent" to begin.',
            'Got it'
        ).then(() => {
            context.globalState.update('mini-swe-agent.hasShownWelcome', true);
        });
    }
    
    // Register cleanup
    context.subscriptions.push({
        dispose: () => {
            agentManager.dispose();
        }
    });
}

export function deactivate() {
    console.log('mini-swe-agent extension is now deactivated');
}

