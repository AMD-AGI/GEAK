import * as vscode from 'vscode';
import { PythonBridge } from './pythonBridge';
import { AgentState, AgentMessage, InitializeParams, ConfirmationResponse } from './types';

export class AgentManager {
    private bridge: PythonBridge;
    private state: AgentState = {
        status: 'idle',
        currentStep: 0,
        totalCost: 0,
        mode: 'confirm',
        messages: []
    };
    private stateChangeEmitter = new vscode.EventEmitter<AgentState>();
    public onStateChange = this.stateChangeEmitter.event;
    
    constructor(private context: vscode.ExtensionContext) {
        this.bridge = new PythonBridge(context);
        this.setupBridgeHandlers();
    }
    
    private setupBridgeHandlers() {
        // Handle agent messages
        this.bridge.onNotification('agent/message', (params) => {
            this.state.messages.push({
                role: params.role,
                content: params.content,
                step: params.step,
                cost: params.cost
            });
            this.state.currentStep = params.step;
            this.state.totalCost = params.cost;
            this.emitStateChange();
        });
        
        // Handle confirmation requests (blocking on Python side)
        this.bridge.onRequest('agent/requestConfirmation', async (params) => {
            this.state.status = 'waiting_approval';
            this.state.pendingAction = {
                action: params.action,
                step: this.state.currentStep
            };
            this.emitStateChange();
            
            // Show quick pick for approval
            const choice = await vscode.window.showQuickPick([
                { label: '$(check) Approve', value: 'approve', description: 'Execute this command' },
                { label: '$(close) Reject', value: 'reject', description: 'Don\'t execute this command' },
                { label: '$(zap) Switch to YOLO', value: 'yolo', description: 'Auto-execute all future commands' },
                { label: '$(person) Switch to Human', value: 'human', description: 'Enter commands manually' }
            ], {
                placeHolder: `Execute command?`,
                title: 'Agent wants to execute a command'
            });
            
            this.state.status = 'running';
            this.state.pendingAction = undefined;
            this.emitStateChange();
            
            const response: ConfirmationResponse = {};
            
            if (!choice || choice.value === 'reject') {
                response.approved = false;
                response.reason = 'User rejected';
            } else if (choice.value === 'approve') {
                response.approved = true;
            } else if (choice.value === 'yolo') {
                this.state.mode = 'yolo';
                response.approved = true;
                response.switchMode = true;
                response.newMode = 'yolo';
                vscode.window.showInformationMessage('Switched to YOLO mode - all commands will execute automatically');
            } else if (choice.value === 'human') {
                this.state.mode = 'human';
                response.switchMode = true;
                response.newMode = 'human';
                vscode.window.showInformationMessage('Switched to Human mode - you can now enter commands manually');
            }
            
            return response;
        });
        
        // Handle human command requests
        this.bridge.onRequest('agent/requestHumanCommand', async (params) => {
            this.state.status = 'waiting_input';
            this.emitStateChange();
            
            const command = await vscode.window.showInputBox({
                prompt: 'Enter bash command to execute:',
                placeHolder: 'e.g., cat app.py',
                title: 'Human Mode - Enter Command'
            });
            
            this.state.status = 'running';
            this.emitStateChange();
            
            return { command: command || '' };
        });
        
        // Handle limits exceeded
        this.bridge.onNotification('agent/limitsExceeded', (params) => {
            vscode.window.showWarningMessage(
                `Agent limits exceeded: ${params.current_steps} steps, $${params.current_cost.toFixed(2)} cost`
            );
        });
        
        // Handle request for new limits
        this.bridge.onRequest('agent/requestNewLimits', async (params) => {
            const newStepLimit = await vscode.window.showInputBox({
                prompt: 'Enter new step limit (0 for unlimited):',
                value: '100',
                validateInput: (value) => {
                    const num = parseInt(value);
                    return isNaN(num) || num < 0 ? 'Please enter a valid number >= 0' : null;
                }
            });
            
            const newCostLimit = await vscode.window.showInputBox({
                prompt: 'Enter new cost limit in dollars (0 for unlimited):',
                value: '5.0',
                validateInput: (value) => {
                    const num = parseFloat(value);
                    return isNaN(num) || num < 0 ? 'Please enter a valid number >= 0' : null;
                }
            });
            
            return {
                step_limit: parseInt(newStepLimit || '100'),
                cost_limit: parseFloat(newCostLimit || '5.0')
            };
        });
        
        // Handle exit confirmation
        this.bridge.onRequest('agent/confirmExit', async (params) => {
            const choice = await vscode.window.showInformationMessage(
                'Agent wants to finish. Continue with a new task?',
                'Finish',
                'Continue'
            );
            
            if (choice === 'Continue') {
                const newTask = await vscode.window.showInputBox({
                    prompt: 'Enter new task:',
                    placeHolder: 'e.g., Now add unit tests'
                });
                
                if (newTask) {
                    return { continue: true, newTask };
                }
            }
            
            return { continue: false };
        });
        
        // Handle agent started
        this.bridge.onNotification('agent/started', () => {
            this.state.status = 'running';
            this.emitStateChange();
            vscode.window.showInformationMessage('Agent started');
        });
        
        // Handle info notifications
        this.bridge.onNotification('agent/info', (params) => {
            vscode.window.showInformationMessage(params.message);
        });
        
        // Handle warning notifications
        this.bridge.onNotification('agent/warning', (params) => {
            vscode.window.showWarningMessage(params.message);
        });
        
        // Handle agent finished
        this.bridge.onNotification('agent/finished', (params) => {
            this.state.status = 'finished';
            this.emitStateChange();
            
            vscode.window.showInformationMessage(
                `Agent finished: ${params.exitStatus}`,
                'View Output'
            ).then(choice => {
                if (choice === 'View Output') {
                    vscode.window.showInformationMessage(params.result);
                }
            });
        });
        
        // Handle errors
        this.bridge.onNotification('agent/error', (params) => {
            this.state.status = 'error';
            this.emitStateChange();
            
            vscode.window.showErrorMessage(`Agent error: ${params.error}`, 'Show Details').then(choice => {
                if (choice === 'Show Details') {
                    const channel = vscode.window.createOutputChannel('mini-swe-agent Error');
                    channel.appendLine(params.traceback);
                    channel.show();
                }
            });
        });
    }
    
    async startAgent(task: string): Promise<void> {
        const config = vscode.workspace.getConfiguration('mini-swe-agent');
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('Please open a workspace folder first');
            return;
        }
        
        // Start Python bridge
        await this.bridge.start();
        
        // Get config file path if specified
        const configFilePath = config.get<string>('configPath');
        
        // Prepare initialization parameters
        const initParams: InitializeParams = {
            workspacePath: workspaceFolder.uri.fsPath,
            modelName: config.get('modelName'), // Can be undefined if set in YAML
            task: task,
            configFilePath: configFilePath || undefined,
            config: {
                agent: {
                    mode: config.get('defaultMode', 'confirm'),
                    cost_limit: config.get('costLimit', 3.0),
                    step_limit: config.get('stepLimit', 50),
                    whitelist_actions: config.get('whitelistActions', ['^ls', '^cat', '^pwd'])
                },
                model: {},
                env: {}
            }
        };
        
        // Send initialization - this will block on Python side until agent completes
        this.bridge.sendRequest('agent/waitInitialize', initParams).catch((err) => {
            vscode.window.showErrorMessage(`Failed to initialize agent: ${err}`);
        });
        
        this.state = {
            status: 'running',
            currentStep: 0,
            totalCost: 0,
            mode: config.get('defaultMode', 'confirm'),
            messages: []
        };
        this.emitStateChange();
    }
    
    stopAgent(): void {
        this.bridge.stop();
        this.state.status = 'idle';
        this.emitStateChange();
        vscode.window.showInformationMessage('Agent stopped');
    }
    
    setMode(mode: 'confirm' | 'yolo' | 'human'): void {
        this.state.mode = mode;
        this.bridge.sendNotification('agent/setMode', { mode });
        this.emitStateChange();
        vscode.window.showInformationMessage(`Switched to ${mode} mode`);
    }
    
    getState(): AgentState {
        return { ...this.state };
    }
    
    private emitStateChange() {
        this.stateChangeEmitter.fire({ ...this.state });
    }
}

