import * as vscode from 'vscode';
import { PythonBridge } from './pythonBridge';
import { AgentState, AgentMessage, InitializeParams, ConfirmationResponse, Task } from './types';

export class AgentManager {
    private bridge: PythonBridge;
    private state: AgentState = {
        status: 'idle',
        currentStep: 0,
        totalCost: 0,
        mode: 'confirm',
        messages: [],
        taskHistory: []
    };
    private stateChangeEmitter = new vscode.EventEmitter<AgentState>();
    public onStateChange = this.stateChangeEmitter.event;
    
    private pendingActionResolvers: Map<string, (response: ConfirmationResponse) => void> = new Map();
    private nextActionId: number = 1;
    
    constructor(private context: vscode.ExtensionContext) {
        this.bridge = new PythonBridge(context);
        this.setupBridgeHandlers();
    }
    
    private setupBridgeHandlers() {
        // Handle agent messages
        this.bridge.onNotification('agent/message', (params) => {
            const message: AgentMessage = {
                role: params.role,
                content: params.content,
                step: params.step,
                cost: params.cost,
                timestamp: new Date()
            };
            
            this.state.messages.push(message);
            this.state.currentStep = params.step;
            this.state.totalCost = params.cost;
            
            // Update current task
            if (this.state.currentTask) {
                this.state.currentTask.totalSteps = params.step;
                this.state.currentTask.totalCost = params.cost;
            }
            
            this.emitStateChange();
        });
        
        // Handle confirmation requests (blocking on Python side)
        this.bridge.onRequest('agent/requestConfirmation', async (params) => {
            // Generate action ID
            const actionId = `action_${this.nextActionId++}_${Date.now()}`;
            
            // Check mode
            if (this.state.mode === 'yolo') {
                // Auto-approve in YOLO mode
                const actionMessage: AgentMessage = {
                    role: 'assistant',
                    content: params.action,
                    step: this.state.currentStep,
                    timestamp: new Date(),
                    isAction: true,
                    actionId: actionId,
                    actionStatus: 'auto-approved',
                    actionCommand: params.action,
                    actionTimestamp: new Date()
                };
                this.state.messages.push(actionMessage);
                this.emitStateChange();
                
                return { approved: true };
            }
            
            // Create pending action message
            const actionMessage: AgentMessage = {
                role: 'assistant',
                content: params.action,
                step: this.state.currentStep,
                timestamp: new Date(),
                isAction: true,
                actionId: actionId,
                actionStatus: 'pending',
                actionCommand: params.action
            };
            
            this.state.messages.push(actionMessage);
            this.state.status = 'waiting_approval';
            this.state.pendingAction = {
                action: params.action,
                actionId: actionId,
                step: this.state.currentStep
            };
            this.emitStateChange();
            
            // Wait for user action
            return new Promise<ConfirmationResponse>((resolve) => {
                this.pendingActionResolvers.set(actionId, resolve);
            });
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
            
            // Mark current task as completed
            if (this.state.currentTask) {
                this.state.currentTask.status = params.exitStatus === 'success' ? 'completed' : 'failed';
                this.state.currentTask.endTime = new Date();
                this.state.currentTask.exitStatus = params.exitStatus;
                
                // Move to history
                this.state.taskHistory.unshift(this.state.currentTask);
                this.state.currentTask = undefined;
            }
            
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
            
            // Mark current task as failed
            if (this.state.currentTask) {
                this.state.currentTask.status = 'failed';
                this.state.currentTask.endTime = new Date();
                this.state.currentTask.exitStatus = 'error: ' + params.error;
                
                // Move to history
                this.state.taskHistory.unshift(this.state.currentTask);
                this.state.currentTask = undefined;
            }
            
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
        
        // Create new task
        const newTask: Task = {
            id: `task_${Date.now()}`,
            query: task,
            status: 'running',
            startTime: new Date(),
            totalSteps: 0,
            totalCost: 0
        };
        
        this.state = {
            status: 'running',
            currentStep: 0,
            totalCost: 0,
            mode: config.get('defaultMode', 'confirm'),
            messages: [],
            currentTask: newTask,
            taskHistory: this.state.taskHistory
        };
        this.emitStateChange();
        
        // Send initialization - this will block on Python side until agent completes
        this.bridge.sendRequest('agent/waitInitialize', initParams).catch((err) => {
            vscode.window.showErrorMessage(`Failed to initialize agent: ${err}`);
            
            // Mark task as failed
            if (this.state.currentTask) {
                this.state.currentTask.status = 'failed';
                this.state.currentTask.endTime = new Date();
                this.state.taskHistory.unshift(this.state.currentTask);
                this.state.currentTask = undefined;
            }
            
            this.state.status = 'error';
            this.emitStateChange();
        });
    }
    
    stopAgent(): void {
        // Mark current task as cancelled
        if (this.state.currentTask) {
            this.state.currentTask.status = 'cancelled';
            this.state.currentTask.endTime = new Date();
            this.state.taskHistory.unshift(this.state.currentTask);
            this.state.currentTask = undefined;
        }
        
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
    
    /**
     * Handle action approval/rejection from the UI
     */
    handleAction(actionId: string, decision: 'approve' | 'reject', reason?: string, editedCommand?: string): void {
        // Find the action message
        const messageIndex = this.state.messages.findIndex(m => m.actionId === actionId);
        if (messageIndex === -1) {
            vscode.window.showErrorMessage('Action not found');
            return;
        }
        
        const message = this.state.messages[messageIndex];
        
        // Update message status
        message.actionStatus = decision === 'approve' ? 'approved' : 'rejected';
        message.actionTimestamp = new Date();
        if (reason) {
            message.actionReason = reason;
        }
        
        // Clear pending action
        this.state.status = 'running';
        this.state.pendingAction = undefined;
        this.emitStateChange();
        
        // Resolve the promise
        const resolver = this.pendingActionResolvers.get(actionId);
        if (resolver) {
            this.pendingActionResolvers.delete(actionId);
            
            const response: ConfirmationResponse = {
                approved: decision === 'approve',
                reason: reason,
                editedCommand: editedCommand
            };
            
            resolver(response);
        }
    }
    
    getState(): AgentState {
        return { ...this.state };
    }
    
    private emitStateChange() {
        this.stateChangeEmitter.fire({ ...this.state });
    }
}
