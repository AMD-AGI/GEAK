/**
 * Type definitions for mini-swe-agent VS Code extension
 */

export interface Task {
    id: string;
    query: string;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    startTime: Date;
    endTime?: Date;
    totalSteps?: number;
    totalCost?: number;
    exitStatus?: string;
}

export interface AgentState {
    status: 'idle' | 'running' | 'waiting_approval' | 'waiting_input' | 'waiting_strategy' | 'finished' | 'error';
    currentStep: number;
    totalCost: number;
    mode: 'confirm' | 'yolo' | 'human';
    messages: AgentMessage[];
    currentTask?: Task;
    taskHistory: Task[];
    pendingAction?: {
        action: string;
        actionId: string;
        step: number;
    };
    currentStrategies: Strategy[];
    waitingForStrategySelection: boolean;
}

export interface AgentMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    step?: number;
    cost?: number;
    timestamp?: Date;
    
    // Action related
    isAction?: boolean;
    actionId?: string;
    actionStatus?: 'pending' | 'approved' | 'rejected' | 'auto-approved';
    actionCommand?: string;
    actionReason?: string;
    actionTimestamp?: Date;
}

export interface JSONRPCMessage {
    jsonrpc: string;
    id?: number;
    method?: string;
    params?: any;
    result?: any;
    error?: {
        code: number;
        message: string;
    };
}

export interface InitializeParams {
    workspacePath: string;
    modelName?: string;
    task: string;
    configFilePath?: string;
    config: {
        agent?: {
            mode?: string;
            cost_limit?: number;
            step_limit?: number;
            whitelist_actions?: string[];
        };
        model?: Record<string, any>;
        env?: Record<string, any>;
    };
}

export interface ConfirmationRequest {
    action: string;
    mode: string;
}

export interface ConfirmationResponse {
    approved?: boolean;
    switchMode?: boolean;
    newMode?: string;
    reason?: string;
    editedCommand?: string;
}

export interface Strategy {
    id: string;
    title: string;
    description: string;
    reasoning?: string;
    isRecommended?: boolean;
}

export interface StrategySelectionRequest {
    strategies: Strategy[];
    step: number;
    cost: number;
}

export interface StrategySelectionResponse {
    selectedStrategyId: string | null;
    action: 'select' | 'skip';
}

export interface StrategyGeneratedNotification {
    strategies: Strategy[];
    autoSelected: string;
    mode: 'auto';
    step: number;
    cost: number;
}

