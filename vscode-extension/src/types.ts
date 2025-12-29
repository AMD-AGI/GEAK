/**
 * Type definitions for mini-swe-agent VS Code extension
 */

export interface AgentState {
    status: 'idle' | 'running' | 'waiting_approval' | 'waiting_input' | 'finished' | 'error';
    currentStep: number;
    totalCost: number;
    mode: 'confirm' | 'yolo' | 'human';
    messages: AgentMessage[];
    pendingAction?: {
        action: string;
        step: number;
    };
}

export interface AgentMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    step?: number;
    cost?: number;
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
}

