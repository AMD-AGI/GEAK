import * as cp from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';
import { JSONRPCMessage } from './types';

export class PythonBridge {
    private process: cp.ChildProcess | null = null;
    private requestId = 0;
    private pendingRequests = new Map<number, {
        resolve: (value: any) => void;
        reject: (reason: any) => void;
    }>();
    private messageHandlers = new Map<string, (params: any) => void>();
    private requestHandlers = new Map<string, (params: any) => any>();
    private outputChannel: vscode.OutputChannel;
    
    constructor(private context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('mini-swe-agent');
    }
    
    async start(): Promise<void> {
        const pythonPath = await this.findPython();
        const scriptPath = path.join(
            this.context.extensionPath,
            'python',
            'main.py'
        );
        
        // Add the parent directory to PYTHONPATH so vscode_agent can be imported
        const pythonDir = path.join(this.context.extensionPath, 'python');
        const env = { ...process.env };
        env.PYTHONPATH = pythonDir + (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '');
        
        this.outputChannel.appendLine(`Starting Python process: ${pythonPath} ${scriptPath}`);
        this.outputChannel.appendLine(`PYTHONPATH: ${env.PYTHONPATH}`);
        
        this.process = cp.spawn(pythonPath, [scriptPath], {
            cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
            env
        });
        
        // Handle stdout (JSON-RPC messages)
        this.process.stdout?.on('data', (data: Buffer) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const message = JSON.parse(line);
                        this.handleMessage(message);
                    } catch (e) {
                        this.outputChannel.appendLine(`Failed to parse JSON-RPC: ${line}`);
                        this.outputChannel.appendLine(`Error: ${e}`);
                    }
                }
            }
        });
        
        // Handle stderr (errors and debug output)
        this.process.stderr?.on('data', (data: Buffer) => {
            this.outputChannel.appendLine(`Python stderr: ${data.toString()}`);
        });
        
        // Handle process exit
        this.process.on('exit', (code) => {
            this.outputChannel.appendLine(`Python process exited with code ${code}`);
            this.process = null;
            
            // Reject all pending requests
            this.pendingRequests.forEach(({ reject }) => {
                reject(new Error('Python process exited'));
            });
            this.pendingRequests.clear();
        });
        
        // Wait a bit for process to start
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    private handleMessage(message: JSONRPCMessage) {
        if (message.id !== undefined && this.pendingRequests.has(message.id)) {
            // This is a response to our request
            const pending = this.pendingRequests.get(message.id)!;
            if (message.error) {
                pending.reject(new Error(message.error.message));
            } else {
                pending.resolve(message.result);
            }
            this.pendingRequests.delete(message.id);
        } else if (message.method) {
            // This is a notification or request from Python
            if (message.id !== undefined) {
                // Request - need to send response
                const handler = this.requestHandlers.get(message.method);
                if (handler) {
                    Promise.resolve(handler(message.params))
                        .then(result => {
                            this.sendResponse(message.id!, result);
                        })
                        .catch(error => {
                            this.sendError(message.id!, error);
                        });
                } else {
                    this.outputChannel.appendLine(`No handler for request: ${message.method}`);
                    this.sendError(message.id!, new Error(`No handler for ${message.method}`));
                }
            } else {
                // Notification - no response needed
                const handler = this.messageHandlers.get(message.method);
                if (handler) {
                    handler(message.params);
                } else {
                    this.outputChannel.appendLine(`No handler for notification: ${message.method}`);
                }
            }
        }
    }
    
    sendNotification(method: string, params?: any): void {
        if (!this.process) {
            this.outputChannel.appendLine('Cannot send notification: Python process not running');
            return;
        }
        
        const message: JSONRPCMessage = {
            jsonrpc: '2.0',
            method,
            params
        };
        
        this.process.stdin?.write(JSON.stringify(message) + '\n');
    }
    
    sendRequest(method: string, params?: any): Promise<any> {
        return new Promise((resolve, reject) => {
            if (!this.process) {
                reject(new Error('Python process not running'));
                return;
            }
            
            this.requestId++;
            const id = this.requestId;
            
            this.pendingRequests.set(id, { resolve, reject });
            
            const message: JSONRPCMessage = {
                jsonrpc: '2.0',
                id,
                method,
                params
            };
            
            this.process.stdin?.write(JSON.stringify(message) + '\n');
        });
    }
    
    private sendResponse(id: number, result: any): void {
        if (!this.process) {
            return;
        }
        
        const message: JSONRPCMessage = {
            jsonrpc: '2.0',
            id,
            result
        };
        
        this.process.stdin?.write(JSON.stringify(message) + '\n');
    }
    
    private sendError(id: number, error: any): void {
        if (!this.process) {
            return;
        }
        
        const message: JSONRPCMessage = {
            jsonrpc: '2.0',
            id,
            error: {
                code: -1,
                message: String(error)
            }
        };
        
        this.process.stdin?.write(JSON.stringify(message) + '\n');
    }
    
    onNotification(method: string, handler: (params: any) => void): void {
        this.messageHandlers.set(method, handler);
    }
    
    onRequest(method: string, handler: (params: any) => any): void {
        this.requestHandlers.set(method, handler);
    }
    
    stop(): void {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
    }
    
    private async findPython(): Promise<string> {
        // First check user configuration
        const config = vscode.workspace.getConfiguration('mini-swe-agent');
        const configuredPath = config.get<string>('pythonPath');
        if (configuredPath) {
            return configuredPath;
        }
        
        // Try to use Python extension's Python
        try {
            const pythonExt = vscode.extensions.getExtension('ms-python.python');
            if (pythonExt) {
                await pythonExt.activate();
                const execCommand = pythonExt.exports?.settings?.getExecutionDetails?.()?.execCommand;
                if (execCommand) {
                    // execCommand can be a string or an array like ['python', '-u']
                    const pythonPath = Array.isArray(execCommand) ? execCommand[0] : execCommand;
                    this.outputChannel.appendLine(`Using Python from Python extension: ${pythonPath}`);
                    return pythonPath;
                }
            }
        } catch (e) {
            this.outputChannel.appendLine(`Failed to get Python from extension: ${e}`);
        }
        
        // Default to 'python3'
        this.outputChannel.appendLine('Using default python3');
        return 'python3';
    }
}

