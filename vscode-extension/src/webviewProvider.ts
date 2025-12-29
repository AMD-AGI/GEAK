import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { AgentState } from './types';

export class WebviewProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;
    
    constructor(
        private context: vscode.ExtensionContext,
        private agentManager: AgentManager
    ) {
        // Listen to state changes
        agentManager.onStateChange(state => {
            this.updateWebview(state);
        });
    }
    
    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        token: vscode.CancellationToken
    ): void {
        this.view = webviewView;
        
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.context.extensionUri, 'media'),
                vscode.Uri.joinPath(this.context.extensionUri, 'webview')
            ]
        };
        
        webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
        
        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(message => {
            switch (message.command) {
                case 'start':
                    vscode.commands.executeCommand('mini-swe-agent.start');
                    break;
                case 'stop':
                    vscode.commands.executeCommand('mini-swe-agent.stop');
                    break;
                case 'setMode':
                    this.agentManager.setMode(message.mode);
                    break;
            }
        });
        
        // Send initial state
        this.updateWebview(this.agentManager.getState());
    }
    
    private updateWebview(state: AgentState) {
        if (this.view) {
            this.view.webview.postMessage({
                type: 'stateUpdate',
                state
            });
        }
    }
    
    private getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mini-swe-agent</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            padding: 0;
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            background: var(--vscode-editor-background);
        }
        
        .status-bar {
            position: sticky;
            top: 0;
            padding: 12px;
            background: var(--vscode-sideBar-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            z-index: 100;
        }
        
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .status-row:last-child {
            margin-bottom: 0;
        }
        
        .status-label {
            font-weight: 600;
            margin-right: 8px;
        }
        
        .status-value {
            color: var(--vscode-descriptionForeground);
        }
        
        .status-value.running {
            color: var(--vscode-charts-green);
        }
        
        .status-value.waiting {
            color: var(--vscode-charts-yellow);
        }
        
        .status-value.error {
            color: var(--vscode-errorForeground);
        }
        
        .mode-buttons {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 13px;
            flex: 1;
        }
        
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        
        button:active {
            background: var(--vscode-button-activeBackground);
        }
        
        button.active {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        
        .messages-container {
            padding: 12px;
            overflow-y: auto;
        }
        
        .message {
            margin-bottom: 16px;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid transparent;
        }
        
        .message.user {
            background: var(--vscode-textBlockQuote-background);
            border-left-color: var(--vscode-textBlockQuote-border);
        }
        
        .message.assistant {
            background: var(--vscode-textCodeBlock-background);
            border-left-color: var(--vscode-charts-blue);
        }
        
        .message.system {
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-left-color: var(--vscode-charts-purple);
        }
        
        .message-header {
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .message-meta {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        
        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: var(--vscode-editor-font-family);
            font-size: 13px;
            line-height: 1.5;
        }
        
        .message-content code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--vscode-descriptionForeground);
        }
        
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .empty-state-text {
            font-size: 14px;
            line-height: 1.6;
        }
        
        .start-button {
            margin-top: 16px;
            padding: 8px 16px;
        }
    </style>
</head>
<body>
    <div class="status-bar">
        <div class="status-row">
            <div>
                <span class="status-label">Status:</span>
                <span class="status-value" id="status">idle</span>
            </div>
            <div>
                <span class="status-label">Step:</span>
                <span class="status-value" id="step">0</span>
                <span class="status-label" style="margin-left: 12px;">Cost:</span>
                <span class="status-value" id="cost">$0.00</span>
            </div>
        </div>
        <div class="status-row">
            <span class="status-label">Mode:</span>
            <span class="status-value" id="mode">confirm</span>
        </div>
        <div class="mode-buttons">
            <button id="btn-confirm" onclick="setMode('confirm')">Confirm</button>
            <button id="btn-yolo" onclick="setMode('yolo')">YOLO</button>
            <button id="btn-human" onclick="setMode('human')">Human</button>
        </div>
    </div>
    
    <div id="messages-container" class="messages-container">
        <div class="empty-state">
            <div class="empty-state-icon">🤖</div>
            <div class="empty-state-text">
                <p><strong>mini-swe-agent</strong></p>
                <p>AI coding agent ready to help!</p>
                <p style="margin-top: 8px;">Click "Start Agent" command to begin.</p>
            </div>
        </div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        
        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'stateUpdate') {
                updateUI(message.state);
            }
        });
        
        function updateUI(state) {
            // Update status
            const statusEl = document.getElementById('status');
            statusEl.textContent = state.status;
            statusEl.className = 'status-value';
            
            if (state.status === 'running') {
                statusEl.classList.add('running');
            } else if (state.status.includes('waiting')) {
                statusEl.classList.add('waiting');
            } else if (state.status === 'error') {
                statusEl.classList.add('error');
            }
            
            // Update step and cost
            document.getElementById('step').textContent = state.currentStep;
            document.getElementById('cost').textContent = '$' + state.totalCost.toFixed(2);
            
            // Update mode
            document.getElementById('mode').textContent = state.mode;
            
            // Update mode buttons
            document.querySelectorAll('.mode-buttons button').forEach(btn => {
                btn.classList.remove('active');
            });
            const activeBtn = document.getElementById('btn-' + state.mode);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
            
            // Update messages
            const container = document.getElementById('messages-container');
            
            if (state.messages.length === 0) {
                // Show empty state
                container.innerHTML = \`
                    <div class="empty-state">
                        <div class="empty-state-icon">🤖</div>
                        <div class="empty-state-text">
                            <p><strong>mini-swe-agent</strong></p>
                            <p>AI coding agent ready to help!</p>
                            <p style="margin-top: 8px;">Use Command Palette (Ctrl+Shift+P) and run "mini: Start Agent"</p>
                        </div>
                    </div>
                \`;
            } else {
                // Show messages
                container.innerHTML = '';
                state.messages.forEach(msg => {
                    const div = document.createElement('div');
                    div.className = 'message ' + msg.role;
                    
                    const header = document.createElement('div');
                    header.className = 'message-header';
                    
                    const roleName = msg.role === 'assistant' ? 'mini-swe-agent' : msg.role;
                    header.innerHTML = '<span>' + roleName.toUpperCase() + '</span>';
                    
                    if (msg.step !== undefined) {
                        const meta = document.createElement('span');
                        meta.className = 'message-meta';
                        meta.textContent = 'Step ' + msg.step;
                        if (msg.cost !== undefined) {
                            meta.textContent += ' • $' + msg.cost.toFixed(2);
                        }
                        header.appendChild(meta);
                    }
                    
                    const content = document.createElement('div');
                    content.className = 'message-content';
                    content.textContent = msg.content;
                    
                    div.appendChild(header);
                    div.appendChild(content);
                    container.appendChild(div);
                });
                
                // Scroll to bottom
                container.scrollTop = container.scrollHeight;
            }
        }
        
        function setMode(mode) {
            vscode.postMessage({ command: 'setMode', mode: mode });
        }
    </script>
</body>
</html>`;
    }
}

