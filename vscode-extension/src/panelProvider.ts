import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { AgentState } from './types';

export class PanelProvider {
    private panel?: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];
    
    constructor(
        private context: vscode.ExtensionContext,
        private agentManager: AgentManager
    ) {
        // Listen to state changes
        agentManager.onStateChange(state => {
            this.updatePanel(state);
        });
    }
    
    public isVisible(): boolean {
        return this.panel !== undefined;
    }
    
    public show(taskId?: string) {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.One);
            if (taskId) {
                this.scrollToTask(taskId);
            }
            return;
        }
        
        this.panel = vscode.window.createWebviewPanel(
            'miniSweAgentChat',
            'mini-swe-agent Chat',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(this.context.extensionUri, 'media')
                ]
            }
        );
        
        this.panel.webview.html = this.getHtmlForWebview(this.panel.webview);
        
        // Handle messages from webview
        this.panel.webview.onDidReceiveMessage(
            message => this.handleMessage(message),
            null,
            this.disposables
        );
        
        // Clean up when panel is closed
        this.panel.onDidDispose(
            () => {
                this.panel = undefined;
                this.disposables.forEach(d => d.dispose());
                this.disposables = [];
            },
            null,
            this.disposables
        );
        
        // Send initial state
        this.updatePanel(this.agentManager.getState());
        
        if (taskId) {
            // Wait a bit for webview to load
            setTimeout(() => this.scrollToTask(taskId), 100);
        }
    }
    
    private handleMessage(message: any) {
        switch (message.command) {
            case 'startTask':
                // Start agent from panel
                this.agentManager.startAgent(message.task);
                break;
                
            case 'approveAction':
                this.agentManager.handleAction(message.actionId, 'approve');
                break;
                
            case 'rejectAction':
                vscode.window.showInputBox({
                    prompt: 'Why are you rejecting this action? (optional)',
                    placeHolder: 'e.g., This command is too risky'
                }).then(reason => {
                    if (reason !== undefined) {
                        this.agentManager.handleAction(message.actionId, 'reject', reason);
                    }
                });
                break;
                
            case 'editAction':
                this.handleEditAction(message.actionId);
                break;
                
            case 'setMode':
                this.agentManager.setMode(message.mode);
                break;
        }
    }
    
    private async handleEditAction(actionId: string) {
        const state = this.agentManager.getState();
        const message = state.messages.find(m => m.actionId === actionId);
        
        if (!message || !message.actionCommand) {
            return;
        }
        
        const edited = await vscode.window.showInputBox({
            prompt: 'Edit the command:',
            value: message.actionCommand,
            valueSelection: [0, message.actionCommand.length]
        });
        
        if (edited) {
            this.agentManager.handleAction(actionId, 'approve', undefined, edited);
        }
    }
    
    public scrollToTask(taskId: string) {
        if (this.panel) {
            this.panel.webview.postMessage({
                type: 'scrollToTask',
                taskId: taskId
            });
        }
    }
    
    private updatePanel(state: AgentState) {
        if (this.panel) {
            this.panel.webview.postMessage({
                type: 'stateUpdate',
                state: state
            });
        }
    }
    
    private getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mini-swe-agent Chat</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            background: var(--vscode-editor-background);
        }
        
        .toolbar {
            position: sticky;
            top: 0;
            padding: 12px 16px;
            background: var(--vscode-editor-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-info {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-badge.running {
            background: rgba(0, 255, 0, 0.1);
            color: var(--vscode-charts-green);
        }
        
        .status-badge.waiting {
            background: rgba(255, 255, 0, 0.1);
            color: var(--vscode-charts-yellow);
        }
        
        .metric {
            font-size: 12px;
            color: var(--vscode-descriptionForeground);
        }
        
        .mode-badge {
            padding: 4px 8px;
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            border-radius: 4px;
            font-size: 10px;
            text-transform: uppercase;
        }
        
        .toolbar-actions {
            display: flex;
            gap: 8px;
        }
        
        button {
            padding: 6px 12px;
            font-size: 12px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        
        button.secondary {
            background: transparent;
            border: 1px solid var(--vscode-button-border);
        }
        
        .messages-container {
            padding: 16px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .task-separator {
            margin: 24px 0;
            padding: 12px;
            background: var(--vscode-textBlockQuote-background);
            border-left: 4px solid var(--vscode-charts-purple);
            border-radius: 4px;
        }
        
        .task-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .task-badge {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--vscode-charts-purple);
        }
        
        .task-title {
            font-size: 14px;
            font-weight: 600;
        }
        
        .task-time {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        
        .message {
            margin-bottom: 20px;
            padding: 16px;
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 8px;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .role-badge {
            font-size: 11px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }
        
        .role-badge.user {
            background: var(--vscode-textBlockQuote-background);
        }
        
        .role-badge.agent {
            background: var(--vscode-textCodeBlock-background);
        }
        
        .role-badge.observation {
            background: var(--vscode-editor-inactiveSelectionBackground);
        }
        
        .step-info {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        
        .message-content {
            line-height: 1.6;
        }
        
        .thought-section {
            margin-bottom: 12px;
            padding: 12px;
            background: rgba(100, 100, 255, 0.05);
            border-left: 3px solid var(--vscode-charts-blue);
            border-radius: 4px;
        }
        
        .thought-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--vscode-charts-blue);
            margin-bottom: 6px;
        }
        
        .action-section {
            margin-top: 12px;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 6px;
            overflow: hidden;
        }
        
        .action-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: var(--vscode-editor-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .action-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
        }
        
        .action-status.pending {
            color: var(--vscode-charts-yellow);
        }
        
        .code-block {
            background: var(--vscode-textCodeBlock-background);
        }
        
        .code-header {
            display: flex;
            justify-content: space-between;
            padding: 6px 12px;
            background: rgba(0, 0, 0, 0.2);
            font-size: 11px;
        }
        
        .copy-btn {
            padding: 2px 8px;
            font-size: 10px;
            background: transparent;
            border: 1px solid var(--vscode-button-border);
        }
        
        .code-block pre {
            padding: 12px;
            margin: 0;
            overflow-x: auto;
        }
        
        .code-block code {
            font-family: var(--vscode-editor-font-family);
            font-size: 13px;
            line-height: 1.6;
        }
        
        .action-controls {
            display: flex;
            gap: 8px;
            padding: 12px;
            background: var(--vscode-editor-background);
        }
        
        .action-controls button {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 10px 16px;
        }
        
        .approve-btn {
            background: var(--vscode-button-background);
        }
        
        .approve-btn:hover {
            background: var(--vscode-button-hoverBackground);
            transform: translateY(-1px);
        }
        
        .reject-btn {
            background: transparent;
            border: 1px solid var(--vscode-errorForeground);
            color: var(--vscode-errorForeground);
        }
        
        .reject-btn:hover {
            background: var(--vscode-errorForeground);
            color: var(--vscode-editor-background);
        }
        
        .edit-btn {
            background: transparent;
            border: 1px solid var(--vscode-button-border);
        }
        
        .action-result {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .action-result.approved {
            background: rgba(0, 255, 0, 0.1);
            color: var(--vscode-charts-green);
            border: 1px solid var(--vscode-charts-green);
        }
        
        .action-result.rejected {
            background: rgba(255, 0, 0, 0.1);
            color: var(--vscode-errorForeground);
            border: 1px solid var(--vscode-errorForeground);
        }
        
        .action-result.auto-approved {
            background: rgba(255, 255, 0, 0.1);
            color: var(--vscode-charts-yellow);
            border: 1px solid var(--vscode-charts-yellow);
        }
        
        .output-section pre {
            padding: 12px;
            background: var(--vscode-textCodeBlock-background);
            border-radius: 4px;
            overflow-x: auto;
            font-family: var(--vscode-editor-font-family);
            font-size: 13px;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--vscode-descriptionForeground);
        }
        
        .empty-state-icon {
            font-size: 64px;
            margin-bottom: 16px;
        }
        
        /* Task input area styles */
        .task-input-area {
            padding: 40px 20px;
            max-width: 900px;
            margin: 0 auto;
        }
        
        .task-input-card {
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 12px;
            padding: 32px;
        }
        
        .task-input-card h2 {
            margin: 0 0 10px 0;
            font-size: 24px;
            color: var(--vscode-foreground);
        }
        
        .task-input-card .hint {
            margin: 0 0 20px 0;
            color: var(--vscode-descriptionForeground);
            font-size: 14px;
        }
        
        #task-input {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            font-family: var(--vscode-font-family);
            font-size: 14px;
            line-height: 1.5;
            resize: vertical;
            margin-bottom: 16px;
            border-radius: 4px;
        }
        
        #task-input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }
        
        #start-task-btn {
            width: 100%;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            cursor: pointer;
            border-radius: 6px;
        }
        
        #start-task-btn:hover {
            background: var(--vscode-button-hoverBackground);
        }
        
        #start-task-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .task-examples {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--vscode-panel-border);
            color: var(--vscode-descriptionForeground);
            font-size: 13px;
            line-height: 1.8;
        }
        
        .task-examples strong {
            display: block;
            margin-bottom: 8px;
        }
        
        /* Message collapse styles */
        .message-preview {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .message-full {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .expand-btn {
            margin-top: 8px;
            padding: 4px 12px;
            font-size: 11px;
            background: transparent;
            border: 1px solid var(--vscode-button-border);
            cursor: pointer;
            color: var(--vscode-textLink-foreground);
            border-radius: 3px;
        }
        
        .expand-btn:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }
        
        .thought-collapsed {
            position: relative;
        }
        
        .thought-preview {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .thought-full {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <div class="status-info">
            <span class="status-badge" id="status-badge">idle</span>
            <span class="metric">Step: <span id="current-step">0</span></span>
            <span class="metric">Cost: <span id="total-cost">$0.00</span></span>
            <span class="mode-badge" id="mode-badge">confirm</span>
        </div>
        <div class="toolbar-actions">
            <button onclick="setMode('confirm')">Confirm</button>
            <button onclick="setMode('yolo')">YOLO</button>
            <button onclick="setMode('human')">Human</button>
        </div>
    </div>
    
    <!-- Task input area (shown when idle with no messages) -->
    <div class="task-input-area" id="task-input-area" style="display: none;">
        <div class="task-input-card">
            <h2>🚀 Start New Task</h2>
            <p class="hint">Describe what you want the agent to help you with</p>
            <textarea 
                id="task-input" 
                placeholder="e.g., Fix the bug in user_auth.py that causes login failures..."
                rows="6"
            ></textarea>
            <button id="start-task-btn">▶ Start Agent</button>
            <div class="task-examples">
                <strong>Examples:</strong>
                • Optimize the device_binary_search function in the kernel<br>
                • Add error handling to the API endpoints<br>
                • Write unit tests for the authentication module
            </div>
        </div>
    </div>
    
    <!-- Messages container (shown when agent is running or has history) -->
    <div class="messages-container" id="messages-container" style="display: none;">
    </div>
    
    <!-- Empty state (fallback) -->
    <div class="empty-state" id="empty-state">
        <div class="empty-state-icon">🤖</div>
        <p><strong>mini-swe-agent</strong></p>
        <p>Click "Show Full Chat" to start a new task</p>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        let currentState = null;
        
        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'stateUpdate') {
                currentState = message.state;
                updateUI(message.state);
            } else if (message.type === 'scrollToTask') {
                scrollToTask(message.taskId);
            }
        });
        
        // Task input button handler
        document.getElementById('start-task-btn').addEventListener('click', () => {
            const taskInput = document.getElementById('task-input');
            const task = taskInput.value.trim();
            if (!task) {
                return;
            }
            
            // Send task to extension
            vscode.postMessage({
                command: 'startTask',
                task: task
            });
            
            // Disable button to prevent double-click
            document.getElementById('start-task-btn').disabled = true;
            document.getElementById('start-task-btn').textContent = '⏳ Starting...';
        });
        
        // Support Ctrl+Enter to submit
        document.getElementById('task-input').addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                document.getElementById('start-task-btn').click();
            }
        });
        
        function updateUI(state) {
            // Update toolbar
            const statusBadge = document.getElementById('status-badge');
            statusBadge.textContent = state.status;
            statusBadge.className = 'status-badge ' + (state.status === 'running' ? 'running' : 
                                                      state.status.includes('waiting') ? 'waiting' : '');
            
            document.getElementById('current-step').textContent = state.currentStep || 0;
            document.getElementById('total-cost').textContent = '$' + (state.totalCost || 0).toFixed(2);
            document.getElementById('mode-badge').textContent = state.mode;
            
            // Update messages and UI areas
            updateUIAreas(state);
            updateMessages(state);
        }
        
        function updateUIAreas(state) {
            const taskInputArea = document.getElementById('task-input-area');
            const messagesContainer = document.getElementById('messages-container');
            const emptyState = document.getElementById('empty-state');
            const taskInput = document.getElementById('task-input');
            const startBtn = document.getElementById('start-task-btn');
            
            // Determine which area to show
            if (state.status === 'idle' && (!state.messages || state.messages.length === 0)) {
                // Show task input area
                taskInputArea.style.display = 'block';
                messagesContainer.style.display = 'none';
                emptyState.style.display = 'none';
                
                // Reset input if idle
                taskInput.value = '';
                startBtn.disabled = false;
                startBtn.textContent = '▶ Start Agent';
                
                // Auto-focus on input
                setTimeout(() => taskInput.focus(), 100);
            } else if (state.messages && state.messages.length > 0) {
                // Show messages
                taskInputArea.style.display = 'none';
                messagesContainer.style.display = 'block';
                emptyState.style.display = 'none';
            } else {
                // Show empty state (fallback)
                taskInputArea.style.display = 'none';
                messagesContainer.style.display = 'none';
                emptyState.style.display = 'flex';
            }
        }
        
        function updateMessages(state) {
            const container = document.getElementById('messages-container');
            
            if (!state.messages || state.messages.length === 0) {
                container.innerHTML = '';
                return;
            }
            
            let html = '';
            let currentTaskId = null;
            
            state.messages.forEach((msg, index) => {
                // Add task separator for user messages
                if (msg.role === 'user' && state.taskHistory) {
                    const task = state.taskHistory.find(t => 
                        new Date(t.startTime).getTime() <= new Date(msg.timestamp).getTime()
                    );
                    if (task && task.id !== currentTaskId) {
                        currentTaskId = task.id;
                        html += \`
                            <div class="task-separator" data-task-id="\${task.id}">
                                <div class="task-header">
                                    <span class="task-badge">Task</span>
                                    <span class="task-title">\${escapeHtml(task.query)}</span>
                                    <span class="task-time">\${formatDateTime(task.startTime)}</span>
                                </div>
                            </div>
                        \`;
                    }
                }
                
                html += renderMessage(msg);
            });
            
            container.innerHTML = html;
            container.scrollTop = container.scrollHeight;
        }
        
        function renderMessage(msg) {
            const roleClass = msg.role === 'user' ? 'user' : 
                             msg.role === 'assistant' ? 'agent' : 'observation';
            const roleName = msg.role === 'assistant' ? '🤖 AGENT' :
                            msg.role === 'user' ? '👤 USER' : '📋 OBSERVATION';
            
            const COLLAPSE_THRESHOLD = 150;
            let contentHTML = '';
            
            if (msg.isAction) {
                // This is an action message with approve/reject buttons
                const thought = extractThought(msg.content);
                const command = msg.actionCommand || extractCommand(msg.content);
                
                // Render thought with collapse if long
                let thoughtHTML = '';
                if (thought) {
                    if (thought.length > COLLAPSE_THRESHOLD) {
                        const preview = thought.substring(0, COLLAPSE_THRESHOLD);
                        thoughtHTML = \`
                            <div class="thought-section thought-collapsed">
                                <div class="thought-label">💭 THOUGHT:</div>
                                <div class="thought-preview">\${escapeHtml(preview)}...</div>
                                <div class="thought-full" style="display: none;">\${escapeHtml(thought)}</div>
                                <button class="expand-btn" onclick="toggleThought(this)">▼ Show more</button>
                            </div>
                        \`;
                    } else {
                        thoughtHTML = \`
                            <div class="thought-section">
                                <div class="thought-label">💭 THOUGHT:</div>
                                \${escapeHtml(thought)}
                            </div>
                        \`;
                    }
                }
                
                // Action section always fully expanded
                contentHTML = \`
                    \${thoughtHTML}
                    <div class="action-section">
                        <div class="action-label">
                            <span>⚡ PROPOSED ACTION</span>
                            \${msg.actionStatus === 'pending' ? '<span class="action-status pending">● Waiting for approval</span>' : ''}
                        </div>
                        <div class="code-block">
                            <div class="code-header">
                                <span>bash</span>
                                <button class="copy-btn" onclick="copyCode('\${escapeHtml(command)}')">📋 Copy</button>
                            </div>
                            <pre><code>\${escapeHtml(command)}</code></pre>
                        </div>
                        \${renderActionControls(msg)}
                    </div>
                \`;
            } else {
                // Regular message with collapse if long
                const content = msg.content;
                if (content.length > COLLAPSE_THRESHOLD) {
                    const preview = content.substring(0, COLLAPSE_THRESHOLD);
                    contentHTML = \`
                        <div class="message-preview">\${escapeHtml(preview).replace(/\\n/g, '<br>')}...</div>
                        <div class="message-full" style="display: none;">\${escapeHtml(content).replace(/\\n/g, '<br>')}</div>
                        <button class="expand-btn" onclick="toggleMessageContent(this)">▼ Show more</button>
                    \`;
                } else {
                    contentHTML = escapeHtml(content).replace(/\\n/g, '<br>');
                }
            }
            
            return \`
                <div class="message" data-action-id="\${msg.actionId || ''}">
                    <div class="message-header">
                        <span class="role-badge \${roleClass}">\${roleName}</span>
                        <span class="step-info">
                            \${msg.step ? 'Step ' + msg.step : ''}
                            \${msg.cost ? ' | $' + msg.cost.toFixed(2) : ''}
                            \${msg.timestamp ? ' | ' + formatTime(msg.timestamp) : ''}
                        </span>
                    </div>
                    <div class="message-content">\${contentHTML}</div>
                </div>
            \`;
        }
        
        function renderActionControls(msg) {
            if (msg.actionStatus === 'pending') {
                return \`
                    <div class="action-controls">
                        <button class="approve-btn" onclick="approveAction('\${msg.actionId}')">
                            <span>✓</span><span>Approve</span>
                        </button>
                        <button class="reject-btn" onclick="rejectAction('\${msg.actionId}')">
                            <span>✗</span><span>Reject</span>
                        </button>
                        <button class="edit-btn" onclick="editAction('\${msg.actionId}')">
                            <span>✏</span><span>Edit</span>
                        </button>
                    </div>
                \`;
            } else if (msg.actionStatus === 'approved') {
                return \`
                    <div class="action-controls">
                        <div class="action-result approved">
                            <span>✓</span>
                            <span>Approved by you</span>
                            \${msg.actionTimestamp ? '<span style="margin-left: auto;">' + formatTime(msg.actionTimestamp) + '</span>' : ''}
                        </div>
                    </div>
                \`;
            } else if (msg.actionStatus === 'rejected') {
                return \`
                    <div class="action-controls">
                        <div class="action-result rejected">
                            <span>✗</span>
                            <span>Rejected\${msg.actionReason ? ' (' + escapeHtml(msg.actionReason) + ')' : ''}</span>
                        </div>
                    </div>
                \`;
            } else if (msg.actionStatus === 'auto-approved') {
                return \`
                    <div class="action-controls">
                        <div class="action-result auto-approved">
                            <span>⚡</span>
                            <span>Auto-approved (YOLO mode)</span>
                        </div>
                    </div>
                \`;
            }
            return '';
        }
        
        function approveAction(actionId) {
            vscode.postMessage({ command: 'approveAction', actionId });
        }
        
        function rejectAction(actionId) {
            vscode.postMessage({ command: 'rejectAction', actionId });
        }
        
        function editAction(actionId) {
            vscode.postMessage({ command: 'editAction', actionId });
        }
        
        function setMode(mode) {
            vscode.postMessage({ command: 'setMode', mode });
        }
        
        function copyCode(code) {
            navigator.clipboard.writeText(code);
        }
        
        function scrollToTask(taskId) {
            const element = document.querySelector('[data-task-id="' + taskId + '"]');
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function extractThought(content) {
            const match = content.match(/THOUGHT[:${'\\s'}]+([\\s\\S]*?)\`\`\`/i);
            return match ? match[1].trim() : '';
        }
        
        function extractCommand(content) {
            const match = content.match(/\`\`\`bash\\s*\\n([\\s\\S]*?)\\n\`\`\`/);
            return match ? match[1].trim() : content;
        }
        
        function formatTime(dateString) {
            const date = new Date(dateString);
            return date.toLocaleTimeString();
        }
        
        function formatDateTime(dateString) {
            const date = new Date(dateString);
            return date.toLocaleString();
        }
        
        function toggleMessageContent(button) {
            const messageContent = button.parentElement;
            const preview = messageContent.querySelector('.message-preview');
            const full = messageContent.querySelector('.message-full');
            const isExpanded = full.style.display !== 'none';
            
            if (isExpanded) {
                preview.style.display = 'block';
                full.style.display = 'none';
                button.innerHTML = '▼ Show more';
            } else {
                preview.style.display = 'none';
                full.style.display = 'block';
                button.innerHTML = '▲ Show less';
            }
        }
        
        function toggleThought(button) {
            const thoughtSection = button.parentElement;
            const preview = thoughtSection.querySelector('.thought-preview');
            const full = thoughtSection.querySelector('.thought-full');
            const isExpanded = full.style.display !== 'none';
            
            if (isExpanded) {
                preview.style.display = 'block';
                full.style.display = 'none';
                button.innerHTML = '▼ Show more';
            } else {
                preview.style.display = 'none';
                full.style.display = 'block';
                button.innerHTML = '▲ Show less';
            }
        }
    </script>
</body>
</html>`;
    }
}

