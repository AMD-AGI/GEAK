import * as vscode from 'vscode';
import { AgentManager } from './agentManager';
import { AgentState } from './types';

export class SidebarProvider implements vscode.WebviewViewProvider {
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
                vscode.Uri.joinPath(this.context.extensionUri, 'media')
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
                case 'showPanel':
                    vscode.commands.executeCommand('mini-swe-agent.showPanel');
                    break;
                case 'showTask':
                    vscode.commands.executeCommand('mini-swe-agent.showPanel', message.taskId);
                    break;
                case 'selectStrategy':
                    this.agentManager.handleStrategySelection(message.strategyId, 'select');
                    break;
                case 'skipStrategy':
                    this.agentManager.handleStrategySelection(null, 'skip');
                    break;
            }
        });
        
        // Send initial state
        this.updateWebview(this.agentManager.getState());
    }
    
    private updateWebview(state: AgentState) {
        if (this.view) {
            const sidebarData = this.extractSidebarData(state);
            this.view.webview.postMessage({
                type: 'stateUpdate',
                state: sidebarData
            });
        }
    }
    
    private extractSidebarData(state: AgentState) {
        return {
            status: state.status,
            currentStep: state.currentStep,
            totalCost: state.totalCost,
            mode: state.mode,
            currentTask: state.currentTask,
            recentTasks: state.taskHistory.slice(0, 5),
            hasPendingAction: !!state.pendingAction,
            currentStrategies: state.currentStrategies,
            waitingForStrategySelection: state.waitingForStrategySelection
        };
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
            padding: 12px;
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 6px;
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--vscode-descriptionForeground);
        }
        
        .status-dot.running {
            background: var(--vscode-charts-green);
            animation: pulse 2s infinite;
        }
        
        .status-dot.waiting {
            background: var(--vscode-charts-yellow);
            animation: pulse 2s infinite;
        }
        
        .status-dot.error {
            background: var(--vscode-errorForeground);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-text {
            font-weight: 600;
            text-transform: capitalize;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .metric-card {
            padding: 8px;
            background: var(--vscode-editor-background);
            border-radius: 4px;
            border: 1px solid var(--vscode-panel-border);
        }
        
        .metric-label {
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 14px;
            font-weight: 600;
        }
        
        .control-buttons {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            margin-bottom: 8px;
        }
        
        .mode-buttons {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            margin-bottom: 12px;
        }
        
        button {
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 500;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        
        button.secondary {
            background: transparent;
            border: 1px solid var(--vscode-button-border);
        }
        
        button.active {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        
        .section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 16px 0 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .task-count {
            color: var(--vscode-descriptionForeground);
        }
        
        .task-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .strategy-section {
            margin: 16px 0;
            padding: 12px;
            background: var(--vscode-textBlockQuote-background);
            border-radius: 4px;
            border: 1px solid var(--vscode-charts-yellow);
        }
        
        .strategy-list {
            margin: 12px 0;
        }
        
        .strategy-item {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-editor-background);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .strategy-item:hover {
            background: var(--vscode-list-hoverBackground);
            border-color: var(--vscode-focusBorder);
        }
        
        .strategy-item.selected {
            background: var(--vscode-list-activeSelectionBackground);
            border-color: var(--vscode-focusBorder);
        }
        
        .strategy-item.recommended {
            border-color: var(--vscode-charts-green);
        }
        
        .strategy-title {
            font-weight: bold;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .strategy-description {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 4px;
        }
        
        .strategy-reasoning {
            font-size: 0.85em;
            opacity: 0.6;
            font-style: italic;
        }
        
        .strategy-buttons {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        
        .strategy-buttons button {
            flex: 1;
        }
        
        .task-card {
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid transparent;
            background: var(--vscode-editor-background);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .task-card:hover {
            background: var(--vscode-list-hoverBackground);
        }
        
        .task-card.current {
            border-left-color: var(--vscode-charts-green);
            background: var(--vscode-textBlockQuote-background);
        }
        
        .task-card.completed {
            border-left-color: var(--vscode-charts-blue);
        }
        
        .task-card.failed {
            border-left-color: var(--vscode-errorForeground);
        }
        
        .task-status {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-bottom: 6px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .task-status.running {
            color: var(--vscode-charts-green);
        }
        
        .task-status.completed {
            color: var(--vscode-charts-blue);
        }
        
        .task-status.failed {
            color: var(--vscode-errorForeground);
        }
        
        .task-query {
            font-size: 12px;
            line-height: 1.4;
            margin-bottom: 6px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        
        .task-meta {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
        }
        
        .show-full-btn {
            width: 100%;
            margin-top: 12px;
            padding: 10px;
        }
        
        .empty-state {
            text-align: center;
            padding: 20px;
            color: var(--vscode-descriptionForeground);
        }
        
        .action-pending-notice {
            padding: 8px 12px;
            margin: 12px 0;
            background: rgba(255, 255, 0, 0.1);
            border: 1px solid var(--vscode-charts-yellow);
            border-radius: 4px;
            font-size: 11px;
            color: var(--vscode-charts-yellow);
            text-align: center;
        }
    </style>
</head>
<body>
    <!-- Status Indicator -->
    <div class="status-indicator">
        <div class="status-dot" id="status-dot"></div>
        <div class="status-text" id="status-text">Idle</div>
            </div>
    
    <!-- Metrics -->
    <div class="metrics">
        <div class="metric-card">
            <span class="metric-label">Step</span>
            <span class="metric-value" id="metric-step">0</span>
            </div>
        <div class="metric-card">
            <span class="metric-label">Cost</span>
            <span class="metric-value" id="metric-cost">$0.00</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">Mode</span>
            <span class="metric-value" id="metric-mode">confirm</span>
        </div>
    </div>
    
    <!-- Action Pending Notice -->
    <div class="action-pending-notice" id="action-notice" style="display: none;">
        ⚡ Action waiting for approval
    </div>
    
    <!-- Quick Actions -->
    <div class="control-buttons">
        <button onclick="startAgent()">▶ Start</button>
        <button onclick="stopAgent()">⏸ Stop</button>
    </div>
    
        <div class="mode-buttons">
        <button id="btn-yolo" onclick="setMode('yolo')" class="secondary">⚡ YOLO</button>
        <button id="btn-human" onclick="setMode('human')" class="secondary">👤 Human</button>
    </div>
    
    <!-- Strategy Selection (shown when waiting) -->
    <div id="strategy-section" class="strategy-section" style="display: none;">
        <div class="section-title">
            <span>⚡ Choose a Strategy</span>
        </div>
        <div id="strategy-list" class="strategy-list"></div>
        <div class="strategy-buttons">
            <button onclick="confirmStrategy()" class="primary">✓ Confirm</button>
            <button onclick="skipStrategy()" class="secondary">→ Let LLM Choose</button>
        </div>
    </div>
    
    <!-- Recent Tasks -->
    <div class="section-title">
        <span>Recent Tasks</span>
        <span class="task-count" id="task-count">(0)</span>
    </div>
    
    <div class="task-list" id="task-list">
        <div class="empty-state">
            No tasks yet
        </div>
    </div>
    
    <!-- Show Full Chat Button -->
    <button class="show-full-btn" onclick="showPanel()">
        📖 Show Full Chat
    </button>
    
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
            const statusDot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            statusDot.className = 'status-dot';
            
            if (state.status === 'running') {
                statusDot.classList.add('running');
                statusText.textContent = 'Running';
            } else if (state.status.includes('waiting')) {
                statusDot.classList.add('waiting');
                statusText.textContent = 'Waiting';
            } else if (state.status === 'error') {
                statusDot.classList.add('error');
                statusText.textContent = 'Error';
            } else {
                statusText.textContent = 'Idle';
            }
            
            // Update metrics
            document.getElementById('metric-step').textContent = state.currentStep || 0;
            document.getElementById('metric-cost').textContent = '$' + (state.totalCost || 0).toFixed(2);
            document.getElementById('metric-mode').textContent = state.mode;
            
            // Update mode buttons
            document.querySelectorAll('.mode-buttons button').forEach(btn => {
                btn.classList.remove('active');
            });
            if (state.mode === 'yolo') {
                document.getElementById('btn-yolo').classList.add('active');
            } else if (state.mode === 'human') {
                document.getElementById('btn-human').classList.add('active');
            }
            
            // Show/hide action notice
            const actionNotice = document.getElementById('action-notice');
            if (state.hasPendingAction) {
                actionNotice.style.display = 'block';
            } else {
                actionNotice.style.display = 'none';
            }
            
            // Update strategy section
            updateStrategies(state.waitingForStrategySelection, state.currentStrategies || []);
            
            // Update tasks
            updateTasks(state.currentTask, state.recentTasks || []);
        }
        
        let selectedStrategyId = null;
        
        function updateStrategies(waiting, strategies) {
            const strategySection = document.getElementById('strategy-section');
            const strategyList = document.getElementById('strategy-list');
            
            if (!waiting || !strategies || strategies.length === 0) {
                strategySection.style.display = 'none';
                return;
            }
            
            strategySection.style.display = 'block';
            selectedStrategyId = null;
            
            strategyList.innerHTML = strategies.map(strategy => {
                const recommendedBadge = strategy.isRecommended ? '<span style="color: var(--vscode-charts-green);">⭐</span>' : '';
                const recommendedClass = strategy.isRecommended ? 'recommended' : '';
                
                return \`
                    <div class="strategy-item \${recommendedClass}" 
                         data-strategy-id="\${strategy.id}"
                         onclick="selectStrategyItem('\${strategy.id}')">
                        <div class="strategy-title">
                            \${recommendedBadge}
                            \${escapeHtml(strategy.title)}
                        </div>
                        <div class="strategy-description">\${escapeHtml(strategy.description)}</div>
                        \${strategy.reasoning ? '<div class="strategy-reasoning">' + escapeHtml(strategy.reasoning) + '</div>' : ''}
                    </div>
                \`;
            }).join('');
        }
        
        function selectStrategyItem(strategyId) {
            selectedStrategyId = strategyId;
            document.querySelectorAll('.strategy-item').forEach(item => {
                if (item.dataset.strategyId === strategyId) {
                    item.classList.add('selected');
            } else {
                    item.classList.remove('selected');
                }
            });
        }
        
        function confirmStrategy() {
            if (!selectedStrategyId) {
                // If no strategy selected, auto-select recommended one
                const recommendedItem = document.querySelector('.strategy-item.recommended');
                if (recommendedItem) {
                    selectedStrategyId = recommendedItem.dataset.strategyId;
                } else {
                    // Select first one
                    const firstItem = document.querySelector('.strategy-item');
                    if (firstItem) {
                        selectedStrategyId = firstItem.dataset.strategyId;
                    }
                }
            }
            
            vscode.postMessage({
                command: 'selectStrategy',
                strategyId: selectedStrategyId
            });
        }
        
        function skipStrategy() {
            vscode.postMessage({
                command: 'skipStrategy'
            });
        }
        
        function updateTasks(currentTask, recentTasks) {
            const taskList = document.getElementById('task-list');
            const taskCount = document.getElementById('task-count');
            
            const allTasks = [];
            if (currentTask) {
                allTasks.push(currentTask);
            }
            allTasks.push(...recentTasks.filter(t => !currentTask || t.id !== currentTask.id));
            
            taskCount.textContent = '(' + allTasks.length + ')';
            
            if (allTasks.length === 0) {
                taskList.innerHTML = '<div class="empty-state">No tasks yet</div>';
                return;
                        }
            
            taskList.innerHTML = allTasks.map(task => {
                const isCurrent = currentTask && task.id === currentTask.id;
                const statusClass = task.status;
                const statusIcon = task.status === 'completed' ? '✓' : 
                                   task.status === 'failed' ? '✗' : 
                                   task.status === 'running' ? '●' : '⊘';
                
                return \`
                    <div class="task-card \${isCurrent ? 'current' : statusClass}" 
                         onclick="showTask('\${task.id}')">
                        <div class="task-status \${statusClass}">
                            <span>\${statusIcon}</span>
                            <span>\${task.status}</span>
                        </div>
                        <div class="task-query">\${escapeHtml(task.query)}</div>
                        <div class="task-meta">
                            <span>\${formatTime(task.startTime)}</span>
                            \${task.totalCost ? '<span>$' + task.totalCost.toFixed(2) + '</span>' : ''}
                        </div>
                    </div>
                \`;
            }).join('');
        }
        
        function startAgent() {
            vscode.postMessage({ command: 'start' });
        }
        
        function stopAgent() {
            vscode.postMessage({ command: 'stop' });
        }
        
        function setMode(mode) {
            vscode.postMessage({ command: 'setMode', mode: mode });
        }
        
        function showPanel() {
            vscode.postMessage({ command: 'showPanel' });
        }
        
        function showTask(taskId) {
            vscode.postMessage({ command: 'showTask', taskId: taskId });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatTime(dateString) {
            const date = new Date(dateString);
            const now = new Date();
            const diff = now - date;
            const seconds = Math.floor(diff / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);
            
            if (seconds < 60) return 'Just now';
            if (minutes < 60) return minutes + 'm ago';
            if (hours < 24) return hours + 'h ago';
            if (days === 1) return 'Yesterday';
            if (days < 7) return days + 'd ago';
            
            return date.toLocaleDateString();
        }
    </script>
</body>
</html>`;
    }
}
