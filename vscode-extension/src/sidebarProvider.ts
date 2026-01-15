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
        webviewView.webview.onDidReceiveMessage(async message => {
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
                case 'refreshStrategies':
                    this.agentManager.refreshStrategyList();
                    break;
                case 'exploreStrategies':
                    this.agentManager.exploreStrategies(message.indices || []);
                    break;
                case 'editStrategyPath':
                    {
                        const pathInfo = this.agentManager.getStrategyFilePath();
                        if (!pathInfo.isModifiable) {
                            vscode.window.showWarningMessage('Cannot change path while agent is running.');
                            return;
                        }
                        
                        const newPath = await vscode.window.showInputBox({
                            prompt: 'Enter strategy file path (relative to workspace or absolute path)',
                            value: pathInfo.relative,
                            placeHolder: '.optimization_strategies.md or /absolute/path/to/file.md',
                            validateInput: (value) => {
                                if (!value) { return 'Path cannot be empty'; }
                                return null;
                            }
                        });
                        
                        if (newPath && newPath !== pathInfo.relative) {
                            await this.agentManager.updateStrategyFilePath(newPath);
                        }
                    }
                    break;
                case 'openStrategyFile':
                    {
                        const fs = require('fs');
                        let filePath: string;
                        
                        // Try to get path from strategy data (from Python agent) first
                        const state = this.agentManager.getState();
                        if (state.strategyData?.filePath) {
                            filePath = state.strategyData.filePath;
                            console.log('[SidebarProvider] Using file path from Python agent:', filePath);
                        } else {
                            // Fallback to calculated path
                            const pathInfo = this.agentManager.getStrategyFilePath();
                            filePath = pathInfo.absolute;
                            console.log('[SidebarProvider] Using calculated file path:', filePath);
                        }
                        
                        const uri = vscode.Uri.file(filePath);
                        
                        if (fs.existsSync(filePath)) {
                            await vscode.window.showTextDocument(uri);
                        } else {
                            const choice = await vscode.window.showWarningMessage(
                                `Strategy file does not exist: ${filePath}\nCreate it?`,
                                'Create',
                                'Cancel'
                            );
                            if (choice === 'Create') {
                                vscode.window.showInformationMessage('Use agent command: optool create ...');
                            }
                        }
                    }
                    break;
            }
        });
        
        // Send initial state
        this.updateWebview(this.agentManager.getState());
    }
    
    private updateWebview(state: AgentState) {
        if (this.view) {
            const sidebarData = this.extractSidebarData(state);
            console.log('[SidebarProvider] Updating webview with state:', JSON.stringify(sidebarData, null, 2));
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
            waitingForStrategySelection: state.waitingForStrategySelection,
            strategyData: state.strategyData,
            strategyFilePath: state.strategyFilePath
        };
    }
    
    private getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>geak-agent</title>
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
        
        /* Optimization strategies section */
        .optimization-section {
            margin: 12px 0;
            padding: 12px;
            background: var(--vscode-editor-background);
            border-radius: 6px;
            border: 1px solid var(--vscode-input-border);
        }
        
        .optimization-section .section-title {
            margin: 0 0 12px 0;
            position: relative;
        }
        
        .refresh-btn {
            padding: 4px 8px;
            font-size: 14px;
            min-width: 28px;
            margin-left: auto;
        }
        
        .optimization-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .optimization-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 8px;
            margin-bottom: 6px;
            background: var(--vscode-textBlockQuote-background);
            border-radius: 4px;
            transition: background 0.2s;
        }
        
        .optimization-item:hover {
            background: var(--vscode-list-hoverBackground);
        }
        
        .optimization-item.disabled {
            opacity: 0.6;
        }
        
        .optimization-item.disabled:hover {
            background: transparent;
        }
        
        .optimization-checkbox {
            margin-top: 2px;
            cursor: pointer;
        }
        
        .optimization-checkbox:disabled {
            cursor: not-allowed;
            opacity: 0.5;
        }
        
        .optimization-content {
            flex: 1;
            min-width: 0;
        }
        
        .optimization-header {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 2px;
        }
        
        .optimization-name {
            font-weight: 500;
            font-size: 0.9em;
        }
        
        .optimization-status {
            font-size: 0.75em;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        }
        
        .status-pending {
            background: var(--vscode-charts-yellow);
            color: var(--vscode-editor-background);
        }
        
        .status-exploring {
            background: var(--vscode-charts-blue);
            color: var(--vscode-editor-background);
        }
        
        .status-successful {
            background: var(--vscode-charts-green);
            color: var(--vscode-editor-background);
        }
        
        .status-failed {
            background: var(--vscode-charts-red);
            color: var(--vscode-editor-background);
        }
        
        .status-partial {
            background: var(--vscode-charts-orange);
            color: var(--vscode-editor-background);
        }
        
        .status-combined {
            background: var(--vscode-charts-purple);
            color: var(--vscode-editor-background);
        }
        
        .status-skipped {
            background: var(--vscode-descriptionForeground);
            color: var(--vscode-editor-background);
        }
        
        .optimization-description {
            font-size: 0.85em;
            opacity: 0.7;
            line-height: 1.3;
            word-break: break-word;
        }
        
        .optimization-result {
            margin-top: 4px;
            font-size: 0.8em;
            padding: 4px 6px;
            background: var(--vscode-textBlockQuote-background);
            border-left: 2px solid var(--vscode-textBlockQuote-border);
            color: var(--vscode-descriptionForeground);
            font-style: italic;
            line-height: 1.3;
        }
        
        .empty-strategies {
            text-align: center;
            padding: 20px;
            opacity: 0.6;
            font-size: 0.9em;
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
        
        .strategy-file-path {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 8px;
            margin-bottom: 8px;
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 4px;
            font-size: 11px;
        }
        
        .path-text {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--vscode-descriptionForeground);
        }
        
        .path-button {
            padding: 2px 6px;
            font-size: 10px;
            min-width: auto;
        }
        
        .lock-icon {
            opacity: 0.7;
            font-size: 10px;
        }
        
        button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
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
    
    <!-- Optimization Strategies -->
    <div id="optimization-section" class="optimization-section" style="display: none;">
        <div class="section-title">
            <span>🎯 Optimization Strategies</span>
            <div style="display: flex; gap: 4px;">
                <button onclick="refreshStrategies()" class="path-button" title="Refresh strategy list">↻</button>
                <button id="edit-path-btn" onclick="editStrategyPath()" class="path-button" title="Change file path">⚙️</button>
            </div>
        </div>
        <div id="strategy-file-path" class="strategy-file-path">
            <span>📄</span>
            <span class="path-text" id="path-text" title=""></span>
            <button onclick="openStrategyFile()" class="path-button">Open</button>
            <span class="lock-icon" id="path-lock-icon"></span>
        </div>
        <div id="optimization-list" class="optimization-list"></div>
        <div class="strategy-buttons">
            <button onclick="exploreSelected()" class="primary">🚀 Explore Selected</button>
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
            console.log('[Webview] Received message:', message);
            if (message.type === 'stateUpdate') {
                console.log('[Webview] State update:', message.state);
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
            
            // Update strategy file path (pass strategy data to get actual file path)
            updateStrategyFilePath(state.strategyFilePath, state.strategyData);
            
            // Update optimization strategies section
            updateOptimizationStrategies(state.strategyData);
            
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
        
        let selectedOptimizationIndices = new Set();
        
        function updateOptimizationStrategies(strategyData) {
            console.log('[Webview] updateOptimizationStrategies called with:', strategyData);
            const optimizationSection = document.getElementById('optimization-section');
            const optimizationList = document.getElementById('optimization-list');
            
            if (!strategyData || !strategyData.exists) {
                console.log('[Webview] Strategy data not exists, hiding section');
                optimizationSection.style.display = 'none';
                return;
            }
            
            console.log('[Webview] Strategy data exists, showing section');
            
            optimizationSection.style.display = 'block';
            
            // Show all strategies, not just pending ones
            const strategies = strategyData.strategies;
            
            if (strategies.length === 0) {
                optimizationList.innerHTML = '<div class="empty-strategies">No strategies found</div>';
                return;
            }
            
            optimizationList.innerHTML = strategies.map(strategy => {
                const isPending = strategy.status === 'pending';
                const isDisabled = !isPending;
                
                return \`
                <div class="optimization-item \${isDisabled ? 'disabled' : ''}">
                    <input 
                        type="checkbox" 
                        class="optimization-checkbox" 
                        id="opt-\${strategy.index}"
                        onchange="toggleOptimizationStrategy(\${strategy.index})"
                        \${selectedOptimizationIndices.has(strategy.index) ? 'checked' : ''}
                        \${isDisabled ? 'disabled' : ''}
                    />
                    <label for="opt-\${strategy.index}" class="optimization-content">
                        <div class="optimization-header">
                            <span class="optimization-name">#\${strategy.index} \${escapeHtml(strategy.name)}</span>
                            <span class="optimization-status status-\${strategy.status}">\${strategy.status}</span>
                        </div>
                        <div class="optimization-description">\${escapeHtml(strategy.description)}</div>
                        \${strategy.result ? \`<div class="optimization-result">\${escapeHtml(strategy.result)}</div>\` : ''}
                    </label>
                </div>
                \`;
            }).join('');
        }
        
        function toggleOptimizationStrategy(index) {
            if (selectedOptimizationIndices.has(index)) {
                selectedOptimizationIndices.delete(index);
            } else {
                selectedOptimizationIndices.add(index);
            }
        }
        
        function refreshStrategies() {
            vscode.postMessage({ command: 'refreshStrategies' });
        }
        
        function exploreSelected() {
            if (selectedOptimizationIndices.size === 0) {
                return;
            }
            
            vscode.postMessage({ 
                command: 'exploreStrategies', 
                indices: Array.from(selectedOptimizationIndices)
            });
            
            // Clear selection
            selectedOptimizationIndices.clear();
        }
        
        function updateStrategyFilePath(pathInfo, strategyData) {
            const pathText = document.getElementById('path-text');
            const editBtn = document.getElementById('edit-path-btn');
            const lockIcon = document.getElementById('path-lock-icon');
            
            // If strategy data has actual file path from Python agent, use that
            if (strategyData && strategyData.filePath) {
                const actualPath = strategyData.filePath;
                const parts = actualPath.split('/');
                const fileName = parts[parts.length - 1];
                
                pathText.textContent = fileName;
                pathText.title = actualPath;
                
                // When agent is running with actual file, lock is always on
                lockIcon.textContent = '🔒';
                lockIcon.title = 'Locked (agent running)';
                editBtn.disabled = true;
            } else if (pathInfo) {
                // Fallback to calculated path info
                pathText.textContent = pathInfo.relative;
                pathText.title = pathInfo.absolute;
                
                // Update lock icon and button state
                if (pathInfo.isModifiable) {
                    lockIcon.textContent = '✏️';
                    lockIcon.title = 'Editable';
                    editBtn.disabled = false;
                } else {
                    lockIcon.textContent = '🔒';
                    lockIcon.title = 'Locked (agent running)';
                    editBtn.disabled = true;
                }
            }
        }
        
        function editStrategyPath() {
            vscode.postMessage({ command: 'editStrategyPath' });
        }
        
        function openStrategyFile() {
            vscode.postMessage({ command: 'openStrategyFile' });
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
