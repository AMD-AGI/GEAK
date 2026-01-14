import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface OptimizationStrategy {
    index: number;
    name: string;
    status: 'baseline' | 'pending' | 'exploring' | 'successful' | 'failed' | 'partial' | 'skipped' | 'combined';
    description: string;
    expected?: string;
    target?: string;
    result?: string;
    details?: string;
}

export interface StrategyListData {
    exists: boolean;
    baseline?: {
        metrics: Record<string, string>;
        logFile?: string;
    };
    strategies: OptimizationStrategy[];
    notes: string[];
}

export class StrategyManagerClient {
    private currentData?: StrategyListData;
    private changeHandlers: ((data: StrategyListData) => void)[] = [];
    private pythonBridgePath: string;
    
    constructor(
        private workspacePath: string,
        extensionPath: string
    ) {
        this.pythonBridgePath = path.join(extensionPath, 'python', 'strategy_bridge.py');
    }
    
    /**
     * Start event-driven mode (no polling)
     */
    startEventDriven(): void {
        console.log('[StrategyManager] Starting event-driven mode');
        this.loadAndNotify();
    }
    
    /**
     * Stop
     */
    stop(): void {
        // Nothing to clean up in event-driven mode
    }
    
    /**
     * Trigger refresh (for event-driven)
     */
    async triggerRefresh(): Promise<void> {
        await this.loadAndNotify();
    }
    
    /**
     * Register change callback
     */
    onChange(handler: (data: StrategyListData) => void): void {
        this.changeHandlers.push(handler);
        if (this.currentData) {
            handler(this.currentData);
        }
    }
    
    /**
     * Get current data
     */
    getCurrentData(): StrategyListData | null {
        return this.currentData || null;
    }
    
    /**
     * Mark strategy status
     */
    async markStrategy(index: number, status: string, result?: string, details?: string): Promise<boolean> {
        try {
            const strategyFilePath = path.join(this.workspacePath, '.optimization_strategies.md');
            const args = [
                'python',
                this.pythonBridgePath,
                'mark',
                strategyFilePath,
                index.toString(),
                status
            ];
            
            if (result) {
                args.push(result);
            }
            if (details) {
                args.push(details);
            }
            
            const { stdout } = await execAsync(args.join(' '));
            const response = JSON.parse(stdout);
            
            if (response.success) {
                await this.triggerRefresh();
                return true;
            } else {
                console.error('Failed to mark strategy:', response.error);
                return false;
            }
        } catch (error) {
            console.error('Error marking strategy:', error);
            return false;
        }
    }
    
    /**
     * Load and notify
     */
    private async loadAndNotify(): Promise<void> {
        try {
            console.log('[StrategyManager] Loading strategy list...');
            const data = await this.loadStrategyList();
            console.log('[StrategyManager] Loaded data:', JSON.stringify(data, null, 2));
            
            if (JSON.stringify(data) !== JSON.stringify(this.currentData)) {
                this.currentData = data;
                console.log('[StrategyManager] Data changed, notifying', this.changeHandlers.length, 'handlers');
                this.notifyChange(data);
            } else {
                console.log('[StrategyManager] Data unchanged, skipping notification');
            }
        } catch (error) {
            console.error('[StrategyManager] Failed to load strategy list:', error);
        }
    }
    
    /**
     * Notify all listeners
     */
    private notifyChange(data: StrategyListData): void {
        for (const handler of this.changeHandlers) {
            handler(data);
        }
    }
    
    /**
     * Call Python bridge to load strategy list
     */
    private async loadStrategyList(): Promise<StrategyListData> {
        const strategyFilePath = path.join(this.workspacePath, '.optimization_strategies.md');
        const command = `python ${this.pythonBridgePath} get ${strategyFilePath}`;
        
        console.log('[StrategyManager] Executing command:', command);
        console.log('[StrategyManager] Strategy file path:', strategyFilePath);
        
        try {
            const { stdout, stderr } = await execAsync(command);
            
            if (stderr) {
                console.log('[StrategyManager] Python stderr:', stderr);
            }
            
            console.log('[StrategyManager] Python stdout (raw):', stdout);
            
            // Extract JSON from stdout (skip any non-JSON lines like welcome messages)
            // Find the first line that starts with { and parse from there
            const lines = stdout.split('\n');
            let jsonLine = '';
            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('{')) {
                    jsonLine = trimmed;
                    break;
                }
            }
            
            if (!jsonLine) {
                console.error('[StrategyManager] No JSON found in output');
                return { exists: false, strategies: [], notes: [] };
            }
            
            console.log('[StrategyManager] Extracted JSON:', jsonLine);
            const result = JSON.parse(jsonLine);
            
            if (result.error) {
                console.error('[StrategyManager] Strategy bridge error:', result.error);
                return { exists: false, strategies: [], notes: [] };
            }
            
            return result;
            
        } catch (error) {
            console.error('[StrategyManager] Error executing Python bridge:', error);
            return { exists: false, strategies: [], notes: [] };
        }
    }
}


