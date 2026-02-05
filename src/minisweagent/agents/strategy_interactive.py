"""Interactive CLI agent with strategy support.

This agent extends StrategyAgent for command-line usage with rich console output.
"""

import sys
from rich.console import Console

from minisweagent.agents.strategy_agent import StrategyAgent


console = Console(highlight=False)


class StrategyInteractiveAgent(StrategyAgent):
    """Strategy agent for CLI with rich console output.
    
    This agent implements the strategy management interface for command-line usage,
    providing visual feedback through the rich console library.
    """

    def notify_strategy_changed(self, strategy_data: dict):
        """Display strategy changes in the console."""
        strategies = strategy_data.get("strategies", [])
        file_path = strategy_data.get("filePath", "")
        
        console.print(f"\n[bold green]Strategy list updated:[/bold green] {len(strategies)} strategies")
        console.print(f"[dim]File: {file_path}[/dim]")
        
        # Show summary of strategies
        if strategies:
            console.print("\n[bold]Current Strategies:[/bold]")
            for s in strategies[:5]:  # Show first 5
                status_color = {
                    "pending": "yellow",
                    "exploring": "blue",
                    "successful": "green",
                    "failed": "red",
                    "partial": "orange",
                    "skipped": "dim"
                }.get(s["status"], "white")
                
                console.print(f"  [{status_color}]{s['index']}. {s['name']}[/{status_color}] - {s['status']}")
            
            if len(strategies) > 5:
                console.print(f"  [dim]... and {len(strategies) - 5} more[/dim]")
        
        console.print()  # Empty line for readability

