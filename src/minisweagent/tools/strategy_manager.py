"""Strategy list management tool for kernel optimization.

This module provides a structured way to manage optimization strategies,
avoiding manual markdown editing and ensuring consistent formatting.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import typer
from typing_extensions import Annotated

class StrategyStatus(Enum):
    """Strategy status enumeration."""
    BASELINE = "baseline"
    PENDING = "pending"
    EXPLORING = "exploring"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    COMBINED = "combined"
    
    @classmethod
    def from_string(cls, s: str) -> "StrategyStatus":
        """Create status from string."""
        return cls[s.upper()]

@dataclass
class Strategy:
    """Single optimization strategy."""
    name: str
    status: StrategyStatus
    description: str
    priority: int = 50  # Default: Normal priority (High=100, Normal=50)
    expected: str | None = None
    result: str | None = None
    details: str | None = None
    target: str | None = None
    
    @property
    def priority_label(self) -> str:
        """Get human-readable priority label."""
        if self.priority >= 100:
            return "high"
        else:
            return "normal"
    
    @staticmethod
    def priority_from_label(label: str) -> int:
        """Convert priority label to numeric value."""
        label_lower = label.lower()
        if label_lower == "high":
            return 100
        else:  # "normal" or any other value
            return 50
    
    def to_markdown(self, index: int) -> str:
        """Convert to markdown format."""
        lines = [f"## Strategy {index}: {self.name}"]
        lines.append(f"[priority:{self.priority_label}][{self.status.value}] {self.description}")
        
        if self.expected:
            lines.append(f"- Expected: {self.expected}")
        if self.target:
            lines.append(f"- Target: {self.target}")
        if self.result:
            lines.append(f"- Result: {self.result}")
        if self.details:
            lines.append(f"- Details: {self.details}")
            
        return "\n".join(lines)

@dataclass
class Baseline:
    """Baseline performance."""
    metrics: dict[str, str] = field(default_factory=dict)
    log_file: str | None = None
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = ["## Baseline Performance", "[baseline]"]
        for key, value in self.metrics.items():
            lines.append(f"- {key}: {value}")
        if self.log_file:
            lines.append(f"- Detailed results: {self.log_file}")
        return "\n".join(lines)

@dataclass
class StrategyList:
    """Complete strategy list."""
    baseline: Baseline
    strategies: list[Strategy] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert to complete markdown file."""
        lines = ["# Kernel Optimization Strategies", ""]
        
        lines.append(self.baseline.to_markdown())
        lines.append("")
        
        for i, strategy in enumerate(self.strategies, start=1):
            lines.append(strategy.to_markdown(i))
            lines.append("")
        
        if self.notes:
            for note in self.notes:
                lines.append(f"# Note: {note}")
            lines.append("")
        
        return "\n".join(lines)

class StrategyManager:
    """Strategy list manager."""
    
    def __init__(
        self, 
        filepath: Path | str = ".optimization_strategies.md",
        on_change_callback = None
    ):
        """Initialize strategy manager.
        
        Args:
            filepath: Path to strategy markdown file
            on_change_callback: Optional callback function called when strategies change.
                                Receives StrategyList as argument.
        """
        self.filepath = Path(filepath)
        self.on_change_callback = on_change_callback
    
    def create(self, baseline: Baseline, strategies: list[Strategy]) -> None:
        """Create new strategy list file."""
        strategy_list = StrategyList(baseline=baseline, strategies=strategies)
        self.save(strategy_list)
    
    def load(self) -> StrategyList:
        """Load strategy list from markdown file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Strategy file not found: {self.filepath}")
        
        content = self.filepath.read_text()
        return self._parse_markdown(content)
    
    def save(self, strategy_list: StrategyList) -> None:
        """Save strategy list to markdown file."""
        content = strategy_list.to_markdown()
        self.filepath.write_text(content)
        
        # Trigger callback if provided
        if self.on_change_callback:
            try:
                self.on_change_callback(strategy_list)
            except Exception as e:
                # Don't let callback errors break the save operation
                import sys
                print(f"Warning: on_change_callback failed: {e}", file=sys.stderr)
    
    def exists(self) -> bool:
        """Check if strategy file exists."""
        return self.filepath.exists()
    
    def add_strategy(
        self, 
        name: str,
        description: str,
        expected: str,
        position: int | None = None,
        target: str | None = None
    ) -> None:
        """Add new strategy."""
        strategy_list = self.load()
        new_strategy = Strategy(
            name=name,
            status=StrategyStatus.PENDING,
            description=description,
            expected=expected,
            target=target
        )
        
        if position is None:
            strategy_list.strategies.append(new_strategy)
        else:
            strategy_list.strategies.insert(position - 1, new_strategy)
        
        self.save(strategy_list)
    
    def remove_strategy(self, index: int, method: str = "skip") -> None:
        """Remove strategy."""
        strategy_list = self.load()
        idx = index - 1
        
        if method == "skip":
            strategy_list.strategies[idx].status = StrategyStatus.SKIPPED
            strategy_list.strategies[idx].result = "Not applicable"
        elif method == "delete":
            del strategy_list.strategies[idx]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'skip' or 'delete'")
        
        self.save(strategy_list)
    
    def update_strategy(
        self,
        index: int,
        status: str | None = None,
        result: str | None = None,
        details: str | None = None,
        **kwargs
    ) -> None:
        """Update strategy fields."""
        strategy_list = self.load()
        idx = index - 1
        strategy = strategy_list.strategies[idx]
        
        if status:
            strategy.status = StrategyStatus.from_string(status)
        if result is not None:
            strategy.result = result
        if details is not None:
            strategy.details = details
        
        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        self.save(strategy_list)
    
    def mark_status(
        self,
        index: int,
        status: str,
        result: str | None = None,
        details: str | None = None
    ) -> None:
        """Mark strategy status."""
        self.update_strategy(index, status=status, result=result, details=details)
    
    def get_strategy(self, index: int) -> Strategy:
        """Get specific strategy."""
        strategy_list = self.load()
        return strategy_list.strategies[index - 1]
    
    def list_strategies(
        self,
        status: str | None = None
    ) -> list[tuple[int, Strategy]]:
        """List strategies with indices."""
        strategy_list = self.load()
        result = []
        
        for i, strategy in enumerate(strategy_list.strategies, start=1):
            if status is None or strategy.status.value == status:
                result.append((i, strategy))
        
        return result
    
    def get_summary(self) -> dict:
        """Get strategy list summary statistics."""
        strategy_list = self.load()
        
        summary = {
            "total": len(strategy_list.strategies),
            "by_status": {}
        }
        
        for status in StrategyStatus:
            count = sum(1 for s in strategy_list.strategies if s.status == status)
            if count > 0:
                summary["by_status"][status.value] = count
        
        return summary
    
    def get_full_content(self) -> str:
        """Get full content of the strategy file as formatted text."""
        if not self.exists():
            return "Strategy file does not exist"
        
        strategy_list = self.load()
        lines = []
        
        # Baseline section
        if strategy_list.baseline:
            lines.append("## Baseline Performance")
            for key, value in strategy_list.baseline.metrics.items():
                lines.append(f"- {key}: {value}")
            if strategy_list.baseline.log_file:
                lines.append(f"- Detailed results: {strategy_list.baseline.log_file}")
            lines.append("")
        
        # Strategies
        for idx, strategy in enumerate(strategy_list.strategies, start=1):
            priority_label = "🔴 HIGH" if strategy.priority >= 100 else ""
            lines.append(f"## Strategy {idx}: {strategy.name} {priority_label}")
            lines.append(f"[{strategy.status.value}] {strategy.description}")
            if strategy.expected:
                lines.append(f"- Expected: {strategy.expected}")
            if strategy.target:
                lines.append(f"- Target: {strategy.target}")
            if strategy.result:
                lines.append(f"- Result: {strategy.result}")
            if strategy.details:
                lines.append(f"- Details: {strategy.details}")
            lines.append("")
        
        # Notes
        if strategy_list.notes:
            lines.append("## Notes")
            for note in strategy_list.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        return "\n".join(lines)
    
    def add_note(self, note: str) -> None:
        """Add note to strategy list."""
        strategy_list = self.load()
        strategy_list.notes.append(note)
        self.save(strategy_list)
    
    def update_priority(self, index: int, priority: int) -> None:
        """Update strategy priority."""
        strategy_list = self.load()
        idx = index - 1
        
        if idx < 0 or idx >= len(strategy_list.strategies):
            raise IndexError(f"Strategy index {index} out of range")
        
        strategy_list.strategies[idx].priority = priority
        self.save(strategy_list)
    
    def _parse_markdown(self, content: str) -> StrategyList:
        """Parse markdown content to StrategyList object."""
        lines = content.split("\n")
        
        baseline = None
        strategies = []
        notes = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("## Baseline Performance"):
                baseline, i = self._parse_baseline(lines, i + 1)
            elif line.startswith("## Strategy"):
                match = re.match(r"## Strategy \d+: (.+)", line)
                if match:
                    strategy, i = self._parse_strategy(lines, i + 1, match.group(1))
                    strategies.append(strategy)
                else:
                    i += 1
            elif line.startswith("# Note:"):
                note = line.replace("# Note:", "").strip()
                if note:
                    notes.append(note)
                i += 1
            else:
                i += 1
        
        if baseline is None:
            baseline = Baseline()
        
        return StrategyList(baseline=baseline, strategies=strategies, notes=notes)
    
    def _parse_baseline(self, lines: list[str], start_idx: int) -> tuple[Baseline, int]:
        """Parse baseline section."""
        metrics = {}
        log_file = None
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            # Stop at next section
            if line.startswith("##") or line.startswith("# Note:") or (line.startswith("#") and not line.startswith("##")):
                break
            
            if line.startswith("- "):
                content = line[2:].strip()
                if ":" in content:
                    key, value = content.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "Detailed results":
                        log_file = value
                    else:
                        metrics[key] = value
            i += 1
        
        return Baseline(metrics=metrics, log_file=log_file), i
    
    def _parse_strategy(self, lines: list[str], start_idx: int, name: str) -> tuple[Strategy, int]:
        """Parse strategy section."""
        status = StrategyStatus.PENDING
        priority = 50  # Default: Normal priority
        description = ""
        expected = None
        result = None
        details = None
        target = None
        
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            # Stop at next strategy, note, or any other top-level markdown heading
            if line.startswith("##") or line.startswith("# Note:") or (line.startswith("#") and not line.startswith("##")):
                break
            
            if line.startswith("[") and "]" in line:
                # Parse priority if present: [priority:high][status] description
                remaining = line
                
                # Try to parse priority (supports both numeric and text labels)
                priority_match = re.match(r'\[priority:(high|normal|\d+)\]', remaining, re.IGNORECASE)
                if priority_match:
                    priority_value = priority_match.group(1)
                    if priority_value.isdigit():
                        # Numeric value (backward compatibility)
                        priority = int(priority_value)
                    else:
                        # Text label (high/normal)
                        priority = Strategy.priority_from_label(priority_value)
                    remaining = remaining[priority_match.end():]
                
                # Parse status
                if remaining.startswith("[") and "]" in remaining:
                    status_str = remaining[remaining.index("[") + 1:remaining.index("]")]
                    try:
                        status = StrategyStatus(status_str)
                    except ValueError:
                        status = StrategyStatus.PENDING
                    description = remaining[remaining.index("]") + 1:].strip()
            elif line.startswith("- Expected:"):
                expected = line.replace("- Expected:", "").strip()
            elif line.startswith("- Target:"):
                target = line.replace("- Target:", "").strip()
            elif line.startswith("- Result:"):
                result = line.replace("- Result:", "").strip()
            elif line.startswith("- Details:"):
                details = line.replace("- Details:", "").strip()
            
            i += 1
        
        return Strategy(
            name=name,
            status=status,
            priority=priority,
            description=description,
            expected=expected,
            result=result,
            details=details,
            target=target
        ), i


# CLI Interface
app = typer.Typer(help="Strategy list management tool for kernel optimization")

@app.command()
def create(
    baseline_metrics: Annotated[list[str], typer.Option(
        help="Baseline metrics in format 'Key:Value'"
    )],
    baseline_log: Annotated[str | None, typer.Option(
        help="Path to baseline benchmark log file"
    )] = None,
    strategies: Annotated[list[str] | None, typer.Option(
        help="Strategies in format 'Name|Description|Expected|Target'"
    )] = None,
    file: Annotated[str, typer.Option(
        help="Output file path"
    )] = ".optimization_strategies.md"
):
    """Create a new strategy list file."""
    manager = StrategyManager(file)
    
    metrics = {}
    for metric in baseline_metrics:
        if ":" not in metric:
            typer.echo(f"Warning: Invalid metric format '{metric}', skipping", err=True)
            continue
        key, value = metric.split(":", 1)
        metrics[key.strip()] = value.strip()
    
    baseline = Baseline(metrics=metrics, log_file=baseline_log)
    
    strategy_list = []
    if strategies:
        for s in strategies:
            parts = s.split("|")
            if len(parts) < 2:
                typer.echo(f"Warning: Invalid strategy format '{s}', skipping", err=True)
                continue
            strategy = Strategy(
                name=parts[0].strip(),
                status=StrategyStatus.PENDING,
                description=parts[1].strip(),
                expected=parts[2].strip() if len(parts) > 2 else None,
                target=parts[3].strip() if len(parts) > 3 else None
            )
            strategy_list.append(strategy)
    
    manager.create(baseline, strategy_list)
    typer.echo(f"✓ Created strategy list: {file}")

@app.command()
def add(
    name: Annotated[str, typer.Argument(help="Strategy name")],
    description: Annotated[str, typer.Argument(help="Strategy description")],
    expected: Annotated[str, typer.Argument(help="Expected improvement")],
    position: Annotated[int | None, typer.Option(help="Insert position (1-based)")] = None,
    target: Annotated[str | None, typer.Option(help="Optimization target")] = None,
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Add a new strategy."""
    manager = StrategyManager(file)
    manager.add_strategy(name, description, expected, position, target)
    typer.echo(f"✓ Added strategy: {name}")

@app.command()
def mark(
    index: Annotated[int, typer.Argument(help="Strategy index (1-based)")],
    status: Annotated[str, typer.Argument(help="New status")],
    result: Annotated[str | None, typer.Option(help="Result description")] = None,
    details: Annotated[str | None, typer.Option(help="Additional details")] = None,
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Mark strategy status."""
    manager = StrategyManager(file)
    manager.mark_status(index, status, result, details)
    typer.echo(f"✓ Marked Strategy {index} as [{status}]")

@app.command()
def update(
    index: Annotated[int, typer.Argument(help="Strategy index (1-based)")],
    status: Annotated[str | None, typer.Option(help="New status")] = None,
    result: Annotated[str | None, typer.Option(help="Result")] = None,
    details: Annotated[str | None, typer.Option(help="Details")] = None,
    expected: Annotated[str | None, typer.Option(help="Expected improvement")] = None,
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Update strategy fields."""
    manager = StrategyManager(file)
    kwargs = {}
    if expected:
        kwargs["expected"] = expected
    
    manager.update_strategy(index, status, result, details, **kwargs)
    typer.echo(f"✓ Updated Strategy {index}")

@app.command()
def remove(
    index: Annotated[int, typer.Argument(help="Strategy index (1-based)")],
    method: Annotated[str, typer.Option(help="Remove method: 'skip' or 'delete'")] = "skip",
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Remove or skip a strategy."""
    manager = StrategyManager(file)
    manager.remove_strategy(index, method)
    action = "Skipped" if method == "skip" else "Deleted"
    typer.echo(f"✓ {action} Strategy {index}")

@app.command()
def show(
    status: Annotated[str | None, typer.Option(help="Filter by status")] = None,
    summary: Annotated[bool, typer.Option(help="Show summary only")] = False,
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Show strategy list."""
    manager = StrategyManager(file)
    
    if summary:
        summary_data = manager.get_summary()
        typer.echo(f"Total strategies: {summary_data['total']}")
        typer.echo("By status:")
        for status_name, count in summary_data['by_status'].items():
            typer.echo(f"  [{status_name}]: {count}")
    else:
        strategies = manager.list_strategies(status)
        for idx, strategy in strategies:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Strategy {idx}: {strategy.name}")
            typer.echo(f"Status: [{strategy.status.value}]")
            typer.echo(f"Description: {strategy.description}")
            if strategy.expected:
                typer.echo(f"Expected: {strategy.expected}")
            if strategy.target:
                typer.echo(f"Target: {strategy.target}")
            if strategy.result:
                typer.echo(f"Result: {strategy.result}")
            if strategy.details:
                typer.echo(f"Details: {strategy.details}")

@app.command()
def note(
    message: Annotated[str, typer.Argument(help="Note message")],
    file: Annotated[str, typer.Option(help="Strategy file path")] = ".optimization_strategies.md"
):
    """Add a note to the strategy list."""
    manager = StrategyManager(file)
    manager.add_note(message)
    typer.echo("✓ Added note")

if __name__ == "__main__":
    app()

