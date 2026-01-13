# Strategy Manager Tool

A command-line tool for managing optimization strategy lists in kernel optimization tasks.

## Overview

The Strategy Manager provides a structured way to create, track, and manage optimization strategies. It generates and parses markdown files in a consistent format, eliminating manual editing errors and ensuring compatibility across different optimization workflows.

## Installation

The tool is already available as part of minisweagent. No additional installation is required.

## Quick Start

```bash
# Create a new strategy list
python -m minisweagent.tools.strategy_manager create \
  --baseline-metrics "Bandwidth:152.3 GB/s" \
  --baseline-metrics "Throughput:2.3 TFLOPS" \
  --baseline-log "baseline_benchmark.log" \
  --strategies "Memory Coalescing|Optimize memory access|+20-30%|Memory-bound" \
  --strategies "SIMD Vectorization|Add vector intrinsics|+15-25%|Compute-bound"

# Mark strategy as being explored
python -m minisweagent.tools.strategy_manager mark 1 exploring

# Record successful result
python -m minisweagent.tools.strategy_manager mark 1 successful \
  --result "+21.1% improvement (KEEP)" \
  --details "strategy1_benchmark.log"

# View all strategies
python -m minisweagent.tools.strategy_manager show

# View summary statistics
python -m minisweagent.tools.strategy_manager show --summary
```

## Commands

### create

Create a new strategy list file.

```bash
python -m minisweagent.tools.strategy_manager create \
  --baseline-metrics "Key:Value" \
  --baseline-log "path/to/log" \
  --strategies "Name|Description|Expected|Target" \
  --file ".optimization_strategies.md"
```

**Options:**
- `--baseline-metrics`: Baseline metrics (can be specified multiple times)
- `--baseline-log`: Path to baseline benchmark log file
- `--strategies`: Strategy definition (can be specified multiple times)
- `--file`: Output file path (default: `.optimization_strategies.md`)

### mark

Mark strategy status with optional result and details.

```bash
python -m minisweagent.tools.strategy_manager mark INDEX STATUS \
  --result "Result description" \
  --details "Additional details" \
  --file ".optimization_strategies.md"
```

**Status values:** `exploring`, `successful`, `failed`, `partial`, `skipped`

**Examples:**
```bash
# Start exploring
python -m minisweagent.tools.strategy_manager mark 1 exploring

# Record success
python -m minisweagent.tools.strategy_manager mark 1 successful \
  --result "+21.1% improvement (KEEP)"

# Record failure
python -m minisweagent.tools.strategy_manager mark 2 failed \
  --result "-3.2% regression (REVERTED)" \
  --details "Register pressure increased"
```

### add

Add a new strategy to the list.

```bash
python -m minisweagent.tools.strategy_manager add \
  "Strategy Name" \
  "Description" \
  "Expected improvement" \
  --target "Optimization target" \
  --position N \
  --file ".optimization_strategies.md"
```

**Options:**
- `--target`: Optimization target description
- `--position`: Insert position (1-based, default: append to end)

### update

Update strategy fields.

```bash
python -m minisweagent.tools.strategy_manager update INDEX \
  --status STATUS \
  --result "Result" \
  --details "Details" \
  --expected "Expected" \
  --file ".optimization_strategies.md"
```

### remove

Remove or skip a strategy.

```bash
python -m minisweagent.tools.strategy_manager remove INDEX \
  --method METHOD \
  --file ".optimization_strategies.md"
```

**Methods:**
- `skip`: Mark as [skipped], preserves strategy in list (recommended)
- `delete`: Remove completely and renumber remaining strategies

### show

Display strategy list.

```bash
python -m minisweagent.tools.strategy_manager show \
  --status STATUS \
  --summary \
  --file ".optimization_strategies.md"
```

**Options:**
- `--status`: Filter by status (e.g., `successful`, `pending`)
- `--summary`: Show only summary statistics

### note

Add a note to document decisions.

```bash
python -m minisweagent.tools.strategy_manager note \
  "Note message" \
  --file ".optimization_strategies.md"
```

## Strategy Status Labels

- `[baseline]` - Original performance (measure first!)
- `[pending]` - Strategy not yet tried
- `[exploring]` - Currently implementing/testing
- `[successful]` - Performance improved, keep it
- `[failed]` - Performance degraded, reverted
- `[partial]` - Some improvement but below expectation
- `[skipped]` - Not applicable or skipped
- `[combined]` - Combining multiple successful strategies

## Complete Workflow Example

```bash
# 1. Create initial strategy list
python -m minisweagent.tools.strategy_manager create \
  --baseline-metrics "Bandwidth:152.3 GB/s" \
  --baseline-log "baseline.log" \
  --strategies "Memory Coalescing|Optimize access|+20-30%|Memory-bound" \
  --strategies "SIMD Vectorization|Add vector intrinsics|+15-25%|Compute-bound" \
  --strategies "Shared Memory|Cache data|+30-40%|Memory traffic"

# 2. Start exploring Strategy 1
python -m minisweagent.tools.strategy_manager mark 1 exploring

# 3. After implementation and benchmarking
python -m minisweagent.tools.strategy_manager mark 1 successful \
  --result "+21.1% improvement (KEEP)" \
  --details "strategy1_benchmark.log"

# 4. Explore Strategy 2
python -m minisweagent.tools.strategy_manager mark 2 exploring
python -m minisweagent.tools.strategy_manager mark 2 failed \
  --result "-3.2% regression (REVERTED)" \
  --details "Register pressure increased"

# 5. Skip Strategy 3 based on analysis
python -m minisweagent.tools.strategy_manager remove 3 --method skip
python -m minisweagent.tools.strategy_manager note "Skipped after profiling showed compute-bound bottleneck"

# 6. Add new strategy based on results
python -m minisweagent.tools.strategy_manager add \
  "Advanced Memory Coalescing" \
  "Build on Strategy 1 success" \
  "+5-10%"

# 7. View current status
python -m minisweagent.tools.strategy_manager show --summary
python -m minisweagent.tools.strategy_manager show --status successful
```

## Output Format

The tool generates markdown files in a consistent format:

```markdown
# Kernel Optimization Strategies

## Baseline Performance
[baseline]
- Bandwidth: 152.3 GB/s
- Throughput: 2.3 TFLOPS
- Detailed results: baseline_benchmark.log

## Strategy 1: Memory Coalescing
[successful] Optimize memory access patterns
- Expected: +20-30%
- Target: Memory-bound
- Result: +21.1% improvement (KEEP)
- Details: strategy1_benchmark.log

## Strategy 2: SIMD Vectorization
[failed] Use vector intrinsics
- Expected: +15-25%
- Target: Compute-bound
- Result: -3.2% regression (REVERTED)
- Details: Register pressure increased

# Note: Prioritized memory optimizations based on profiling
```

## Python API

You can also use the tool programmatically:

```python
from minisweagent.tools import StrategyManager, Strategy, Baseline, StrategyStatus

# Create manager
manager = StrategyManager(".optimization_strategies.md")

# Create strategy list
baseline = Baseline(
    metrics={"Bandwidth": "152.3 GB/s"},
    log_file="baseline.log"
)
strategies = [
    Strategy(
        name="Memory Coalescing",
        status=StrategyStatus.PENDING,
        description="Optimize memory access patterns",
        expected="+20-30%",
        target="Memory-bound"
    )
]
manager.create(baseline, strategies)

# Mark status
manager.mark_status(1, "successful", "+21.1% improvement")

# Add strategy
manager.add_strategy(
    "Loop Unrolling",
    "Unroll inner loops",
    "+10-15%",
    target="Reduce overhead"
)

# Load and query
strategy_list = manager.load()
summary = manager.get_summary()
```

## Benefits

✅ **Consistent Format**: All strategy lists use the same structure  
✅ **Error Prevention**: Type checking and validation prevent mistakes  
✅ **Easy to Use**: Simple commands for all operations  
✅ **Human Readable**: Generates clean markdown files  
✅ **Programmatic Access**: Can be used as Python library  
✅ **Integrated Workflow**: Designed for mini-swe-agent optimization workflows

## Troubleshooting

**Q: File not found error**  
A: Make sure you're in the correct directory or specify the full path with `--file`

**Q: Invalid status error**  
A: Use one of the valid status values: `exploring`, `successful`, `failed`, `partial`, `skipped`

**Q: Strategy index out of range**  
A: Run `show --summary` to see how many strategies exist. Indices are 1-based.

## Testing

Run the test suite:

```bash
pytest tests/test_strategy_manager.py -v
```

