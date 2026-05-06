# Agent implementations

* `optimization_agent.py` - Main kernel-optimization agent (standalone; used by the `fixed`, `planned`, and `auto` execution modes).
* `default.py` - Base class used by preprocess subagents (`SelectPatchAgent`, `UnitTestAgent`, `ShapeFixerAgent`).
* `parallel_agent.py` - Orchestrator shell that spawns N `OptimizationAgent` workers across a GPU pool.

The `homogeneous/` and `heterogeneous/` sub-directories are implementation folders that hold mode-specific glue code (task generator, prompts, result scanning). They will be renamed / merged into `run/` internals in a later PR; the public execution vocabulary is `fixed` / `planned` / `auto` / `translate`, defined in `run/compose.py`.
