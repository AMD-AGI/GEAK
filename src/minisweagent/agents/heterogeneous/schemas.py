"""Tool JSON schemas for the heterogeneous orchestrator LLM."""

_ORCHESTRATOR_SWE_TOOLS = {"bash", "str_replace_editor", "profile_kernel", "strategy_manager"}

_ORCHESTRATOR_ONLY_TOOLS: list[dict] = [
    {
        "name": "generate_tasks",
        "description": (
            "Generate optimisation task files for a round.  Returns a JSON "
            "object with a 'tasks' list of file paths.  An empty list means "
            "convergence – no more optimisations to try."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "round_num": {
                    "type": "integer",
                    "description": "Round number (1-based).",
                },
                "previous_results_dir": {
                    "type": "string",
                    "description": "Path to previous round's results directory (optional for round 1).",
                },
            },
            "required": ["round_num"],
        },
    },
    {
        "name": "dispatch_tasks",
        "description": (
            "Dispatch a list of task files to available GPUs for parallel "
            "execution.  Returns a JSON summary of results.  If task_files "
            "is omitted, auto-discovers from the latest round's task directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task file paths to dispatch (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "collect_results",
        "description": (
            "Read results from a completed round's output directory.  "
            "Returns a Markdown summary of patches, test outputs, and logs.  "
            "If results_dir is omitted, auto-discovers the latest round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results_dir": {
                    "type": "string",
                    "description": "Path to the results directory to scan (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "finalize",
        "description": (
            "Signal that optimisation is complete.  Provide a summary of "
            "what was achieved, the best patch, and total speedup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the optimisation.",
                },
                "best_patch": {
                    "type": "string",
                    "description": "Path or identifier of the best patch.",
                },
                "total_speedup": {
                    "type": "string",
                    "description": "Total speedup achieved (e.g. '15%').",
                },
            },
            "required": ["summary"],
        },
    },
]


def build_tools_schema(toolruntime) -> list[dict]:
    """Merge ToolRuntime schemas (allowlisted) with orchestrator-specific tools."""
    swe_tools = [t for t in toolruntime.get_tools_schema() if t["name"] in _ORCHESTRATOR_SWE_TOOLS]
    return swe_tools + _ORCHESTRATOR_ONLY_TOOLS
