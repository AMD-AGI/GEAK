"""Interactive configuration editor for auto-detected settings."""

import sys
from pathlib import Path
from select import select
from typing import Any

from minisweagent.run.utils.task_parser import generate_patch_output_dir


def input_with_timeout(prompt: str, timeout_s: float, default: str) -> tuple[str, bool]:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select([sys.stdin], [], [], timeout_s)
    if ready:
        return sys.stdin.readline().rstrip("\n"), False
    return default, True


def parse_edit_command(command: str) -> tuple[str | None, Any]:
    """Parse user edit command like '--test_command=python test.py'.
    
    Returns (field_name, value) or (None, None) if invalid.
    """
    command = command.strip()
    
    # Check if it's an edit command (starts with --)
    if not command.startswith('--'):
        return None, None
    
    # Remove leading --
    command = command[2:]
    
    # Split by = sign
    if '=' not in command:
        return None, None
    
    field_name, value = command.split('=', 1)
    field_name = field_name.strip()
    value = value.strip()
    
    # Validate field name
    valid_fields = {
        'kernel_name', 'repo', 'test_command', 
        'metric', 'num_parallel', 'gpu_ids'
    }
    
    if field_name not in valid_fields:
        return None, None
    
    # Parse value based on field type
    if field_name == 'num_parallel':
        try:
            value = int(value)
        except ValueError:
            return None, None
    elif value.lower() in ('none', 'null', ''):
        value = None
    
    return field_name, value


def display_edit_help() -> str:
    """Display help message for editing configuration."""
    # Use markup=False or escape angle brackets for Rich console
    return """[bold yellow]Edit Commands:[/bold yellow]
  --kernel_name=VALUE      - Set kernel name
  --repo=PATH              - Set repository path
  --test_command=CMD       - Set test command
  --metric=DESCRIPTION     - Set metric description
  --num_parallel=NUMBER    - Set number of parallel agents
  --gpu_ids=IDS            - Set GPU IDs (e.g., "0,1,2,3")
  
[bold green]Other Commands:[/bold green]
  y or Enter               - Proceed with current configuration
  q                        - Abort
  h                        - Show this help message
"""


def display_config_with_sources(merged_config: dict, console):
    """Display configuration with sources and conflicts."""
    lines = [
        "\n" + "=" * 80,
        "Configuration (Priority: Prompt > CLI > YAML):",
        "=" * 80,
    ]
    
    # Show conflicts if any
    conflicts = merged_config.get("_conflicts", {})
    if conflicts:
        lines.append("\n[bold yellow]⚠ Conflicts detected:[/bold yellow]")
        for field, sources in conflicts.items():
            lines.append(f"  {field}:")
            for source, value in sources.items():
                marker = "→" if source == "prompt" else "✗"
                lines.append(f"    {marker} {source}: {value}")
        lines.append("")
    
    # Show final configuration
    sources = merged_config.get("_sources", {})
    fields = [
        ("kernel_name", merged_config.get("kernel_name") or "Not detected"),
        ("repo", merged_config.get("repo") or "Not detected"),
        ("test_command", merged_config.get("test_command") or "Auto-create via UnitTestAgent"),
        ("metric", merged_config.get("metric") or "Auto-extract from test output"),
        ("num_parallel", str(merged_config.get("num_parallel") or "1 (default)")),
        ("gpu_ids", merged_config.get("gpu_ids") or "0 (default)"),
        ("patch_output_dir", merged_config.get("_patch_output_dir", "optimization_logs")),
    ]
    
    for key, value in fields:
        source = sources.get(key)
        source_label = f" [{source}]" if source else ""
        lines.append(f"  {key + ':':<20} {value}{source_label}")
    
    lines.append("=" * 80)
    console.print("\n".join(lines))


def interactive_config_edit_with_sources(merged_config: dict, console) -> tuple[dict, str, bool]:
    """Interactive configuration editor with source tracking and conflict warnings."""
    display_config_with_sources(merged_config, console)
    
    # Show conflicts warning if any
    conflicts = merged_config.get("_conflicts", {})
    if conflicts:
        console.print("\n[bold yellow]⚠ Configuration conflicts detected (see above).[/bold yellow]")
        console.print("[dim]Prompt-detected values will be used by default (highest priority).[/dim]")
    
    current_config = {k: v for k, v in merged_config.items() if not k.startswith("_")}
    current_patch_dir = merged_config["_patch_output_dir"]
    
    while True:
        console.print("\n[bold cyan]Options:[/bold cyan] (y) to proceed, (q) to abort, (h) for help, or --field=value to edit")
        user_input, timed_out = input_with_timeout("Your choice: ", timeout_s=60, default="y")
        user_input = user_input.strip().lower()
        if timed_out:
            console.print("[dim]No input for 60s, defaulting to 'y'.[/dim]")
        
        if not user_input or user_input == 'y':
            return current_config, current_patch_dir, True
        
        elif user_input == 'q':
            return current_config, current_patch_dir, False
        
        elif user_input == 'h':
            console.print(display_edit_help())
            continue
        
        elif user_input.startswith('--'):
            field_name, value = parse_edit_command(user_input)
            
            if field_name is None:
                console.print("[bold red]Invalid command format. Use --field=value (e.g., --test_command=python test.py)[/bold red]")
                console.print("[dim]Type 'h' to see all available commands[/dim]")
                continue
            
            current_config[field_name] = value
            console.print(f"[bold green]✓ Updated {field_name} = {value}[/bold green]")
            
            if field_name == 'kernel_name':
                current_patch_dir = generate_patch_output_dir(value)
                console.print(f"[bold green]✓ Updated patch_output_dir = {current_patch_dir}[/bold green]")
            
            # Update merged_config for next display
            merged_config[field_name] = value
            merged_config["_patch_output_dir"] = current_patch_dir
            display_config_with_sources(merged_config, console)
            continue
        
        else:
            console.print(f"[bold red]Unknown command: '{user_input}'. Type 'h' for available commands.[/bold red]")
            continue


def interactive_config_edit(parsed_config: dict, patch_output_dir: str, console) -> tuple[dict, str, bool]:
    """Interactive configuration editor.
    
    Returns:
        (updated_config, updated_patch_output_dir, proceed)
        - proceed=True: user confirmed
        - proceed=False: user aborted
    """
    from minisweagent.run.utils.task_parser import display_parsed_config, generate_patch_output_dir
    
    current_config = parsed_config.copy()
    current_patch_dir = patch_output_dir
    
    while True:
        # Display current configuration
        console.print(display_parsed_config(current_config, current_patch_dir))
        
        # Prompt for input
        console.print("\n[bold cyan]Options:[/bold cyan] (y) to proceed, (q) to abort, (h) for help, or --field=value to edit")
        user_input, timed_out = input_with_timeout("Your choice: ", timeout_s=60, default="y")
        user_input = user_input.strip().lower()
        if timed_out:
            console.print("[dim]No input for 60s, defaulting to 'y'.[/dim]")
        
        # Handle different inputs
        if not user_input or user_input == 'y':
            # Proceed with current config
            return current_config, current_patch_dir, True
        
        elif user_input == 'q':
            # Abort
            return current_config, current_patch_dir, False
        
        elif user_input == 'h':
            # Show help
            console.print(display_edit_help())
            continue
        
        elif user_input.startswith('--'):
            # Edit command
            field_name, value = parse_edit_command(user_input)
            
            if field_name is None:
                console.print("[bold red]Invalid command format. Use --field=value (e.g., --test_command=python test.py)[/bold red]")
                console.print("[dim]Type 'h' to see all available commands[/dim]")
                continue
            
            # Update configuration
            current_config[field_name] = value
            console.print(f"[bold green]✓ Updated {field_name} = {value}[/bold green]")
            
            # Regenerate patch output dir if kernel_name changed
            if field_name == 'kernel_name':
                current_patch_dir = generate_patch_output_dir(value)
                console.print(f"[bold green]✓ Updated patch_output_dir = {current_patch_dir}[/bold green]")
            
            # Show updated config in next iteration
            continue
        
        else:
            console.print(f"[bold red]Unknown command: '{user_input}'. Type 'h' for available commands.[/bold red]")
            continue


def apply_config_changes(
    parsed_config: dict,
    repo: Path | None,
    test_command: str | None,
    metric: str | None,
    num_parallel: int | None,
    gpu_ids: str | None,
    patch_output: Path | None,
) -> tuple[Path | None, str | None, str | None, int | None, str | None, Path | None]:
    """Apply parsed configuration to command-line arguments.
    
    Only updates arguments that are not already set by command-line.
    Returns updated values.
    """
    # Override command-line arguments with auto-detected values (if not already specified)
    if not repo and parsed_config.get("repo"):
        repo = Path(parsed_config["repo"])
    
    if not test_command and parsed_config.get("test_command"):
        test_command = parsed_config["test_command"]
    
    if not metric and parsed_config.get("metric"):
        metric = parsed_config["metric"]
    
    if num_parallel is None and parsed_config.get("num_parallel"):
        num_parallel = parsed_config["num_parallel"]
    
    if not gpu_ids and parsed_config.get("gpu_ids"):
        gpu_ids = parsed_config["gpu_ids"]
    
    if not patch_output and parsed_config.get("_patch_output_dir"):
        patch_output = Path(parsed_config["_patch_output_dir"])
    
    return repo, test_command, metric, num_parallel, gpu_ids, patch_output


def load_and_merge_configs(
    config: dict,
    repo: Path | None,
    test_command: str | None,
    metric: str | None,
    num_parallel: int | None,
    gpu_ids: str | None,
    patch_output: Path | None,
    task_content: str | None,
    yolo: bool,
    model,
    console,
) -> tuple[Path | None, str | None, str | None, int | None, list[int], Path | None, str | None]:
    """Load and merge configurations from multiple sources.
    
    Configuration priority (highest to lowest):
    1. Prompt auto-detect (from task description)
    2. CLI arguments (--repo, --test-command, etc.)
    3. YAML parallel_config
    
    When conflicts exist, user is prompted for confirmation.
    
    Args:
        config: Loaded configuration dict from yaml
        repo, test_command, metric, num_parallel, gpu_ids, patch_output: Command-line arguments
        task_content: Task description for auto-detection
        yolo: Whether in yolo mode (skip interactive editing)
        model: Model instance for LLM-based parsing
        console: Rich console for output
    
    Returns:
        Updated tuple of (repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name)
        Note: gpu_ids is returned as a list[int], not str
    """
    from minisweagent.run.utils.task_parser import generate_patch_output_dir, parse_task_info
    
    # Track kernel_name for returning
    kernel_name = None

    # Track config sources for each field
    config_sources: dict[str, dict[str, Any]] = {
        "repo": {},
        "test_command": {},
        "metric": {},
        "num_parallel": {},
        "gpu_ids": {},
        "patch_output": {},
    }
    
    # Step 1: Collect values from all sources
    parallel_config = config.get("parallel_config") or {}
    
    # Source 1: CLI arguments (if provided)
    if repo:
        config_sources["repo"]["cli"] = repo
    if test_command:
        config_sources["test_command"]["cli"] = test_command
    if metric:
        config_sources["metric"]["cli"] = metric
    if num_parallel is not None:
        config_sources["num_parallel"]["cli"] = num_parallel
    if gpu_ids:
        config_sources["gpu_ids"]["cli"] = gpu_ids
    if patch_output:
        config_sources["patch_output"]["cli"] = patch_output
    
    # Source 2: YAML parallel_config
    if parallel_config.get("repo"):
        config_sources["repo"]["yaml"] = Path(parallel_config["repo"])
    if parallel_config.get("test_command"):
        config_sources["test_command"]["yaml"] = parallel_config["test_command"]
    if parallel_config.get("metric"):
        config_sources["metric"]["yaml"] = parallel_config["metric"]
    if parallel_config.get("num_parallel") is not None:
        config_sources["num_parallel"]["yaml"] = parallel_config["num_parallel"]
    if parallel_config.get("gpu_ids"):
        gpu_ids_value = parallel_config["gpu_ids"]
        if isinstance(gpu_ids_value, list):
            config_sources["gpu_ids"]["yaml"] = ",".join(map(str, gpu_ids_value))
        else:
            config_sources["gpu_ids"]["yaml"] = str(gpu_ids_value)
    if parallel_config.get("patch_output_dir"):
        config_sources["patch_output"]["yaml"] = Path(parallel_config["patch_output_dir"])
    
    # Step 2: Auto-detect from task content (highest priority if present)
    parsed_config = None
    missing_in_cli_yaml = []
    if task_content:
        # Check what's missing from CLI+YAML
        if not config_sources["repo"]:
            missing_in_cli_yaml.append("repo")
        if not config_sources["test_command"]:
            missing_in_cli_yaml.append("test_command")
        if not config_sources["metric"]:
            missing_in_cli_yaml.append("metric")
        if not config_sources["num_parallel"]:
            missing_in_cli_yaml.append("num_parallel")
        if not config_sources["gpu_ids"]:
            missing_in_cli_yaml.append("gpu_ids")
        
        # Always run auto-detect if there's task content (to show user what was detected)
        if missing_in_cli_yaml:
            console.print(f"[bold cyan]Auto-detecting configuration from task: {', '.join(missing_in_cli_yaml)}...[/bold cyan]")
            parsed_config = parse_task_info(task_content, model)
            
            # Source 3: Prompt auto-detect (highest priority)
            if parsed_config.get("repo"):
                config_sources["repo"]["prompt"] = Path(parsed_config["repo"])
            if parsed_config.get("test_command"):
                config_sources["test_command"]["prompt"] = parsed_config["test_command"]
            if parsed_config.get("metric"):
                config_sources["metric"]["prompt"] = parsed_config["metric"]
            if parsed_config.get("num_parallel") is not None:
                config_sources["num_parallel"]["prompt"] = parsed_config["num_parallel"]
            if parsed_config.get("gpu_ids"):
                config_sources["gpu_ids"]["prompt"] = parsed_config["gpu_ids"]
            if parsed_config.get("kernel_name"):
                kernel_name = parsed_config["kernel_name"]
    
    # Step 3: Merge configurations with priority: prompt > cli > yaml
    # Apply highest priority source for each field
    def get_highest_priority(field_sources: dict[str, Any]) -> tuple[Any, str | None]:
        """Get value from highest priority source. Returns (value, source)"""
        if "prompt" in field_sources:
            return field_sources["prompt"], "prompt"
        elif "cli" in field_sources:
            return field_sources["cli"], "cli"
        elif "yaml" in field_sources:
            return field_sources["yaml"], "yaml"
        return None, None
    
    # Detect conflicts (when multiple sources have different values for the same field)
    conflicts: dict[str, dict[str, Any]] = {}
    for field, sources in config_sources.items():
        if len(sources) > 1:
            # Check if values are actually different
            values = list(sources.values())
            if len({str(v) for v in values}) > 1:
                conflicts[field] = sources
    
    # Apply merged configuration
    repo_value, repo_source = get_highest_priority(config_sources["repo"])
    if repo_value is None:
        repo_value, repo_source = Path.cwd(), "cwd"
    test_command_value, test_command_source = get_highest_priority(config_sources["test_command"])
    metric_value, metric_source = get_highest_priority(config_sources["metric"])
    num_parallel_value, num_parallel_source = get_highest_priority(config_sources["num_parallel"])
    gpu_ids_value, gpu_ids_source = get_highest_priority(config_sources["gpu_ids"])
    patch_output_value, patch_output_source = get_highest_priority(config_sources["patch_output"])
    
    # Generate patch output directory if not provided
    if not patch_output_value:
        patch_output_value = Path(generate_patch_output_dir(kernel_name))
        patch_output_source = "auto-generated"
    
    # Prepare display config with sources
    merged_config = {
        "kernel_name": kernel_name,
        "repo": repo_value,
        "test_command": test_command_value,
        "metric": metric_value,
        "num_parallel": num_parallel_value,
        "gpu_ids": gpu_ids_value,
        "_patch_output_dir": str(patch_output_value),
        "_sources": {
            "repo": repo_source,
            "test_command": test_command_source,
            "metric": metric_source,
            "num_parallel": num_parallel_source,
            "gpu_ids": gpu_ids_source,
            "patch_output": patch_output_source,
        },
        "_conflicts": conflicts,
    }
    
    # Step 4: Interactive confirmation (unless in yolo mode)
    if not yolo and (parsed_config or conflicts):
        updated_config, updated_patch_dir, proceed = interactive_config_edit_with_sources(
            merged_config, console
        )
        
        if not proceed:
            console.print("[bold red]Aborted by user.[/bold red]")
            return None, None, None, None, None, None, None
        
        # Apply user-confirmed values
        repo = Path(updated_config["repo"]) if updated_config.get("repo") else None
        test_command = updated_config.get("test_command")
        metric = updated_config.get("metric")
        num_parallel = updated_config.get("num_parallel")
        gpu_ids = updated_config.get("gpu_ids")
        patch_output = Path(updated_patch_dir) if updated_patch_dir else None
        kernel_name = updated_config.get("kernel_name")
    elif yolo and (parsed_config or missing_in_cli_yaml):
        # In yolo mode, just display and auto-apply
        display_config_with_sources(merged_config, console)
        repo = repo_value
        test_command = test_command_value
        metric = metric_value
        num_parallel = num_parallel_value
        gpu_ids = gpu_ids_value
        patch_output = patch_output_value
    else:
        # No auto-detect needed and no conflicts
        console.print("[bold green]Using configuration from command-line and/or config file.[/bold green]")
        repo = repo_value
        test_command = test_command_value
        metric = metric_value
        num_parallel = num_parallel_value
        gpu_ids = gpu_ids_value
        patch_output = patch_output_value

    # Parse GPU IDs into list[int]
    parsed_gpu_ids = []
    if gpu_ids:
        try:
            parsed_gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",") if gpu_id.strip()]
        except ValueError:
            console.print(f"[bold red]Warning: Invalid GPU IDs format '{gpu_ids}'. Expected comma-separated integers (e.g., '0,1,2,3'). Using default \\[0].[/bold red]")
            parsed_gpu_ids = [0]
    else:
        # Try to get from config file
        config_gpu_ids = config.get("patch", {}).get("gpu_ids")
        if config_gpu_ids:
            if isinstance(config_gpu_ids, list):
                parsed_gpu_ids = config_gpu_ids
            else:
                try:
                    parsed_gpu_ids = [int(gpu_id.strip()) for gpu_id in str(config_gpu_ids).split(",") if gpu_id.strip()]
                except ValueError:
                    parsed_gpu_ids = [0]
        else:
            # Default to GPU 0
            parsed_gpu_ids = [0]
    
    # Auto-detect num_parallel from gpu_ids when not explicitly provided.
    # If the user passed --gpu-ids 4,5,6,7 but didn't set --num-parallel,
    # default to one agent per GPU instead of forcing single-agent mode.
    if num_parallel is None and parsed_gpu_ids and len(parsed_gpu_ids) > 1:
        num_parallel = len(parsed_gpu_ids)
        console.print(
            f"[bold cyan]Auto-detected num_parallel={num_parallel} from gpu_ids {parsed_gpu_ids}[/bold cyan]"
        )

    return repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name
