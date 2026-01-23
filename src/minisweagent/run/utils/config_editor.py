"""Interactive configuration editor for auto-detected settings."""

import re
from pathlib import Path
from typing import Any


def parse_edit_command(command: str) -> tuple[str | None, Any]:
    """Parse user edit command like '/test_command=python test.py'.
    
    Returns (field_name, value) or (None, None) if invalid.
    """
    command = command.strip()
    
    # Check if it's an edit command (starts with /)
    if not command.startswith('/'):
        return None, None
    
    # Remove leading /
    command = command[1:]
    
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
    return """
[bold yellow]Edit Commands:[/bold yellow]
  /kernel_name=<name>      - Set kernel name
  /repo=<path>             - Set repository path
  /test_command=<cmd>      - Set test command
  /metric=<description>    - Set metric description
  /num_parallel=<number>   - Set number of parallel agents
  /gpu_ids=<ids>           - Set GPU IDs (e.g., "0,1,2,3")
  
[bold green]Other Commands:[/bold green]
  y, yes, [Enter]          - Proceed with current configuration
  n, no                    - Abort
  h, help                  - Show this help message
"""


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
        console.print("\n[bold cyan]Options:[/bold cyan] [y]es to proceed, [n]o to abort, [h]elp for edit commands, or /field=value to edit")
        user_input = input("Your choice: ").strip().lower()
        
        # Handle different inputs
        if not user_input or user_input in ('y', 'yes'):
            # Proceed with current config
            return current_config, current_patch_dir, True
        
        elif user_input in ('n', 'no'):
            # Abort
            return current_config, current_patch_dir, False
        
        elif user_input in ('h', 'help'):
            # Show help
            console.print(display_edit_help())
            continue
        
        elif user_input.startswith('/'):
            # Edit command
            field_name, value = parse_edit_command(user_input)
            
            if field_name is None:
                console.print("[bold red]Invalid command format. Use /field=value (e.g., /test_command=python test.py)[/bold red]")
                console.print("[dim]Type 'help' to see all available commands[/dim]")
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
            console.print(f"[bold red]Unknown command: '{user_input}'. Type 'help' for available commands.[/bold red]")
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
) -> tuple[Path | None, str | None, str | None, int | None, list[int], Path | None]:
    """Load and merge configurations from multiple sources.
    
    Configuration priority: Command-line > extra_config from yaml > auto-detect
    
    Args:
        config: Loaded configuration dict from yaml
        repo, test_command, metric, num_parallel, gpu_ids, patch_output: Command-line arguments
        task_content: Task description for auto-detection
        yolo: Whether in yolo mode (skip interactive editing)
        model: Model instance for LLM-based parsing
        console: Rich console for output
    
    Returns:
        Updated tuple of (repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output)
        Note: gpu_ids is returned as a list[int], not str
    """
    from minisweagent.run.utils.task_parser import parse_task_info, generate_patch_output_dir, display_parsed_config
    
    # Step 1: Get extra_config from yaml (if exists)
    extra_config = config.get("extra_config", {})
    
    # Step 2: Apply extra_config values for missing command-line arguments
    if not repo and extra_config.get("repo"):
        repo = Path(extra_config["repo"])
        console.print(f"[dim]Using repo from config: {repo}[/dim]")
    if not test_command and extra_config.get("test_command"):
        test_command = extra_config["test_command"]
        console.print(f"[dim]Using test_command from config[/dim]")
    if not metric and extra_config.get("metric"):
        metric = extra_config["metric"]
        console.print(f"[dim]Using metric from config[/dim]")
    if num_parallel is None and extra_config.get("num_parallel"):
        num_parallel = extra_config["num_parallel"]
        console.print(f"[dim]Using num_parallel from config: {num_parallel}[/dim]")
    if not gpu_ids and extra_config.get("gpu_ids"):
        gpu_ids_value = extra_config.get("gpu_ids")
        if isinstance(gpu_ids_value, list):
            gpu_ids = ",".join(map(str, gpu_ids_value))
        else:
            gpu_ids = gpu_ids_value
        console.print(f"[dim]Using gpu_ids from config: {gpu_ids}[/dim]")
    if not patch_output and extra_config.get("patch_output_dir"):
        patch_output = Path(extra_config["patch_output_dir"])
        console.print(f"[dim]Using patch_output_dir from config: {patch_output}[/dim]")
    
    # Step 3: Auto-detect remaining missing configurations
    if task_content:
        missing_fields = []
        if not repo:
            missing_fields.append("repo")
        if not test_command:
            missing_fields.append("test_command")
        if not metric:
            missing_fields.append("metric")
        if num_parallel is None:
            missing_fields.append("num_parallel")
        if not gpu_ids:
            missing_fields.append("gpu_ids")
        
        if missing_fields:
            console.print(f"[bold cyan]Auto-detecting missing configuration from task: {', '.join(missing_fields)}...[/bold cyan]")
            parsed_config = parse_task_info(task_content, model)
            
            # Generate patch output directory based on kernel name (only if not set)
            if not patch_output:
                auto_patch_output = generate_patch_output_dir(parsed_config.get("kernel_name"))
                parsed_config["_patch_output_dir"] = auto_patch_output
            else:
                parsed_config["_patch_output_dir"] = str(patch_output)
            
            # Interactive configuration editor (unless in yolo mode)
            if not yolo:
                updated_config, updated_patch_dir, proceed = interactive_config_edit(
                    parsed_config, parsed_config["_patch_output_dir"], console
                )
                
                if not proceed:
                    console.print("[bold red]Aborted by user.[/bold red]")
                    return None, None, None, None, None, None
                
                parsed_config = updated_config
                parsed_config["_patch_output_dir"] = updated_patch_dir
            else:
                # In yolo mode, just display the config
                console.print(display_parsed_config(parsed_config, parsed_config["_patch_output_dir"]))
            
            # Apply parsed configuration to command-line arguments (only for missing values)
            repo, test_command, metric, num_parallel, gpu_ids, patch_output = apply_config_changes(
                parsed_config, repo, test_command, metric, num_parallel, gpu_ids, patch_output
            )
        else:
            console.print("[bold green]All configuration provided via command-line or config file. Skipping auto-detection.[/bold green]")
    
    # Parse GPU IDs into list[int]
    parsed_gpu_ids = []
    if gpu_ids:
        try:
            parsed_gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",") if gpu_id.strip()]
        except ValueError:
            console.print(f"[bold red]Warning: Invalid GPU IDs format '{gpu_ids}'. Expected comma-separated integers (e.g., '0,1,2,3'). Using default [0].[/bold red]")
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
    
    return repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output
