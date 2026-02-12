"""Parse optimization task information from user input."""

import json
import re
from datetime import datetime
from pathlib import Path


def _resolve_path_case(path: Path) -> Path | None:
    """Resolve path to filesystem case (e.g. geak -> GEAK on case-sensitive filesystems).
    Walks each component and matches case-insensitively against directory listing.
    Returns None if any component is not found.
    """
    if not path.is_absolute():
        return None
    parts = path.parts[1:]  # drop leading /
    resolved = Path(path.anchor)
    for name in parts:
        if not resolved.is_dir():
            return None
        found = None
        for entry in resolved.iterdir():
            if entry.name.lower() == name.lower():
                found = entry
                break
        if found is None:
            return None
        resolved = found
    return resolved


def parse_task_info(task_content: str, model) -> dict:
    """Parse task content to extract optimization configuration.
    
    Extracts:
    - kernel_name: Name of the kernel being optimized
    - repo: Repository path
    - test_command: Command to test the optimization
    - metric: Performance metric to extract
    - num_parallel: Number of parallel agents
    - gpu_ids: GPU IDs for parallel execution
    
    Returns dict with extracted values (None if not found).
    """
    prompt = f"""Analyze the following optimization task and extract configuration information.

Task:
{task_content}

Extract the following information (return null if not found):
1. kernel_name: The name of the kernel/function being optimized (e.g., "gemm", "matmul", "conv2d")
2. repo: The repository path mentioned in the task (absolute path or relative path)
3. test_command: The command to run tests or benchmarks
4. metric: The performance metric to measure (e.g., "bandwidth in GB/s", "latency in ms", "throughput")
5. num_parallel: Number of parallel optimization agents to run (integer)
6. gpu_ids: Comma-separated GPU IDs for parallel execution (e.g., "0,1,2,3")

Return ONLY a valid JSON object with these keys. Example:
{{
  "kernel_name": "matmul",
  "repo": "/path/to/repo",
  "test_command": "python test.py",
  "metric": "Extract throughput in GFLOPS",
  "num_parallel": 4,
  "gpu_ids": "0,1,2,3"
}}

If any field cannot be determined from the task, set it to null.
"""
    
    try:
        response = model.query([
            {"role": "system", "content": "You are a helpful assistant that extracts structured configuration from optimization tasks. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ])
        content = response.get("content", "").strip()
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        parsed = json.loads(content)
        
        # Validate and normalize the parsed data
        result = {
            "kernel_name": parsed.get("kernel_name"),
            "repo": parsed.get("repo"),
            "test_command": parsed.get("test_command"),
            "metric": parsed.get("metric"),
            "num_parallel": parsed.get("num_parallel"),
            "gpu_ids": parsed.get("gpu_ids"),
        }
        
        # Normalize repo path and preserve filesystem case (LLM often returns lowercase)
        if result["repo"]:
            repo_path = Path(result["repo"])
            if repo_path.exists():
                result["repo"] = str(repo_path.resolve())
            else:
                # Path may have wrong case (e.g. geak -> GEAK); resolve via directory listing
                resolved = _resolve_path_case(repo_path)
                if resolved is not None:
                    result["repo"] = str(resolved.resolve())
        
        return result
        
    except (json.JSONDecodeError, Exception):
        # If parsing fails, return all None
        return {
            "kernel_name": None,
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
        }


def generate_patch_output_dir(kernel_name: str | None, base_dir: str = "optimization_logs") -> str:
    """Generate patch output directory based on kernel name and timestamp.
    
    Format: optimization_logs/kernelname_timestamp
    If kernel_name is None, use "optimization_timestamp"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if kernel_name:
        # Clean kernel name (replace special characters with underscores)
        clean_name = re.sub(r'[^\w\-]', '_', kernel_name)
        dir_name = f"{clean_name}_{timestamp}"
    else:
        dir_name = f"optimization_{timestamp}"
    
    return str(Path(base_dir) / dir_name)


def display_parsed_config(parsed_info: dict, patch_output_dir: str) -> str:
    """Display parsed configuration in a formatted way for user confirmation."""
    lines = [
        "\n" + "=" * 70,
        "Auto-detected Configuration:",
        "=" * 70,
        "  Note: no input for 60s will default to 'y' (proceed).",
    ]

    fields: list[tuple[str, str]] = [
        (
            "kernel_name",
            parsed_info["kernel_name"]
            or "Not detected. Please use --kernel-name to specify the kernel name",
        ),
        ("repo", parsed_info["repo"] or "Not detected. Please use --repo to specify the repository path"),
        (
            "test_command",
            parsed_info["test_command"]
            or "Not detected. Automatically search or create the test command via UnitTestAgent",
        ),
        (
            "metric",
            parsed_info["metric"] or "Not detected. Automatically extract the metric from the test output",
        ),
        ("num_parallel", str(parsed_info["num_parallel"] or "Not detected. Default to 1.")),
        ("gpu_ids", parsed_info["gpu_ids"] or "Not detected. Default to 0."),
        ("patch_output_dir", patch_output_dir),
    ]
    key_width = max(len(k) for k, _ in fields)
    for key, value in fields:
        lines.append(f"  {key + ':':<{key_width + 1}}  {value}")
    lines.append("=" * 70)
    
    return "\n".join(lines)
