#!/usr/bin/env python3
"""
Runtime Environment Manager for GEAK Agent

Detects and manages runtime environments (local or Docker) for GPU kernel operations.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

logger = logging.getLogger(__name__)
console = Console()


class RuntimeType(Enum):
    """Type of runtime environment."""
    LOCAL = "local"
    DOCKER = "docker"
    UNKNOWN = "unknown"


@dataclass
class RuntimeEnvironment:
    """Configuration for a runtime environment."""
    runtime_type: RuntimeType
    docker_image: Optional[str] = None
    docker_devices: list[str] = None
    docker_volumes: list[str] = None
    has_gpu: bool = False
    has_triton: bool = False
    has_torch: bool = False
    
    def __post_init__(self):
        if self.docker_devices is None:
            self.docker_devices = []
        if self.docker_volumes is None:
            self.docker_volumes = []


# Default configuration
DEFAULT_DOCKER_IMAGE = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
DEFAULT_DOCKER_DEVICES = ["/dev/kfd", "/dev/dri"]


def check_docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_local_gpu() -> bool:
    """Check if GPU is available locally (ROCm or CUDA)."""
    # Check for ROCm
    rocm_paths = [
        "/opt/rocm",
        "/usr/lib/rocm",
        Path.home() / ".rocm"
    ]
    has_rocm = any(Path(p).exists() for p in rocm_paths)
    
    # Check for CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        has_cuda = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        has_cuda = False
    
    return has_rocm or has_cuda


def check_local_dependencies() -> dict[str, bool]:
    """Check if required dependencies (torch, triton) are available locally."""
    deps = {}
    
    # Check torch
    try:
        import torch
        deps['torch'] = True
        deps['torch_cuda'] = torch.cuda.is_available()
    except ImportError:
        deps['torch'] = False
        deps['torch_cuda'] = False
    
    # Check triton
    try:
        import triton
        deps['triton'] = True
    except ImportError:
        deps['triton'] = False
    
    return deps


def detect_runtime_environment() -> RuntimeEnvironment:
    """
    Automatically detect the current runtime environment.
    
    Returns:
        RuntimeEnvironment: Configuration of detected environment
    """
    # Check local environment
    has_gpu = check_local_gpu()
    deps = check_local_dependencies()
    
    if deps.get('torch') and deps.get('triton') and (has_gpu or deps.get('torch_cuda')):
        logger.info("Detected complete local environment with GPU support")
        return RuntimeEnvironment(
            runtime_type=RuntimeType.LOCAL,
            has_gpu=True,
            has_triton=deps['triton'],
            has_torch=deps['torch']
        )
    
    logger.info("Local environment incomplete (missing GPU or dependencies)")
    return RuntimeEnvironment(
        runtime_type=RuntimeType.UNKNOWN,
        has_gpu=has_gpu,
        has_triton=deps.get('triton', False),
        has_torch=deps.get('torch', False)
    )


def prompt_runtime_environment(auto_confirm: bool = False) -> RuntimeEnvironment:
    """
    Prompt user to select runtime environment if not suitable.
    
    Args:
        auto_confirm: If True, automatically use default Docker without prompting
    
    Returns:
        RuntimeEnvironment: Selected runtime configuration
    """
    # First detect current environment
    current_env = detect_runtime_environment()
    
    # Show detection results
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Runtime Environment Detection[/bold cyan]")
    console.print("=" * 60)
    
    status_icon = "✅" if current_env.runtime_type == RuntimeType.LOCAL else "⚠️"
    console.print(f"\n{status_icon} [bold]Current Environment Status:[/bold]")
    console.print(f"  • GPU Available: {'✅ Yes' if current_env.has_gpu else '❌ No'}")
    console.print(f"  • PyTorch Installed: {'✅ Yes' if current_env.has_torch else '❌ No'}")
    console.print(f"  • Triton Installed: {'✅ Yes' if current_env.has_triton else '❌ No'}")
    
    # If local environment is complete, use it
    if current_env.runtime_type == RuntimeType.LOCAL:
        console.print("\n✅ [bold green]Local environment is ready![/bold green]")
        if auto_confirm or Confirm.ask("Use local environment?", default=True):
            return current_env
    
    # Check if Docker is available
    docker_available = check_docker_available()
    
    if not docker_available:
        console.print("\n[bold red]❌ Docker not available![/bold red]")
        console.print("\nTo run GPU kernels, you need either:")
        console.print("  1. Local GPU with torch + triton installed")
        console.print("  2. Docker with GPU support")
        console.print("\n[yellow]Continuing with local environment (some features may not work)[/yellow]")
        return current_env
    
    # Offer Docker option
    console.print("\n[bold yellow]⚠️  Local environment incomplete[/bold yellow]")
    console.print("\nDocker is available. Options:")
    console.print(f"  1. Use Docker (default image: {DEFAULT_DOCKER_IMAGE})")
    console.print("  2. Use Docker (custom image)")
    console.print("  3. Continue with local environment (limited functionality)")
    
    if auto_confirm:
        console.print(f"\n[bold green]Using default Docker image:[/bold green] {DEFAULT_DOCKER_IMAGE}")
        choice = "1"
    else:
        choice = Prompt.ask(
            "Select option",
            choices=["1", "2", "3"],
            default="1"
        )
    
    if choice == "1":
        # Use default Docker image
        console.print(f"\n✅ [bold green]Using Docker:[/bold green] {DEFAULT_DOCKER_IMAGE}")
        return RuntimeEnvironment(
            runtime_type=RuntimeType.DOCKER,
            docker_image=DEFAULT_DOCKER_IMAGE,
            docker_devices=DEFAULT_DOCKER_DEVICES.copy(),
            has_gpu=True,
            has_triton=True,
            has_torch=True
        )
    
    elif choice == "2":
        # Custom Docker image
        custom_image = Prompt.ask("Enter Docker image name")
        
        # Ask about GPU devices
        if Confirm.ask("Add GPU devices (/dev/kfd, /dev/dri)?", default=True):
            devices = DEFAULT_DOCKER_DEVICES.copy()
        else:
            devices = []
        
        console.print(f"\n✅ [bold green]Using Docker:[/bold green] {custom_image}")
        return RuntimeEnvironment(
            runtime_type=RuntimeType.DOCKER,
            docker_image=custom_image,
            docker_devices=devices,
            has_gpu=len(devices) > 0,
            has_triton=True,  # Assume Docker image has it
            has_torch=True    # Assume Docker image has it
        )
    
    else:
        # Continue with local
        console.print("\n[yellow]Continuing with local environment[/yellow]")
        return current_env


def get_runtime_config_for_agent(runtime_env: RuntimeEnvironment, workspace_path: str = "") -> dict:
    """
    Convert RuntimeEnvironment to agent-compatible config.
    
    Args:
        runtime_env: Runtime environment configuration
        workspace_path: Path to workspace directory to mount in Docker
    
    Returns:
        dict: Configuration for the agent's environment
    """
    if runtime_env.runtime_type == RuntimeType.DOCKER:
        # Build Docker run arguments
        run_args = ["--rm"]
        
        # Add GPU devices
        for device in runtime_env.docker_devices:
            run_args.extend(["--device", device])
        
        # Add workspace volume if specified
        if workspace_path:
            workspace_path = os.path.abspath(workspace_path)
            run_args.extend(["-v", f"{workspace_path}:/workspace"])
        
        # Add volumes
        for volume in runtime_env.docker_volumes:
            run_args.extend(["-v", volume])
        
        return {
            "type": "docker",
            "image": runtime_env.docker_image,
            "cwd": "/workspace" if workspace_path else "/",
            "run_args": run_args,
            "env": {
                "PYTHONUNBUFFERED": "1",
            }
        }
    
    else:  # LOCAL or UNKNOWN
        return {
            "type": "local",
            "cwd": workspace_path or os.getcwd(),
        }


def display_runtime_info(runtime_env: RuntimeEnvironment):
    """Display runtime environment information in a nice panel."""
    if runtime_env.runtime_type == RuntimeType.DOCKER:
        info = f"""[bold]Runtime:[/bold] Docker
[bold]Image:[/bold] {runtime_env.docker_image}
[bold]GPU Devices:[/bold] {', '.join(runtime_env.docker_devices) if runtime_env.docker_devices else 'None'}
[bold]Status:[/bold] ✅ Ready for GPU operations"""
    else:
        status = "✅ Ready" if runtime_env.has_gpu and runtime_env.has_torch and runtime_env.has_triton else "⚠️ Limited"
        info = f"""[bold]Runtime:[/bold] Local
[bold]GPU:[/bold] {'✅ Available' if runtime_env.has_gpu else '❌ Not detected'}
[bold]PyTorch:[/bold] {'✅ Installed' if runtime_env.has_torch else '❌ Not installed'}
[bold]Triton:[/bold] {'✅ Installed' if runtime_env.has_triton else '❌ Not installed'}
[bold]Status:[/bold] {status}"""
    
    console.print(Panel(info, title="[bold cyan]Runtime Environment[/bold cyan]", expand=False))


if __name__ == "__main__":
    # Test the detection
    logging.basicConfig(level=logging.INFO)
    
    console.print("[bold]Testing Runtime Environment Detection[/bold]\n")
    
    env = detect_runtime_environment()
    display_runtime_info(env)
    
    console.print("\n[bold]Testing Interactive Prompt[/bold]\n")
    selected_env = prompt_runtime_environment(auto_confirm=False)
    display_runtime_info(selected_env)
    
    console.print("\n[bold]Agent Config:[/bold]")
    config = get_runtime_config_for_agent(selected_env, "/home/sapmajum/test_geak_workflow")
    console.print(config)
