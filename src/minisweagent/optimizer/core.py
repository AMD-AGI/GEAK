"""
Compact optimizer core - wraps existing optimizers with unified interface.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


class OptimizerType(Enum):
    """Available optimizer types."""
    OPENEVOLVE = "openevolve"      # LLM-guided evolution
    AUTOTUNE = "autotune"          # Parameter search (future)
    AUTO = "auto"                   # Auto-select best


@dataclass
class OptimizeResult:
    """Compact optimization result."""
    optimized_code: str
    metrics: Dict[str, float]
    optimizer_used: str
    iterations: int = 1


def optimize_kernel(
    kernel_code: str,
    *,
    optimizer: OptimizerType = OptimizerType.AUTO,
    bottleneck: str = "balanced",
    strategy: Optional[str] = None,
    target_speedup: float = 2.0,
    budget_usd: float = 1.0,
    **kwargs
) -> OptimizeResult:
    """
    Optimize a kernel using specified optimizer.
    
    Args:
        kernel_code: Kernel code to optimize
        optimizer: Which optimizer to use (auto-selects if AUTO)
        bottleneck: Performance bottleneck type
        strategy: Specific optimization strategy (optional)
        target_speedup: Target speedup multiplier
        budget_usd: Max cost budget
        **kwargs: Additional optimizer-specific args
    
    Returns:
        OptimizeResult with optimized code and metrics
    
    Example:
        >>> result = optimize_kernel(
        ...     kernel_code=my_kernel,
        ...     bottleneck="latency",
        ...     target_speedup=2.0
        ... )
        >>> print(result.optimized_code)
    """
    # Auto-select optimizer if needed
    if optimizer == OptimizerType.AUTO:
        optimizer = _select_optimizer(bottleneck, budget_usd)
    
    # Route to appropriate optimizer
    if optimizer == OptimizerType.OPENEVOLVE:
        return _optimize_with_openevolve(
            kernel_code, bottleneck, strategy, **kwargs
        )
    elif optimizer == OptimizerType.AUTOTUNE:
        return _optimize_with_autotune(
            kernel_code, **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def _select_optimizer(bottleneck: str, budget: float) -> OptimizerType:
    """Auto-select best optimizer based on context."""
    # Simple heuristic for now
    if budget < 0.1:
        return OptimizerType.AUTOTUNE  # Fast, cheap
    else:
        return OptimizerType.OPENEVOLVE  # Powerful, expensive


def _optimize_with_openevolve(
    kernel_code: str,
    bottleneck: str,
    strategy: Optional[str],
    **kwargs
) -> OptimizeResult:
    """Use OpenEvolve optimizer via MCP."""
    import subprocess
    import json
    
    # Get kernel path
    kernel_path = kwargs.get("kernel_path")
    if not kernel_path:
        # Create temp file if just code provided
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(kernel_code)
            kernel_path = Path(f.name)
    else:
        kernel_path = Path(kernel_path)
    
    # Prepare MCP call
    mcp_request = {
        "kernel_path": str(kernel_path),
        "test_path": kwargs.get("test_path"),
        "max_iterations": kwargs.get("max_iterations", 50),
        "config": kwargs.get("openevolve_config"),
        "docker_image": kwargs.get("docker_image", "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x")
    }
    
    # Call OpenEvolve MCP tool
    # Try proper MCP protocol first, fallback to direct import
    try:
        # Option 1: Use real MCP protocol (recommended)
        from mcp_client import MCPClient
        import asyncio
        
        async def run_mcp():
            """Run MCP client asynchronously."""
            # Get server config from registry
            mcp_path = Path(__file__).parent.parent.parent.parent / "mcp_tools" / "openevolve-mcp"
            
            server_config = {
                "command": ["python3", "-m", "openevolve_mcp.server"],
                "cwd": str(mcp_path),
                "env": {
                    "PYTHONPATH": str(mcp_path / "src")
                }
            }
            
            async with MCPClient("openevolve-mcp", server_config) as client:
                result = await client.call_tool("optimize_kernel", mcp_request)
                return result
        
        # Run async MCP call
        result = asyncio.run(run_mcp())
        
    except ImportError:
        # Option 2: Fallback to direct import (backward compatibility)
        import sys
        # Path from core.py → optimizer/ → minisweagent/ → src/ → repo root/ → mcp_tools/openevolve-mcp/src/
        mcp_path = Path(__file__).parent.parent.parent.parent / "mcp_tools" / "openevolve-mcp" / "src"
        if str(mcp_path) not in sys.path:
            sys.path.insert(0, str(mcp_path))
        
        from openevolve_mcp.server import _optimize_kernel_impl as mcp_optimize_kernel
        
        # Call MCP implementation function directly (no protocol overhead)
        result = mcp_optimize_kernel(
            kernel_path=str(kernel_path),
            test_path=mcp_request.get("test_path"),
            max_iterations=mcp_request["max_iterations"],
            config=mcp_request.get("config"),
            docker_image=mcp_request["docker_image"]
        )
    
    # Process result
    if result.get("success"):
        return OptimizeResult(
            optimized_code=result["optimized_code"],
            metrics=result["metrics"],
            optimizer_used="openevolve",
            iterations=result["iterations"]
        )
    else:
        raise RuntimeError(f"OpenEvolve MCP failed: {result.get('error')}")


def _optimize_with_autotune(kernel_code: str, **kwargs) -> OptimizeResult:
    """Use autotune optimizer (placeholder for now)."""
    # TODO: Implement simple parameter search
    raise NotImplementedError("Autotune optimizer not yet implemented")
