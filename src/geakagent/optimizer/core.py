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
    """Use OpenEvolve optimizer via MCP (openevolve-mcp calls run_openevolve.py from geak-oe)."""
    kernel_path = kwargs.get("kernel_path")
    if not kernel_path:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(kernel_code)
            kernel_path = Path(f.name)
    else:
        kernel_path = Path(kernel_path)

    # openevolve-mcp tool API: kernel_path, max_iterations, gpu, output_dir, commandment_path, baseline_metrics_path
    mcp_request: Dict[str, Any] = {
        "kernel_path": str(kernel_path.resolve()),
        "max_iterations": kwargs.get("max_iterations", 10),
        "gpu": kwargs.get("gpu", 0),
        "output_dir": kwargs.get("output_dir"),
        "commandment_path": kwargs.get("commandment_path"),
        "baseline_metrics_path": kwargs.get("baseline_metrics_path"),
    }
    # Drop None values so MCP tool uses its defaults
    mcp_request = {k: v for k, v in mcp_request.items() if v is not None}

    try:
        from mcp_client import MCPClient
        import asyncio

        mcp_path = Path(__file__).parent.parent.parent.parent / "mcp_tools" / "openevolve-mcp"
        server_config = {
            "command": ["python3", "-m", "openevolve_mcp.server"],
            "cwd": str(mcp_path),
            "env": {"PYTHONPATH": str(mcp_path / "src")},
        }
        async def run_mcp():
            async with MCPClient("openevolve-mcp", server_config) as client:
                return await client.call_tool("optimize_kernel", mcp_request)
        result = asyncio.run(run_mcp())
    except ImportError:
        # Fallback: invoke run_openevolve.py via openevolve-mcp server module (same as MCP tool does)
        import os
        import sys
        mcp_src = Path(__file__).parent.parent.parent.parent / "mcp_tools" / "openevolve-mcp" / "src"
        if str(mcp_src) not in sys.path:
            sys.path.insert(0, str(mcp_src))
        os.environ.setdefault("GEAK_OE_ROOT", str(Path(__file__).parent.parent.parent.parent / "geak-oe"))
        from openevolve_mcp.server import optimize_kernel as _mcp_optimize_kernel
        # optimize_kernel is the @mcp.tool() function; call it with same args as MCP
        result = _mcp_optimize_kernel(**mcp_request)

    if not result.get("success"):
        raise RuntimeError(result.get("error", "OpenEvolve failed"))

    # Map MCP response to OptimizeResult (tool returns best_kernel_path, speedup, commandment_path, etc.)
    best_path = result.get("best_kernel_path") or ""
    optimized_code = ""
    if best_path and Path(best_path).is_file():
        optimized_code = Path(best_path).read_text()
    metrics = {
        "speedup": float(result.get("speedup", 1.0)),
        "best_score": float(result.get("best_score", 1.0)),
        "baseline_latency_us": float(result.get("baseline_latency_us", 0)),
        "best_latency_us": float(result.get("best_latency_us", 0)),
    }
    return OptimizeResult(
        optimized_code=optimized_code,
        metrics=metrics,
        optimizer_used="openevolve",
        iterations=int(result.get("iterations_completed", 0)),
    )


def _optimize_with_autotune(kernel_code: str, **kwargs) -> OptimizeResult:
    """Use autotune optimizer (placeholder for now)."""
    # TODO: Implement simple parameter search
    raise NotImplementedError("Autotune optimizer not yet implemented")
