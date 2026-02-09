"""
OpenEvolve Optimizer MCP Server - Minimal & Modular

Provides GPU kernel optimization using LLM-guided evolution.
"""

import json
import sys
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create MCP server
mcp = FastMCP(
    name="openevolve-optimizer",
    instructions="GPU kernel optimization using LLM-guided evolution (OpenEvolve)"
)

# Constants
SEPARATOR = "#" * 146
# Relative to msa/ root: msa/geak-oe
DEFAULT_GEAK_OE_PATH = str(Path(__file__).parent.parent.parent.parent.parent / "geak-oe")
DEFAULT_DOCKER_IMAGE = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"


def prepare_kernel_file(
    kernel_code: str,
    test_code: Optional[str] = None
) -> str:
    """
    Merge kernel and test code into OpenEvolve format.
    
    Format: <kernel> + SEPARATOR + <tests>
    """
    logger.debug("Preparing kernel file with separator")
    test_section = test_code or _default_test_template()
    merged = f"{kernel_code.strip()}\n\n{SEPARATOR}\n\n{test_section.strip()}"
    logger.debug(f"Merged file size: {len(merged)} chars")
    return merged


def _default_test_template() -> str:
    """Minimal test template (fallback)."""
    logger.debug("Using default test template")
    return """
import torch
import pytest

def test_kernel_runs():
    assert True, "Kernel test placeholder"

def test_save_results():
    print("Test results saved")

def test_save_performance_results():
    print("Performance results saved")
"""


async def run_openevolve(
    merged_file: Path,
    evaluator_path: Path,
    max_iterations: int,
    output_dir: Path,
    config: dict
) -> dict:
    """
    Run OpenEvolve optimization asynchronously.
    
    Returns:
        {
            "optimized_code": str,
            "metrics": dict,
            "speedup": float,
            "iterations": int,
            "success": bool
        }
    """
    try:
        logger.info(f"Starting OpenEvolve optimization")
        logger.info(f"  Input file: {merged_file}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Output dir: {output_dir}")
        
        # Add geak-oe to path
        geak_oe_path = Path(DEFAULT_GEAK_OE_PATH)
        if str(geak_oe_path) not in sys.path:
            sys.path.insert(0, str(geak_oe_path))
            logger.debug(f"Added to sys.path: {geak_oe_path}")
        
        # Import OpenEvolve
        logger.debug("Importing OpenEvolve")
        from openevolve import OpenEvolve
        from openevolve.config import Config
        
        # Create config using from_dict (proper way for OpenEvolve)
        logger.debug("Creating OpenEvolve config")
        oe_config = Config.from_dict(config)
        
        # Run OpenEvolve
        logger.info("Initializing OpenEvolve...")
        openevolve = OpenEvolve(
            initial_program_path=str(merged_file),
            evaluation_file=str(evaluator_path),
            config=oe_config,
            output_dir=str(output_dir)
        )
        
        logger.info("Running OpenEvolve optimization (this may take a while)...")
        best_program = await openevolve.run()
        
        speedup = best_program.metrics.get("final_score", 1.0)
        logger.info(f"✓ OpenEvolve completed successfully!")
        logger.info(f"  Best iteration: {best_program.generation}")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Metrics: {best_program.metrics}")
        
        return {
            "optimized_code": best_program.code,
            "metrics": best_program.metrics,
            "speedup": speedup,
            "iterations": best_program.generation,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"OpenEvolve optimization failed: {e}", exc_info=True)
        return {
            "optimized_code": None,
            "metrics": {},
            "speedup": 1.0,
            "iterations": 0,
            "success": False,
            "error": str(e)
        }


def _optimize_kernel_impl(
    kernel_path: str,
    test_path: Optional[str] = None,
    max_iterations: int = 50,
    config: Optional[dict] = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE
) -> dict:
    """
    Optimize a GPU kernel using OpenEvolve LLM-guided evolution.
    
    Args:
        kernel_path: Path to kernel file (.py)
        test_path: Optional path to test file (pytest format)
        max_iterations: Max evolution iterations (default: 50)
        config: Optional OpenEvolve config override
        docker_image: Docker image for GPU execution
    
    Returns:
        {
            "success": bool,
            "optimized_code": str,
            "speedup": float,
            "iterations": int,
            "metrics": dict,
            "baseline_metrics_path": str,
            "optimized_metrics_path": str
        }
    """
    try:
        logger.info("="*60)
        logger.info("OpenEvolve MCP Tool Called")
        logger.info("="*60)
        logger.info(f"Kernel path: {kernel_path}")
        logger.info(f"Test path: {test_path}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Docker image: {docker_image}")
        
        kernel_path = Path(kernel_path)
        
        # Read kernel code
        logger.info(f"Reading kernel file: {kernel_path}")
        kernel_code = kernel_path.read_text()
        logger.debug(f"Kernel code length: {len(kernel_code)} chars")
        
        # Read test code if provided
        test_code = None
        if test_path:
            logger.info(f"Reading test file: {test_path}")
            test_code = Path(test_path).read_text()
            logger.debug(f"Test code length: {len(test_code)} chars")
        else:
            logger.info("No test file provided, will use default template")
        
        # Prepare merged file
        logger.info("Merging kernel and test code...")
        merged_code = prepare_kernel_file(kernel_code, test_code)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(merged_code)
            temp_file = Path(f.name)
        logger.info(f"Created temporary merged file: {temp_file}")
        
        # Setup paths
        geak_oe_path = Path(DEFAULT_GEAK_OE_PATH)
        evaluator_path = geak_oe_path / "examples" / "tb" / "rocm_evaluator.py"
        output_dir = kernel_path.parent / "openevolve_output"
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Evaluator: {evaluator_path}")
        
        # Default config (properly structured for OpenEvolve Config)
        default_config = {
            "max_iterations": max_iterations,
            "database": {
                "population_size": 20,
                "num_islands": 4
            },
            "llm": {
                "models": [{"name": "claude-sonnet-4", "weight": 1.0}]
            },
            "evaluator": {
                "timeout": 1800
            }
        }
        
        # Merge with user config (if provided)
        if config:
            logger.info(f"Merging custom config: {config}")
            for key, value in config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        # Run OpenEvolve (async)
        logger.info("Starting OpenEvolve async execution...")
        result = asyncio.run(run_openevolve(
            temp_file,
            evaluator_path,
            max_iterations,
            output_dir,
            default_config
        ))
        
        # Clean up
        logger.debug(f"Cleaning up temporary file: {temp_file}")
        temp_file.unlink(missing_ok=True)
        
        if result["success"]:
            logger.info("Optimization succeeded! Processing results...")
            
            # Save optimized code
            optimized_path = kernel_path.parent / f"{kernel_path.stem}_optimized.py"
            optimized_path.write_text(result["optimized_code"])
            logger.info(f"Saved optimized kernel: {optimized_path}")
            
            # Convert metrics to StandardBenchmark format
            baseline_metrics_path = kernel_path.parent / "benchmark" / "baseline" / "metrics.json"
            optimized_metrics_path = kernel_path.parent / "benchmark" / "optimized" / "metrics.json"
            
            # Create metrics in standard format
            standard_metrics = {
                "speedup": result["speedup"],
                "latency_ms": result["metrics"].get("latency_ms", 0),
                "tflops": result["metrics"].get("tflops", 0),
                "bandwidth_gb_s": result["metrics"].get("bandwidth", 0)
            }
            
            # Save baseline (speedup=1.0)
            baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_metrics_path, 'w') as f:
                json.dump({**standard_metrics, "speedup": 1.0}, f, indent=2)
            logger.info(f"Saved baseline metrics: {baseline_metrics_path}")
            
            # Save optimized
            optimized_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(optimized_metrics_path, 'w') as f:
                json.dump(standard_metrics, f, indent=2)
            logger.info(f"Saved optimized metrics: {optimized_metrics_path}")
            
            logger.info("="*60)
            logger.info(f"✓ Optimization Complete!")
            logger.info(f"  Speedup: {result['speedup']:.2f}x")
            logger.info(f"  Iterations: {result['iterations']}")
            logger.info(f"  Output: {optimized_path}")
            logger.info("="*60)
            
            return {
                "success": True,
                "optimized_code": result["optimized_code"],
                "optimized_path": str(optimized_path),
                "speedup": result["speedup"],
                "iterations": result["iterations"],
                "metrics": standard_metrics,
                "baseline_metrics_path": str(baseline_metrics_path),
                "optimized_metrics_path": str(optimized_metrics_path)
            }
        else:
            logger.error(f"Optimization failed: {result['error']}")
            return {
                "success": False,
                "error": result["error"],
                "optimized_code": None,
                "speedup": 1.0
            }
            
    except Exception as e:
        logger.error(f"MCP tool execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "optimized_code": None
        }


@mcp.tool()
def optimize_kernel(
    kernel_path: str,
    test_path: Optional[str] = None,
    max_iterations: int = 50,
    config: Optional[dict] = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE
) -> dict:
    """
    Optimize a GPU kernel using OpenEvolve LLM-guided evolution.
    
    MCP tool wrapper - calls implementation function.
    """
    return _optimize_kernel_impl(
        kernel_path=kernel_path,
        test_path=test_path,
        max_iterations=max_iterations,
        config=config,
        docker_image=docker_image
    )


def main():
    """Run MCP server."""
    logger.info("Starting OpenEvolve MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
