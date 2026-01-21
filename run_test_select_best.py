#!/usr/bin/env python3
"""Quick test script for _select_best_from_parallel_runs method.

This script can be run directly without pytest to test the functionality.
It uses the metric_model configuration to select the best patch from parallel runs.
The metric_model configuration is read from mini_patch_agent.yaml.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.config import get_config_path


def main():
    """Run test with actual data and metric_model config."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test _select_best_from_parallel_runs with metric_model config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="mini_patch_agent",
        help="Config file name (without .yaml extension). Default: mini_patch_agent"
    )
    parser.add_argument(
        "--base-patch-dir",
        type=str,
        default="/data/users/yueliu14/mini-swe-agent/20251230_v3_device_segmented_reduce/",
        help="Base directory containing parallel_* subdirectories"
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Number of parallel runs"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="To select the best patch, you should Calculate speedup point-by-point between PATCH and BASELINE, \
          then take the mean of those speedup. Do not average the bandwidth values before computing speedup.",
        help="Metric name for evaluation"
    )
    
    args = parser.parse_args()
    
    base_patch_dir = Path(args.base_patch_dir)
    
    if not base_patch_dir.exists():
        print(f"ERROR: Test data directory not found: {base_patch_dir}")
        print("Please ensure the test data directory exists.")
        return 1
    
    print(f"Testing with data from: {base_patch_dir}")
    print(f"Available parallel directories: {list(base_patch_dir.glob('parallel_*'))}\n")
    
    # Load config file
    print("=" * 80)
    print("Loading config file...")
    print("=" * 80)
    
    try:
        config_path = get_config_path(args.config)
        print(f"✓ Config file found: {config_path}")
        config = yaml.safe_load(config_path.read_text())
        print(f"✓ Config loaded\n")
    except Exception as e:
        print(f"✗ Failed to load config file: {e}")
        return 1
    
    # Get metric model config
    print("=" * 80)
    print("Loading metric_model config...")
    print("=" * 80)
    
    metric_model_config = config.get("metric_model", {})
    if not metric_model_config:
        print("✗ Warning: No metric_model config found, using default")
        metric_model_config = {}
    else:
        print(f"✓ Metric model config loaded")
        print(f"  Model name: {metric_model_config.get('model_name', 'not specified')}")
        print(f"  Model class: {metric_model_config.get('model_class', 'not specified')}\n")
    
    # Test with metric model config
    print("=" * 80)
    print("Calling _select_best_from_parallel_runs with metric_model config...")
    print("=" * 80)
    
    result = ParallelAgent._select_best_from_parallel_runs(
        base_patch_dir=base_patch_dir,
        num_parallel=args.num_parallel,
        metric=args.metric,
        metric_model_config=metric_model_config
    )
    
    if result:
        print(f"result: {result}")
    else:
        print("\n✗ Failed: Result is None")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

