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
from minisweagent.models import get_model


def main():
    """Run test with actual data and metric_model config."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test _select_best_from_parallel_runs with metric_model config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="mini_select_patch",
        help="Config file name/path (with or without .yaml). Default: mini_select_patch"
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
        print("  Tip: available builtin configs include:")
        print("   - mini")
        print("   - mini_kernel")
        print("   - mini_select_patch")
        print("   - mini_system_prompt")
        print("  You can also pass a direct path to a .yaml file via --config.\n")
        config = {}
    
    # Resolve model config for SelectPatchAgent
    print("=" * 80)
    print("Resolving model config...")
    print("=" * 80)

    # Optional override: some configs may provide a dedicated metric_model section
    metric_model_config = config.get("metric_model", {}) or {}

    # Default: use the config's model/model_name (same as normal runner)
    model_config = dict(config.get("model", {}) or {})
    model_name = config.get("model_name")
    if model_name and "model_name" not in model_config:
        model_config["model_name"] = model_name

    if metric_model_config:
        print("✓ Using metric_model config override\n")
        model_config = metric_model_config
    else:
        print("✓ Using config.model / model_name\n")

    print(f"  Model name: {model_config.get('model_name', 'not specified')}")
    print(f"  Model class: {model_config.get('model_class', model_config.get('model_class', 'not specified'))}\n")
    
    # Test with metric model config
    print("=" * 80)
    print("Calling _select_best_from_parallel_runs with metric_model config...")
    print("=" * 80)
    
    model_factory = lambda: get_model(config=model_config)

    result = ParallelAgent._select_best_from_parallel_runs(
        base_patch_dir=base_patch_dir,
        num_parallel=args.num_parallel,
        metric=args.metric,
        model_factory=model_factory,
    )
    
    if result:
        print(f"result: {result}")
    else:
        print("\n✗ Failed: Result is None")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

