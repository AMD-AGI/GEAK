#!/usr/bin/env python3
"""Quick test script for _select_best_from_parallel_runs method.

This script can be run directly without pytest to test the functionality.
It uses a real model to select the best patch from parallel runs.
The model configuration is read from mini_patch_agent.yaml.
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
    """Run test with actual data and real model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test _select_best_from_parallel_runs with real model")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="mini_patch_agent",
        help="Config file name (without .yaml extension). Default: mini_patch_agent"
    )
    parser.add_argument(
        "--base-patch-dir",
        type=str,
        default="/mnt/raid0/yueliu14/rocprim/20251218_device_binary_search",
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
        default="bytes_per_second",
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
    
    # Initialize real model from config
    print("=" * 80)
    print("Initializing model from config...")
    print("=" * 80)
    
    try:
        model = get_model(config=config.get("model", {}))
        print(f"✓ Model initialized: {model.config.model_name}")
        print(f"  Model class: {type(model).__name__}\n")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        print(f"\nPlease check the model configuration in: {config_path}")
        return 1
    
    # Test with real model
    print("=" * 80)
    print("Calling _select_best_from_parallel_runs with real model...")
    print("=" * 80)
    
    result = ParallelAgent._select_best_from_parallel_runs(
        base_patch_dir=base_patch_dir,
        num_parallel=args.num_parallel,
        metric=args.metric,
        model=model
    )
    
    if result:
        print(f"\n✓ Success!")
        print(f"  Selected: agent_{result.agent_id}/{result.patch_id}")
        print(f"  Test passed: {result.test_passed}")
        print(f"  Return code: {result.returncode}")
        if result.llm_conclusion:
            print(f"  Analysis preview: {result.llm_conclusion[:200]}...")
        print(f"  Model calls: {model.n_calls}")
        print(f"  Model cost: ${model.cost:.4f}")
        
        # Show patch file path if available
        patch_file = base_patch_dir / f"parallel_{result.agent_id}" / f"{result.patch_id}.patch"
        if patch_file.exists():
            print(f"  Patch file: {patch_file}")
            print(f"  Patch size: {patch_file.stat().st_size} bytes")
    else:
        print("\n✗ Failed: Result is None")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

