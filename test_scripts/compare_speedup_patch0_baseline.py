#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def parse_log(path: Path):
    """Parse rocPRIM benchmark output, return list of bytes_per_second (converted to G/s) in order"""
    results = []

    # Support G/s + T/s
    pattern = re.compile(
        r"^(?P<name>.+?)/manual_time.*?bytes_per_second=(?P<bps>[\d\.]+)(?P<unit>[GT])/s",
        re.MULTILINE,
    )

    try:
        text = path.read_text(encoding="utf8")
    except Exception as e:
        print(f"❌ Error reading file: {path} ({e})")
        return results

    for m in pattern.finditer(text):
        bps = float(m.group("bps"))
        unit = m.group("unit")

        # Convert T/s → G/s
        if unit == "T":
            bps *= 1024.0

        results.append(bps)

    return results


def find_patch0_baseline(agent_dirs: list[tuple[int, Path]]):
    """Find patch_0 file to use as baseline (first valid one)."""
    for agent_id, agent_dir in sorted(agent_dirs, key=lambda x: x[0]):
        patch0_candidates = list(agent_dir.glob("patch_0_test.txt")) + list(agent_dir.glob("patch_0.patch"))
        for patch0_file in patch0_candidates:
            if patch0_file.stat().st_size > 0:
                values = parse_log(patch0_file)
                if values:
                    print(f"✅ Using baseline from: agent{agent_id}/{patch0_file.name}")
                    return agent_id, patch0_file, values
    
    return None, None, None


def compare_with_baseline(log_file: Path, baseline_values: list[float]):
    """Compare log file with baseline values, return avg speedup or None if error"""
    log_values = parse_log(log_file)

    if not log_values:
        print(f"  ❌ No valid data in log: {log_file.name}")
        return None

    if len(log_values) != len(baseline_values):
        print(f"  ❌ Mismatch: log has {len(log_values)} tests, baseline has {len(baseline_values)} tests")
        return None

    speedups = []
    for i, (opt_val, base_val) in enumerate(zip(log_values, baseline_values)):
        if base_val > 0:
            speedup = opt_val / base_val
            speedups.append(speedup)
        else:
            print(f"  ⚠ Warning: baseline value at index {i} is 0, skipping")

    if not speedups:
        print(f"  ❌ No valid speedups calculated")
        return None

    return sum(speedups) / len(speedups)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_speedup_patch0_baseline.py <base_dir>")
        print("Example: python compare_speedup_patch0_baseline.py 20260109_device_histogram/")
        print("  Will use the first valid patch_0 as baseline")
        sys.exit(1)

    base_input = sys.argv[1]
    base_path = Path(base_input)

    if not base_path.exists() or not base_path.is_dir():
        print(f"❌ Directory not found: {base_path}")
        sys.exit(1)

    # Find all parallel_* directories
    agent_dirs = []
    for i in range(10):
        parallel_dir = base_path / f"parallel_{i}"
        if parallel_dir.exists():
            agent_dirs.append((i, parallel_dir))

    if not agent_dirs:
        print(f"❌ No parallel_* directories found in {base_path}")
        sys.exit(1)

    print(f"✅ Found {len(agent_dirs)} parallel directory(ies)\n")

    # Find patch_0 baseline
    baseline_agent_id, baseline_file, baseline_values = find_patch0_baseline(agent_dirs)

    if baseline_file is None:
        print("❌ No valid patch_0 file found to use as baseline")
        sys.exit(1)

    print(f"📊 Baseline has {len(baseline_values)} test values\n")

    # Find all patch files (excluding the baseline itself)
    log_files = []
    for agent_id, agent_dir in agent_dirs:
        for log_file in sorted(agent_dir.glob("patch_*_test.txt")):
            # Skip the baseline file
            if agent_id == baseline_agent_id and log_file == baseline_file:
                continue
            
            if log_file.stat().st_size == 0:
                print(f"⚠ Skipping empty file: {log_file.name} (agent{agent_id})")
                continue
            
            log_files.append((agent_id, log_file))
        
        # Also check for .patch files
        for log_file in sorted(agent_dir.glob("patch_*.patch")):
            # Skip the baseline file
            if agent_id == baseline_agent_id and log_file == baseline_file:
                continue
            
            if log_file.stat().st_size == 0:
                print(f"⚠ Skipping empty file: {log_file.name} (agent{agent_id})")
                continue
            
            log_files.append((agent_id, log_file))

    if not log_files:
        print("❌ No other patch files found to compare")
        sys.exit(1)

    print(f"✅ Found {len(log_files)} patch file(s) to compare\n")

    best_speedup = 0.0
    best_log = None
    best_agent = None
    results = []

    for agent_id, log_file in log_files:
        print(f"📊 Processing: agent{agent_id}/{log_file.name}")

        avg_speedup = compare_with_baseline(log_file, baseline_values)

        if avg_speedup is not None:
            print(f"  ✅ Average speedup = {avg_speedup:.3f}x\n")
            results.append((agent_id, log_file, avg_speedup))
            if avg_speedup > best_speedup:
                best_speedup = avg_speedup
                best_log = log_file
                best_agent = agent_id
        else:
            print()

    print("=" * 60)
    if best_log:
        print(f"🏆 BEST speedup = {best_speedup:.3f}x")
        print(f"🏆 Best patch: agent{best_agent}/{best_log.name}")
    else:
        print("⚠ No valid speedups calculated")
    print("=" * 60)

    # Print summary
    if results:
        print(f"\n📋 Summary ({len(results)} valid results):")
        for agent_id, log_file, speedup in sorted(results, key=lambda x: x[2], reverse=True):
            print(f"  {speedup:.3f}x  agent{agent_id}/{log_file.name}")

