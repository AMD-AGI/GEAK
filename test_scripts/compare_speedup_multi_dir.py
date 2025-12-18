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


def parse_baseline(path: Path):
    """Parse baseline file, return list of bandwidth values (G/s)"""
    try:
        lines = path.read_text(encoding="utf8").strip().split("\n")
        return [float(line.strip()) for line in lines if line.strip()]
    except Exception as e:
        print(f"❌ Error reading baseline file: {path} ({e})")
        return []


def extract_benchmark_name(log_filename: str):
    """Extract benchmark name from log filename
    Example: benchmark_device_merge_sort_1.log -> device_merge_sort
    """
    # Remove .log extension
    name = log_filename.replace(".log", "")
    # Remove benchmark_ prefix
    if name.startswith("benchmark_"):
        name = name[len("benchmark_"):]
    # Remove _N suffix (where N is a number)
    name = re.sub(r"_\d+$", "", name)
    return name


def compare_with_baseline(log_file: Path, baseline_file: Path):
    """Compare log file with baseline, return avg speedup or None if error"""
    log_values = parse_log(log_file)
    baseline_values = parse_baseline(baseline_file)

    if not log_values:
        print(f"  ❌ No valid data in log: {log_file.name}")
        return None

    if not baseline_values:
        print(f"  ❌ No valid data in baseline: {baseline_file.name}")
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
        print("Usage: python compare_speedup_multi_dir.py <base_name> [baseline_dir]")
        print("Example: python compare_speedup_multi_dir.py 20251209_block_histogram_agent")
        print("  Will search in: base_name0, base_name1, ..., base_name4")
        sys.exit(1)

    base_input = sys.argv[1]
    baseline_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./baseline_bandwidth")

    if not baseline_dir.exists():
        print(f"❌ Baseline directory not found: {baseline_dir}")
        sys.exit(1)

    # Determine base name and parent directory
    base_path = Path(base_input)
    if base_path.exists() and base_path.is_dir():
        # If input is an existing directory, use its name and parent
        base_name = base_path.name
        parent_dir = base_path.parent
    else:
        # If input is just a name, use it directly and current directory as parent
        base_name = base_path.name if base_path.name else base_input
        parent_dir = Path(".")

    # Search in agent0-4 directories
    # Directory format: base_name0, base_name1, etc. (e.g., 20251209_block_histogram_agent0)
    agent_dirs = []
    for i in range(5):
        agent_dir = parent_dir / f"{base_name}{i}"
        if agent_dir.exists():
            agent_dirs.append((i, agent_dir))
        else:
            print(f"⚠ Agent directory not found: {agent_dir}")

    if not agent_dirs:
        print(f"❌ No agent directories found (looking for {base_name}0-4)")
        sys.exit(1)

    print(f"✅ Found {len(agent_dirs)} agent directory(ies) to search\n")

    # Find all benchmark_*_N.log files from all agent directories
    log_files = []
    for agent_id, agent_dir in agent_dirs:
        for log_file in sorted(agent_dir.glob("benchmark_*_[0-9]*.log")):
            if log_file.stat().st_size == 0:
                print(f"⚠ Skipping empty file: {log_file.name} (agent{agent_id})")
                continue
            log_files.append((agent_id, log_file))

    if not log_files:
        print("❌ No valid log files found (looking for benchmark_*_N.log)")
        sys.exit(1)

    print(f"✅ Found {len(log_files)} log file(s) to compare\n")

    best_speedup = 0.0
    best_log = None
    best_agent = None
    results = []

    for agent_id, log_file in log_files:
        print(f"📊 Processing: {log_file.name} (agent{agent_id})")
        
        # Extract benchmark name
        bench_name = extract_benchmark_name(log_file.name)
        baseline_file = baseline_dir / f"{bench_name}.txt"

        if not baseline_file.exists():
            print(f"  ❌ Baseline file not found: {baseline_file.name}\n")
            continue

        print(f"  📌 Baseline: {baseline_file.name}")

        avg_speedup = compare_with_baseline(log_file, baseline_file)

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
        print(f"🏆 Best log: {best_log.name}")
        print(f"🏆 Best agent: agent{best_agent}")
    else:
        print("⚠ No valid speedups calculated")
    print("=" * 60)

    # Print summary
    if results:
        print(f"\n📋 Summary ({len(results)} valid results):")
        for agent_id, log_file, speedup in sorted(results, key=lambda x: x[2], reverse=True):
            print(f"  {speedup:.3f}x  agent{agent_id}/{log_file.name}")

