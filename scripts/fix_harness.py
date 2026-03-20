#!/usr/bin/env python3
"""Fix kernel_oe.py harness to match test_topk_harness.py exactly."""

path = "/home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/topk/kernel_oe.py"
with open(path) as f:
    lines = f.readlines()

sep_line = "#" * 146
sep_idx = None
for i, line in enumerate(lines):
    if line.strip() == sep_line:
        sep_idx = i
        break

assert sep_idx is not None, "Separator not found"
print(f"Separator at line {sep_idx + 1}")

harness_lines = lines[sep_idx + 1:]

# Fix 1: Revert speedup output format
for i, line in enumerate(harness_lines):
    if "Speedup (geomean):" in line:
        harness_lines[i] = '        print(f"{\'Geometric mean speedup:\':<22} {geomean_speedup:.2f}x")\n'
        print(f"Fix 1 (speedup format) at harness line {i}")
        break

# Fix 2: Revert default mode to benchmark-only
for i, line in enumerate(harness_lines):
    if "# Default: correctness then benchmark" in line:
        end_i = i + 1
        while end_i < len(harness_lines):
            if "print(" in harness_lines[end_i] and "62" in harness_lines[end_i]:
                break
            end_i += 1
        new_block = [
            "        # Default: benchmark (harness shapes)\n",
            '        print("\\n[Benchmark Mode]")\n',
            "        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)\n",
            "\n",
        ]
        harness_lines[i:end_i] = new_block
        print(f"Fix 2 (default mode) replaced lines {i}-{end_i}")
        break

# Fix 3: Remove check_correctness function
new_harness = []
skip = False
for i, line in enumerate(harness_lines):
    if "def check_correctness(" in line:
        skip = True
        print(f"Fix 3: removing check_correctness at harness line {i}")
        continue
    if skip:
        if line.startswith("def ") and "check_correctness" not in line:
            skip = False
        elif line.strip() == "" and i + 1 < len(harness_lines) and harness_lines[i + 1].startswith("def "):
            skip = False
            continue
        else:
            continue
    new_harness.append(line)

final = lines[: sep_idx + 1] + new_harness
with open(path, "w") as f:
    f.writelines(final)

print("\nDone! Verifying...")
with open(path) as f:
    content = f.read()

h = content.split("#" * 146, 1)[1]
assert "Geometric mean speedup:" in h, "Fix 1 failed"
assert "Speedup (geomean)" not in h, "Fix 1 incomplete"
assert "# Default: benchmark (harness shapes)" in h, "Fix 2 failed"
assert "def check_correctness" not in h, "Fix 3 failed"
assert "def run_correctness" in h
assert "def run_benchmark" in h
assert "def run_profile" in h
assert "def make_input" in h
assert "def reference_topk" in h
print("All checks passed!")
