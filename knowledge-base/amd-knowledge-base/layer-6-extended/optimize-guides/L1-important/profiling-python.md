---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "python"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Profiling Python Scripts

## Getting Started

The ROCm Systems Profiler Python package is installed in `lib/pythonX.Y/site-packages/rocprofsys`. To ensure proper package discovery, add this path to your `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=/opt/rocprofiler-systems/lib/python3.8/site-packages:${PYTHONPATH}
```

Both the setup script and module file automatically configure this path.

## Running ROCm Systems Profiler on a Python Script

ROCm Systems Profiler provides a `rocprof-sys-python` helper script that manages `PYTHONPATH` configuration and selects the correct Python interpreter. These commands are equivalent:

```bash
rocprof-sys-python --help
```

and

```bash
export PYTHONPATH=/opt/rocprofiler-systems/lib/python3.8/site-packages:${PYTHONPATH}
python3.8 -m rocprofsys --help
```

### Command Line Options

Key options available via `rocprof-sys-python --help` include:

- `-h, --help`: Display help information
- `-v VERBOSITY`: Set logging verbosity
- `-b, --builtin`: Enable decorator-based profiling with `@profile`
- `-c FILE`: Specify configuration file
- `-F [BOOL]`: Use full filepath instead of basename
- `--label`: Encode function arguments, filename, or line number in labels
- `-I, --function-include`: Include specific function names
- `-E, --function-exclude`: Exclude specific function names
- `-R, --function-restrict`: Profile only specific functions
- `-MI, --module-include`: Include entries from specified files
- `-ME, --module-exclude`: Exclude entries from specified files
- `-MR, --module-restrict`: Profile only specified modules
- `--trace-c [BOOL]`: Enable C function profiling within the Python interpreter

### Selective Instrumentation

Restrict profiling scope using command-line filters or the `@profile` decorator with the `-b` flag. For example:

```python
@profile
def inefficient(n):
   a = 0
   for i in range(n):
      a += i
      for j in range(n):
            a += j
   return a
```

Running with `rocprof-sys-python -b -- ./example.py` narrows instrumentation to this function and its children.

## ROCm Systems Profiler Python Source Instrumentation

Add source-level profiling by importing and decorating functions:

```python
import rocprofsys

@rocprofsys.profile()
def run(n):
   # function body
```

Alternatively, use context-manager syntax:

```python
if __name__ == "__main__":
   with rocprofsys.profile():
      run(20)
```

### Configuration

Configure the profiler within source code by modifying `rocprofsys.profiler.config`:

```python
from rocprofsys.profiler import config
from rocprofsys import profile

config.include_args = True
config.include_filename = False
config.include_line = False
config.restrict_functions += ["fib", "run"]

with profile():
   run(5)
```

Available configuration fields control argument inclusion, filename display, line number inclusion, and function filtering.
