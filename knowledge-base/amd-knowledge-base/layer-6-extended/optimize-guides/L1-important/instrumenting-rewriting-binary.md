Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprof-sys", "instrumentation", "binary-rewrite"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/instrumenting-rewriting-binary-application.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Instrumenting and Rewriting a Binary Application

## Overview

The ROCm Systems Profiler provides three methods for performing instrumentation using the `rocprof-sys-instrument` executable:

1. **Runtime instrumentation** - instruments the target executable and loaded shared libraries
2. **Process attachment** - connects to an already running process
3. **Binary rewrite** - generates a new executable or library with instrumentation embedded

## Runtime Instrumentation

Runtime instrumentation represents the default mode when neither `-p` nor `-o` options are specified. This approach instruments both the target executable and its dynamically-linked dependencies, offering comprehensive analysis coverage.

However, this mode carries trade-offs: "Runtime instrumentation supports instrumenting not only the target executable but also the shared libraries loaded by the target executable. Consequently, this mode consumes more memory, takes longer to perform the instrumentation, and tends to add more significant overhead to the runtime of the application."

**Syntax:**
```bash
rocprof-sys-instrument <rocprof-sys-options> -- <exe> [<exe-options>...]
```

## Process Attachment

This alpha feature attaches to active processes using the `-p <PID>` flag, similar to `gdb -p`. The same memory and overhead considerations apply as with runtime instrumentation. Note: "detaching from the target process without ending the target process is not currently supported."

**Syntax:**
```bash
rocprof-sys-instrument <rocprof-sys-options> -p <PID> -- <exe-name>
```

## Binary Rewrite

Binary rewriting generates instrumented executables or libraries with the `-o` option. This method exclusively modifies the text section, avoiding dynamic library instrumentation. The benefits include significantly faster instrumentation and reduced runtime overhead.

**Advantages:** "Binary rewriting is the recommended mode when the target executable uses process-level parallelism (for example, MPI)"

**Syntax:**
```bash
rocprof-sys-instrument <rocprof-sys-options> -o <output-file> -- <exe-or-library>
```

### Library Rewriting Workflow

For applications with minimal main routines and substantial functionality in dynamic libraries, follow this process:

1. Identify dynamically-linked libraries using `ldd`
2. Generate binary rewrites of both the executable and target libraries
3. Maintain matching library names (e.g., `libfoo.so.2`)
4. Output instrumented libraries to a separate directory
5. Prepend `LD_LIBRARY_PATH` with the instrumented library directory
6. Verify resolution with `ldd`

## Selective Instrumentation

The tool applies default heuristics to determine which functions receive instrumentation:

- Skip dynamic call-sites (function pointers) by default
- Exclude functions with fewer than 1024 instructions
- Skip instrumentation points requiring traps
- Exclude loop-level instrumentation
- Avoid functions with overlapping bodies or multiple entry points

Options like `--min-instructions`, `--min-address-range`, `--traps`, and `--allow-overlapping` modify these behaviors.

### Filtering Functions and Modules

Six command-line options accept regular expressions for scope customization:

- `--module-include` / `--function-include` - force inclusion without excluding others
- `--module-restrict` / `--function-restrict` - exclusively select matching patterns
- `--module-exclude` / `--function-exclude` - always applied regardless of other options

## Diagnostic Output

Running `rocprof-sys-instrument` generates files detailing available, instrumented, excluded, and overlapping functions in the `rocprof-sys-<NAME>-output` directory.

Use `--simulate` to generate these files without execution or binary generation:

```bash
rocprof-sys-instrument --simulate -- foo
rocprof-sys-instrument --simulate -o foo.inst -- foo
```

## Default Configuration Embedding

The `--env` option embeds configuration defaults into generated binaries, preserving settings across sessions:

```bash
rocprof-sys-instrument -o ./foo.samp \
  --env ROCPROFSYS_USE_SAMPLING=ON ROCPROFSYS_SAMPLING_FREQ=5 -- ./foo
```

These defaults can be overridden in subsequent sessions through environment variable reassignment.

## Sampling Mode

The deprecated `--mode sampling` option instruments only the main function while activating CPU call-stack and system-level thread sampling. Modern implementations should use call stack sampling instead.

## Troubleshooting

### RPATH Management

For binary-rewritten libraries, verify library resolution using `ldd` to ensure instrumented versions load correctly. Modify RPATH settings if necessary to point to instrumented library locations.
