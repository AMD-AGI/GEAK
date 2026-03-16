---
tags: ["optimization", "performance", "profiling", "pc-sampling", "rocprofv3"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/rocprofiler-sdk/how-to/using-pc-sampling.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Using PC sampling

## Overview

PC (Program Counter) sampling is a GPU profiling technique that periodically samples the program counter during kernel execution. This method helps identify performance bottlenecks, understand kernel behavior, analyze code coverage, and find heavily executed code paths.

**Note:** PC sampling is only supported on AMD GPUs with architectures gfx90a and later.

## PC sampling availability and configuration

To check GPU support for PC sampling, run:

```bash
rocprofv3 -L
# or
rocprofv3 --list-avail
```

This displays available methods and configuration parameters. Example output for gfx90a shows support for the `host_trap` method with `time` unit and configurable intervals.

### Firmware Requirements for MI300X

Important firmware updates for MI300X are included in ROCm 7.0:
- **Host-trap PC sampling:** PSP TOS Firmware >= 00.36.02.59
- **Stochastic PC sampling:** MEC Firmware feature version 50, firmware >= 0x0000001a

Check firmware versions using:
```bash
sudo cat /sys/kernel/debug/dri/0/amdgpu_firmware_info | grep SOS
sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info | grep MEC
```

### Running PC sampling

Basic command structure:

```bash
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap \
  --pc-sampling-unit time --pc-sampling-interval 1 \
  --output-format csv -- <application_path>
```

This generates `agent_info.csv` and `pc_sampling_host_trap.csv` files prefixed with the process ID.

## PC sampling fields

Output files contain these fields:

- **Sample_Timestamp:** When the sample was captured
- **Exec_Mask:** Active SIMD lanes at sampling time
- **Dispatch_Id:** Source kernel dispatch identifier
- **Instruction:** Assembly instruction text
- **Instruction_Comment:** Source line mapping (requires debug symbols)
- **Correlation_Id:** API launch call ID matching dispatch ID

For comprehensive output, use JSON format:

```bash
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap \
  --pc-sampling-unit time --pc-sampling-interval 1 \
  --output-format json -- <application_path>
```

**Enabling Debug Symbols:** Compile applications with debug symbols to populate the `Instruction_Comment` field with source-line information, improving hotspot identification.

## Host-trap PC sampling and sampling skid

Host-trap uses a background kernel thread to periodically interrupt running waves and capture the PC. While effective, this software-based approach has a limitation: interrupt processing delays can cause sampling skid, where PC samples may be attributed to instructions up to two positions away from the actual latency source. This results in non-precise intra-kernel sampling.

When analyzing profiles, consider not only the highest-cost instruction but also adjacent instructions. If near branch instructions, examine both the branch target and the following instruction.

To address these limitations, hardware-based stochastic PC sampling provides precise intra-kernel sampling with zero skid.

**Host-trap Support:** MI200, MI300, MI325, MI350, and MI355 architectures.

## Hardware-based (stochastic) PC sampling method

Introduced for gfx942 architecture, `ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC` employs specialized hardware to probe actively running GPU waves. Beyond standard hotspot information, it provides:

- Whether a wave issued the sampled instruction
- The reason an instruction wasn't issued (stall reason)
- Additional state information for understanding instruction-per-cycle (IPC) behavior

### gfx942 Requirements

Stochastic sampling on gfx942 requires cycle-based intervals as powers of 2. Verify availability with:

```bash
rocprofv3 -L
```

### Profiling with stochastic sampling

```bash
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic \
  --pc-sampling-unit cycles --pc-sampling-interval 1048576 \
  --output-format csv,json -- <application_path>
```

This generates `pc_sampling_stochastic.csv` and `out_results.json`.

### Additional stochastic sampling fields

Compared to host-trap output:

- **Wave_Issued_Instruction:** 1 if instruction was issued, 0 otherwise
- **Instruction_Type:** Type of issued instruction (when Wave_Issued_Instruction = 1)
- **Stall_Reason:** Reason instruction wasn't issued (when Wave_Issued_Instruction = 0)
- **Wave_Count:** Total actively running waves on the compute unit

JSON output includes arbiter state fields (`arb_state_issue_*` and `arb_state_stall_*`) indicating instruction types issued or stalled at sampling time.

**Stochastic Support:** MI300, MI325, MI350, and MI355 architectures.
