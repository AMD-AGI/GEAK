Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "pc-sampling", "rocprofiler"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/rocprofiler-sdk/api-reference/pc_sampling.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK PC Sampling Method

## Overview

Program Counter (PC) sampling is a profiling technique that uses statistical approximation by "periodically sampling GPU program counters to create a histogram of kernel instruction execution." This device-wide sampling mechanism captures data on every compute unit simultaneously.

### Beta Feature Warning

This feature requires enabling the `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable. Users should be aware of potential risks including "hardware freeze" and possible need for cold restart in case of system instability.

## ROCprofiler-SDK PC Sampling Service

### Setting Up tool_init()

The configuration process involves three main steps:

1. **Create a context and buffer:**
   ```c
   rocprofiler_context_id_t ctx{0};
   rocprofiler_buffer_id_t buff;
   ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
   ROCPROFILER_CALL(rocprofiler_create_buffer(ctx, 8192, 2048,
                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                    pc_sampling_callback, user_data, &buff),
                    "buffer creation failed");
   ```

2. **Query available GPU agents:**
   Use `rocprofiler_query_available_agents` to identify GPU agents (filter by `ROCPROFILER_AGENT_TYPE_GPU`).

3. **Check agent PC sampling support:**
   Call `rocprofiler_query_pc_sampling_agent_configurations` to verify compatibility. "Only newer GPU architectures (MI200 onwards) support this feature."

4. **Configure the service:**
   ```c
   rocprofiler_configure_pc_sampling_service(ctx, agent_id, picked_cfg->method,
                                             picked_cfg->unit, 1000, buffer_id, 0);
   ```

### Important Consideration

On shared systems, multiple processes may conflict when configuring the service. "It is advisable for process A to repeat the querying process to observe configuration CB and reuse it for configuring the PC sampling service."

## Processing PC Samples

Samples arrive asynchronously through the `pc_sampling_callback` function. Process samples by iterating through headers and checking for the `ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING` category:

```c
void pc_sampling_callback(rocprofiler_context_id_t ctx,
                         rocprofiler_buffer_id_t buff,
                         rocprofiler_record_header_t** headers,
                         size_t num_headers, void* data, uint64_t drop_count) {
    for(size_t i = 0; i < num_headers; i++) {
        if(headers[i]->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) {
            // Process individual sample
        }
    }
}
```

Buffers can be flushed synchronously using `rocprofiler_buffer_flush` to trigger the callback.
