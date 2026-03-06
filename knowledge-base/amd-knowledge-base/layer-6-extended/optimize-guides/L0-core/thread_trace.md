---
tags: ["optimization", "performance", "profiling", "thread-trace", "rocprofiler"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/rocprofiler-sdk/api-reference/thread_trace.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprof Trace Decoder and Thread Trace APIs

## Overview

Thread trace provides detailed insight into GPU kernel execution by collecting traces of shader instructions. It captures GPU occupancy, instruction execution times, performance counters, and wave execution behavior using hardware instrumentation.

**Note:** Thread trace generates substantial data volumes. Implementation of filtering strategies is recommended to focus on specific application areas of interest.

**Note:** ROCprof Trace Decoder is a binary-only library available at the [rocprof-trace-decoder releases](https://github.com/ROCm/rocprof-trace-decoder/releases) repository.

## Thread Trace Service API

### Setup in tool_init()

**Step 1: Configure callback tracing for code objects**

```c
rocprofiler_context_id_t ctx{0};
ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

ROCPROFILER_CALL(
    rocprofiler_configure_callback_tracing_service(ctx,
                                               ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                               nullptr,
                                               0,
                                               tool_codeobj_tracing_callback,
                                               nullptr),
    "code object tracing service configure");
```

**Step 2: Query available GPU agents**

```c
std::vector<rocprofiler_agent_id_t> agents{};

ROCPROFILER_CALL(
    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents,
           size_t _num_agents, void* _data) {
            auto* agent_v = static_cast<std::vector<rocprofiler_agent_id_t>*>(_data);
            for(size_t i = 0; i < _num_agents; ++i) {
                auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    agent_v->emplace_back(agent->id);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &agents),
    "Failed to iterate agents");
```

**Step 3: Configure optional parameters**

```c
std::vector<rocprofiler_thread_trace_parameter_t> params{};

params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0xF});
params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, 0});
params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, 0xF});
params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, 1u<<30}); // 1 GB
```

**Configuration Parameters:**

| Parameter | Description |
|-----------|-------------|
| SHADER_ENGINE_MASK | Determines which Shader Engines to trace (bitmask). Single SE tracing recommended to avoid data loss. |
| TARGET_CU | Specifies target Compute Unit or WGP. Only one CU/WGP per tracing session. |
| SIMD_SELECT | For gfx9: bitmask per SIMD lane; for gfx10+: single SIMD ID (mod4 for compatibility). |
| BUFFER_SIZE | Buffer size shared across all configured SEs. Larger sizes have minimal overhead. |

### Device Thread Trace

Enable asynchronous, device-wide thread trace:

```c
for(auto agent_id : agents) {
    ROCPROFILER_CALL(
        rocprofiler_configure_device_thread_trace_service(
            ctx,
            agent_id,
            params.data(),
            params.size(),
            shader_data_callback,
            nullptr),
        "thread trace service configure");
}
```

Start data collection:

```c
auto status = rocprofiler_start_context(ctx);
// Run application workload
status = rocprofiler_stop_context(ctx);
```

### Dispatch Thread Trace

Enable selective tracing based on kernel dispatches:

```c
// Optional: serialize all kernels
params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL, 1});

rocprofiler_thread_trace_control_flags_t
dispatch_callback(rocprofiler_agent_id_t agent_id,
                  rocprofiler_queue_id_t queue_id,
                  rocprofiler_async_correlation_id_t correlation_id,
                  rocprofiler_kernel_id_t kernel_id,
                  rocprofiler_dispatch_id_t dispatch_id,
                  void* userdata,
                  rocprofiler_user_data_t* dispatch_userdata) {
    if(target_kernel_id == kernel_id)
        return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
    return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
}

for(auto agent_id : agents) {
    ROCPROFILER_CALL(
        rocprofiler_configure_dispatch_thread_trace_service(
            ctx,
            agent_id,
            params.data(),
            params.size(),
            dispatch_callback,
            shader_data_callback,
            nullptr),
        "thread trace service configure");
}
```

## ROCprof Trace Decoder API

### Trace Decoder Setup

```c
rocprofiler_thread_trace_decoder_id_t decoder{};

ROCPROFILER_CALL(
    rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib"),
    "thread trace decoder creation");
```

### Code Object Tracking

Register code objects during load events:

```c
void tool_codeobj_tracing_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* /* user_data */,
    void* /* userdata */) {

    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
       record.operation != ROCPROFILER_CODE_OBJECT_LOAD)
        return;

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    auto* data = static_cast<
        rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
        return;

    auto* memorybase = reinterpret_cast<const void*>(data->memory_base);

    ROCPROFILER_CALL(
        rocprofiler_thread_trace_decoder_codeobj_load(
            decoder,
            data->code_object_id,
            data->load_delta,
            data->load_size,
            memorybase,
            data->memory_size),
        "code object loading to decoder");
}
```

## Processing Thread Trace Data

**Important:** The provided samples process data immediately within callbacks for simplicity. Production systems should save data and process after application completion, as trace generation rates (GB/s) exceed processing rates (MB/s).

### Shader Data Callback

```c
void shader_data_callback(rocprofiler_agent_id_t agent,
                         int64_t shader_engine_id,
                         void* data,
                         size_t data_size,
                         rocprofiler_user_data_t userdata) {
    auto status = rocprofiler_trace_decode(decoder_handle,
                                           trace_decoder_callback,
                                           data,
                                           data_size,
                                           userdata);
}
```

### Decoder Callback

```c
void trace_decoder_callback(
    rocprofiler_thread_trace_decoder_record_type_t record_type,
    void* trace_events,
    uint64_t trace_size,
    void* userdata) {

    switch(record_type) {
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE: {
            auto* waves = static_cast<
                rocprofiler_thread_trace_decoder_wave_t*>(trace_events);
            for(uint64_t i = 0; i < trace_size; ++i) {
                // Process wave data
            }
            break;
        }
    }
}
```

### Trace Decoder Info Events

The decoder provides quality information through `ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO` events:

**DATA_LOST Event**
Indicates trace data was dropped due to hardware bandwidth or buffer overflow.

*Causes:* Buffer too small; memory bandwidth exceeded

*Actions:* Increase buffer sizes; reduce SEs/SIMD lanes traced; disable or adjust performance counters

**STITCH_INCOMPLETE Event**
Indicates Program Counter addresses could not be found for some instructions (pc field set to zero).

*Causes:* Trace started mid-kernel; missing code object registration; runtime kernels; prior DATA_LOST events; decoder bugs

## Reference Headers

- [trace_decoder.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder.h)
- [trace_decoder_types.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder_types.h)
- [core.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/core.h)
- [dispatch.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/dispatch.h)
- [agent.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/agent.h)
