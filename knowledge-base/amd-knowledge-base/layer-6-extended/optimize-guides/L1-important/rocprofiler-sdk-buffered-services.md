---
tags: ["optimization", "performance", "profiling", "rocprofiler-sdk", "buffering"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/buffered_services.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK Buffered Services

## Overview

The buffered approach in ROCprofiler-SDK employs background threads to deliver callbacks containing batches of records. The framework supports various buffer record categories via `rocprofiler_buffer_category_t` and buffer tracing services through `rocprofiler_buffer_tracing_kind_t`. Buffer flushing—whether implicit or explicit using `rocprofiler_flush_buffer`—triggers tool callbacks with record arrays.

## Creating a Buffer

```c
rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                          size_t                          size,
                          size_t                          watermark,
                          rocprofiler_buffer_policy_t     policy,
                          rocprofiler_buffer_tracing_cb_t callback,
                          void*                           callback_data,
                          rocprofiler_buffer_id_t*        buffer_id);
```

### Buffer Parameters

- **size**: Buffer capacity in bytes, rounded to nearest memory page (typically 4096 bytes on Linux)
- **watermark**: Byte threshold triggering buffer flush callbacks. Safe values range from zero to buffer size
- **policy**: Determines behavior for records exceeding available space. `ROCPROFILER_BUFFER_POLICY_DISCARD` drops oversized records until explicit flush; `ROCPROFILER_BUFFER_POLICY_LOSSLESS` swaps buffers automatically
- **callback**: Invoked upon buffer flush
- **callback_data**: User-provided argument passed to callback
- **buffer_id**: Output handle for successful creation

## Creating Dedicated Callback Threads

By default, ROCprofiler-SDK uses a single background thread for all buffer callbacks. Custom threads are created via:

```c
rocprofiler_status_t
rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id);
```

Assign buffers to threads with:

```c
rocprofiler_status_t
rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                   rocprofiler_callback_thread_t cb_thread_id);
```

## Configuring Buffer Tracing Services

```c
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                             rocprofiler_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*  operations,
                                             size_t                            operations_count,
                                             rocprofiler_buffer_id_t           buffer_id);
```

### Configuration Parameters

- **kind**: High-level service specification (e.g., HIP API, HSA API, kernel dispatches)
- **operations**: API-specific function restrictions. Use `nullptr` and `0` to trace all operations, or provide a C-array for subset tracing
- **buffer_id**: Target buffer for records

Multiple services can share the same buffer. Configuring duplicate services for a context returns an error.

## Buffer Tracing Callback Function

```c
typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t      context,
                                                rocprofiler_buffer_id_t       buffer_id,
                                                rocprofiler_record_header_t** headers,
                                                size_t                        num_headers,
                                                void*                         data,
                                                uint64_t                      drop_count);
```

### Record Header Fields

- **category**: Classifies buffer records (e.g., `ROCPROFILER_BUFFER_CATEGORY_TRACING`, `ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING`, `ROCPROFILER_BUFFER_CATEGORY_COUNTERS`)
- **kind**: Depends on category; for tracing, specifies the tracing type
- **payload**: Cast after determining category and kind

### Callback Implementation Example

```c
void buffer_callback_func(rocprofiler_context_id_t      context,
                         rocprofiler_buffer_id_t       buffer_id,
                         rocprofiler_record_header_t** headers,
                         size_t                        num_headers,
                         void*                         user_data,
                         uint64_t                      drop_count)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(
                header->payload);
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                header->payload);
        }
    }
}
```

## Buffer Tracing Record

Unlike callback tracing records, buffer records lack common fields across all types. However, many contain `kind` and `operation` fields. Query functions include:

- `rocprofiler_query_buffer_tracing_kind_name`: Retrieves tracing kind names
- `rocprofiler_query_buffer_tracing_kind_operation_name`: Retrieves operation names
- `rocprofiler_iterate_buffer_tracing_kinds`: Iterates all tracing kinds
- `rocprofiler_iterate_buffer_tracing_kind_operations`: Iterates operations per kind

Record data types are defined in `rocprofiler-sdk/buffer_tracing.h`.
