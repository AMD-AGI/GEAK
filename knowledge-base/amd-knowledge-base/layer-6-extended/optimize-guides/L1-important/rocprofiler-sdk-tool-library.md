Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler-sdk", "tool-development"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/tool_library.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK Tool Library

## Overview

The tool library leverages APIs from `rocprofiler-sdk` and `rocprofiler-register` to enable profiling and tracing of HIP applications. The document explains how to design tools using these libraries efficiently. The command-line tool `rocprofv3` is built on `librocprofiler-sdk-tool.so.X.Y.Z`, which utilizes these same libraries.

## ROCm Runtimes Design

ROCm runtimes communicate with a helper library called `rocprofiler-register` during initialization. This library detects whether a tool requires ROCprofiler-SDK services by checking for `rocprofiler_configure` in the tool's symbol table or the `ROCP_TOOL_LIBRARIES` environment variable. This approach represents a significant improvement over previous designs that relied on tools setting runtime-specific environment variables like `HSA_TOOLS_LIB` before runtime initialization.

## Tool Library Design

When ROCprofiler-SDK detects `rocprofiler_configure`, it invokes this function with parameters including the SDK version, count of previously invoked tools, and a unique tool identifier. The tool returns a pointer to a `rocprofiler_tool_configure_result_t` struct containing:

- Initialization function (context creation opportunity)
- Finalization function
- Data pointer for initialization and finalization callbacks

The SDK provides a `rocprofiler-sdk/registration.h` header that forward declares `rocprofiler_configure` with appropriate compiler attributes for symbol visibility.

### Basic Implementation Pattern

```c
#include <rocprofiler-sdk/registration.h>

namespace {
  struct ToolData {
    uint32_t version;
    const char* runtime_version;
    uint32_t priority;
    rocprofiler_client_id_t client_id;
  };

  int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data_v);
  void tool_fini(void* tool_data_v);
}

extern "C" {
  rocprofiler_tool_configure_result_t*
  rocprofiler_configure(uint32_t version,
                        const char* runtime_version,
                        uint32_t priority,
                        rocprofiler_client_id_t* client_id) {
    if(priority > 0) return nullptr;

    client_id->name = "ExampleTool";

    static auto data = ToolData{version, runtime_version, priority, client_id};

    static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t),
      &tool_init,
      &tool_fini,
      static_cast<void*>(&data)
    };

    return &cfg;
  }
}
```

**Important:** The SDK prohibits calling any runtime functions (HSA, HIP, etc.) during tool initialization, as this causes deadlocks.

### Initialization Phase

After scanning for all `rocprofiler_configure` symbols, ROCprofiler-SDK invokes the initialization callback. This is the appropriate time to create contexts:

```c
#include <rocprofiler-sdk/rocprofiler.h>

namespace {
  int tool_init(rocprofiler_client_finalize_t fini_func, void* data_v) {
    auto ctx = rocprofiler_context_id_t{0};
    rocprofiler_create_context(&ctx);
    // associate services with context
    return 0;
  }
}
```

Tools should store context handles to manage data collection for associated services.

## Tool Finalization

During initialization, ROCprofiler-SDK provides a `rocprofiler_client_finalize_t` function pointer. Tools can invoke this to explicitly trigger the finalize callback:

```c
namespace {
  int tool_init(rocprofiler_client_finalize_t fini_func, void* data_v) {
    auto explicit_finalize = [](rocprofiler_client_finalize_t finalizer,
                                rocprofiler_client_id_t* client_id) {
      std::this_thread::sleep_for(std::chrono::seconds{10});
      finalizer(client_id);
    };

    rocprofiler_start_context(ctx);
    std::thread{explicit_finalize, fini_func,
                static_cast<ToolData*>(data_v)->client_id}.detach();

    return 0;
  }
}
```

If explicit finalization is not called, ROCprofiler-SDK invokes the callback via an `atexit` handler.
