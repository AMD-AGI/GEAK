Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "kernel", "cpp"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/kernel_language_cpp_support.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Kernel Language C++ Support

## Supported Kernel Language C++ Features

### General C++ Features

**Exception Handling**
Device code doesn't support exceptions due to hardware constraints. Error handling must use return codes instead.

**Assertions**
The `assert` function works in device code for debugging. When an input expression equals zero, execution stops. HIP also provides `abort()` for terminating applications during critical failures, implemented via `__builtin_trap()`.

**printf Support**
Standard printf functionality is available in device code:

```cpp
#include <hip/hip_runtime.h>

__global__ void run_printf() { printf("Hello World\n"); }

int main() {
  run_printf<<<dim3(1), dim3(1), 0, 0>>>();
}
```

**Device-Side Dynamic Memory Allocation**
Device code supports `new`, `delete`, `malloc`, and `free` for dynamic global memory management.

**Classes**
Classes function on both host and device with some restrictions. Member functions with appropriate qualifiers execute the corresponding overload. Virtual functions are supported but calling them across host/device boundaries creates undefined behavior. Memory space qualifiers cannot be applied to member variables.

### C++11 Support

- **constexpr**: Full device support; implicitly marks functions as `__host__ __device__`
- **Lambdas**: Implicitly marked `__host__ __device__`; restricted variable capture
- **std::function**: Not supported

### C++14 Support
All C++14 language features are supported.

### C++17 Support
All C++17 language features are supported.

### C++20 Support
Most C++20 features work, except coroutines aren't available in device code.

## Compiler Features

**Pragma Unroll**
```cpp
#pragma unroll 16  /* unroll loop by 16 */
for (int i=0; i<16; i++) ...

#pragma unroll 1   /* never unroll */
for (int i=0; i<16; i++) ...

#pragma unroll     /* completely unroll */
for (int i=0; i<16; i++) ...
```

**In-Line Assembly**
GCN ISA inline assembly is supported in device code with careful usage recommended.

**Kernel Compilation**
Binary code objects (`.co` files) can be generated:
```bash
hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]
```

**Architecture-Specific Code**
The `amdclang++` compiler defines `__gfx*__` macros based on target GPU architecture for conditional compilation.
