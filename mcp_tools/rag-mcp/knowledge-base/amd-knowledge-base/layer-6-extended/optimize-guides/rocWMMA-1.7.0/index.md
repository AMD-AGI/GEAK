# rocWMMA 1.7.0 Documentation Index

**Version**: rocWMMA 1.7.0  
**ROCm Release**: 6.4.3  
**Source**: [ROCm/rocWMMA](https://github.com/ROCm/rocWMMA) (tag: rocm-6.4.3)  
**Last Updated**: 2026-01-12

---

## 📚 Documentation Files

| File | Description | Size |
|------|-------------|------|
| [index.md](./index.md) | Main documentation entry | 1.2 KB |
| [what-is-rocwmma.md](./what-is-rocwmma.md) | Overview and introduction | 2.5 KB |
| [license.md](./license.md) | License information | 1.2 KB |
| [install/installation.md](./install/installation.md) | Installation guide | 29 KB |
| [conceptual/programmers-guide.md](./conceptual/programmers-guide.md) | Developer's guide | 16 KB |
| [api-reference/api-reference-guide.md](./api-reference/api-reference-guide.md) | API documentation | 14 KB |

---

## 💻 Code Examples (samples/)

### Simple Examples
- `simple_sgemm.cpp` - FP32 matrix multiplication
- `simple_hgemm.cpp` - FP16 matrix multiplication  
- `simple_dgemm.cpp` - FP64 matrix multiplication
- `simple_sgemv.cpp` - FP32 matrix-vector multiplication
- `simple_dgemv.cpp` - FP64 matrix-vector multiplication
- `simple_dlrm.cpp` - Deep Learning Recommendation Model

### Performance Examples
- `perf_sgemm.cpp` - Optimized FP32 GEMM benchmark
- `perf_hgemm.cpp` - Optimized FP16 GEMM benchmark
- `perf_dgemm.cpp` - Optimized FP64 GEMM benchmark

### Runtime Compilation
- `hipRTC_gemm.cpp` - HIP Runtime Compilation example

### Build Files
- `CMakeLists.txt` - CMake configuration
- `common.hpp` - Shared utilities

---

## 🚀 Quick Start

1. Read [what-is-rocwmma.md](./what-is-rocwmma.md)
2. Follow [install/installation.md](./install/installation.md)
3. Study [conceptual/programmers-guide.md](./conceptual/programmers-guide.md)
4. Try examples in `samples/`
5. Refer to [api-reference/api-reference-guide.md](./api-reference/api-reference-guide.md)

---

*Converted from reStructuredText using Pandoc 3.8.3*
