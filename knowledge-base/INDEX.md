Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# AMD AI DevTool Knowledge Base Index

*Comprehensive navigation for 94 knowledge files covering the complete AMD GPU AI stack*

---

## 💻 Verified Code Examples

**29 Production-Ready Examples** targeting MI300X and H100/H200:

### Intermediate Examples (13)

- **vLLM Basic Serving on AMD ROCm** ([basic_serving.py](../examples/amd/vllm/basic_serving.py)) - Complete example showing how to serve an LLM using vLLM on AMD GPUs with sampling parameters
- **PyTorch BF16 Training on MI300X** ([bf16_training_mi300x.py](../examples/amd/pytorch/bf16_training_mi300x.py)) - Automatic mixed precision training with BF16 on MI300X, leveraging 192GB HBM3 and rocBLAS acceleration
- **Nsight Systems Multi-GPU Profiling** ([nsight_multigpu_profile.py](../examples/nvidia/profiling/nsight_multigpu_profile.py)) - Multi-GPU performance profiling with Nsight Systems: NVLink analysis, NCCL communication, and kernel optimization
- **Optimized Matrix Multiplication with LDS on AMD GPUs** ([optimized_matmul_lds.cpp](../examples/amd/hip/optimized_matmul_lds.cpp)) - Matrix multiplication with LDS tiling, bank conflict avoidance, and memory coalescing for CDNA architectures. Achieves 800+ GFLOPS on MI300X.
- **Optimized Parallel Reduction on AMD GPUs** ([parallel_reduction.cpp](../examples/amd/hip/parallel_reduction.cpp)) - Multiple parallel reduction strategies using wavefront primitives (__shfl_down), sequential addressing, and grid-stride loops for 64-wide wavefronts.
- **Optimized 2D Convolution on AMD GPUs** ([convolution_2d.cpp](../examples/amd/hip/convolution_2d.cpp)) - 2D convolution with constant memory for filters, LDS tiling with halo loading, and optimized boundary handling. 500-800 GB/s effective bandwidth.
- **Advanced Vector Addition with Wave-Level Optimizations** ([advanced_vector_add.cpp](../examples/amd/hip/advanced_vector_add.cpp)) - Vector addition demonstrating 64-thread wave indexing, float4 vectorization, coalesced memory access, and inline assembly for instruction control.
- **Optimized Matrix Multiplication with Shared Memory on Nvidia GPUs** ([optimized_matmul_shared.cu](../examples/nvidia/cuda/optimized_matmul_shared.cu)) - Matrix multiplication with shared memory tiling, bank conflict avoidance, and memory coalescing for 32-wide warps. Achieves 1000+ GFLOPS on H100.
- **Optimized Parallel Reduction on Nvidia GPUs** ([parallel_reduction.cu](../examples/nvidia/cuda/parallel_reduction.cu)) - Multiple parallel reduction strategies using warp primitives (__shfl_down_sync), sequential addressing, and grid-stride loops for 32-wide warps.
- **Optimized 2D Convolution on Nvidia GPUs** ([convolution_2d.cu](../examples/nvidia/cuda/convolution_2d.cu)) - 2D convolution with constant memory for filters, shared memory tiling with halo loading, and optimized boundary handling. 600-1000 GB/s effective bandwidth.
- **Vectorized Type Conversions (BF16/FP16/FP8)** ([type_conversions.cu](../examples/nvidia/cuda/type_conversions.cu)) - Vectorized type conversion operations: BF16 ↔ Float, FP16 ↔ Float, FP8 support for efficient mixed-precision computation.
- **Vector Broadcast and Normalization Operations** ([vector_broadcast_ops.cu](../examples/nvidia/cuda/vector_broadcast_ops.cu)) - Row-wise and column-wise broadcast operations, layer normalization, RMS normalization, and fused operations for transformer models.
- **Optimized Matrix Transpose with Shared Memory** ([transpose_optimized.cu](../examples/nvidia/cuda/transpose_optimized.cu)) - Efficient matrix transpose using shared memory tiling with bank conflict avoidance through padding and in-place transpose optimization.

### Advanced Examples (16)

- **Wavefront-Level Primitives on AMD GPUs** ([wavefront_primitives.cpp](../examples/amd/hip/wavefront_primitives.cpp)) - Comprehensive wavefront primitives demonstration: shuffle operations, voting (__ballot, __any, __all), reduction, scan, and broadcast for 64-wide wavefronts.
- **Matrix Multiplication using MFMA Instructions** ([mfma_gemm_bf16.cpp](../examples/amd/hip/mfma_gemm_bf16.cpp)) - Low-level MFMA instruction usage (v_mfma_f32_16x16x32_bf16) with explicit AGPR management and register tile concepts for matrix multiplication.
- **Wave-Level Reduction and Softmax** ([wave_reduction_softmax.cpp](../examples/amd/hip/wave_reduction_softmax.cpp)) - Numerically stable softmax implementation using wave-level reductions (__shfl_xor) for max and sum operations across 64 threads.
- **Direct Buffer-to-LDS Transfer Optimization** ([buffer_to_lds_direct.cpp](../examples/amd/hip/buffer_to_lds_direct.cpp)) - Key HipKittens optimization: llvm_amdgcn_raw_buffer_load_lds intrinsic for direct global-to-LDS transfer, bypassing VGPRs with readfirstlane hoisting.
- **Warp-Level Primitives on Nvidia GPUs** ([warp_primitives.cu](../examples/nvidia/cuda/warp_primitives.cu)) - Comprehensive warp primitives demonstration: shuffle operations, voting (__ballot_sync, __any_sync, __all_sync), reduction, scan, and broadcast for 32-wide warps.
- **Tensor Core Matrix Multiplication with MMA** ([tensor_cores_mma.cu](../examples/nvidia/cuda/tensor_cores_mma.cu)) - Low-level warp-level MMA (mma.sync) instructions for BF16 tensor core operations with register fragment management and performance comparison.
- **Shared Memory Bank Conflict Avoidance with Swizzling** ([shared_memory_swizzling.cu](../examples/nvidia/cuda/shared_memory_swizzling.cu)) - XOR-based address swizzling techniques to eliminate bank conflicts in shared memory, with padding tricks and performance measurements.
- **Asynchronous Copy and Software Pipelining** ([async_copy_pipeline.cu](../examples/nvidia/cuda/async_copy_pipeline.cu)) - cp.async PTX instructions for asynchronous global-to-shared memory copy with multi-stage software pipelining to hide memory latency.
- **Warp Shuffle Operations for Fast Reductions** ([warp_shuffle_ops.cu](../examples/nvidia/cuda/warp_shuffle_ops.cu)) - Complete warp shuffle operation patterns: reductions, butterfly exchanges, broadcast, rotate, and shared-memory-free collective operations.
- **Flash Attention 2 Complete Implementation** ([flash_attention_v2.cu](../examples/nvidia/cuda/flash_attention_v2.cu)) - Complete Flash Attention 2 kernel with online softmax algorithm, tile-based computation, and combined optimization techniques for memory-efficient attention.
- **MFMA Matrix Multiplication on MI300X** ([mfma_matmul.cpp](../examples/amd/hip/mfma_matmul.cpp)) - Optimized matrix multiplication using MFMA intrinsics with BF16 precision, targeting 1307 TFLOPS on MI300X
- **FSDP Multi-GPU Training on MI300X Cluster** ([fsdp_multi_mi300x.py](../examples/amd/distributed/fsdp_multi_mi300x.py)) - Fully Sharded Data Parallel training for 70B+ models on 8x MI300X with RCCL, gradient checkpointing, and CPU offloading
- **rocprof Performance Analysis for MI300X** ([rocprof_analysis.py](../examples/amd/profiling/rocprof_analysis.py)) - Comprehensive performance profiling tool with MFMA utilization, memory bandwidth analysis, and automated optimization suggestions
- **Tensor Core FP8 GEMM on H100** ([tensorcore_fp8_gemm.cu](../examples/nvidia/cuda/tensorcore_fp8_gemm.cu)) - 4th generation Tensor Core matrix multiplication with FP8 precision, achieving 3,958 TFLOPS on H100
- **Transformer Engine FP8 Training on H100** ([transformer_engine_fp8.py](../examples/nvidia/pytorch/transformer_engine_fp8.py)) - Automatic FP8/FP16 precision training with Transformer Engine, achieving 2x speedup for LLM training on H100/H200
- **DeepSpeed ZeRO-3 Training on H100 with NVLink** ([deepspeed_zero3_h100.py](../examples/nvidia/distributed/deepspeed_zero3_h100.py)) - DeepSpeed ZeRO-3 for 175B+ models on 8x H100 with NVLink 4.0, FP8 training, and CPU/NVMe offloading

*Access via: `~/.amd-ai/examples/` after running `amd-ai-devtool update`*

---

## 🚀 Quick Start Guides

Get up and running fast with these practical guides:

- [Fine-tune an LLM in 15 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

## 📚 By Use Case

### Serving LLMs for Inference

- [Docker Deployment for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/deployment/docker-deployment.md)
- [Production LLM Serving on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/deployment/production-serving.md)
- [Attention Mechanism Optimization for AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/optimization/attention-optimization.md)
- [Serving Optimization for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/optimization/serving-optimization.md)
- [SGLang on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/serving-engines/sglang-serving.md)

### Fine-tuning and Training

- [FSDP Training on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/distributed/fsdp-training.md)
- [Full Model Fine-tuning](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/full-finetuning.md)
- [LoRA Fine-tuning on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/lora-finetuning.md)
- [QLoRA: Quantized LoRA Fine-tuning](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/qlora-finetuning.md)
- [Memory Optimization for Training](amd-knowledge-base/layer-5-llm/03-training/optimization/memory-optimization.md)

## 🎯 By Model Architecture


### LLAMA
- [LLaMA Model Optimization on AMD GPUs](amd-knowledge-base/layer-5-llm/04-models/llama/llama-optimization.md)

### MISTRAL
- [Mistral Model Optimization](amd-knowledge-base/layer-5-llm/04-models/mistral/mistral-optimization.md)

## 📊 By Technology Stack Layer


### Layer 1: Hardware Architecture & Specifications

**GPU Architectures:**
- [CDNA Architecture and MI Series Guide](amd-knowledge-base/layer-1-hardware/amd-gpu-arch/cdna-architecture.md) - *Comprehensive guide to AMD CDNA architecture and Instinct MI series accelerators for HPC and AI ...
- [Blackwell Architecture Guide](nvidia-knowledge-base/layer-1-hardware/nvidia-gpu-arch/blackwell-architecture.md) - *Next-generation Nvidia datacenter GPU architecture for AI workloads*
- [Nvidia GPU Architecture Comparison](nvidia-knowledge-base/layer-1-hardware/nvidia-gpu-arch/gpu-comparison.md) - *Comparison of latest Nvidia datacenter GPU architectures: Blackwell and Hopper*
- [Hopper Architecture Guide](nvidia-knowledge-base/layer-1-hardware/nvidia-gpu-arch/hopper-architecture.md) - *Comprehensive guide to Nvidia Hopper architecture and H-series datacenter GPUs for AI workloads*

### Layer 2: Compute Stack & Programming Models

**ROCm Ecosystem:**
- [AMD System Management Interface (AMD SMI)](amd-knowledge-base/layer-2-compute-stack/rocm-systems/amd-smi-usage.md) - The AMD System Management Interface (AMD SMI) is a unified library and toolset for managing and m...
- [ROCm Systems Usage Guide](amd-knowledge-base/layer-2-compute-stack/rocm-systems/rocm-systems-usage.md) - The ROCm Systems super-repo consolidates multiple ROCm systems projects into a single repository,...
- [ROCm Docker Images - Complete Reference (ROCm 7.x)](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-docker-images.md) - Official AMD ROCm Docker images for containerized GPU development, ML/AI workloads, and productio...
- [ROCm Installation Guide](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-installation.md) - Comprehensive guide for installing ROCm (Radeon Open Compute), AMD's open-source compute stack fo...
- [ROCm Profiling and Performance Analysis](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-profiling.md) - Comprehensive guide to profiling AMD GPU applications using ROCm tools.
- [ROCm GitHub Repositories - Complete Reference](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-repositories.md) - Comprehensive guide to all official AMD ROCm repositories on GitHub, organized by the 5-layer AMD...
- [ROCm Compute Stack Overview](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-stack-overview.md) - ROCm (Radeon Open Compute) is AMD's complete **open-source** software stack for GPU computing, de...

**HIP Programming:**
- [CUDA to HIP Porting Guide](amd-knowledge-base/layer-2-compute-stack/hip/cuda-to-hip-porting.md) - Comprehensive guide to migrating CUDA code to run on AMD GPUs with HIP and ROCm.
- [HIP Debugging Guide](amd-knowledge-base/layer-2-compute-stack/hip/hip-debugging.md) - Comprehensive guide to debugging HIP applications on AMD GPUs.
- [HIP GPU Programming Fundamentals](amd-knowledge-base/layer-2-compute-stack/hip/hip-gpu-programming-fundamentals.md) - *Foundation concepts for GPU programming with HIP on AMD GPUs*
- [HIP Memory Management](amd-knowledge-base/layer-2-compute-stack/hip/hip-memory-management.md) - *Comprehensive guide to memory types, patterns, and optimization for AMD GPUs*
- [HIP Thread Synchronization](amd-knowledge-base/layer-2-compute-stack/hip/hip-thread-synchronization.md) - *Mastering thread coordination, wavefront primitives, and atomic operations on AMD GPUs*

**TheRock Packaging:**
- [TheRock: The HIP Environment and ROCm Kit](amd-knowledge-base/layer-2-compute-stack/therock/therock-overview.md) - TheRock is a lightweight open source build system for HIP and ROCm that provides a unified way to...

### Layer 3: Libraries & Computational Primitives

**Linear Algebra & Mathematics:**
- [ROCm Libraries Usage Guide](amd-knowledge-base/layer-3-libraries/rocm-libraries/rocm-libraries-usage.md) - The ROCm Libraries monorepo is AMD's unified repository that consolidates all ROCm math and ML li...
- [hipBLAS Usage Guide](amd-knowledge-base/layer-3-libraries/blas/hipblas-usage.md) - hipBLAS provides a portable BLAS interface that works on both AMD and NVIDIA GPUs.
- [rocBLAS Usage Guide](amd-knowledge-base/layer-3-libraries/blas/rocblas-usage.md) - rocBLAS is AMD's optimized BLAS (Basic Linear Algebra Subprograms) library for ROCm.
- [rocFFT Usage Guide](amd-knowledge-base/layer-3-libraries/fft/rocfft-usage.md) - rocFFT is AMD's Fast Fourier Transform library for ROCm, providing highly optimized FFT implement...
- [rocSOLVER Usage Guide](amd-knowledge-base/layer-3-libraries/solver/rocsolver-usage.md) - rocSOLVER provides LAPACK functionality for ROCm, enabling linear system solving and matrix facto...
- [rocSPARSE Usage Guide](amd-knowledge-base/layer-3-libraries/sparse/rocsparse-usage.md) - rocSPARSE provides sparse linear algebra operations for ROCm.
- [cuBLAS Usage Guide](nvidia-knowledge-base/layer-3-libraries/blas/cublas-usage.md) - *GPU-accelerated Basic Linear Algebra Subprograms (BLAS) library*

**Communication & Multi-GPU:**
- [RCCL Usage Guide](amd-knowledge-base/layer-3-libraries/communications/rccl-usage.md) - RCCL (ROCm Communication Collectives Library) enables multi-GPU and multi-node communication for ...
- [NCCL Usage Guide](nvidia-knowledge-base/layer-3-libraries/communications/nccl-usage.md) - *Nvidia Collective Communications Library for multi-GPU and multi-node training*

**ML Primitives & Compilers:**
- [Triton on AMD GPUs](amd-knowledge-base/layer-3-libraries/compilers/triton-on-rocm.md) - Triton is a Python-based language and compiler for writing efficient GPU kernels with ROCm support.
- [MIOpen Usage Guide](amd-knowledge-base/layer-3-libraries/ml-primitives/miopen-usage.md) - MIOpen is AMD's library for high-performance deep learning primitives, providing optimized implem...

### Layer 4: ML Frameworks & Runtimes

**Major ML Frameworks:**
- [JAX with ROCm - Getting Started](amd-knowledge-base/layer-4-frameworks/jax/jax-rocm-basics.md) - ## Overview
- [PyTorch with ROCm](amd-knowledge-base/layer-4-frameworks/pytorch/pytorch-rocm-basics.md) - Complete guide to using PyTorch with AMD GPUs via ROCm.
- [TensorFlow with ROCm - Getting Started](amd-knowledge-base/layer-4-frameworks/tensorflow/tensorflow-rocm-basics.md) - ## Overview
- [JAX with CUDA](nvidia-knowledge-base/layer-4-frameworks/jax/jax-cuda-basics.md) - *High-performance numerical computing with JAX on Nvidia GPUs*
- [PyTorch with CUDA](nvidia-knowledge-base/layer-4-frameworks/pytorch/pytorch-cuda-basics.md) - *Complete guide to using PyTorch with Nvidia GPUs*
- [TensorFlow with CUDA](nvidia-knowledge-base/layer-4-frameworks/tensorflow/tensorflow-cuda-basics.md) - *Guide to using TensorFlow with Nvidia GPUs*

### Layer 5: Large Language Models & Advanced AI

**Quickstart Guides:**
- [Fine-tune an LLM in 15 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-inference.md) - 5min
- [Fine-tune an LLM in 15 Minutes](nvidia-knowledge-base/layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](nvidia-knowledge-base/layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

**Foundations & Setup:**
- [Docker and Containers for ROCm](amd-knowledge-base/layer-5-llm/01-foundations/docker-basics.md) - Complete guide to containerizing AMD GPU applications with Docker and ROCm.
- [Hugging Face Transformers on AMD GPUs](amd-knowledge-base/layer-5-llm/01-foundations/transformers-rocm.md) - Complete guide to using the Transformers library with AMD ROCm for model inference and training.
- [Docker and Containers for CUDA](nvidia-knowledge-base/layer-5-llm/01-foundations/docker-basics.md) - *Complete guide to containerizing LLM applications with Docker and CUDA*
- [Hugging Face Transformers on Nvidia GPUs](nvidia-knowledge-base/layer-5-llm/01-foundations/transformers-cuda.md) - *Complete guide to using Transformers library with CUDA*

**Inference & Serving:**
- [Docker Deployment for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/deployment/docker-deployment.md) - Production-ready Docker deployment patterns for serving LLMs on AMD GPUs.
- [Production LLM Serving on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/deployment/production-serving.md) - Comprehensive guide to deploying and operating LLM inference at production scale.
- [Attention Mechanism Optimization for AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/optimization/attention-optimization.md) - Advanced techniques for optimizing attention mechanisms in LLM inference on AMD hardware.
- [Serving Optimization for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/optimization/serving-optimization.md) - Comprehensive guide to optimizing LLM serving performance on AMD GPUs.
- [SGLang on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/serving-engines/sglang-serving.md) - SGLang (Structured Generation Language) is a fast serving framework for large language models and...
- [vLLM Deployment on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/serving-engines/vllm-serving.md) - vLLM is a fast and easy-to-use library for LLM inference and serving, with excellent AMD GPU supp...
- [TensorRT-LLM Serving](nvidia-knowledge-base/layer-5-llm/02-inference/serving-engines/tensorrt-llm.md) - *Nvidia's optimized LLM inference engine for maximum performance*
- [vLLM Deployment on Nvidia GPUs](nvidia-knowledge-base/layer-5-llm/02-inference/serving-engines/vllm-serving.md) - *Fast and easy-to-use library for LLM inference and serving with excellent Nvidia GPU support*

**Training & Fine-tuning:**
- [FSDP Training on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/distributed/fsdp-training.md) - Fully Sharded Data Parallel (FSDP) enables training large models that don't fit on a single GPU.
- [Full Model Fine-tuning](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/full-finetuning.md) - Complete guide to full parameter fine-tuning of large language models on AMD GPUs.
- [LoRA Fine-tuning on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/lora-finetuning.md) - Complete guide to efficient fine-tuning using Low-Rank Adaptation (LoRA) on AMD hardware.
- [QLoRA: Quantized LoRA Fine-tuning](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/qlora-finetuning.md) - Train large language models with 4-bit quantization and LoRA on AMD GPUs with minimal memory.
- [Memory Optimization for Training](amd-knowledge-base/layer-5-llm/03-training/optimization/memory-optimization.md) - Techniques to optimize memory usage during LLM training on AMD GPUs.
- [Dataset Preparation for LLM Training](amd-knowledge-base/layer-5-llm/03-training/preparation/dataset-preparation.md) - Complete guide to preparing datasets for fine-tuning LLMs on AMD GPUs.
- [Training Environment Setup](amd-knowledge-base/layer-5-llm/03-training/preparation/environment-setup.md) - Complete guide to setting up an optimal training environment for LLMs on AMD GPUs.
- [LoRA Fine-tuning on Nvidia GPUs](nvidia-knowledge-base/layer-5-llm/03-training/fine-tuning/lora-finetuning.md) - *Complete guide to efficient fine-tuning using Low-Rank Adaptation (LoRA)*
- [QLoRA: Quantized LoRA Fine-tuning](nvidia-knowledge-base/layer-5-llm/03-training/fine-tuning/qlora-finetuning.md) - *Train large language models with 4-bit quantization and LoRA on Nvidia GPUs*

**Model Architectures & Optimization:**
- [LLaMA Model Optimization on AMD GPUs](amd-knowledge-base/layer-5-llm/04-models/llama/llama-optimization.md) - Comprehensive guide to optimizing LLaMA and LLaMA-2 models on AMD hardware.
- [Mistral Model Optimization](amd-knowledge-base/layer-5-llm/04-models/mistral/mistral-optimization.md) - Guide to optimizing Mistral and Mixtral models on AMD GPUs.

**Advanced Techniques:**
- [Custom Kernels with Triton](amd-knowledge-base/layer-5-llm/05-advanced/custom-kernels/triton-kernels.md) - Guide to writing custom GPU kernels for AMD using Triton.
- [MongoDB MCP Integration for AI/ML Workflows](amd-knowledge-base/layer-5-llm/05-advanced/data-management/mongodb-mcp-integration.md) - Query MongoDB databases directly from your IDE for experiment tracking, GPU monitoring data, and ...

## 🛠 Best Practices & Optimization

**Performance & Debugging:**
- [Debugging Best Practices](amd-knowledge-base/best-practices/debugging/debugging-guide.md) - ## Systematic Debugging Approach
- [GPU Performance Optimization Best Practices](amd-knowledge-base/best-practices/performance/gpu-optimization.md) - ## General Principles
- [Kernel Optimization for AMD GPUs](amd-knowledge-base/best-practices/performance/kernel-optimization.md) - ## Occupancy Optimization
- [Memory Optimization for AMD GPUs](amd-knowledge-base/best-practices/performance/memory-optimization.md) - ## Memory Hierarchy
- [GPU Performance Optimization Best Practices](nvidia-knowledge-base/best-practices/performance/gpu-optimization.md) - *General principles for optimizing Nvidia GPU performance*
- [Kernel Optimization for Nvidia GPUs](nvidia-knowledge-base/best-practices/performance/kernel-optimization.md) - *Best practices for optimizing CUDA kernels*
- [Memory Optimization for Nvidia GPUs](nvidia-knowledge-base/best-practices/performance/memory-optimization.md) - *Optimize memory usage and bandwidth on Nvidia GPUs*

**Testing & Development:**
- [GitHub Actions for AMD GPU Projects](amd-knowledge-base/best-practices/ci-cd/github-actions.md) - CI/CD pipelines for GPU-accelerated projects using GitHub Actions.
- [GPU Testing Best Practices](amd-knowledge-base/best-practices/testing/gpu-testing.md) - Comprehensive guide to testing GPU-accelerated applications on AMD hardware.

## 📈 By Difficulty Level


### Beginner

- [ROCm Docker Images - Complete Reference (ROCm 7.x)](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-docker-images.md) - 20min
- [ROCm GitHub Repositories - Complete Reference](amd-knowledge-base/layer-2-compute-stack/rocm/rocm-repositories.md) - 15min
- [Fine-tune an LLM in 15 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](amd-knowledge-base/layer-5-llm/00-quickstart/quickstart-inference.md) - 5min
- [CUDA Docker Images - Complete Reference](nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-docker-images.md) - 20min
- [Fine-tune an LLM in 15 Minutes](nvidia-knowledge-base/layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](nvidia-knowledge-base/layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

### Intermediate

- [GitHub Actions for AMD GPU Projects](amd-knowledge-base/best-practices/ci-cd/github-actions.md) - 35min
- [GPU Testing Best Practices](amd-knowledge-base/best-practices/testing/gpu-testing.md) - 40min
- [TensorFlow with ROCm - Getting Started](amd-knowledge-base/layer-4-frameworks/tensorflow/tensorflow-rocm-basics.md) - 45min
- [Docker and Containers for ROCm](amd-knowledge-base/layer-5-llm/01-foundations/docker-basics.md) - 40min
- [Hugging Face Transformers on AMD GPUs](amd-knowledge-base/layer-5-llm/01-foundations/transformers-rocm.md) - 45min
- [Docker Deployment for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/deployment/docker-deployment.md) - 45min
- [SGLang on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/serving-engines/sglang-serving.md) - 30min
- [vLLM Deployment on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/serving-engines/vllm-serving.md) - 30min
- [FSDP Training on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/distributed/fsdp-training.md) - 30min
- [LoRA Fine-tuning on AMD GPUs](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/lora-finetuning.md) - 45min

### Advanced

- [JAX with ROCm - Getting Started](amd-knowledge-base/layer-4-frameworks/jax/jax-rocm-basics.md) - 60min
- [Production LLM Serving on AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/deployment/production-serving.md) - 60min
- [Attention Mechanism Optimization for AMD GPUs](amd-knowledge-base/layer-5-llm/02-inference/optimization/attention-optimization.md) - 50min
- [Serving Optimization for LLM Inference](amd-knowledge-base/layer-5-llm/02-inference/optimization/serving-optimization.md) - 55min
- [Full Model Fine-tuning](amd-knowledge-base/layer-5-llm/03-training/fine-tuning/full-finetuning.md) - 50min
- [Memory Optimization for Training](amd-knowledge-base/layer-5-llm/03-training/optimization/memory-optimization.md) - 45min
- [LLaMA Model Optimization on AMD GPUs](amd-knowledge-base/layer-5-llm/04-models/llama/llama-optimization.md) - 45min
- [TensorRT-LLM Serving](nvidia-knowledge-base/layer-5-llm/02-inference/serving-engines/tensorrt-llm.md) - 60min

## 🔍 Navigation & Search Tips

### Finding Content
- **New to AMD GPUs?** Start with [Quick Start Guides](#-quick-start-guides)
- **Hardware deep dive?** Explore [Layer 1: Hardware Architecture](#layer-1-hardware-architecture--specifications)
- **Library usage?** Check [Layer 3: Libraries & Primitives](#layer-3-libraries--computational-primitives)
- **Framework integration?** See [Layer 4: ML Frameworks](#layer-4-ml-frameworks--runtimes)
- **LLM deployment?** Visit [Layer 5: LLM & AI](#layer-5-large-language-models--advanced-ai)

### Search Commands
```bash
# Search knowledge base
amd-ai-devtool search "rocBLAS matrix multiplication"
amd-ai-devtool search "LLaMA fine-tuning memory"
amd-ai-devtool search "RDNA vs CDNA architecture"

# Browse documentation
amd-ai-devtool docs                # Show layer overview and presets
amd-ai-devtool docs --categories   # Show detailed categories
```

### By Expertise Level
- **👋 Beginner**: Quick starts, installation guides, basic tutorials
- **⚡ Intermediate**: Framework integration, optimization basics, deployment
- **🚀 Advanced**: Multi-GPU, production serving, custom optimizations
- **🔬 Expert**: Custom kernels, architecture deep-dives, research techniques

### Total Knowledge Base
- **94 Comprehensive Guides** covering the complete AMD GPU AI stack
- **5 Technology Layers** from hardware to applications
- **40+ ROCm Repositories** documented and organized by layer
- **10+ Model Architectures** with optimization guides
- **Multiple Frameworks** (PyTorch, TensorFlow, JAX, ONNX)
- **Best Practices** for production deployment and debugging