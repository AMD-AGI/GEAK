---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Nvidia CUDA Knowledge Base Index

*Navigation for 28+ guides focusing on Hopper, Blackwell, and CUDA 13.x*

---

## 🚀 Quick Start Guides

Get up and running fast with these practical guides:

- [Fine-tune an LLM in 15 Minutes](layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

## 📚 By Use Case

### Serving LLMs for Inference

- [Docker Deployment for LLM Inference](layer-5-llm/02-inference/deployment/docker-deployment.md)
- [Production LLM Serving on Nvidia GPUs](layer-5-llm/02-inference/deployment/production-serving.md)
- [Attention Mechanism Optimization for Nvidia GPUs](layer-5-llm/02-inference/optimization/attention-optimization.md)
- [Serving Optimization for LLM Inference](layer-5-llm/02-inference/optimization/serving-optimization.md)
- [vLLM on Nvidia GPUs](layer-5-llm/02-inference/serving-engines/vllm-serving.md)
- [TensorRT-LLM Serving](layer-5-llm/02-inference/serving-engines/tensorrt-llm.md)
- [Triton Inference Server](layer-5-llm/02-inference/serving-engines/triton-inference-server.md)

### Fine-tuning and Training

- [FSDP Training on Nvidia GPUs](layer-5-llm/03-training/distributed/fsdp-training.md)
- [Full Model Fine-tuning](layer-5-llm/03-training/fine-tuning/full-finetuning.md)
- [LoRA Fine-tuning on Nvidia GPUs](layer-5-llm/03-training/fine-tuning/lora-finetuning.md)
- [QLoRA: Quantized LoRA Fine-tuning](layer-5-llm/03-training/fine-tuning/qlora-finetuning.md)
- [Memory Optimization for Training](layer-5-llm/03-training/optimization/memory-optimization.md)

## 🎯 By Model Architecture

### LLAMA
- [LLaMA Model Optimization on Nvidia GPUs](layer-5-llm/04-models/llama/llama-optimization.md)

### MISTRAL
- [Mistral Model Optimization](layer-5-llm/04-models/mistral/mistral-optimization.md)

## 📊 By Technology Stack Layer

### Layer 1: Hardware Architecture & Specifications

**Latest GPU Architectures:**
- [Blackwell Architecture Guide](layer-1-hardware/nvidia-gpu-arch/blackwell-architecture.md) - Next-gen with FP4, 5th gen Tensor Cores (B100/B200)
- [Hopper Architecture Guide](layer-1-hardware/nvidia-gpu-arch/hopper-architecture.md) - Current gen with FP8, Transformer Engine (H100/H200)
- [GPU Comparison](layer-1-hardware/nvidia-gpu-arch/gpu-comparison.md) - Blackwell vs Hopper comparison

### Layer 2: Compute Stack & Programming Models

**CUDA 13.x Ecosystem:**
- [CUDA 13.0 Installation Guide](layer-2-compute-stack/cuda/cuda-installation.md) - Installing CUDA 13.x Toolkit and drivers
- [CUDA Programming Basics](layer-2-compute-stack/cuda/cuda-basics.md) - Core CUDA 13.x programming concepts
- [CUDA Advanced Optimization](layer-2-compute-stack/cuda/cuda-advanced-optimization.md) - Native CUDA optimization techniques: tensor cores, shared memory swizzling, async copy, warp shuffles, Flash Attention 2 for peak performance.
- [CUDA Native Examples Guide](layer-2-compute-stack/cuda/cuda-native-examples-guide.md) - 8 runnable examples demonstrating native CUDA optimization: tensor cores MMA, shared memory swizzling, async copy pipeline, warp shuffles, Flash Attention V2, type conversions, vector ops, and transpose (see `examples/nvidia/cuda/`).
- [CUDA Profiling and Performance Analysis](layer-2-compute-stack/cuda/cuda-profiling.md) - Nsight tools for Hopper/Blackwell
- [CUDA Docker Images Reference](layer-2-compute-stack/cuda/cuda-docker-images.md) - CUDA 13.x container images

**Nsight Tools:**
- [Nsight Systems Guide](layer-2-compute-stack/nsight/nsight-systems.md) - System-wide performance analysis
- [Nsight Compute Guide](layer-2-compute-stack/nsight/nsight-compute.md) - Kernel-level profiling and optimization

### Layer 3: Libraries & Computational Primitives

**Linear Algebra & Mathematics:**
- [cuBLAS Usage Guide](layer-3-libraries/blas/cublas-usage.md) - GPU-accelerated linear algebra operations
- [cuDNN Usage Guide](layer-3-libraries/dnn/cudnn-usage.md) - Deep learning primitives library

**Communication & Multi-GPU:**
- [NCCL Usage Guide](layer-3-libraries/communications/nccl-usage.md) - Multi-GPU and multi-node collective operations

**ML Primitives & Compilers:**
- [Triton on CUDA GPUs](layer-3-libraries/compilers/triton-on-cuda.md) - Python-based GPU kernel compiler

### Layer 4: ML Frameworks & Runtimes

**Major ML Frameworks:**
- [PyTorch with CUDA](layer-4-frameworks/pytorch/pytorch-cuda-basics.md) - Complete guide to using PyTorch with Nvidia GPUs
- [TensorFlow with CUDA](layer-4-frameworks/tensorflow/tensorflow-cuda-basics.md) - TensorFlow on Nvidia GPUs
- [JAX with CUDA](layer-4-frameworks/jax/jax-cuda-basics.md) - JAX for high-performance numerical computing

### Layer 5: Large Language Models & Advanced AI

**Quickstart Guides:**
- [Fine-tune an LLM in 15 Minutes](layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

**Foundations & Setup:**
- [Docker and Containers for CUDA](layer-5-llm/01-foundations/docker-basics.md) - Containerizing CUDA applications
- [Hugging Face Transformers on Nvidia GPUs](layer-5-llm/01-foundations/transformers-cuda.md) - Transformers library with CUDA

**Inference & Serving:**
- [Docker Deployment for LLM Inference](layer-5-llm/02-inference/deployment/docker-deployment.md) - Production Docker deployment patterns
- [Production LLM Serving on Nvidia GPUs](layer-5-llm/02-inference/deployment/production-serving.md) - Production-scale LLM serving
- [Attention Mechanism Optimization](layer-5-llm/02-inference/optimization/attention-optimization.md) - Optimizing attention for Nvidia hardware
- [Serving Optimization for LLM Inference](layer-5-llm/02-inference/optimization/serving-optimization.md) - Comprehensive serving optimization
- [vLLM Deployment on Nvidia GPUs](layer-5-llm/02-inference/serving-engines/vllm-serving.md) - Fast LLM inference with vLLM
- [TensorRT-LLM Serving](layer-5-llm/02-inference/serving-engines/tensorrt-llm.md) - Nvidia's optimized LLM inference engine
- [Triton Inference Server](layer-5-llm/02-inference/serving-engines/triton-inference-server.md) - Scalable inference serving platform

**Training & Fine-tuning:**
- [FSDP Training on Nvidia GPUs](layer-5-llm/03-training/distributed/fsdp-training.md) - Fully Sharded Data Parallel training
- [Full Model Fine-tuning](layer-5-llm/03-training/fine-tuning/full-finetuning.md) - Complete parameter fine-tuning
- [LoRA Fine-tuning on Nvidia GPUs](layer-5-llm/03-training/fine-tuning/lora-finetuning.md) - Efficient LoRA fine-tuning
- [QLoRA: Quantized LoRA Fine-tuning](layer-5-llm/03-training/fine-tuning/qlora-finetuning.md) - 4-bit quantized LoRA
- [Memory Optimization for Training](layer-5-llm/03-training/optimization/memory-optimization.md) - Training memory optimization
- [Dataset Preparation for LLM Training](layer-5-llm/03-training/preparation/dataset-preparation.md) - Dataset preparation guide
- [Training Environment Setup](layer-5-llm/03-training/preparation/environment-setup.md) - Setting up training environment

**Model Architectures & Optimization:**
- [LLaMA Model Optimization on Nvidia GPUs](layer-5-llm/04-models/llama/llama-optimization.md) - LLaMA optimization guide
- [Mistral Model Optimization](layer-5-llm/04-models/mistral/mistral-optimization.md) - Mistral/Mixtral optimization

**Advanced Techniques:**
- [Custom CUDA Kernels](layer-5-llm/05-advanced/custom-kernels/cuda-kernels.md) - Writing custom CUDA kernels

## 🛠 Best Practices & Optimization

**Performance & Debugging:**
- [GPU Performance Optimization Best Practices](best-practices/performance/gpu-optimization.md) - General GPU optimization
- [Kernel Optimization for Nvidia GPUs](best-practices/performance/kernel-optimization.md) - CUDA kernel optimization
- [Memory Optimization for Nvidia GPUs](best-practices/performance/memory-optimization.md) - Memory hierarchy optimization
- [Debugging Best Practices](best-practices/debugging/debugging-guide.md) - Systematic debugging approach

**Testing & Development:**
- [GPU Testing Best Practices](best-practices/testing/gpu-testing.md) - Testing GPU applications

## 📈 By Difficulty Level

### Beginner

- [CUDA Docker Images Reference](layer-2-compute-stack/cuda/cuda-docker-images.md) - 20min
- [Fine-tune an LLM in 15 Minutes](layer-5-llm/00-quickstart/quickstart-finetuning.md) - 15min
- [LLM Inference in 5 Minutes](layer-5-llm/00-quickstart/quickstart-inference.md) - 5min

### Intermediate

- [TensorFlow with CUDA](layer-4-frameworks/tensorflow/tensorflow-cuda-basics.md) - 45min
- [Docker and Containers for CUDA](layer-5-llm/01-foundations/docker-basics.md) - 40min
- [Hugging Face Transformers on Nvidia GPUs](layer-5-llm/01-foundations/transformers-cuda.md) - 45min
- [Docker Deployment for LLM Inference](layer-5-llm/02-inference/deployment/docker-deployment.md) - 45min
- [vLLM Deployment on Nvidia GPUs](layer-5-llm/02-inference/serving-engines/vllm-serving.md) - 30min
- [FSDP Training on Nvidia GPUs](layer-5-llm/03-training/distributed/fsdp-training.md) - 30min
- [LoRA Fine-tuning on Nvidia GPUs](layer-5-llm/03-training/fine-tuning/lora-finetuning.md) - 45min

### Advanced

- [JAX with CUDA](layer-4-frameworks/jax/jax-cuda-basics.md) - 60min
- [Production LLM Serving on Nvidia GPUs](layer-5-llm/02-inference/deployment/production-serving.md) - 60min
- [Attention Mechanism Optimization](layer-5-llm/02-inference/optimization/attention-optimization.md) - 50min
- [Serving Optimization for LLM Inference](layer-5-llm/02-inference/optimization/serving-optimization.md) - 55min
- [TensorRT-LLM Serving](layer-5-llm/02-inference/serving-engines/tensorrt-llm.md) - 60min
- [Full Model Fine-tuning](layer-5-llm/03-training/fine-tuning/full-finetuning.md) - 50min
- [Memory Optimization for Training](layer-5-llm/03-training/optimization/memory-optimization.md) - 45min
- [LLaMA Model Optimization on Nvidia GPUs](layer-5-llm/04-models/llama/llama-optimization.md) - 45min

## 🔍 Navigation & Search Tips

### Finding Content
- **New to Nvidia GPUs?** Start with [Quick Start Guides](#-quick-start-guides)
- **Hardware deep dive?** Explore [Layer 1: Hardware Architecture](#layer-1-hardware-architecture--specifications)
- **Library usage?** Check [Layer 3: Libraries & Primitives](#layer-3-libraries--computational-primitives)
- **Framework integration?** See [Layer 4: ML Frameworks](#layer-4-ml-frameworks--runtimes)
- **LLM deployment?** Visit [Layer 5: LLM & AI](#layer-5-large-language-models--advanced-ai)

### Search Commands
```bash
# Search knowledge base
amd-ai-devtool search "cuBLAS matrix multiplication"
amd-ai-devtool search "LLaMA fine-tuning memory"
amd-ai-devtool search "Hopper vs Ampere architecture"

# Browse documentation
amd-ai-devtool docs --vendor nvidia
```

### By Expertise Level
- **👋 Beginner**: Quick starts, installation guides, basic tutorials
- **⚡ Intermediate**: Framework integration, optimization basics, deployment
- **🚀 Advanced**: Multi-GPU, production serving, custom optimizations
- **🔬 Expert**: Custom kernels, architecture deep-dives, research techniques

### Total Knowledge Base
- **28+ Focused Guides** for latest Nvidia GPU AI stack
- **5 Technology Layers** from hardware to applications
- **Latest GPU Architectures**: Hopper (H100/H200) and Blackwell (B100/B200)
- **CUDA 13.0+**: Latest CUDA version only
- **Multiple Frameworks** (PyTorch, TensorFlow, JAX)
- **Best Practices** for modern GPU optimization

