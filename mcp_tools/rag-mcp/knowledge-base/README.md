# Knowledge Base

This directory contains curated knowledge documents used by the RAG pipeline. Content is organized by GPU vendor and topic layer.

## Directory Structure

```
knowledge-base/
├── README.md
├── INDEX.md
│
├── amd-knowledge-base/           # AMD / ROCm knowledge (primary)
│   ├── layer-1-hardware/         # GPU architecture fundamentals
│   │   └── amd-gpu-arch/         #   CDNA / RDNA architecture docs (3 files)
│   │
│   ├── layer-2-compute-stack/    # ROCm platform & programming model
│   │   ├── rocm/                 #   Installation, profiling, Docker images, stack overview (5 files)
│   │   ├── rocm-systems/         #   AMD SMI, runtime systems (2 files)
│   │   ├── hip/                  #   HIP programming: basics, memory, sync, porting, optimization (7 files)
│   │   └── therock/              #   TheRock build system overview (1 file)
│   │
│   ├── layer-3-libraries/        # ROCm math & ML libraries
│   │   ├── rocm-libraries/       #   Unified monorepo usage guide
│   │   ├── blas/                 #   rocBLAS, hipBLAS
│   │   ├── fft/                  #   rocFFT
│   │   ├── solver/               #   rocSOLVER
│   │   ├── sparse/               #   rocSPARSE
│   │   ├── random/               #   rocRAND
│   │   ├── algorithms/           #   rocThrust, HIP performance optimization
│   │   ├── communications/       #   RCCL (multi-GPU)
│   │   ├── ml-primitives/        #   MIOpen
│   │   └── compilers/            #   Triton on ROCm
│   │
│   ├── layer-4-frameworks/       # ML framework integration
│   │   ├── pytorch/              #   PyTorch on ROCm
│   │   ├── tensorflow/           #   TensorFlow on ROCm
│   │   ├── jax/                  #   JAX on ROCm
│   │   └── onnx/                 #   ONNX Runtime on ROCm
│   │
│   ├── layer-5-llm/              # LLM inference, training & deployment
│   │   ├── 00-quickstart/        #   Quick-start guides (inference, fine-tuning)
│   │   ├── 01-foundations/        #   Docker basics, Transformers on ROCm
│   │   ├── 02-inference/         #   Serving & deployment
│   │   │   ├── serving-engines/  #     vLLM, SGLang
│   │   │   ├── optimization/     #     Attention & serving optimization
│   │   │   └── deployment/       #     Docker & production deployment
│   │   ├── 03-training/          #   Fine-tuning & distributed training
│   │   │   ├── preparation/      #     Dataset prep, environment setup
│   │   │   ├── fine-tuning/      #     LoRA, QLoRA, full fine-tuning
│   │   │   ├── distributed/      #     FSDP training
│   │   │   └── optimization/     #     Memory optimization
│   │   ├── 04-models/            #   Model-specific guides
│   │   │   ├── llama/            #     LLaMA optimization
│   │   │   ├── mistral/          #     Mistral optimization
│   │   │   ├── gpt-models/       #     GPT optimization
│   │   │   └── other-models/     #     Diverse architectures
│   │   └── 05-advanced/          #   Advanced topics
│   │       ├── custom-kernels/   #     Triton kernel development
│   │       └── data-management/  #     MongoDB MCP integration
│   │
│   ├── layer-6-extended/         # Extended optimization resources (239 files)
│   │   └── optimize-guides/
│   │       ├── L0-core/          #   Core optimization docs — profiling, memory, perf counters (28 files)
│   │       ├── L1-important/     #   Important references — library tuning, SDK tools (38 files)
│   │       ├── L2-optional/      #   Optional/supplementary docs (19 files)
│   │       ├── silu_optim/       #   SiLU kernel optimization case study — bf16, coalescing, occupancy (16 files)
│   │       ├── rocWMMA-1.7.0/    #   rocWMMA library docs — API, concepts, samples (6 files)
│   │       └── customer-case/    #   Real-world kernel optimization reports (131 files)
│   │           ├── customer/     #     Internal customer kernels (20 kernels)
│   │           │                 #       e.g. silu_mul, rms_norm, convolution, knn, histogram ...
│   │           └── github_repo/  #     Open-source project kernel analyses
│   │               ├── vllm/     #       vLLM kernels (paged_attention, fused_moe, rope, ...)
│   │               ├── sglang/   #       SGLang kernels (decode_attention, fp8_gemm, moe, ...)
│   │               ├── rtp-llm/  #       RTP-LLM kernels (FP8_GEMM, MLA, MoE, ...)
│   │               ├── llama.cpp/#       llama.cpp kernels (softmax, quantize, matmul, ...)
│   │               ├── aiter/    #       AITer kernels (causal_conv1d, flash_attention, ...)
│   │               ├── composable_kernel/  # CK library kernels
│   │               └── warp-hip/ #       Warp-HIP kernels (BVH, radix sort, mesh query)
│   │
│   └── best-practices/           # Cross-cutting best practices
│       ├── performance/          #   GPU, memory & kernel optimization (3 files)
│       ├── debugging/            #   Debugging workflows (1 file)
│       ├── testing/              #   GPU testing strategies (1 file)
│       └── ci-cd/                #   GitHub Actions CI/CD (1 file)
│
├── nvidia-knowledge-base/        # NVIDIA / CUDA knowledge (mirror structure)
│   ├── layer-1-hardware/         #   NVIDIA GPU architecture
│   ├── layer-2-compute-stack/    #   CUDA programming
│   ├── layer-3-libraries/        #   cuBLAS, cuDNN, NCCL
│   ├── layer-4-frameworks/       #   PyTorch, TensorFlow, JAX on CUDA
│   ├── layer-5-llm/              #   LLM inference & training on NVIDIA
│   └── best-practices/           #   NVIDIA performance best practices
│
└── comparisons/                  # Cross-vendor comparisons
    ├── rocm-vs-cuda.md           #   ROCm vs CUDA ecosystem comparison
    └── hip-cuda-programming-comparison.md  # HIP vs CUDA programming API comparison
```
