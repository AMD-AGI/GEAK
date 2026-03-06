#!/bin/bash
set -e
cd /workspace/GEAK/68_gemm_add
./build/example_gemm_add_xdl_fp16 1 1 0 1024 1024 1024 1024 1024 1024 1024 && ./build/example_gemm_add_xdl_fp16 1 1 1 2048 2048 2048 2048 2048 2048 2048
