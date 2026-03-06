#!/bin/bash
# Script to run OpenEvolve optimization when GPU is available
# This uses auto-build mode since kernel.py has triton_op, torch_op, EVAL_CONFIGS, and --profile flag

set -e

KERNEL_DIR="/workspace/GEAK/add_kernel"
cd ${KERNEL_DIR}

echo "========================================"
echo "Starting OpenEvolve Optimization"
echo "========================================"
echo ""
echo "Kernel: ${KERNEL_DIR}/kernel.py"
echo "Mode: Auto-build (standard AIG-Eval interface detected)"
echo ""

# Run OpenEvolve with auto-build mode
python3 /workspace/geak-oe/examples/geak_eval/run_openevolve.py \
  kernel.py \
  --iterations 10 \
  --gpu 0 \
  --output optimization_output

echo ""
echo "========================================"
echo "Optimization Complete!"
echo "========================================"
echo ""
echo "Results available in:"
echo "  - optimization_output/best_kernel.py"
echo "  - optimization_output/openevolve_result.json"
echo "  - optimization_output/progress.log"
echo ""
