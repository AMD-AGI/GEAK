#!/bin/bash
# Preprocess 4 representative kernels ONCE.
# Output is reused by all ablation experiments for consistent baselines.
#
# Usage: bash scripts/preprocess_once.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PREPROCESS_DIR="/home/sapmajum/geak-experiments/preprocess"
IMAGE="lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"

KEY1="${AMD_LLM_API_KEY:-fa273d4402b74a9c830c9e9fc4ebfb54}"
CONTAINER="geak-preprocess"

declare -A KERNEL_URLS
KERNEL_URLS[rope]="https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/rope/kernel.py"
KERNEL_URLS[fused_rms_fp8]="https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/fused_rms_fp8/kernel.py"
KERNEL_URLS[topk]="https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/topk/kernel.py"
KERNEL_URLS[nsa_backward]="https://github.com/AMD-AGI/AIG-Eval/blob/geak-eval-kernels/tasks/geak_eval/nsa_backward/kernel.py"

echo "=== GEAK Preprocessing (one-time, 4 kernels) ==="
echo "Output: $PREPROCESS_DIR"
mkdir -p "$PREPROCESS_DIR"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    -v "$REPO_DIR:/workspace" \
    -v /home/sapmajum/AIG-Eval:/workspace/AIG-Eval \
    -v "$PREPROCESS_DIR:/preprocess" \
    -e AMD_LLM_API_KEY="$KEY1" \
    -e HIP_FORCE_DEV_KERNARG=1 \
    "$IMAGE" \
    sleep infinity

sleep 3

docker exec "$CONTAINER" bash -c '
pip install -e /workspace 2>&1 | tail -1
# Install Metrix profiler
if ! python3 -c "from metrix import Metrix" 2>/dev/null; then
    git clone --depth 1 https://github.com/AMDResearch/intellikit.git /tmp/intellikit 2>&1 | tail -1
    cd /tmp/intellikit/metrix && pip install -e . 2>&1 | tail -1 && cd /workspace
fi
pip install -e /workspace/mcp_tools/metrix-mcp 2>&1 | tail -1
pip install -e /workspace/mcp_tools/profiler-mcp 2>&1 | tail -1
mkdir -p /workspace/.geak_resolved
if [ ! -d "/workspace/.geak_resolved/AMD-AGI_AIG-Eval" ]; then
    ln -sf /workspace/AIG-Eval /workspace/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null || true
fi
git config --global --add safe.directory "*"
mkdir -p /root/.config/mini-swe-agent
cat > /root/.config/mini-swe-agent/.env << ENVEOF
MODEL=openai/amd-llama-4-maverick
OPENAI_API_KEY=$AMD_LLM_API_KEY
OPENAI_BASE_URL=https://api.amd.com/llm/v1
MSWEA_CONFIGURED=true
MSWEA_MODEL_NAME=openai/amd-llama-4-maverick
ENVEOF
echo "Setup complete"
'

echo ""
for kernel in rope fused_rms_fp8 topk nsa_backward; do
    URL="${KERNEL_URLS[$kernel]}"
    echo "=== Preprocessing: $kernel ==="
    docker exec "$CONTAINER" bash -c "
        mkdir -p /preprocess/$kernel/.geak_resolved
        # Symlink existing AIG-Eval clone instead of re-cloning from GitHub
        ln -sf /workspace/.geak_resolved/AMD-AGI_AIG-Eval /preprocess/$kernel/.geak_resolved/AMD-AGI_AIG-Eval 2>/dev/null || true
        geak-preprocess '$URL' -o /preprocess/$kernel/ --gpu 0 2>&1 | tail -20
        echo '---'
        echo 'Files created:'
        ls /preprocess/$kernel/
    "
    echo ""
done

echo "=== Preprocessing complete ==="
echo "Verify:"
for kernel in rope fused_rms_fp8 topk nsa_backward; do
    harness=$(docker exec "$CONTAINER" cat "/preprocess/$kernel/harness_path.txt" 2>/dev/null || echo "MISSING")
    baseline=$(docker exec "$CONTAINER" bash -c 'grep GEAK_RESULT_LATENCY_MS /preprocess/'$kernel'/full_benchmark_baseline.txt 2>/dev/null | tail -1' || echo "MISSING")
    profile=$(docker exec "$CONTAINER" python3 -c "import json; p=json.load(open('/preprocess/$kernel/profile.json')); print('success:', p.get('success'))" 2>/dev/null || echo "MISSING")
    echo "  $kernel: harness=$harness baseline=$baseline profile=$profile"
done

echo ""
echo "Preprocessing output saved to: $PREPROCESS_DIR"
echo "This will be reused by all experiments via geak-orchestrate --preprocess-dir"
