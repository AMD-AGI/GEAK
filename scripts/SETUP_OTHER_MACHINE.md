# GEAK Cross-Session Memory + RAG Setup (for another MI355X / 8-GPU machine)

This documents the exact setup currently running on this machine so it can be reproduced.

## Branch state
- Repo: `git@github.com:AMD-AGI/GEAK.git`
- Branch: `fix/eval-patch-apply-3way-fallback` (latest commit: `a3d47c16`)
- Includes: cross-session memory + RAG hook + patch-apply 3-way fallback fix

## Repo + container layout

```bash
# Required repos
git clone -b fix/eval-patch-apply-3way-fallback git@github.com:AMD-AGI/GEAK.git /data/GEAK
git clone -b geak-triton-common-benchmark git@github.com:AMD-AGI/AgentKernelArena.git /data/AgentKernelArena
```

## Two-slot containers (8 GPUs total, 4 per slot)

```bash
docker pull lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x

# Slot 1: GPUs 0-3
docker run -d --name geak_slot1 \
  --privileged --network host --ipc host \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/kfd --device /dev/dri \
  -v /dev/shm:/dev/shm -v /home:/home -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x sleep infinity

# Slot 2: GPUs 4-7
docker run -d --name geak_slot2 \
  --privileged --network host --ipc host \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/kfd --device /dev/dri \
  -v /dev/shm:/dev/shm -v /home:/home -v /data:/data \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x sleep infinity
```

For MI300X: change `gfx950` to `gfx942` AND swap `mi35x` for `mi30x` in the image tag.

## Per-container install (run for each slot)

```bash
SLOT=geak_slot1   # or geak_slot2

# Install GEAK from our branch
docker exec $SLOT pip install -e /data/GEAK

# Configure agent
docker exec $SLOT bash -c '
mkdir -p /root/.config/mini-swe-agent
cat > /root/.config/mini-swe-agent/.env <<EOF
MSWEA_MODEL_NAME=claude-opus-4.6
MSWEA_MODEL_CLASS=amd_llm
AMD_LLM_API_KEY=YOUR_KEY_HERE
MSWEA_CONFIGURED=true
EOF
'

# Pin aiter (REQUIRED — 12 of 18 kernels use it)
docker exec $SLOT bash -c '
cd /sgl-workspace/aiter &&
git fetch && git reset --hard && git clean -fd &&
git checkout 22122345c03991cb8026947b8df05e02f50d1f88
'

# Verify
docker exec $SLOT geak --version  # Should print: GEAK-v3 agent v0.1 (core: mini-swe-agent 1.14.4)
```

## Knowledge base

Lives at `/data/GEAK/src/minisweagent/memory/cross_session/knowledge_base.json` (pulled with the branch).
Currently 30 entries, all with verified real diffs (3KB-37KB each), full original_kernel_code (2KB-48KB), and best_speedup ≥ 1.10x. No empty fields across 870 cells.

When `GEAK_SAVE_TO_KNOWLEDGE_BASE=1`:
- The container's local SQLite DB at `/root/.cache/geak/memory.db` is auto-seeded from the JSON on first read
- New winning runs (≥1.10x verified) are auto-appended to the local DB
- The host JSON is NOT touched at runtime — sync back manually if you want persistent updates

## Run all 18 kernels (recommended sequencing)

### 1. Queue files

`/data/slot1_queue.txt`:
```
# Slot1 (GPUs 0-3): L1 + early L2 (lighter compute, faster runs)
L1:fused_append_shared_experts:on
L1:llama_ff_triton:on
L1:mla_decode:on
L1:moe_routing_sigmoid_top1:on
L1:refk_fp8_blockwise_mm:on
L1:refk_identity:on
L2:fast_rms_layernorm:on
L2:ff_backward:on
L2:lean_atten_paged:on
```

`/data/slot2_queue.txt`:
```
# Slot2 (GPUs 4-7): late L2 + all L3 (heavier compute)
L2:topk:on
L3:fused_moe_mxfp4:on
L3:fused_mxfp4_quant_moe_sort:on
L3:fused_qk_rope_cache_mla:on
L3:fused_qkv_rope:on
L3:fused_rms_fp8:on
L3:gemm:on
L3:gemm_a16w16_atomic:on
L3:gemm_a16wfp4:on
```

Format: `LEVEL:KERNEL:MEM_MODE`. MEM_MODE = `on` (KB+RAG read+write) | `off` (memory disabled) | `on-norag` (KB only, RAG off).

### 2. Master queue runner

The host-side runner is `scripts/multi_slot_runner.sh` (in this repo). It:
- Waits for slot idle
- Per-kernel: snapshots prior winner, clears outputs + Triton/torch_extensions cache, warms baseline, runs `geak -t '...'` with the right MEM_ENV, then snapshots outputs
- Sets `GEAK_USE_KNOWLEDGE_BASE=1` + `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` for `mem=on`
- Sets `GEAK_MEMORY_DISABLE=1` for `mem=off`
- Sets `GEAK_RAG_HOOK_DISABLE=1` for `mem=on-norag`

```bash
# Launch both slots in parallel
setsid bash /data/GEAK/scripts/multi_slot_runner.sh geak_slot1 /data/slot1_queue.txt </dev/null > /tmp/mq_slot1.log 2>&1 &
disown
setsid bash /data/GEAK/scripts/multi_slot_runner.sh geak_slot2 /data/slot2_queue.txt </dev/null > /tmp/mq_slot2.log 2>&1 &
disown

# Watch progress
tail -f /data/triton_runs/master_geak_slot1.log
tail -f /data/triton_runs/master_geak_slot2.log
```

### 3. Expected runtime
- ~60-120 min per kernel (5 rounds × 4 parallel sub-agents, each ~10-25 min)
- 9 kernels per slot × ~90 min average = ~13.5 hours per slot, parallel
- 2 failures expected (timeouts) — master_queue auto-advances

## Memory + RAG flags

| MEM_MODE | KB read | KB write | RAG hook | Use case |
|---|---|---|---|---|
| `on` | yes | yes | yes | Production runs |
| `on-norag` | yes | yes | no | Isolate KB contribution |
| `off` | no | no | no | Baseline / regression check |

Per-run env:
- `GEAK_USE_KNOWLEDGE_BASE=1` — read KB at task start
- `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` — auto-save winning run to local DB
- `GEAK_MEMORY_DISABLE=1` — disable all memory paths
- `GEAK_RAG_HOOK_DISABLE=1` — disable RAG snippets (KB-only)
- `GEAK_MEMORY_MIN_SPEEDUP=1.10` — single-source-of-truth speedup threshold (mirrored in formatter, retriever, KB)

## Output locations

Per-kernel run output (inside container): `/workspace/outputs/<kernel>/`
- `final_report.json` — overall best
- `geak_agent.log` — full trace
- `results/round_N/<task>/best_results.json` — per-strategy speedup + patch path
- `results/round_N/<task>/patch_*.patch` — actual diff applied

Host-side snapshots (cleared per kernel by master queue):
- `/data/triton_runs/winner_snapshots/<slot>_<kernel>_<TS>/best_patch.diff` — winning diff
- `/data/triton_runs/canonical_snapshots/<slot>_<kernel>_mem<mode>_<TS>/` — full output dir copy
