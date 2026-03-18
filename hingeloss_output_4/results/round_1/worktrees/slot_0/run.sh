#!/bin/bash
export PYTHONPATH=/workspace/GEAK/hingeloss_output_4/results/round_1/worktrees/slot_0:/workspace/GEAK/HingeLoss:${PYTHONPATH}
export HIP_VISIBLE_DEVICES=0
exec python3 "$@"
