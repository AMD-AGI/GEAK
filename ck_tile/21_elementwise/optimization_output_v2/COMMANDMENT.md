## SETUP
export HIP_VISIBLE_DEVICES=${GEAK_GPU_DEVICE}
export PYTHONPATH=${GEAK_WORK_DIR}:${PYTHONPATH}

## CORRECTNESS
python3 /workspace/GEAK/ck_tile/21_elementwise/test_harness.py --correctness

## PROFILE
python3 /workspace/GEAK/ck_tile/21_elementwise/test_harness.py --profile > /dev/null 2>&1 || true
python3 /workspace/GEAK/ck_tile/21_elementwise/test_harness.py --profile > /dev/null 2>&1 || true
kernel-profile "python3 /workspace/GEAK/ck_tile/21_elementwise/test_harness.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --replays 5
