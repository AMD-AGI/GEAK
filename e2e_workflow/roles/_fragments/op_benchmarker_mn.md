# Op Benchmarker — multi-node serving

Read this only when `SERVING_TOPOLOGY` is in your inputs.

**PHASE=bakeoff step 2.** This machine sees no GPU and the nodes have no `$SKILL_DIR`: copy
`$SKILL_DIR/scripts/op_bench.py` and `harness_lib.py` into `<OP_TASK_DIR>`, then run that copy on a
node card, same flags:
`<MN_GPU_EXEC> --target master --gpu <GPU_ID> -- python3 "<OP_TASK_DIR>/op_bench.py" --task "<OP_TASK_DIR>" ...`
