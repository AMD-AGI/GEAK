# Every role — multi-node serving

Read this only when `MN_STATE` is in your inputs.

This run serves ONE instance on `MN_NODES` machines (`SERVING_TOPOLOGY`), pinned in `MN_STATE`.
bench_e2e.sh adds `BENCH_LAUNCHER=multinode` / `GEAK_MN_STATE` itself and refuses any other launcher
here. Each machine's TP and cards come from the pin (`MN_TP_BY_ROLE`, when present, is each PD
engine's TP).

You and every Bash command you run are on the driver: it sees NO GPU and the nodes have no GEAK code.
Keep files on the shared fs (`MN_SHARED_FS`, same path on the nodes), not /tmp: a node sees only that
and its own image. gpu_lock.sh and profile_kernel.sh run on a node (the master for a bare id, that
engine's machine for `<engine>:<id>`, e.g. `decode:3`). Anything else needing a GPU or a node
(rocm-smi) goes through the `MN_GPU_EXEC` command from your inputs:

    <MN_GPU_EXEC> --target <master|rank<N>|prefill|decode|host> [--gpu <id>] -- <command>

(or `--script <file>`). `--gpu` exports HIP_VISIBLE_DEVICES/GPU_ID and holds the card's lock.
An isolated server is

    <MN_GPU_EXEC> --target <T> --gpu <id> -- env BACKEND=<BACKEND> BENCH_LAUNCHER=native GPU=<id> TP=1 ... bash <eval_dir>/bench_e2e.sh

## PD (`SERVING_TOPOLOGY=pd_disagg`)

Pass each `GEAK_MN_PD_*` input (one engine's accepted config) VERBATIM in the env of every SERVING
bench_e2e.sh on the current config (a true-baseline leg omits them). `EXTRA_SERVER_ARGS` /
`EXTRA_ENV` / `OVERLAY_PYTHONPATH` reach BOTH engines: never copy a `GEAK_MN_PD_*` value into them.
