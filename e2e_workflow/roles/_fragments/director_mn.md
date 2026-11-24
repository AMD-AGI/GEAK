# Director — multi-node serving

Read this only when `SERVING_TOPOLOGY` is in your inputs. It adds to the role file's steps as follows.

**PHASE=setup step 4.** This machine sees no GPU and the nodes have no GEAK code: copy
`gpu_identity.py` into `$EVAL_DIR`, and run that copy and every GPU / stack probe of this step
(including `env_info.txt`) on the master, prefixed with `<MN_GPU_EXEC> --target master --`.

## PD (`SERVING_TOPOLOGY=pd_disagg`)

**PHASE=setup step 5.** Before you accept a baseline, grep both engines' `server.log` (the `log` of
each entry under `pd.engines` in the leg's `mn_launch.json`) for `KVTransferError` /
`transfer failed`. If either appears, the KV handoff died and the requests came back empty, though the
client still credits their full length: the number is not a measurement. Re-run on a fresh server (up
to 2 more attempts) and report the log line in `notes`; never return it as `baseline_throughput_tok_s`.

**PHASE=validate, the final block.** Also export the `GEAK_MN_PD_SWEEP_ARGS_*` /
`GEAK_MN_PD_SWEEP_ENV_*` inputs. Those are per-engine config; `EXTRA_SERVER_ARGS` / `EXTRA_ENV` are the
shared layer and must not absorb them. `OVERLAY_PYTHONPATH` is the one code tree both engines run.
