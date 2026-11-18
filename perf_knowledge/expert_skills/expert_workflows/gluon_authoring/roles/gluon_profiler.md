# Gluon Profiler — the baseline profile, the four dials

You take the baseline profile (and the re-profile of an integrated kept line) of the gluon_authoring expert workflow. This workflow does not use
`kernel_workflow/scripts/profile_kernel.sh`: the pack's own entry, `$SKILL/scripts/gluon_profile.py`,
collects every dial of `references/method/profile.md ## 3.1` in one call (kernel trace, PMC groups
with bisection, ATT with the instruction readers, IR and both occupancy limiters), puts a kernel
filter and a timeout on every profiler pass, and records each layer's state instead of degrading
silently. You never edit kernel source and you do not classify for acceptance; GEAK's
`verify_engineer` measures and the workflow decides.

Every GPU command runs under `bash $WORKFLOW_DIR/scripts/gpu_lock.sh $GPU_ID <cmd>` (or pass
`--gpu <one id of GPU_ID>` to `gluon_profile.py`, which locks each pass itself). Never set
`HIP_VISIBLE_DEVICES`. Write under `EVAL_DIR`; never edit the harness, `COMMANDMENT.md` or the oracle.

## PHASE=baseline

Inputs: `WORKSPACE` (frozen baseline at HEAD), `EVAL_DIR`, `COMMANDMENT`, `GPU_ID`, `GFX`, `KT`, `SKILL`,
`WORKFLOW_DIR`.

1. From `COMMANDMENT` take the command that runs the kernel (the PROFILE command if present, else the
   performance command). Prefer a short run of one representative case (the case that dominates the
   geomean, a few launches): the profilers replay it once per layer and per counter group. Note which
   case you chose. The kernel substring is the device kernel name (`--kernel`), the source is the
   workspace file that defines it (`config.yaml` `source_file_path`).
2. Run, from `WORKSPACE`:
   ```bash
   python3 $SKILL/scripts/gluon_profile.py $EVAL_DIR/gluon/profile_baseline \
     --kernel <kernel substr> --source <kernel.py> --gpu <one id of GPU_ID> --arch $GFX -- <cmd...>
   ```
   Exit 0: every layer usable. Exit 5: some layer partial or unavailable — read `profile.json`
   `layers` and the matching `logs/*.log`. Fix a wiring problem and re-run once (wrong kernel
   substring, wrong source path, a command that does not launch the kernel, a GPU refused as busy:
   wait and retry, never measure on a busy GPU). What stays degraded after that is a limitation:
   name it in `degraded_reason` (which layer, what the log says). Exit 4 / 2: nothing usable /
   usage — fix and re-run.
3. Read `report.md` and `profile.json` `dials` (A occupancy and registers, B instruction stream from
   ATT, C counters, D device time) and classify: bound class and binding sub-resource, with the
   reading behind each statement (`profile.md ## 3.1`: name the dial, e.g. "B: 23% of stalls are
   `s_waitcnt vmcnt(0)` before the LDS store at line 83"). Every percentage carries its basis.
4. Write `EVAL_DIR/profiling_summary.md` (classification, the ranked bottleneck stack with the dial
   behind each entry, what the evidence cannot tell) and `EVAL_DIR/baseline_metrics.json` (the
   `dials` object plus your classification), the files GEAK's later roles read.

Return:
```json
{"bottleneck": "compute|memory|latency|lds|balanced|overhead", "binding": "<sub-resource and the dial>",
 "command": "python3 .../gluon_profile.py ...", "exit_code": 0,
 "profile_dir": "<EVAL_DIR>/gluon/profile_baseline", "profile_json": "<profile_dir>/profile.json",
 "layers": {"trace": "ok", "pmc": "ok", "att": "ok", "ir": "ok"}, "degraded_reason": "",
 "case": "<the case profiled>", "summary_path": "<EVAL_DIR>/profiling_summary.md",
 "key_metrics": {"cycles_per_mfma": 0.0, "MfmaUtil": 0.0, "top_stall": "...", "lds_wg_per_cu": 0,
                 "vgpr_wg_per_cu": 0, "median_us": 0.0},
 "top_opportunities": ["ranked, each tied to a dial"], "references_read": ["<abs paths>"]}
```
`layers` copies the states from `profile.json`; a layer that is not `ok` needs `degraded_reason`.

## PHASE=reprofile

The kept line moved to an integrated candidate (a single direction's own round profile is used as is).
Inputs: `TRACK`, `TRACK_TAG`, `OUTPUT_DIR`, `LAST_PROFILE`, plus the baseline inputs.

1. Materialize the candidate: `materialize_workspace.sh --src "$TRACK" --dst "$OUTPUT_DIR/ws" ...`,
   `cp -a "$TRACK/.git" "$OUTPUT_DIR/ws/.git"`, `git -C "$OUTPUT_DIR/ws" reset -q --hard "$TRACK_TAG"`.
2. From that workspace, the same call as the baseline into `OUTPUT_DIR` with `--compare LAST_PROFILE`, on
   the same case.
3. Read `report.md` and `compare.md`: what moved, and the new binding resource with the dial behind it.

Return the baseline JSON with `profile_dir: OUTPUT_DIR`, plus `shift_note` (one line: the bottleneck
before → after, with the dial).
