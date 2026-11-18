# hip-ep Extractor — HeadKernel task

You generate the GEAK kernel task for one op and record its baseline, so that
kernel_workflow starts from a task that compiles, passes, and measures what the model runs.

Export every `TOOL_ENV` variable first. Steps marked **LONG** follow "Long commands"
in your prompt.

## PHASE=extract (OP = matmul_nbits_u4_prefill)
1. **Task.** Generate it at `TASK_DIR`, starting from the current accepted overlay:
   ```
   python3 TOOLS_DIR/make_task.py --op OP --arch ARCH --model MODEL_PATH --m LENS --out TASK_DIR \
       [--overlay CUR_OVERLAY]
   ```
   Pass `--overlay` only when `CUR_OVERLAY` is non-empty. The tool reads the model's
   MatMulNBits shapes with hip-ep's own `extract_shapes.py`, at the prompt lengths the
   A/B gate times (`LENS`).
2. **Baseline.** In `TASK_DIR`:
   1. `python3 scripts/task_runner.py compile` (about 2.5 minutes; foreground, Bash timeout 600000).
   2. Correctness (LONG; DIR `LANE_DIR`, NAME `correctness`), the command being
      `cd TASK_DIR && python3 scripts/task_runner.py correctness`. It runs every model shape, the
      M=1 guards, the leaf's full coverage sweep with every extended WMMA entry, and the
      qmoe_ragged_mm leaf, which instantiates the same WMMA core.
   3. Performance (LONG; DIR `LANE_DIR`, NAME `performance`), the command being
      `cd TASK_DIR && python3 scripts/task_runner.py performance`.

   If any step here or in step 3 fails, return `status` as `fail: <the step>: <the error>`.
   A task that fails at baseline is a tooling problem, not a kernel opportunity.
3. **Workload weights.** These are calls per forward × baseline ms:
   ```
   python3 TOOLS_DIR/make_task.py --out TASK_DIR --workload-from TASK_DIR/build/performance_report.json
   ```
4. **Steer.** Write the `steer` text for the kernel lane. Name the shapes with the
   largest weights, and state these facts:
   - the baseline already uses the fastest config the kernel's own autotune finds, so
     changing only which config is picked does not count;
   - the K whose row stride production pads (see "Not timed here" in the task's
     `README.md`) runs a padded-B variant that this harness does not time;
   - every WMMA config must stay correct, extended ones included, and so must the
     qmoe_ragged_mm kernel that instantiates the WMMA core.

## Return
```json
{"status": "pass", "workload_json": "TASK_DIR/workload.json", "steer": "..."}
```
