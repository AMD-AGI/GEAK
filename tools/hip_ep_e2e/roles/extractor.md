# hip-ep Extractor — HeadKernel task

You generate the GEAK kernel task for one op and record its baseline, so that
kernel_workflow starts from a task that compiles, passes, and measures what the model runs.

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
   1. Compile (LONG; DIR `LANE_DIR`, NAME `compile`), the command being
      `cd TASK_DIR && python3 scripts/task_runner.py compile`.
   2. Correctness (LONG; DIR `LANE_DIR`, NAME `correctness`), the command being
      `cd TASK_DIR && python3 scripts/task_runner.py correctness`.
   3. Performance (LONG; DIR `LANE_DIR`, NAME `performance`), the command being
      `cd TASK_DIR && python3 scripts/task_runner.py performance`.

   If any step here or in step 3 fails, return `status` as `fail: <the step>: <the error>`.
   A task that fails at baseline is a tooling problem, not a kernel opportunity.
3. **Workload weights.** These are calls per forward × baseline ms:
   ```
   python3 TOOLS_DIR/make_task.py --out TASK_DIR --workload-from TASK_DIR/build/performance_report.json
   ```
4. **Steer.** Write the `steer` text for the kernel lane: the shapes with the largest
   weights, and the task `README.md`'s "How it is checked", "Not timed here" and "How it is
   measured", in a few lines.

## Return
```json
{"status": "pass", "workload_json": "TASK_DIR/workload.json", "steer": "..."}
```
