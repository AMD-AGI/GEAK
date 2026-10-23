# UT contracts: per-operator templates for extracted kernel tests

A delivered task used to be ~500 hand-written lines of `unittest.py` plus a `cases.py`
whose operand construction, tolerance and gate logic were reinvented per kernel. Most UT
defects found in review came from that reinvention: a dropped `is_shuffled` flag that timed
the wrong kernel, one task-wide `tol` applied to integer top-k indices, uniform MoE routing
timed as if served, a negative control that only failed to compile, a padded graph tail
compared raw. This directory moves what is the same for every kernel of one operator family
into a declared **contract** and one **driver**, the way SIKL's `OpContract`
(`sikl/core/contract`) does for its kernels, so a task only writes what is specific to it.

| Piece | What it is |
|---|---|
| `families/<family>.json` | the contract of one operator family (17 today, `scaffold_ut.py list`) |
| `ut_contract.py` | contract loading, CPU lint, seeded operand construction, input checks, the bound compares |
| `reference_f64.py` | float64 references, one per family that has one, row-subsettable |
| `ut_driver.py` | every gate, driven by the frozen contract; a task's `unittest.py` calls it |
| `scaffold_ut.py` | `new` / `sync-meta` / `lint` / `list` |
| `templates/` | the generated `cases.py` and `unittest.py` |
| `tests/` | CPU tests, including a scaffolded task run through every gate |

`ut_contract.py`, `reference_f64.py` and `ut_driver.py` are copied into each task beside
`harness_lib.py` and frozen with it; the family contract is frozen as `run/ut_contract.json`.
Nothing in a task imports this directory at run time.

## Workflow

```bash
UTC="$SKILL_ROOT/ut_contracts"
python3 "$UTC/scaffold_ut.py" list
python3 "$UTC/scaffold_ut.py" new --family gemm_a8w8_blockscale --task "$EVAL_DIR/tasks/<name>" \
    --target <module:entry> --bind-module <module> --bind-file kernel_src/<file>.py \
    --attr fp8_dtype=float8_e4m3fn --axis n=7168 --axis k=2048 \
    --manifest <cases.json> --phase-regimes <regimes.json> [--override <overrides.json>]
# fill the FILL markers: cases.py HOOKS and call(), replay_bundle() for graph phases,
# meta.json evidence_path and negative_controls, run/ut_cases.json if no --manifest
python3 "$UTC/scaffold_ut.py" sync-meta --task "$EVAL_DIR/tasks/<name>"
python3 "$EVAL_DIR/tasks/<name>/ut_contract.py" "$EVAL_DIR/tasks/<name>"     # CPU lint, exit 0/1
cd "$EVAL_DIR/tasks/<name>" && bash "$SHARED_TOOLS/gpu_lock.sh" "$GPU_ID" python3 unittest.py
```

With several GPUs visible (`--gpus`, else `$UT_GPUS`, then `$GPU_IDS`) the driver shards
its correctness legs by case and spreads timing buckets over them. `--quick` runs every gate
but timing on the smallest, middle and largest case of each phase (one draw, a few server
calls) and writes `run/ut_report.quick.json` -- an iteration aid, never a pass or a receipt.
`--reuse` keeps the gates that passed in `run/ut_report.json` when nothing they read changed,
e.g. `--skip timing` in a lane, then `--reuse` alone to time it.

The lint needs no GPU and no torch; run it before every GPU session. `unittest.py` writes
`run/ut_report.json` (every gate's measures) and `run/ut_result.json` (verdict, exit code,
snapshot hash, gate statuses), prints `UT_GATE <gate> <status>` lines and, when timing ran,
`GEAK_WEIGHTED_SPEEDUP=<value>`. Exit 0 = every gate passed; 1 = a gate failed or did not
run (`--skip` makes the verdict `partial`, never `pass`); 3 = `UT_HARNESS_INCOMPLETE`, the UT
itself is missing something (a graph phase without a replay bundle) — fix the UT, do not
blame the candidate.

## The case manifest (`run/ut_cases.json`)

```json
{"cases": [{"sig": "DECODE_m4", "regime": "DECODE", "axes": {"m": 4},
            "effective_calls": 1840.0, "timed": true, "random": true,
            "evidence": "observed_workload.json#..."}],
 "replay_sigs": ["DECODE_m1", "DECODE_m64"]}
```

`axes` holds the case's var axes (and may restate const ones, which must agree with the
task's). `regime` is the phase label and must be a `meta.phase_regimes` key. Timed cases need
finite `effective_calls` from the observed workload; `random` cases feed the value-parity and
negative-control gates; `replay_sigs` must hold at least two cases of every graph-deployed
phase. `meta.workload.cases` is derived from this file by `sync-meta` and the lint rejects
any drift between them.

## What a contract declares

| Field | Meaning |
|---|---|
| `match` | `{kernel: [regex], op: [regex]}`: device-kernel names and launching ops this family covers; `scripts/draft_task_decls.py` proposes the family whose patterns fit most of a task's kernels |
| `axes` | `{lo, hi, multiple_of}`; `var: true` marks an axis that varies by case (tokens, context), others are pinned per task with `--axis`; `expr` derives one (`"ceil_div(k, 128)"`) |
| `attrs` | variant selectors with their legal `values` (dtype, layout, activation...); pinned per task with `--attr`, default = first value |
| `constraints` | expressions every case must satisfy (`"topk <= n_experts"`) |
| `inputs.<name>` | `shape` (axis expressions, or `{"by": attr, "cases": {...}}`), `dtype` (`"attr:<name>"` allowed), `dist`, `lifetime` (`weight`/`per_call`/`state`), optional `layout` hook, `present_when`, `writable` (the live entry writes it in place: `call()` must clone) |
| `outputs.<name>` | `shape`, `dtype`, `bound` (below), optional `max_violation_fraction`, `sort_by` (compare rows in another output's order: weights follow their ids) |
| `compare` | `determinism` (`bitwise` = the candidate must reproduce itself byte for byte), `max_violation_fraction`, `element_order` |
| `layout_facts` | attributes the dispatcher reads that the built operands must carry (`b.is_shuffled`) |
| `reference`, `reference_rows` | the float64 reference and how many rows of the token axis it computes |
| `hooks`, `call` | what the task must write, and how |
| `timing`, `negative_controls`, `pitfalls` | the cache condition to record, controls known to work, and failures seen on this family |

A task departs from its family only through `--override` items `{"path", "value", "why"}`;
an override without a `why` is refused, and every applied override is printed by the lint.
For an operator no family covers, write a contract file in the same format and pass
`--family-file`; the same driver and lint apply.

### Bounds

Every compare happens on the CPU in float64. `y` is the float64 reference, `b` the frozen
baseline's output, `u` the output dtype's unit roundoff.

| `kind` | Passes when | Use for |
|---|---|---|
| `dot_product` | `|c - y| <= 2(6 sqrt(n_eff) + 2) u_acc (|a|@|b|) + u|y|`, n_eff from `length` and `max_split_k`; unbiased | GEMMs whose accumulation follows the IEEE model (bf16, MXFP4 operands) |
| `baseline_relative` | `|c - y| <= 2 max(|b - y|, rms_row(b - y)) + u|y|`; unbiased or no more biased than the baseline | the default where the installed kernel is itself inexact (FP8 MFMA, atomics, attention) |
| `output_rounding` | `|c - y| <= 2u|y|` | ops every correct kernel computes to one rounding (residual add) |
| `tolerance` | `|c - y| <= atol + (rtol + u)|y|` (`against: baseline` uses `b`) | norms, activations, rope |
| `noise_relative` | `|c - b| <= m max(|b2 - b|, rms_row(b2 - b)) + k u|b|`, b2 a second baseline run | quantized outputs whose last code flips at rounding boundaries |
| `sqnr` | `20 log10(|y|/|c - y|) >= min_db` | whole-output checks when no elementwise model holds |
| `exact`, `multiset_rows` | equal, or equal as per-row multisets | integer outputs, routing ids |
| `topk_equivalent` | per row: same count of valid entries, distinct, in range, and the selected `scores` equal the baseline's selected scores (within `score_atol/rtol`) | top-k indices; ties may pick either index; `scores: "ref:<name>"` for a fused score+top-k |
| `bytes_exact` | equal bytes (up to `max_violation_fraction`) | packed FP4, caches |
| `harness` | `harness_lib.correct` with `tol` | legacy; the lint warns |

"Unbiased" is SIKL's test `|mean(e)| / rms(e) <= 5 / sqrt(n)`: a candidate whose errors all
lean one way (a dropped scale, a truncating cast) fails even inside the elementwise bound.

### Distributions

`normal {mean, std}`, `lecun_normal {fan_in}` (unit-variance GEMM outputs), `uniform`,
`log_uniform {log2_lo, log2_hi}` (scales), `int_range {lo, hi, distinct}`, `permutation`
(tie-free rows), `routing {experts, load: uniform | zipf:<s> | observed, counts}`, `simplex`
(routing weights), `cu_seqlens`, `seq_lens`, `page_table {pool}` (a shuffled pool),
`sparse_indices {context, pad}` (distinct slots stratified over the whole context), `constant`,
`hook` (a task function; may produce several inputs, e.g. packed values and their scales),
`produced {by}` and `fixture` (structural state loaded by cases.py). Parameters may be
expressions over the case's axes. Every draw comes from the leg's seeded generator, so the
baseline leg, the candidate leg and the driver rebuild identical operands.

## Gates (`ut_driver.py`)

| Gate | Process | Fails when |
|---|---|---|
| `contract` | driver, CPU | the lint finds an error |
| `inputs` | driver | a case does not rebuild byte-identically from its seed, a layout fact is missing, an integer input leaves its range |
| `binding` | legs | both legs import the same code, or the baseline resolves inside the task |
| `correctness` | baseline + candidate oracle legs (fresh processes, deployment compile mode applied) | any random case/draw violates its output bound; with `bitwise` determinism, two candidate runs differ |
| `isolation` | candidate | `call()` modified an input, or two calls share an output buffer |
| `replay` | baseline eager refs, candidate graph replay | a graph phase lacks a bundle (exit 3), capture fails, or a replayed case violates its bound |
| `server_calls` | baseline + candidate replay of `run/server_calls.json` (server evidence only) | the candidate raises or differs from the baseline (beyond `server_call_tol`, default `replay_tol`) on a call the baseline replays; calls the baseline cannot rebuild are listed, and their variants fall to `compile_variants` |
| `compile_variants` | the baseline legs above, under `compile_watch` | a Triton variant in `run/server_variants.json` is built by no baseline leg and not waived (`compile_variant_waivers: [{id, why}]`), or the legs' Triton/torch namespaces differ from the server's (`server_preimports`) |
| `negative_controls` | candidate oracle per mutation | a mutation passes the compare, fails by < `min_control_margin` (default 10x the bound), or does not run at all; the source is restored and re-hashed either way |
| `restore` | driver | any `kernel_src` byte differs from the start of the run |
| `timing` | `harness_lib.measure_legs` | no measurable weighted speedup |

Timing is unchanged from the shared harness (`measure_legs` + `serving_weighted_speedup`) so
the optimize stage reads the same number it always has; the report adds which buckets
stopped before `--max-reps` and the recorded cache condition.

## Limits

- A family contract is not a substitute for reading the live call: shapes, dtypes and the
  boundary come from the trace and the installed source; the contract checks them.
- `all_reduce` is declared but the single-process driver cannot run a collective; use the
  rank harness of `references/collective-ut.md` with the contract's bound.
- Hooks that quantize or shuffle must call the installed functions; the CPU tests use
  stand-ins and prove nothing about a real quantizer.
- The float64 references are the operator's mathematics. When the installed baseline fails
  its own family bound, find out why before overriding it.
