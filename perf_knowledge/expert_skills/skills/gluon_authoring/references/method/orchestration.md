# Orchestration — GEAK roles, GPU discipline, context lifecycle

**What this chapter decides.** *Which GEAK role* carries each stage of a Gluon track, how the one
engineer that carries the track holds a GPU, what it may run in parallel (only bounded measurements in its
own loop), what it returns, how its records survive a lost return, and what keeps a long session
affordable. It decides nothing about the kernel: no lever, gate or verdict lives here.

**When you are here.** Before the first edit of a track (who owns what, which launch args), when a
measurement needs a GPU, when a round ends and its result must be handed back, and when a long session
needs to stay cheap.

This skill follows GEAK's usage of expert skills: it is **injected into GEAK's existing
`kernel_workflow` roles** and adds no roles and no agents. GEAK's phases (`kernel_workflow/kernel_lane.js`:
Setup → Benchmark → Profile → Optimize → Verify → Merge → Report → Validate) are the stage machine; the
method's stages ([`index.md`](index.md)) are the work one GEAK role does inside them. The pack's record
tools (`toolctl.py`, stage cards, context lease, `canonical_record.py`, `round_record.py`,
`close_audit.py`) are **optional bookkeeping** the deep_engineer may use for its own records — they are not
a run mode ([`## Optional bookkeeping tools`](#optional-bookkeeping-tools)).

---

## Who does which stage

**Injection.** Off by default (opt-in `use_expert_skills="true"`; when off nothing is injected and the run
is unchanged). Only the planning and authoring roles receive it (`EXPERT_SKILL_ROLES` in `kernel_lane.js`:
`tech_lead`, `author_engineer`, `engineer`, `deep_engineer`). They read
`kernel_workflow/roles/_fragments/expert_skills.md`, match the skill through `EXPERT_SKILLS_DIR/index.yaml`
(`scope: kernel`, operator, gen, dtype, regime, `validation_status == validated`), and treat its procedure
as a high-prior candidate to reproduce — advisory, never overriding the isolated A/B against the oracle.
On a declared or detected gfx1201 (R9700) run the CDNA skills are isolated and not injected.

| method stage | GEAK phase | GEAK role | what this skill asks of it |
| --- | --- | --- | --- |
| entry (comparator) | Setup, Benchmark | Director (`PHASE=setup`, isolated workspace, `mode: optimize`); `benchmark_engineer` (harness, `COMMANDMENT`, baseline timing) | the comparator is the tuned plain champion (`champion_ms`), pinned before any Gluon edit ([`entry.md`](entry.md)) |
| budget, baseline evidence | Profile | `profile_engineer` (`kernel_workflow/scripts/profile_kernel.sh`, baseline) | the four dials ([`profile.md`](profile.md)); `hw_budget.py` before authoring ([`budget.md`](budget.md)) |
| launch | Optimize — plan | `tech_lead` | gate **G0** launch args ([`index.md`](index.md), [`entry.md`](entry.md)); dispatch **one** `deep_explore` direction for the port, no specialists beside it |
| entry assertion → transcribe → recover → evidence → climb | Optimize — execute | `deep_engineer` (`specialty=deep_explore`) | the whole track in one engineer's own measure → profile → rewrite loop ([`## The deep_engineer's method boundary`](#the-deep_engineers-method-boundary)) |
| measure (acceptance) | Verify | `verify_engineer` | independent re-benchmark of `best_patch.diff` in a clean workspace (`harness_lib`); its number is the round result |
| commit | Merge | `tech_lead` commit gate | `MIN_IMPROVE` (default 2%) over the cumulative best; a measured noise band only makes it stricter |
| re-profile, memory | Optimize — after merge | `profile_engineer` (`reprofile`), `tech_lead` (`update_memory`) | the next round plans from the new bottleneck and from the deep_engineer's notes (requests, closure self-review) |
| close | Report, Validate | the deep_engineer states the verdict and writes the closure self-review; Director validates and arbitrates | [`close.md`](close.md); Director quotes the self-review's strongest counter-evidence |

**GEAK owns the loop.** `Setup → Analyze → Benchmark (COMMANDMENT + baseline) → Profile → [Research] →
LOOP[ plan round → (Optimize ‖ Verify, pipelined) → Merge → re-profile → update memory ] → Report →
Validate`. The budget is GEAK's (directions; a `deep_explore` direction costs `DEEP_COST`, default 2, and
always runs in a dedicated round), not a pack round count; acceptance is GEAK's verify + commit gate +
Director, against `geomean(baseline_ms / optimized_ms)` in absolute latency.

**The evidence obligations do not lapse because GEAK drives.** Budget/roofline before authoring
([`budget.md`](budget.md)) and a re-profile every round with the four dials ([`profile.md`](profile.md)),
through `kernel_workflow/scripts/profile_kernel.sh` under `gpu_lock.sh`.

### The deep_engineer's method boundary

The deep_engineer in the single `deep_explore` direction owns entry → climb. Its input is an asserted
`plain_champion.json` (or, on an incumbent entry, the measured explicit kernel), not a plain kernel to
retune.

1. **Assert the champion / comparator first.** If the champion ref is absent or `champion_gate.py` fails,
   make no edit and return `status: "failed"` with `blocked_missing_context: <the missing item>` in
   `notes` ([`entry.md`](entry.md)).
2. **Recover a faithful anchor** from the champion's identified artifacts (transcribe, then recover —
   [`transcribe.md`](transcribe.md), [`recover.md`](recover.md)).
3. **Record recovery debt separately** from performance against `champion_ms`; reach the declared recovery
   condition (parity) before treating an anchor improvement as a deep win.
4. **Climb one coupled explicit-tile layer at a time** ([`climb.md`](climb.md)), keeping only oracle-gated,
   same-boundary evidence with lowered-code support.
5. **No config sweep, no branch, no re-derived front-end structure.** A suspect structure or config is
   returned as `structure_suspect` / `resweep_request` in the result; tech_lead turns it into the next
   round's direction (a tile retune goes to GEAK's own plain tuning).
6. **A result slower than `champion_ms` is a negative** (`negative_revert_plain`); it does not revise the
   plain champion.

### Required return

The return is GEAK's `worker_result` (`ENG_SCHEMA` in `kernel_lane.js`: `status`, `speedup_geomean`,
`measurement_valid`, `per_case`, `patch_file`, `strategy`, `strategies_tried`, `notes`), written to
`OUTPUT_DIR/worker_result.json` and returned as StructuredOutput, plus `best_patch.diff` and `report.md`
([`kernel_workflow/roles/deep_engineer.md`](../../../../../../kernel_workflow/roles/deep_engineer.md)). The
skill adds these to its `notes` and artifacts:

- champion / gate refs (`plain_champion.json`, the `champion_gate.py` result);
- anchor and best measurements against `champion_ms`;
- recovery state — `parity` or `parity_unreached` with the attributed residual split;
- checkpoint (path of the last confirmed-kept diff + metrics);
- rounds spent against the direction's budget;
- `structure_suspect` / `resweep_request` (with evidence paths), blocker, deferred work;
- at a ceiling / keep-baseline / negative close: the closure self-review (`closure_review.md` path,
  verdict, strongest counter-evidence, recommended next —
  [close.md `## Closure challenge (self-review before a ceiling / keep-baseline / negative close)`](close.md)).

An unmeasured or unverified candidate is a blocker / unknown, not a win. Director performs final
arbitration.

---

## GPU lock

The single owner of GPU locking is GEAK's **`kernel_workflow/scripts/gpu_lock.sh`** (lock dir
`/tmp/team_gpu_locks`, one `gpu_<id>.lock` per GPU). The pack's `scripts/gpu_lock.sh` is a shim that execs
it, so pack docs and GEAK roles take the same locks. **One GPU per engineer:** the lane hands each engineer
its `GPU_ID`; every compile / correctness / timing / profile command of the deep_engineer — including its
bounded experiments — runs under that one id.

```bash
cd <workspace> && bash kernel_workflow/scripts/gpu_lock.sh <gpu_id|pool> <command...>
```

- **Single id** — `flock -x -w 1200` on that GPU for the whole command (error after 1200 s).
- **Pool** (comma list of the GPUs THIS run was allocated — never an example list) — takes the first lane
  that is both unlocked and idle (`GEAK_GPU_REQUIRE_IDLE=1` default; idle = `gpu_busy_percent ≤
  GEAK_GPU_MAX_BUSY_PCT` (5) and VRAM used ≤ `GEAK_GPU_MAX_VRAM_MB` (1024)), waits up to
  `GEAK_GPU_POOL_WAIT` (1200 s), and appends the lane actually taken plus `wait_s` to `GEAK_GPU_USE_LOG`
  when that names a file.
- It exports `HIP_VISIBLE_DEVICES` for the command itself — **never inline `HIP_VISIBLE_DEVICES`** in a
  benchmark or profiler command; let the lock place it, and never nest a second lock on the same id inside
  the command (it would wait on its own flock). It also isolates `TORCH_EXTENSIONS_DIR` per workspace, pins
  `PYTORCH_ROCM_ARCH` to the selected GPU's arch (ROCR-scoped), reaps orphaned `rocm_agent_enumerator`
  processes, and runs a source-provenance guard before and after the command — exit 86 /
  `GEAK_SOURCE_INVALID` means discard all measurement output from that invocation, even if it printed PASS.
- Long runs: `bash scripts/wait_for.sh --run --log "$LOG" -- bash kernel_workflow/scripts/gpu_lock.sh <gpu_id> <cmd>`
  (paths from the GEAK repo root; the pack-relative `scripts/gpu_lock.sh` shim resolves to the same script).
- Profiling goes through `kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>`, which takes the
  same lock ([`profile.md`](profile.md)).

This serializes benchmark access per GPU, so GEAK engineers sharing GPUs do not corrupt each other's timing.

### GPU broker (opt-in)

`scheduler/` (in this pack) is a per-box broker that adds what flock cannot: a queue with priority bands
(`interactive` 10 < `wake` 20 < `verify` 30 < `measure` 40 < `sweep` 50 < `bulk` 60, aging one band per
`--aging-s`, default 30 s), per-kernel fair share, first-class RESIDENT leases with a short exclusive
`wake()` window per reading, TTL reclaim of wedged holders, and a journal (`wait_s` / `hold_s`, audited
with `scheduler/audit_journal.py`). There is deliberately no shared-compute lease kind.

It is **off by default** in GEAK: enable it with `GEAK_GPU_BROKER=1` (autostart stays off,
`GEAK_GPU_BROKER_AUTOSTART=0`, unless you set it); its locks are shared with GEAK at
`/tmp/team_gpu_locks` (the same directory `gpu_lock.sh` flocks). With no daemon listening,
`gpu_lock.sh` takes the flock path and every call site behaves exactly as before; the broker's failure
posture is degrade, never block. Fair-share key: `GEAK_GPU_GROUP` if set, else inferred
(`GEAK_GPU_REQUIRE_GROUP=1` fails rather than guessing). Full operating notes: `scheduler/README.md`.

---

## Parallelism: bounded measurements only, never a fan-out of the port

The engine is **one engineer running the sequential layer loop** ([`climb.md`](climb.md)). This pack is
`breadth_enabled: false` (`scripts/pack_facts.json`, `lever-cards.json _search_policy`): there is no plain
tier here (the champion arrives swept and pinned by GEAK's plain rounds), no structural branch, and the
comparator is `champion_ms` from the bundle, never a `plain_baseline` computed in this track.

| level | what | how in GEAK |
| --- | --- | --- |
| plain search (L0) | diverse plain hypotheses, config sweeps | GEAK's own plain-Triton rounds (`tech_lead` + specialist `engineer`s), before or after the Gluon track — never during it |
| Gluon backbone (L1) | transcribe → recover → climb over coupled layers | **one** `deep_explore` direction; the deep_engineer's own sequential greedy climb |
| within a layer (L2) | a layer variant that is not budget-decidable | evaluated sequentially in the same loop; prefer a budget decision over sampling |

**Why the port is never fanned out.** Coupled transforms (layout → mempath → pipeline → slicing) are
sequential because each layer's output is the next layer's input, and half-built coupled candidates cannot be
compared fairly. Transcription is deterministic — parallel arms measure the same layouts and share the same
bug. Splitting the track would fork the checkpoint, break the enabling-step look-ahead, and corrupt
attribution. So tech_lead dispatches the Gluon port as a single `deep_explore` direction (the lane runs it in
a dedicated round anyway) and never pairs it with specialist directions expecting a merge. Decision rule for
anything else: **split only what is independent; keep coupled work whole** — when in doubt, treat it as
coupled; the cost of wrongly parallelizing coupled work is corrupted evidence, not just lost speed.

**Bounded experiments are the deep_engineer's own work.** A timing across shapes, a knob / tile probe, a
Tier-0 / A1–A4 anomaly check — self-contained measurements that do not advance the coupled backbone — are run
by the deep_engineer itself, on its own `GPU_ID` under `gpu_lock.sh`, following
[records.md `## 9. Experiment / hypothesis-test contract`](records.md) and its robustness defaults
(same-session interleave + noise-floor gate per [`benchmark-hygiene.md`](benchmark-hygiene.md); the median
is the reported statistic, min-over-repeats a screening-only extra column; baseline identity — only the
variable-under-test differs, no stored cross-run baseline; the correctness oracle on every cell;
profiler-degraded-mode classification). It reuses the existing timing / harness function verbatim, never
rewrites timing, and returns the [records.md `### Experiment result`](records.md) shape into its own
records. On compile / OOM / GPU-busy / import error it records the exact cause, runs the runnable subset and
continues. A `ship` recommendation from such an experiment is a screening result: **acceptance numbers come
only from `verify_engineer`** (`harness_lib`). A config the experiment shows is mis-tuned is a
`resweep_request`, not something to sweep inside the track.

**One process per patched variant:** a variant that patches the toolchain runs in its own process; there is
no in-process `cache_tag` interleave ([`benchmark-hygiene.md`](benchmark-hygiene.md)).

---

## Round budget

The budget is GEAK's: `budget` counts directions across rounds and a `deep_explore` direction costs
`DEEP_COST` (2). A port needs the G0 launch args (`candidate_floor`, `max_no_improve`, `budget`,
`progress_delta`) so recovery below the comparator is not read as a stall ([`entry.md`](entry.md)). Within
its round the deep_engineer follows its role's stop rule (target reached; ~6–8 iterations each < 1 %; a hard
cap of ~40 measured iterations), and the pack's thresholds govern how it spends that round
([climb.md `Self-monitoring`](climb.md)): warn **70 %** / wrap-up **85 %** / timeout **95 %** of the
iteration budget; stall 10/15/20; ceiling = 3 profiles within 1 %; a regression budget of 2–3 steps, then
back to the checkpoint. Budget left is not a stop ([`close.md`](close.md#hard-stops)).

---

## Records and resumability

GEAK's layout governs: `<exp_root>/team_<kernel>_<ts>/<kernel>/round_N/engineer_i/`, i.e. the
deep_engineer's private `KERNEL_PATH` workspace and its `OUTPUT_DIR`. The pack's ledgers
([`records.md`](records.md)) live inside them:

```text
OUTPUT_DIR/                      # GEAK-owned path, written by the deep_engineer
  worker_result.json  report.md  best_patch.diff   # GEAK's return channel + recovery backstop
  closure_review.md              # the closure self-review, at a ceiling / keep-baseline / negative close
  round log                      # per-round entries (records.md ## 1c), keep/revert with evidence refs
  checkpoint/                    # last confirmed-kept diff + metrics  <- resume anchor
  profile_rN/  ir/  budget/  exp_<id>/              # per-round evidence; bounded experiments
```

- **Two channels.** The StructuredOutput `worker_result` is what the lane harvests and enters tech_lead's
  context; the on-disk files are the full-fidelity record. Keep the return compact (paths + conclusions);
  the reader opens a file only when it needs the detail.
- **Save `best_patch.diff` the moment a new best is set**, not at the end: if the return is lost, the lane
  re-measures whatever patch is on disk, and verify is the source of truth.
- **Resume.** The checkpoint is authoritative for *what was tried* (diff, metrics, round log). When a
  `deep_explore` round ends `partial` and tech_lead issues a follow-on direction, the prior `OUTPUT_DIR`'s
  checkpoint and round log come forward through GEAK's cross-round `INSIGHTS` / the direction prompt; the
  next deep_engineer reloads them instead of re-deriving the track, and coupled work is resumed as the same
  line, never re-split. A cold resume works from the on-disk ledger alone
  ([records.md `## 1d. Residual / Deferred-Task & Resume ledger`](records.md)).

## Isolation + safety

- The deep_engineer works only in its private workspace and `OUTPUT_DIR`; it never edits the harness,
  `COMMANDMENT`, oracle or anything outside `KERNEL_PATH`.
- Do not auto-init git, auto-commit, or run destructive checkout/reset on a user's checkout. Clear only
  run-local artifacts, and move them aside rather than `rm` (GEAK roles must not prompt).
- One process per patched variant: a variant that patches the toolchain never shares a process (or a JIT
  cache) with another variant.

---

## Optional bookkeeping tools

**Not a run mode.** GEAK's phases drive the track; nothing here launches, schedules or gates a role. A
deep_engineer may use these tools to keep its records auditable and its context bounded.

**Discovery.** `cd "$SKILL_ROOT"` (in GEAK `perf_knowledge/expert_skills/skills/gluon_authoring`). `$WORK`
MUST be absolute (e.g. the engineer's `OUTPUT_DIR`) — after the `cd`, a relative `--work` resolves inside the
skill tree and writes the run's artifacts there.

```bash
python3 scripts/toolctl.py stage --work "$WORK" --role deep --json       # stage card + its method section
python3 scripts/toolctl.py describe "$TOOL" --json
python3 scripts/toolctl.py exec --work "$WORK" --role deep "$TOOL" -- --help
python3 scripts/toolctl.py transition --work "$WORK" --role deep --to <next_stage> [--artifact NAME=PATH] [--receipt R]
```

The stage response carries the current stage, allowed tool names, `artifact_refs`, the stage's own
`reference`, and the `standing_references` and `evidence_roles` that apply at every stage. `stage_source:
default` means nothing in `$WORK` named a stage and you are seeing the graph's first stage as a fallback,
not an assignment. `exec` records a receipt (what was requested and the child result); it does not isolate,
authorize or sandbox the child, and long benchmark / profile / build jobs still run under `gpu_lock.sh`.

**Role ids in the record schema.** The stage cards' `role_policy` and the record tools carry upstream role
ids (`deep`, `captain`, `direct_owner`, `bounded_bench`, `skeptic`, …) as schema values. They are not GEAK
roles and nothing spawns them. The deep_engineer passes `--role deep`; final arbitration in GEAK is
Director's.

**Stage cards.** Generated into `runtime/skill-index.json` and one card per stage (`runtime/stages/*.json`);
each names the method section that stage operates:

| # | stage | purpose | stage document |
| --- | --- | --- | --- |
| 1 | `entry` | settle the entry mode (ported champion vs already-explicit incumbent), assert that bundle, pin the comparator | [`entry.md#Stage-Entry: the champion assertion`](entry.md) |
| 2 | `transcribe` | recover the champion layouts into a faithful explicit anchor | [`transcribe.md#Stage-Anchor: the transcription, and what it is allowed to be`](transcribe.md) |
| 3 | `recover` | attribute and repay transcription debt before a win claim | [`recover.md#Stage-Recover: pay the transcription debt, then satisfy the parity gate`](recover.md) |
| 4 | `evidence` | refresh the budget, profile, timing, correctness, and lowered-code evidence | [`profile.md#3.1 Required evidence — the four dials, every round`](profile.md) |
| 5 | `climb` | climb one coupled explicit-tile layer without sweep or branch | [`climb.md#Stage-Climb: one search kind, and it is depth`](climb.md) |
| 6 | `close` | close against `champion_ms` with audits and deferred work recorded | [`close.md#Stop conditions`](close.md) |

**Standing references** (handed to every stage, applying at each one that touches the kernel): **budget**
[`budget.md`](budget.md) · **roofline** [`../hardware/roofline-models.md`](../hardware/roofline-models.md) ·
**profile** [`profile.md`](profile.md) · **hardware**
[`../hardware/capability-matrix.md`](../hardware/capability-matrix.md) · **experience**
[`../pitfalls/negative-patterns.md`](../pitfalls/negative-patterns.md). `scripts/lever_index.py --bound
<class> --arch gfx950` indexes the experience cards by the bound YOUR profile named — experience, not verdict.

**Canonical record.** `scripts/canonical_record.py` (worker result, structure-suspect / resweep requests,
run state, final report), `round_record.py`, `close_audit.py`, `report_lint.py`. If you keep them, write each
producer receipt when its artifact lands ([close.md `## Process provenance: where each producer receipt goes`](close.md)).
They add auditability; they never replace GEAK's return or verify.

### Context lifecycle

**One bounded lease per stage.** Liveness and staleness come from stage events and recorded bytes, never
from elapsed time, a model window, or a cache lifetime.

```text
acquire -> bounded query -> record -> recall -> rotate -> resume
```

- **acquire / rotate / resume** — `toolctl.py stage` opens the lease; rotate settles it and emits a resume
  capsule when the stage or context generation changes; resume verifies the capsule's hashes before
  continuing. Durable state wins over conversational memory whenever they disagree.
- **bounded query** — one routed section per stage via `toolctl.py context query --markdown-heading`. Never a
  full reference tree, transcript, profile dump, or script body. The `file.md#Heading` form in a stage card is
  an address; register the reference once and query it:

  ```bash
  python3 scripts/toolctl.py context acquire --work "$WORK" --role deep \
      --artifact "mref=pack:/references/method/close.md"
  python3 scripts/toolctl.py context query --work "$WORK" --role deep \
      --artifact mref --markdown-heading "Stop conditions"
  python3 scripts/toolctl.py context rotate --work "$WORK" --role deep --json
  python3 scripts/toolctl.py context resume --work "$WORK" --role deep [--capsule CAPSULE] --json
  ```

- **record** — every valid measurement and decision, before moving on. An unrecorded measurement keeps the
  stage open; a long command completes when its receipt says so, not when time passes.
- **recall** — recover prior valid, stale and unfinished obligations before forming the next hypothesis.
  Missing context stays `unknown`.

Normative contracts: `runtime/context-lease.schema.json` (`lease_id`, `run_id`, `generation`, `role`,
`stage`, `policy_ref`, `context_ref`, `context_sha256`, `remaining`, `status`),
`runtime/context-policy.schema.json` (host-independent byte and event budgets — `max_context_bytes`,
`max_capsule_bytes`, `max_query_return_bytes`, `lease`; token telemetry optional and never controls
correctness) and `runtime/resume-capsule.schema.json` (a disposable, bounded rehydration index —
`checkpoint_ref(s)`, `journal_head_ref` + `journal_head_sha256`, `recall_ref`, `open_obligation_refs`,
`next_refs`, `capsule_sha256`). Refusal conditions: `toolctl.py context --help`, not the script body.

---

## Context and cost discipline

Everything in this section changes what a session **costs**, never whether it is **correct**: none of it
may become a gate, a skipped dial, or a reason to close. The deep_engineer is the long-lived session of a
Gluon track (many measure → profile → rewrite iterations), so its context is where the cost accrues.

Measured on a four-kernel campaign: cache-read 39.9 %, cache-write 32.4 %, output including thinking
27.5 %, uncached input 0.2 %. The cost of a session is **turns times prefix**, not the size of any one
read — the largest single tool result was 51 KB while sessions reached 355 K tokens.

- **Nothing is sent once.** A file read or command result is re-sent on every later turn. Prefer a bounded
  query (a heading, a selector, `grep`, `head`) over reading a whole artifact; never `cat` a full
  `worker_result.json`, sweep document, profiler dump or IR listing into the conversation.
- **A command's own text is part of the prefix.** Inlining a heredoc script into a Bash argument was 38 % of
  one role's accumulated context. Write the script to a file once and invoke the file.
- **An idle gap can cost more than the work.** Past the cache lifetime the whole prefix is rewritten; in the
  measured campaign the turn after a five-minute gap wrote 225 K tokens where a normal turn wrote 3 K. Wait
  on long jobs with `scripts/wait_for.sh` rather than idling, and dispose a round's bulk before a long wait.
- **Carry the pointers, drop the traffic.** Keep, verbatim: the asserted comparator and champion (number
  AND SHA), the current best with its evidence ref, open obligations (a requested resweep, an unresolved
  structure suspect, the closure self-review's open items), the budget spent, and **every attempt already
  measured, including every reverted one**, as `round | lever | verdict | delta vs comparator with noise
  band | evidence ref`, each rejection with the reading that killed it — a reverted lever that drops out of
  context gets retried, and a retry is indistinguishable from progress. Drop shell output, profiler dumps,
  IR / ISA listings, compiler logs and superseded intermediate numbers: the evidence ref is the memory, the
  dump is not. Never carry a number that cannot be attributed to a kept line; mark it `unknown` and re-read
  it from the artifact.
- **Experiment memory is the round log, not host memory.** "Do not try X again" written into a host memory
  file outlives the body and layout it was measured against; verdicts live in the round log and GEAK's
  `INSIGHTS`, where a structure change can expire them.
- **Compaction is configured, not automatic.** With no window configured, compaction fires at the model's
  context limit, so a session that peaks well below a 1 M window never compacts. If the host compacts, set
  the window from the role's measured peak (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` on Claude Code) and verify a
  compaction actually happened before relying on it; after one, re-derive what was tried from the round log,
  not from the summary.

---

## See also

- [`index.md`](index.md) — the flow end to end and the stage → method file → tools table.
- [`climb.md`](climb.md) `Self-monitoring` — thresholds, checkpoint.
- [`records.md`](records.md) — the ledger templates; the Experiment contract.
- [`close.md`](close.md) — hard stops, the closure self-review, `final_report` schema.
- `kernel_workflow/roles/{tech_lead,deep_engineer,verify_engineer,director}.md`, `kernel_workflow/kernel_lane.js`.
- `kernel_workflow/scripts/gpu_lock.sh` — per-GPU serialization (`scripts/gpu_lock.sh` shim);
  `scheduler/README.md` — the opt-in broker.

## Sources

Merged into this chapter: upstream `references/orchestration.md` (provider-agnostic model),
`references/claude-code-orchestration.md` (Claude Code binding + cost adapter), `tile-programming-gluon.md`
`## Context lifecycle` and `## Runtime contract`; the upstream `gluon-direction` and `gluon-bench` agent
contracts (method boundary, required return, experiment discipline — now the deep_engineer's); GEAK facts
from `kernel_workflow/kernel_lane.js` (phases, `EXPERT_SKILL_ROLES`, `ENG_SCHEMA`, `DEEP_COST`),
`kernel_workflow/roles/{tech_lead,deep_engineer,verify_engineer,director}.md`,
`kernel_workflow/roles/_fragments/expert_skills.md`, `kernel_workflow/scripts/gpu_lock.sh`;
`runtime/stages/*.json`, `runtime/context-*.schema.json`, `runtime/resume-capsule.schema.json`,
`scripts/toolctl.py --help`, `scripts/pack_facts.json`, `scheduler/README.md`.

Dropped, with reason (this skill adds no roles and no agents; GEAK's phases are the stage machine):

- **The EMBEDDED / STANDALONE run-mode split** (`## Which run mode you are in`, both mode subsections) — one
  mode remains, GEAK's. The pack's `toolctl` stage spine is no longer a way to drive a run; its tools are
  kept as optional bookkeeping (`## Optional bookkeeping tools`).
- **The pack's own role table and agents** (`captain` = `kernel-opt-run`, `deep` = `gluon-direction`,
  `bounded_bench` = `gluon-bench`, `skeptic` = `amd-closure-skeptic`, `arm`) and the per-agent discovery
  and fallback-`SKILL_ROOT` blocks — replaced by the GEAK role table. Their contracts were folded in: the
  direction's method boundary and return → `## The deep_engineer's method boundary` / `### Required return`;
  the bench's experiment discipline → `## Parallelism`; the skeptic → close.md `## Closure challenge`.
- **Agent spawning:** `## When to use a subagent`, `## Mandatory handoff`, `### Delegating an experiment`,
  `## Level → Claude Code mechanism`, `## Max parallel agents` (`max_parallel_subagents`,
  `orchestration.mode: parallel_subagents | sequential_fallback`), "who launches `gluon-bench`", the
  parent-merges rule, BRANCH / best-of-N arm rosters and `branch_request.json` (triton front end only),
  `job_description.md` handoffs, re-spawning a direction agent to resume — GEAK's JS owns parallelism and
  dispatch; the only parallelism left is bounded measurement inside the deep_engineer's loop.
- **Captain / fleet supervision:** captain arbitration and close, the fleet compaction-recall hook
  (`compact_recall_hook.py`, `SessionStart` hook JSON), supervisor-vs-direction compact lists — final
  arbitration is Director's; the direction's keep-list survives under `## Context and cost discipline`.
- **`.claude` host install and subagent permissions:** subagent model / effort override env vars
  (`CLAUDE_CODE_SUBAGENT_MODEL[_FORCE]`, `CLAUDE_CODE_EFFORT_LEVEL`), gateway model-availability probe and
  pinned model ids, spawn-depth and session caps, agent frontmatter `tools:` enforcement, project
  allow-lists, `maxTurns`, `agent-<id>.jsonl` resume, the campaign `CLAUDE.md` compact-instructions
  template, the three-memory-stores table and `CLAUDE_CODE_DISABLE_AUTO_MEMORY` — host configuration of a
  separate agent runtime, not of GEAK's roles. The underlying rules (keep-list, memory is not the record,
  compaction must be configured) are kept in short form.
- **`## Contract additions`** (`gpu_devices`, `max_parallel_subagents`, `work_root`) — GEAK allocates the
  GPU and workspace (`GPU_ID`, `KERNEL_PATH`, `OUTPUT_DIR`); the `work_root/dir_<id>/` layout is replaced by
  GEAK's `round_N/engineer_i/`.
- Earlier merge notes kept in effect: `gpu_lock.sh` is GEAK's (`flock -w 1200`, pool wait 1200 s), never
  inline `HIP_VISIBLE_DEVICES`; min-over-repeats is a screening statistic and acceptance numbers are
  `harness_lib` medians; the broker is opt-in (`GEAK_GPU_BROKER=1`, autostart off, lock dir
  `/tmp/team_gpu_locks`); no `.geak_gpu_group` marker or `kernel_fleet.js` is claimed; hard stops are owned
  by [`close.md`](close.md#hard-stops).
