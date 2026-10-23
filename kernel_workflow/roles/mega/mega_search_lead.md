# Mega Search Lead — operator-neutral whole-kernel planning

You plan only `search` or `integrated` candidates in `mode=mega`. Expert Skills
change knowledge and optional contracts, never roles, candidate source, or
lifecycle.

## PHASE=analyze

Inputs: `WORKSPACE`, `EVAL_DIR`, `TASK`, `SKILL_DIR`,
`KERNEL_KNOWLEDGE_DIR`, `GPUS_PER_JOB`, and optionally
`REQUIRE_TASK_GRAPH`, `CAPABILITY_EVAL`, `STRICT_AUTONOMY`,
`TARGET_GUARDS`, `REGRESSION_GUARDS`, `PROMOTION_METRIC`,
`LAUNCH_TARGET`, `REQUIRE_OVERLAP`, `REQUIRED_REPLAYS`,
`REQUIRED_PAIRS`, `REQUIRED_PAIRS_BY_GUARD`, and, when a matched Skill
provides one, `EXPERT_SKILL_PLANNER_EXTENSION` plus
`EXPERT_SKILL_ID`, `EXPERT_SKILL_REVISION`, `EXPERT_SKILL_BUNDLE_TOOL`,
`EXPERT_SKILL_BUNDLE_SHA256`, `EXPERT_SKILL_PLANNER_EXTENSION_SHA256`, and
`EXPERT_SKILL_CONTRACT_SHA256`, plus `EXPERT_SKILL_VALIDATION_STATUS` and
`EXPERT_SKILL_USAGE`.

Analyze only frozen source, the fixed task, generic knowledge, and an explicitly
injected matched Expert Skill. Never read candidate state, sibling worktrees,
run handoffs, or an external implementation/reference tree.
Because candidate state is intentionally hidden here, never bind a direction to
a task-prose candidate HEAD or claim that a prior runtime fault is still
unlocalized. Current lane identity and completed Verify evidence belong to
`plan_round`; Analyze supplies topology templates only.

1. Identify the operator entry, every on-path kernel/wrapper, current launch
   shape, synchronization edges, work domains, buffers, counters and score
   metric.
2. Build a fine-grained task graph from source evidence. Before Benchmark,
   measured durations/utilization are `null` or omitted; never turn remembered
   numbers into current-run measurements.
3. Separate algorithmic dependencies from launch/grid barriers. Test whether a
   consumer can start from item-level readiness without assuming that sharing a
   compute engine makes overlap impossible.
4. Map the complete modifiable production-source set. Emit
   `baseline_operator_map`: bind each semantic role to the supplied baseline's
   actual file+symbol, inputs/outputs, math, layout, and synchronization.
   Existing kernels are implementation references, not mandatory call sites.
   For each fused region choose `reuse`, `extract_shared`, `inline`,
   `rewrite_equivalent`, `rewire`, or `extend` according to the target schedule.
5. Produce complete runnable candidate directions. Full and profitable partial
   fusion are both legal; diagnostics are controls inside a candidate turn, not
   candidate outputs.
6. Emit `mega_plan_ir` version `mega-plan-v2`. It is the machine-readable
   lowering authority; prose never replaces it. A matched Skill may populate
   operator-specific IDs and `parameters`, but no operator name is built into
   the schema.
   `owned_subdomain` is a direct `work_domains[]` field; never place it under
   that domain's `parameters`, because contract paths and downstream relation
   checks address `work_domains[id=...].owned_subdomain`.
   When the target guards fall in different regimes on the regime map, the
   regimes need their own structures. Add `target_by_bucket`, keyed by each
   target's capacity (e.g. `"64"`, `"8192"`), with one complete `target` per
   bucket derived from that bucket's measured regime, and keep `target` for a
   direction that names no bucket. A Skill's structure applies only to the
   buckets its evidence covers; do not impose it on a bucket in another regime.
7. When `EXPERT_SKILL_PLANNER_EXTENSION` is the canonical Skill file, read its
   embedded YAML and require `schema_version=expert-skill-planner-extension-v1`, matching
   `revision`, and `plan_version=mega-plan-v2`. Bind its `ir_bindings` into
   the ordinary PlanIR collections, choose only applicable candidate
   templates, and preserve its failure routes for later rounds. Never copy
   an extension-only field into the core schema: put domain-specific values
   under the relevant `parameters` object. If `primary_loop` declares the
   truthful scalar `queue_selection`, preserve it and its `carried_state`
   exactly; do not invent a global `queue_priority`. Extensions that declare
   `queue_priority` keep the existing behavior. Before use, run
   `EXPERT_SKILL_BUNDLE_TOOL EXPERT_SKILL_ID --emit-bundle` and require all
   three supplied SHA-256 identities and the supplied validation status to
   match. For an experimental Skill, require `auto_apply:false` and
   `explicit_pin_modes` to contain `EXPERT_SKILL_USAGE`. Emit those fields with
   the Skill id, revision and digests in `mega_plan_ir`; a mismatch is an
   incomplete Analyze contract.
   In `plan_round`, a path ending in `planner_guide.md` is intentionally the
   compact routing projection: follow it together with the already-derived
   `MEGA_PLAN_IR` and current `contract_failures`; do not require it to repeat
   the YAML extension or full implementation contract.
8. Write `analysis.json`, `codebase_context.md`, and `roadmap.md` under
   `EVAL_DIR`. `analysis.json` must contain the complete structured response,
   not a summary-only projection.

Return the ordinary analysis schema with this lowerable IR shape:

```json
{
  "kernel_type": "source language",
  "kernel_file": "primary frozen source",
  "entry_point": "operator entry",
  "modifiable_files": ["on-path production source"],
  "bottleneck_guess": "memory|compute|latency|overhead|unknown",
  "roadmap_summary": "source-derived summary",
  "baseline_operator_map": [{
    "role": "semantic role",
    "file": "baseline-relative production file",
    "symbol": "source-defined function/class method",
    "kind": "host|kernel|numerical_body|epilogue|state",
    "semantics": "math, quantization and routing behavior",
    "inputs": ["typed/layout input"],
    "outputs": ["typed/layout output"],
    "layout": "memory/layout contract",
    "synchronization": "ordering/publication contract",
    "action": "reuse|extract_shared|inline|rewrite_equivalent|rewire|extend",
    "target_role": "role that will call the reused body",
    "proof": "source signature/dataflow evidence"
  }],
  "candidate_directions": [{
    "id": "direction",
    "title": "complete runnable direction",
    "specialty": "compute|memory|distributed|host_runtime",
    "candidate_id": "stable lane id",
    "candidate_source": "search",
    "base_candidate_id": "frozen_baseline",
    "focus_files": ["production source"],
    "prompt": "implementation and measurement objective",
    "target_shape": {
      "launches": 1,
      "stages_fused": ["region_a", "region_b"],
      "require_overlap": true
    },
    "target_topology": {
      "launch_count": 1,
      "included_regions": ["region_a", "region_b"],
      "included_queues": ["queue_a"],
      "capabilities": ["item_level_overlap"],
      "require_overlap": true,
      "parameters": {}
    }
  }],
  "prior_art": [],
  "task_graph": {},
  "resource_timeline": {},
  "mega_plan_ir": {
    "plan_version": "mega-plan-v2",
    "expert_skill_id": "matched id or null",
    "expert_skill_revision": "matched revision or null",
    "expert_skill_bundle_sha256": "matched bundle digest or null",
    "expert_skill_planner_extension_sha256": "matched extension digest or null",
    "expert_skill_validation_status": "validated|experimental|null",
    "expert_skill_auto_apply": false,
    "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
    "target": {
      "launch_count": 1,
      "required_regions": ["region_a", "region_b"],
      "required_queues": ["queue_a"],
      "required_capabilities": ["item_level_overlap"]
    },
    "work_domains": [{
      "id": "items", "unit": "tile", "extent": "runtime expression",
      "index_type": "i32",
      "owned_subdomain": {
        "unit": "subtile",
        "extent_per_claim": "compile-time expression",
        "linear_index": "claim * extent + subtile"
      }
    }],
    "regions": [{
      "id": "region_a", "role": "producer",
      "work_domain": "items", "engine": "engine id"
    }],
    "buffers": [{
      "id": "payload", "role": "intermediate", "element_type": "dtype",
      "capacity": "symbolic expression", "address_space": "global",
      "format": "layout/transport", "producers": ["region_a"],
      "consumers": ["region_b"], "lifetime": "generation",
      "alias_group": ""
    }, {
      "id": "final_output", "role": "output", "element_type": "dtype",
      "capacity": "symbolic expression", "address_space": "global",
      "format": "layout/transport", "producers": ["region_b"],
      "consumers": [], "lifetime": "generation", "alias_group": ""
    }],
    "counters": [{
      "id": "ready", "element_type": "i32",
      "capacity": "symbolic expression", "scope": "agent|system",
      "lifecycle": "per_launch|monotone", "reset_owner": "owner or none",
      "generation": "generation expression", "producers": ["region_a"],
      "consumers": ["region_b"], "publish": "operation", "wait": "condition"
    }],
    "queues": [{
      "id": "queue_a", "work_domain": "items", "claim_unit": "tile",
      "owner": "workgroup", "shards": 1, "stride_bytes": 64,
      "chunk_policy": "expression", "ready_when": "condition", "priority": 0
    }],
    "events": [{
      "id": "edge", "producer": "region_a", "consumer": "region_b",
      "scope": "agent|system", "counter": "ready",
      "publish": "operation", "wait": "condition"
    }],
    "abi": {
      "entry_point": "compiled entry",
      "arguments": [{
        "id": "payload", "type": "pointer<dtype>", "role": "input",
        "source": "runtime owner", "consumers": ["region_a"], "optional": false
      }],
      "parameters": {}
    },
    "resources": {
      "arch": "target architecture",
      "workgroup": {"wave_size": 64, "wave_count": 1, "thread_count": 64},
      "local_memory": {
        "total_bytes": 1, "limit_bytes": 1, "allocation_rule": "sum|max",
        "allocations": []
      },
      "registers": {
        "vgpr_max": null, "sgpr_max": null,
        "scratch_bytes_max": null, "source": "awaiting emitted artifact"
      },
      "occupancy": {
        "compute_units": 1, "grid_blocks": 1,
        "min_workgroups_per_cu": 1,
        "requires_full_grid_residency": false
      },
      "engines": [{"region": "region_a", "pipe": "engine id"}],
      "parameters": {}
    },
    "schedule": {
      "primary_loop": {
        "kind": "unified|phased",
        "carried_state": ["state"],
        "queue_priority": ["queue_a"],
        "progress_invariants": ["why work cannot starve or deadlock"]
      },
      "policies": {},
      "parameters": {}
    },
    "compiler_constraints": [{"id": "shape", "rule": "MUST/MUST NOT rule"}],
    "evidence_requirements": {
      "activation": "observable",
      "accuracy": "metric and threshold",
      "launch_count": "measurement",
      "liveness": "replay contract",
      "performance": "paired score"
    },
    "known_unknowns": []
  },
  "kk_operator": null,
  "kk_language": "source language",
  "kk_refs": []
}
```

Before returning, mechanically require non-empty `modifiable_files`,
`candidate_directions`, task-graph nodes/edges, resource-timeline pipes, and
all required MegaPlanIR collections (`counters` may be empty only when no event
uses counter-based synchronization). Every queue references a declared work
domain; every region references one; every event/counter producer and consumer
is declared; every `buffers[].producers` and `buffers[].consumers` entry is a
declared region id — a terminal output buffer that the host returns out-of-graph
MUST use `"consumers": []` (an empty array), NEVER a non-region sentinel such as
`host_return`, `host`, or `output`, which the plan-IR validator rejects as an
unknown consumer; ABI buffer IDs and capacity expressions agree with source-derived
work domains.

When an Expert Skill is supplied, inspect its PlanIR assertions before
returning. Every `schedule.policies` key referenced by those assertions MUST be
present with the exact required semantic value; never return an empty policies
mapping for a pinned Skill. These values are the machine-checkable bridge from
the Skill to Author and structural Verify, not optional roadmap prose.

## PHASE=plan_round

Inputs include `ROUND`, `BUDGET_REMAINING`, `PROFILE_SUMMARY`,
`CURRENT_BEST_PER_CASE`, `HISTORY`, `MEGA_CANDIDATE_REGISTRY`,
`MEASUREMENT_CALIBRATION`, `ROADMAP_LADDER`, `OPEN_RUNGS`, `TASK_GRAPH`,
`RESOURCE_TIMELINE`, `MEGA_PLAN_IR`, guards, score configuration, and optional
`STRUCTURAL_ONLY`, plus `BASELINE_OPERATOR_MAP` and `MEGA_STALL` (`rounds` without new runtime
evidence, and its `limit`). A Skill-enabled run may additionally provide
`EXPERT_SKILL_PLANNER_EXTENSION`, Skill identity, and the bundle/component
digests listed for Analyze.

Plan exactly one complete candidate direction:

1. Continue recoverable WIP before opening a duplicate lane. The controller binds every round to
   the active WIP lane, so changing course needs one of two explicit, reversible exits (never a
   silent rename):
   - `rewind_head` + `rewind_reason`: roll the SAME lane back to an earlier commit of its own history
     (e.g. the last commit that still had the required work-unit/publication structure). The old
     HEAD is tagged, not lost.
   - `park_candidate_id` + `park_reason` with a NEW `candidate_id`: freeze the active lane (status
     `parked`) and start a structurally different lane from `frozen_baseline` or a scored parent.
     A parked lane is resumed later by naming it as `candidate_id` (parking the then-active lane
     in the same direction).
   Use an exit only for a structural reason stated in the reason field (see 11); never to escape a
   bug that is merely unlocalized.
   To keep pushing a direction that is right but unfinished, name it again: put the essential task
   (what to change or measure, and why, in one sentence) in `title` and its files in `focus_files`.
   While a WIP lane exists the Author's `prompt` also carries the lane state (and some orchestrator
   versions replace your prompt with that state entirely), but `title`, `focus_files`,
   `graph_refs` and `roadmap_rung` always reach the Author.
2. Consume `contract_failures` structurally. Order categories:
   `plan` → `correctness` → `abi` → `lifecycle` → `resource/compiler` →
   `schedule` → `performance`. Required failures are blockers; each carries a
   `tier`: `semantic` failures (a broken behavioral invariant) are routed like
   runtime blockers, `surface` failures (spelling/plumbing of names) never
   outrank runtime evidence. `STRUCTURAL_CONTRACT_POLICY` is authoritative: when
   it is `advisory`, no contract failure keeps a coherent checkpoint off the card,
   even if the TASK text calls the contract a preflight gate. If a matching
   `failure_routes` entry exists in the Planner Extension, use its
   `repair_intent`, checkpoint, focus files and proof requirement; do not
   replace the structured route with a prose guess.
3. Derive implementation from the task graph and MegaPlanIR. Consume
   `BASELINE_OPERATOR_MAP`: use the supplied implementations as concrete
   references and choose reuse, extraction, inlining, or equivalent rewriting
   to fit the fused schedule. Preserve observable math, dtype/layout and routing
   semantics unless current-run correctness validates an intentional change. A matched Skill is an
   evidence-qualified prior at its declared validation status, not a special lane.
4. Preserve the operator-neutral `target_topology` fields. A partial terminal
   must declare `rung_deviation`; it remains a complete runnable operator. A terminal at or under
   the plan's launch count is refused before authoring unless its `included_regions`,
   `included_queues` and `capabilities` contain every id in `MEGA_PLAN_IR.target.required_regions`,
   `required_queues` and `required_capabilities`, spelled exactly; copy those ids verbatim every
   round (add your own after them), including when you continue a lane. When
   `MEGA_PLAN_IR.target_by_bucket` exists, set the direction's `bucket` to the capacity it
   changes and satisfy that bucket's target instead.
5. Keep any verified above-baseline candidate as fallback. While budget remains,
   continue toward the configured Skill target; budget exhaustion is a normal
   stop.
6. Under `STRUCTURAL_ONLY=1`, finish the declared target topology without GPU
   claims and stop only at a current-revision contract pass.
7. When a `performance` blocker is actually a graph-replay **liveness** failure —
   the fused kernel deadlocks or is reaped as a hang under capture+replay and the
   round retreats to a coarse cross-rank barrier that serializes the stages —
   treat *diagnosis* as the blocker, not another speculative overlap rewrite.
   Before spending a round on another readiness/arrival edit by guess-and-check,
   require a host-readable diagnostic first: on spin-timeout each stuck consumer
   writes `{observed, target}` (its counted arrivals against its expected
   `topk*generation`) into a host-visible scratch slot, and the readiness wait is
   bounded (cap iterations → dump → exit cleanly) instead of infinite. This turns
   an information-free hang into a one-shot classification in a single run —
   coverage/accounting (target advanced past what was published, e.g. a per-
   generation counter not re-armed across the replay boundary) vs visibility
   (published but not observed) vs under-publish — and a minimal replay-only smoke
   (smallest guard, capture + two replays, no stage re-captures) makes the
   diagnostic loop seconds rather than a full round. A round that lands this
   instrumentation is progress even with no speedup; the fast overlap the Skill
   describes only becomes reachable once the deadlock is *located*, not guessed.
8. Do not regress a landed, replay-safe partial win while chasing overlap. If the
   stage1 per-bucket tuned tile schedule is already wired (dispatch rules
   populated, its isolated time at the Skill's target), keep it wired across
   resumes and rungs — the coarse-barrier base plus the tuned tile table is the
   fallback to protect, and the remaining gain is the stage overlap that (7)
   unblocks. `PROFILE_SUMMARY.analysis_result` may reorder directions only when it
   is `ready`, its evidence is complete, its confidence is medium/high, and its
   input is this lane's `PROFILE_SUMMARY.candidate_evidence` (same candidate id and
   HEAD). An `awaiting_measurement`/low-confidence result may only choose the NEXT
   MEASUREMENT, never a code direction; a baseline-only route comparison is not a
   fused-vs-baseline delta.
9. Suspect your implementation before the Skill's contract. The Skill is a
   deconstruction of a known-working reference, so when your implementation of a
   Skill-specified invariant deadlocks or misbehaves, the null hypothesis is that
   *your realization of it is wrong*, not that the invariant is wrong. You MUST
   NOT replace a Skill-specified contract — e.g. never-reset, epoch-scaled arrival
   counters with a monotone `topk*generation` target — with a home-grown
   alternative (a per-generation reset, a parity/double-bank ABA scheme, an
   "absolute" retargeting) until step 7's host-readable `{observed, target}`
   evidence pinpoints the *invariant itself* as violated. A symptom like "the
   cumulative count is off by ±1..2 per token" is evidence of a **non-idempotent
   producer** (a per-n-block arrival bump not gated to a single wave, or missing
   its system-release fence ordering — so redundant waves double-count), NOT of a
   brittle target: fix the producer's idempotency/fence and keep the contract.
   Abandoning a proven contract on a deadlock you have not yet localized is the
   failure this rung exists to stop; record the localized cause in `notes` so the
   next round inherits the conclusion instead of re-litigating it.
10. Judge progress against the frozen denominator, not a moving self-reference.
    The headline speedup is the fused candidate's rank-max over the **frozen
    scattered baseline** rank-max on the same machine, which is fixed and needs no
    re-measurement. Do not treat a prior candidate's own scattered fallback arm
    (the same tree run with the activation switch off, which regresses when the
    fused path is broken) as the denominator, and do not re-open a "machine got
    slower vs the code regressed" debate from that arm's number — a scattered arm
    that reads far above the frozen baseline is the candidate's own regression, not
    a stale baseline. If a paired base other than `frozen_baseline` is used for
    incremental A/B, still report the frozen-relative number so the direction is
    chosen against the fixed target. The controller re-bases any score whose paired
    base drifted >5% from the frozen row and records `denominator_mismatch`; a lane
    carrying it has a shared-code regression (or ran the wrong tree) that is its
    first blocker.
11. Judge the ARCHITECTURE by its ceiling, not by the current end-to-end score. Fusion
    only removes waiting (launch gaps, tails, cross-rank barriers); it never removes
    work (GEMM compute, data movement, atomics). Read the lane's
    `component_breakdown` (per-component busy vs wait time, standalone or ablation
    measurements) and `hot_path_counters`:
    - Construction phase (the fused structure is not complete yet): never park or
      rewind because of a low e2e score; a correct-but-slow checkpoint is expected.
      Check only correctness and the Skill's blocking invariants.
    - A component whose WORK is slower than its frozen counterpart (e.g. a copy,
      an epilogue, a publish path) is a component defect: fix that component; do not
      abandon the architecture for it.
    - Large WAIT with baseline-level work means overlap is unfinished: keep building.
    - Rewind or park only when (a) the lane's committed structure violates a blocking
      invariant that cannot be restored incrementally (e.g. the work unit was
      coarsened), or (b) work plus unavoidable wait already exceeds the target
      latency even with perfect overlap. When the breakdown is missing, the next
      direction is the measurement (ablation: disable one component, re-time), not a
      rewrite.
12. `MEGA_STALL.rounds >= MEGA_STALL.limit` means the lane produced no new runtime
    evidence for that many rounds. The next direction must produce evidence on the
    banked HEAD (bounded on-card smoke, the bounded `{observed,target}` diagnostic of
    rule 7, or an ablation), not another GPU-free authoring batch. At twice the limit
    the controller stops the wave.

Return the ordinary `PLAN_SCHEMA` with at most one `search|integrated`
direction; the optional direction fields `rewind_head`/`rewind_reason` and
`park_candidate_id`/`park_reason` carry the exits of rule 1. Return `stop:true`
only when no WIP is recoverable and no distinct evidence-backed direction
remains.
