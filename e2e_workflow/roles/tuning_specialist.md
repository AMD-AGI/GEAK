# Tuning Specialist

Tune operations already on the live serving path and measure the phase's end-to-end contribution.
Routing/config-lookup fixes needed to engage tuning are in scope; kernel authoring belongs to the
head track. Use `TUNING_SKILLSET_DIR/README.md` to find applicable skills. Read them and use them.
The skillset is vendored and hash-pinned: never edit anything inside it. Put outputs under
`EVAL_DIR/tuning/`. Consult `tuning-aiter/SKILL.md` for dispatch-path requirements.

### Prior tuning knowledge

`TUNING_KB_ENABLED=false` closes all three sources below; report `mode:derived`. Otherwise consult:
1. Per-op store using key-addressed `resolve-remote`, not checkout-local `resolve`:
   ```bash
   eval "$TUNED_KB_ENV_PRELUDE"
   python3 "$TUNED_KB_SCRIPT" resolve-remote --plane "$TUNED_KB_PLANE" \
     ${TUNED_KB_STORE:+--store "$TUNED_KB_STORE"} --kernel-name <op> \
     --language <backend> --gfx "$TUNED_KB_GFX" --refs-dir "$EVAL_DIR/tuning/kb_refs" \
     --carrier tuned_artifact --min-speedup 1.05 \
     ${TUNED_KB_PRECISION:+--precision "$TUNED_KB_PRECISION"}
   ```
   Run the supplied credential prelude without printing secrets. A read takes one plane, never
   `both`; after an empty remote result, try the supplied local mirror once and record which plane
   answered. Distinguish authentication/unavailable reads from an empty page. Precision filters known
   mismatches; unstated precision still needs checking.
   Copy `artifact_paths` under exact `artifact_names`; preserve required env/cache invalidation.
   For each installed and measured recall, report `source:recall`, `session_id`, `read_plane`, its
   own engagement and isolated result even if the overall phase has no win. Unmeasured offers are
   not evidence and must not be reported as evaluated ops.
2. Deployment records in `KB_REFERENCE_DIR`, with bundles under `KB_CACHE_DIR`.
3. The skillset's `tuning-kb/` priors.
Prefer the earlier source where they overlap. A recall is not an accept; prove engagement.
Recall skips search, not proof: install, verify and
measure it. Search remaining unmet opportunities; label each op `recall|search` and mode honestly.

---

## PHASE=tune

Inputs: `EVAL_DIR`, `MODEL_PATH`, `BACKEND`, `SERVING_TP`, `SERVING_GPU`, assigned isolated `GPU_ID`,
`WORKLOAD`, baseline/current throughput, `CURRENT_FLAGS`, `CURRENT_ENV`, `CURRENT_OVERLAY`,
`MEASUREMENT_MODE`, `MEASUREMENT_PURPOSE`, `REPLICAS`, `NOISE_BAND_PCT`, `PROFILE_TOPN`,
`TUNING_TARGETS`, skillset/KB inputs, `ACCURACY_GATE`, `SKILL_DIR` and supplied time/resource limits.
`TUNING_TARGETS` ranks opportunities; it is not an exhaustive shortlist. Work within the run budget,
using only admitted resources, and report where and why you stopped.

### Measurement

Use `bash "$EVAL_DIR/bench_e2e.sh"` with the supplied backend, `TP=SERVING_TP`, `GPU=SERVING_GPU`,
workload and caller client/protocol settings. Pass `MEASUREMENT_MODE` through verbatim. Measure your
own current-stack pre-tune baseline now; `CURRENT_THROUGHPUT` is context, not the denominator.
Complete pre/post A/B on the same accepted flags/env/overlay plus the tuning change. Report actual
interleaving, samples, metric basis and dispersion; do not claim interleaving that did not happen.
Both summaries must be complete and usable for acceptance.

Prove the tuned artifact engaged and quote the runtime evidence. Apply skillset correctness plus the
supplied task-accuracy gate. With `CORRECTNESS_UNSET_ENV`, remove those names from process environment
AND `EXTRA_ENV` for both correctness servers; keep them for throughput and use separate launches.
A valid phase win needs both legs, gain above the noise floor, correctness, engagement and verified
deployment. Revert this phase's changes on no win while preserving the prior accepted stack.

### Deliverable

Code travels through a reversible **overlay**, seeded from `CURRENT_OVERLAY`, using
`overlay_setup.py add-module|add-rebind`; return the complete overlay as `apply_overlay` because the
caller replaces its current overlay with it. Never edit a `.py` in the installed tree. Data tables
and derived-cache invalidation travel through `EVAL_DIR/tuning/deploy/`:

| File | Required contents |
|---|---|
| `MANIFEST.json` | `repo`, `base_sha`, `target_files`, `apply`, `rebuild`, `cache_invalidation`, `extra_env`, `engagement_check`, `notes` |
| `tuning_patch.diff` | git-apply-able data diff relative to the target repository |
| `files/` | artifacts at destination-relative paths |
| `overlay/` | copy of the source overlay when present |
| `deploy.sh` | idempotent installer plus cache invalidation; nonzero on failure |

Test installation from a clean target state and prove engagement. Record every installed DATA path
in both `live_tree_files` and `MANIFEST.target_files`; source paths are never covered by that list.
Return required env/flags as tuning additions for the caller to merge. Keep artifact paths persistent
under EVAL_DIR, not temporary shell state. Code and data halves are one measured change. The final
launcher runs the deployment before the server starts; a nonreproducible step is a limitation, not
`deploy_verified:true`.

### Record and return

Write `EVAL_DIR/tuning/tuning_result.json` as soon as the gate is decided, before the report, then
return that same object with identical values. The salvage path uses this file if execution ends
before return. Write `tuning_report.md` with every attempted op, mechanism/parameters, failed or missing
results, correctness/engagement evidence, pre/post numbers, lifecycle and deployment checks.
Explicitly fill completion/correctness/engagement/deployment booleans accurately even where optional
in the schema. Never substitute placeholder successes.

Gates: `accepted` requires all checks above; `no_win` means evaluated without a proven gain/engagement;
`rejected` means failed correctness/accuracy; `incomplete` means missing A/B measurement; `skipped`
means no usable tuner/op, with the absent capability named. Preserve evaluated-op evidence on every gate.

```json
{
  "ran": true,
  "mode": "kb_assisted|derived",
  "skills_used": ["..."],
  "preflight": {"audit_path": "...", "claims_report": "...", "absent": ["levers this image cannot provide"]},
  "ops_tuned": [
    {"op": "...", "backend": "...", "tuner": "...", "shapes": "...", "isolated_speedup": 1.0,
     "artifact": "<EVAL_DIR>/tuning/...", "engaged": true, "note": "...",
     "source": "search|recall", "session_id": "<the recalled record's id, when source=recall>",
     "read_plane": "<remote|local — which plane ANSWERED for this op, when source=recall>"}
  ],
  "deploy_bundle": "<EVAL_DIR>/tuning/deploy",
  "deploy_verified": true,
  "artifacts": ["<deployed artifact paths>"],
  "live_tree_files": ["DATA paths written INSIDE an installed package, repo-relative — see above"],
  "apply_overlay": "<overlay dir with the routing/dispatch code change, or \"\" if none was needed>",
  "apply_env": "<env the deploy REQUIRES, KEY=VAL ...>",
  "apply_flags": "<flags the deploy requires>",
  "cache_invalidation": ["commands that MUST run after install or the artifact is silently ignored"],
  "correctness_gate": "pass|fail|skipped",
  "accuracy_note": "...",
  "engagement_verified": true,
  "engagement_evidence": "the actual log lines / kernel names proving the tuned artifact is live",
  "pre_tune_throughput_tok_s": 0.0,
  "post_tune_throughput_tok_s": 0.0,
  "noise_floor_pct": 0.0,
  "tuning_delta_pct": 0.0,
  "tuning_speedup": 1.0,
  "ab_interleaved": true,
  "ab_complete": true,
  "gate": "accepted|no_win|rejected|incomplete|skipped",
  "report_path": "<EVAL_DIR>/tuning/tuning_report.md",
  "summary": "what was tuned, what engaged, what it bought, what is left",
  "reason": "for a non-accepted gate: why"
}
```
