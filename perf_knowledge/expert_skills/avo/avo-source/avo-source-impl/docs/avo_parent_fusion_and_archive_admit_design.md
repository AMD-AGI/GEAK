# AVO lineage keep + parent-direction port — Design (for review)

**Status:** design only. No code in this drop. Implementation later on the
fork `/shared_nfs/liuji/avo_proj/PerfSkills-avo-opt` (copy of `PerfSkills-avo`
when coding starts). Default-off flags so trunk behavior is unchanged until a
handoff opts in.

**Review scope (this version):**

| ID | Feature | This review |
|----|---------|-------------|
| **F0** | Same parent → keep one; different parents that **clear the global champion gate** → **all enter the DAG**; unique `best_id` | **In scope — implement first** |
| **F1** | Off-path **lever review** (deterministic) then absorb that direction on champion (no git apply) | In scope as **second** feature; §4 is the review surface |
| **F2** | Archive nodes that did **not** beat global best (0.98× near-misses) | **Out of scope. Do not implement.** Appendix A only. |

F0 is safer than F2: every extra tag already passed `passesCommitGate(ver,
cumulative)` vs the **round-start global best**. No new numeric threshold,
no “almost as good” clones. New tags stay **speedup-monotone in `avo-vN`**
by committing also_rans first (still > old best) and the unique champion last.

Companions: [`avo_start_points_design.md`](avo_start_points_design.md),
[`avo_on_team.md`](avo_on_team.md), `lib/commit_gate.mjs`.

---

## 0. Review summary (read this first)

Today a team round with 8 starts × 8 directions may produce **several**
workers that independently beat the global best, from **different parents**.
`commitRoundResult` commits **only `round.best`** and drops the rest.

That is the wrong compression:

- **Same parent, several directions:** keeping only the best child is right
  (near-duplicate edits on one local optimum).
- **Different parents, each beating global best:** dropping all but the
  max throws away independently verified, globally improving mechanisms.
  Those nodes are the natural donors for F1.

F0 does **not** create multiple champions. It creates **one champion** plus
**sibling tags** that also beat the *old* best, parented on their own
`startRef`. `cumulative` / `best_id` / reprofile follow only the max.

F2 (keep 5.82× when best is 5.88×) is deferred: it needs a new alpha gate
and changes `avo_lineage.py`’s `geomean < best` refusal. F0 does not.

---

## 1. Problem

### 1.1 Diverse starts already exist; the DAG forgets extra winners

`start_policy=diverse` starts workers from different committed tags. The
commit gate is vs **global** `cumulative`. Integrator only merges same-base,
different-regime passers (almost never on these kernels). Then a single
`commitCandidate(round.best)` runs and **returns**.

MiniMax prefill tree: PACK_Q lives on `v4←v1`; champion path is
`v0→v1→v2→v3→v8→v9`. Extra global winners from other parents in the same
round would have been discarded the same way.

### 1.2 Invariants we will not change

- Unique champion: only the round’s highest gated geomean updates
  `best_id`, `cumulative`, `bestVersion`, `reprofileBest`.
- `passesCommitGate` vs global best is **not** relaxed.
- COMMANDMENT / independent verify / `assertCommittedParent` unchanged.
- Integrator path unchanged.

---

## 2. Goals / non-goals

**Goals**

1. **F0 — multi-parent keep:** among workers that pass the champion gate
   vs **round-start** `cumulative`:
   - group by `startRef`;
   - keep **one** (highest `verified_geomean`) per parent;
   - commit **all** those group-winners;
   - only the global max is `role=champion`; others `role=also_ran`.
2. **F1 — direction port (after F0):** bounded slot to re-implement a
   donor’s *strategy* on the current champion source, then verify.
   Default trigger: champion **stall** (`noImprove >= 3`), **before**
   the existing supervisor REDIRECT at `noImprove >= 5` (§4.8.1).

**Non-goals (this review)**

- F2 near-best archive (`α × cumulative` without beating global best).
- `git apply` of a donor patch onto champion.
- Replacing Integrator **or** the LLM supervisor / ESCALATE.
- Multiple `best_id`s.

---

## 3. Feature F0 — Keep every parent’s best global winner

### 3.1 Rule

Let `G` = `cumulative` at the **start** of `commitRoundResult` (old
champion). Let `P` = `{ workers | passesCommitGate(ver, G) }`.

```
byParent = group P by startRef
picks = { argmax(speedup) in each group }     # one per father
champion = argmax(speedup) among picks
also_ran = picks \ { champion }

commit champion  → updates cumulative, HEAD, reprofile
for each also_ran (speedup desc):
  commit also_ran → tag + lineage, do NOT update cumulative
  checkout HEAD back to champion
```

Empty `P` → no commit (same as today).

Integrator: if a merge **commits as champion**, F0 also-rans are still the
**variation** group-winners from `P` that are not that merge. If the merge
**fails** the extra gate, fall back to F0 on `P` as above (today’s fallback
is only `round.best`; F0 replaces that fallback).

### 3.2 Why this is conservative

| Candidate | F0 |
|-----------|-----|
| From v9, 5.90×, beats G=5.876 | group(v9) winner; may be champion |
| From v9, 5.88×, beats G | **dropped** (same father, worse than 5.90) |
| From v8, 5.89×, beats G | **also_ran** commit, parent v8 |
| From v4, 5.10×, fails gate | **not in P** — gone (F2 would have discussed this; we do not) |
| From v9, 5.85×, fails gate | **not in P** |

Every extra tag is already a **global improvement** over the previous
champion. Director pre-commit: **yes for champion**; for also_ran, **yes
in v1**. If GPU cost hurts, later `also_ran_director=false`; default on.

### 3.2 Tag order: keep `avo-vN` speedup non-decreasing

Trunk only commits when `geomean > cumulative`, so **every new tag is
strictly faster than every previous tag**. F0 must keep that:

`score(avo-vK) ≤ score(avo-vK+1)` for all committed nodes (champion and
also_ran).

That **forbids** champion-first (v10=5.95, v11=5.90). Use **ascending
speedup**, champion **last**:

1. Sort `picks` by `verified_geomean` **ascending**.
2. All but the last → `role=also_ran`. Do **not** move `best_id`.
   Each still beats **old** `G`, so `avo_lineage.py` `geomean < best`
   stays happy while `best_id` is still v9.
3. **Last** (highest) → `role=champion`, update `best_id` / HEAD /
   reprofile.
4. After each also_ran, `checkoutAvoRepo(oldChampion)` until the final
   champion commit.

Example vs `G=5.876` (v9):

```
v10 also_ran  5.88×  parent v8    5.88 > 5.876
v11 also_ran  5.90×  parent v4    5.90 > 5.88
v12 champion  5.95×  parent v9    5.95 > 5.90   ← best_id
```

Git parents are still each worker’s `startRef`, not the previous tag.
Monotone tags ≠ a linear chain.

Do not assign `best_id` to an also_ran in step 2 even though its number
is a global improvement over G — the unique champion is the max.

### 3.3 `commitCandidate` split (F0 only)

```js
opts.role = 'champion' | 'also_ran'
```

| side effect | champion | also_ran |
|-------------|---------|----------|
| director pre-commit | yes | yes (v1) |
| git tag `avo-vN` | yes | yes |
| `assertCommittedParent` | yes | yes (parent = that worker’s `startRef`) |
| `commits = N` | yes | yes (monotonic tags) |
| `cumulative`, `bestVersion` | **yes** | **no** |
| `reprofileBest` | yes | **no** |
| `noImprove = 0` | yes | **no** (round already improved if championed) |
| `committed.push` | `role: champion` | `role: also_ran` |
| checkout after | HEAD = new best | back to `bestVersion` |

`commitRoundResult` **must not `return true` before also_rans are tagged.**

```js
const G = cumulative;
const picksAsc = selectPerParentWinners(round.passers); // unique startRef, speedup ASC
const champ = picksAsc[picksAsc.length - 1];
const alsoRans = picksAsc.slice(0, -1);
let championed = false;
if (integrator committed) championed = true;
else {
  for (const p of alsoRans)
    await commitCandidate(p, { role: 'also_ran' }); // no best_id update
  if (champ)
    championed = await commitCandidate(champ, { role: 'champion' });
}
return championed;
```

If integrator became champion, still tag remaining per-parent winners
(not the merge patch) as also_ran **before** or **after**? Integrator
output is usually the new max. Then also_rans (variation parents) must
be tagged **first** (lower speedups), integrator/champion **last**, to
keep tag monotonicity. v1: run F0 picks excluding the champion patch,
ascending, then champion (merge or variation max) last.

### 3.4 Pure helper `lib/parent_keep.mjs`

```js
selectPerParentWinners(passers) → Passer[]  // unique startRef, speedup ASC
```

Tests: two passers same parent → one; two parents → two sorted asc; empty → [].

### 3.5 Starts after F0

`also_ran` nodes **are** committed versions with real tags. Existing
`start_policy=diverse` already buckets by `strategy` and ranks by speedup.
They naturally become explore starts **without** a new start policy.

`bestNode()` must ignore `role=also_ran` when picking exploit start
(defense: also_ran speedup ≤ new champion, so max is still champion, but
filter explicitly).

### 3.6 Lineage fields

```json
{
  "id": "v11",
  "parent_id": "v8",
  "git_ref": "avo-v11",
  "role": "also_ran",
  "strategy": "…",
  "score": { "speedup": 5.89, "verified": true }
}
```

`best_id` always a `role=champion` node (missing `role` ⇒ champion, old
files). Resume: `cumulative` from max among non-`also_ran` (and non-archive
if F2 ever lands).

Git subject: `avo: avo-v11 [also_ran] | 5.890x | …`

### 3.7 Knobs

| knob | default | meaning |
|------|---------|---------|
| `multi_parent_keep` | `false` | master switch for F0 |
| `also_ran_director` | `true` | director pre-commit on also_ran |

Off → identical to trunk (single `round.best` commit).

### 3.8 Cost

Worst case: 8 different parents all beat G → 1 champion + 7 also_ran
director benches. Unlikely; typical 1–2 extra. Cap later if needed
(`also_ran_max`, default unlimited in v1 because `|P|` is already gated).

---

## 4. Feature F1 — Path review, then port *direction* onto champion

Implement **after** F0. F0 only *stores* extra global winners. F1 decides
**which of those (and older off-path champions) are fusion-eligible on the
current champion**, then **rewrites that lever in C’s source** and measures.

This section is the review surface: selection must be deterministic and
auditable. The agent does not pick the donor.

### 4.1 What we are *not* doing

- `git apply` / three-way merge of `parent(D)→D` onto C.
- Copying `D`’s `topk_sparse.py` over C.
- Asking tech_lead in prose to “compose onto v6” with no git parent.

Those smash context. Fusion is **specification transfer**: a reviewed
lever, regenerated on C.

### 4.2 Path inventory (orchestrator, no LLM)

At fusion time let `C` = current `best_id`.

**Champion path** `Anc(C)`: follow `parent_id` from C to v0.
Each node contributes a **strategy card** (already in lineage):

```
{ id, ref, parent_id, role, strategy, speedup,
  expected, rationale_excerpt }   # rationale from commit message / lineage
```

**Off-path pool** `Off(C)`: committed nodes (champion-role **or**
`also_ran`) whose `id` ∉ `Anc(C)` ∪ {C}.

These are the only fusion-donor candidates. F0’s `also_ran` tags matter
because they sit in `Off(C)` with **global-best-beating** evidence.

### 4.3 Lever identity (how we know “already on the path”)

Free-text `strategy` is too noisy to match by equality. F1 uses a
**normalized lever key**:

```
lever_key = norm(strategy)
  lowercase, squeeze space,
  strip port: / fused: / integrated: prefixes,
  then map through an optional alias table
```

v1 alias examples (kernel-agnostic buckets + MiniMax-ish names):

| key | typical strategy text |
|-----|------------------------|
| `launch_pin` | pinned num_warps / num_stages / waves_per_eu |
| `qk_mask` | fused qk mask / mask elision |
| `int32_gather` | int32 paged gather / slot index |
| `index_arith` | umin clamp / KV_SLOT_STRIDE constexpr |
| `mfma_shape` | matrix_instr_nonkdim / instrShape |
| `mfma_denom` | mfma-denominator / delete key-axis softmax sum |
| `pack_q_union` | PACK_Q / packed-q-union-topk |
| `compact_rank` | compacted-rank union walk |
| `cache_cg` | cache_modifier .cg |
| `autotune` | strategy `autotune` (weak donor — skip unless only option) |

Unmapped strings fall back to the full `norm(strategy)` (unique, so they
never collide with an ancestor unless the text is identical).

**Already absorbed:** `lever_key(D)` ∈ `{ lever_key(n) | n ∈ Anc(C) }`.
Skip. (v6 `mfma_shape` vs v9 `mfma_denom` are **different** keys — both
can be eligible relative to each other; do not over-merge MFMA.)

Optional later: embedding / LLM classify into keys. **Not in v1.** v1 is
table + exact norm, so a human can diff the review file.

### 4.4 Fusion eligibility (pure: `reviewFusionDonors`)

For each `D ∈ Off(C)`, emit `{ donor, status, reasons[] }`.

**Reject**

| status | when |
|--------|------|
| `on_champion_path` | D ∈ Anc(C) (should not be in Off) |
| `lever_already_on_path` | `lever_key(D)` already in Anc(C) |
| `already_tried` | `(D.id, C.id)` in `fusion_attempts.jsonl` |
| `autotune_weak` | `lever_key==autotune` and a non-autotune Off exists |
| `role_baseline` | D is v0 |

**Accept (eligible)**

All rejects fail, and D has a real `git_ref`. Also_ran and old
off-path champions are equal citizens.

**Rank eligible** (pick `fusion_k`, default 1):

1. Higher `speedup` first (stronger demonstrated lever).
2. Tie-break: lever **not** mentioned in C’s last `N` failed
   `attempts.jsonl` labels.
3. Tie-break: older donor first (longer-ignored branch).

Output a durable review artifact **before** any agent call:

```
avo_state/fusion_review_step<NN>.json
```

```json
{
  "champion": "avo-v9",
  "champion_path": ["v0","v1","v2","v3","v8","v9"],
  "champion_levers": ["launch_pin","qk_mask","int32_gather","index_arith","mfma_denom"],
  "reviewed": [
    { "id": "v4", "lever": "pack_q_union", "status": "eligible", "speedup": 4.65, "rank": 2 },
    { "id": "v7", "lever": "compact_rank", "status": "eligible", "speedup": 5.52, "rank": 1 },
    { "id": "v6", "lever": "mfma_shape", "status": "eligible", "speedup": 5.35, "rank": 3 },
    { "id": "v5", "lever": "qk_mask", "status": "lever_already_on_path" }
  ],
  "selected": ["v7"]
}
```

This file is what humans review: **why this donor, why those skips.**
A campaign can set `fusion_require_review=true` to **stop after writing
the JSON** (no agent) for dry-run audits. Default `false` (auto-port the
`selected` list).

### 4.5 MiniMax v9 — example review (for this design review)

Champion path levers ≈ pin launch, fused mask, int32 gather, index
arith, mfma-denominator.

| Off-path node | lever | vs current path | F1 v1 |
|---------------|-------|-----------------|-------|
| v4 PACK_Q union | `pack_q_union` | not on path; orthogonal axis (query packing) | **eligible** |
| v7 compacted-rank | `compact_rank` | not on path; related to v4 but different walk | **eligible** (higher speedup than v4 → rank 1) |
| v6 instrShape-16 | `mfma_shape` | not the same key as `mfma_denom` | **eligible** (may conflict in practice; verify decides) |
| v5 redundant-mask | ~`qk_mask` | v2 already fused mask on path | **skip** if alias maps it to `qk_mask` |
| v8/v3/… | — | on `Anc(v9)` | **skip** |

Selected donor v7: agent must **re-express compacted-rank union on the
v9 kernel** (which already has int32 gather + mfma-denom), with PACK_Q=1
/ old body as fallback where the donor did. It must **not** paste v7’s
file (that file lacks v9’s denominator change).

If v7 port fails correctness or does not beat v9, log
`fusion_attempts` and next step may try v4 (`pack_q_union`).

**Human check:** if you disagree that v6 is eligible next to v9, add
alias `mfma_shape`→`mfma_family` and `mfma_denom`→`mfma_family` so both
count as “already on path” once either lands. That is a table edit, not
an agent guess.

### 4.6 Evidence pack (for the chosen donor only)

Orchestrator, no LLM:

```
FUSION_DIR/
  fusion_review.json   # copy of §4.4 file (full Off review)
  donor_card.json      # selected D: lever_key, strategy, speedup, parent
  donor_kernel/        # modifiable files at D (read-only)
  champion_wt/         # writable worktree at C
  donor.diff           # git diff D.parent D, header DO_NOT_APPLY
```

`donor_card.lever_key` is the **spec**. Source dump is to see *how* D
did it. Diff is illustration only.

### 4.7 Agent contract (`roles/avo_fusion.md`)

- Implement **only** `lever_key` on C.
- Do not `git apply` / overwrite C with D.
- Preserve C’s existing levers (everything in `champion_levers`).
- Gate new behavior so shapes D did not claim still run C’s body.
- `no_change` if the lever is already structurally present in C
  (missed by `lever_key`).
- Independent `avo_verify` on base C. Champion-commit iff gate vs
  **current** cumulative → tag `port:<D>@<C>`, `parent_id=C`,
  `fused_from=[D.id]`. Else drop (no F2).

Git parent is always C. Donor is metadata only.

### 4.8 Frequent fusion vs path diversity

**Yes, fusing every step would collapse diversity.** Two different
notions get mixed:

| | Meaning | Fusion does |
|--|---------|-------------|
| **Mechanism diversity in the champion** | C contains PACK_Q *and* mfma-denom | **increases** (the point of F1) |
| **Search diversity of the tree** | live starts still sit on distinct local optima | **decreases** if every useful lever is copied onto C and later workers only edit C |

After a successful port, `lever_already_on_path` skips that donor, so
the *same* lever is not re-fused. The damage is the **next** team round:
`diverse` still lists old tags, but exploit (best first) + fused C often
wins, and off-path starts look strictly slower even if they had unique
headroom (occupancy vs packing). The tree’s old nodes remain, but the
**search mass** concentrates on C — a chain in practice.

F0 without F1 **preserves** path diversity (also_ran stay slower
siblings). F1 **spends** that diversity to enrich C. Cadence must be
low.

**v1 conservation rules** (all default on when `fusion_enabled`):

1. **Primary trigger = champion stall, not “every win”.** Fuse when
   the **global champion has not moved** for `fusion_stall_no_improve`
   rounds (default **3**). Do **not** fuse after every champion
   commit (`fusion_only_after_champion_commit=false`). Climbing C
   with variation is cheaper and keeps search mass on live starts;
   porting a side lever is the **bypass** when that climb is stuck.
2. **Cooldown:** `fusion_cooldown_rounds=2` — after a fusion *attempt*
   (success or fail), skip F1 for 2 evolve steps so variation (and
   the supervisor, if it fires) can act on the new state.
3. **Do not steal all explore GPUs.** Fusion is 0–1 extra slot, never
   replace `operator_directions`. Off-path `also_ran` remain in
   `selectStartPoints` even after their lever is on C (workers on that
   tag can still find *new* levers the champion does not have).
4. **Stop when Off is empty of eligible donors** (all levers already
   on path or tried). That is success, not a bug — then only normal
   diverse variation remains. Stall then falls through to the
   **existing supervisor** (§4.8.1).
5. **Kill switch:** if last `K` fusion successes each add `<
   fusion_min_gain` (default same as commit margin) vs C, disable F1
   for the rest of the campaign (ports are cloning noise). Supervisor
   and ESCALATE keep running as today.

Validation metric for collapse: entropy of `startRef` among workers
that produce gated passers, T2 vs T1. If T2’s passers are ~100% from
the fused champion, fusion is too aggressive even if C is faster.

### 4.8.1 Compatible with the existing supervisor

F1 is a **deterministic stall action**, same layer as the detector in
`docs/avo_supervisor_design.md` — it does **not** replace the LLM
supervisor, `backtrack_to`, or ESCALATE.

Today (`avo_workflow.js`):

| signal | default | action |
|--------|---------|--------|
| `noImprove` | counter of rounds with **no champion commit** | — |
| stall | `noImprove >= stall_no_improve` (**5**) or patch cycling | LLM **REDIRECT** |
| escalate | `supervisorCycles >= 3` without a commit | wider `fanOutEvolve` |
| commit | anyone beats global `cumulative` | `noImprove=0`, `supervisorCycles=0` |

With F0, “主卡不动” **is** this `noImprove`: an also_ran still had to
beat round-start `G` to be tagged, and the unique best becomes the new
champion, which already resets `noImprove`. A second “exploit-arm
only” counter is unnecessary unless F2 (near-miss) is ever added.

**Ladder** (fusion threshold **strictly below** supervisor stall):

```
noImprove = 1, 2     → team / single as today; no F1
noImprove >= 3       → F1 stall-fusion IF eligible Off donors
                       AND cooldown / caps / kill-switch allow
                       AND not patch-cycling
noImprove >= 5       → existing supervisor REDIRECT (unchanged)
supervisorCycles >= 3 → existing ESCALATE (unchanged)
```

Clamp: if `fusion_stall_no_improve >= stall_no_improve`, log and set
`fusion_stall_no_improve = stall_no_improve - 1` so fusion always has
a window **before** REDIRECT. Defaults 3 vs 5 already do this.

**Same-step exclusion.** Patch **cycling** is the wrong medicine for
a donor port (the edit is repeating, not missing a side lever). Skip
F1 when `cycling`; let the supervisor fire as today.

If both would be eligible (`noImprove` somehow ≥ both thresholds in
one step), **supervisor wins** — do not run F1 and REDIRECT in the
same iteration. With defaults this does not happen: fusion at 3,
REDIRECT at 5, cooldown 2 covers steps 4–5 so F1 is idle when the
supervisor runs.

**Failed fusion must not starve the supervisor.**

| fusion outcome | `noImprove` | `supervisorCycles` | cooldown |
|----------------|-------------|--------------------|----------|
| port champion-commits (beats `G`) | **0** (same as any commit) | **0** | start 2 |
| port fails / `no_change` / Off empty | **unchanged** (keep stalling) | unchanged | start 2 |
| skipped (cooldown, cycling, dry-run) | unchanged | unchanged | — |

Resetting `noImprove` on a failed port would loop fusion forever and
never reach REDIRECT. Empty Off is not a fusion success: fall through
until `noImprove=5`.

**Budget.** Reuse the supervisor’s remaining-wall-clock idea: skip F1
if there is not room for fusion agent + one `avo_verify` (same
`estStepS` / `remainingS` style, or a smaller `fusion_overhead_s`).
Do not skip the later REDIRECT solely because fusion was skipped for
budget — those gates stay independent.

**`backtrack_to`.** After a REDIRECT that moves the exploration base,
recompute `Anc(C)` / `Off(C)` against the **current** champion (or the
backtrack target if that is now `bestVersion`). Do not fuse using a
stale donor list from before the backtrack.

**Prompt / memory.** On REDIRECT, pass a short `FUSION_STALL` line:
last fusion attempt (donor `lever_key`, pass/fail, skip reason). The
supervisor can avoid re-proposing a lever that just failed to port.
Do not teach the supervisor to *request* fusion; the orchestrator owns
the trigger.

**Optional second trigger** (`fusion_only_after_champion_commit`,
default **false**): also fuse immediately after a champion move when
Off is non-empty. That is the old “enrich C while winning” mode; it
is more aggressive on diversity. Stall-fusion is the default because
it only spends Off when C is already stuck.

### 4.9 Placement and knobs

Two hooks, both after F0 tagging in that step:

- **Stall hook (default):** Evolve loop, after a round that did **not**
  champion-commit, when `noImprove >= fusion_stall_no_improve`,
  **before** the supervisor block.
- **Win hook (off):** after `commitRoundResult` when `best_id` moved.

First validation: stall hook only, `fusion_k=1`, cooldown 2, stall
threshold 3.

| knob | default | meaning |
|------|---------|---------|
| `fusion_enabled` | `false` | master |
| `fusion_k` | `1` | donors to port per step |
| `fusion_max_per_campaign` | `8` | cap agent calls |
| `fusion_require_review` | `false` | write JSON and skip agent (audit) |
| `fusion_lever_alias_file` | optional | JSON map for §4.3 |
| `fusion_on_stall` | `true` | F1 when champion `noImprove` hits threshold |
| `fusion_stall_no_improve` | `3` | must be `< stall_no_improve` (5) |
| `fusion_only_after_champion_commit` | `false` | extra F1 after a win |
| `fusion_cooldown_rounds` | `2` | skip F1 after an attempt |
| `fusion_min_gain` | gate margin | disable F1 if ports stop helping |

---

## 5. Implementation map (when coding on the fork)


**F0 (first PR)**

| File | Change |
|------|--------|
| `lib/parent_keep.mjs` | `selectPerParentWinners` |
| `tests/parent_keep.test.mjs` | grouping |
| `avo_workflow.js` | `commitRoundResult` no early return; also_ran loop |
| `scripts/avo_lineage.py` | `--role also_ran`; resume ignores also_ran for `best` |
| `lib/start_select.mjs` | `bestNode` skips `also_ran` |
| `run_geak_avo.py` | `multi_parent_keep`, `also_ran_director` |

**F1 (second PR)**

| File | Change |
|------|--------|
| `lib/fusion_select.mjs` | `reviewFusionDonors` + rank; alias table |
| `tests/fusion_select.test.mjs` | path skip, lever alias, also_ran eligible |
| `roles/avo_fusion.md` | implement selected lever on C; forbid git apply |
| `avo_workflow.js` | stall hook **before** supervisor; write `fusion_review_step*.json`; optional dry-run; `FUSION_STALL` on REDIRECT |
| `run_geak_avo.py` | fusion knobs |

Do **not** add `lib/archive_admit.mjs` / α-gate in these PRs.

---

## 6. Validation

Fork + MiniMax snapshot
`MiniMax-M3-MXFP4_gqa_share_sparse_fwd_kernel_resume_5.8x`, 8 GPU, 1 day.

| arm | F0 | F1 |
|-----|----|----|
| T0 | off | off |
| T1 | **on** | off |
| T2 | on | on |

**F0 worked if:** some rounds have `also_ran` tags; `best_id` never an
also_ran; later `diverse` starts use those refs; champion speedup ≥ T0
(not required to win immediately — tree richness is the metric); no
correctness miss.

**F0 failed if:** `also_ran` updates `cumulative`; same parent produces
two tags; HEAD left on also_ran; a later tag has **lower** speedup than
an earlier tag (`avo-vN` not monotone).

**F1** only after T1 looks healthy.

---

## 7. Worked example

Round-start `G = 5.876` (`avo-v9`). Eight workers, champion gate vs G:

| start | × | in P? | per-parent keep | commit |
|-------|---|-------|-----------------|--------|
| v8 | 5.90 | yes | winner(v8) | **also_ran v10** (first) |
| v9 | 5.95 | yes | winner(v9) | **champion v11** (last) |
| v4 | 5.10 | no | — | — |
| v9 | 5.85 | no | — | — |

Commit order (speedup ASC, champion last): **v10 also_ran 5.90× parent v8**,
then **v11 champion 5.95× parent v9**. Both > 5.876; v11 > v10.
HEAD / cumulative = v11. F2 would have kept 5.10 / 5.85 — **we do not**.

---

## 8. Sequencing

1. Review this doc.
2. Copy tree to `PerfSkills-avo-opt`.
3. Implement **F0 only**, flag default false, tests, T0 vs T1.
4. Implement **F1**, T2.
5. F2 (Appendix A) only if T1 still cannot feed F1 enough donors.

---

## Appendix A — F2 (deferred, do not implement)

Keep workers that are correct, beat **their own parent**, ≥ 0.98 ×
global best, but **failed** the champion gate. Same-parent collapse,
exploit-arm (start=champion and lost) excluded.

Requires a new `verified` pool (today non-gated workers never reach
commit), `avo_lineage.py` allow-below-best, and α/CV tuning (decode
v14/v15 noise). Explicitly **out of this review.**
