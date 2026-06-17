---
name: add-expert-skill-to-geak
description: Contribute a human-authored, e2e-validated optimization recipe (an "expert skill") to GEAK — scaffold, fill, validate by scope, and open a PR.
---

# Skill: add an expert skill to GEAK

Use this when a human expert has a **reusable optimization recipe** worth capturing so the
`e2e_workflow` / `kernel_workflow` can reproduce it automatically — e.g. "port MLA decode from TileLang
to Triton on gfx942", or "the FlyDSL fp8 a8w8 blockscale down-proj playbook (+67% e2e)".

Read the contract first: [`../README.md`](../README.md). Key rule: a skill is an **advisory prior**,
never a mandate; it must pass a validation gate (efficacy + do-no-harm) before it lands as `validated`,
and the consuming workflow always decides the winner by on-box measurement.

## Steps (works for a human or an agent)

### 1. Scaffold
```bash
python _contribute/scaffold.py --id <slug> --operator <op> --scope <kernel|e2e> \
  --title "..." --author <you> --gens gfx942 --dtypes fp8_e4m3_fnuz --regimes prefill,decode \
  [--from-backend tilelang --to-backend triton]   # migration skills
```
- `--operator` MUST be a name that exists in `../index/capability_index.yaml` (the selector matches on
  it). The scaffolder rejects unknown operators.
- `--scope kernel` → validated by `kernel_workflow` (isolated A/B vs the oracle), consumed by the kernel
  layer. `--scope e2e` → validated by `e2e_workflow` (Director same-session A/B), consumed by routing.
- This writes `skills/<slug>/skill.md` (status: `draft`) and regenerates `index.yaml`.

### 2. Fill the recipe
Edit `skills/<slug>/skill.md`. The body sections are required and must be non-empty:
- **When to use** — the exact bottleneck/shape/arch.
- **Mechanism** — *why* it works (hardware/numerics/scheduling) so it transfers.
- **Procedure** — the regulated steps an author-agent reproduces: entrypoints, kernel structure, the
  named lever (e.g. "fuse the dequant into one fp8 MFMA").
- **Knobs & pitfalls**, **Do-no-harm notes** (where it must stay OFF), **Sources** (every claim pointed).

### 3. Validate (two-sided gate)
```bash
python _contribute/validate_skill.py <slug> --static          # schema/operator/sections (no GPU)
python _contribute/validate_skill.py <slug> --emit-plan --model <MODEL>   # prints the on-box command
# ... run that Workflow command on a box; it produces an eval dir with the measured delta ...
python _contribute/validate_skill.py <slug> --record --artifact <eval_dir> \
  --gpu gfx942/MI300X --model <name> --date 2026-06-17 \
  --e2e-pct 2.1 --parity pass            # (kernel scope: --isolated 1.27 instead of --e2e-pct)
```
- **Efficacy**: the measured delta must meet the skill's `expects` (`isolated_speedup_min` or
  `e2e_delta_min_pct`) with parity. Otherwise `--record` stamps `status: failed` and exits non-zero.
- **Do-no-harm**: also run the control scenario from `_emit-plan` (a model/shape that does NOT match
  the selector) with `use_expert_skills=true` and confirm `|e2e delta|` stays within the noise band —
  i.e. the skill is inert when not triggered. Record that eval dir in the skill's Sources.
- `--record` writes the `validation:` block and reindexes; only `validated` skills are auto-applied.

### 4. Open a PR
```bash
bash _contribute/make_pr.sh <slug>          # refuses unless status==validated (use --allow-draft for WIP)
```
Branches `expert-skill/<slug>`, commits the skill + regenerated `index.yaml`, pushes, and opens a PR
(via `gh` if present, else prints the compare URL).

## Notes
- **Skills are opt-in.** The workflows ignore `expert_skills/` unless a run passes
  `use_expert_skills=true` (default OFF). So: validation runs (`--emit-plan` above) already pass it,
  and to benefit from a landed skill in a normal optimization run you must enable it explicitly. With
  the flag OFF the workflow behaves byte-identically to a build without this feature.
- Maintain skills ONLY in the canonical `geak_v4/GEAK` tree; other snapshots sync from here.
- A skill that later regresses (aiter/triton upgrade, box drift) should be re-validated; staleness
  demotes it to a plain reference until refreshed.
