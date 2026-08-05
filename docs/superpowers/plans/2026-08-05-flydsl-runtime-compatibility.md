# FlyDSL Runtime Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while implementing each task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GEAK expert skills distinguish AITER-vendored FlyDSL from standalone FlyDSL, require and validate the bounded `>=0.2.2,<0.3` runtime, and teach the two gfx950 MoE skills how to use the 0.2.2 typed-coordinate API safely.

**Architecture:** Add a machine-readable `runtime` contract to expert-skill frontmatter and the generated selector index. A small Python compatibility module validates contracts, selects the installed-version profile with PEP 440 semantics, and probes provider-specific imports/symbols without installing or replacing packages. Workflow role guidance runs this probe before applying a matched skill; incompatible skills are skipped explicitly, while unvalidated profiles must be remeasured.

**Tech Stack:** Python 3.8+, PyYAML, `packaging`, standard-library `importlib`, GEAK Markdown/YAML expert skills.

## Global Constraints

- Work only in `/sgl-workspace/upstream_GEAK/GEAK_moe_skill/GEAK`.
- Do not modify or install into the active AITER/FlyDSL environment.
- Do not claim unbounded future compatibility: the new typed-coordinate profile is `>=0.2.2,<0.3`.
- Validate the skills on FlyDSL 0.2.2; older runtime results are not retained in the updated recipes.
- Treat the serving provider as `aiter_vendored_flydsl`; do not require standalone `kernels.moe_gemm_2stage`.
- Use `packaging.version.Version` and `packaging.specifiers.SpecifierSet`; do not implement custom semantic-version parsing.
- A runtime mismatch must produce an explicit incompatibility result, not silently fall back or auto-upgrade.
- Do not create commits unless the user explicitly requests them.

---

### Task 1: Runtime contract model and tests

**Files:**
- Create: `perf_knowledge/expert_skills/_contribute/test_runtime_compat.py`
- Create: `perf_knowledge/expert_skills/_contribute/runtime_compat.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `validate_runtime_contract(runtime: dict) -> list[str]`
- Produces: `select_runtime_profile(runtime: dict, installed_version: str) -> dict | None`
- Produces: `probe_runtime(runtime: dict) -> dict`
- Produces CLI: `python runtime_compat.py <skill-id> --json`

- [ ] Write failing unit tests for valid AITER-vendored profiles, rejection of unbounded `>=0.2.2`, PEP 440 profile selection, missing-version behavior, provider-specific import/symbol checks, and the absence of any standalone-kernel requirement.
- [ ] Run `python3 -m pytest perf_knowledge/expert_skills/_contribute/test_runtime_compat.py -q` and verify failures are caused by the missing module.
- [ ] Add `packaging>=23` to GEAK runtime dependencies.
- [ ] Implement the minimal contract validator, selector, runtime probe, and JSON CLI.
- [ ] Re-run the focused tests and verify all pass.

### Task 2: Indexing and static validation

**Files:**
- Modify: `perf_knowledge/expert_skills/_contribute/scaffold.py`
- Modify: `perf_knowledge/expert_skills/_contribute/validate_skill.py`
- Modify: `perf_knowledge/expert_skills/_template/SKILL_TEMPLATE.md`
- Modify: `perf_knowledge/expert_skills/index.yaml` (generated)
- Test: `perf_knowledge/expert_skills/_contribute/test_runtime_compat.py`

**Interfaces:**
- Consumes: `validate_runtime_contract`
- Produces: generated index entries retaining non-empty `runtime` contracts
- Produces: static validation errors for malformed runtime contracts

- [ ] Extend the tests to assert malformed FlyDSL contracts fail static validation and reindexing preserves runtime data.
- [ ] Run the focused tests and verify the new assertions fail.
- [ ] Teach `scaffold.py` to copy non-empty `runtime` frontmatter and update the index schema header.
- [ ] Teach `validate_skill.py` to validate non-empty runtime contracts and include runtime-profile guidance in emitted plans.
- [ ] Add an optional runtime-contract block to the skill template.
- [ ] Re-run focused tests and regenerate `index.yaml`.

### Task 3: Workflow consumption and documentation

**Files:**
- Modify: `e2e_workflow/roles/_fragments/expert_skills.md`
- Modify: `kernel_workflow/roles/_fragments/expert_skills.md`
- Modify: `perf_knowledge/expert_skills/README.md`
- Test: `perf_knowledge/expert_skills/_contribute/test_runtime_compat.py`

**Interfaces:**
- Consumes: `runtime_compat.py <skill-id> --json`
- Produces: explicit `backend_incompatible` handling and `revalidation_required` behavior in role guidance

- [ ] Add text-level regression tests requiring both workflow fragments to invoke the runtime probe and distinguish incompatible from revalidation-required profiles.
- [ ] Run focused tests and verify they fail before documentation changes.
- [ ] Update both workflow fragments to gate matched skills on the runtime probe before authoring/integration.
- [ ] Document provider/version profiles, staleness behavior, and no-auto-install policy in the expert-skills contract.
- [ ] Re-run focused tests.

### Task 4: Port the two gfx950 MoE skill recipes

**Files:**
- Modify: `perf_knowledge/expert_skills/skills/flydsl_decode_moe_stage1_blkmap/skill.md`
- Modify: `perf_knowledge/expert_skills/skills/flydsl_prefill_moe_stage2_fp8partial/skill.md`
- Modify: `perf_knowledge/expert_skills/index.yaml` (generated)
- Test: `perf_knowledge/expert_skills/_contribute/test_runtime_compat.py`

**Interfaces:**
- Produces `==0.2.2` as `validated` and `>0.2.2,<0.3` as
  `revalidation_required`, so evidence never transfers to an untested release.
- Records fresh FlyDSL 0.2.2 strict-segment evidence alongside each skill.
- Produces 0.2.2 port instructions using `fx.Int32` coordinates and `flydsl.expr.typing.as_ir_value`

- [ ] Add tests requiring both skill frontmatters to expose the provider, both profiles, AITER-vendored imports, and no standalone-kernel import.
- [ ] Run focused tests and verify they fail.
- [ ] Add runtime contracts and a compatibility section to both skill recipes.
- [ ] For decode, document every `idx2crd`/`crd2idx` dynamic-coordinate conversion and preserve `_am2` plus descriptor runtime signatures.
- [ ] For prefill, document typed-coordinate conversion, official `as_ir_value`, one compile-time partial scale shared by producer/reducer, and inclusion of that scale in both JIT cache identities.
- [ ] State that 0.2.2+ performance/parity evidence must be freshly produced before the profile can become validated.
- [ ] Regenerate `index.yaml` and re-run focused tests.

### Task 5: Full static verification

**Files:**
- Verify all modified files.

- [ ] Run `python3 -m pytest perf_knowledge/expert_skills/_contribute/test_runtime_compat.py -q`.
- [ ] Run static validation for both modified MoE skills.
- [ ] Run static validation for every existing expert skill to detect schema regressions.
- [ ] Run `python3 -m py_compile` on all modified Python files.
- [ ] Run `node e2e_workflow/scripts/test_expert_skills_off_identical.js` if Node.js is available; otherwise report the unavailable executable without claiming that test passed.
- [ ] Run `git diff --check` and inspect `git diff --stat` plus the complete diff.
