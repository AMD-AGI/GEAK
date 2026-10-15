#!/usr/bin/env python3
"""test_kb_switch.py — `use_learned_kb=false` must reach EVERY reader of the learned KB.

The switch existed and did not work. It removed `LEARNED_KB_BUDGET` from the planner's inputs — the
block that CONSTRAINS how much of a round the KB may steer — while roles/tech_lead.md still listed
`knowledge/learned/INDEX.md` among the files to read, and roles/author_engineer.md did too. So
`use_learned_kb=false` dropped the limit and kept the instruction, and the KB-off control arm the
flag exists to produce was never actually KB-off.

The failure is not that a check was missing, it is that the switch was enforced per call site and a
call site lapsed. So this test derives the readers FROM THE TREE — any role file that names the
learned KB is a reader and must gate on the `LEARNED_KB` input — rather than from a list that a
third reader could be added without joining.

    python3 kernel_workflow/scripts/test_kb_switch.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(HERE)
ROLES = os.path.join(WF, "roles")
LANE = os.path.join(WF, "kernel_lane.js")
DISPATCH = os.path.join(WF, "kernel_workflow.js")

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


lane = open(LANE).read()

# The switch must be handed to the roles in BOTH positions. Passing it only when the KB is on is what
# the old code did with the budget, and it is why the off case was never expressed to the agent.
check("the lane passes LEARNED_KB unconditionally, not only when the KB is on",
      re.search(r"LEARNED_KB:\s*USE_LEARNED_READ\s*\?\s*'on'\s*:\s*'off'", lane) is not None,
      "expected `LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off'` in the role inputs")

# Readers are DERIVED, not listed: every role that names the learned KB has to gate on the input.
# `update_experience` is the writer and is governed by its own flag, so it is exempt.
WRITERS = {"update_experience.md"}
readers, ungated = [], []
for fn in sorted(os.listdir(ROLES)):
    if not fn.endswith(".md") or fn in WRITERS:
        continue
    text = open(os.path.join(ROLES, fn)).read()
    if "knowledge/learned" not in text:
        continue
    readers.append(fn)
    # The gate must be AT the reference and must say what `off` means. Merely mentioning the token
    # somewhere in the file is not a gate: the first version of this check passed a tech_lead.md
    # whose KB bullet had been reverted to "read it unconditionally", because the word LEARNED_KB
    # still appeared two lines further down. A check that survives deleting the thing it guards is
    # not a check.
    where = text.index("knowledge/learned")
    window = text[max(0, where - 200):where + 400]
    if "LEARNED_KB" not in window or "off" not in window:
        ungated.append(fn)

check("at least one role reads the learned KB (else this test proves nothing)", bool(readers),
      "no role mentions knowledge/learned — did the tree move?")
check("every role that reads the learned KB gates on LEARNED_KB",
      not ungated, f"ungated: {ungated}; readers: {readers}")

# Each gated role must be handed the input by the lane, or the gate reads an undefined value and the
# agent decides for itself what an absent switch means. Inputs may be built inline OR by a helper
# (`planInputs(...)`), so follow one level of indirection rather than only matching the literal —
# the first version of this check reported tech_lead as ungated when it was not, which would have
# taught the next reader to distrust the test.
def inputs_of(role):
    """The text of the inputs expression handed to `role`, helper bodies inlined."""
    out = []
    for m in re.finditer(r"roleAgent\('" + role + r"'", lane):
        seg = lane[m.start():m.start() + 4000]
        out.append(seg)
        for helper in set(re.findall(r"\b([a-zA-Z_]\w*)\(", seg)):
            h = re.search(r"(?:function\s+" + helper + r"\s*\(|const\s+" + helper + r"\s*=)", lane)
            if h:
                out.append(lane[h.start():h.start() + 4000])
    return "\n".join(out)


missing = [r[:-3] for r in readers if "LEARNED_KB" not in inputs_of(r[:-3])]
check("the lane hands LEARNED_KB to every gated role",
      not missing, f"gated but never given the input: {missing}")

# The bake-off path builds its lane args explicitly instead of spreading the caller's, so anything
# left out silently reverts to the lane default. A caller asking for a KB-off bake-off used to get
# eight KB-on lanes and no error.
disp = open(DISPATCH).read()
check("the bake-off dispatcher forwards use_learned_kb to each lane",
      "use_learned_kb:" in disp, "kernel_workflow.js builds lane args explicitly; the flag must be there")

# The always-on perf_knowledge corpus needs its own run-level control arm. It is not the learned KB:
# turning one off must not silently leave the other on, and an explicit empty path must stay empty
# rather than falling through JavaScript's `|| default`.
check("the lane defaults perf knowledge on but exposes an override",
      re.search(r"A\.use_perf_knowledge\s*!=\s*null\s*\?\s*A\.use_perf_knowledge\s*:\s*'true'",
                lane) is not None,
      "expected args.use_perf_knowledge with historical default true")
check("an explicit empty perf knowledge dir does not fall back to the default",
      "A.perf_knowledge_dir != null ? String(A.perf_knowledge_dir)" in lane
      and "USE_PERF_KNOWLEDGE ? REQUESTED_PERF_KNOWLEDGE_DIR : ''" in lane,
      "use nullish selection plus an explicit off branch, not `A.perf_knowledge_dir || default`")
check("planning and authoring receive the perf knowledge switch",
      lane.count("PERF_KNOWLEDGE: USE_PERF_KNOWLEDGE ? 'on' : 'off'") >= 3,
      "author, analyze and plan_round must all see the control-arm state")
# Counted as CALL SITES, not as bare occurrences. The earlier `count(NAME) >= 4` meant "one
# definition plus the three role inputs", which stopped being true the moment the three stopped
# being constants: the isolate branches turn the authoring KB off AFTER the top of the lane, so a
# constant captured the pre-isolate value and the paths had to become call-time helpers. The count
# fell to 3 and the check failed while the property it stands for was intact.
check("planning and authoring receive the corpus catalog",
      lane.count("CORPUS_CATALOG: corpusCatalog()") >= 3,
      "new operator families must be discoverable without another GEMM hard-code")
check("planning and authoring receive measured decision outcomes",
      lane.count("MEASURED_DECISION_OUTCOMES: measuredDecisionOutcomes()") >= 3,
      "the verifier feedback aggregate must reach the candidate selector")
check("planning and authoring receive measured implementation bundles",
      lane.count("MEASURED_IMPLEMENTATION_REGISTRY: measuredImplementationRegistry()") >= 3,
      "normalized AITER benchmark winners must reach author and optimize")
# The helpers only matter if they actually read the switch — a helper that ignored it would pass the
# three checks above and still leak the KB into the control arm.
check("the corpus paths are read through the switch at call time",
      "KERNEL_KNOWLEDGE_DIR ? `${KERNEL_KNOWLEDGE_DIR}/corpus/catalog.yaml` : ''" in lane
      and lane.count("USE_PERF_KNOWLEDGE && A.") >= 2,
      "corpusCatalog/measured* must gate on the live switch, not on a value captured at lane start")
check("the bake-off dispatcher forwards use_perf_knowledge to each lane",
      "use_perf_knowledge:" in disp,
      "kernel_workflow.js builds bake-off lane args explicitly; the flag must be there")
check("the bake-off dispatcher forwards the corpus exploration control",
      "corpus_cold_direction:" in disp and "corpus_dir_cap:" not in disp,
      "all directions may use corpus; only a profile/free hypothesis is reserved")
check("the bake-off dispatcher forwards measured decision outcomes",
      "decision_outcomes_path:" in disp,
      "explicit lane args must not silently drop the measured prior")
check("the bake-off dispatcher forwards measured implementation bundles",
      "implementation_registry_path:" in disp,
      "explicit lane args must not silently drop exact-context implementation priors")

# Perf-decision attribution is separate from the learned-card citation ledger. A page path (`kk_refs`)
# cannot identify which curated card or cfg row seeded a direction, so exact IDs must survive the
# entire plan -> verify -> history -> return path.
tech = open(os.path.join(ROLES, "tech_lead.md")).read()
check("the planner schema accepts exact decision refs",
      "decision_refs: { type: 'array'" in lane and "`decision_refs`" in tech,
      "PLAN_SCHEMA and the TechLead output contract must both name decision_refs")
check("decision refs reach engineers and round history",
      "decision_refs: d.decision_refs || []" in lane
      and "decision_refs: r.d.decision_refs || []" in lane,
      "attribution must not stop at the planner output")
check("verifier outcomes are joined to decision refs",
      "decisionCitations.push" in lane and "decision_citations: decisionCitations" in lane,
      "formal validation needs exact per-decision outcomes in the lane result")
check("decision outcomes distinguish frozen-baseline score from round increment",
      "incremental_vs_incumbent" in lane
      and "ratio_of_frozen_baseline_speedups" in lane
      and "bundle_size" in lane,
      "one final speedup must not be copied onto every referenced decision as causal credit")
check("the perf-KB-off arm mechanically strips decision attribution",
      lane.count("USE_PERF_KNOWLEDGE ? normalizeDecisionRefs") >= 2,
      "author and plan outputs must not smuggle decision refs into the control arm")
check("every completed validation attempts an environment manifest",
      "capture_validation_env.py" in lane
      and "validation_environment.yaml" in lane
      and "validation_environment:" in lane,
      "software/GPU/shape/method capture must not depend on earning a learned card")
check("corpus hypothesis diversity is checked before engineer dispatch",
      "CORPUS_EXPLORATION_CONTRACT" in lane
      and lane.index("corpusPlanViolation(plan)") < lane.index("const results = await pipeline("),
      "a post-run attribution edit cannot undo hypothesis anchoring")
check("a multi-direction round keeps a hypothesis neither source seeded",
      "CORPUS_COLD_DIRECTION" in lane
      and "no executable direction is seeded by neither corpus nor learned cards" in lane
      and "rawDirections = rawDirections.filter(" in lane
      and "hypothesis_source" in lane,
      "semantic/constraint cards are unlimited; performance history must not decide every hypothesis")
# One invariant, one switch, one predicate. Both halves of "recorded knowledge" have to be checked
# together: a round split between a card-seeded and a corpus-seeded direction once satisfied the two
# knobs separately while holding no control at all.
check("the cold-direction invariant has a single switch",
      "KB_COLD_DIRECTION &&" in lane
      and lane.count("const CORPUS_COLD_DIRECTION") == 1,
      "corpus_cold_direction must narrow kb_cold_direction, never act as a second independent knob")
check("an independent direction is seeded by neither corpus nor cards",
      "const independentDirection =" in lane
      and "learned_refs) || []).length" in lane,
      "hypothesis_source alone reads a card-only direction as `profile` and misses the violation")
check("there is no fixed cap on corpus-using directions",
      "CORPUS_DIR_CAP" not in lane and "corpus_dir_cap" not in lane,
      "API and constraint knowledge should be available to every direction")
check("language detection excludes the frozen baseline tree",
      re.search(r"detectSourceLanguage\(\s*`\$\{CANONICAL\}/kernel_src`", lane) is not None
      and "detect_language.py ${CANONICAL} " not in lane,
      "baseline_src may be Triton while the authored candidate is FlyDSL")
check("author language mismatch is a hard result gate",
      "validation_status: 'language_mismatch'" in lane
      and "validation_status: effectiveValidationStatus" in lane,
      "recording a mismatch while returning accepted poisons language-indexed knowledge")
check("FlyDSL API compatibility is checked before optimization",
      lane.count("checkFlydslCompatibility(") >= 3
      and "validation_status: 'flydsl_incompatible'" in lane
      and "ACTIVE_FLYDSL_VERSION" in lane,
      "version-map advice needs an active-surface gate before GPU optimization")
check("FlyDSL package availability is checked before the author agent",
      lane.index("await probeFlydslPackage('author:flydsl-package')") <
      lane.index("roleAgent('author_engineer'"),
      "a missing FlyDSL install should not consume an author run")
check("experience writes use the observed source language",
      "--language ${JSON.stringify(observedLanguage)}" in lane,
      "the optimize-mode requested language defaults to triton and is not an observation")
check("author search starts from the measured seed rather than a fictional 1.0",
      "MODE === 'author' ? authorSeedSpeedup : 1.0" in lane
      and "seed_per_case" in lane,
      "sub-baseline migration improvements cannot compound if the incumbent is fixed at 1.0")
check("author search and shipping use different gates",
      "REQUIRE_SPEEDUP: MODE === 'author' ? 'true' : 'false'" in lane
      and "'no_speedup'" in lane,
      "a sub-baseline candidate may advance search but must not be shipped")
check("mode is normalized in both dispatcher and worker",
      ".trim().toLowerCase() || 'optimize'" in lane
      and "{ ...A, mode: MODE, workflow_dir: WORKFLOW_DIR }" in disp,
      "mixed-case AUTHOR must not enter a half-author/half-optimize state")
check("round state advances only after a successful commit",
      "landed = !!(commitResult && commitResult.committed)" in lane
      and lane.index("if (landed) {") < lane.index("cumulative = winner.geomean"),
      "a measured patch that failed to apply is not the next canonical kernel")
check("final source gate runs before Director may apply",
      lane.index("'lang:detect-final'") < lane.index("roleAgent('director', 'validate'")
      and "APPLY_TO_ORIGINAL: sourceGatePassed ? APPLY_TO_ORIGINAL : 'false'" in lane,
      "language/API mismatch must disable external mutation before Director validation")
check("author seed measurement uses the optimization primary metric",
      "author_measurement_invalid" in lane
      and "authored.seed_metric_kind === expectedSeedMetric" in lane
      and "seed_metric_kind" in lane,
      "a geomean seed cannot initialize a weighted optimization climb")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("all green")
