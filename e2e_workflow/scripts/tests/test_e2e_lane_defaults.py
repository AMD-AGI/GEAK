#!/usr/bin/env python3
"""test_e2e_lane_defaults.py — e2e launches the kernel lane with the learned KB OFF by default.

Same worker, different prior. A kernel_workflow campaign re-optimizes a fixed benchmark set, where a
card distilled from an earlier run of the same kernel is exactly the point. e2e extracts whatever the
profiler surfaces from a live server, so a card carrying another run's conclusions about a
superficially similar op is a prior nobody asked for — and the e2e layer has never been measured with
it on. So kernel_workflow defaults on, e2e defaults off, and `args.use_learned_kb` overrides either.

The interesting failure is not "the default is wrong" — it is "the default is right at six call
sites and absent at the seventh". e2e invokes the lane from seven places today; anything that misses
the injection silently takes the LANE's default, which is on. So this derives the call sites from the
file rather than checking a list, and requires each one to route through `laneArgs`.

    python3 e2e_workflow/scripts/tests/test_e2e_lane_defaults.py
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(E2E, "e2e_workflow.js")

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


s = open(SRC).read()

check("e2e defaults the lane's learned KB to off",
      re.search(r"LANE_USE_LEARNED_KB\s*=\s*String\(A\.use_learned_kb\s*!=\s*null\s*\?\s*"
                r"A\.use_learned_kb\s*:\s*'false'\)", s) is not None,
      "expected LANE_USE_LEARNED_KB to fall back to 'false'")

check("the default is overridable per run",
      "A.use_learned_kb" in s, "the caller must be able to turn it back on")

check("one injection point, not one per call site",
      s.count("const laneArgs = (wfArgs) =>") == 1 and "use_learned_kb: LANE_USE_LEARNED_KB" in s,
      "laneArgs is where the default is applied")

check("gfx1201 disables the E2E learned-card corpus",
      "const E2E_LEARNED_KB_ENABLED = !RDNA4_ISOLATE" in s
      and "E2E_LEARNED_KB: E2E_LEARNED_KB_ENABLED ? 'on' : 'off'" in s,
      "every role prompt must receive one architecture-gated switch")

check("RDNA4 matrix-core guidance is WMMA, not hard-coded MFMA",
      "const MATRIX_CORE_NAME = RDNA4_ISOLATE ? 'WMMA' : 'MFMA'" in s
      and "one fp8 MFMA" not in s,
      "R9700-reachable task strings must use MATRIX_CORE_NAME")

# Every lane invocation must go through it. Two shapes exist: the bounded wrappers, which inject
# internally, and raw `workflow(...)` calls, which must wrap their args explicitly.
WRAPPERS = ("fastBoundedWorkflow", "deepBoundedWorkflow")
for w in WRAPPERS:
    body = s[s.index(f"function {w}("):][:600]
    check(f"{w} injects the lane defaults",
          "workflow(ref, laneArgs(wfArgs))" in body,
          "the wrapper calls workflow() with raw wfArgs")

raw = []
for m in re.finditer(r"workflow\(\{ scriptPath: KERNEL_WF_SCRIPT \}, (.{0,12})", s):
    caller = s.rfind("\n", 0, m.start())
    line = s[caller + 1:s.index("\n", m.start())].strip()
    if any(w in line for w in WRAPPERS):
        continue                      # covered by the wrapper's own injection
    if not m.group(1).lstrip().startswith("laneArgs("):
        raw.append(s[:m.start()].count("\n") + 1)

check("every raw lane invocation wraps its args in laneArgs",
      not raw, f"unwrapped `workflow({{scriptPath: KERNEL_WF_SCRIPT}}, ...)` at line(s) {raw}")

# Execute the actual argument boundary without starting a workflow. A dropped
# selector silently runs ordinary GEAK, so check behavior as well as call routing.
boundary = s[s.index("const RDNA4_ISA ="):s.index("// EXP_ROOT =")]
worker_src = os.path.join(os.path.dirname(E2E), "kernel_workflow", "src", "kernel_lane")
worker_args = open(os.path.join(worker_src, "01b_args.js")).read().split("const BUDGET =")[0]
dispatch = open(os.path.join(worker_src, "14b_optimize_dispatch.js")).read()
cases = [{}, {"optimizer": "greedy"}, {"optimizer": "evo"}]
cases += [{"optimizer": value} for value in (None, "", "population", "EVO", True, 1, [], {})]
probe = subprocess.run(["node", "-e", """
const fs = require('node:fs'), vm = require('node:vm');
const {boundary, worker_args, dispatch, cases} = JSON.parse(fs.readFileSync(0, 'utf8'));
Promise.all(cases.map(async A => {
  try {
    const child = vm.runInNewContext(boundary + '\\nlaneArgs({kernel_path: "/task"})', {
      A, EXPECTED_GFX: '', EXPECTED_TARGET: '', EXPECTED_DEVICE_NAME: '',
      EXPECTED_PHYSICAL_CU_COUNT: 0,
    });
    const calls = [];
    // Real worker parsing/dispatch, with engineering and evolution creation stubbed.
    await vm.runInNewContext('(async () => {\\n' + worker_args + dispatch + '\\n})()', {
      A: child, cumulative: 1, bestSeen: 1, bestPerCase: {}, finalWinner: null,
      profileSummary: '', history: {},
      createEvolution: async () => { calls.push('createEvolution'); return 'evolution'; },
      greedyOptimize: async (_start, evolution) => { calls.push(evolution || 'ordinary'); },
      applyOptimizeResult: () => {},
    });
    return {...child, policy_calls: calls};
  } catch (error) { return {error: error.message}; }
})).then(results => console.log(JSON.stringify(results)));
"""], input=json.dumps({"boundary": boundary, "worker_args": worker_args,
                        "dispatch": dispatch, "cases": cases}),
    text=True, capture_output=True, check=True)
results = json.loads(probe.stdout)
check("omitted optimizer preserves the worker default", "optimizer" not in results[0])
for i, optimizer in enumerate(("greedy", "evo"), 1):
    check(f"{optimizer} reaches the existing nested worker",
          results[i].get("optimizer") == optimizer and results[i].get("kernel_path") == "/task")
check("omitted and greedy selectors enter the ordinary worker policy",
      all(result.get("policy_calls") == ["ordinary"] for result in results[:2]))
check("evo selector enters the existing evolution policy in the shared loop",
      results[2].get("policy_calls") == ["createEvolution", "evolution"])
check("invalid optimizer is rejected before workflow dispatch",
      all(result.get("error") == "optimizer must be greedy or evo" for result in results[3:]))

budget_declaration = re.search(r"^const KERNEL_BUDGET =.*$", s, re.MULTILINE).group(0)
budget_cases = [{}, {"kernel_budget": 1}, {"kernel_budget": 12}, {"kernel_budget": "12"}, {"kernel_budget": " +012 "}]
budget_cases += [{"kernel_budget": value} for value in (None, "12.0", "12x", "1e3", True, False, 0, -1, 1.5, [], {}, 2**53)]
budget_probe = subprocess.run(["node", "-e", """
const fs = require('node:fs'), vm = require('node:vm');
const {boundary, budget_declaration, cases} = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = [false, true].map(FAST_MODE => cases.map(A => {
  try {
    return vm.runInNewContext(boundary + '\\n' + budget_declaration + '\\nlaneArgs({budget: KERNEL_BUDGET})', {
      A, FAST_MODE, EXPECTED_GFX: '', EXPECTED_TARGET: '', EXPECTED_DEVICE_NAME: '',
      EXPECTED_PHYSICAL_CU_COUNT: 0,
    });
  } catch (error) { return {error: error.message}; }
}));
console.log(JSON.stringify(results));
"""], input=json.dumps({"boundary": boundary, "budget_declaration": budget_declaration, "cases": budget_cases}),
    text=True, capture_output=True, check=True)
budgets = json.loads(budget_probe.stdout)
check("omitted kernel budget keeps full/fast defaults", budgets[0][0].get("budget") == 6 and budgets[1][0].get("budget") == 3)
check("explicit kernel budgets reach the existing lane unchanged",
      all(row[1].get("budget") == 1 and row[2].get("budget") == 12 for row in budgets))
check("direct e2e integer strings remain accepted without truncation",
      all(row[3].get("budget") == 12 and row[4].get("budget") == 12 for row in budgets))
check("invalid kernel budgets are rejected without coercion",
      all(result.get("error") == "kernel_budget must be a positive safe integer" for row in budgets for result in row[5:]))

# The kernel side must still default ON — this test would otherwise pass just as well on a tree where
# somebody turned the KB off everywhere.
lane = open(os.path.join(os.path.dirname(E2E), "kernel_workflow", "kernel_lane.js")).read()
check("the kernel lane itself still defaults the KB ON",
      re.search(r"A\.use_learned_kb\s*!=\s*null\s*\?\s*A\.use_learned_kb\s*:\s*'true'", lane) is not None,
      "kernel_workflow's own default must stay on")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("all green")
