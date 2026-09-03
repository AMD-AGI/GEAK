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
import os
import re
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

# The fast multi-GPU path builds its integration input independently of the serial helper. Keep the
# optional EvoK winner contract aligned in both places or only fast runs will lose the rebind data.
fast_start = s.index("const headWinnerInputs = {")
fast_inputs = s[fast_start:s.index("const integ =", fast_start)]
check("fast head integration forwards EvoK winner metadata",
      all(token in fast_inputs for token in
          ("evok_callable:", "evok_signature_form:", "evok_hit_json:", "...EVOK_INPUTS")),
      "headWinnerInputs must carry the same EvoK fields as mkIntegrateInputs")

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
