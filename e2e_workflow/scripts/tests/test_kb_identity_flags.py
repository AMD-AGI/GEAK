#!/usr/bin/env python3
"""test_kb_identity_flags.py — e2e KB identity argv is formatted in exactly one place.

kb/identity.py's header names the failure this guards: the `geak` scheme is client-defined and
supports EXACT canonical-id lookup only. There is no search. A reader and a writer that disagree by
a single segment do not raise — they address two different pages, and the only symptom is that
history quietly stops existing, indistinguishable from "nobody ever recorded this".

e2e_workflow.js calls the store from both sides (Module A reads, Module B writes). As long as both
go through `kbIdentityFlags()`, they cannot disagree; the moment a second call site formats
`--tp`/`--isl`/... itself, they can, and nothing fails until months of records are stranded. That
rule had no automated defence, so this file supplies one by deriving the call sites from the source
rather than checking a list.

It also pins the two ends against each other: every flag the JS emits must be a flag
e2e_store.py's `_identity_args` actually accepts. A rename on either side is otherwise a
`unrecognized arguments` SystemExit inside a writer that only runs after a multi-hour campaign.

    python3 e2e_workflow/scripts/tests/test_kb_identity_flags.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(os.path.dirname(E2E), "e2e_workflow.js")
STORE = os.path.join(E2E, "e2e_store.py")

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


s = open(SRC).read()

block = re.search(r"function kbIdentityFlags\(\) \{.*?\n\}", s, re.S)
check("kbIdentityFlags() exists", block is not None)
if block is None:
    sys.exit(1)
body = block.group(0)

# The flags that address an e2e page. Each is specific enough to this scheme that a second
# occurrence anywhere in the file is, by construction, a second formatting site.
#
# --model / --gfx / --framework-version / --precision / --rocm-version are deliberately NOT here:
# the kernel KB is a different scheme that legitimately spells some of them too (see the
# `--language ... --gfx ...` sites), so "appears once" is the wrong assertion for them. They are
# covered by the membership check below instead.
E2E_ONLY = ("--tp", "--ep", "--isl", "--osl", "--conc", "--workload-kind")

for flag in E2E_ONLY:
    hits = [s[:m.start()].count("\n") + 1 for m in re.finditer(re.escape(flag) + r"[ `]", s)]
    check(f"{flag} is formatted exactly once in the whole file",
          len(hits) == 1, f"found at line(s) {hits}")
    check(f"{flag} is formatted inside kbIdentityFlags()",
          flag + " " in body or flag + "`" in body)

# ── The JS and the Python argparse must spell the same flags ───────────────────────────────────
emitted = sorted(set(re.findall(r"`(--[a-z-]+) ", body)))
check("kbIdentityFlags() emits a non-trivial argv", len(emitted) >= 8, f"emitted {emitted}")

store = open(STORE).read()
ident = re.search(r"def _identity_args\(p\):.*?\n\n\n", store, re.S)
check("e2e_store._identity_args() exists", ident is not None)
accepted = set(re.findall(r'p\.add_argument\("(--[a-z-]+)"', ident.group(0) if ident else ""))
unknown = [f for f in emitted if f not in accepted]
check("every flag the workflow emits is accepted by e2e_store's identity parser",
      not unknown, f"{unknown} would be an `unrecognized arguments` SystemExit at write time")

# ep and the workload kind are the two dimensions the agentx address depends on; losing either
# silently reverts the address to the pre-ep scheme, which still parses and still writes.
for flag in ("--ep", "--workload-kind"):
    check(f"{flag} reaches the store", flag in emitted)

# The kind must come from the DECLARATION, not from the shape provenance: the provenance values
# include agentx_pending_baseline and synthetic_fallback_on_agentx, which would mis-file precisely
# the degraded AgentX runs that most need a stable address.
check("--workload-kind is fed by WORKLOAD_KIND, never WORKLOAD_SHAPE_PROVENANCE",
      "--workload-kind ${shq(WORKLOAD_KIND)}" in body
      and "WORKLOAD_SHAPE_PROVENANCE" not in body)

# ep is read once at startup, like tp. Reading it from a live value would reintroduce the drift the
# whole change exists to remove, because Config Tuner may move --ep-size mid-run.
check("--ep is fed by the startup constant SERVING_EP",
      "--ep ${SERVING_EP}" in body)
check("SERVING_EP is a startup constant parsed from args",
      re.search(r"const SERVING_EP = parseInt\(A\.ep != null \? A\.ep : 1, 10\);", s) is not None)

# ── The graded axis is a LABEL, not an address ────────────────────────────────────────────────
# bench_summarize.py grades on five bases and keeps adding them. In the address that would be five
# sparse pages nobody can compare; as a recorded field it is one page a reader can filter. So
# --metric-basis must reach the store from the READ, and must never reach kbIdentityFlags().
check("the graded axis is not an addressing dimension",
      "--metric-basis" not in body,
      "a basis in the address splits one deployment's history across five pages")
check("a declared basis is stated on the read",
      re.search(r"function kbMetricBasisFlag\(\) \{[\s\S]*?--metric-basis", s) is not None)
check("the resolve command carries it",
      "kbMetricBasisFlag()" in s and re.search(
          r"resolve \$\{kbIdentityFlags\(\)\}[\s\S]{0,120}kbMetricBasisFlag\(\)", s) is not None)
check("only a declared basis is stated",
      re.search(r"AGENTX && AGENTX_METRIC_BASIS \?", s) is not None,
      "a synthetic run asserts no basis and must keep its present behaviour")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("all green")
