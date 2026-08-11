#!/usr/bin/env python3
"""test_kb.py — guards the learned-KB contract: the generated index, and the gates that admit a card.

Supersedes test_learned_index.js, whose assertions about the index are all kept below. It grew the
gate tests because the index contract was the only part under test: a malformed card could not break
the index (the generator skips it) but could still reach the read path, so "the index is correct" and
"the KB is correct" were different claims and only one had a test.

Pure stdlib on a throwaway tmp dir: no GPU, no agent, no repo mutation, no network.

    python3 kernel_workflow/scripts/test_kb.py
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kb", os.path.join(HERE, "kb.py"))
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def card(name, **over):
    fm = {
        "name": name,
        "description": f"{name}: a lever on some op; +14% on the large shapes.",
        "keywords": "[gather, prologue]",
        "kernels": "[some_kernel]",
        "platforms": "[gfx950]",
        "kernel_class": "moe_grouped_gemm",
        "regime": "large-batch",
        "key": "bf16 fused-MoE grouped GEMM on gfx950, vLLM, large token counts",
        "lifecycle": "active",
        "type": "lever",
        "confidence": "★★",
        "effect": "+11.5% geomean; per-case +14.8% and +14.3% on the two large token counts.",
        "confirms_cited": "0", "confirms_blind": "1", "losses": "0", "attempts": "6",
        "source": "campaign20 2026-08-11", "last_seen": "2026-08-11",
    }
    fm.update({k: str(v) for k, v in over.items()})
    body = "# " + name + "\n- lever: change who produces the operand.\n- verify: confirm it engaged.\n"
    return "---\n" + "\n".join(f"{k}: {v}" for k, v in fm.items()) + "\n---\n" + body


def write(d, fname, text):
    with open(os.path.join(d, fname), "w") as f:
        f.write(text)


def run(d, *args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "kb.py"), "--kb-dir", d, *args],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def fresh():
    d = tempfile.mkdtemp(prefix="kbtest-")
    os.makedirs(os.path.join(d, "_inbox"), exist_ok=True)
    write(d, "_archive.md", "")
    return d


# --- index contract (inherited from test_learned_index.js) -------------------
d = fresh()
write(d, "a-card.md", card("a-card"))
write(d, "b-card.md", card("b-card", confidence="★★★", kernel_class="dense_gemm"))
write(d, "m-card.md", card("m-card", kernel_class="method"))
run(d, "index")
idx = open(os.path.join(d, "INDEX.md")).read()

check("index is derived from the cards (every active card appears)",
      all(f"({n}.md)" in idx for n in ("a-card", "b-card", "m-card")))
check("index carries the card's own description, kernels and keywords",
      "a-card: a lever on some op" in idx and "kernels: some_kernel" in idx and "kw: gather" in idx)
check("grouping is by kernel_class", "## moe_grouped_gemm" in idx and "## dense_gemm" in idx)
check("cross-cutting 'method' group sorts last",
      idx.index("## method") > max(idx.index("## dense_gemm"), idx.index("## moe_grouped_gemm")))
check("keyword vocabulary appendix is published", "## keyword vocabulary" in idx)
check("regeneration is deterministic", build := kb.build_index(d), "")
check("regeneration is byte-stable", kb.build_index(d) == build)

write(d, "z-card.md", card("z-card", keywords="[split_k, Split K, splitk]"))
run(d, "index")
idx2 = open(os.path.join(d, "INDEX.md")).read()
check("keywords are normalized mechanically (Split K -> split-k)", "split-k" in idx2)
check("surviving near-duplicate spellings are FLAGGED, not auto-merged",
      "Near-duplicate keywords" in idx2 and "splitk" in idx2)

# A lost append is the failure the generated index exists to prevent: hand-mangling INDEX.md must not
# survive a regen, and --check must notice before anyone trusts it.
write(d, "INDEX.md", "# hand-edited nonsense\n")
rc, _, _ = run(d, "index", "--check")
check("--check reports a stale index (exit 1)", rc == 1)
check("--check writes nothing when stale",
      open(os.path.join(d, "INDEX.md")).read() == "# hand-edited nonsense\n")
run(d, "index")
rc, _, _ = run(d, "index", "--check")
check("--check reports up-to-date after a regen (exit 0)", rc == 0)

write(d, "a-card.md", card("a-card", lifecycle="archived"))
run(d, "index")
idx3 = open(os.path.join(d, "INDEX.md")).read()
check("an archived card leaves the index", "(a-card.md)" not in idx3)
check("...but keeps its file (it holds the evidence that retired it)",
      os.path.exists(os.path.join(d, "a-card.md")))
shutil.rmtree(d)


# --- admission gates: delete the check and this test must go red -------------
def gate(name, expect_substr, **over):
    dd = fresh()
    write(dd, "c.md", card("c", **over))
    _, out, _ = run(dd, "lint", "--cards")
    fails = json.loads(out)["failures"].get("c.md", [])
    check(f"gate: {name}", any(expect_substr in f for f in fails),
          f"got {fails!r}")
    shutil.rmtree(dd)


gate("wall-clock is refused", "wall-clock",
     effect="0.0140 ms baseline; per-case +14.8% on the large shapes.")
gate("absolute throughput is refused", "absolute throughput",
     effect="1451 TFLOP/s; per-case +14.8% on the large shapes.")
gate("absolute bandwidth is refused", "absolute bandwidth",
     effect="4.9 TB/s sustained; per-case +14.8% on the large shapes.")
gate("a mandate is refused", "a mandate", effect="you must use this; per-case +14.8% on large shapes.")
gate("an eval-dir path is refused", "eval-dir", source="/shared_nfs/exp/kb_on_0810 run")
gate("a bare class·gfx·regime key is refused", "bare class",
     key="moe_grouped_gemm · gfx950 · large-batch")
gate("an over-long description is refused", "description is", description="x" * 170)
gate("an empty list in the header is refused", "missing 'keywords'", keywords="[]")
gate("an unknown lifecycle is refused", "lifecycle must be", lifecycle="retired")
gate("a bare geomean with no per-case evidence is refused", "per-case", effect="1.15x geomean.")
gate("★★★ without a blind confirmation is refused", "confirms_blind",
     confidence="★★★", confirms_blind="0")

# The audit must SEE the cards most likely to be broken. all_cards() filters to active for every other
# caller, so an unknown lifecycle would otherwise make a card invisible to the very check that would
# have flagged it.
dd = fresh()
write(dd, "c.md", card("c", lifecycle="archived", effect="0.0140 ms and nothing else."))
_, out, _ = run(dd, "lint", "--cards")
check("the audit inspects archived cards too",
      "c.md" in json.loads(out)["failures"])
shutil.rmtree(dd)

# A well-formed card must pass: a gate that rejects everything is not a gate.
dd = fresh()
write(dd, "c.md", card("c"))
_, out, _ = run(dd, "lint", "--cards")
check("a well-formed card is admitted", json.loads(out)["cards_failing"] == 0,
      json.dumps(json.loads(out)["failures"]))
shutil.rmtree(dd)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("all green")
