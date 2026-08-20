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
    title = over.pop("_title", name)
    body = "# " + str(title) + "\n- lever: change who produces the operand.\n- verify: confirm it engaged.\n"
    return "---\n" + "\n".join(f"{k}: {v}" for k, v in fm.items()) + "\n---\n" + body


def write(d, fname, text):
    with open(os.path.join(d, fname), "w") as f:
        f.write(text)


def run(d, *args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "kb.py"), "--kb-dir", d, *args],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def cite(d, run_id, kernel, payload):
    r = subprocess.run([sys.executable, os.path.join(HERE, "kb.py"), "--kb-dir", d, "cite",
                        "--run-id", run_id, "--kernel", kernel, "--citations", "-"],
                       input=payload, capture_output=True, text=True, timeout=120)
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

# `source` is provenance and is exempt from the class-level kernel-name rule. Requiring both "every
# claim needs a run id" and "never name a kernel" is unsatisfiable when run ids are named after their
# kernel, which is how real campaigns name them. Delete the exemption and this goes red.
dd = fresh()
write(dd, "c.md", card("c", source="run _fwd_grouped_kernel_stage1-chuschen16h 2026-08-11"))
_, out, _ = run(dd, "lint", "--cards")
check("a kernel symbol in `source` is allowed (provenance, not the principle)",
      json.loads(out)["cards_failing"] == 0, json.dumps(json.loads(out)["failures"]))
shutil.rmtree(dd)
gate("a bare class·gfx·regime key is refused", "bare class",
     key="moe_grouped_gemm · gfx950 · large-batch")
gate("an over-long description is refused", "description is", description="x" * 170)
gate("an empty list in the header is refused", "missing 'keywords'", keywords="[]")
gate("an unknown lifecycle is refused", "lifecycle must be", lifecycle="retired")
gate("a bare geomean with no per-case evidence is refused", "per-case", effect="1.15x geomean.")
gate("★★★ without a blind confirmation is refused", "confirms_blind",
     confidence="★★★", confirms_blind="0")
# Optional fields: the lint checks FORMAT, never presence. A field documented as part of the schema
# that no card carries and no gate wants teaches a curator that the schema is approximate.
gate("a malformed cost tier is refused", "cost must be", cost="L9")

# A missing title is written out as `none-<scope>.md` with an H1 of "# None" — non-empty, so a
# truthiness check passes it. Two cards reached disk that way on the first real bulk import.
dd = fresh()
write(dd, "c.md", card("c", _title="None"))
_, out, _ = run(dd, "lint", "--cards")
check("a literal 'None' title is refused (str(None), not a title)",
      any("no usable title" in f for f in json.loads(out)["failures"].get("c.md", [])),
      json.dumps(json.loads(out)["failures"]))
shutil.rmtree(dd)

# ...and the audit must read the title from the body's H1, not substitute the filename, or the check
# above can never fire on a real card.
dd = fresh()
write(dd, "c.md", card("c").replace("# c\n", ""))
_, out, _ = run(dd, "lint", "--cards")
check("a card with no H1 at all is refused",
      any("no usable title" in f for f in json.loads(out)["failures"].get("c.md", [])),
      json.dumps(json.loads(out)["failures"]))
shutil.rmtree(dd)
gate("an unparseable verified_on is refused", "verified_on must be", verified_on="yesterday")

# `kernels` must stay OPTIONAL. Requiring a field the writer cannot derive does not produce blanks,
# it produces plausible fiction: requiring this one while migrating class-level cards invented 11
# symbols, one of them in 17 cards. Deleting the exemption must turn this red.
dd = fresh()
write(dd, "c.md", card("c", kernels="[]"))
_, out, _ = run(dd, "lint", "--cards")
check("an empty kernels list is ALLOWED (a grep aid, not a provenance claim)",
      json.loads(out)["cards_failing"] == 0, json.dumps(json.loads(out)["failures"]))
shutil.rmtree(dd)

# The audit must SEE the cards most likely to be broken. all_cards() filters to active for every other
# caller, so an unknown lifecycle would otherwise make a card invisible to the very check that would
# have flagged it.
dd = fresh()
write(dd, "c.md", card("c", lifecycle="archived", effect="0.0140 ms and nothing else."))
_, out, _ = run(dd, "lint", "--cards")
check("the audit inspects archived cards too",
      "c.md" in json.loads(out)["failures"])
shutil.rmtree(dd)

# The eviction budget must be computed on the axis the index GROUPS by. Bucketing finer (adding
# platforms/regime) silently let one heading hold 11 cards under a cap of 8, while the generator
# warned about a limit nothing enforced. 12 cards of one class, 3 regimes: exactly 8 may stay active.
# Sized from CLASS_CAP, not from a literal: the first version hard-coded 12 cards against a cap of 8
# and went red the moment the cap was raised — a test that fails when a knob it is not testing
# changes is a test nobody will keep.
dd = fresh()
N = kb.CLASS_CAP + 4
for i in range(N):
    write(dd, f"k{i}.md", card(f"k{i}", regime=["decode", "mixed", "large-batch"][i % 3],
                               last_seen=f"2026-08-{(i % 28) + 1:02d}"))
    prop = {"run_id": f"r{i}", "date": "2026-08-11", "kernel_names": [], "validation_status": "accepted",
            "box_quiet": True, "held_out": False, "citations": [], "cards": []}
    with open(os.path.join(dd, "_inbox", f"r{i}.json"), "w") as f:
        json.dump(prop, f)
run(dd, "drain", "--apply", "--validated-runs", str(N))
active = 0
for f in os.listdir(dd):
    if f.endswith(".md") and f not in ("INDEX.md", "README.md") and not f.startswith("_"):
        if "lifecycle: active" in open(os.path.join(dd, f)).read():
            active += 1
check("the per-class cap binds on the axis the index groups by",
      active == kb.CLASS_CAP, f"{active} active of {N} written, cap {kb.CLASS_CAP}")
shutil.rmtree(dd)

# `doctor` must report the cap on the axis `drain` ENFORCES it. It grouped by `key` instead, and a
# key is a plain-English sentence unique to each card, so every bucket held exactly one card:
# headroom read CLASS_CAP-1 forever and `classes_at_cap` was permanently empty. The one failure this
# check exists to catch — a class quietly at its budget, the next card in it evicting a sibling — was
# the failure it could never report. Fill one class to the cap and require doctor to say so.
dd = fresh()
for i in range(kb.CLASS_CAP):
    write(dd, f"full{i}.md", card(f"full{i}", key=f"a distinct plain-English situation number {i}"))
write(dd, "other.md", card("other", kernel_class="dense_gemm", key="some other situation entirely"))
_, out, _ = run(dd, "doctor")
rep = json.loads(out)
check("doctor reports a class at its cap",
      "moe_grouped_gemm" in rep["classes_at_cap"],
      f"classes_at_cap={rep['classes_at_cap']}")
check("doctor's headroom is counted per kernel_class, not per card",
      rep["class_headroom"].get("moe_grouped_gemm") == 0
      and rep["class_headroom"].get("dense_gemm") == kb.CLASS_CAP - 1,
      json.dumps(rep["class_headroom"]))
shutil.rmtree(dd)

# The citation ledger must actually reach the counters. `drain` has always known how to apply
# citations and nothing ever handed it any — the lane passed them to a curator prompt, and the same
# arithmetic sat in roles/update_experience.md as prose. Measured result across two campaigns: 292
# citations, 126 of them at or below the frozen baseline, 7 recorded losses. `kb.py cite` is the
# producer; this checks the whole path, ledger -> drain -> counters.
dd = fresh()
write(dd, "cited.md", card("cited", confirms_cited=0, confirms_blind=0, losses=0, attempts=1,
                           origin_kernels="[kern_a]"))
cites = json.dumps([
    {"card": "cited.md", "cited_then_verified": 0.0, "became_winner": False},      # a real loss
    {"card": "cited.md", "cited_then_verified": 1.4, "became_winner": False},      # neutral
])
rc, out, err = cite(dd, "r1", "kern_a", cites)
check("cite files a ledger into the inbox", rc == 0 and "citations" in out, err)
run(dd, "drain", "--apply", "--validated-runs", "1")
meta = open(os.path.join(dd, "cited.md")).read()
check("a failed citation is recorded as a loss",
      "losses: 1" in meta, meta.split("---")[1])
check("a neutral citation is an attempt and nothing more",
      "attempts: 3" in meta and "confirms_cited: 0" in meta, meta.split("---")[1])
shutil.rmtree(dd)

# ★★★ has to be reachable or the tier is decoration. It was not: `confirms_blind` was only ever
# granted by a run with the KB off, and every arm runs with it on, so doctor reported all 135 cards
# self-confirmed-only. The distinction that matters for leakage is same-kernel vs cross-kernel — a
# card winning on the kernel it was distilled from is memorisation; winning on another is transfer.
dd = fresh()
write(dd, "xfer.md", card("xfer", confirms_cited=0, confirms_blind=0, attempts=1,
                          origin_kernels="[kern_a]"))
win = json.dumps([{"card": "xfer.md", "cited_then_verified": 2.0, "became_winner": True}])
cite(dd, "r2", "kern_b", win)
run(dd, "drain", "--apply", "--validated-runs", "1")
meta = open(os.path.join(dd, "xfer.md")).read()
check("a win on a DIFFERENT kernel is a blind confirmation",
      "confirms_blind: 1" in meta and "confirms_cited: 0" in meta, meta.split("---")[1])
shutil.rmtree(dd)

dd = fresh()
write(dd, "self.md", card("self", confirms_cited=0, confirms_blind=0, attempts=1,
                          origin_kernels="[kern_a]"))
cite(dd, "r3", "kern_a", json.dumps([{"card": "self.md", "cited_then_verified": 2.0,
                                       "became_winner": True}]))
run(dd, "drain", "--apply", "--validated-runs", "1")
meta = open(os.path.join(dd, "self.md")).read()
check("a win on the card's OWN kernel is not blind",
      "confirms_blind: 0" in meta and "confirms_cited: 1" in meta, meta.split("---")[1])
shutil.rmtree(dd)

# Confirmations are earned, never asserted. 129 of 135 cards in the tree carry `confirms_cited: 1`
# written by their author at creation, which is why `rank`'s standing term cannot order anything.
dd = fresh()
prop = {"run_id": "assert1", "date": "2026-08-18", "kernel_names": ["k"],
        "validation_status": "accepted", "box_quiet": True, "held_out": False, "citations": [],
        "cards": [{"title": "T", "name": "t-card",
                   "key": "a plainly worded gfx950 situation with an op and a regime",
                   "type": "lever", "confidence": "★★", "effect": "+11% geomean over 4 shapes",
                   "attempts": 3, "confirms_cited": 1, "confirms_blind": 0, "losses": 0,
                   "description": "a lever on an op: +11% on the large shapes",
                   "keywords": ["tiling"], "kernels": [], "platforms": ["gfx950"],
                   "kernel_class": "dense_gemm", "regime": "decode", "lifecycle": "active",
                   "source": "campaign 2026-08-18", "last_seen": "2026-08-18",
                   "lever": "x", "verify": "y"}]}
pf = os.path.join(dd, "p.json")
with open(pf, "w") as f:
    json.dump(prop, f)
_, out, _ = run(dd, "lint", "--file", pf)
check("a new card may not assert its own confirmations",
      "confirms_cited must be 0" in out, out[:300])
shutil.rmtree(dd)

# MERGE IS BY KEY. It was `if <generated filename> in cards`, and the filename is slugify(title) plus
# a scope slug — so the same situation under two titles became two cards, each starting fresh
# counters. Reproduced in review of #411 and reproduced here: same key, different titles, one card.
dd = fresh()
for i, title in enumerate(("First wording of it", "Completely different wording")):
    prop = {"run_id": f"m{i}", "date": "2026-08-19", "kernel_names": ["kern_a"],
            "validation_status": "accepted", "box_quiet": True, "held_out": False, "citations": [],
            "cards": [{"title": title, "name": f"card-{i}",
                       "key": "one and the same plainly worded situation on gfx950",
                       "type": "lever", "confidence": "★★", "effect": "+11% geomean over 4 shapes",
                       "attempts": 2, "confirms_cited": 0, "confirms_blind": 0, "losses": 0,
                       "description": f"{title}: a lever on an op, +11% on the large shapes",
                       "keywords": ["tiling"], "kernels": [], "platforms": ["gfx950"],
                       "kernel_class": "dense_gemm", "regime": "decode", "lifecycle": "active",
                       "source": "campaign 2026-08-19", "last_seen": "2026-08-19",
                       "lever": "x", "verify": "y"}]}
    with open(os.path.join(dd, "_inbox", f"m{i}.json"), "w") as f:
        json.dump(prop, f)
run(dd, "drain", "--apply", "--validated-runs", "2")
files = [f for f in os.listdir(dd) if f.endswith(".md") and f not in ("INDEX.md", "README.md")
         and not f.startswith("_")]
check("two proposals with one key produce ONE card", len(files) == 1, f"got {sorted(files)}")
if len(files) == 1:
    meta = open(os.path.join(dd, files[0])).read()
    check("a merge does not fabricate a confirmation",
          "confirms_cited: 0" in meta and "confirms_blind: 0" in meta, meta.split("---")[1])
shutil.rmtree(dd)

# A verifier that produced no number is an ATTEMPT, not a loss. Encoding "no evidence" as 0 charged
# cards for crashed or timed-out verifiers.
dd = fresh()
write(dd, "noev.md", card("noev", confirms_cited=0, confirms_blind=0, losses=0, attempts=1,
                          origin_kernels="[kern_a]"))
cite(dd, "r9", "kern_a", json.dumps([{"card": "noev.md", "cited_then_verified": None,
                                      "became_winner": False}]))
run(dd, "drain", "--apply", "--validated-runs", "1")
meta = open(os.path.join(dd, "noev.md")).read()
check("a missing verifier result is not scored as a loss",
      "losses: 0" in meta and "attempts: 2" in meta, meta.split("---")[1])
shutil.rmtree(dd)

# `rank` must see a blind confirmation. It read only confirms_cited, so the cross-kernel transfer the
# whole design is built to demonstrate bought a card no standing at all.
blind_card = {"meta": {"confidence": "★★", "confirms_cited": 0, "confirms_blind": 1, "losses": 0,
                       "last_seen": str(kb.date.today())}}
plain_card = {"meta": {"confidence": "★★", "confirms_cited": 0, "confirms_blind": 0, "losses": 0,
                       "last_seen": str(kb.date.today())}}
check("a blind confirmation outranks no confirmation",
      kb.rank(blind_card) > kb.rank(plain_card),
      f"blind={kb.rank(blind_card):.3f} plain={kb.rank(plain_card):.3f}")

# `lint --cards` must FAIL loudly. It returned 0 unconditionally, so a CI step or a && chain saw
# success while printing failures.
dd = fresh()
write(dd, "bad.md", card("bad", effect="0.0140 ms baseline; per-case +14.8% on the large shapes."))
rc, _, _ = run(dd, "lint", "--cards")
check("lint --cards exits non-zero when a card fails", rc == 1, f"exit {rc}")
write(dd, "bad.md", card("good"))
rc, _, _ = run(dd, "lint", "--cards")
check("lint --cards still exits 0 on a clean tree", rc == 0, f"exit {rc}")
shutil.rmtree(dd)

# The cross-kernel rule is worthless to the cards already in the tree unless their provenance is
# recovered, and it must be RECOVERED, not invented: a guessed origin hands out blind credit nobody
# earned. `backfill-origins` derives it from the drained proposals, and reports what it cannot.
dd = fresh()
write(dd, "known.md", card("known", key="a situation the drained proposal also names",
                           confirms_blind=0))
write(dd, "orphan.md", card("orphan", key="a situation no proposal accounts for"))
with open(os.path.join(dd, "_inbox", "p.json.drained"), "w") as f:
    json.dump({"run_id": "p", "date": "2026-08-19", "kernel_names": ["kern_origin"],
               "validation_status": "accepted", "citations": [],
               "cards": [{"name": "known", "title": "known",
                          "key": "a situation the drained proposal also names"}]}, f)
_, out, _ = run(dd, "backfill-origins", "--apply")
rep = json.loads(out)
check("backfill attributes a card from its drained proposal",
      rep["backfilled"] == 1 and "origin_kernels: ['kern_origin']" in
      open(os.path.join(dd, "known.md")).read(), out[:300])
check("backfill leaves an unattributable card alone and NAMES it",
      rep["no_provenance"] == ["orphan.md"]
      and "origin_kernels" not in open(os.path.join(dd, "orphan.md")).read(),
      f"no_provenance={rep['no_provenance']}")

# ...and the recovered provenance actually unlocks the thing it was recovered for.
cite(dd, "later", "a_different_kernel",
     json.dumps([{"card": "known.md", "cited_then_verified": 2.0, "became_winner": True}]))
run(dd, "drain", "--apply", "--validated-runs", "1")
check("a backfilled card can now earn a blind confirmation",
      "confirms_blind: 1" in open(os.path.join(dd, "known.md")).read(),
      open(os.path.join(dd, "known.md")).read().split("---")[1])
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
