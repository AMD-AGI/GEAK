#!/usr/bin/env python3
"""Roll up the learned card trees into base rates, and publish them as `index/run_recurrence.md`.

Run from the repo root (or anywhere — paths are resolved from this file):

    python3 perf_knowledge/index/_gen_run_recurrence.py            # rewrite the digest
    python3 perf_knowledge/index/_gen_run_recurrence.py --check    # exit 1 if stale, write nothing
    python3 perf_knowledge/index/_gen_run_recurrence.py --json     # the same data, machine-readable

WHY THIS EXISTS
A learned card is one run's conclusion, and the card tree has no way to say "this held again".
`confirms_cited` counts re-citations of the SAME card, so the far more common shape — twelve
independent runs on twelve different kernels each concluding, separately, that the dispatch path was
already at its floor — leaves twelve cards and no aggregate anywhere. The single most useful thing
that pile knows is a base rate, and a base rate is exactly what a per-card format cannot hold.

So this reads all the cards and reports, per optimization axis: on how many DISTINCT kernels it paid,
on how many it was already closed, and which cards say so. That is the number a planner wants before
spending a round, and it lives in `perf_knowledge/` because it is a prior about the hardware and the
operator, not about any one run.

WHAT IT WILL NOT DO
It never restates a claim. Every row is counts plus links; the sentences stay in the cards. That is
deliberate and it is the whole safety argument: a generator that paraphrased 135 cards into curated
prose would be inventing the one thing nobody measured — the generalization — and `perf_knowledge`'s
consumption contract (advisory evidence and candidate cards, never final verdicts) forbids exactly
that. It also makes the
"ratios, never absolutes" rule automatic, since no card text is copied at all.

It reports BOTH sides of every axis, always. A digest of only the wins is a worse prior than no
digest: 56 of the 135 cards here are anti-patterns, and on several axes the closed side is the
majority verdict. Hiding that would sell an axis the tree has already measured shut.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
OUT = os.path.join(HERE, "run_recurrence.md")

# One card parser, imported rather than reimplemented. `learned/README.md` records what the second
# implementation cost the last time there were two: the JS reader parsed `keywords: []` as an empty
# list and the Python one as the truthy STRING "[]", so one side dropped every tag while the other
# reported the header complete, and neither side could see the disagreement.
#
# Loaded BY PATH, under a name nothing else uses: the repo also has a root `kb/` PACKAGE, and
# `sys.path.insert` loses to an already-populated `sys.modules["kb"]`. In any process that imported
# the package first -- `kb/tests/test_store_env.py` does, in the same pytest session -- this module
# would bind the package and die later on `kb.all_cards`.
def _load_card_parser():
    import importlib.util
    path = os.path.join(ROOT, "kernel_workflow", "scripts", "kb.py")
    spec = importlib.util.spec_from_file_location("_gen_run_recurrence_kb", path)
    assert spec and spec.loader, f"cannot load the card parser at {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "all_cards"), f"{path} has no all_cards(); it is not the card parser"
    return mod


kb = _load_card_parser()

# The sinks whose gate is a kernel-level A/B. Both learned trees are read: e2e cards carry
# throughput evidence for the same axes, and an axis measured at both levels is a stronger prior
# than one measured at either. Which tree a card came from is kept and printed per row.
#
# `language_gated` is whether that tree's WRITE PATH enforces `language:` on a new card. The kernel
# tree goes through `kb.py propose`, whose lint refuses a card without one; the e2e tree is
# hand-maintained markdown with no such gate, so its cards keep arriving without a language. Saying
# "cards written from here on carry a language" across both was false for a sixth of the corpus.
# Named rather than a bare tuple: `("kernel", path, True)` does not say what the True is.
Tree = collections.namedtuple("Tree", "name root language_gated")

CARD_TREES = (
    Tree("kernel", os.path.join(ROOT, "kernel_workflow", "knowledge", "learned"), True),
    Tree("e2e", os.path.join(ROOT, "e2e_workflow", "knowledge", "learned"), False),
)

# Publish threshold, counted in DISTINCT KERNELS, never in cards: one campaign distils into many
# cards sharing a few kernel symbols, so a card count answers "how much did we write" when the
# question is "how many separate things did we measure". Three is the smallest number for which "it
# keeps happening" is defensible; below it the row still appears under `pending`, with its count.
MIN_KERNELS = 3
# An axis is a cross-operator prior only if it was seen in this many distinct kernel classes.
# Otherwise it is knowledge about one operator, and it routes there.
MIN_CLASSES_FOR_GENERAL = 3
# Above this share of an axis's kernels in one class, the axis is that operator's, not the tree's.
OPERATOR_CONCENTRATION = 0.8

# `type:` is the verdict. Named here because "lever" and "anti-pattern" say nothing to a reader who
# has not read the card schema, and the digest is read by people who have not.
VERDICT = {
    "lever": "paid",
    "anti-pattern": "closed",
    "method": "how to measure",
    "routing": "backend choice",
}
VERDICT_ORDER = ("paid", "closed", "how to measure", "backend choice", "unclassified")

# Keywords that restate a field the card header already carries: scope, not axes. Every card here
# is `gfx950`, so an axis whose evidence is "86 cards mention gfx950" describes the box we own. Most
# of the set is DERIVED (see `scope_vocabulary`); this table holds only the ones a curator spelled
# differently from the field they duplicate, and names that field. `test_run_recurrence.py` asserts
# every value is a real card field — a growing hand-list here is vocabulary drift.
HEADER_SYNONYMS = {
    "anti-pattern": "type",       # `type: anti-pattern`
    "closed-axis": "type",        # the curator's other word for the same verdict
    "lever": "type",
    "method": "type",
    "latency-bound": "regime",    # a bound class; REGIME_VOCAB carries the other three
}

# kernel_class (learned cards) -> operator id (perf_knowledge taxonomy). A bridge between two
# vocabularies, so it is explicit and it is checked: every target must exist under `operators/`.
# Classes deliberately absent map to nothing and are REPORTED as unroutable rather than filed
# somewhere plausible — see `kb.normalize_class` for what silent mis-filing costs. `method` and
# `composable` have no operator by construction: they are about how to measure, not what.
CLASS_TO_OPERATOR = {
    "dense_gemm": "dense_gemm",
    "quantized_gemm": "scaled_quant_gemm",
    "moe_grouped_gemm": "fused_moe_grouped_gemm",
    "attention_decode": "attention_decode_paged",
    "attention": "attention_prefill_fmha",
    "linear_attention": "linear_attention_gated_delta",
    "quantize_cast": "quant_dequant_fp8",
    "moe_router_topk": "moe_routing_topk",
    "topk_router": "moe_routing_topk",
}


def taxonomy_ids():
    """Every backtick-quoted id in `index/taxonomy.md`, plus the FAMILY of each dtype id.

    Read rather than restated: the file says its ids are authoritative, and a second copy of them
    here would be the drift this whole module is trying to measure.

    The family split is dtypes ONLY, and the section is parsed to keep it that way. Cards tag `fp8`
    where the taxonomy has `fp8_e4m3_fnuz` — same scope, finer id — so dtypes need it. Applying it
    to every section was the first version and it silently deleted a real axis: the operator id
    `splitk_streamk_gemm` contributed `splitk`, which is how `split-k` — one of the levers this
    tree has measured most — stopped being an axis at all. An over-broad scope filter fails in the
    direction that is hardest to notice, since the row simply is not there to look wrong.
    """
    path = os.path.join(HERE, "taxonomy.md")
    ids, section = set(), ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                section = line[3:].strip().lower()
            for tok in re.findall(r"`([a-z0-9_]+)`", line):
                ids.add(tok)
                if section.startswith("dtypes") and "_" in tok:
                    ids.add(tok.split("_")[0])
    return ids


def fold(term):
    """`split_k` / `Split K` / `split--k` are one concept. Same normalization the index generator
    uses, imported for the same reason the card parser is.

    It is a matching key and NOT a label: it deletes hyphens and a trailing `s`, so `isa-census`
    folds to `isacensu` and `num-stages` to `numstage`. Every table here groups by the fold and
    prints the card's own most common spelling, because a digest whose row headings are visibly
    mangled words reads as a broken tool no matter how right the counts are.
    """
    return kb._kw_fold(term)


def scope_vocabulary(cards):
    """Folded terms that describe WHERE a card applies rather than WHAT was tried.

    Derived from the taxonomy plus the header values actually present in the trees, so it tracks the
    vocabulary instead of freezing a snapshot of it: add a gfx or a kernel class and its keyword
    stops being mistaken for an axis without anyone editing this file.
    """
    scope = {fold(t) for t in taxonomy_ids()}
    scope |= {fold(r) for r in kb.REGIME_VOCAB}
    scope |= {fold(k) for k in HEADER_SYNONYMS}
    for c in cards:
        m = c["meta"]
        for field in ("kernel_class", "regime", "type", "language", "toolchain"):
            if m.get(field):
                scope.add(fold(m[field]))
        for field in ("platforms",):
            for v in m.get(field) or []:
                scope.add(fold(v))
    # A bound class is a regime however it is spelled: `roofline` is an axis (go measure the roof),
    # `memory-bound` is a place on it.
    return scope


def load_cards():
    out = []
    for t in CARD_TREES:
        if not os.path.isdir(t.root):
            continue
        for c in kb.all_cards(type("KB", (), {"root": t.root})()):
            c["tree"] = t.name
            out.append(c)
    return out


def axis_rows(cards, scope):
    """One row per axis keyword, holding the distinct kernels behind each verdict.

    Distinct kernels, not cards, at every level — see MIN_KERNELS. A card with no `kernels` still
    counts toward its verdict via a synthetic id built from its own name, because dropping it would
    quietly shrink the denominator of a base rate; it is counted once and cannot inflate anything.

    `classes` maps kernel_class -> the distinct kernels seen under it, so concentration is a share
    of kernels and not a share of cards. Those differ by a lot: one campaign writes several cards
    over the same three kernels, and counting cards would read that as a broadly-measured axis.
    """
    rows = {}
    for c in cards:
        m = c["meta"]
        verdict = VERDICT.get(str(m.get("type")), "unclassified")
        kernels = {str(k) for k in (m.get("kernels") or [])} or {f"card:{m.get('name')}"}
        for raw in m.get("keywords") or []:
            axis = fold(raw)
            if not axis or axis in scope:
                continue
            r = rows.setdefault(axis, {
                "axis": axis,
                "spellings": collections.Counter(),
                "kernels_by_verdict": collections.defaultdict(set),
                "kernels": set(),
                "classes": collections.defaultdict(set),
                "platforms": set(),
                "languages": set(),
                "trees": set(),
                "cards": [],
                "losses": 0,
                "confirms": 0,
            })
            r["spellings"][str(raw)] += 1
            r["kernels_by_verdict"][verdict] |= kernels
            r["kernels"] |= kernels
            r["classes"][kb.class_of(m)] |= kernels
            r["platforms"] |= {str(p) for p in (m.get("platforms") or [])}
            r["trees"].add(c["tree"])
            if m.get("language"):
                r["languages"].add(str(m["language"]))
            r["cards"].append({"tree": c["tree"], "name": str(m.get("name")),
                               "slug": os.path.splitext(os.path.basename(c["path"]))[0],
                               "verdict": verdict, "class": kb.class_of(m),
                               "confidence": str(m.get("confidence", ""))})
            r["losses"] += int(m.get("losses", 0) or 0)
            r["confirms"] += (int(m.get("confirms_cited", 0) or 0)
                              + int(m.get("confirms_blind", 0) or 0))
    for r in rows.values():
        r["label"] = r["spellings"].most_common(1)[0][0]
        r["n_kernels"] = len(r["kernels"])
        r["n_classes"] = len(r["classes"])
        by_size = sorted(r["classes"].items(), key=lambda kv: (-len(kv[1]), kv[0]))
        r["top_class"] = by_size[0][0]
        r["concentration"] = len(by_size[0][1]) / r["n_kernels"]
        r["scope"] = ("general" if r["n_classes"] >= MIN_CLASSES_FOR_GENERAL
                      else "operator" if r["concentration"] >= OPERATOR_CONCENTRATION
                      else "narrow")
    return rows


def arch_split(rows):
    """Axes whose verdict differs BY PLATFORM — the architecture-difference lane.

    Needs two platforms in one axis to say anything at all. With a single-gfx tree this is
    structurally empty, and the digest says so with the platform census rather than omitting the
    section: "no rows" and "no evidence that could produce a row" are different states, and a reader
    who cannot tell them apart will read the first as "nothing differs across architectures".
    """
    out = []
    for r in rows.values():
        if len(r["platforms"]) < 2:
            continue
        out.append({"axis": r["axis"], "label": r["label"],
                    "platforms": sorted(r["platforms"]), "cards": r["cards"]})
    return out


def card_link(c):
    """Cards live in the tree that measured them, so the link has to name the tree.

    Relative to HERE, the directory the digest is written into — not to `perf_knowledge/`. Getting
    that wrong is silent: `../kernel_workflow/...` from `perf_knowledge/index/` resolves to
    `perf_knowledge/kernel_workflow/...`, and the first version of this file shipped 135 links that
    all pointed one level short. `test_run_recurrence.py` resolves every link on disk now, because
    a broken audit trail is worse than a missing one — it looks checkable and is not.

    The TARGET is the card's filename, never its `name:`. Both tree READMEs say the two are the
    same string, and in the kernel tree `kb.py propose` enforces it — but the e2e tree is
    hand-maintained, and four of its cards carry a prose `name:` ("mxfp4 fused-MoE on sglang+aiter
    (gfx950) — ..."). Linking by `name:` produced four targets that do not exist and, because the
    prose holds parentheses, markdown truncated two of them mid-sentence. The filename is what
    resolves, so the filename is what we link; `name:` stays as the visible label.
    """
    roots = {t.name: t.root for t in CARD_TREES}
    root = os.path.relpath(roots.get(c["tree"], ""), HERE)
    return f"[{c['name']}]({root}/{c['slug']}.md)"


# The prose lives here rather than inline so `render()` reads as layout and the wording can be
# reviewed as wording. Every `{}` is filled from the corpus, never hardcoded — a threshold quoted in
# prose that no longer matches the constant is how a generated file starts lying.
PROSE = {
    "intro": (
        "A learned card is one run's conclusion. This file is the part no single card can hold: for "
        "each optimization axis, on how many *distinct kernels* it paid, and on how many it was "
        "already closed. Rows are counts and links only — the claims stay in the cards, because the "
        "generalization is the one thing no run measured."
    ),
    "contract": (
        "Read it as a **prior about where to look first**, never as a verdict. "
        "[`README.md`](../README.md) applies unchanged: this base proposes candidates, the box decides."
    ),
    "general_intro": (
        "An axis reaches this table at >= {min_kernels} distinct kernels and >= {min_classes} "
        "distinct kernel classes: measured widely enough that it is a statement about the hardware "
        "rather than about one operator."
    ),
    "general_note": (
        "`paid` / `closed` count DISTINCT KERNELS, and they overlap on purpose: the same axis can "
        "pay on one kernel and be shut on another, which is the base rate. Where `closed` leads, "
        "the axis is not useless — it is the one to price before funding a round on it."
    ),
    "operator_intro": (
        "Concentrated (>= {pct}% of the axis's kernels in one class) and therefore knowledge about "
        "that operator. `operator` is the [`taxonomy.md`](taxonomy.md) id, so the row points at a "
        "directory that exists."
    ),
    "unroutable": (
        "Kernel classes with cards but no operator id, left unfiled rather than filed plausibly: "
        "{classes}. `method` and `composable` have no operator by construction — they are about how "
        "to measure, and a method row filed under an operator would be read as being about that "
        "operator. `other` is the parser's fallback for a card with no `kernel_class` at all, which "
        "is every e2e card. Anything else here wants a line in `CLASS_TO_OPERATOR`."
    ),
    "language_intro": (
        "The lane that lets a FlyDSL run see FlyDSL evidence instead of Triton evidence. It is fed by "
        "the card `language:` field, which is deliberately NOT backfilled onto older cards — a guessed "
        "language is worse than an absent one. Which write paths require it is not uniform; the "
        "breakdown below says which."
    ),
    "language_empty": (
        "**Empty, and not because nothing was measured:** 0 of {n} active cards carry a `language:` "
        "field. It fills from one side only, so which side matters:\n"
        "\n"
        "{per_tree}\n"
        "\n"
        "So this lane will populate for {enforced} cards as they land, and will stay silent for "
        "{unenforced} ones until that write path gains the same gate. Read a future entry here as "
        "evidence from {enforced} specifically — not as the whole corpus having been language-tagged."
    ),
    "language_tree_enforced": (
        "- **{tree}** ({n} cards today): `language:` is required. `kb.py propose` refuses a new card "
        "without one, and the lane fills it from `kernel_workflow/scripts/detect_language.py` reading "
        "the produced source rather than echoing the request."
    ),
    "language_tree_open": (
        "- **{tree}** ({n} cards today): `language:` is **not** required — these cards are "
        "hand-maintained markdown with no lint gate and no detection step on the write path, so they "
        "will keep arriving without one. A real gap, named here rather than averaged away."
    ),
    "arch_intro": (
        "Axes whose verdict differs across `gfx` targets — the rows worth carrying into "
        "[`hardware/`](../hardware/). Needs one axis measured on two platforms."
    ),
    "arch_empty": (
        "**Structurally empty on this corpus, not measured-and-equal:** the platform census is "
        "{census}. A row here needs one axis measured on two gfx targets; with a single-platform "
        "corpus no comparison exists to report. Read this as *unmeasured*, not as *no difference*."
    ),
    "pending_intro": (
        "Seen, but on fewer than {min_kernels} distinct kernels. Listed so the pipeline is visible: "
        "an axis absent from this file entirely has never been tried, which is a different thing "
        "from tried twice."
    ),
    "evidence_intro": (
        "Every published axis with the cards behind it, so a row can be audited back to the run "
        "that measured it. `paid` / `closed` / `how to measure` is the card's `type:`."
    ),
    "evidence_format": (
        "One card per line on purpose: this file is regenerated on every run that writes a card and "
        "CI diffs it, so a single new card should produce a one-line diff rather than rewrite a "
        "paragraph."
    ),
    "banner": (
        "**GENERATED. Never hand-edit.** Regenerate with "
        "`python3 perf_knowledge/index/_gen_run_recurrence.py`; CI runs it with `--check`."
    ),
    "sources": [
        ("- Generated from the learned card trees: `kernel_workflow/knowledge/learned/` and "
         "`e2e_workflow/knowledge/learned/` (schema and evidence rules: their `README.md`)."),
        ("- Card evidence is a frozen-baseline isolated A/B plus oracle parity (kernel tree) or the "
         "e2e Director's A/B (e2e tree). No number is copied here; each row links the cards."),
        ("- Axis vocabulary is filtered against [`taxonomy.md`](taxonomy.md) and the card header "
         "fields, so scope tags (`gfx950`, `decode`, `fp8`) do not appear as axes."),
    ],
}


def render(rows, cards, unroutable):
    published = {k: v for k, v in rows.items() if v["n_kernels"] >= MIN_KERNELS}
    pending = {k: v for k, v in rows.items() if v["n_kernels"] < MIN_KERNELS}
    platforms = collections.Counter(p for c in cards for p in (c["meta"].get("platforms") or []))
    languages = collections.Counter(str(c["meta"].get("language")) for c in cards
                                    if c["meta"].get("language"))
    # Per tree, how many cards carry the discovery header this file reads. A tree whose cards have
    # no `keywords` contributes no axes, and stating that is the difference between "e2e measured
    # nothing" and "e2e cards predate the header schema" — only one of those is true.
    per_tree = ", ".join(
        "{}={} ({} with keywords)".format(
            t.name, sum(1 for c in cards if c["tree"] == t.name),
            sum(1 for c in cards if c["tree"] == t.name and c["meta"].get("keywords")))
        for t in CARD_TREES)

    L = ["---", "title: Run recurrence — base rates distilled from the learned card trees",
         "kind: reference", "generated_by: index/_gen_run_recurrence.py", "---", "",
         "# Run recurrence — what keeps holding across runs", "",
         PROSE["banner"], "", PROSE["intro"], "", PROSE["contract"], "",
         (f"Corpus: **{len(cards)} active cards** across {per_tree} · "
          f"**{len(published)} axes published** (>= {MIN_KERNELS} distinct kernels), "
          f"{len(pending)} below threshold."), ""]

    # ---- the material that exists: cross-operator base rates -------------------------------------
    L += ["## Cross-operator axes", "",
          PROSE["general_intro"].format(min_kernels=MIN_KERNELS,
                                        min_classes=MIN_CLASSES_FOR_GENERAL), "",
          "| axis | paid on | closed on | how to measure | kernels | classes | cards |",
          "|---|---|---|---|---|---|---|"]
    gen = sorted((r for r in published.values() if r["scope"] == "general"),
                 key=lambda r: (-r["n_kernels"], r["axis"]))
    for r in gen:
        v = r["kernels_by_verdict"]
        L.append(f"| `{r['label']}` | {len(v['paid'])} | {len(v['closed'])} | "
                 f"{len(v['how to measure'])} | {r['n_kernels']} | {r['n_classes']} | "
                 f"{len(r['cards'])} |")
    if not gen:
        L.append("| _(none yet)_ | | | | | | |")
    L += ["", PROSE["general_note"], ""]

    # ---- per-operator ----------------------------------------------------------------------------
    L += ["## Per-operator axes", "",
          PROSE["operator_intro"].format(pct=int(OPERATOR_CONCENTRATION * 100)), "",
          "| operator | axis | paid on | closed on | kernels | cards |", "|---|---|---|---|---|---|"]
    per_op = collections.defaultdict(list)
    for r in published.values():
        if r["scope"] != "operator":
            continue
        op = CLASS_TO_OPERATOR.get(r["top_class"])
        per_op[op or f"(unmapped: {r['top_class']})"].append(r)
    any_op = False
    for op in sorted(per_op):
        for r in sorted(per_op[op], key=lambda r: (-r["n_kernels"], r["axis"])):
            v = r["kernels_by_verdict"]
            link = (f"[`{op}`](../operators/{op}/)" if not op.startswith("(") else op)
            L.append(f"| {link} | `{r['label']}` | {len(v['paid'])} | {len(v['closed'])} | "
                     f"{r['n_kernels']} | {len(r['cards'])} |")
            any_op = True
    if not any_op:
        L.append("| _(none yet)_ | | | | | |")
    if unroutable:
        listed = ", ".join(f"`{c}` ({n} cards)" for c, n in sorted(unroutable.items()))
        L += ["", PROSE["unroutable"].format(classes=listed)]
    L.append("")

    # ---- language lane ---------------------------------------------------------------------------
    L += ["## By authoring language", "", PROSE["language_intro"], ""]
    if languages:
        L += ["| language | cards | axes |", "|---|---|---|"]
        for lang, n in sorted(languages.items(), key=lambda kv: (-kv[1], kv[0])):
            axes = sorted(r["label"] for r in published.values() if lang in r["languages"])
            L.append(f"| `{lang}` | {n} | {', '.join(f'`{a}`' for a in axes) or '—'} |")
    else:
        by_tree = collections.Counter(c["tree"] for c in cards)
        lines, enforced, unenforced = [], [], []
        for t in CARD_TREES:
            key = "language_tree_enforced" if t.language_gated else "language_tree_open"
            lines.append(PROSE[key].format(tree=t.name, n=by_tree.get(t.name, 0)))
            (enforced if t.language_gated else unenforced).append(f"`{t.name}`")
        L.append(PROSE["language_empty"].format(
            n=len(cards), per_tree="\n".join(lines),
            enforced=" and ".join(enforced) or "no",
            unenforced=" and ".join(unenforced) or "no"))
    L.append("")

    # ---- architecture lane -----------------------------------------------------------------------
    arch = arch_split(published)
    L += ["## By architecture", "", PROSE["arch_intro"], ""]
    if arch:
        L += ["| axis | platforms | cards |", "|---|---|---|"]
        for a in sorted(arch, key=lambda a: a["axis"]):
            L.append(f"| `{a['label']}` | {', '.join(a['platforms'])} | {len(a['cards'])} |")
    else:
        census = ", ".join(f"`{p}`={n}" for p, n in sorted(platforms.items())) or "empty"
        L.append(PROSE["arch_empty"].format(census=census))
    L.append("")

    # ---- pending ---------------------------------------------------------------------------------
    L += ["## Below threshold", "", PROSE["pending_intro"].format(min_kernels=MIN_KERNELS), ""]
    if pending:
        buckets = collections.defaultdict(list)
        for r in sorted(pending.values(), key=lambda r: r["label"]):
            buckets[r["n_kernels"]].append(r["label"])
        for n in sorted(buckets, reverse=True):
            L.append(f"- **{n} kernel(s)**: " + " · ".join(f"`{a}`" for a in buckets[n]))
    else:
        L.append("- _(none)_")
    L.append("")

    # ---- the evidence itself ---------------------------------------------------------------------
    L += ["## Evidence", "", PROSE["evidence_intro"], "", PROSE["evidence_format"], ""]
    for r in sorted(published.values(), key=lambda r: (-r["n_kernels"], r["axis"])):
        spell = [s for s, _ in r["spellings"].most_common()]
        alias = f" (also spelled {', '.join(f'`{s}`' for s in spell[1:])})" if len(spell) > 1 else ""
        L.append(f"### `{r['label']}`{alias}")
        L.append(f"{r['n_kernels']} distinct kernels · {r['n_classes']} classes · scope "
                 f"`{r['scope']}` · platforms {', '.join(sorted(r['platforms'])) or '—'}")
        L.append("")
        for verdict in VERDICT_ORDER:
            cs = sorted((c for c in r["cards"] if c["verdict"] == verdict), key=lambda c: c["name"])
            if not cs:
                continue
            L.append(f"**{verdict}** — {len(r['kernels_by_verdict'][verdict])} kernels, "
                     f"{len(cs)} cards:")
            L += [f"- {card_link(c)}" for c in cs]
        L.append("")

    L += ["## Sources", *PROSE["sources"], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if run_recurrence.md is stale and write nothing (for CI)")
    ap.add_argument("--json", action="store_true", help="dump the rows instead of rendering")
    a = ap.parse_args()

    cards = load_cards()
    scope = scope_vocabulary(cards)
    rows = axis_rows(cards, scope)
    unroutable = collections.Counter(
        kb.class_of(c["meta"]) for c in cards
        if kb.class_of(c["meta"]) not in CLASS_TO_OPERATOR)

    if a.json:
        def clean(r):
            d = {k: v for k, v in r.items() if k != "kernels_by_verdict"}
            d["spellings"] = [s for s, _ in r["spellings"].most_common()]
            d["kernels"] = sorted(r["kernels"])
            d["platforms"] = sorted(r["platforms"])
            d["languages"] = sorted(r["languages"])
            d["trees"] = sorted(r["trees"])
            d["classes"] = {k: sorted(v) for k, v in r["classes"].items()}
            d["verdicts"] = {k: sorted(v) for k, v in r["kernels_by_verdict"].items()}
            return d
        print(json.dumps({"cards": len(cards), "min_kernels": MIN_KERNELS,
                          "unroutable_classes": dict(unroutable),
                          "axes": [clean(r) for r in sorted(rows.values(),
                                                            key=lambda r: -r["n_kernels"])]},
                         ensure_ascii=False, indent=2))
        return 0

    text = render(rows, cards, unroutable)
    published = sum(1 for r in rows.values() if r["n_kernels"] >= MIN_KERNELS)
    if a.check:
        old = None
        if os.path.exists(OUT):
            with open(OUT, encoding="utf-8") as f:
                old = f.read()
        if old == text:
            print(f"OK: {os.path.relpath(OUT, ROOT)} is up to date "
                  f"({len(cards)} cards, {published} axes)")
            return 0
        print(f"STALE: {os.path.relpath(OUT, ROOT)} does not match the cards — regenerate with "
              f"`python3 perf_knowledge/index/_gen_run_recurrence.py`", file=sys.stderr)
        return 1
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OK: {len(cards)} cards -> {published} axes published "
          f"({len(rows) - published} below threshold) -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
