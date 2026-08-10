#!/usr/bin/env python3
"""Learned-knowledge store. ONE implementation, serving every `knowledge/learned/` tree in the repo.

    kb.py --kb-dir <tree> match   --operator <op> --device <str> [--regime <r>] [--max 3] [--explain]
    kb.py --kb-dir <tree> lint    --file <proposal.json>        # validate a proposal
    kb.py --kb-dir <tree> lint    --cards                       # audit the cards already in the tree
    kb.py --kb-dir <tree> propose --file <proposal.json>        # validate + place in _inbox/
    kb.py --kb-dir <tree> drain   [--apply] [--validated-runs N]
    kb.py --kb-dir <tree> doctor  [--toolchain <fingerprint>]
    kb.py --kb-dir <tree> stats

`--kb-dir` is REQUIRED and has no default. There are two trees in this repo
(`kernel_workflow/knowledge/learned/` for kernel-level levers, `e2e_workflow/knowledge/learned/` for
e2e routing/config) and they must not be mixed. A default would quietly make this a single-tree tool
again, which is how the contract drifted apart in the first place: two copies of a rule, and the
second one lapsed — the e2e INDEX now carries a "MANDATED LEVER" and a "do NOT use it" that its own
README forbids. One implementation, two data dirs.

Why the commands split the way they do
--------------------------------------
`match` and `propose` run inside a live campaign — up to 8 drivers per host, on two hosts sharing one
NFSv3 mount. Neither takes a lock: `match` only reads, and `propose` creates one file whose name
contains the run id, so two writers can never collide.

`drain` is the ONLY writer of INDEX.md / cards / _archive.md, run by one operator between campaigns.
That is not merely lock-avoidance: "MERGE if the key already exists" is not implementable
concurrently, because 20 curators cannot see each other's proposals and would each insert a
near-duplicate for the same key. One writer holding the whole inbox dedupes correctly, and gives a
human a review gate over what enters the KB.
"""
import argparse
import glob as globmod
import json
import os
import re
import sys
from datetime import date, datetime

INDEX_CAP = 40
STARS = {"★": 1, "★★": 2, "★★★": 3}
STAR_OF = {1: "★", 2: "★★", 3: "★★★"}


class KB:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise SystemExit(f"--kb-dir {self.root} does not exist")
        self.inbox = os.path.join(self.root, "_inbox")
        self.index = os.path.join(self.root, "INDEX.md")
        self.archive = os.path.join(self.root, "_archive.md")


# ---------------------------------------------------------------------------
# Key normalization. Here, not in an agent, on purpose.
#
# The inputs are unreliable: the operator id comes from the analyze phase and is nullable, and the
# device string is free text ("MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"). If two runs of the same
# kernel class produce two different key strings, nothing ever matches — and the failure is SILENT.
# You conclude "the KB doesn't help" when in truth it was never read. So keys go through a closed
# vocabulary, and anything unrecognised lands in an `unmatched` bucket a human inspects, rather than
# being quietly coerced into the nearest class.
# ---------------------------------------------------------------------------
CLASS_VOCAB = {
    "dense gemm": ["gemm_a16_w16", "dense gemm", "a16w16", "wvsplitk", "skinny gemm", "gemv"],
    "quantized gemm": ["a8w8", "w8a8", "blockscale", "fp8 gemm", "int4 gemm", "w4a16", "mxfp4",
                       "scaled_quant", "quantized gemm"],
    "moe grouped gemm": ["fused_moe", "moe_gemm", "moe_stage", "grouped gemm", "moe"],
    "attention": ["attention", "paged_attention", "flash", "mla", "sdpa"],
    "linear attention": ["chunk_scaled_dot", "kkt", "fla", "mamba", "linear attention", "delta rule"],
    "quantize / cast": ["per_token_group_quant", "quant", "cast", "dynamic_quant"],
    "topk / routing": ["topk", "router", "argmax", "sort", "sampling"],
    "memory movement": ["write_req", "token_pool", "copy", "gather", "scatter", "reshape", "rope"],
    "reduction / norm": ["rmsnorm", "layernorm", "softmax", "reduce", "norm"],
}
REGIME_VOCAB = ["decode", "prefill", "mixed", "launch-bound", "memory-bound", "compute-bound",
                "small-batch", "large-batch", "unknown"]
UNMATCHED = "unmatched"


def normalize_class(operator, kernel_name=""):
    hay = f"{operator or ''} {kernel_name or ''}".lower()
    for cls, needles in CLASS_VOCAB.items():
        for n in needles:
            if n in hay:
                return cls
    return UNMATCHED


def normalize_gfx(device):
    m = re.search(r"(gfx\d+[a-z]*)", str(device or ""), re.I)
    return m.group(1).lower() if m else UNMATCHED


def normalize_regime(regime):
    r = str(regime or "").strip().lower()
    return r if r in REGIME_VOCAB else "unknown"


def make_key(operator, device, regime, kernel_name=""):
    return " · ".join([normalize_class(operator, kernel_name), normalize_gfx(device),
                       normalize_regime(regime)])


def slugify(s):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(s).lower())).strip("-")[:60]


# ---------------------------------------------------------------------------
# The lint. ONE implementation, called from `propose` (fail fast, at the source, while the run that
# can explain itself is still alive), from `drain` (defence against hand-written or older proposals),
# and from `lint --cards` (audit a tree that predates these rules). Three call sites, one rule.
# ---------------------------------------------------------------------------

# Instance identifiers. A card carrying one is memorising a specific run rather than distilling a
# principle. Campaigns re-run the SAME kernels, so such a card is read next time by the very kernel it
# came from: an A/B over it would look spectacular and mean nothing.
LEAK_PATTERNS = [
    (r"/exp/|eval_dir|/worktree/|\bexp/\w+", "an eval-dir or experiment path"),
    (r"\b\w*_patch\.diff\b|current_best\.diff|best_patch\.diff", "a patch-file path"),
    (r"\btest_cases\.json\b", "the harness test-case file"),
    (r"\b(?:perf|corr)_[A-Za-z]\d+_[A-Za-z]?\d+", "a verbatim harness case id"),
]
# Mandate / blocklist language. The contract is ADD-only: a card may add a candidate, never remove
# one. This is a check and not a paragraph because the paragraph already failed once.
MANDATE_PATTERNS = [
    (r"\bnever use\b|\bdo not use\b|\bdon't use\b|\bforbidden\b|\bbanned\b", "a prohibition"),
    (r"\bmandated?\b|\bmust use\b|\balways use\b|\brequired lever\b", "a mandate"),
    (r"\bdeprecated for this op\b|\bblocklist\b", "a blocklist"),
]
CARD_BODY_FIELDS = ("title", "lever", "apply", "verify", "caution", "effect", "source")
MAX_CARD_LINES = 15


def lint_card(card, kernel_names=(), strict_source=True):
    """Return a list of rejection reasons; empty == accepted."""
    errs = []
    text = "\n".join(str(card.get(f, "")) for f in CARD_BODY_FIELDS)

    for pat, what in LEAK_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            errs.append(f"leaks an instance identifier ({what}): {m.group(0)!r}")
    # The most direct leak, and the one a well-meaning curator writes without noticing.
    for kn in kernel_names:
        if kn and len(kn) > 4 and re.search(re.escape(kn), text, re.I):
            errs.append(f"names a specific kernel ({kn!r}); keys and bodies must be class level")
    for pat, what in MANDATE_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            errs.append(f"contains {what}: {m.group(0)!r} — a caution must read 'also verify X'")

    if strict_source and not str(card.get("source", "")).strip():
        errs.append("no source: every claim needs a run id + date")
    if not str(card.get("effect", "")).strip():
        errs.append("no effect")
    elif not re.search(r"\d", str(card["effect"])):
        errs.append("effect cites no number")
    # A geomean alone hides a lever that helped one shape and did nothing elsewhere. The director
    # returns per_case[], so there is no excuse for not saying where it held.
    elif not re.search(r"(per-case|shape|case|S\s*[<>=]|batch|decode|prefill|M=|N=|K=|conc)",
                       str(card["effect"]), re.I):
        errs.append("effect gives no per-case evidence (a bare geomean is not enough)")

    conf = str(card.get("confidence", ""))
    if conf not in STARS:
        errs.append(f"confidence must be one of {list(STARS)}, got {conf!r}")
    attempts = card.get("attempts")
    if not isinstance(attempts, int) or attempts < 1:
        errs.append("attempts must be an int >= 1 (the base rate is not optional)")
    # Self-confirmation cannot buy authority: a card only ever confirmed by runs it steered is capped
    # at two stars however many times it 'reproduces'.
    if STARS.get(conf, 0) >= 3 and int(card.get("confirms_blind", 0) or 0) < 1:
        errs.append("★★★ requires confirms_blind >= 1 (self-confirmation cannot promote)")

    body_lines = sum(len(str(card.get(f, "")).splitlines()) or 1 for f in CARD_BODY_FIELDS)
    if body_lines > MAX_CARD_LINES:
        errs.append(f"body is {body_lines} lines (>{MAX_CARD_LINES}): that is narrative, distil it")
    return errs


def lint_proposal(prop):
    """Whole-run gates: is this a run we are willing to learn from at all?"""
    errs = []
    if prop.get("validation_status") != "accepted":
        errs.append(f"validation_status={prop.get('validation_status')!r}; only 'accepted' may "
                    f"produce cards (a flagged run is the one most likely to overstate)")
    if prop.get("box_quiet") is False:
        errs.append("box_quiet=false: a contended run encodes contention as kernel physics")
    if not str(prop.get("run_id", "")).strip():
        errs.append("no run_id")
    if prop.get("held_out"):
        errs.append("kernel is in the HELD-OUT split; distilling from it destroys the A/B")
    return errs


# ---------------------------------------------------------------------------
# Card files: a tiny front-matter format (no PyYAML dependency on these boxes).
# ---------------------------------------------------------------------------
def read_card(path):
    raw = open(path).read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    for k in ("confirms_cited", "confirms_blind", "attempts", "losses"):
        try:
            meta[k] = int(meta.get(k, 0))
        except ValueError:
            meta[k] = 0
    return {"path": path, "meta": meta, "body": m.group(2)}


def write_card(path, meta, body):
    order = ["key", "type", "confidence", "effect", "confirms_cited", "confirms_blind", "losses",
             "attempts", "toolchain", "last_seen"]
    lines = ["---"]
    for k in order:
        if k in meta:
            lines.append(f"{k}: {meta[k]}")
    for k, v in meta.items():
        if k not in order:
            lines.append(f"{k}: {v}")
    lines.append("---")
    open(path, "w").write("\n".join(lines) + "\n" + body.rstrip() + "\n")


def all_cards(kb):
    out = []
    for f in sorted(os.listdir(kb.root)):
        if f.endswith(".md") and not f.startswith("_") and f not in ("README.md", "INDEX.md"):
            c = read_card(os.path.join(kb.root, f))
            if c:
                out.append(c)
    return out


def freshness(meta):
    try:
        d = datetime.strptime(meta.get("last_seen", ""), "%Y-%m-%d").date()
    except ValueError:
        return 0.0
    return max(0.0, 1.0 - (date.today() - d).days / 365.0)


def rank(c):
    return STARS.get(c["meta"].get("confidence", "★"), 1) * (0.25 + 0.75 * freshness(c["meta"]))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_match(kb, a):
    """At most --max cards. Never the whole index.

    plan_round already carries four other advisory channels (knowledge/, perf_knowledge, the deep-mode
    blackboards, its own HISTORY). Pasting a 40-line index in as a fifth would dilute the profile
    evidence that makes the workflow work with no KB at all.
    """
    key = make_key(a.operator, a.device, a.regime, a.kernel_name)
    cls, gfx, regime = key.split(" · ")
    scored = []
    for c in all_cards(kb):
        parts = [p.strip() for p in c["meta"].get("key", "").split("·")]
        if len(parts) != 3 or parts[0] != cls:
            continue
        why = ["class matches"]
        score = rank(c)
        if parts[1] == gfx:
            score += 1.0; why.append("same gfx")
        if parts[2] == regime:
            score += 0.5; why.append("same regime")
        scored.append((score, c, why))
    scored.sort(key=lambda t: -t[0])
    out = {"key": key,
           # Named loudly: an unmatched dimension means this lookup could not have matched anything,
           # which reads exactly like "the KB had nothing useful" unless it is reported.
           "unmatched_dims": [d for d, v in (("kernel_class", cls), ("gfx", gfx)) if v == UNMATCHED],
           "cards": []}
    for score, c, why in scored[:a.max]:
        m = c["meta"]
        card = {"path": c["path"], "key": m.get("key"), "confidence": m.get("confidence"),
                "effect": m.get("effect"), "attempts": m.get("attempts"),
                "confirms_cited": m.get("confirms_cited"), "confirms_blind": m.get("confirms_blind"),
                "losses": m.get("losses", 0)}
        if a.explain:
            card["why_matched"] = why
            card["score"] = round(score, 3)
        out["cards"].append(card)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_lint(kb, a):
    if a.cards:
        # Audit a tree that predates these rules. `source` is not required here: the point is to find
        # contract violations in existing cards, not to fail every card for a missing field.
        bad = {}
        for c in all_cards(kb):
            card = dict(c["meta"])
            card["title"] = os.path.basename(c["path"])
            for f in ("lever", "apply", "verify", "caution", "source"):
                m = re.search(rf"^- {f}:\s*(.*?)(?=\n- \w+:|\Z)", c["body"], re.S | re.M)
                if m:
                    card[f] = m.group(1)
            if "attempts" not in c["meta"] or not c["meta"].get("attempts"):
                card["attempts"] = 1  # older schema had no attempts; don't report that 17 times
            e = lint_card(card, strict_source=False)
            if e:
                bad[os.path.basename(c["path"])] = e
        print(json.dumps({"cards_audited": len(all_cards(kb)), "cards_failing": len(bad),
                          "failures": bad}, ensure_ascii=False, indent=2))
        return 0
    prop = json.load(open(a.file))
    errs = {"proposal": lint_proposal(prop)}
    names = prop.get("kernel_names") or ([prop["kernel_name"]] if prop.get("kernel_name") else [])
    for i, card in enumerate(prop.get("cards", [])):
        e = lint_card(card, names)
        if e:
            errs[f"card[{i}] {card.get('title', '?')}"] = e
    errs = {k: v for k, v in errs.items() if v}
    print(json.dumps({"ok": not errs, "rejections": errs}, ensure_ascii=False, indent=2))
    return 1 if errs else 0


def cmd_propose(kb, a):
    prop = json.load(open(a.file))
    if cmd_lint(kb, a):
        print("REJECTED — nothing written to the inbox.", file=sys.stderr)
        return 1
    os.makedirs(kb.inbox, exist_ok=True)
    # Create-once, name carries the run id: two writers can never collide, so no lock is needed even
    # across hosts on NFSv3, where cross-host NLM locking has never been exercised on this mount.
    dest = os.path.join(kb.inbox, f"{slugify(prop['run_id'])}.json")
    with open(dest, "x") as f:
        json.dump(prop, f, ensure_ascii=False, indent=2)
    print(json.dumps({"written": dest, "cards": len(prop.get("cards", [])),
                      "citations": len(prop.get("citations", []))}))
    return 0


def _render_index(cards):
    by_key = {}
    for c in cards:
        by_key.setdefault(c["meta"].get("key", "?"), []).append(c)
    out = []
    for key in sorted(by_key):
        out.append(f"\n## {key}")
        for c in sorted(by_key[key], key=lambda x: -rank(x)):
            m = c["meta"]
            out.append(f"- {m.get('confidence','★')} {os.path.basename(c['path'])[:-3]} — "
                       f"{m.get('effect','')} "
                       f"[cited {m.get('confirms_cited',0)} / blind {m.get('confirms_blind',0)} / "
                       f"lost {m.get('losses',0)} / attempts {m.get('attempts',0)}]")
    return "\n".join(out).strip() or "_(empty — no cards distilled yet)_"


def _splice(path, begin, end, content):
    """Read-modify-write between two markers, in that order.

    Do NOT inline this as `open(p, "w").write(_splice(p, ...))`: Python evaluates `open(p, "w")`
    first, truncating the file before _splice reads it, and the index silently becomes empty.
    """
    raw = open(path).read()
    if begin not in raw or end not in raw:
        raise SystemExit(f"{path}: missing {begin}/{end} markers — refusing to overwrite it")
    new = re.sub(f"{re.escape(begin)}.*?{re.escape(end)}",
                 lambda _: f"{begin}\n{content}\n{end}", raw, flags=re.S)
    with open(path, "w") as f:
        f.write(new)


def cmd_drain(kb, a):
    proposals, skipped = [], []
    for f in sorted(os.listdir(kb.inbox)) if os.path.isdir(kb.inbox) else []:
        if not f.endswith(".json"):
            continue
        p = json.load(open(os.path.join(kb.inbox, f)))
        errs = lint_proposal(p)
        (skipped if errs else proposals).append((f, p, errs))

    cards = {os.path.basename(c["path"]): c for c in all_cards(kb)}
    merged, inserted, rejected, demoted = [], [], [], []

    for fname, prop, _ in proposals:
        key = make_key(prop.get("kernel_class") or prop.get("operator"),
                       prop.get("gfx") or prop.get("device"), prop.get("regime"), "")
        names = prop.get("kernel_names") or ([prop["kernel_name"]] if prop.get("kernel_name") else [])

        # ---- new cards -----------------------------------------------------
        for card in prop.get("cards", []):
            errs = lint_card(card, names)
            if errs:
                rejected.append({"run": prop.get("run_id"), "title": card.get("title"),
                                 "reasons": errs})
                continue
            fn = f"{slugify(card.get('title'))}-{slugify(key)}.md"
            blind = 1 if card.get("blind") else 0
            cited = 0 if card.get("blind") else 1
            if fn in cards:
                m = cards[fn]["meta"]
                m["confirms_cited"] = int(m.get("confirms_cited", 0)) + cited
                m["confirms_blind"] = int(m.get("confirms_blind", 0)) + blind
                m["attempts"] = int(m.get("attempts", 0)) + int(card.get("attempts", 1))
                m["last_seen"] = prop.get("date", str(date.today()))
                want = STARS.get(card.get("confidence", "★"), 1)
                if want == 3 and int(m.get("confirms_blind", 0)) < 1:
                    want = 2      # a merge must not raise a star no single card could claim
                m["confidence"] = STAR_OF[max(want, STARS.get(m.get("confidence", "★"), 1))]
                cards[fn]["body"] += f"\n- source: {card.get('source')}\n"
                merged.append({"card": fn, "run": prop.get("run_id")})
            else:
                if STARS.get(card.get("confidence", "★"), 1) < 2:
                    rejected.append({"run": prop.get("run_id"), "title": card.get("title"),
                                     "reasons": ["INSERT requires >=★★ (merge-only at ★)"]})
                    continue
                meta = {"key": key, "type": card.get("type", "lever"),
                        "confidence": card.get("confidence"), "effect": card.get("effect"),
                        "confirms_cited": cited, "confirms_blind": blind, "losses": 0,
                        "attempts": int(card.get("attempts", 1)),
                        "toolchain": prop.get("toolchain", "unknown"),
                        "last_seen": prop.get("date", str(date.today()))}
                body = f"# {card.get('title')}\n"
                for f_ in ("lever", "apply", "verify", "caution", "source"):
                    if card.get(f_):
                        body += f"- {f_}: {card[f_]}\n"
                cards[fn] = {"path": os.path.join(kb.root, fn), "meta": meta, "body": body}
                inserted.append({"card": fn, "run": prop.get("run_id")})

        # ---- citations: the NEGATIVE half of the loop ----------------------
        # Without this the KB has only an up escalator. The curator reports what worked, so `attempts`
        # would be whatever it remembered; meanwhile every KB-on run already knows exactly which card
        # seeded which direction and what the verifier measured. A card cited ten times that lost nine
        # must not look like one that won.
        for cite in prop.get("citations", []):
            fn = cite.get("card")
            if not fn:
                continue
            fn = fn if fn.endswith(".md") else fn + ".md"
            if fn not in cards:
                continue                      # card was evicted/archived since it was cited
            m = cards[fn]["meta"]
            m["attempts"] = int(m.get("attempts", 0)) + 1
            won = float(cite.get("cited_then_verified") or 0) > 1.0
            if won:
                m["confirms_cited"] = int(m.get("confirms_cited", 0)) + 1
            else:
                m["losses"] = int(m.get("losses", 0)) + 1
            m["last_seen"] = prop.get("date", str(date.today()))
            # A lever that keeps being tried and keeps not paying is a weaker hint than its stars say.
            # Demote rather than delete, and record the condition — never a blocklist.
            if m["losses"] >= 3 and m["losses"] > int(m.get("confirms_cited", 0)):
                old = STARS.get(m.get("confidence", "★"), 1)
                if old > 1:
                    m["confidence"] = STAR_OF[old - 1]
                    demoted.append({"card": fn, "to": m["confidence"],
                                    "losses": m["losses"], "wins": m.get("confirms_cited", 0)})
                note = (f"\n- caution: cited {m['attempts']} time(s) with "
                        f"{m['losses']} non-improving outcome(s) as of "
                        f"{prop.get('date', date.today())} — also verify it engages on your shapes "
                        f"before spending a round on it.\n")
                if "cited " not in cards[fn]["body"]:
                    cards[fn]["body"] += note

    live = sorted(cards.values(), key=lambda c: -rank(c))
    evicted = []
    if len(live) > INDEX_CAP:
        keep, drop = [], []
        for c in live:
            (keep if (len(keep) < INDEX_CAP or c["meta"].get("confidence") == "★★★")
             else drop).append(c)
        evicted = [os.path.basename(c["path"]) for c in drop]
        live = keep

    validated = a.validated_runs if a.validated_runs is not None else len(proposals) + len(skipped)
    report = {
        "kb_dir": kb.root,
        "dry_run": not a.apply,
        # Curators degrade to null silently on API faults. Without this line a KB built from 6 of 20
        # runs is indistinguishable from one built from 20.
        "coverage": f"{len(proposals)}/{validated} validated runs produced a usable proposal",
        "skipped_proposals": [{"file": f, "reasons": e} for f, _, e in skipped],
        "merged": merged, "inserted": inserted, "rejected_cards": rejected,
        "demoted_by_citations": demoted, "evicted": evicted, "index_lines": len(live),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not a.apply:
        return 0

    for c in live:
        write_card(c["path"], c["meta"], c["body"])
    _splice(kb.index, "<!-- CARDS:BEGIN -->", "<!-- CARDS:END -->", _render_index(live))
    if evicted:
        with open(kb.archive, "a") as f:
            f.write(f"\n### Evicted {date.today()} (INDEX at cap {INDEX_CAP})\n")
            for e in evicted:
                f.write(f"- {e}\n")
    for fname, _, _ in proposals:
        os.rename(os.path.join(kb.inbox, fname), os.path.join(kb.inbox, fname + ".drained"))
    return 0


def cmd_doctor(kb, a):
    """Make rot visible. Silent no-match is the most likely way this whole thing dies."""
    cs = all_cards(kb)
    def base(c):
        return os.path.basename(c["path"])
    report = {
        "kb_dir": kb.root,
        "cards": len(cs),
        "index_headroom": INDEX_CAP - len(cs),
        # A card only ever confirmed by runs it steered. Capped at ★★ by the lint, listed here so the
        # cap is visible rather than merely enforced.
        "self_confirmed_only": [base(c) for c in cs if int(c["meta"].get("confirms_blind", 0)) == 0],
        # Tried often, rarely paid. Not wrong — just a weaker hint than its stars suggest.
        "weak_base_rate": [
            {"card": base(c), "cited": c["meta"].get("confirms_cited", 0),
             "attempts": c["meta"].get("attempts", 0)}
            for c in cs if int(c["meta"].get("attempts", 0)) >= 5
            and int(c["meta"].get("confirms_cited", 0)) * 3 < int(c["meta"].get("attempts", 0))],
        "losing": [{"card": base(c), "losses": c["meta"].get("losses", 0)}
                   for c in cs if int(c["meta"].get("losses", 0)) >= 2],
        # A card from an older ROCm/Triton can be flatly false on the current one.
        "stale_toolchain": ([base(c) for c in cs if a.toolchain
                             and c["meta"].get("toolchain", "unknown") not in (a.toolchain, "unknown")]
                            if a.toolchain else "pass --toolchain to check"),
        # If a key has an `unmatched` dimension, no lookup can ever reach it.
        "unreachable_keys": [base(c) for c in cs if UNMATCHED in c["meta"].get("key", "")],
        "inbox_pending": len([f for f in os.listdir(kb.inbox) if f.endswith(".json")])
                         if os.path.isdir(kb.inbox) else 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_stats(kb, a):
    cs = all_cards(kb)
    print(json.dumps({
        "kb_dir": kb.root,
        "cards": len(cs),
        "by_confidence": {s: sum(1 for c in cs if c["meta"].get("confidence") == s) for s in STARS},
        "total_attempts": sum(int(c["meta"].get("attempts", 0)) for c in cs),
        "total_cited_wins": sum(int(c["meta"].get("confirms_cited", 0)) for c in cs),
        "total_losses": sum(int(c["meta"].get("losses", 0)) for c in cs),
        "inbox_pending": len([f for f in os.listdir(kb.inbox) if f.endswith(".json")])
                         if os.path.isdir(kb.inbox) else 0,
    }, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb-dir", required=True,
                    help="the knowledge/learned tree to operate on. REQUIRED, no default: this tool "
                         "serves several trees and a default would silently pick one.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match"); m.set_defaults(fn=cmd_match)
    m.add_argument("--operator", default=""); m.add_argument("--language", default="")
    m.add_argument("--device", default=""); m.add_argument("--regime", default="")
    m.add_argument("--kernel-name", dest="kernel_name", default="")
    m.add_argument("--max", type=int, default=3)
    m.add_argument("--explain", action="store_true", help="say why each card matched")

    l = sub.add_parser("lint"); l.set_defaults(fn=cmd_lint)
    l.add_argument("--file"); l.add_argument("--cards", action="store_true",
                                             help="audit the cards already in --kb-dir")

    p = sub.add_parser("propose"); p.set_defaults(fn=cmd_propose)
    p.add_argument("--file", required=True); p.add_argument("--cards", action="store_false",
                                                            default=False, help=argparse.SUPPRESS)

    d = sub.add_parser("drain"); d.set_defaults(fn=cmd_drain)
    d.add_argument("--apply", action="store_true", help="without this it is a dry run")
    d.add_argument("--validated-runs", type=int, default=None,
                   help="denominator for the coverage line")

    doc = sub.add_parser("doctor"); doc.set_defaults(fn=cmd_doctor)
    doc.add_argument("--toolchain", default="", help="current stack fingerprint to compare against")

    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)

    a = ap.parse_args()
    sys.exit(a.fn(KB(a.kb_dir), a) or 0)


if __name__ == "__main__":
    main()
