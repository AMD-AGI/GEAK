#!/usr/bin/env python3
"""Learned-knowledge store. ONE implementation, serving every `knowledge/learned/` tree in the repo.

    kb.py --kb-dir <tree> match   --operator <op> --device <str> [--regime <r>] [--max 3] [--explain]
                                  # a GREP SHORTCUT, not the read path. The read path is: read the
                                  # generated INDEX.md and judge relevance by meaning (README
                                  # "Discovery"). A matcher returns what a string query asked for;
                                  # a reader returns what is actually relevant, and a split-k card
                                  # is worth opening for a problem nobody would query as split-k.
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
from collections import Counter
import glob as globmod
import json
import os
import re
import sys
from datetime import date, datetime

INDEX_CAP = 40   # retained: the archive header still reports the budget it was written under
CLASS_CAP = int(os.environ.get("KB_CLASS_CAP", "32"))
# Active cards kept per `kernel_class` — the axis the generated index GROUPS by and the one its
# over-cap warning counts. See the eviction block in `drain`.
#
# Was 8, and 8 was measured to be wrong. On a 121-card corpus it archived 60 cards — half the KB —
# and it took them from exactly the classes that had the most to say: 24 of 32 from
# `moe_grouped_gemm`, 16 of 24 from `attention_decode`. The comparison arm read 78 cards with none
# trimmed; this one read 61, and lost worst (-32%) on `fused_moe_kernel`, a `moe_grouped_gemm` kernel
# whose class had been cut by 75%.
#
# The reasoning behind 8 was that a prolific class would otherwise crowd out a sparse one. That
# confused fairness with usefulness: a class has more cards mostly because more was learned about it,
# and `match`/the reader select BY CLASS anyway, so a big `moe_grouped_gemm` shelf never competes for
# attention with `quantize_cast`. Trimming it only destroys priors.
#
# 32 is the largest class in the current corpus, i.e. nothing is trimmed today. It stays a cap rather
# than becoming unbounded because the index is read whole by a planner and an index that grows without
# limit eventually costs recall — but the next value should be chosen from a measurement, not from a
# tidiness instinct, and it should be raised before it is allowed to bind. Overridable via
# KB_CLASS_CAP so an operator can test a different budget without editing code.
def class_of(card):
    """The eviction/index bucket for a card. ONE definition, because three call sites had three.

    `drain`'s eviction, the index generator and `doctor` each derived this inline and they had
    already drifted: two mapped a missing OR empty kernel_class to "other", the third mapped empty
    to a bucket named "" that only ever held the malformed cards, and `doctor` grouped by `key`
    instead — a sentence unique per card, so its buckets held one card each and its cap warning
    could never fire. Accepts either a raw meta mapping or a card dict wrapping one.
    """
    meta = card.get("meta", card) if hasattr(card, "get") else card
    return str(meta.get("kernel_class") or "other").strip() or "other"


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
    """Longest matching needle wins — NEVER first-match-in-dict-order.

    The classes overlap by construction: "linear attention" contains "attention", "moe grouped gemm"
    contains "gemm". With first-match, whichever class happened to be declared first in the dict won,
    so a curator that correctly said "linear attention" had its card filed under "attention" — and a
    later attention kernel would then be handed linear-attention cards. That mis-filing is silent:
    the card exists, the lookup succeeds, and only the content is wrong. Observed on the first real
    batch. Specificity, not declaration order, decides.
    """
    hay = f"{operator or ''} {kernel_name or ''}".lower()
    best, best_len = UNMATCHED, 0
    for cls, needles in CLASS_VOCAB.items():
        for n in needles:
            if n in hay and len(n) > best_len:
                best, best_len = cls, len(n)
    return best


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
# Absolute machine numbers. README "Content rules" #1 already forbids these in prose; this is the
# same rule with teeth. Wall-clock and absolute throughput are properties of the box that produced
# them — clock state, driver, neighbour load — so a figure copied into a card is stale on arrival and
# reads to the next run as a target it should hit. Ratios, percent deltas and fractions of achievable
# peak survive the move to another box; `1451 TFLOP/s` does not. Measured against the cards this repo
# already had: 47 of 78 carried at least one, so the prose rule was not holding on its own.
# `0.5x`, `+18%`, `62% of achievable HBM BW`, `M=4096`, `num_warps=8`, a date, a gfx id all pass.
ABSOLUTE_UNIT_PATTERNS = [
    (r"\b\d[\d.,]*\s*(?:ms|µs|us|ns)\b", "wall-clock (ms/µs/ns)"),
    (r"\b\d[\d.,]*\s*(?:T|G|M)?FLOP(?:/s|s\b)", "absolute throughput (FLOP/s)"),
    (r"\b\d[\d.,]*\s*(?:T|G|M|K)?B/s\b", "absolute bandwidth (B/s)"),
    (r"\b\d[\d.,]*\s*(?:GHz|MHz)\b", "an absolute clock"),
    # `W` must be a UNIT, not the start of another token. `\d[\d.,]*\s*W\b` matched "BLOCK=2, W..."
    # — the comma was read as a thousands separator and `W` as watts — and refused a legitimate card.
    # A gate that fires on correct input costs more than the one bad card it was meant to catch.
    (r"\b\d+(?:\.\d+)?\s?(?:W|watts?)(?![A-Za-z0-9_-])", "absolute power"),
]
# Their discovery header is what the generated INDEX.md is built from: a card missing one of these is
# either invisible (no description => "(no description)") or unfindable (no keywords/kernels), which
# looks identical to "the KB had nothing for this run".
REQUIRED_HEADER_FIELDS = ("name", "description", "keywords", "platforms",
                          "kernel_class", "regime", "lifecycle")
# `kernels` is DELIBERATELY not in that list, and the reason generalises: a required field whose
# value the writer has no way to know does not get left blank, it gets filled with something
# plausible. Measured here — requiring it while migrating 78 class-level cards (whose bodies are
# forbidden from naming a kernel, so the symbol genuinely was not recoverable) produced 11 invented
# symbols across the corpus, one of them, `fused_moe_grouped_gemm`, in 17 cards. A grep aid that
# sends the reader to the wrong card is worse than an absent one. Optional, and checked when given.
MAX_DESCRIPTION_CHARS = 160          # README: the description IS the index line
CARD_BODY_FIELDS = ("title", "lever", "apply", "stack", "verify", "pitfall", "caution", "effect",
                    "source")
MAX_CARD_LINES = 20                  # README: ">20 lines means you're storing narrative"


def lint_card(card, kernel_names=(), strict_source=True):
    """Return a list of rejection reasons; empty == accepted."""
    errs = []
    text = "\n".join(str(card.get(f, "")) for f in CARD_BODY_FIELDS)
    # `source` is PROVENANCE, and provenance is exempt from the class-level rule below. Those two
    # requirements contradict each other otherwise, and it is not hypothetical: a campaign whose run
    # ids are named after their kernel ("_fwd_grouped_kernel_stage1-chuschen16h") cannot satisfy both
    # "every claim needs a run id" and "never name a kernel". Hit on the first real bulk import; the
    # curator complied by de-identifying `source`, which destroyed exactly the run id the other rule
    # demands. A kernel symbol in the LEVER is memorising a run; a kernel symbol in the CITATION is
    # what makes the claim checkable.
    principle_text = "\n".join(str(card.get(f, ""))
                               for f in CARD_BODY_FIELDS if f != "source")

    for pat, what in LEAK_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            errs.append(f"leaks an instance identifier ({what}): {m.group(0)!r}")
    # The most direct leak, and the one a well-meaning curator writes without noticing.
    for kn in kernel_names:
        if kn and len(kn) > 4 and re.search(re.escape(kn), principle_text, re.I):
            errs.append(f"names a specific kernel ({kn!r}); keys and bodies must be class level")
    for pat, what in MANDATE_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            errs.append(f"contains {what}: {m.group(0)!r} — a caution must read 'also verify X'")
    for pat, what in ABSOLUTE_UNIT_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            errs.append(f"cites {what}: {m.group(0)!r} — record a ratio, a percent delta, or a "
                        f"fraction of achievable peak; the raw timing belongs in EVAL_DIR")

    # Discovery header. Checked here rather than left to the generator: the generator DEFAULTS a
    # missing description to "(no description)" and skips a card with no frontmatter entirely, so an
    # incomplete header degrades to an unfindable card instead of an error anyone sees.
    for f in REQUIRED_HEADER_FIELDS:
        v = card.get(f)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            errs.append(f"discovery header missing {f!r}: the index line is built from it")
    desc = str(card.get("description", ""))
    if len(desc) > MAX_DESCRIPTION_CHARS:
        errs.append(f"description is {len(desc)} chars (>{MAX_DESCRIPTION_CHARS}): it IS the index "
                    f"line, so it has to read as one")
    if str(card.get("lifecycle", "active")) not in ("active", "archived"):
        errs.append(f"lifecycle must be active|archived, got {card.get('lifecycle')!r}")

    # OPTIONAL fields, validated only when present. README used to document `layer`, `levers`,
    # `cost`, `verified_on` and `roofline` as part of the schema while nothing wrote them and nothing
    # checked them — 0 of 78 real cards carried any. A documented field that no card has and no gate
    # wants is not a schema, it is a wish, and it teaches a curator that the schema is approximate.
    # So: the README now marks them optional, and anything a card DOES carry has to be well-formed.
    cost = str(card.get("cost", "")).strip()
    if cost and not re.fullmatch(r"L[0-3]", cost):
        errs.append(f"cost must be L0|L1|L2|L3 (env/flag · config · host rewrite · new kernel), "
                    f"got {cost!r}")
    von = str(card.get("verified_on", "")).strip()
    if von and von != "null" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", von):
        errs.append(f"verified_on must be YYYY-MM-DD or null, got {von!r} — it means the date an "
                    f"on-box A/B confirmed this, so an unparseable one is worse than none")
    # `key` is their plain-English identity and the merge target. A rigid triple defeats its purpose:
    # it collapses a vLLM MXFP8 card and an sglang bf16 card of the same class onto one key and
    # invites a wrong merge. The machine-readable slots live in the header above.
    key = str(card.get("key", "")).strip()
    if not key:
        errs.append("no key: one line of plain English saying what this card is about")
    elif re.fullmatch(r"[\w /-]+·[\w /-]+·[\w /-]+", key):
        errs.append(f"key {key!r} is a bare class·gfx·regime triple — the header already carries "
                    f"those; write what a person would say (op, arch, framework, dtype, regime)")

    # "None" is not a title, it is `str(None)`. Checking only for emptiness misses the way this
    # actually fails: an absent key reaches slugify() and the card lands on disk as `none-<scope>.md`
    # with an H1 reading "# None" — non-empty, so a truthiness check waves it through. Two cards did
    # exactly that on the first real bulk import.
    if str(card.get("title", "")).strip().lower() in ("", "none", "null", "n/a"):
        errs.append(f"no usable title ({card.get('title')!r}): it becomes the card's H1 and its "
                    f"filename slug, and an absent one is written out as 'none-<scope>.md'")
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
        if line.strip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        # Parse `[a, b]` into a list, exactly as the JS parseFrontmatter() this replaced did.
        # Without this the two readers disagree about the same file: the generator saw an empty
        # `keywords: []` and dropped the card's tags, while this side saw the STRING "[]" — truthy,
        # non-empty — and reported the header complete. A lint that cannot see an empty list cannot
        # enforce a header, and the disagreement is silent in both directions.
        if v.startswith("[") and v.endswith("]"):
            meta[k] = [x.strip().strip('"\'') for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k] = v.strip('"\'')
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


def all_cards(kb, include_archived=False):
    """Active cards only, by default — the SAME predicate `build_index()` uses.

    If these two disagree about what is live, the index and every check in this file are describing
    different knowledge bases, and the disagreement is invisible: `doctor` would report a healthy
    tree while the reader sees a different one. One rule, two implementations, is how that happens.
    """
    out = []
    for f in sorted(os.listdir(kb.root)):
        if f.endswith(".md") and not f.startswith("_") and f not in ("README.md", "INDEX.md"):
            c = read_card(os.path.join(kb.root, f))
            if c and (include_archived or str(c["meta"].get("lifecycle", "active")) == "active"):
                out.append(c)
    return out


def freshness(meta):
    try:
        d = datetime.strptime(meta.get("last_seen", ""), "%Y-%m-%d").date()
    except ValueError:
        return 0.0
    return max(0.0, 1.0 - (date.today() - d).days / 365.0)


def rank(c):
    """Eviction order. Stars and freshness, weighted by what the card has actually EARNED.

    The citation machinery exists to let a card prove itself: `confirms_cited` counts the times a
    cited card's direction went on to win its round, `losses` the times it did not even beat the
    baseline. Ranking on confidence and freshness alone threw all of that away at exactly the moment
    it mattered — a card cited and confirmed three times lost its index slot to one written this
    morning that nothing had ever tested, purely because the newcomer was fresher. That inverts the
    point of keeping score. Observed when a 20-kernel ingest pushed the index from 28 to 100 cards
    against a cap of 40: without this term the survivors were chosen by date.

    Bounded both ways on purpose. The floor keeps a card with a bad run from vanishing before its
    demotion to ★ has been earned through the confidence ladder; the ceiling stops a well-cited card
    from becoming unevictable and freezing the index against better new work.
    """
    m = c["meta"]
    standing = (1.0 + int(m.get("confirms_cited", 0) or 0)) / (1.0 + int(m.get("losses", 0) or 0))
    standing = max(0.5, min(2.0, standing))
    return STARS.get(m.get("confidence", "★"), 1) * (0.25 + 0.75 * freshness(m)) * standing


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
        # Audit EVERY card, archived included. Auditing only the active ones excludes exactly the
        # cards most likely to be malformed — a card whose `lifecycle` is a typo is not active, so
        # the reader that filters on active never sees it and reports a clean tree.
        for c in all_cards(kb, include_archived=True):
            card = dict(c["meta"])
            # Take the title from the body's H1, which is where it actually lives — NOT from the
            # filename. Substituting the filename here made "this card has no title" unobservable to
            # the only check that would report it, and two cards reached disk named `none-*.md`.
            h1 = re.search(r"^#\s+(.+)$", c["body"], re.M)
            card["title"] = h1.group(1).strip() if h1 else ""
            for f in ("lever", "apply", "verify", "caution", "source"):
                m = re.search(rf"^- {f}:\s*(.*?)(?=\n- \w+:|\Z)", c["body"], re.S | re.M)
                if m:
                    card[f] = m.group(1)
            if "attempts" not in c["meta"] or not c["meta"].get("attempts"):
                card["attempts"] = 1  # older schema had no attempts; don't report that 17 times
            e = lint_card(card, strict_source=False)
            if e:
                bad[os.path.basename(c["path"])] = e
        # Count what was actually audited. Reporting len(all_cards(kb)) here counted ACTIVE cards
        # while the loop above walks archived ones too, so an audit of 32 cards announced 23 — and
        # the number a reader checks against "did it look at everything?" was the wrong one.
        print(json.dumps({"cards_audited": len(all_cards(kb, include_archived=True)),
                          "cards_failing": len(bad),
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


# INDEX.md has exactly one writer in this repo: `build_index()` below. The splice-based renderer
# that used to live here was the second one, and a second writer of a generated file is how the index
# and the cards drift apart. `drain` now shells out to the generator instead (see the end of it).



# ---------------------------------------------------------------------------
# INDEX.md generation. A PORT of build_learned_index.js, which this replaces.
#
# The JS original was the right design — the index is DERIVED from the cards, so parallel lanes can
# never drop each other's append and the listing can never drift from the files. What it could not do
# is run: the containers these agents execute in have no JS runtime at all (node/nodejs/bun/npx/deno
# all absent), so `update_experience`'s instruction to regenerate after writing a card failed at the
# shell, and a card that never reaches INDEX.md is a card the read path cannot see. The design was
# sound and the runtime assumption was not.
#
# Ported rather than duplicated: the JS generator and its test are deleted in the same change. Two
# generators for one generated file drift, and a drifted index is exactly the failure the generated
# index exists to prevent.
#
# Output is byte-compatible with the JS version: same header, same grouping, same ordering, same
# vocabulary appendix, same ⚠ blocks.
# ---------------------------------------------------------------------------
GEN_CAP = 40
GEN_SKIP = {"INDEX.md", "README.md", "_archive.md"}
LAST_GROUPS = ["method", "other"]


def _owner(d):
    parts = os.path.abspath(d).split(os.sep)
    return parts[parts.index("knowledge") - 1] if "knowledge" in parts[1:] else parts[-1]


def _regen_cmd(d):
    o = _owner(d)
    return ("python3 kernel_workflow/scripts/kb.py --kb-dir " +
            ("kernel_workflow/knowledge/learned" if o == "kernel_workflow"
             else f"{o}/knowledge/learned") + " index")


def _kw_normalize(x):
    return re.sub(r"-{2,}", "-", re.sub(r"[\s_]+", "-", str(x).lower().strip())).strip("-")


def _kw_fold(x):
    return re.sub(r"s$", "", _kw_normalize(x).replace("-", ""))


def _index_header(d):
    o = _owner(d)
    return (f"""# Learned — index of distilled {o} experience cards

<!-- GENERATED FILE — do not hand-edit. Regenerate with:
       {_regen_cmd(d)}
     Every line below is derived from one card's discovery frontmatter. To change a line, edit the
     card's `description`/`keywords`/`confidence` and regenerate. -->

Open the cards matching your run as **additional, advisory priors** — they only ADD candidate levers to
try, never remove any and never replace measurement. The frozen-baseline isolated A/B + oracle parity is
always the judge (see `README.md`). **Cap: <={GEN_CAP} card lines.** Confidence (a hint strength, not
authority): ★ noise/unverified · ★★ single non-overlap or >=2 consistent · ★★★ >=2 non-overlap.

Effects are **ratios or percent deltas only, never wall-clock or absolute throughput** — those vary box
to box and stay in the run's `EVAL_DIR` (see `README.md` -> "Content rules").

**How to use this file: READ it, then open the 0–3 cards that look relevant.** Each line carries the
card's own description, the kernel symbols it was measured on, and its keywords — enough to judge
relevance without opening anything. Match on *meaning*, not on an exact string: a card written for
`split-k on skinny-M GEMM` is worth opening for a tall-K GEMM too. If nothing matches, that is a real
answer — plan cold, exactly as this workflow does without any KB.
""")


def build_index(kb_dir):
    """Render INDEX.md text from the active cards on disk. Pure function of the directory."""
    cards = []
    for f in sorted(os.listdir(kb_dir)):
        if not f.endswith(".md") or f in GEN_SKIP or f.startswith("_"):
            continue
        c = read_card(os.path.join(kb_dir, f))
        if not c:
            continue                      # no frontmatter = not indexable; skipped, not guessed at
        m = c["meta"]
        if str(m.get("lifecycle", "active")) != "active":
            continue                      # archived keeps its file (and its refuting source), not its line
        arr = lambda v: v if isinstance(v, list) else ([str(v)] if v else [])
        st = re.search(r"★+", str(m.get("confidence", "")))
        cards.append({
            "file": f, "name": m.get("name") or f[:-3],
            "description": m.get("description") or m.get("effect") or "(no description)",
            "keywords": list(dict.fromkeys(filter(None, (_kw_normalize(k) for k in arr(m.get("keywords")))))),
            "kernels": arr(m.get("kernels")), "platforms": arr(m.get("platforms")),
            "kernel_class": class_of(m), "regime": m.get("regime") or "",
            "confidence": st.group(0) if st else "★",
        })

    groups = {}
    for c in cards:
        groups.setdefault(c["kernel_class"], []).append(c)
    def gkey(g):
        r = LAST_GROUPS.index(g) if g in LAST_GROUPS else -1
        return (0 if r == -1 else 1, r, g)
    body = ""
    for g in sorted(groups, key=gkey):
        rows = sorted(groups[g], key=lambda c: (-len(c["confidence"]), c["name"]))
        body += f"\n## {g}\n"
        for c in rows:
            scope = " · ".join([x for x in ["/".join(c["platforms"]), c["regime"]] if x])
            body += f"- {'[' + scope + '] ' if scope else ''}{c['description']} {c['confidence']} — ({c['file']})\n"
            tags = " · ".join([x for x in [
                f"kernels: {', '.join(c['kernels'])}" if c["kernels"] else "",
                f"kw: {', '.join(c['keywords'])}" if c["keywords"] else ""] if x])
            if tags:
                body += f"  - {tags}\n"
    if not cards:
        body = "\n## (no cards yet)\n"

    counts = Counter(k for c in cards for k in c["keywords"])
    vocab = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if vocab:
        body += ("\n## keyword vocabulary (generated — REUSE these before coining a new term)\n" +
                 " · ".join(f"{k}({n})" if n > 1 else k for k, n in vocab) + "\n")
        byfold = {}
        for k, _ in vocab:
            byfold.setdefault(_kw_fold(k), []).append(k)
        dupes = [v for v in byfold.values() if len(v) > 1]
        if dupes:
            # Flagged, never auto-merged: collapsing `mfma`/`mfmas` behind the curator's back would
            # be a worse failure than a visible warning.
            body += ("\n> ⚠ **Near-duplicate keywords** — same concept, different spelling. Pick one, edit the\n"
                     "> cards, regenerate:\n" + "".join(f"> - {' / '.join(v)}\n" for v in dupes))

    # Per-class, matching what `drain` actually enforces; a global total would warn on a healthy tree
    # the moment the KB grew past one campaign's worth of cards.
    over = {g: len(v) for g, v in groups.items() if len(v) > CLASS_CAP}
    footer = ""
    if over:
        footer = (f"\n> ⚠ **Over the per-class cap of {CLASS_CAP}:** " +
                  ", ".join(f"{g} ({n})" for g, n in sorted(over.items())) +
                  ". Archive the lowest `confidence × freshness × standing` card in that class\n"
                  "> (★★★ is never auto-evicted), then regenerate.\n")
    return _index_header(kb_dir) + body + footer


def cmd_index(kb, a):
    out = os.path.join(kb.root, "INDEX.md")
    nxt = build_index(kb.root)
    prev = open(out).read() if os.path.exists(out) else ""
    n = nxt.count("\n— (") + sum(1 for l in nxt.splitlines() if l.startswith("- ") and l.endswith(")"))
    if a.check:
        if prev == nxt:
            print(f"INDEX.md up to date."); return 0
        print("INDEX.md is stale — run: " + _regen_cmd(kb.root), file=sys.stderr); return 1
    if prev != nxt:
        open(out, "w").write(nxt)
    print(f"INDEX.md {'unchanged' if prev == nxt else 'written'}.")
    return 0


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
        # FALLBACK only. `key` belongs to the card and is plain English by design; deriving a
        # class·gfx·regime triple here and writing it over the curator's line destroyed exactly what
        # the schema asks for — and the lint, which rejects bare triples, then failed all 58 cards
        # `drain` had just written. Seen on the first real bulk import through this path. The triple
        # still earns its keep as the eviction bucket (from the header fields) and as a filename
        # slug when a card supplies no key of its own.
        fallback_key = make_key(prop.get("kernel_class") or prop.get("operator"),
                       prop.get("gfx") or prop.get("device"), prop.get("regime"), "")
        names = prop.get("kernel_names") or ([prop["kernel_name"]] if prop.get("kernel_name") else [])

        # ---- new cards -----------------------------------------------------
        for card in prop.get("cards", []):
            errs = lint_card(card, names)
            if errs:
                rejected.append({"run": prop.get("run_id"), "title": card.get("title"),
                                 "reasons": errs})
                continue
            key = str(card.get("key", "")).strip() or fallback_key
            fn_scope = "-".join(x for x in [card.get("kernel_class", ""),
                                            "-".join(card.get("platforms") or []),
                                            card.get("regime", "")] if x)
            fn = f"{slugify(card.get('title'))}-{slugify(fn_scope or key)}.md"
            blind = 1 if card.get("blind") else 0
            cited = 0 if card.get("blind") else 1
            if fn in cards:
                m = cards[fn]["meta"]
                # A merge refreshes the header too: a card whose description still describes the
                # first run it came from is a stale index line, and the index line is the read path.
                for hf in ("description", "kernel_class", "regime"):
                    if card.get(hf):
                        m[hf] = card[hf]
                for lf in ("keywords", "kernels", "platforms"):
                    if card.get(lf):
                        m[lf] = sorted(set(list(m.get(lf) or []) + list(card[lf])))
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
                # Carry the DISCOVERY HEADER through. This list used to stop at `key` and the
                # counters, which predates the header — so `drain` wrote cards with no `description`,
                # no `keywords`, no `kernel_class`, and three things broke at once, none loudly:
                # the generated index rendered them as "(no description)" under a group called
                # `other`; the per-class budget saw every card in ONE bucket and evicted 50 of 60 to
                # a cap meant for one class; and `lint --cards` then failed the very cards `drain`
                # had just written. The bulk-import path was producing cards that violate the schema
                # it enforces, and it went unnoticed because the first 87 cards were placed by hand.
                # If a field is part of the header, it belongs here — not in a list to keep in sync.
                meta = {
                    "name": fn[:-3],
                    "description": card.get("description", ""),
                    "keywords": card.get("keywords", []),
                    "kernels": card.get("kernels", []),
                    "platforms": card.get("platforms", []),
                    "kernel_class": card.get("kernel_class", "other"),
                    "regime": card.get("regime", ""),
                    "key": key, "layer": "learned",
                    "lifecycle": card.get("lifecycle", "active"),
                    "type": card.get("type", "lever"),
                    "confidence": card.get("confidence"), "effect": card.get("effect"),
                    "confirms_cited": cited, "confirms_blind": blind, "losses": 0,
                    "attempts": int(card.get("attempts", 1)),
                    "toolchain": prop.get("toolchain", "unknown"),
                    "last_seen": prop.get("date", str(date.today())),
                }
                for opt in ("cost", "verified_on", "roofline", "levers"):
                    if card.get(opt):
                        meta[opt] = card[opt]
                body = f"# {card.get('title')}\n"
                # `stack` and `pitfall` are part of the card body in this schema — a stacked win's
                # per-direction attribution and the traps actually hit are the two things the next
                # run needs most, and they were being dropped on the floor here.
                for f_ in ("lever", "apply", "stack", "verify", "pitfall", "caution", "source"):
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
            # Three states, not two. `cited_then_verified` is measured against the ORIGINAL frozen
            # baseline, so once a kernel sits at 2.5x cumulative EVERY non-regressing direction
            # measures >1.0 — scoring that as a win would let a card accumulate credit for advancing
            # nothing. Observed on the first KB-on run: a card cited twice, verified 2.548x and
            # 2.555x, `became_winner` false both times, i.e. it never moved the cumulative best.
            #   advanced  -> the direction became the round winner. The only real confirm.
            #   failed    -> the direction did not even beat the original baseline. A real loss.
            #   neutral   -> everything between: counted as an attempt, credited to neither side.
            # Deliberately conservative in the direction that penalises the KB: a card must move the
            # number to gain standing, and the ambiguous middle never inflates it.
            v = float(cite.get("cited_then_verified") or 0)
            if cite.get("became_winner"):
                m["confirms_cited"] = int(m.get("confirms_cited", 0)) + 1
            elif v <= 1.0:
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

    # Budget per class, not globally. A global cap is decided by whichever class happened to be
    # optimized most: ingesting one 20-kernel campaign put 15 cards in `moe grouped gemm ·
    # compute-bound` and 2 in `quantize / cast · mixed`, and under a flat cap the prolific class
    # evicts the sparse ones wholesale — the KB narrows toward whatever was worked on last, which is
    # the opposite of what it is for. `match` retrieves BY CLASS and only ever injects a few cards, so
    # breadth across classes is worth more than depth inside one.
    # Group by the MACHINE slots in the discovery header, not by `key`. `key` is deliberately plain
    # English here ("MXFP8 E8M0 dense linear, decode-bound · gfx950"), so two cards about the same
    # class would land in different buckets on a wording difference and neither would ever hit its
    # cap. The header fields are the ones the generator groups the index by, so the budget is
    # enforced on the same axis the reader browses.
    # Bucket by `kernel_class` — the SAME axis the generated index groups by, and the same one its
    # over-cap warning counts. Bucketing finer (class · platforms · regime) looked more careful and
    # was wrong: `moe_grouped_gemm · both`, `· mixed` and `· large-batch` are three buckets of 8, so a
    # class the reader sees as one heading held 11 cards under a cap of 8, and the generator warned
    # about a limit the evictor was not enforcing. The budget exists to bound what a reader scans, so
    # it has to be computed on what a reader scans.
    live, evicted = [], []
    by_class = {}
    for c in cards.values():
        by_class.setdefault(class_of(c), []).append(c)
    for _key, group in sorted(by_class.items()):
        group.sort(key=lambda c: -rank(c))
        keep, drop = [], []
        for c in group:
            # ★★★ is never evicted: it marks a principle that has been confirmed across runs, and a
            # busy class should not be able to age one out.
            (keep if (len(keep) < CLASS_CAP or c["meta"].get("confidence") == "★★★")
             else drop).append(c)
        live.extend(keep)
        evicted.extend(os.path.basename(c["path"]) for c in drop)
    live.sort(key=lambda c: -rank(c))

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

    # Eviction = `lifecycle: archived`, and the card file STAYS. In this sink the index is generated
    # and `collect()` keeps only `lifecycle: active`, so flipping the field is what actually removes a
    # card from the read path — the reader reads INDEX.md. That is worth stating because the obvious
    # implementation (drop the index line, leave the file) does NOT work in a sink where retrieval
    # globs the directory: in the tree this code came from, all 9 evicted cards of one drain were
    # still being returned by the matcher afterwards, because the cap had been enforced on the
    # listing and on nothing else. Here the projection IS the read path, so the field is the fix.
    # The file is kept on purpose: an archived card holds the evidence that retired it.
    if evicted:
        with open(kb.archive, "a") as f:
            f.write(f"\n### Archived {date.today()} (per-class cap {CLASS_CAP})\n")
            for c in cards.values():
                if os.path.basename(c["path"]) in set(evicted):
                    c["meta"]["lifecycle"] = "archived"
                    write_card(c["path"], c["meta"], c["body"])
            for e in evicted:
                f.write(f"- {e}\n")

    for fname, _, _ in proposals:
        os.rename(os.path.join(kb.inbox, fname), os.path.join(kb.inbox, fname + ".drained"))

    # INDEX.md is THEIR generated artifact and this script is not a second writer of it. Regenerating
    # here rather than leaving it to the operator is the difference between "the index can never
    # drift" and "the index can never drift as long as everyone remembers"; a stale index after a
    # bulk import is indistinguishable from a KB that learned nothing.
    open(os.path.join(kb.root, "INDEX.md"), "w").write(build_index(kb.root))
    return 0


def cmd_doctor(kb, a):
    """Make rot visible. Silent no-match is the most likely way this whole thing dies."""
    cs = all_cards(kb)
    def base(c):
        return os.path.basename(c["path"])
    report = {
        "kb_dir": kb.root,
        "cards": len(cs),
        # Headroom is per class, because eviction is. A single total went negative the moment the
        # index outgrew the old flat cap and told the operator the KB was 38 cards over budget when
        # in fact only one class was full — a derived number reported against a rule that no longer
        # decides anything is worse than no number.
        # Group by kernel_class, the axis `drain` actually evicts on. This read `key` — a
        # plain-English sentence unique to each card — so every bucket held one card, headroom was
        # CLASS_CAP-1 for everything and `classes_at_cap` was structurally empty. The tree reached
        # attention_decode 32/32 while doctor reported 31 free, i.e. the monitor for the one failure
        # that has already cost this project a regression (a cap of 8 archiving half the KB, and the
        # worst-hit class was the one carrying the biggest win) could never fire.
        "class_headroom": {k: CLASS_CAP - n for k, n in sorted(
            Counter(class_of(c) for c in cs).items())},
        "classes_at_cap": [k for k, n in Counter(
            class_of(c) for c in cs).items() if n >= CLASS_CAP],
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

    ix = sub.add_parser("index", help="regenerate INDEX.md from the cards — the ONLY writer of it")
    ix.set_defaults(fn=cmd_index)
    ix.add_argument("--check", action="store_true",
                    help="exit 1 if INDEX.md is stale and write nothing (for CI)")

    doc = sub.add_parser("doctor"); doc.set_defaults(fn=cmd_doctor)
    doc.add_argument("--toolchain", default="", help="current stack fingerprint to compare against")

    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)

    a = ap.parse_args()
    sys.exit(a.fn(KB(a.kb_dir), a) or 0)


if __name__ == "__main__":
    main()
