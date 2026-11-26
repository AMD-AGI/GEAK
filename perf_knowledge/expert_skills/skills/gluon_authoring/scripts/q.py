#!/usr/bin/env python3
"""q.py - query the Gluon/AMD fact cards (facts.jsonl next to this skill's root).

  q.py "<words or a pasted error>" [--arch gfx950] [--dtype fp8] [--kind error] [-n 6] [--full]
  q.py --id <card-id> [<card-id> ...]     print whole cards (incl. "method": how it was verified)
  q.py --toc [--kind tool]                 groups -> ids (with --kind: that kind only)
  q.py --kind tool                         one line per tool card (id: purpose); --full for the cards

Ranking: a pasted error/message that contains a card's verbatim key ranks first; then query words
matched as whole words, weighted by rarity (IDF), against keys (x3), the id (x2) and claim/signal
(x1), plus a bonus when a key or the card id appears whole in the query. An arch named in the
query (gfx950, MI300X, ...) favours cards scoped to that arch. Cards with evidence "unverified"
rank after verified hits and are labelled UNVERIFIED. The top 3 hits print in full (claim, check,
see); further hits print as id lines (`--id` to expand). Exit 0 with hits, 1 with none, 2 on bad
usage. ISA opcode / SKU questions: the footer names amd_isa.py / extract_sku.py. stdlib only.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE = os.environ.get("GLUON_FACTS") or os.path.join(ROOT, "facts.jsonl")
# Where amd_isa.py / extract_sku.py live: next to this script (standalone) or the shared kernel_tools
# directory of the GEAK pack (perf_knowledge/expert_skills/skills/<skill>/scripts -> kernel_workflow/...).
TOOL_DIRS = [HERE, os.path.normpath(os.path.join(ROOT, "..", "..", "..", "..", "kernel_workflow",
                                                 "scripts", "kernel_tools"))]
STOP = set("a an the of to in on for and or is are be with by at as it its this that from not no "
           "what why how does do did when which i my me get got error use using can there vs "
           "into out has have than then".split())
ARCH_WORDS = {"gfx950": "gfx950", "mi350x": "gfx950", "mi355x": "gfx950", "cdna4": "gfx950",
              "gfx942": "gfx942", "mi300x": "gfx942", "mi325x": "gfx942", "mi308x": "gfx942", "cdna3": "gfx942"}
WORD = re.compile(r"[a-z0-9_.$/+-]*[a-z0-9_]")
FULL_SHOWN = 3


def load(path):
    cards = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    cards.append(json.loads(line))
                except ValueError as exc:
                    sys.exit(f"q.py: {path}:{n}: bad JSON ({exc})")
    return cards


def words(text):
    """whole words plus their '.', '_', '-', '/' parts ('gl.amd.slice' -> also 'gl', 'amd', 'slice')."""
    out = set()
    for w in WORD.findall(text.lower()):
        w = w.strip(".-/")
        if not w:
            continue
        out.add(w)
        for p in re.split(r"[._/+-]+", w):
            if p:
                out.add(p)
    return out


def qtokens(query):
    toks = []
    for w in WORD.findall(query.lower()):
        w = w.strip(".-/")
        if w and w not in STOP and len(w) > 1 and w not in toks:
            toks.append(w)
    return toks


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip(" '\"`")


def tool_path(name):
    for d in TOOL_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.relpath(p) if not os.path.relpath(p).startswith("../../..") else p
    return name


def in_scope(card, key, want):
    if not want:
        return True
    vals = card.get("scope", {}).get(key) or []
    vals = vals if isinstance(vals, list) else [vals]
    if not vals:
        return True
    want = want.lower()
    return any(v in ("*", "any") or want in str(v).lower() or str(v).lower() in want for v in vals)


class Index:
    def __init__(self, cards):
        self.docs = []
        df = {}
        for c in cards:
            kw = words(" ".join(c.get("keys", [])))
            iw = words(c["id"].split(".", 1)[1]) | {c["id"].split(".")[0]}
            bw = words(c.get("claim", "") + " " + c.get("signal", ""))
            self.docs.append((c, kw, iw, bw))
            for w in kw | iw | bw:
                df[w] = df.get(w, 0) + 1
        n = len(cards)
        self.idf = {w: math.log((n + 1) / (d + 0.5)) for w, d in df.items()}

    def score(self, i, query, toks, arch):
        c, kw, iw, bw = self.docs[i]
        q = norm(query)
        s = 0.0
        for k in c.get("keys", []):
            kn = norm(k)
            if len(kn) >= 12 and (kn in q or (len(q) >= 12 and q in kn)):
                s += 100 + len(kn) / 4        # verbatim error / message string
            elif len(kn) >= 3 and kn == q:
                s += 20                         # the query is exactly a key
            elif len(kn) >= 4 and re.search(r"(?<![\w.])" + re.escape(kn) + r"(?![\w])", q):
                s += 2 + sum(self.idf.get(w, 0) for w in words(kn)) / 2   # key phrase whole in query
        if c["id"].lower() in q:
            s += 50
        matched = 0
        for t in toks:
            parts = {t} | set(p for p in re.split(r"[._/+-]+", t) if p and p not in STOP)
            best = 0.0
            for p in parts:
                w = self.idf.get(p, 0) * (1.0 if p == t else 0.6)
                best = max(best, (3 * w if p in kw else 0) + (2 * w if p in iw else 0) + (w if p in bw else 0))
            if best:
                matched += 1
            s += best
        if s and arch:
            archs = [a.lower() for a in (c.get("scope", {}).get("arch") or [])]
            if archs and arch in archs:
                s *= 1.35 if len(archs) == 1 else 1.1
            elif archs and arch not in archs and not any(a in ("*", "any") for a in archs):
                s *= 0.6
        return s, matched


def evid(card):
    return card.get("evidence", "?")


def head(card):
    ev = evid(card)
    tag = "UNVERIFIED " if ev == "unverified" else ""
    arch = ",".join(card.get("scope", {}).get("arch") or []) or "any"
    tri = card.get("scope", {}).get("triton") or ""
    return f"{card['id']}  [{tag}{card.get('kind', '?')}; {ev.split(':')[0]}; {arch}{'; ' + tri if tri else ''}]"


def terse(card):
    lines = [head(card), "  " + card.get("claim", "")]
    if card.get("check"):
        lines.append("  check: " + card["check"])
    if card.get("overrides"):
        lines.append("  overrides: " + ", ".join(card["overrides"]))
    if card.get("see"):
        lines.append("  see: " + " ".join(card["see"]))
    return "\n".join(lines)


def one_line(card, width=110):
    s = re.split(r"(?<=[.;])\s", card.get("claim", ""), maxsplit=1)[0]
    s = s if len(s) <= width else s[:width - 3].rstrip() + "..."
    return f"{card['id']}: {s}"


def full(card):
    return json.dumps(card, indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("query", nargs="*", help="words or a pasted error message (quoting optional)")
    ap.add_argument("--arch", help="gfx950 | gfx942 | ...")
    ap.add_argument("--dtype", help="fp8 | bf16 | fp4 | ...")
    ap.add_argument("--kind", help="fact|error|api|isa|hw|tool|trap")
    ap.add_argument("--id", nargs="+", metavar="ID", help="print these cards in full")
    ap.add_argument("--toc", action="store_true", help="list groups and ids")
    ap.add_argument("--full", action="store_true", help="print whole cards (incl. method)")
    ap.add_argument("-n", type=int, default=6, help="max hits (default 6; the first 3 in full)")
    ap.add_argument("--store", default=STORE, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not os.path.exists(a.store):
        print(f"q.py: no fact store at {a.store}", file=sys.stderr)
        return 2
    cards = load(a.store)
    byid = {c["id"]: c for c in cards}

    if a.id:
        missing = [i for i in a.id if i not in byid]
        for i in a.id:
            if i in byid:
                print(full(byid[i]))
        for i in missing:
            near = [c for c in byid if i.split(".")[-1] in c][:5]
            print(f"q.py: no card {i}" + (f" (near: {' '.join(near)})" if near else ""), file=sys.stderr)
        return 1 if missing else 0

    pool = [c for c in cards if in_scope(c, "arch", a.arch) and in_scope(c, "dtype", a.dtype)
            and (not a.kind or c.get("kind") == a.kind)]

    if a.toc or (not a.query and a.kind):
        if a.toc:
            groups = {}
            for c in pool:
                groups.setdefault(c["id"].split(".")[0], []).append(c["id"])
            for g in sorted(groups):
                print(f"{g} ({len(groups[g])}): " + " ".join(i.split(".", 1)[1] for i in groups[g]))
        else:
            for c in pool:
                print(full(c) if a.full else one_line(c))
            if not a.full:
                print(f"-- {len(pool)} cards; q.py --id <id> for one card, q.py --kind {a.kind} <words> to search")
        return 0 if pool else 1

    if not a.query:
        ap.print_usage(sys.stderr)
        print("q.py: give query words, --id, --kind or --toc", file=sys.stderr)
        return 2

    query = " ".join(a.query)
    toks = qtokens(query)
    arch = (a.arch or "").lower() or next((ARCH_WORDS[t] for t in toks if t in ARCH_WORDS), "")
    idx = Index(pool)
    hits = []
    for i, c in enumerate(pool):
        s, matched = idx.score(i, query, toks, arch)
        if s > 0:
            hits.append((s, c))
    hits.sort(key=lambda h: (evid(h[1]) == "unverified", -h[0], h[1]["id"]))
    best = max((h[0] for h in hits), default=0)
    hits = [h for h in hits if h[0] >= max(1.0, best / 4)][:a.n]
    for k, (s, c) in enumerate(hits):
        if a.full:
            print(full(c))
        elif k < FULL_SHOWN:
            print(terse(c))
        else:
            print(head(c))
    if len(hits) > FULL_SHOWN and not a.full:
        print("-- more: q.py --id <id>")
    foot = []
    ql = query.lower()
    if re.search(r"\b(v|s|ds|buffer|global|flat|scratch)_[a-z0-9_]+", ql) or \
            re.search(r"\b(opcode|encoding|hazard|wait ?states|waitcnt|isa)\b", ql):
        foot.append(f"ISA: python3 {tool_path('amd_isa.py')} --arch {arch or 'gfx950'} search \"<words>\""
                    " (then facts/layout/lds/hazard/waitcnt; --help)")
    if re.search(r"\bmi3\d\d|\bsku\b|\bpeak|\bclock|\bxcd|\bcus?\b|\bhbm\b", ql):
        foot.append(f"SKU: python3 {tool_path('extract_sku.py')} --facts --sku <MI350X|MI355X|MI300X|MI325X>"
                    " (--help)")
    if foot:
        print("--\n" + "\n".join(foot))
    if not hits:
        print(f"q.py: no card matches {query!r}" + ("" if foot else "; try other words or --toc"),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
