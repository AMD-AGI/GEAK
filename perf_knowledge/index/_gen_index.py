#!/usr/bin/env python3
"""Generate the structured index over ALL THREE knowledge layers (plan Part 1.4 / P2).

Reads the frontmatter of every content file in
  * perf_knowledge/**              (layer: reference)
  * e2e_workflow/knowledge/learned (layer: learned)
  * kernel_workflow/knowledge/**   (layer: reference — static guides)
and produces, under perf_knowledge/index/:
  * kb_manifest.yaml               one machine record per file — the resolver's ONLY data source.
  * views/by_platform/<gen>.md     grouped by gfx (+ SKU subsections)
  * views/by_kernel_class/<c>.md   grouped by kernel_class
  * views/by_lever/<lever>.md      grouped by lever, sectioned by cost L0->L3
  * views/by_bound_type/<b>.md     grouped by roofline bound_type

_gen_registry.py (sota_registry / capability_index / sota_matrix) is left UNTOUCHED — this is additive.
Run from perf_knowledge/:  python3 index/_gen_index.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _kb_vocab as V  # noqa: E402

try:
    import yaml
except Exception:
    yaml = None

PK_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PK_ROOT)
LEARNED_ROOT = os.path.join(REPO, "e2e_workflow", "knowledge", "learned")
KW_KNOW = os.path.join(REPO, "kernel_workflow", "knowledge")
VIEWS = os.path.join(HERE, "views")

FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.S)
EXCLUDE = (os.sep + "expert_skills" + os.sep, os.sep + "index" + os.sep, os.sep + "_templates" + os.sep)
SKIP_NAMES = {"INDEX.md", "README.md", "_archive.md"}


def parse_fm(text):
    m = FM_RE.match(text)
    if not m:
        return None
    body = m.group(1)
    if yaml is not None:
        try:
            d = yaml.safe_load(body)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    # fallback: minimal key: value / key: [list] parser
    fm, cur = {}, None
    for line in body.splitlines():
        lm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if lm:
            k, v = lm.group(1), lm.group(2).strip()
            if v.startswith("[") and v.endswith("]"):
                fm[k] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
            elif v == "":
                fm[k] = []
                cur = k
            else:
                fm[k] = v
                cur = None
        else:
            lm2 = re.match(r"^\s*-\s*(.*)$", line)
            if lm2 and cur:
                fm.setdefault(cur, [])
                if isinstance(fm[cur], list):
                    fm[cur].append(lm2.group(1).strip())
    return fm


def as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def hook_of(text, fm):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()[:120]
    return str(fm.get("title", "")).strip()[:120]


def collect():
    files = []
    roots = [PK_ROOT, LEARNED_ROOT, KW_KNOW]
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            if any(x in dp + os.sep for x in EXCLUDE):
                continue
            for f in fn:
                if f.endswith(".md") and f not in SKIP_NAMES:
                    files.append(os.path.join(dp, f))
    recs = []
    for path in sorted(set(files)):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        fm = parse_fm(text)
        if not fm:
            continue
        rel = os.path.relpath(path, REPO)
        # Files under ANY .../knowledge/learned/ dir are ALWAYS layer:learned (e2e AND kernel_workflow
        # sinks), even if a card omits `layer:` — the directory is the source of truth, symmetrically.
        is_learned = (os.sep + "knowledge" + os.sep + "learned" + os.sep) in os.path.abspath(path)
        layer = "learned" if is_learned else str(fm.get("layer", "reference")).strip()
        platforms = V.gens_to_platforms(as_list(fm.get("platforms")) or as_list(fm.get("gens")))
        if not platforms:
            # Best-effort platform from a gfxNNNN token in the filename / key line (learned cards).
            toks = re.findall(r"gfx\d+", os.path.basename(path) + " " + str(fm.get("key", "")))
            platforms = V.gens_to_platforms(toks)
        recs.append({
            "path": rel,
            "title": str(fm.get("title", "")).strip(),
            "layer": layer,
            "kind": str(fm.get("kind", "")).strip(),
            "platforms": platforms,
            "skus": [s for s in as_list(fm.get("skus")) if s in V.SKU_IDS],
            "kernel_class": str(fm.get("kernel_class", "")).strip(),
            "levers": as_list(fm.get("levers")),
            "cost": str(fm.get("cost", "")).strip(),
            "risk": str(fm.get("risk", "")).strip(),
            "bound_type": as_list(fm.get("bound_type")),
            "lifecycle": str(fm.get("lifecycle", "")).strip() or "active",
            "status": str(fm.get("status", "")).strip(),
            "verified_on": str(fm.get("verified_on", "")).strip() or None,
            "hook": hook_of(text, fm),
        })
    return recs


def _yqs(s):
    """Quote a scalar for YAML only when needed."""
    s = str(s)
    if s == "":
        return '""'
    if re.search(r"[:#\[\]{}&*!|>'\"%@`]", s) or s[0] in "-? " or s[-1] == " ":
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


def write_manifest(recs):
    out = ["# kb_manifest.yaml — one record per KB file across all three layers (AUTO-GENERATED by",
           "# index/_gen_index.py). This is the SINGLE data source kb_resolve.py consumes. Do not hand-edit;",
           "# edit file frontmatter and re-run the generator.", "",
           f"# files: {len(recs)}", "records:"]
    listkeys = ("platforms", "skus", "levers", "bound_type")
    for r in recs:
        out.append(f"  - path: {_yqs(r['path'])}")
        for k in ("title", "layer", "kind", "kernel_class", "cost", "risk",
                  "lifecycle", "status", "verified_on", "hook"):
            v = r[k]
            out.append(f"    {k}: {'null' if v is None else _yqs(v)}")
        for k in listkeys:
            out.append(f"    {k}: [{', '.join(_yqs(x) for x in r[k])}]")
    open(os.path.join(HERE, "kb_manifest.yaml"), "w", encoding="utf-8").write("\n".join(out) + "\n")


def _lifemark(r):
    m = {"stale": " ⚠stale", "archived": " ⛔archived", "candidate": " ·candidate"}.get(r["lifecycle"], "")
    if r["verified_on"] is None and r["layer"] == "artifact":
        m += " ⚠unverified"
    return m


def _line(r, base):
    """`base` is the directory the view file is WRITTEN to, not VIEWS itself. Every view lands in
    VIEWS/<subdir>/, one level deeper, so relpath-ing against VIEWS drops a `../` from every link."""
    rel = os.path.relpath(os.path.join(REPO, r["path"]), base)
    return f"- [{r['title'] or r['path']}]({rel}) — {r['hook']}{_lifemark(r)}"


def rmtree_views():
    if not os.path.isdir(VIEWS):
        return
    for dp, _dn, fn in os.walk(VIEWS, topdown=False):
        for f in fn:
            os.unlink(os.path.join(dp, f))
        os.rmdir(dp)


def write_views(recs):
    rmtree_views()
    # by_platform (with SKU subsections)
    d = os.path.join(VIEWS, "by_platform")
    os.makedirs(d, exist_ok=True)
    for gen in V.GENS:
        sub = [r for r in recs if gen in r["platforms"]]
        agn = [r for r in recs if not r["platforms"]]
        lines = [f"# by_platform — {gen} ({V.GENS[gen]['arch']})", "",
                 "AUTO-GENERATED by index/_gen_index.py.", ""]
        lines.append(f"## {gen} — cards ({len(sub)})")
        lines += [_line(r, d) for r in sub] or ["- (none)"]
        for sku in V.GENS[gen]["skus"]:
            ss = [r for r in sub if sku in r["skus"]]
            if ss:
                lines += ["", f"### SKU {sku} ({len(ss)})"] + [_line(r, d) for r in ss]
        lines += ["", f"## platform-independent ({len(agn)})"] + [_line(r, d) for r in agn[:200]]
        open(os.path.join(d, f"{gen}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # by_kernel_class
    d = os.path.join(VIEWS, "by_kernel_class")
    os.makedirs(d, exist_ok=True)
    for kc in sorted(V.KERNEL_CLASSES):
        sub = [r for r in recs if r["kernel_class"] == kc]
        lines = [f"# by_kernel_class — {kc}", "", "AUTO-GENERATED by index/_gen_index.py.", "",
                 f"{len(sub)} cards."] + [""] + ([_line(r, d) for r in sub] or ["- (none)"])
        open(os.path.join(d, f"{kc}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # by_lever (sectioned by cost L0->L3)
    d = os.path.join(VIEWS, "by_lever")
    os.makedirs(d, exist_ok=True)
    for lev in sorted(V.LEVERS):
        sub = [r for r in recs if lev in r["levers"]]
        lines = [f"# by_lever — {lev}", "", "AUTO-GENERATED by index/_gen_index.py. Cheapest cost first.", ""]
        for cost in V.COSTS:
            cc = [r for r in sub if r["cost"] == cost]
            if cc:
                lines += [f"## {cost} — {V.COST_MEANING[cost]}"] + [_line(r, d) for r in cc] + [""]
        nocost = [r for r in sub if r["cost"] not in V.COSTS]
        if nocost:
            lines += ["## (cost unset)"] + [_line(r, d) for r in nocost]
        if not sub:
            lines += ["- (none)"]
        open(os.path.join(d, f"{lev}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # by_bound_type
    d = os.path.join(VIEWS, "by_bound_type")
    os.makedirs(d, exist_ok=True)
    for b in sorted(V.BOUND_TYPES):
        sub = [r for r in recs if b in r["bound_type"]]
        lines = [f"# by_bound_type — {b}", "", "AUTO-GENERATED by index/_gen_index.py.", "",
                 f"{len(sub)} cards.", ""] + ([_line(r, d) for r in sub] or ["- (none)"])
        open(os.path.join(d, f"{b}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    recs = collect()
    write_manifest(recs)
    write_views(recs)
    kc = sum(1 for r in recs if r["kernel_class"])
    lv = sum(1 for r in recs if r["levers"])
    bt = sum(1 for r in recs if r["bound_type"])
    print(f"OK: {len(recs)} records -> kb_manifest.yaml + views/. "
          f"kernel_class set on {kc}, levers on {lv}, bound_type on {bt}.")


if __name__ == "__main__":
    main()
