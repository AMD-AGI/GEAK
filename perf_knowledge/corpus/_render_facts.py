#!/usr/bin/env python3
"""Render extracted source evidence into the human-readable evidence index.

Split from the extractor on purpose. The extractor reads a tree it does not own and needs an aiter
checkout to run; this reads only committed YAML, so it runs in CI, on a laptop, and in a container
with no aiter and no GPU. That is what makes `--check` a usable gate: a doc that disagrees with the
evidence it projects from is a silent lie, and catching it must not depend on having the source tree.

What this page is allowed to say is the whole design question. It reports, per implementation
question, WHICH LANGUAGES STATE AN ANSWER IN SOURCE and where — nothing about which answer is better. The
distinction sounds pedantic until you look at the coverage table: `scheduling` has 132 FlyDSL hits
and none anywhere else, and the tempting sentence — "FlyDSL needs manual scheduling, the others do
not" — is only half right. Triton and CK schedule too; their scheduling is chosen by a compiler pass
or a pipeline enum rather than written per-instruction, so it is not in the grep. The table shows
where an answer is WRITTEN DOWN. Reading it as where an implementation choice EXISTS is the error this file is
built to make hard, which is why every count links to the lines behind it and no count carries an
adjective.

The same rule applies to shipped configuration data: a value shared by every selected shape config is
reported as agreement in those files, not as a measured winner. No benchmark archive or rejected
alternatives are attached here.
"""
import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
EVIDENCE = os.path.join(HERE, "evidence")
DOC = os.path.join(HERE, "gemm_source_evidence.md")

# Language order, and it is not alphabetical. FlyDSL first because this corpus exists to serve FlyDSL
# authors, then the languages in rough order of how much they hide: Triton and Gluon abstract the
# most, CK parameterises, HIP is explicit, asm is total control. Reading a row left to right is then
# reading the same implementation question at descending levels of abstraction.
LANG_ORDER = ["flydsl", "triton", "gluon", "ck", "hip", "asm"]

# What each implementation question asks. A gloss that stated a
# preference ("prefer 4 stages") would be a claim, and claims belong in expert skills where a card
# can carry the measurement that earned them. A question is safe and still orients the reader.
QUESTIONS = {
    "mfma_intrinsic": "Which matrix instruction is issued",
    "mfma_shape": "What M/N/K shape that instruction covers",
    "tile_shape": "How much of the output one workgroup owns",
    "waves": "How many waves cooperate on that tile",
    "split_k": "Whether K is split across workgroups, and how",
    "persistent": "Whether workgroups persist and loop over tiles",
    "grid_mapping": "How workgroup IDs are mapped onto tiles and XCDs",
    "scale_operand": "Which quantization-scale granularity the kernel consumes",
    "scale_in_k_loop": "Where a per-K-block scale multiplies each partial product inside the K loop",
    "scale_after_k_loop": "Where a per-token/per-channel scale multiplies the finished accumulator",
    "atomic_combine": "Where partial results are accumulated into the output with atomics",
    "grid_fill": "How the workgroup count is sized against the CU count to pick a tile or split",
    "lane_reduction": "Where partial sums are reduced across the lanes of a wave",
    "fragment_layout": "Which lane holds which row and column of an MFMA fragment",
    "jit_cache": "How a compiled kernel (or a looked-up config) is cached and keyed",
    "parity_tolerance": "Which tolerance the implementation's own test accepts against its reference",
    "dispatch_padding": "How the dispatcher pads a shape before looking up a tuned config",
    "config_lookup": "How a shape is mapped to a shipped tuned configuration",
    "config_space": "Which configuration values the library declares it will build",
    "config_limit": "Where that configuration space is capped",
    "config_validity": "Which configurations are rejected before compilation, and why",
    "kernel_family": "Which kernel family a shape is routed to",
    "arch_gate": "Which code path exists only on some architectures",
    "lds_stage": "How many pipeline stages are buffered in LDS",
    "lds_swizzle": "How LDS addresses are permuted to dodge bank conflicts",
    "lds_pad": "How LDS rows are padded for the same reason",
    "lds_alloc": "Where LDS is reserved",
    "async_copy": "Whether global->LDS copies are asynchronous",
    "access_width": "How wide a single memory access is",
    "cache_policy": "Which cache hint loads and stores carry",
    "layout_preshuffle": "Whether an operand is pre-permuted before the kernel runs",
    "layout_transpose": "Which operand is transposed",
    "layout_epilogue": "How the accumulator is rearranged on the way out",
    "scheduling": "Where instruction order is pinned by hand",
    "tunable_param": "Which knobs the kernel exposes (values live elsewhere)",
    "instance_name": "Configuration encoded in a generated kernel's name",
    "pipeline_variant": "Which named pipeline variant is being built",
    "ck_instance": "Which CK device-op template is instantiated",
    "ck_pipeline": "CK's loop scheduler and pipeline version",
    "asm_object": "Which precompiled object the launcher loads",
    "asm_launch": "Workgroup size and grid formula the assembly assumes",
    "asm_argblock": "Byte layout of the argument block it expects",
    "asm_tile": "Tile size the selected object was built for",
}

PROSE = {
    "header": (
        "# GEMM source evidence: how six implementations answer each development question\n"
        "\n"
        "GENERATED by `perf_knowledge/corpus/_render_facts.py` from `evidence/gemm_source.yaml`.\n"
        "Do not hand-edit. To refresh:\n"
        "\n"
        "```bash\n"
        "python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /path/to/aiter --emit\n"
        "python3 perf_knowledge/corpus/_render_facts.py --emit\n"
        "```\n"
    ),
    "what_this_is": (
        "## What this is, and what it is not\n"
        "\n"
        "This is the **evidence layer**, not the final development advice: an index of the same operator\n"
        "family written six ways. It lets a FlyDSL author inspect how somebody else answered a question\n"
        "before answering it from scratch. Every row carries a content-bound `src_…` identity and\n"
        "`file:line` in an AITER checkout at a named commit.\n"
        "\n"
        "It contains **no performance claims and no rankings.** Nothing here says a tile is good, or\n"
        "that one language beats another. A source statement is mechanically reproducible; a recommendation\n"
        "needs conditions and evidence strength. Those actionable cards live in "
        "[`gemm_decisions.md`](gemm_decisions.md). Measured claims are not copied into that always-on "
        "page; they remain behind the learned-card/expert-skill switches with their validation artifact.\n"
        "\n"
        "Read a count as **\"this many source locations matched this implementation question\"** — never as\n"
        "\"this language does this and the others do not.\" A compiler-chosen answer still exists; it\n"
        "is simply not in the source text. See the coverage note below.\n"
    ),
    "coverage_note": (
        "The zeros are the interesting part, and they are easy to misread. `scheduling` is dense in\n"
        "FlyDSL and empty elsewhere, but Triton kernels are scheduled too — by a compiler pass, and CK\n"
        "by a `BlockGemmPipelineScheduler` enum, neither of which is a per-instruction statement a grep\n"
        "can find. Likewise `tile_shape` is nearly empty for Triton not because Triton kernels have no\n"
        "tiles, but because their tiles arrive as `tl.constexpr` arguments whose values are in the\n"
        "shipped JSON (see the tunable-surface and tuned-starting-point sections).\n"
        "\n"
        "So: a filled cell means *an answer is written here, at these lines*. An empty cell means\n"
        "*not stated in source in a form this extractor recognises* — which may mean the answer is\n"
        "delegated, encoded in a name, or simply that no rule exists for that idiom yet. The rules are\n"
        "listed in `_extract_impl_facts.py`; a missing idiom is a fixable gap, not a finding.\n"
    ),
    "questions_intro": (
        "## Implementation questions\n"
        "\n"
        "One section per question. Within a section, each language lists the distinct source values and the\n"
        "lines that state them. Values are verbatim; where a rule captured a sub-expression, that is\n"
        "what is shown.\n"
    ),
    "arch_note": (
        "Values marked `[gfx…]` sit inside an architecture-conditional branch. That is a hint about\n"
        "where to look, not a filter: the same number outside a branch may still be arch-dependent in\n"
        "practice, and a number inside one is not necessarily wrong elsewhere.\n"
    ),
    "tuned_intro": (
        "## Shipped tuned starting points\n"
        "\n"
        "AITER's GEMM Triton kernels carry no `@triton.autotune`. Selected configurations ship instead\n"
        "as per-shape JSON, one file per `gfx`-and-variant, each mapping an M bucket to a knob set. The\n"
        "files provide concrete seeds known to be selected by AITER, but contain no benchmark archive,\n"
        "rejected alternatives or environment sufficient to call any row a measured winner.\n"
        "\n"
        "Grouped by `(gfx, variant, M bucket)`. **`same across configs`** means every shipped shape\n"
        "config in that group carries the value — useful seed evidence, not proof of a constraint.\n"
        "Knobs that moved are listed as varying, with value counts. Exact source JSON paths are retained\n"
        "in `evidence/gemm_tuned_configs.yaml` for per-shape lookup.\n"
    ),
    "gaps_intro": (
        "## Gaps, stated\n"
        "\n"
        "What the extractor looked for and did not find. Recorded because a silently-empty section and a\n"
        "genuinely-empty one are indistinguishable to a reader, and the difference decides whether the\n"
        "next person re-runs the search or trusts it.\n"
    ),
}


def load_yaml(path):
    """Read the generated evidence. PyYAML if present, else the subset this file emits.

    The fallback exists because the extractor hand-rolls its emitter to keep diffs readable and to
    stay importable without PyYAML; a renderer that then required PyYAML would put the pair back in
    the dependency it was avoiding.
    """
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    return _mini_yaml(path)


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        body = s[1:-1]
        # Order matters: `\\n` is a literal backslash followed by n, not a newline, so the escape
        # sequences have to be resolved in one left-to-right pass rather than by chained replaces.
        out, i = [], 0
        while i < len(body):
            if body[i] == "\\" and i + 1 < len(body):
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(body[i + 1],
                                                                            body[i + 1]))
                i += 2
            else:
                out.append(body[i])
                i += 1
        return "".join(out)
    if s == "true":
        return True
    if s == "false":
        return False
    if s == "{}":
        return {}
    if s == "[]":
        return []
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def _mini_yaml(path):
    """Parse exactly the shape `_extract_impl_facts.py` writes, and nothing more.

    Scoped to that shape deliberately: top-level map, `key:` opening either a list of maps or a
    nested map, one level of nesting inside a row, plus inline `[a, b]`. A general YAML parser is a
    bad thing to write by hand, and the test alongside this asserts equivalence with PyYAML on the
    real generated files so the two paths cannot drift apart unnoticed.

    The one subtlety is that `key:` with nothing after it does not say whether a list or a map
    follows — `evidence:` opens a list, `knobs:` opens a map, and both look identical on their own
    line. So the container type is left undecided until the first child arrives and its leading `- `
    (or absence) settles it. An earlier version guessed map, which quietly turned every evidence
    excerpt into a one-key dict whose key was the excerpt.
    """
    root = {}
    stack = [(-1, root)]          # (indent, container) — innermost last
    pending = None                # (indent, parent, key): container type not yet known
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raw = line.rstrip("\n")
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip())

            if pending is not None and indent > pending[0]:
                _, parent, key = pending
                container = [] if stripped.startswith("- ") else {}
                parent[key] = container
                stack.append((indent, container))
                pending = None
            elif pending is not None:
                # Nothing indented under it: an empty container. `[]` and `{}` both read as falsy
                # downstream, so the distinction does not matter to any caller here.
                pending[1][pending[2]] = []
                pending = None

            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            container = stack[-1][1]

            if stripped.startswith("- "):
                body = stripped[2:]
                if isinstance(container, list) and ":" in body and not body.startswith('"'):
                    row = {}
                    container.append(row)
                    stack.append((indent + 2, row))
                    key, _, rest = body.partition(":")
                    pending = _put(row, key.strip(), rest, indent + 2)
                elif isinstance(container, list):
                    container.append(_unquote(body))
                continue

            key, _, rest = stripped.partition(":")
            if isinstance(container, list):
                # Continuation of the row opened by the most recent `- `.
                container = stack[-1][1] if isinstance(stack[-1][1], dict) else {}
            pending = _put(container, key.strip(), rest, indent)
    if pending is not None:
        pending[1][pending[2]] = []
    return root


def _put(target, key, rest, indent):
    """Set `key`, or return a pending marker when its container type is not yet decidable."""
    rest = rest.strip()
    if rest == "":
        return (indent, target, key)
    if rest.startswith("[") and rest.endswith("]"):
        body = rest[1:-1].strip()
        target[key] = [_unquote(x) for x in _split_top(body)] if body else []
    elif rest == "{}":
        target[key] = {}
    else:
        target[key] = _unquote(rest)
    return None


def _split_top(body):
    out, depth, cur, quoted = [], 0, "", False
    for ch in body:
        if ch == '"':
            quoted = not quoted
        if not quoted:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append(cur)
                cur = ""
                continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


def cite(fact):
    """A `file:line` pointer. Plain text, not a link: the target is in another repo at another commit.

    A markdown link would render as clickable and resolve to nothing, or worse, to whatever happens
    to be at that path in the reader's checkout. The commit is in the provenance block above; the
    pair is what makes the pointer meaningful.
    """
    identity = f" (`{fact['evidence_id']}`)" if fact.get("evidence_id") else ""
    return f"`{location(fact)}`{identity}"


def location(fact):
    """`file:line`, or `file:line-end` when the extractor recorded the statement's extent."""
    end = fact.get("end_line")
    return f"{fact['file']}:{fact['line']}" + (f"-{end}" if end else "")


def value_of(fact):
    """What the rule captured, or the matched text when it captured nothing.

    Trimmed hard, because some matches are a whole template argument list and a table cell holding
    400 characters is a table nobody reads. Truncation is marked so the reader knows to open the file
    rather than trusting the fragment.
    """
    got = fact.get("captured") or []
    text = ", ".join(str(g) for g in got) if got else str(fact.get("match", ""))
    text = " ".join(text.split())
    return (text[:70] + "…") if len(text) > 71 else text


def arch_scopes(record):
    """Normalise extractor output without iterating a string into `g, f, x, 9, 5, 0`.

    The extractor emits one enclosing gfx branch as a scalar today; older/local artifacts may carry a
    list. Rendering accepts both, but always treats `gfx950` as one architecture label.
    """
    raw = record.get("arch_scope")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw if value]


def coverage_table(evidence):
    counts = collections.Counter((f["language"], f["category"]) for f in evidence)
    langs = [l for l in LANG_ORDER if any(k[0] == l for k in counts)]
    langs += sorted({k[0] for k in counts} - set(langs))
    rows = ["| implementation question | " + " | ".join(langs) + " |",
            "|---|" + "---|" * len(langs)]
    for cat in sorted({k[1] for k in counts}):
        cells = [str(counts.get((l, cat), 0)) if counts.get((l, cat)) else "·" for l in langs]
        rows.append(f"| `{cat}` | " + " | ".join(cells) + " |")
    tot = [str(sum(v for (l, _), v in counts.items() if l == L)) for L in langs]
    rows.append("| **total** | " + " | ".join(f"**{t}**" for t in tot) + " |")
    return rows


def question_section(cat, evidence):
    """One implementation question, grouped by language then by distinct source value.

    Grouped by value rather than listed per hit because the same tile appearing in nine files is one
    piece of information plus a note that it is common, not nine pieces. The line list is capped: the
    point is to give the reader an entry point, and a wall of fifty citations defeats that.
    """
    out = [f"### `{cat}` — {QUESTIONS.get(cat, 'uncategorised')}", ""]
    by_lang = collections.defaultdict(list)
    for f in evidence:
        by_lang[f["language"]].append(f)
    order = [l for l in LANG_ORDER if l in by_lang] + sorted(set(by_lang) - set(LANG_ORDER))
    for lang in order:
        group = collections.defaultdict(list)
        for f in by_lang[lang]:
            group[value_of(f)].append(f)
        out.append(f"**{lang}** — {len(by_lang[lang])} in {len({f['file'] for f in by_lang[lang]})} file(s)")
        out.append("")
        for val, hits in sorted(group.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:12]:
            arch = sorted({scope for hit in hits for scope in arch_scopes(hit)})
            tag = f" [{', '.join(arch)}]" if arch else ""
            where = ", ".join(cite(h) for h in hits[:3])
            more = f" +{len(hits) - 3} more" if len(hits) > 3 else ""
            out.append(f"- `{val}`{tag} — {where}{more}")
        if len(group) > 12:
            out.append(f"- …{len(group) - 12} further distinct value(s); see the YAML")
        out.append("")
    return out


def tuned_section(tuned, limit):
    """The grouped shipped configs, biggest groups first — more shape configs is broader support."""
    out = []
    ranked = sorted(tuned, key=lambda g: (-int(g.get("shape_configs") or 0),
                                          str(g.get("gfx")), str(g.get("tags"))))
    for g in ranked[:limit]:
        n = int(g.get("shape_configs") or 0)
        head = (f"**{g.get('gfx')} · {g.get('tags')} · {g.get('m_bucket')}** — "
                f"`{g.get('config_id')}` ({n} shipped shape config(s))")
        out += [head, ""]
        fixed = g.get("same_across_configs") or g.get("knobs") or {}
        varies = g.get("varies_by_shape") or {}
        if fixed:
            label = "same across shipped configs" if n > 1 else "the single shipped config"
            out.append(f"- {label}: " + ", ".join(f"`{k}={v}`" for k, v in sorted(fixed.items())))
        if varies:
            out.append("- varies by shape: "
                       + ", ".join(f"`{k}` ({v})" for k, v in sorted(varies.items())))
        sources = g.get("source_files") or []
        if sources:
            listed = ", ".join(f"`{path}`" for path in sources[:3])
            suffix = f" +{len(sources) - 3} more" if len(sources) > 3 else ""
            out.append(f"- source JSON: {listed}{suffix}")
        out.append("")
    if len(ranked) > limit:
        out += [f"…{len(ranked) - limit} further group(s) in `evidence/gemm_tuned_configs.yaml`.", ""]
    return out


def render(data, tuned_data):
    prov = data.get("provenance", {})
    evidence = data.get("source_evidence") or []
    L = [PROSE["header"], ""]
    L += ["| provenance | |", "|---|---|"]
    for k in ("aiter_origin", "aiter_commit", "operator_family", "evidence_records"):
        if k in prov:
            L.append(f"| {k} | `{prov[k]}` |")
    # Stated as a sentence rather than a field, because the empty case is the one that needs saying:
    # "these citations resolve" is the corpus's whole claim, and a reader should not have to infer it
    # from an absent list.
    dirt = prov.get("aiter_dirty_sources") or []
    if dirt:
        L += ["", f"**{len(dirt)} source file(s) had uncommitted changes when this was extracted**, so "
                  f"records from them are marked `unreproducible` and do not resolve at the commit "
                  f"above for anyone else: " + ", ".join(f"`{p}`" for p in dirt) + "."]
    else:
        L += ["", ("Every citation below resolves at that commit — no contributing file had "
                   "uncommitted changes when this was extracted.")]
    L += ["", PROSE["what_this_is"], "", "## Source-evidence coverage", ""]
    L += coverage_table(evidence)
    L += ["", PROSE["coverage_note"], "", PROSE["questions_intro"], "", PROSE["arch_note"], ""]

    by_cat = collections.defaultdict(list)
    for f in evidence:
        by_cat[f["category"]].append(f)
    # Questions in the declared order, so the page reads instruction -> tile -> memory -> layout ->
    # scheduling rather than alphabetically. Unknown categories land at the end, visibly.
    for cat in list(QUESTIONS) + sorted(set(by_cat) - set(QUESTIONS)):
        if cat in by_cat:
            L += question_section(cat, by_cat[cat])

    tuned = (tuned_data or {}).get("tuned_configs") or []
    if tuned:
        tprov = (tuned_data or {}).get("provenance", {})
        L += [PROSE["tuned_intro"], ""]
        L.append(f"Source: `{tprov.get('source', '?')}` — "
                 f"{tprov.get('rows_summarised', '?')} rows in {len(tuned)} groups.")
        L.append("")
        L += tuned_section(tuned, limit=24)

    gaps = data.get("missing") or []
    L += [PROSE["gaps_intro"], ""]
    if gaps:
        for g in gaps:
            L.append(f"- **{g.get('language')}** in `{g.get('subtree')}` — {g.get('why')}")
    else:
        L.append("None: every declared subtree matched at least one file.")
    L.append("")
    L += sources_section(prov, tuned_data)
    return "\n".join(L) + "\n"


def sources_section(prov, tuned_data):
    """The `## Sources` every content doc in this knowledge base carries.

    For a generated file this is not decoration: the whole page is derived, and a reader who wants to
    check a row needs the repo, the commit, and the intermediate YAML — the doc itself cites
    `file:line` with no repo attached, which is unresolvable without exactly these three.
    """
    origin, commit = prov.get("aiter_origin", "?"), prov.get("aiter_commit", "?")
    out = [
        "## Sources",
        "",
        (f"- **AITER** `{origin}` at commit `{commit}` — every `file:line` on this page is that "
         f"repo at that commit. Nothing here is quoted from anywhere else."),
        ("- Extracted source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml), produced by "
         "[`_extract_impl_facts.py`](_extract_impl_facts.py)."),
    ]
    if (tuned_data or {}).get("tuned_configs"):
        out.append("- Shipped tuning tables: "
                   "[`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml), summarised from "
                   f"`{(tuned_data.get('provenance') or {}).get('source', '?')}` in the same tree.")
    out += [
        ("- This page is generated by [`_render_facts.py`](_render_facts.py); edit the extractor or "
         "the source evidence, never this file."),
        "",
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true", help="write gemm_source_evidence.md")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if gemm_source_evidence.md differs from the evidence (CI gate)")
    a = ap.parse_args()

    src = os.path.join(EVIDENCE, "gemm_source.yaml")
    if not os.path.exists(src):
        print(f"no source evidence to render: {os.path.relpath(src, ROOT)} "
              f"(run _extract_impl_facts.py --emit first)", file=sys.stderr)
        return 2
    tuned_src = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
    text = render(load_yaml(src),
                  load_yaml(tuned_src) if os.path.exists(tuned_src) else None)

    if a.check:
        old = ""
        if os.path.exists(DOC):
            with open(DOC, encoding="utf-8") as f:
                old = f.read()
        if old != text:
            print(f"STALE: {os.path.relpath(DOC, ROOT)} does not match "
                  f"{os.path.relpath(src, ROOT)}. Re-run with --emit.", file=sys.stderr)
            return 1
        print(f"OK: {os.path.relpath(DOC, ROOT)} matches the source evidence")
        return 0

    if not a.emit:
        sys.stdout.write(text)
        return 0
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OK: {len(text.splitlines())} lines -> {os.path.relpath(DOC, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
