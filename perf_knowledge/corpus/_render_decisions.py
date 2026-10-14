#!/usr/bin/env python3
"""Render actionable GEMM development cards from curated decisions and shipped tuning evidence.

The source-evidence index answers "what is written in AITER?". This renderer answers the next
question an author actually has: "given my conditions, what should I try, why, what are the
alternatives, and how strong is the evidence?". It does not infer that from regex hits. Curated cards
state those semantics explicitly, and every citation — source line, shipped selection, background
doc — is checked against what it names.

The page follows the order an author writes a kernel in — `development_order` in the performance
axes, from the math contract to validation. A card sits under the step of its first axis, which is
the question it answers, and is linked from every other step its axes touch, so a decision taken in
one step shows up where its consequences land. A card's `solutions` block carries the same question
as other backends answer it.

AITER's shipped tuning tables are safe to derive mechanically: a knob shared by every selected config
in a group is a useful value to seed; a knob that varied is one to sweep; a backend-selection count
says where AITER's tuner chose each backend. No benchmark archive is attached, so "selected" never
becomes "winner" or a ranking. The Triton JSON seeds are rendered to their own page, because they are
a different backend's knob names and most of the old decision page by volume.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)

sys.path.insert(0, HERE)
import _render_facts as source_renderer

DECISIONS = os.path.join(HERE, "decisions", "gemm.yaml")
SOURCE_EVIDENCE = os.path.join(HERE, "evidence", "gemm_source.yaml")
TUNED_EVIDENCE = os.path.join(HERE, "evidence", "gemm_tuned_configs.yaml")
TUNED_DB_EVIDENCE = os.path.join(HERE, "evidence", "gemm_csv_tuned_configs.yaml")
PERFORMANCE_AXES = os.path.join(HERE, "performance_axes", "gemm.yaml")
DOC = os.path.join(HERE, "gemm_decisions.md")
TRITON_DOC = os.path.join(HERE, "gemm_triton_seeds.md")

SCHEMA_VERSION = 4
REQUIRED = {
    "id", "type", "axes", "match", "question", "evidence_level", "status", "conditions",
    "actions", "alternatives", "why", "source_evidence", "measurement_evidence", "limitations",
}
OPTIONAL = {"solutions", "references", "shipped_evidence"}
LEVELS = {"source_observed"}
CARD_TYPES = {"semantic", "constraint", "performance_candidate"}
# Within a step, what must hold is read before what to try.
TYPE_RANK = {"constraint": 0, "semantic": 1, "performance_candidate": 2}
# Same vocabulary as the source evidence and `kernel_workflow/scripts/kb.py`, so a solution's backend
# can be checked against the language of the evidence it cites.
BACKENDS = {"flydsl", "triton", "gluon", "ck", "hip", "asm"}
MATCH_LIST_FIELDS = {
    "operator_families", "target_languages", "gfx", "exclude_gfx", "dtypes", "regimes",
    "bottlenecks", "requires",
}
SHAPE_FIELDS = {"m_min", "m_max", "n_min", "n_max", "k_min", "k_max"}
# Exact spellings only. A `fp8*` family once made every per-token A8 card eligible for 128x128
# block-scale tasks, which is a different math contract rather than a looser match.
DTYPE_PATTERN = re.compile(r"[a-z0-9_]+")
TRAIT_NAME = re.compile(r"[a-z][a-z0-9_]*")
REFERENCE = re.compile(r"(?P<path>[^:\s]+\.md)(?::(?P<lo>\d+)(?:-(?P<hi>\d+))?)?")
GFX_ORDER = {"gfx950": 0, "gfx942": 1, "gfx1250": 2, "gfx1201": 3}
LEVEL_TEXT = {
    "source_observed": (
        "Source-observed candidate — the cited implementation exists; no performance preference is "
        "implied."
    ),
}
M_BUCKET_ORDER = ("M<=16", "M17-64", "M65-256", "M257-1024", "M1025-4096", "M>4096")


def validate_match(label, match):
    """Validate the deterministic filter without pretending it is a performance verdict."""
    errors = []
    if not isinstance(match, dict) or not match:
        return [f"{label}: match must be a non-empty mapping"]
    unknown = sorted(set(match) - MATCH_LIST_FIELDS - {"shape"})
    if unknown:
        errors.append(f"{label}: unknown match fields {unknown}")
    for field in MATCH_LIST_FIELDS:
        if field not in match:
            continue
        values = match[field]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            errors.append(f"{label}: match.{field} must be a non-empty string list")
    for value in match.get("dtypes") or []:
        if isinstance(value, str) and not DTYPE_PATTERN.fullmatch(value):
            errors.append(f"{label}: match.dtypes value {value!r} is not an exact dtype spelling")
    if not match.get("operator_families"):
        errors.append(f"{label}: match.operator_families is required")
    if not match.get("target_languages"):
        errors.append(f"{label}: match.target_languages is required")
    if set(match.get("gfx") or []) & set(match.get("exclude_gfx") or []):
        errors.append(f"{label}: match.gfx and match.exclude_gfx overlap")
    shape = match.get("shape")
    if shape is not None:
        if not isinstance(shape, dict) or not shape:
            errors.append(f"{label}: match.shape must be a non-empty mapping")
        else:
            shape_unknown = sorted(set(shape) - SHAPE_FIELDS)
            if shape_unknown:
                errors.append(f"{label}: unknown match.shape fields {shape_unknown}")
            for field, value in shape.items():
                if not isinstance(value, int) or value < 0:
                    errors.append(f"{label}: match.shape.{field} must be a non-negative integer")
            for axis in ("m", "n", "k"):
                low, high = shape.get(f"{axis}_min"), shape.get(f"{axis}_max")
                if isinstance(low, int) and isinstance(high, int) and low > high:
                    errors.append(f"{label}: match.shape.{axis}_min exceeds {axis}_max")
    return errors


def _reference_errors(label, ref):
    m = REFERENCE.fullmatch(str(ref))
    if not m:
        return [f"{label}: reference {ref!r} is not `path.md[:line[-line]]`"]
    path = os.path.normpath(os.path.join(PK, m.group("path")))
    if not path.startswith(PK + os.sep) or not os.path.isfile(path):
        return [f"{label}: reference {ref!r} does not name a file under perf_knowledge/"]
    if m.group("lo"):
        with open(path, encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        lo, hi = int(m.group("lo")), int(m.group("hi") or m.group("lo"))
        if not 1 <= lo <= hi <= count:
            return [f"{label}: reference {ref!r} is outside the file's {count} lines"]
    return []


def validate_traits(traits):
    """The design properties `match.requires` may name, each with the sentence a planner reads."""
    if not isinstance(traits, dict) or not traits:
        return ["traits must be a non-empty mapping of name -> description"]
    errors = []
    for name, text in traits.items():
        if not TRAIT_NAME.fullmatch(str(name)):
            errors.append(f"trait {name!r} is not a lower_snake_case name")
        if not str(text or "").strip():
            errors.append(f"trait {name!r} has no description")
    return errors


def development_steps(axes_data):
    """`development_order` from the axes file, and the step each axis belongs to.

    The steps order the page; they add no selection semantics. Every axis must sit in exactly one
    step, or a card whose first axis was left out would have nowhere to be rendered.
    """
    steps = (axes_data or {}).get("development_order") or []
    axis_ids = [str(axis.get("id")) for axis in (axes_data or {}).get("axes") or []]
    errors = [] if steps else ["performance axes publish no development_order"]
    step_of = {}
    for i, step in enumerate(steps, 1):
        if step.get("step") != i:
            errors.append(f"development_order: step {step.get('step')!r} is out of order; expected {i}")
        if not str(step.get("title") or "").strip():
            errors.append(f"development_order: step {i} has no title")
        for axis in step.get("axes") or []:
            if axis not in axis_ids:
                errors.append(f"development_order: step {i} names unknown axis {axis!r}")
            elif axis in step_of:
                errors.append(f"development_order: axis {axis!r} is in steps {step_of[axis]} and {i}")
            else:
                step_of[axis] = i
    unplaced = [axis for axis in axis_ids if axis not in step_of]
    if steps and unplaced:
        errors.append(f"development_order: axes in no step: {unplaced}")
    return steps, step_of, errors


def validate(cards, source_evidence, axis_ids=None, shipped=None, traits=None):
    """Return validation errors; never silently render an ungrounded decision card.

    `axis_ids`, `shipped` and `traits` are optional so the check can run on the source evidence
    alone (as the older tests do); when given, a card's axes, shipped-selection IDs and required
    traits must resolve too.
    """
    errors = []
    evidence_ids = [record.get("evidence_id") for record in source_evidence]
    available = {record.get("evidence_id"): record for record in source_evidence}
    if None in available:
        errors.append("source evidence record has no evidence_id; regenerate with the current extractor")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("source evidence contains duplicate evidence_id values")

    def check_source_refs(label, refs, backend=None):
        for ref in refs or []:
            if not re.fullmatch(r"src_[0-9a-f]{16}", str(ref)):
                errors.append(f"{label}: source_evidence must use a content-bound `src_…` ID: {ref}")
            elif ref not in available:
                errors.append(f"{label}: source evidence ID not present in extracted evidence: {ref}")
            elif backend and available[ref].get("language") != backend:
                errors.append(
                    f"{label}: {ref} is {available[ref].get('language')} evidence, not {backend}"
                )

    ids = set()
    for i, card in enumerate(cards):
        label = card.get("id") or f"card[{i}]"
        missing = sorted(REQUIRED - set(card))
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        unknown = sorted(set(card) - REQUIRED - OPTIONAL)
        if unknown:
            errors.append(f"{label}: unknown fields {unknown}")
        if label in ids:
            errors.append(f"{label}: duplicate id")
        ids.add(label)
        if card["type"] not in CARD_TYPES:
            errors.append(f"{label}: unknown card type {card['type']!r}")
        axes = card.get("axes")
        if not isinstance(axes, list) or not axes:
            errors.append(f"{label}: axes must be a non-empty list")
        elif axis_ids is not None:
            for axis in axes:
                if axis not in axis_ids:
                    errors.append(f"{label}: unknown performance axis {axis!r}")
            if len(set(axes)) != len(axes):
                errors.append(f"{label}: axes repeat an axis")
        errors.extend(validate_match(label, card["match"]))
        if traits is not None:
            for trait in (card["match"] or {}).get("requires") or []:
                if trait not in traits:
                    errors.append(f"{label}: match.requires names undefined trait {trait!r}")
        if card["evidence_level"] not in LEVELS:
            errors.append(f"{label}: unknown evidence_level {card['evidence_level']!r}")
        for field in ("conditions", "actions", "why", "limitations"):
            if not card.get(field):
                errors.append(f"{label}: {field} must not be empty")
        if not card.get("source_evidence"):
            errors.append(f"{label}: source_observed card needs source_evidence")
        if card.get("measurement_evidence"):
            errors.append(
                f"{label}: measured evidence belongs in learned cards/expert skills behind their "
                "feature switch, not in the always-on corpus"
            )
        check_source_refs(label, card.get("source_evidence"))
        for j, solution in enumerate(card.get("solutions") or []):
            slabel = f"{label}.solutions[{j}]"
            if not isinstance(solution, dict):
                errors.append(f"{slabel}: must be a mapping")
                continue
            if solution.get("backend") not in BACKENDS:
                errors.append(f"{slabel}: backend must be one of {sorted(BACKENDS)}")
            if not str(solution.get("mechanism") or "").strip():
                errors.append(f"{slabel}: mechanism must not be empty")
            if not solution.get("source_evidence"):
                errors.append(f"{slabel}: a solution needs its own source evidence")
            check_source_refs(slabel, solution.get("source_evidence"), solution.get("backend"))
        for ref in card.get("references") or []:
            errors.extend(_reference_errors(label, ref))
        for ref in card.get("shipped_evidence") or []:
            if not re.fullmatch(r"(cfg|sel)_[0-9a-f]{16}", str(ref)):
                errors.append(f"{label}: shipped_evidence must be a `cfg_…` or `sel_…` ID: {ref}")
            elif shipped is not None and ref not in shipped:
                errors.append(f"{label}: shipped evidence ID not present in tuned evidence: {ref}")
    return errors


def _bullets(items):
    return [f"- {item}" for item in items]


def render_match(match):
    out = []
    for field in (
        "operator_families", "target_languages", "gfx", "exclude_gfx", "dtypes", "regimes",
        "bottlenecks", "requires",
    ):
        if match.get(field):
            out.append(f"- `{field}`: " + ", ".join(f"`{value}`" for value in match[field]))
    if match.get("shape"):
        values = ", ".join(f"`{key}={value}`" for key, value in sorted(match["shape"].items()))
        out.append(f"- `shape`: {values}")
    return out


def _backend_counts(counts):
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    return " · ".join(f"{name} {n}" for name, n in ordered)


def describe_shipped(ref, shipped):
    record = shipped[ref]
    kind = record["_kind"]
    if kind == "flydsl":
        return (f"`{ref}` — FlyDSL `{record['family']}` `{record['dtype']}` selections in "
                f"`{record['db_kind']}` on `{record['gfx']}`/{record['cu_num']} CUs, "
                f"`{record['m_bucket']}`: {record['shape_configs']} shipped row(s)")
    if kind == "selection":
        return (f"`{ref}` — backend selected by AITER's tuner in `{record['db_kind']}` on "
                f"`{record['gfx']}`/{record['cu_num']} CUs, `{record['m_bucket']}`: "
                f"{_backend_counts(record['selected_backend_rows'])} (rows)")
    return (f"`{ref}` — Triton seed `{' '.join(record.get('tags') or [])}` on `{record['gfx']}`, "
            f"`{record['m_bucket']}` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))")


def render_card(card, evidence_by_id, shipped):
    level = card["evidence_level"]
    out = [
        f"### {card['question']}",
        "",
        (f"**Card:** `{card['id']}` · **type:** `{card['type']}` · "
         f"**evidence:** `{level}` · **status:** `{card['status']}` · "
         f"**axes:** " + ", ".join(f"`{a}`" for a in card["axes"])),
        "",
        LEVEL_TEXT[level],
        "",
        "#### Machine match",
        "",
        *render_match(card["match"]),
        "",
        "#### Use when",
        "",
        *_bullets(card["conditions"]),
        "",
        "#### Try",
        "",
        *_bullets(card["actions"]),
        "",
        "#### Why this is a candidate",
        "",
        *_bullets(card["why"]),
    ]
    if card.get("alternatives"):
        out += ["", "#### Keep as alternatives", "", *_bullets(card["alternatives"])]
    if card.get("solutions"):
        out += ["", "#### Same question in other implementations", ""]
        for solution in card["solutions"]:
            refs = ", ".join(
                f"`{ref}` (`{source_renderer.location(evidence_by_id[ref])}`)"
                for ref in solution["source_evidence"]
            )
            out.append(f"- **{solution['backend']}** — {solution['mechanism']}. Evidence: {refs}")
    out += ["", "#### Evidence", ""]
    for ref in card.get("source_evidence") or []:
        record = evidence_by_id[ref]
        out.append(
            f"- `{ref}` — `{record['category']}` `{source_renderer.value_of(record)}` at "
            f"`{source_renderer.location(record)}`"
        )
    for ref in card.get("shipped_evidence") or []:
        out.append(f"- {describe_shipped(ref, shipped)}")
    if card.get("references"):
        out += ["", "#### Background (always-on reference docs, not measurements)", ""]
        for ref in card["references"]:
            path = REFERENCE.fullmatch(ref).group("path")
            out.append(f"- [`{ref}`](../{path})")
    out += ["", "#### Limits", "", *_bullets(card["limitations"]), ""]
    return out


def _fixed_values(group):
    return group.get("same_across_configs") or group.get("knobs") or {}


def _value(value):
    return "true" if value is True else "false" if value is False else str(value)


def _top_values(text, limit=3):
    parts = [p for p in str(text).split(", ") if p]
    shown = ", ".join(parts[:limit])
    return shown + (f" (+{len(parts) - limit} more)" if len(parts) > limit else "")


def render_flydsl_seed_table(groups):
    """FlyDSL configurations AITER shipped, as seed-candidate / vary-next rows per M bucket."""
    out = [
        "## Shipped configuration seeds: FlyDSL (AITER tuned databases)",
        "",
        ("Generated from the FlyDSL rows of AITER's tuned GEMM databases (`aiter/configs/**/*tuned*gemm*"
         ".csv`), decoded from AITER's own kernel-name serialisation. Match gfx, CU count, database, "
         "family, dtype and M bucket before using a row. **Seed candidate** means every selected row in "
         "the group carries the value; **vary next** lists knobs that moved with the shape, with the "
         "most frequent values first. The tuner's timings are not carried, so these are concrete "
         "candidates, not measured winners."),
        "",
    ]
    by_db = {}
    for group in groups:
        key = (group["gfx"], group["cu_num"], group["db_kind"], group["family"], group["dtype"])
        by_db.setdefault(key, []).append(group)
    for key in sorted(by_db, key=lambda k: (GFX_ORDER.get(k[0], 99), k)):
        gfx, cu, db_kind, family, dtype = key
        rows = sorted(by_db[key], key=lambda g: M_BUCKET_ORDER.index(g["m_bucket"])
                      if g["m_bucket"] in M_BUCKET_ORDER else 99)
        out += [
            f"### `{gfx}` · {cu} CUs · `{db_kind}` · FlyDSL `{family}` · `{dtype}`",
            "",
            "| decision ref | M bucket | shipped rows | seed candidate | vary next |",
            "|---|---|---|---|---|",
        ]
        for group in rows:
            fixed = _fixed_values(group)
            seed = ", ".join(f"`{k}={_value(v)}`" for k, v in sorted(fixed.items())) or "none shared"
            varies = group.get("varies_by_shape") or {}
            if varies:
                vary = "; ".join(f"`{k}`: {_top_values(v)}" for k, v in sorted(varies.items()))
            else:
                vary = "one row; nothing to compare"
            out.append(f"| `{group['config_id']}` | `{group['m_bucket']}` | {group['shape_configs']} | "
                       f"{seed} | {vary} |")
        out.append("")
    return out


def render_selection_table(selection, inventory, gaps):
    """Which backend AITER's tuner selected per shape bucket: counts of rows, never speedups."""
    out = [
        "## Backend AITER's tuner selected, by shape bucket",
        "",
        ("Each row counts the shipped tuned-database rows whose selected backend (`libtype`) was each "
         "implementation, per gfx, CU count, database and M bucket. It says where AITER chose a "
         "backend, not by how much: timings, the competing set and the tuner's box are not recorded, "
         "and a shape shipped in two model databases counts twice. Use it to see which implementation "
         "a FlyDSL kernel competes with in production for a case — and where FlyDSL has never been "
         "selected."),
        "",
    ]
    by_db = {}
    for group in selection:
        by_db.setdefault((group["gfx"], group["cu_num"], group["db_kind"]), []).append(group)
    for key in sorted(by_db, key=lambda k: (GFX_ORDER.get(k[0], 99), k)):
        gfx, cu, db_kind = key
        rows = sorted(by_db[key], key=lambda g: M_BUCKET_ORDER.index(g["m_bucket"])
                      if g["m_bucket"] in M_BUCKET_ORDER else 99)
        out += [
            f"### `{gfx}` · {cu} CUs · `{db_kind}`",
            "",
            "| selection ref | M bucket | rows | selected backend (rows) |",
            "|---|---|---|---|",
        ]
        for group in rows:
            out.append(f"| `{group['selection_id']}` | `{group['m_bucket']}` | {group['rows']} | "
                       f"{_backend_counts(group['selected_backend_rows'])} |")
        out.append("")
    if gaps:
        out += ["Databases read but not counted as selections:", ""]
        out += [f"- `{gap['subtree']}` — {gap['why']}" for gap in gaps]
        out.append("")
    out.append(f"Databases inventoried: {len(inventory)} "
               f"({sum(int(i.get('rows') or 0) for i in inventory)} rows).")
    out.append("")
    return out


def render_triton_tuning_table(tuned):
    """Turn each shipped Triton config group into a concrete seed-candidate / vary-next instruction."""
    out = [
        "## Shipped configuration seeds: Triton",
        "",
        ("These rows are generated from AITER's shipped selected configs. Match all three condition "
         "columns before using a row. **Seed candidate** means every config in that group carries the "
         "value; **vary next** lists knobs that changed by shape. No benchmark archive or rejected "
         "alternatives are attached, so these are concrete candidates, not measured winners."),
        "",
        ("**These are Triton knobs.** The shipped selected configs live under "
         "`aiter/ops/triton/configs/gemm`, so the names are Triton's — `BLOCK_SIZE_M/N/K`, "
         "`num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, "
         "`cache_modifier`. Only the block tile carries over to another backend more or less "
         "directly; the rest have no one-to-one FlyDSL equivalent and several have none at all. If "
         "you are authoring FlyDSL, the FlyDSL seeds on [`gemm_decisions.md`](gemm_decisions.md) are "
         "the ones in your knob names, and a row here is at best a hint about which tile shapes "
         "somebody found worth shipping for a given M bucket."),
        "",
    ]
    by_gfx = {}
    for group in tuned:
        by_gfx.setdefault(str(group.get("gfx")), []).append(group)
    for gfx in sorted(by_gfx, key=lambda value: (GFX_ORDER.get(value, 99), value)):
        groups = sorted(by_gfx[gfx], key=lambda g: (str(g.get("tags")), str(g.get("m_bucket"))))
        out += [
            f"### `{gfx}`",
            "",
            "| decision ref | variant | M bucket | seed candidate | vary next | shipped support |",
            "|---|---|---|---|---|---|",
        ]
        for group in groups:
            fixed = _fixed_values(group)
            shapes = int(group.get("shape_configs") or 0)
            if fixed:
                seed = ", ".join(f"`{k}={v}`" for k, v in sorted(fixed.items()))
            else:
                seed = "none shared by all configs"
            varies = group.get("varies_by_shape") or {}
            if varies:
                vary = ", ".join(f"`{k}`" for k in sorted(varies))
            elif shapes == 1:
                vary = "all exposed knobs; one config cannot establish agreement"
            else:
                vary = "no varying knob recorded"
            source_count = len(group.get("source_files") or [])
            out.append(
                f"| `{group.get('config_id')}` | `{group.get('tags')}` | "
                f"`{group.get('m_bucket')}` | {seed} | {vary} | "
                f"{shapes} selected shape config(s) in {source_count} JSON file(s) |"
            )
        out.append("")
    return out


def shipped_index(tuned_data, tuned_db_data):
    out = {}
    for group in tuned_data.get("tuned_configs") or []:
        out[group["config_id"]] = dict(group, _kind="triton")
    for group in tuned_db_data.get("tuned_configs") or []:
        out[group["config_id"]] = dict(group, _kind="flydsl")
    for group in tuned_db_data.get("backend_selection") or []:
        out[group["selection_id"]] = dict(group, _kind="selection")
    return out


def axis_ids_of(axes_data):
    return {str(axis.get("id")) for axis in (axes_data or {}).get("axes") or []}


def _checked(decision_data, source_data, tuned_data, tuned_db_data, axes_data):
    schema = (decision_data.get("provenance") or {}).get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported decisions schema_version {schema!r}; expected {SCHEMA_VERSION}")
    cards = decision_data.get("cards") or []
    source_evidence = source_data.get("source_evidence") or []
    shipped = shipped_index(tuned_data, tuned_db_data)
    axis_ids = axis_ids_of(axes_data) if axes_data is not None else None
    traits = decision_data.get("traits") or {}
    errors = validate_traits(traits)
    errors += validate(cards, source_evidence, axis_ids, shipped, traits)
    if axes_data is not None:
        errors += development_steps(axes_data)[2]
    if errors:
        raise ValueError("\n".join(errors))
    return cards, source_evidence, shipped


def render_arch_coverage(cards, source_evidence, tuned_data, tuned_db_data):
    """How much of the corpus speaks to each architecture: counted per gfx, never ranked."""
    triton = tuned_data.get("tuned_configs") or []
    flydsl = tuned_db_data.get("tuned_configs") or []
    selection = tuned_db_data.get("backend_selection") or []
    gfxs = {str(g.get("gfx")) for g in triton + flydsl + selection}
    gfxs |= {r["arch_scope"] for r in source_evidence if r.get("arch_scope")}
    for card in cards:
        gfxs |= set(card["match"].get("gfx") or []) | set(card["match"].get("exclude_gfx") or [])
    gfxs = sorted((g for g in gfxs if g.startswith("gfx")),
                  key=lambda g: (GFX_ORDER.get(g, 99), g))
    out = [
        "## Coverage by architecture",
        "",
        ("What this corpus holds for each target, so a plan knows how much it is standing on. A card "
         "counts for a gfx when its match does not exclude it. *Arch-specific source* counts gates "
         "that name the gfx (or a prefix of it, such as `gfx95`) plus records whose code sits under "
         "such a gate. ASM kernels are the rows of AITER's per-arch kernel inventories "
         "(`hsa/<gfx>/<kind>/*.csv`). Counts, not rankings."),
        "",
        "| gfx | cards that can apply | arch-specific source records | ASM kernels by kind | FlyDSL seed groups | backend-selection groups | Triton seed groups |",
        "|---|---|---|---|---|---|---|",
    ]
    for gfx in gfxs:
        applies = sum(
            1 for c in cards
            if (not c["match"].get("gfx") or gfx in c["match"]["gfx"])
            and gfx not in (c["match"].get("exclude_gfx") or [])
        )
        gated = sum(
            1 for r in source_evidence if r["category"] != "asm_tile" and (
                r.get("arch_scope") == gfx
                or (r["category"] == "arch_gate" and any(
                    str(g).startswith("gfx") and gfx.startswith(str(g)) for g in r["captured"]))
            )
        )
        kinds = {}
        for r in source_evidence:
            if r["category"] == "asm_tile" and r.get("arch_scope") == gfx:
                parts = r["file"].split("/")
                kind = parts[2] if len(parts) > 3 and parts[0] == "hsa" else "launcher"
                kinds[kind] = kinds.get(kind, 0) + 1
        inventory = ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) or "—"
        out.append(
            f"| `{gfx}` | {applies} of {len(cards)} | {gated} | {inventory} | "
            f"{sum(1 for g in flydsl if g.get('gfx') == gfx)} | "
            f"{sum(1 for g in selection if g.get('gfx') == gfx)} | "
            f"{sum(1 for g in triton if str(g.get('gfx')) == gfx)} |"
        )
    out.append("")
    return out


def render_traits(traits):
    out = [
        "## Design traits",
        "",
        ("A card whose match `requires` a trait is **deferred** by `_select_candidates.py` until you "
         "pass `--trait <name>`. A trait describes your design, which the selector cannot see, so "
         "deferred means \"applies once your kernel does this\", not \"does not apply\"."),
        "",
    ]
    out += [f"- `{name}` — {text}" for name, text in traits.items()]
    out.append("")
    return out


def render_steps(cards, axes_data, evidence_by_id, shipped):
    """The cards in development order: an index, then one section per step."""
    steps, step_of, _ = development_steps(axes_data)
    questions = {str(axis.get("id")): axis.get("question") for axis in axes_data.get("axes") or []}
    home = {card["id"]: step_of[card["axes"][0]] for card in cards}
    here = {step["step"]: sorted((c for c in cards if home[c["id"]] == step["step"]),
                                 key=lambda c: TYPE_RANK[c["type"]])
            for step in steps}
    touches = {step["step"]: [c for c in cards if home[c["id"]] != step["step"]
                              and any(step_of[a] == step["step"] for a in c["axes"][1:])]
               for step in steps}
    out = [
        "## Development order",
        "",
        "| step | axes | cards that answer it | cards from other steps that bear on it |",
        "|---|---|---|---|",
    ]
    for step in steps:
        n = step["step"]
        axes = ", ".join(f"`{a}`" for a in step.get("axes") or []) or "—"
        own = ", ".join(f"`{c['id']}` ({c['type']})" for c in here[n]) or "—"
        other = ", ".join(f"`{c['id']}`" for c in touches[n]) or "—"
        out.append(f"| {n}. {step['title']} | {axes} | {own} | {other} |")
    out.append("")
    for step in steps:
        n = step["step"]
        out += [f"## {n}. {step['title']}", ""]
        for axis in step.get("axes") or []:
            out.append(f"- `{axis}` — {questions.get(axis)}")
        if step.get("axes"):
            out.append("")
        if step.get("note"):
            out += [step["note"], ""]
        if touches[n]:
            refs = ", ".join(f"`{c['id']}` (step {home[c['id']]})" for c in touches[n])
            out += [f"Also bears on this step: {refs}.", ""]
        if not here[n]:
            out += ["No card answers this step yet.", ""]
        for card in here[n]:
            out += render_card(card, evidence_by_id, shipped)
    return out


def render(decision_data, source_data, tuned_data, tuned_db_data=None, axes_data=None):
    if axes_data is None:
        raise ValueError("the decision page is ordered by the performance axes' development_order")
    tuned_db_data = tuned_db_data or {}
    cards, source_evidence, shipped = _checked(
        decision_data, source_data, tuned_data, tuned_db_data, axes_data)
    provenance = source_data.get("provenance") or {}
    evidence_by_id = {record["evidence_id"]: record for record in source_evidence}
    flydsl_groups = tuned_db_data.get("tuned_configs") or []
    selection = tuned_db_data.get("backend_selection") or []
    triton_groups = tuned_data.get("tuned_configs") or []
    cited = {ref for card in cards for ref in card["source_evidence"]}
    cited |= {ref for card in cards for s in card.get("solutions") or [] for ref in s["source_evidence"]}
    out = [
        "# GEMM development decision cards",
        "",
        ("This is the actionable layer. It does not dump regex matches and ask the reader to infer a "
         "recommendation. Every curated card states **when it applies, what to try, why, alternatives, "
         "evidence strength and limits**. The raw, reproducible source observations remain in "
         "[`gemm_source_evidence.md`](gemm_source_evidence.md)."),
        "",
        (f"Source baseline: AITER `{provenance.get('aiter_commit', '?')}` · "
         f"{len(cards)} curated decision card(s) citing {len(cited)} of {len(source_evidence)} "
         f"source-evidence records · {len(flydsl_groups)} FlyDSL shipped-seed groups · "
         f"{len(selection)} backend-selection groups · {len(triton_groups)} Triton seed groups "
         f"(on [`gemm_triton_seeds.md`](gemm_triton_seeds.md))."),
        "",
        "How to read this page:",
        "",
        ("- Steps follow the order a kernel is written in, from its math contract to validation. A "
         "card sits under the step of its first axis — the question it answers — and every other step "
         "its axes touch links back to it: choosing split-K in step 3 is listed again in step 8, where "
         "the partials are combined, and step 9, where the combine's host state is written."),
        ("- Within a step, constraint cards come first. Read every constraint card before costing a "
         "candidate: architecture gates are in step 2 and build legality in step 3."),
        ("- A card's *Same question in other implementations* block shows how Triton, Gluon, CK or ASM "
         "answer the same question in AITER, with their own source lines. Transfer the intent, not the "
         "spelling: use the target language's docs to write it."),
        ("- Evidence is `file:line`, or `file:line-end` when the cited construct spans lines; open the "
         "range to read the whole mechanism."),
        ("- The two shipped tables after the cards are what AITER's tuner selected — FlyDSL "
         "configurations per bucket, and which backend it chose per bucket. They seed a search; they "
         "never end one."),
        "",
        "Evidence levels:",
        "",
        "- `source_observed`: implementation precedent only; add a candidate and measure it.",
        ("- `shipped_config` / shipped selection: parameter seeds and backend choices recorded in AITER's "
         "shipped tables; alternatives and benchmark results are not attached, so vary and measure "
         "locally."),
        ("- Background references point at always-on docs under `perf_knowledge/` (hardware facts, "
         "how-to), never at measurements."),
        ("- Measured guidance is deliberately not copied here: it stays in learned cards/expert skills "
         "behind their existing feature switches."),
        "",
    ]
    out += render_traits(decision_data.get("traits") or {})
    out += render_arch_coverage(cards, source_evidence, tuned_data, tuned_db_data)
    out += render_steps(cards, axes_data, evidence_by_id, shipped)
    out += render_flydsl_seed_table(flydsl_groups)
    out += render_selection_table(selection, tuned_db_data.get("inventory") or [],
                                  tuned_db_data.get("missing") or [])
    out += [
        "## Other backends' shipped seeds",
        "",
        (f"The {len(triton_groups)} Triton configuration groups AITER ships under "
         "`aiter/ops/triton/configs/gemm` are rendered on [`gemm_triton_seeds.md`](gemm_triton_seeds.md). "
         "They are Triton knob names; open that page when the Triton baseline's own config is the "
         "question, not when choosing FlyDSL knobs."),
        "",
        "## Sources",
        "",
        "- Curated cards: [`decisions/gemm.yaml`](decisions/gemm.yaml).",
        "- Source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml).",
        ("- Shipped tuned databases (FlyDSL seeds and backend selection): "
         "[`evidence/gemm_csv_tuned_configs.yaml`](evidence/gemm_csv_tuned_configs.yaml)."),
        ("- Shipped Triton configs: "
         "[`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml)."),
        "- Performance axes: [`performance_axes/gemm.yaml`](performance_axes/gemm.yaml).",
        ("- Generated by [`_render_decisions.py`](_render_decisions.py); edit the cards or evidence, "
         "never this file."),
        "",
    ]
    return "\n".join(out)


def render_triton_page(decision_data, source_data, tuned_data, tuned_db_data=None, axes_data=None):
    _checked(decision_data, source_data, tuned_data, tuned_db_data or {}, axes_data)
    provenance = source_data.get("provenance") or {}
    out = [
        "# Shipped Triton GEMM configuration seeds",
        "",
        (f"AITER `{provenance.get('aiter_commit', '?')}`. Split out of "
         "[`gemm_decisions.md`](gemm_decisions.md), which carries the FlyDSL cards and FlyDSL seeds; "
         "these rows are Triton knob names."),
        "",
    ]
    out += render_triton_tuning_table(tuned_data.get("tuned_configs") or [])
    out += [
        "## Sources",
        "",
        ("- Shipped tuning evidence: "
         "[`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml)."),
        ("- Generated by [`_render_decisions.py`](_render_decisions.py); edit the evidence, never this "
         "file."),
        "",
    ]
    return "\n".join(out)


def load_inputs():
    return (
        source_renderer.load_yaml(DECISIONS),
        source_renderer.load_yaml(SOURCE_EVIDENCE),
        source_renderer.load_yaml(TUNED_EVIDENCE),
        source_renderer.load_yaml(TUNED_DB_EVIDENCE),
        source_renderer.load_yaml(PERFORMANCE_AXES),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true",
                        help="write gemm_decisions.md and gemm_triton_seeds.md")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if either generated page differs from its inputs")
    args = parser.parse_args()

    for path in (DECISIONS, SOURCE_EVIDENCE, TUNED_EVIDENCE, TUNED_DB_EVIDENCE, PERFORMANCE_AXES):
        if not os.path.isfile(path):
            print(f"missing decision input: {os.path.relpath(path, ROOT)}", file=sys.stderr)
            return 2
    try:
        inputs = load_inputs()
        pages = {DOC: render(*inputs), TRITON_DOC: render_triton_page(*inputs)}
    except ValueError as exc:
        print(f"invalid GEMM decision cards:\n{exc}", file=sys.stderr)
        return 1

    if args.check:
        stale = []
        for path, text in pages.items():
            old = ""
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as handle:
                    old = handle.read()
            if old != text:
                stale.append(os.path.relpath(path, ROOT))
        if stale:
            print(f"STALE: {', '.join(stale)}; re-run _render_decisions.py --emit", file=sys.stderr)
            return 1
        print("OK: gemm_decisions.md and gemm_triton_seeds.md match decisions and evidence")
        return 0
    if not args.emit:
        sys.stdout.write(pages[DOC])
        return 0
    for path, text in pages.items():
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"OK: {len(text.splitlines())} lines -> {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
