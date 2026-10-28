#!/usr/bin/env python3
"""Select conditioned corpus cards without turning source precedent into a verdict.

The selector is deliberately static.  It answers "which cards apply to this task?" from the
machine-readable `match` block; it never claims that a matching performance candidate is fast.
Measured outcome summaries may rank the returned candidates later, while constraints and the
target-box verifier remain separate gates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import yaml

import _corpus_contract as CONTRACT


HERE = Path(__file__).resolve().parent
# No bundled catalog constant: the corpus lives in the knowledge repository that
# `KERNEL_KNOWLEDGE_DIR` points at, and `CONTRACT.load_catalog(None)` resolves it there. A
# default pointing next to this file would resolve to a vendored copy and recreate the second
# source of truth the split removed.
TYPE_ORDER = {"constraint": 0, "performance_candidate": 1, "semantic": 2}
DTYPE_ALIASES = {
    "float16": "f16",
    "fp16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
}
# One fp8 format, three spellings in use: torch's `float8_e4m3fnuz`, the capability index's
# `fp8_e4m3_fnuz` and the task specs' `fp8_e4m3fnuz[_blockscale]`. Canonicalised, then matched
# exactly — the scale-contract suffix (`_blockscale`) is kept, because it is a different contract.
DTYPE_SPELLINGS = (
    (re.compile(r"^float8_"), "fp8_"),
    (re.compile(r"_(e4m3|e5m2)_fnuz(?=_|$)"), r"_\1fnuz"),
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    return data


def normalize_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    value = DTYPE_ALIASES.get(value, value)
    for pattern, replacement in DTYPE_SPELLINGS:
        value = pattern.sub(replacement, value)
    return value


def normalize_gfx(value: str | None) -> str | None:
    return str(value).split(":", 1)[0] if value is not None else None


DTYPE_TOKEN = re.compile(
    r"^(?:f16|bf16|f32|fp32|int8|uint8|int4|mxfp4|mxfp8|fp8_\w+|fp4_\w+)$"
)


def workload_dtype(meta: dict[str, Any] | None) -> str | None:
    """The operand dtype a task `meta.json` states.

    Most tasks spell it canonically (`bf16`, `fp8_e4m3fnuz_blockscale`). Others describe the whole
    contract (`bf16 in / bf16 weight / bf16 out, fp32 accumulate`); the operand dtype is then the one
    dtype named outside the accumulator. Several different ones is a mixed-precision contract that
    no single card dtype describes, so the answer is None rather than a guess.
    """
    raw = (meta or {}).get("dtype")
    if raw is None:
        return None
    canonical = normalize_dtype(raw)
    if DTYPE_TOKEN.match(canonical):
        return canonical
    tokens = re.findall(r"[a-z0-9_]+", str(raw).lower())
    named = {
        normalize_dtype(token)
        for index, token in enumerate(tokens)
        if DTYPE_TOKEN.match(normalize_dtype(token))
        and not (index + 1 < len(tokens) and tokens[index + 1].startswith("acc"))
    }
    return named.pop() if len(named) == 1 else None


def resolve_family(catalog: dict[str, Any], requested: str) -> dict[str, Any]:
    matches = []
    for family in catalog.get("families") or []:
        names = {str(family.get("id") or "")}
        names.update(str(value) for value in family.get("patterns") or [])
        if requested in names:
            matches.append(family)
    if not matches:
        raise ValueError(f"no corpus family publishes operator/pattern {requested!r}")
    if len(matches) != 1:
        raise ValueError(f"operator/pattern {requested!r} is ambiguous across corpus families")
    return matches[0]


def _dimension(match: dict[str, Any], field: str, value: str | None) -> tuple[str, str | None]:
    allowed = match.get(field) or []
    if not allowed:
        return "eligible", None
    if value is None:
        return "deferred", f"missing {field}"
    if field == "regimes" and str(value) == "both":
        # A `both` workload contains decode AND prefill cases. A card restricted to one of them applies
        # to some of those cases, so it is not rejected here; per-case shape bounds (or a per-case
        # query via --workload) decide which cases it covers.
        if {"decode", "prefill", "both"} & {str(item) for item in allowed}:
            return "eligible", None
        return "rejected", f"{field}={value!r} not in {sorted(allowed)!r}"
    normalized = normalize_dtype(value) if field == "dtypes" else (
        normalize_gfx(value) if field == "gfx" else value
    )
    expected = {
        normalize_dtype(item) if field == "dtypes"
        else normalize_gfx(item) if field == "gfx"
        else str(item)
        for item in allowed
    }
    if normalized not in expected:
        return "rejected", f"{field}={value!r} not in {sorted(expected)!r}"
    return "eligible", None


def classify_card(card: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[str]]:
    """Return eligible/deferred/rejected plus explicit reasons."""
    match = card.get("match") or {}
    reasons: list[str] = []

    dimensions = (
        ("operator_families", context.get("operator_family")),
        ("target_languages", context.get("target_language")),
        ("gfx", context.get("gfx")),
        ("dtypes", context.get("dtype")),
        ("regimes", context.get("regime")),
        ("bottlenecks", context.get("bottleneck")),
    )
    statuses = []
    for field, value in dimensions:
        status, reason = _dimension(match, field, value)
        statuses.append(status)
        if reason:
            reasons.append(reason)

    gfx = normalize_gfx(context.get("gfx"))
    excluded_gfx = {normalize_gfx(value) for value in match.get("exclude_gfx") or []}
    if excluded_gfx and gfx is None:
        statuses.append("deferred")
        reasons.append("missing gfx required to evaluate exclusion")
    elif gfx in excluded_gfx:
        statuses.append("rejected")
        reasons.append(f"gfx={gfx!r} is explicitly excluded")

    required = set(match.get("requires") or [])
    available = set(context.get("traits") or [])
    missing_traits = sorted(required - available)
    if missing_traits:
        statuses.append("deferred")
        reasons.append(f"missing required traits {missing_traits!r}")

    shape = match.get("shape") or {}
    for axis in ("m", "n", "k"):
        low, high = shape.get(f"{axis}_min"), shape.get(f"{axis}_max")
        if low is None and high is None:
            continue
        value = context.get(axis)
        if value is None:
            statuses.append("deferred")
            reasons.append(f"missing shape axis {axis}")
            continue
        if low is not None and value < low:
            statuses.append("rejected")
            reasons.append(f"{axis}={value} is below {low}")
        if high is not None and value > high:
            statuses.append("rejected")
            reasons.append(f"{axis}={value} exceeds {high}")

    if "rejected" in statuses:
        return "rejected", reasons
    if "deferred" in statuses:
        return "deferred", reasons
    return "eligible", reasons


def _op_spec_value(op_spec: dict[str, Any], key: str):
    # Seam op_specs spell the fixed weight dims `N`/`K` and list every timed `M` of the run.
    for name in (key, key.upper()):
        if name in op_spec:
            return op_spec[name]
    shapes = op_spec.get("shapes")
    if isinstance(shapes, dict):
        for name in (key, key.upper()):
            if name in shapes:
                return shapes[name]
    return None


def _regime_word(value: Any) -> str:
    """`decode`, `prefill` or `both` out of a regime field that may carry prose after it."""
    text = str(value).strip().lower()
    found = {word for word in ("decode", "prefill") if word in text}
    if "both" in text or found == {"decode", "prefill"}:
        return "both"
    return found.pop() if found else text


def outcome_context_matches(measured: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Require the durable environment keys and reject any stated task mismatch."""
    if measured.get("observed_language") != requested.get("target_language"):
        return False
    if requested.get("gfx") and normalize_gfx(measured.get("gfx")) != normalize_gfx(requested.get("gfx")):
        return False
    for requested_field, measured_field in (
        ("flydsl_version", "flydsl_version"),
        ("provider", "provider"),
        ("metric_kind", "metric_kind"),
        ("aiter_commit", "aiter_commit"),
        ("rocm_version", "rocm_version"),
        ("protocol_fingerprint", "protocol_fingerprint"),
    ):
        value = requested.get(requested_field)
        if value is not None and measured.get(measured_field) != value:
            return False
    op_spec = measured.get("op_spec")
    if not isinstance(op_spec, dict):
        op_spec = {}
    for key in ("dtype", "regime", "m", "n", "k"):
        want = requested.get(key)
        got = _op_spec_value(op_spec, key)
        if want is None:
            continue
        if got is None:
            return False
        if key == "dtype":
            if normalize_dtype(str(got)) != normalize_dtype(str(want)):
                return False
        elif key == "regime":
            if _regime_word(got) != _regime_word(want):
                return False
        elif isinstance(got, (list, tuple)):
            # One run measured several shapes; it speaks to a requested shape it contains.
            if str(want) not in {str(value) for value in got}:
                return False
        elif str(got) != str(want):
            return False
    return True


def measured_priors(outcomes: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build safe priors from single-ref directions only; bundles never rank one card."""
    values: dict[str, list[float]] = {}
    absolute_values: dict[str, list[float]] = {}
    runs: dict[str, set[str]] = {}
    for row in outcomes.get("single_ref_direction_results") or []:
        decisions = row.get("decisions") or []
        if (
            len(decisions) != 1
            or row.get("attribution_scope") != "single_ref_direction"
            or row.get("quality_eligible_for_ranking") is not True
            or not outcome_context_matches(row.get("context") or {}, context)
        ):
            continue
        incremental = row.get("incremental_vs_incumbent")
        if (
            isinstance(incremental, bool)
            or not isinstance(incremental, (int, float))
            or not math.isfinite(float(incremental))
            or incremental <= 0
        ):
            continue
        decision = str(decisions[0])
        values.setdefault(decision, []).append(float(incremental))
        absolute = row.get("speedup_vs_frozen_baseline")
        if (
            not isinstance(absolute, bool)
            and isinstance(absolute, (int, float))
            and math.isfinite(float(absolute))
            and absolute > 0
        ):
            absolute_values.setdefault(decision, []).append(float(absolute))
        runs.setdefault(decision, set()).add(str(row.get("run") or ""))
    priors = {}
    for decision, samples in values.items():
        count = len(samples)
        priors[decision] = {
            "metric": "incremental_vs_incumbent",
            "basis": "single_ref_direction; planner attribution, not causal proof",
            "count": count,
            "unique_runs": len(runs[decision]),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "confidence": "high" if count >= 5 else "medium" if count >= 3 else "low",
        }
        if absolute_values.get(decision):
            priors[decision]["absolute_median_vs_frozen_baseline"] = statistics.median(
                absolute_values[decision]
            )
    return priors


def compatible_bundles(outcomes: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in outcomes.get("bundled_direction_results") or []:
        if (
            row.get("quality_eligible_for_ranking") is True
            and outcome_context_matches(row.get("context") or {}, context)
        ):
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("incremental_vs_incumbent") or 0),
            tuple(row.get("decisions") or []),
            str(row.get("run") or ""),
        ),
    )


def select(
    catalog_path: Path,
    context: dict[str, Any],
    outcomes_path: Path | None = None,
) -> dict[str, Any]:
    catalog_path, catalog = CONTRACT.load_catalog(catalog_path)
    family = resolve_family(catalog, context["operator_family"])
    # Declared by the catalog, not imported as a sibling: this repo consumes the knowledge
    # repo's interface, and the normalizer is part of that interface.
    PERFORMANCE = CONTRACT.load_declared_module(catalog_path, family, "performance_normalizer")
    decisions_path = catalog_path.parent / family["decisions"]
    decision_data = load_yaml(decisions_path)
    axes_path = (
        catalog_path.parent / family["performance_axes"]
        if family.get("performance_axes")
        else None
    )
    performance_model = (
        PERFORMANCE.model_summary(load_yaml(axes_path))
        if axes_path is not None
        else {"axes": [], "comparison_context": {}, "dependencies": [], "contract": {}}
    )

    buckets: dict[str, list[dict[str, Any]]] = {
        "eligible": [],
        "deferred": [],
        "rejected": [],
    }
    outcomes = load_yaml(outcomes_path) if outcomes_path else {}
    priors = measured_priors(outcomes, context) if outcomes else {}
    for card in decision_data.get("cards") or []:
        status, reasons = classify_card(card, context)
        row = {
            "id": card.get("id"),
            "type": card.get("type"),
            "question": card.get("question"),
            "evidence_level": card.get("evidence_level"),
            "status": card.get("status"),
            "reasons": reasons,
        }
        if str(card.get("id")) in priors:
            row["measured_prior"] = priors[str(card.get("id"))]
        buckets[status].append(row)
    def rank(row):
        prior = row.get("measured_prior") or {}
        median = float(prior.get("median") or 0)
        measured = 0 if prior and median > 1.0 else 2 if prior else 1
        gain = -median
        support = -int(prior.get("count") or 0)
        absolute = -float(prior.get("absolute_median_vs_frozen_baseline") or 0)
        return TYPE_ORDER.get(row["type"], 99), measured, gain, support, absolute, str(row["id"])

    for values in buckets.values():
        values.sort(key=rank)

    return {
        "schema_version": 1,
        "family": family["id"],
        "context": context,
        "performance_axes": performance_model["axes"],
        "performance_model": performance_model,
        "measured_bundles": compatible_bundles(outcomes, context) if outcomes else [],
        **buckets,
        # A misspelt trait defers every card that needs it, silently; name it instead.
        "undefined_traits": sorted(
            set(context.get("traits") or []) - set(decision_data.get("traits") or {})
        ),
        "contract": (
            "Static match selects applicable cards. Compatible single-ref outcomes may rank "
            "performance candidates but are planner attribution, never a target-box verdict. "
            "Shared performance axes align backend questions and spelling only; they do not make "
            "unlike representations equivalent. Constraints alone may reject impossible candidates."
        ),
    }


SHAPE_TEXT = re.compile(r"^\s*(\d+)\s*[x,]\s*(\d+)\s*[x,]\s*(\d+)\s*$")


def _case_regime(signature: str) -> str | None:
    text = signature.lower()
    if "decode" in text:
        return "decode"
    if "prefill" in text:
        return "prefill"
    return None


def workload_cases(meta: dict[str, Any] | None, shapes: list[str] | None = None) -> list[dict[str, Any]]:
    """The (M, N, K) cases of a task `meta.json` plus any `--shape MxNxK`, in order.

    Two meta layouts are in use. Kernel-task metas give every case `sig`, `m`, `n`, `k`. Seam metas
    extracted from a served model give each case `name`, `m` and `regime`, keep the fixed weight
    dims once in `model_dims` (`N`, `K`), and mark correctness-only edges `timed: false`.
    A case's regime is its explicit `regime`, else read from its signature only when the signature
    says so (`decode_m8`); nothing is inferred from M, because where decode ends is exactly what
    the cards disagree on.
    """
    cases = []
    model_dims = (meta or {}).get("model_dims") or {}
    for case in (meta or {}).get("cases") or []:
        dims = [case.get(axis, model_dims.get(axis.upper(), model_dims.get(axis)))
                for axis in ("m", "n", "k")]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in dims):
            continue
        signature = str(case.get("sig") or case.get("name") or "m{}_n{}_k{}".format(*dims))
        regime = case.get("regime")
        cases.append({"sig": signature, "m": dims[0], "n": dims[1], "k": dims[2],
                      "regime": regime if regime in ("decode", "prefill") else _case_regime(signature),
                      "timed": case.get("timed") is not False})
    for text in shapes or []:
        parsed = SHAPE_TEXT.match(str(text))
        if not parsed:
            raise ValueError(f"--shape expects MxNxK, got {text!r}")
        m, n, k = (int(v) for v in parsed.groups())
        cases.append({"sig": f"m{m}_n{n}_k{k}", "m": m, "n": n, "k": k, "regime": None,
                      "timed": True})
    return cases


def select_workload(
    catalog_path: Path,
    context: dict[str, Any],
    cases: list[dict[str, Any]],
    outcomes_path: Path | None = None,
) -> dict[str, Any]:
    """Classify every card for every case of one workload.

    Batch optimisation across shapes is the reason this exists: one static answer for a workload that
    spans decode and prefill is either too narrow for one case or too wide for the other. The result
    keeps each case's own eligible/deferred/rejected split and states which cards hold for every case
    and which only for some.
    """
    if not cases:
        raise ValueError("a workload needs at least one (M, N, K) case")
    per_case = []
    base = None
    for case in cases:
        ctx = dict(context, m=case["m"], n=case["n"], k=case["k"])
        if case.get("regime") and context.get("regime") in (None, "both"):
            ctx["regime"] = case["regime"]
        result = select(catalog_path, ctx, outcomes_path)
        base = base or result
        per_case.append({
            "sig": case["sig"], "m": case["m"], "n": case["n"], "k": case["k"],
            "regime": ctx.get("regime"), "timed": case.get("timed", True),
            "eligible": [row["id"] for row in result["eligible"]],
            "deferred": [{"id": row["id"], "reasons": row["reasons"]} for row in result["deferred"]],
            "rejected": [{"id": row["id"], "reasons": row["reasons"]} for row in result["rejected"]],
            "measured_priors": {
                row["id"]: row["measured_prior"]
                for bucket in ("eligible", "deferred", "rejected") for row in result[bucket]
                if "measured_prior" in row
            },
        })
    # A correctness-only edge (M=1 on a decode task) still gets its own classification, but it is
    # not part of the timed workload: a card that fails only there still holds for every timed case.
    timed = [case for case in per_case if case["timed"]] or per_case
    every = set(timed[0]["eligible"])
    for case in timed[1:]:
        every &= set(case["eligible"])
    some: dict[str, list[str]] = {}
    for case in timed:
        for card_id in case["eligible"]:
            if card_id not in every:
                some.setdefault(card_id, []).append(case["sig"])
    return {
        "schema_version": 1,
        "family": base["family"],
        "context": context,
        "performance_axes": base["performance_axes"],
        "performance_model": base["performance_model"],
        "measured_bundles": base["measured_bundles"],
        "cases": per_case,
        "eligible_in_every_case": sorted(every),
        "eligible_in_some_cases": dict(sorted(some.items())),
        "contract": base["contract"] + (
            " A workload query classifies each case separately; a card eligible for one case says "
            "nothing about the others."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--operator-family", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--gfx")
    parser.add_argument("--flydsl-version")
    parser.add_argument("--provider")
    parser.add_argument("--metric-kind", choices=("geomean", "time_weighted"))
    parser.add_argument("--aiter-commit")
    parser.add_argument("--rocm-version")
    parser.add_argument("--protocol-fingerprint")
    parser.add_argument("--dtype")
    parser.add_argument("--regime")
    parser.add_argument("--bottleneck")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--outcomes", type=Path,
                        help="optional output of _aggregate_decision_outcomes.py")
    parser.add_argument("--workload", type=Path,
                        help="a task meta.json; classify cards separately for each of its cases")
    parser.add_argument("--shape", action="append", default=[],
                        help="an extra MxNxK case (repeatable); implies a workload query")
    args = parser.parse_args()
    meta = None
    if args.workload:
        meta = json.loads(args.workload.read_text(encoding="utf-8"))
    context = {
        "operator_family": args.operator_family,
        "target_language": args.target_language,
        "gfx": args.gfx,
        "flydsl_version": args.flydsl_version,
        "provider": args.provider,
        "metric_kind": args.metric_kind,
        "aiter_commit": args.aiter_commit,
        "rocm_version": args.rocm_version,
        "protocol_fingerprint": args.protocol_fingerprint,
        "dtype": normalize_dtype(args.dtype) if args.dtype else workload_dtype(meta),
        "regime": args.regime,
        "bottleneck": args.bottleneck,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "traits": sorted(set(args.trait)),
    }
    if meta is not None or args.shape:
        result = select_workload(args.catalog, context, workload_cases(meta, args.shape),
                                 args.outcomes)
    else:
        result = select(args.catalog, context, args.outcomes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
