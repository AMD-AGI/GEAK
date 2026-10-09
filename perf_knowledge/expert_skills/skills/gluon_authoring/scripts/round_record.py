#!/usr/bin/env python3
"""Append and verify portable round-ledger records.

The writer validates enum, unit, comparator, and provenance fields without deciding
whether a search direction is worthwhile.  It remains usable by tool-first packs:
older latency rows receive a canonical compatibility measurement with
``source=legacy_latency`` and ``scope=unknown``.  They never assert device timing
until boundary and observed sampling provenance are supplied.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import uuid

from canonical_record import legacy_measurement, validate_measurement

SCHEMA_VERSION = 3
JOURNAL_SCHEMA_VERSION = 4
JOURNAL_SCHEMA = "kernel_opt.optimization_journal/1"
RECALL_RECEIPT_SCHEMA = "kernel_opt.recall_receipt/1"
RECALL_EVENT_SCHEMA = "kernel_opt.recall_event/1"
MEASUREMENT_EVENT_SCHEMA = "kernel_opt.measurement_event/1"
DEFAULT_JOURNAL = "optimization_journal.jsonl"
DEFAULT_MEASUREMENT_EVENTS = ".kod/measurement_events.jsonl"
DEFAULT_RECALL_EVENTS = ".kod/recall_events.jsonl"
ROUND_SUMMARY_SCHEMA = "kernel_opt.round_summary/1"
DECISION_LOG_REF_SCHEMA = "kernel_opt.decision_log_ref/1"
SUMMARY_BUCKETS = ("valid", "stale", "unfinished", "recheck")
SUMMARY_DEFAULT_LIMIT = 8
RECALL_RECEIPT_MAX_BYTES = 32 * 1024
WORK_KINDS = ("sweep_batch", "branch_arm", "climb_round")
IDENTITY_KINDS = (
    "body", "layout", "comparator", "measurement", "environment", "toolchain",
    "skill_build", "policy",
)
RECALL_TRIGGERS = ("new_climb_hypothesis", "identity_change", "branch_reentry", "manual")
JOURNAL_DISPOSITIONS = ("keep", "revert", "defer", "inconclusive", "blocked")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# ---------------------------------------------------------------- the closed vocabularies

# Four states, and every one of the nine spellings seen in real ledgers maps onto one of them. The
# split that made nine necessary was real -- "kept and confirmed" IS different from "kept and still
# provisional" -- but it is a SECOND axis, and folding it into the first is what produced a
# vocabulary where a query cannot ask either question. `confidence` carries it now.
VERDICTS = ("kept", "reverted", "null", "blocked")

# Orthogonal to verdict. `candidate` on a `kept` row is the old "win-candidate"; `confirmed` on a
# `kept` row is the old "win-confirmed"; absent means the round did not say.
CONFIDENCE = ("confirmed", "candidate")
CORRECTNESS = ("ok", "failed", "unknown", "recheck")

# What the round's number was measured AGAINST. Required, because without it a metric is not a
# measurement -- one ledger's `geomean` was an in-batch control geomean wearing a name that reads
# like a champion speed, and two rounds quoting it were never comparable.
COMPARATORS = (
    "in_batch_control",   # a control arm inside THIS timing window. Not comparable across windows.
    "pinned_champion",    # the pinned champion of this direction, same window.
    "golden",             # the frozen baseline the campaign scores against.
    "prev_round",         # the immediately preceding kept state.
    "arm_parent",         # the trunk this arm branched from.
    "isolated_path",      # a mechanism measured out of situ (owes an exposure ratio -- see P10).
)

# A metric field names its own unit, and exactly one may be set. This is the whole fix for the
# milliseconds-in-a-ratio-field case: there is no field that can hold both.
METRIC_FIELDS = {
    "speedup_vs_comparator": "ratio",   # >0, dimensionless. 1.0 == parity with `comparator`.
    "latency_ms": "ms",                 # >0, milliseconds.
}

# Version 3 deliberately contains no prose fields.  A round's explanation belongs
# in an external decision document that can reference its round and summary
# artifact, while this ledger stays queryable and bounded.  Older records retain
# their original schema and remain readable/validatable below.
STRUCTURED_FIELDS = frozenset({
    "schema_version", "round", "measured_at", "comparator", "speedup_vs_comparator",
    "latency_ms", "noise_band_pct", "correctness", "verdict", "confidence",
    "enabling_step", "evidence_layers", "artifacts", "body_sha", "layout_sig",
    "goal", "mode", "arm_id", "measurement", "lever", "summary_ref",
})
JOURNAL_FIELDS = frozenset({
    "schema", "schema_version", "record_id", "sequence", "previous_entry_sha256",
    "entry_sha256", "work_kind", "work_id", "round", "run_id", "generation", "stage",
    "role", "recorded_at", "identity", "hypothesis", "alternatives", "change_ref",
    "comparison", "measurement_summary", "oracle", "disposition", "evidence_refs",
    "invalidations", "next_action", "recall_receipt_ref", "recall_exemption",
    "measurement_event_id",
})

# The lifecycle has three active search modes.  The retired spellings remain readable below so an
# older ledger does not become invalid merely because its work is now classified by change scope.
SEARCH_MODES = ("sweep", "branch", "climb")
LEGACY_SEARCH_MODES = ("resweep", "layout", "launcher", "ir-override")

# Round kinds that are NOT search modes, and the distinction is load-bearing: a `verify` round is
# not a class of move you can exhaust, so it holds no state in the mode ledger. They are in the enum
# because the campaign needed all three -- diagnose 11, verify 11, converge 2 of 70 rounds -- and
# refusing them would only have produced a fourth spelling. `layout`, meanwhile, was entered ZERO
# times in those same 70 rounds while the census carried a 25%-priced `input_layout` deferral.
ROUND_KINDS = ("diagnose", "verify", "converge")

MODES = SEARCH_MODES + LEGACY_SEARCH_MODES + ROUND_KINDS

# Evidence layers, as TOKENS. This field's value space decides whether the mode-switch trigger works
# at all: it counts a layer lit for the FIRST time as the round having built something even when the
# round won nothing. Free text breaks that silently -- 7 rounds wrote sentences here
# ("A1 next_free_vgpr 370->400", "D interleaved A/B with byte-identical control, 12 cells"), and a
# per-round-unique string makes EVERY such round read as productive. So prose is refused. A token
# OUTSIDE this list is allowed and reported by `verify`, because the failure to catch is accidental
# rather than adversarial, and a campaign may genuinely read a layer nobody anticipated.
EVIDENCE_LAYERS = (
    "isa_census",           # A0, seconds, no PMC. The one layer a measured campaign adopted 8/8.
    "occupancy",            # A. resident waves, VGPR/AGPR, spill, LDS per WG.
    "pmc_valu",             # B. issue and bubble buckets.
    "pmc_tcc",              # C. the memory side -- L2 hits, DRAM bytes.
    "rocprof_compute_sol",  # the SOL table.
    "att",                  # the instruction-level timeline.
    "ab_interleaved",       # D1. same-window interleaved A/B against a MEASURED control band.
    "oracle",               # D2. in-process correctness, gating before timing.
    "roofline",             # the analytic budget, read as a ranking tool and not as a gate.
    "floor_probe",          # a ceiling probed at THIS program count rather than extrapolated.
    "compile_probe",        # zero-GPU: metadata.shared, probe.py plan, OutOfResources.
    "ttgir_facts",          # the IR-level read.
    "sweep_envelope",       # a config certificate whose pin_blockers are closed.
    "grid_census",          # the launch geometry against the CU count.
    # The two promoted from a measured run's `unregistered_layer_tokens`. Both are real instrument
    # reads rather than experiment names, and both are distinct from `grid_census`: that one is
    # launch geometry against the CU count, while `worktile_census` is the per-program work
    # distribution a makespan argument needs (P1, load-imbalanced grids) and `xcd_scan` is which XCD
    # a program lands on, which is a CDNA3/4 placement fact no other layer here reports.
    "worktile_census",
    "xcd_scan",
)

_LAYER_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Which refusals say a row DID NOT COME THROUGH THE WRITER, and which say the record is the wrong
# SHAPE. The split exists only for `verify`, and it earns its keep there: the integrity refusals have
# been in `append` from the beginning, so a row on disk that breaks one of them was hand-written,
# while the three field vocabularies below were added later and an older ledger breaking them is a
# record to reconcile rather than a forgery. Lumping the two buries the forgery under the
# reconciliation -- measured on the ledger that motivated this change, 25 of 26 rows break the new
# goal bound and 7 break the confidence enum, and it is the 7 that mean something.
INTEGRITY_RULES = ("verdict_enum", "confidence_enum", "comparator_enum", "prediction_missing",
                   "two_units", "metric_missing", "metric_not_positive", "kept_without_band",
                   "kept_on_latency", "kept_inside_band", "structured_record_extra_fields",
                   "artifact_refs", "lever_shape", "summary_ref", "correctness_enum")

# `goal` is a KEY, not a description, and the difference is measurable. Written as one-sentence task
# descriptions it produced 60 distinct goals over 70 rounds with only 4 repeated -- and `recall
# --goal` matches exactly, so long values undermine the index this field exists to build. The
# reasoning belongs in --hypothesis, which is unbounded.
GOAL_MAX_CHARS = 80


def goal_key(goal) -> str:
    """The normalised form two rows must agree on for `recall --goal` to join them."""
    return re.sub(r"\s+", " ", str(goal or "").strip().lower())


def near_goal_groups(goals, threshold=0.6) -> list:
    """Goals that are ONE goal under several spellings, clustered by word overlap.

    A prefix rule is not sufficient: length-bounded goals can diverge inside any useful prefix.
    Word overlap joins descriptions of the same objective with different trailing detail while
    keeping separate objectives that merely share an opening."""
    uniq = sorted({goal_key(g) for g in goals if g})
    words = {g: set(re.findall(r"[a-z0-9_]+", g)) for g in uniq}
    groups, claimed = [], set()
    for i, a in enumerate(uniq):
        if a in claimed:
            continue
        cluster = [a]
        for b in uniq[i + 1:]:
            if b in claimed or not words[a] or not words[b]:
                continue
            if len(words[a] & words[b]) / len(words[a] | words[b]) >= threshold:
                cluster.append(b)
        if len(cluster) > 1:
            claimed.update(cluster)
            groups.append(cluster)
    return groups


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_hash(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _entry_hash(record: dict) -> str:
    payload = dict(record)
    payload.pop("entry_sha256", None)
    return _canonical_hash(payload)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_text(value, label: str, maximum: int, *, required: bool = False):
    if value is None:
        if required:
            raise Refused(f"{label} is required")
        return None
    text = str(value).strip()
    if required and not text:
        raise Refused(f"{label} is required")
    if "\n" in text or "\r" in text or len(text) > maximum:
        raise Refused(f"{label} must be one line of at most {maximum} characters")
    return text or None


def _full_hash(value, label: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if not _FULL_SHA256.fullmatch(digest):
        raise Refused(f"{label} must be a complete lowercase SHA-256")
    return digest


def _parse_named_hashes(values) -> dict[str, str]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise Refused("--identity-hash must use KIND=SHA256")
        kind, digest = item.split("=", 1)
        if kind not in IDENTITY_KINDS:
            raise Refused(f"identity kind {kind!r} is outside {IDENTITY_KINDS}")
        if kind in out:
            raise Refused(f"identity kind {kind!r} was supplied twice")
        out[kind] = _full_hash(digest, f"identity.{kind}")
    return out


def _hash_files(paths) -> str | None:
    paths = [os.path.abspath(path) for path in (paths or []) if os.path.isfile(path)]
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _layout_hash(specs) -> str | None:
    if not specs:
        return None
    parts = []
    for spec in specs:
        fields = spec.split(":")
        if len(fields) != 3:
            raise Refused(f"--tensor wants name:dtype:stride_pattern, got {spec!r}")
        parts.append(":".join(field.strip() for field in fields))
    return hashlib.sha256("|".join(sorted(parts)).encode("utf-8")).hexdigest()


class Refused(Exception):
    """A record this tool will not append. The message says which of the four, and why."""


# ---------------------------------------------------------------- provenance the writer fills

def body_sha(paths) -> str | None:
    """sha256 over the sources whose bytes this round's number describes.

    A verdict is a measurement of one lever against ONE BODY. Without this field nothing can tell a
    verdict measured on the current kernel from one measured on a kernel that no longer exists, and
    in one recorded run a single launcher-side transpose inverted six config rejections at once
    -- three of which were one decision, not three."""
    paths = [p for p in (paths or []) if os.path.isfile(p)]
    if not paths:
        return None
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.relpath(p).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def parse_layout_sig(specs) -> str | None:
    """`name:dtype:stride_pattern` per tensor -> one stable signature.

    Separate from `body_sha` because it changes WITHOUT the kernel body changing: transposing an
    input in the launcher leaves every byte of the kernel alone and invalidates every config verdict
    measured against the old stride. That is the most expensive miss on record here --
    116 rounds of climbing never questioned the layout, and the round that finally did inverted six
    rejections and was worth +11.6%."""
    if not specs:
        return None
    parts = []
    for s in specs:
        f = s.split(":")
        if len(f) != 3:
            raise Refused(f"--tensor wants name:dtype:stride_pattern, got {s!r}. "
                          f"stride_pattern is e.g. 'row_major' / 'col_major' / 'contig_k'")
        parts.append(":".join(x.strip() for x in f))
    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- the four refusals

def _v(rule, detail) -> dict:
    return {"rule": rule, "detail": detail}


def _metric_violations(rec) -> list:
    """Refusals 2 and 4: two units in one field, and a `kept` its own number does not support."""
    verdict = rec.get("verdict")
    noise_band = rec.get("noise_band_pct")
    set_metrics = [k for k in METRIC_FIELDS if rec.get(k) is not None]
    if len(set_metrics) > 1:
        return [_v("two_units",
                   f"two metric fields set ({', '.join(sorted(set_metrics))}). A field carries "
                   f"ONE unit. One ledger put milliseconds into a ratio field for seven rounds "
                   f"and the ledger then displayed a 4.7x jump at a round whose own evidence "
                   f"called it a repair rather than a speedup")]
    if not set_metrics:
        if verdict in ("kept", "reverted"):
            return [_v("metric_missing",
                       f"verdict={verdict!r} with no metric. A kept or reverted round is "
                       f"a claim about a number; pass --speedup or --latency-ms. Use "
                       f"--verdict null for a round that refuted a hypothesis without one")]
        return []
    metric = set_metrics[0]
    val = rec[metric]
    if not isinstance(val, (int, float)) or val <= 0 or val != val:
        return [_v("metric_not_positive", f"{metric}={val!r} is not a positive finite number")]

    if verdict != "kept":
        return []
    # Refusal 4. `kept` asserts the number moved; the noise band says what "moved" costs here.
    if noise_band is None:
        return [_v("kept_without_band",
                   "verdict=kept without --noise-band. The band is what makes a delta a claim "
                   "rather than a reading, and it is measured per window -- from the control "
                   "arm's own spread (ab_bench --control), not from a constant. Measured on "
                   "one 1208-round campaign the MEDIAN win was +0.833% against a band of the same "
                   "order, and 31.6% of rounds marked a win recorded a non-positive delta")]
    if metric != "speedup_vs_comparator":
        return [_v("kept_on_latency",
                   "verdict=kept on latency_ms alone. A latency is not a comparison: pass "
                   "--speedup against a named --comparator, or record this as --verdict null")]
    delta_pct = (val - 1.0) * 100.0
    if delta_pct > noise_band:
        return []
    if rec.get("enabling_step"):
        return []  # provisionally kept toward a coupled combination; §Layer Backbone allows this.
    return [_v(
        "kept_inside_band",
        f"verdict=kept at {delta_pct:+.3f}% against a declared noise band of {noise_band:.3f}%. "
        f"This is the round the ledger cannot distinguish from a reading, and it is 31.6% of the "
        f"wins in one measured campaign. Three honest dispositions: --verdict null (it did not "
        f"resolve), --enabling-step (provisionally kept toward a coupled combination whose net you will "
        f"confirm against the checkpoint), or widen the window until the delta clears the band")]


def _journal_violations(rec) -> list:
    out = []
    unexpected = sorted(set(rec) - JOURNAL_FIELDS)
    if unexpected:
        out.append(_v("journal_extra_fields", f"journal record has unexpected fields: {unexpected}"))
    if rec.get("schema") != JOURNAL_SCHEMA:
        out.append(_v("journal_schema", f"journal schema must be {JOURNAL_SCHEMA!r}"))
    if rec.get("work_kind") not in WORK_KINDS:
        out.append(_v("work_kind", f"work_kind must be one of {WORK_KINDS}"))
    if not isinstance(rec.get("sequence"), int) or isinstance(rec.get("sequence"), bool) \
            or rec["sequence"] < 1:
        out.append(_v("sequence", "journal sequence must be a positive integer"))
    identity = rec.get("identity")
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY_KINDS):
        out.append(_v("identity_shape", f"identity must contain exactly {IDENTITY_KINDS}"))
    else:
        for kind, digest in identity.items():
            if digest is not None and (not isinstance(digest, str) or not _FULL_SHA256.fullmatch(digest)):
                out.append(_v("identity_hash", f"identity.{kind} must be null or a full SHA-256"))
    if not isinstance(rec.get("hypothesis"), str) or not rec["hypothesis"].strip():
        out.append(_v("hypothesis", "journal hypothesis must be non-empty"))
    alternatives = rec.get("alternatives")
    if (not isinstance(alternatives, list) or len(alternatives) > 8
            or any(not isinstance(item, str) or not item.strip() or len(item) > 160
                   for item in alternatives)):
        out.append(_v("alternatives", "alternatives must be at most eight non-empty one-line summaries"))
    comparison = rec.get("comparison")
    if not isinstance(comparison, dict) or not isinstance(comparison.get("comparable"), bool):
        out.append(_v("comparison", "comparison must declare comparable as a boolean"))
    elif comparison["comparable"]:
        for field in ("improvement", "uncertainty"):
            value = comparison.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
                out.append(_v("comparison", f"comparable comparison needs finite {field}"))
        if isinstance(comparison.get("uncertainty"), (int, float)) \
                and comparison["uncertainty"] < 0:
            out.append(_v("comparison", "comparison uncertainty must be non-negative"))
    measurement = rec.get("measurement_summary")
    if measurement is not None:
        if not isinstance(measurement, dict) or measurement.get("valid") is not True:
            out.append(_v("measurement_summary", "present measurement_summary must be valid"))
    oracle = rec.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("status") not in CORRECTNESS:
        out.append(_v("oracle", f"oracle.status must be one of {CORRECTNESS}"))
    disposition = rec.get("disposition")
    if not isinstance(disposition, dict) or disposition.get("outcome") not in JOURNAL_DISPOSITIONS:
        out.append(_v("disposition", f"disposition.outcome must be one of {JOURNAL_DISPOSITIONS}"))
    refs = rec.get("evidence_refs")
    if (not isinstance(refs, list) or len(refs) > 16
            or any(not isinstance(ref, str) or not ref.strip() or len(ref) > 256
                   or "\n" in ref or "\r" in ref for ref in refs)):
        out.append(_v("evidence_refs", "evidence_refs must be at most 16 bounded one-line references"))
    if rec.get("entry_sha256") != _entry_hash(rec):
        out.append(_v("entry_hash", "journal entry_sha256 does not match its canonical payload"))
    return out


def row_violations(rec) -> list:
    """Every append-time refusal, evaluated against a ROW instead of against argparse.

    ONE function, TWO callers, and that is the repair. `build()` raises on the first violation and
    `verify()` collects them over the rows already on disk. Those used to be different code, so a
    row that did not come through `build()` was checked by nothing that `build()` checks -- and the
    measured consequence is specific: the last SEVEN rounds of one trunk, the stretch that finalised
    its champion, carry `confidence: "high"` (which `append` refuses) and `body_sha: null`, were
    hand-written, copied `schema_version: 1`, and `verify` reported `ok: true`. The version field
    was the only thing distinguishing a written row from a forged one, and it is the one field a
    hand-writer copies first.

    A missing `comparator` is deliberately NOT reported here: `verify` owns that one as a ledger-wide
    count, and reporting it twice under two names is the noise this file exists to remove."""
    if rec.get("schema_version") == JOURNAL_SCHEMA_VERSION:
        return _journal_violations(rec)
    out = []
    structured = rec.get("schema_version") == SCHEMA_VERSION
    if structured:
        unexpected = sorted(set(rec) - STRUCTURED_FIELDS)
        if unexpected:
            out.append(_v(
                "structured_record_extra_fields",
                f"schema_version={SCHEMA_VERSION} only permits fixed fields plus artifact refs; "
                f"found {unexpected}. Put explanatory prose in a separate decision document that "
                f"references the round and summary artifact, not in rounds.jsonl"))
    verdict = rec.get("verdict")
    if verdict not in VERDICTS:
        out.append(_v("verdict_enum",
                      f"verdict={verdict!r} is outside the enum {VERDICTS}. Real ledgers have run "
                      f"nine spellings for these four states, so a query for one positive spelling "
                      f"missed four others. 'confirmed' and 'candidate' are the ORTHOGONAL axis: "
                      f"pass --confidence"))
    confidence = rec.get("confidence")
    if confidence is not None and confidence not in CONFIDENCE:
        out.append(_v("confidence_enum",
                      f"confidence={confidence!r} outside {CONFIDENCE}. This is the field the "
                      f"measured bypass came through: seven hand-written rows carried "
                      f"confidence='high', which is neither of the two states, on rows that also "
                      f"claimed to have been written by this tool"))
    comparator = rec.get("comparator")
    if comparator is not None and comparator not in COMPARATORS:
        out.append(_v("comparator_enum",
                      f"comparator={comparator!r} outside {COMPARATORS}. This field is "
                      f"required because without it a number is not comparable to any other "
                      f"number: one ledger's metric was an in-batch control geomean under a name "
                      f"that reads like a champion speed, and identical bytes read 2.41 to 2.96 "
                      f"across rounds"))
    # Pre-v3 ledgers carry the old explanatory `prediction` inline and still
    # validate it.  v3 accepts it at append time as a write-gate input but does
    # not serialize it into the structured record.
    if not structured and not str(rec.get("prediction") or "").strip():
        out.append(_v("prediction_missing",
                      "--prediction is required: the quantitative expectation written BEFORE the "
                      "edit (which bucket moves, by how much, which reading shows it). It is what "
                      "separates a round building attribution from a round guessing, and it is the "
                      "input to the mode-switch trigger -- which is why the trigger is 'no "
                      "prediction and no new evidence layer in K rounds' and not 'K rounds "
                      "without a win', a signal the measured hit-rate curve refutes directly"))
    mode = rec.get("mode")
    if mode is not None and mode not in MODES:
        out.append(_v("mode_enum",
                      f"mode={mode!r} outside {list(MODES)}. The first three are active SEARCH modes, "
                      f"the next four are legacy-compatible spellings, and the last three are round "
                      f"kinds, which it cannot. An unknown spelling is invisible to the mode "
                      f"ledger, and 24 of 70 measured rounds were filed under words neither file "
                      f"knew"))
    elif mode in LEGACY_SEARCH_MODES:
        out.append(_v("legacy_mode",
                      f"mode={mode!r} remains readable for backward compatibility, but new records "
                      f"must classify the work as sweep, branch, or climb"))
    goal = rec.get("goal")
    if goal is not None and len(str(goal).strip()) > GOAL_MAX_CHARS:
        out.append(_v("goal_too_long",
                      f"goal is {len(str(goal).strip())} chars and the bound is {GOAL_MAX_CHARS}. "
                      f"`goal` is the KEY `recall --goal` joins on, matched exactly, so a "
                      f"one-sentence task description is a key that matches only itself: measured, "
                      f"60 distinct goals over 70 rounds with 4 repeated, and one goal written "
                      f"three times under three different tails. Put the reasoning in "
                      f"--hypothesis, which is unbounded, and make this a short phrase a sibling "
                      f"arm would write the same way"))
    for tok in rec.get("evidence_layers") or []:
        if not isinstance(tok, str) or not _LAYER_TOKEN.match(tok):
            out.append(_v("evidence_layer_not_token",
                          f"evidence_layer {tok!r} is not a token (lowercase slug, <=32 chars). "
                          f"This field feeds the 'lit a NEW layer' half of the mode-switch "
                          f"trigger, so a per-round-unique string switches the trigger off "
                          f"silently -- 7 measured rounds put a sentence here. Canonical tokens: "
                          f"{list(EVIDENCE_LAYERS)}; a new one is allowed if it is a slug, and "
                          f"`verify` will report it"))
            break
    if structured:
        artifacts = rec.get("artifacts") or []
        if (not isinstance(artifacts, list) or len(artifacts) > 16
                or any(not isinstance(ref, str) or not ref.strip() or len(ref) > 256
                       for ref in artifacts)):
            out.append(_v("artifact_refs",
                          "schema_version=3 artifacts must be at most 16 non-empty refs of at "
                          "most 256 chars"))
        lever = rec.get("lever")
        if lever is not None and (not isinstance(lever, str) or not lever.strip()
                                  or len(lever) > 80):
            out.append(_v("lever_shape",
                          "schema_version=3 lever must be a non-empty fixed key of at most 80 chars"))
        summary_ref = rec.get("summary_ref")
        if summary_ref is not None and (not isinstance(summary_ref, str) or not summary_ref.strip()
                                        or len(summary_ref) > 256):
            out.append(_v("summary_ref",
                          "schema_version=3 summary_ref must be a non-empty artifact reference"))
        if rec.get("correctness") not in CORRECTNESS:
            out.append(_v("correctness_enum",
                          f"schema_version=3 correctness must be one of {CORRECTNESS}"))
    out.extend(_metric_violations(rec))
    if rec.get("measurement") is not None:
        out.extend(validate_measurement(rec["measurement"], "measurement"))
    return out


def _journal_rows(ledger: str) -> list[dict]:
    if not os.path.isfile(ledger):
        return []
    return [row for row in _load(ledger) if "_unparseable" not in row]


def journal_head(ledger: str) -> dict:
    rows = _journal_rows(ledger)
    canonical = [row for row in rows if row.get("schema_version") == JOURNAL_SCHEMA_VERSION]
    if not canonical:
        return {"sequence": 0, "entry_sha256": None, "identity": None, "record_id": None}
    row = canonical[-1]
    return {
        "sequence": row.get("sequence"),
        "entry_sha256": row.get("entry_sha256"),
        "identity": row.get("identity"),
        "record_id": row.get("record_id"),
    }


def _measurement_summary(measurement: dict | None) -> dict | None:
    if measurement is None:
        return None
    identity = measurement.get("identity") if isinstance(measurement.get("identity"), dict) else {}
    value = measurement.get("value")
    summary = {
        "valid": not validate_measurement(measurement),
        "measurement_id": measurement.get("measurement_id"),
        "value": value,
        "unit": identity.get("unit"),
        "aggregation": identity.get("aggregation"),
        "boundary": identity.get("boundary"),
        "source": identity.get("source"),
        "scope": identity.get("scope"),
        "sample_count": (identity.get("sample") or {}).get("count")
        if isinstance(identity.get("sample"), dict) else None,
    }
    return {key: value for key, value in summary.items() if value is not None}


def _comparison(args, measurement: dict | None) -> dict:
    if args.speedup is not None:
        return {
            "comparable": True,
            "comparator": args.comparator,
            "improvement": args.speedup - 1.0,
            "uncertainty": (args.noise_band or 0.0) / 100.0,
            "unit": "ratio_delta",
        }
    return {
        "comparable": False,
        "comparator": args.comparator,
        "reason": "no normalized ratio against the named comparator",
        "measurement_id": measurement.get("measurement_id") if measurement else None,
    }


def _disposition(args) -> dict:
    outcome = {
        "kept": "keep",
        "reverted": "revert",
        "null": "inconclusive",
        "blocked": "blocked",
    }[args.verdict]
    if getattr(args, "defer", False):
        outcome = "defer"
    return {
        "outcome": outcome,
        "confidence": args.confidence,
        "enabling_step": bool(args.enabling_step),
    }


def _load_recall_receipt(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused(f"cannot read recall receipt {path}: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != RECALL_RECEIPT_SCHEMA:
        raise Refused(f"{path} is not a {RECALL_RECEIPT_SCHEMA} receipt")
    expected = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if not isinstance(expected, str) or expected != _canonical_hash(payload):
        raise Refused(f"recall receipt hash mismatch: {path}")
    return receipt


def _required_recall_triggers(work_kind: str, previous: dict, current_identity: dict,
                              branch_reentry: bool) -> list[str]:
    required = []
    if work_kind == "climb_round":
        required.append("new_climb_hypothesis")
    if branch_reentry:
        required.append("branch_reentry")
    old_identity = previous.get("identity")
    if isinstance(old_identity, dict) and any(
            old_identity.get(kind) != current_identity.get(kind)
            and (old_identity.get(kind) or current_identity.get(kind))
            for kind in IDENTITY_KINDS):
        required.append("identity_change")
    return required


def _build_journal(args) -> dict:
    ledger = os.path.abspath(args.ledger)
    head = journal_head(ledger)
    named = _parse_named_hashes(getattr(args, "identity_hash", None))
    body = _hash_files(args.body)
    layout = _layout_hash(args.tensor)
    measured_at = _now()
    compatibility = {
        "round": args.round,
        "measured_at": measured_at,
        "comparator": args.comparator,
        "speedup_vs_comparator": args.speedup,
        "latency_ms": args.latency_ms,
        "noise_band_pct": args.noise_band,
        "verdict": args.verdict,
        "enabling_step": bool(args.enabling_step),
    }
    metric_problems = _metric_violations(compatibility)
    if metric_problems:
        raise Refused(metric_problems[0]["detail"])
    measurement = legacy_measurement(
        compatibility,
        shape_set=getattr(args, "shape", None),
        aggregation=getattr(args, "aggregation", "unspecified"),
        boundary=getattr(args, "boundary", "unknown"),
        source=getattr(args, "measurement_source", None),
        sample_count=getattr(args, "sample_count", None),
        measurement_scope=getattr(args, "measurement_scope", None),
        baseline_ref=getattr(args, "baseline_ref", None),
        comparator_ref=getattr(args, "comparator_ref", None),
    )
    if measurement is not None:
        problems = validate_measurement(measurement)
        if problems:
            raise Refused(problems[0]["detail"])
    measurement_event_id = getattr(args, "measurement_event", None)
    if measurement_event_id:
        events = _valid_events(args.measurement_events, MEASUREMENT_EVENT_SCHEMA)
        event = next((
            item for item in reversed(events)
            if item.get("event") == "valid"
            and item.get("measurement_id") == measurement_event_id
        ), None)
        if event is None:
            raise Refused(f"measurement event {measurement_event_id!r} is not declared valid")
        if event.get("work_kind") != args.work_kind or event.get("work_id") != (
                getattr(args, "work_id", None) or f"{args.work_kind}:{args.round}"):
            raise Refused("measurement event work unit does not match this journal append")
        if measurement is None:
            raise Refused("a declared valid measurement event needs a measurement summary")
    identity = {kind: named.get(kind) for kind in IDENTITY_KINDS}
    identity["body"] = named.get("body") or body
    identity["layout"] = named.get("layout") or layout
    identity["measurement"] = named.get("measurement") or (
        _canonical_hash(measurement.get("identity")) if measurement is not None else None
    )
    identity["comparator"] = named.get("comparator") or _canonical_hash({
        "name": args.comparator,
        "ref": getattr(args, "comparator_ref", None),
    })
    required_triggers = _required_recall_triggers(
        args.work_kind, head, identity, bool(getattr(args, "branch_reentry", False))
    )
    receipt_ref = getattr(args, "recall_receipt", None)
    receipt = _load_recall_receipt(receipt_ref) if receipt_ref else None
    if required_triggers:
        if receipt is None:
            raise Refused(f"recall receipt required for triggers {required_triggers}")
        missing = sorted(set(required_triggers) - set(receipt.get("triggers") or []))
        if missing:
            raise Refused(f"recall receipt does not cover triggers {missing}")
        if receipt.get("journal_head_sha256") != head["entry_sha256"]:
            raise Refused("recall receipt was not issued against the current journal head")
        receipt_identity = receipt.get("identity") or {}
        mismatched = [
            kind for kind in IDENTITY_KINDS
            if receipt_identity.get(kind) != identity.get(kind)
            and (receipt_identity.get(kind) or identity.get(kind))
        ]
        if mismatched:
            raise Refused(f"recall receipt identity differs for {mismatched}")
    first_sweep = args.work_kind == "sweep_batch" and head["sequence"] == 0
    if first_sweep and receipt is None:
        recall_exemption = "first_sweep"
    else:
        recall_exemption = None
    alternatives = [
        _bounded_text(item, "alternative", 160, required=True)
        for item in (getattr(args, "alternative", None) or [])
    ]
    evidence_refs = sorted(set(
        [_bounded_text(ref, "evidence ref", 256, required=True)
         for ref in (getattr(args, "artifact", None) or [])]
        + ([_bounded_text(args.summary_ref, "summary ref", 256, required=True)]
           if getattr(args, "summary_ref", None) else [])
    ))
    invalidations = []
    old_identity = head.get("identity")
    if isinstance(old_identity, dict):
        invalidations = [
            {"kind": kind, "previous_sha256": old_identity.get(kind),
             "current_sha256": identity.get(kind)}
            for kind in IDENTITY_KINDS
            if old_identity.get(kind) != identity.get(kind)
            and (old_identity.get(kind) or identity.get(kind))
        ]
    record = {
        "schema": JOURNAL_SCHEMA,
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "record_id": getattr(args, "work_id", None) or uuid.uuid4().hex,
        "sequence": int(head["sequence"] or 0) + 1,
        "previous_entry_sha256": head["entry_sha256"],
        "entry_sha256": None,
        "work_kind": args.work_kind,
        "work_id": getattr(args, "work_id", None) or f"{args.work_kind}:{args.round}",
        "round": args.round,
        "run_id": getattr(args, "run_id", None),
        "generation": getattr(args, "generation", None),
        "stage": getattr(args, "stage", None),
        "role": getattr(args, "role", None),
        "recorded_at": measured_at,
        "identity": identity,
        "hypothesis": _bounded_text(args.hypothesis, "hypothesis", 512, required=True),
        "alternatives": alternatives,
        "change_ref": _bounded_text(getattr(args, "change_ref", None), "change ref", 256,
                                    required=True),
        "comparison": _comparison(args, measurement),
        "measurement_summary": _measurement_summary(measurement),
        "oracle": {
            "status": {
                "pass": "ok", "passed": "ok", "fail": "failed", "error": "failed",
                "pending": "recheck",
            }.get(str(args.correctness).strip().lower(), str(args.correctness).strip().lower()),
            "evidence_ref": _bounded_text(getattr(args, "oracle_ref", None), "oracle ref", 256),
        },
        "disposition": _disposition(args),
        "evidence_refs": evidence_refs,
        "invalidations": invalidations,
        "next_action": _bounded_text(getattr(args, "next_action", None), "next action", 256),
        "recall_receipt_ref": receipt_ref,
        "recall_exemption": recall_exemption,
        "measurement_event_id": measurement_event_id,
    }
    record["entry_sha256"] = _entry_hash(record)
    bad = row_violations(record)
    if bad:
        raise Refused(bad[0]["detail"])
    return record


def build(args) -> dict:
    if getattr(args, "work_kind", None):
        return _build_journal(args)
    # These inputs are still required before an edit, preserving the legacy
    # write-gate contract.  Schema v3 intentionally does not retain them as
    # unbounded ledger prose.
    if not str(args.change or "").strip():
        raise Refused("--change is required for legacy schema_version=3 append")
    if not str(args.prediction or "").strip():
        raise Refused("--prediction is required: write it before the edit; schema_version=3 "
                      "keeps only the fixed result fields and artifact references")
    artifacts = sorted(set(args.artifact or []) |
                       ({args.summary_ref} if getattr(args, "summary_ref", None) else set()))
    correctness = {
        "pass": "ok", "passed": "ok", "fail": "failed", "error": "failed",
        "pending": "recheck",
    }.get(str(args.correctness).strip().lower(), str(args.correctness).strip().lower())
    if correctness not in CORRECTNESS:
        raise Refused(f"--correctness must be one of {CORRECTNESS} (pass/fail aliases accepted)")
    rec = {
        "schema_version": SCHEMA_VERSION,
        "round": args.round,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparator": args.comparator,
        "speedup_vs_comparator": args.speedup,
        "latency_ms": args.latency_ms,
        "noise_band_pct": args.noise_band,
        "correctness": correctness,
        "verdict": args.verdict,
        "confidence": args.confidence,
        "enabling_step": bool(args.enabling_step),
        # Which evidence layers this round actually lit. Read by the mode-switch trigger, where a
        # layer read for the first time counts as the round having BUILT something even if the
        # round won nothing -- and the same layer read for the eighth round running does not.
        "evidence_layers": sorted(args.evidence_layer or []),
        "artifacts": artifacts,
        "body_sha": body_sha(args.body),
        "layout_sig": parse_layout_sig(args.tensor),
        "goal": args.goal,
        "mode": args.mode,
        "arm_id": args.arm_id,
        "lever": getattr(args, "lever", None),
        "summary_ref": getattr(args, "summary_ref", None),
    }
    measurement = legacy_measurement(
        rec,
        shape_set=getattr(args, "shape", None),
        aggregation=getattr(args, "aggregation", "unspecified"),
        boundary=getattr(args, "boundary", "unknown"),
        source=getattr(args, "measurement_source", None),
        sample_count=getattr(args, "sample_count", None),
        measurement_scope=getattr(args, "measurement_scope", None),
        baseline_ref=getattr(args, "baseline_ref", None),
        comparator_ref=getattr(args, "comparator_ref", None),
    )
    if measurement is not None:
        rec["measurement"] = measurement
    if args.comparator not in COMPARATORS:   # argparse enforces it; a direct caller may not
        raise Refused(f"comparator={args.comparator!r} outside {COMPARATORS}. This field is "
                      f"required because without it a number is not comparable to any other "
                      f"number: one ledger's metric was an in-batch control geomean under a name "
                      f"that reads like a champion speed, and identical bytes read 2.41 to 2.96 "
                      f"across rounds")
    bad = row_violations(rec)
    if bad:
        raise Refused(bad[0]["detail"])
    return {k: v for k, v in rec.items() if v is not None}


# ---------------------------------------------------------------- verify / query

def _load(ledger):
    rows = []
    if not os.path.isfile(ledger):
        return rows
    with open(ledger) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_unparseable": str(exc), "_line": i})
    return rows


def _row_identity(row) -> dict:
    identity = row.get("identity")
    if isinstance(identity, dict):
        return {kind: identity.get(kind) for kind in IDENTITY_KINDS}
    return {
        "body": row.get("body_sha"),
        "layout": row.get("layout_sig"),
        **{kind: None for kind in IDENTITY_KINDS if kind not in ("body", "layout")},
    }


def _identity_now(rows, body_sha_now=None, layout_sig=None, identity_now=None) -> dict:
    current = {kind: None for kind in IDENTITY_KINDS}
    canonical = [row for row in rows if row.get("schema_version") == JOURNAL_SCHEMA_VERSION]
    if canonical:
        current.update(_row_identity(canonical[-1]))
    if isinstance(identity_now, dict):
        current.update({kind: identity_now.get(kind) for kind in IDENTITY_KINDS
                        if kind in identity_now})
    if body_sha_now:
        current["body"] = body_sha_now
    if layout_sig:
        current["layout"] = layout_sig
    return current


def _stale_reasons(row, body_sha_now=None, layout_sig=None, identity_now=None):
    current = _identity_now([], body_sha_now, layout_sig, identity_now)
    measured = _row_identity(row)
    return [
        kind for kind in IDENTITY_KINDS
        if current.get(kind) and measured.get(kind) and current[kind] != measured[kind]
    ]


def _is_stale(row, body_sha_now=None, layout_sig=None, identity_now=None):
    return bool(_stale_reasons(row, body_sha_now, layout_sig, identity_now))


def _row_refs(row):
    """The only evidence a recall summary exports: stable, bounded references."""
    refs = []
    candidates = (
        [row.get("_ledger"), row.get("summary_ref"), row.get("change_ref"),
         row.get("recall_receipt_ref")]
        + list(row.get("artifacts") or [])
        + list(row.get("evidence_refs") or [])
    )
    for ref in candidates:
        if isinstance(ref, str) and ref.strip() and ref not in refs:
            refs.append(ref[:256])
        if len(refs) >= 8:
            break
    return refs


def _summary_status(row, body_sha_now=None, layout_sig=None, identity_now=None):
    # The ordering is intentional: a stale result must be re-measured before
    # interpreting its candidate/recheck flag, and malformed/blocked rows are
    # unfinished rather than evidence of a completed result.
    if _is_stale(row, body_sha_now, layout_sig, identity_now):
        return "stale"
    current = _identity_now([], body_sha_now, layout_sig, identity_now)
    measured = _row_identity(row)
    if any(current.get(kind) and not measured.get(kind) for kind in IDENTITY_KINDS):
        return "recheck"
    if row.get("schema_version") == JOURNAL_SCHEMA_VERSION:
        disposition = row.get("disposition") or {}
        oracle = row.get("oracle") or {}
        if disposition.get("outcome") == "blocked":
            return "unfinished"
        if (oracle.get("status") != "ok" or disposition.get("outcome") in ("defer", "inconclusive")
                or row.get("measurement_summary") is None):
            return "recheck"
        return "valid"
    if row.get("_unparseable") or row.get("verdict") not in VERDICTS or row.get("verdict") == "blocked":
        return "unfinished"
    correctness = str(row.get("correctness") or "").strip().lower()
    if (correctness not in ("", "ok", "pass", "passed", "correct", "match")
            or row.get("confidence") == "candidate"
            or bool(row.get("enabling_step"))):
        return "recheck"
    return "valid"


def _summary_entry(row):
    if row.get("schema_version") == JOURNAL_SCHEMA_VERSION:
        entry = {
            "record_id": row.get("record_id"),
            "sequence": row.get("sequence"),
            "work_kind": row.get("work_kind"),
            "work_id": row.get("work_id"),
            "outcome": (row.get("disposition") or {}).get("outcome"),
            "comparison": row.get("comparison"),
            "next_action": row.get("next_action"),
            "refs": list(row.get("evidence_refs") or [])[:8],
        }
        return {key: value for key, value in entry.items() if value is not None}
    entry = {
        "round": row.get("round"),
        "verdict": row.get("verdict"),
        "mode": row.get("mode"),
        "lever": row.get("lever"),
        "refs": _row_refs(row),
    }
    return {key: value for key, value in entry.items() if value is not None}


def summary(ledgers, body_sha_now=None, layout_sig=None, limit=SUMMARY_DEFAULT_LIMIT,
            identity_now=None):
    """A bounded, stable recall handoff — never an implicit ledger replay."""
    if not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError("limit must be an integer in [1, 64]")
    rows = []
    for ledger in sorted(set(ledgers)):
        for row in _load(ledger):
            rows.append(dict(row, _ledger=ledger))
    rows.sort(key=lambda row: (str(row.get("_ledger")), not isinstance(row.get("round"), int),
                               row.get("round") if isinstance(row.get("round"), int) else 0,
                               row.get("_line", 0)))
    current_identity = _identity_now(rows, body_sha_now, layout_sig, identity_now)
    counts = {name: 0 for name in SUMMARY_BUCKETS}
    entries = {name: [] for name in SUMMARY_BUCKETS}
    truncated = {name: 0 for name in SUMMARY_BUCKETS}
    for row in rows:
        bucket = _summary_status(row, identity_now=current_identity)
        counts[bucket] += 1
        if len(entries[bucket]) < limit:
            entries[bucket].append(_summary_entry(row))
        else:
            truncated[bucket] += 1
    return {
        "schema": ROUND_SUMMARY_SCHEMA,
        "ledgers": sorted(set(ledgers))[:16],
        "identity": current_identity,
        "counts": counts,
        "entries": entries,
        "truncated": truncated,
        "refs": sorted(set(ledgers))[:16],
        "journal_head": next((
            {"record_id": row.get("record_id"), "sequence": row.get("sequence"),
             "entry_sha256": row.get("entry_sha256")}
            for row in reversed(rows)
            if row.get("schema_version") == JOURNAL_SCHEMA_VERSION
        ), None),
    }


def _append_jsonl(path: str, value: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mark_measurement(event_log: str, measurement_id: str, work_kind: str, work_id: str,
                     evidence_ref: str) -> dict:
    if work_kind not in WORK_KINDS:
        raise ValueError(f"work_kind must be one of {WORK_KINDS}")
    for label, value in (("measurement_id", measurement_id), ("work_id", work_id),
                         ("evidence_ref", evidence_ref)):
        if not isinstance(value, str) or not value.strip() or "\n" in value:
            raise ValueError(f"{label} must be a non-empty one-line value")
    event = {
        "schema": MEASUREMENT_EVENT_SCHEMA,
        "event": "valid",
        "measurement_id": measurement_id,
        "work_kind": work_kind,
        "work_id": work_id,
        "evidence_ref": evidence_ref,
        "at": _now(),
    }
    event["event_sha256"] = _canonical_hash(event)
    _append_jsonl(event_log, event)
    return event


def _record_measurement_event(event_log: str, measurement_id: str, record: dict) -> None:
    event = {
        "schema": MEASUREMENT_EVENT_SCHEMA,
        "event": "recorded",
        "measurement_id": measurement_id,
        "record_id": record["record_id"],
        "entry_sha256": record["entry_sha256"],
        "at": _now(),
    }
    event["event_sha256"] = _canonical_hash(event)
    _append_jsonl(event_log, event)


def issue_recall_receipt(ledgers: list[str], out: str, triggers: list[str], *,
                         body_sha_now=None, layout_sig=None, identity_now=None,
                         goal=None, mode_ledger=None, limit=SUMMARY_DEFAULT_LIMIT,
                         event_log=None) -> dict:
    unknown = sorted(set(triggers) - set(RECALL_TRIGGERS))
    if unknown or not triggers:
        raise ValueError(f"recall triggers must be a non-empty subset of {RECALL_TRIGGERS}")
    current = []
    for ledger in ledgers:
        current.extend(_load(ledger))
    identity = _identity_now(current, body_sha_now, layout_sig, identity_now)
    effective_limit = min(limit, 2)
    result = recall(
        ledgers, body_sha_now, layout_sig, goal, mode_ledger, effective_limit,
        identity_now=identity
    )
    heads = [journal_head(ledger) for ledger in ledgers]
    head = next((item for item in reversed(heads) if item["entry_sha256"]), heads[-1] if heads else {})
    receipt = {
        "schema": RECALL_RECEIPT_SCHEMA,
        "receipt_id": uuid.uuid4().hex,
        "issued_at": _now(),
        "triggers": sorted(set(triggers)),
        "journal_refs": sorted(set(ledgers))[:16],
        "journal_head_sha256": head.get("entry_sha256"),
        "identity": identity,
        "entry_limit": effective_limit,
        "summary": result["summary"],
        "same_goal": result.get("same_goal"),
        "empty": result["rows_indexed"] == 0,
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    if len(_canonical_bytes(receipt)) > RECALL_RECEIPT_MAX_BYTES:
        raise ValueError(f"bounded recall receipt exceeds {RECALL_RECEIPT_MAX_BYTES} bytes")
    _atomic_json(out, receipt)
    if event_log:
        event = {
            "schema": RECALL_EVENT_SCHEMA,
            "event": "issued",
            "receipt_id": receipt["receipt_id"],
            "receipt_ref": out,
            "receipt_sha256": receipt["receipt_sha256"],
            "at": receipt["issued_at"],
        }
        event["event_sha256"] = _canonical_hash(event)
        _append_jsonl(event_log, event)
    return receipt


def _consume_recall_event(event_log: str, receipt: dict, record: dict) -> None:
    event = {
        "schema": RECALL_EVENT_SCHEMA,
        "event": "consumed",
        "receipt_id": receipt["receipt_id"],
        "record_id": record["record_id"],
        "entry_sha256": record["entry_sha256"],
        "at": _now(),
    }
    event["event_sha256"] = _canonical_hash(event)
    _append_jsonl(event_log, event)


def abandon_recall_receipt(event_log: str, receipt_ref: str, reason: str) -> dict:
    """Close a receipt whose work never happened, on the record and with a reason.

    A receipt is bound to the journal head it was issued against, and the model had exactly two
    terminal states: consumed, or an `error` finding forever. Work that is legitimately dropped --
    a hypothesis abandoned before its round, a re-plan, a stage that closed early -- left a receipt
    that could never be consumed, so the only route to a clean close was to forge a `consumed`
    event for a round that did not exist. Four of four owners hit this in one campaign.

    Abandoning is therefore a first-class outcome and carries the same weight as consuming: a
    hashed event, the receipt it settles, and a stated reason. What it must not do is erase the
    fact that a recall was issued -- the timeline still shows both.
    """
    reason = " ".join(str(reason or "").split())
    if len(reason) < 12:
        raise ValueError("abandoning a recall receipt requires a reason of at least 12 characters "
                         "-- an unexplained abandon is indistinguishable from a forged consume")
    receipt = _load_recall_receipt(receipt_ref)
    event = {
        "schema": RECALL_EVENT_SCHEMA,
        "event": "abandoned",
        "receipt_id": receipt["receipt_id"],
        "receipt_ref": receipt_ref,
        "reason": reason,
        "at": _now(),
    }
    event["event_sha256"] = _canonical_hash(event)
    _append_jsonl(event_log, event)
    return event


def _valid_events(path: str, schema: str) -> list[dict]:
    rows = _load(path)
    return [
        row for row in rows
        if row.get("schema") == schema and row.get("event_sha256") == _canonical_hash(
            {key: value for key, value in row.items() if key != "event_sha256"}
        )
    ]


def lifecycle_check(ledger: str, measurement_events: str, recall_events: str) -> dict:
    """Report only declared pending facts; never infer strategy from missing classifier output."""
    findings = []
    for label, path, schema in (
        ("measurement", measurement_events, MEASUREMENT_EVENT_SCHEMA),
        ("recall", recall_events, RECALL_EVENT_SCHEMA),
    ):
        invalid = [
            row for row in _load(path)
            if row.get("schema") != schema or row.get("event_sha256") != _canonical_hash(
                {key: value for key, value in row.items() if key != "event_sha256"}
            )
        ]
        if invalid:
            findings.append({
                "kind": f"{label}_event_invalid",
                "severity": "error",
                "detail": f"{len(invalid)} {label} lifecycle event(s) are malformed or tampered",
                "ref": path,
            })
    measurement_rows = _valid_events(measurement_events, MEASUREMENT_EVENT_SCHEMA)
    recorded_measurements = {
        row.get("measurement_id") for row in measurement_rows if row.get("event") == "recorded"
    }
    pending_measurements = [
        row for row in measurement_rows
        if row.get("event") == "valid" and row.get("measurement_id") not in recorded_measurements
    ]
    for row in pending_measurements:
        findings.append({
            "kind": "valid_measurement_unrecorded",
            "severity": "error",
            "detail": f"valid measurement {row.get('measurement_id')!r} for "
                      f"{row.get('work_kind')} {row.get('work_id')!r} has no journal record",
            "ref": row.get("evidence_ref"),
        })
    recall_rows = _valid_events(recall_events, RECALL_EVENT_SCHEMA)
    # Abandoning settles a receipt as surely as consuming it. Both are recorded outcomes; only an
    # issued receipt nobody ever answered for is a finding.
    settled = {row.get("receipt_id") for row in recall_rows
               if row.get("event") in ("consumed", "abandoned")}
    pending_recall = [
        row for row in recall_rows
        if row.get("event") == "issued" and row.get("receipt_id") not in settled
    ]
    for row in pending_recall:
        findings.append({
            "kind": "recall_receipt_unconsumed",
            "severity": "error",
            "detail": f"recall receipt {row.get('receipt_id')!r} has no following journal work unit",
            "ref": row.get("receipt_ref"),
        })
    verification = verify(ledger) if os.path.isfile(ledger) else {
        "ok": True, "rows": 0, "findings": [],
    }
    if not verification.get("ok"):
        findings.append({
            "kind": "journal_invalid",
            "severity": "error",
            "detail": "journal verification has error-severity findings",
            "ref": ledger,
        })
    return {
        "schema": "kernel_opt.journal_lifecycle/1",
        "ok": not findings,
        "journal": ledger,
        "journal_rows": verification.get("rows", 0),
        "pending_measurements": len(pending_measurements),
        "pending_recall_receipts": len(pending_recall),
        "findings": findings,
    }


def verify(ledger):
    """Ledger-wide facts a per-row writer structurally cannot see."""
    rows = _load(ledger)
    out = {"ledger": ledger, "rows": len(rows), "findings": []}

    def f(kind, detail, severity="error"):
        out["findings"].append({"kind": kind, "detail": detail, "severity": severity})

    bad = [r for r in rows if "_unparseable" in r]
    if bad:
        f("unparseable_rows", f"{len(bad)} line(s) are not JSON (first at line {bad[0]['_line']})")

    versions = collections.Counter(r.get("schema_version") for r in rows if "_unparseable" not in r)
    if len(versions) > 1 and not set(versions) <= {SCHEMA_VERSION, JOURNAL_SCHEMA_VERSION}:
        f("mixed_schema_versions", f"rows carry {dict(versions)} -- a reader cannot apply one "
                                   f"interpretation to the file")
    if None in versions:
        f("unversioned_rows", f"{versions[None]} row(s) carry no schema_version, so they were not "
                              f"written through this tool and none of its refusals applied to them")

    vocab = collections.Counter(
        r.get("verdict") for r in rows
        if "_unparseable" not in r and r.get("schema_version") != JOURNAL_SCHEMA_VERSION
    )
    stray = {v: n for v, n in vocab.items() if v is not None and v not in VERDICTS}
    if stray:
        f("verdict_vocabulary_drift", f"verdicts outside the enum: {stray}")

    # The two numbering spaces. Contiguity alone is the WRONG check: in one measured ledger a
    # kernel had zero `round` gaps precisely because a second numbering space (`staged_round`) had renumbered
    # 721 rows and folded the gaps away. So compare the spaces rather than counting holes in one.
    nums = [r.get("round") for r in rows if isinstance(r.get("round"), int)]
    if nums:
        dupes = [n for n, c in collections.Counter(nums).items() if c > 1]
        if dupes:
            f("round_number_collision", f"round(s) appended twice: {sorted(dupes)[:10]} -- two "
                                        f"owners with independent counters, and the later row "
                                        f"silently shadows the earlier for any reader keyed on it")
        span = max(nums) - min(nums) + 1
        if span != len(set(nums)):
            f("round_number_gaps", f"{span - len(set(nums))} number(s) missing between "
                                   f"{min(nums)} and {max(nums)}", severity="warn")

    # Comparator coverage. A file whose rows quote different comparators is fine; a file whose rows
    # do not SAY is a file whose numbers cannot be ordered.
    unnamed = sum(
        1 for r in rows
        if "_unparseable" not in r and r.get("schema_version") != JOURNAL_SCHEMA_VERSION
        and not r.get("comparator")
    )
    if unnamed:
        f("comparator_missing", f"{unnamed} row(s) carry a number with no comparator; those rows "
                                f"are not comparable to any other row in this file")

    unit_mixed = [r.get("round") for r in rows
                  if r.get("speedup_vs_comparator") is not None and r.get("latency_ms") is not None]
    if unit_mixed:
        f("two_units_one_row", f"rows {unit_mixed[:10]} set both metric fields")

    live = [r for r in rows if "_unparseable" not in r]
    journal_rows = [r for r in live if r.get("schema_version") == JOURNAL_SCHEMA_VERSION]
    previous = None
    for expected_sequence, row in enumerate(journal_rows, 1):
        if row.get("sequence") != expected_sequence:
            f("journal_sequence", f"journal sequence expected {expected_sequence}, "
                                  f"found {row.get('sequence')!r}")
            break
        if row.get("previous_entry_sha256") != previous:
            f("journal_hash_chain", f"journal sequence {expected_sequence} does not reference "
                                    "the preceding entry hash")
            break
        previous = row.get("entry_sha256")

    # Every refusal `append` makes, re-run against what is on disk. Without this the refusals bound
    # only the rows that chose to come through the writer, which is the population that did not need
    # them.
    integrity, shape = [], collections.defaultdict(list)
    for r in live:
        for bad in row_violations(r):
            if bad["rule"] in INTEGRITY_RULES:
                integrity.append({"round": r.get("round"), "rule": bad["rule"],
                                  "claimed_schema_version": r.get("schema_version"),
                                  "detail": bad["detail"]})
            else:
                shape[bad["rule"]].append(r.get("round"))
    if integrity:
        forged = sorted({x["round"] for x in integrity
                         if x["claimed_schema_version"] == SCHEMA_VERSION and x["round"] is not None})
        by_rule = collections.Counter(x["rule"] for x in integrity)
        f("append_refusals_violated",
          f"{len(integrity)} violation(s) of a refusal `append` has always made, so those rows did "
          f"not come through it: {dict(by_rule)}. {len(forged)} row(s) carry "
          f"schema_version={SCHEMA_VERSION} anyway (rounds {forged[:12]}) -- a hand-written row "
          f"that copies the version inherits this tool's credibility and none of its refusals, and "
          f"a version check alone cannot see that. Measured: the last seven rounds of one trunk, "
          f"the stretch that finalised its champion. Per-row detail in `integrity_violations`")
        out["integrity_violations"] = integrity[:20]
    if shape:
        f("record_shape_violations",
          f"{sum(len(v) for v in shape.values())} row(s) do not meet a field vocabulary this tool "
          f"now enforces at write time: "
          f"{ {k: sorted(x for x in v if x is not None)[:12] for k, v in sorted(shape.items())} }. "
          f"These are NOT evidence the writer was bypassed -- the vocabularies are newer than the "
          f"ledger -- but each one disables a consumer: an unknown `mode` is invisible to the mode "
          f"ledger, an over-long `goal` matches only itself in `recall`, and a prose "
          f"`evidence_layer` makes its round read as productive forever", severity="warn")
        out["shape_violations"] = {k: sorted(x for x in v if x is not None)
                                   for k, v in sorted(shape.items())}

    # The evidence-layer token space, because it is what the mode-switch trigger reads.
    layers = collections.Counter(x for r in live for x in (r.get("evidence_layers") or [])
                                 if isinstance(x, str))
    noncanonical = {k: v for k, v in sorted(layers.items()) if k not in EVIDENCE_LAYERS}
    if noncanonical:
        once = sorted(k for k, v in noncanonical.items() if v == 1)
        f("evidence_layer_drift",
          f"{len(noncanonical)} token(s) outside the canonical set: {noncanonical}. {len(once)} of "
          f"them appear exactly ONCE, and a once-only token is precisely what makes its round read "
          f"as 'lit a new evidence layer' to `search_mode.py check` -- so a token space that is "
          f"unique per round leaves the mode-switch trigger permanently satisfied and silent. "
          f"Reconcile to {list(EVIDENCE_LAYERS)}, or add the layer to that list if it is real",
          severity="warn")

    # `goal` as a key rather than as a sentence. Both halves are about whether `recall` can join.
    goals = [str(r.get("goal")).strip() for r in live if r.get("goal")]
    distinct_goals = len({goal_key(g) for g in goals})
    if goals:
        near = near_goal_groups(goals)
        if distinct_goals > 0.6 * len(goals):
            f("goal_key_inflation",
              f"{distinct_goals} distinct goals over {len(goals)} rows with a goal. `recall --goal` "
              f"matches EXACTLY, so at this ratio the index cannot join two tactics against one "
              f"goal -- which is the single miss it was built for. Measured at 60/70 in one "
              f"campaign", severity="warn")
        if near:
            f("goal_near_duplicates",
              f"{len(near)} goal(s) written under more than one spelling, so they are separate keys "
              f"to `recall` and the rounds under them cannot see each other: "
              f"{json.dumps(near, ensure_ascii=False)[:700]}", severity="warn")

    # Which rows a verdict can be attributed to a body / a layout at all.
    metric_rows = [
        r for r in live if r.get("schema_version") != JOURNAL_SCHEMA_VERSION
        and any(r.get(k) is not None for k in METRIC_FIELDS)
    ]
    no_body = [r.get("round") for r in metric_rows if not r.get("body_sha")]
    if no_body:
        f("body_sha_missing",
          f"{len(no_body)} row(s) carry a number and no body_sha (rounds "
          f"{sorted(x for x in no_body if x is not None)[:12]}). Nothing can then tell that verdict "
          f"from one measured on a kernel that no longer exists, and `recall`'s stale_negatives "
          f"cannot see the row at all -- it needs a body to compare against. Pass --body",
          severity="warn")
    if metric_rows and not any(r.get("layout_sig") for r in live):
        f("layout_sig_absent",
          "no row in this ledger declares a layout_sig, so `recall --layout-sig` will return zero "
          "stale negatives for a reason that is NOT 'nothing is stale'. A launcher-side transpose "
          "changes no byte of the kernel and voids every verdict measured on the old stride; in "
          "one recorded run one such transpose inverted six standing rejections at once. Pass "
          "--tensor",
          severity="warn")

    out["census"] = {
        "modes": dict(sorted(collections.Counter(
            r.get("mode") for r in live if r.get("mode")).items())),
        "search_modes_never_used": sorted(set(SEARCH_MODES) - {r.get("mode") for r in live}),
        "evidence_layers": dict(sorted(layers.items())),
        "goals_distinct_over_rows": [distinct_goals, len(goals)] if goals else None,
        "rows_with_number_lacking_body_sha": len(no_body),
    }
    out["ok"] = not [x for x in out["findings"] if x["severity"] == "error"]
    return out


def recall(ledgers, body_sha_now=None, layout_sig=None, goal=None, mode_ledger=None,
           limit=SUMMARY_DEFAULT_LIMIT, identity_now=None):
    """The RECALL step: one query, three answers, before you form this round's hypothesis.

    WHY A QUERY AND NOT A DOCUMENT READ. The round loop was `profile -> analyze -> edit -> verify`
    with no retrospective step at all: `analyze` requires every claim to name a reading (A1/B2/...),
    and every one of those readings is from the CURRENT profile. Memory was cut three ways --
    `decision_log.md` is freeform markdown inside `dir_<id>/` while the same brief forbids reading
    another `dir_<id>/`; the cross-run wall ledger's read, write and consume tools are all in this
    pack's `drop_paths`; and `deferred[]`'s only status ladder ends at user action, so it is a
    "waiting on a human" list rather than a "worth retrying" one.

    Those prohibitions are CORRECT -- reading another direction's transcripts would blow the
    context budget that makes supervision affordable. So memory has to be a structured index, and
    this is it: a bounded JSON answer, not a tree walk.

    THE INDEX IS DERIVED, NOT MAINTAINED. It is computed from the round ledgers on demand rather
    than kept as a fifth hand-written file, because a hand-written index is exactly the artifact
    that drifts -- which is the defect this whole schema exists to fix.

    KEYED ON GOAL, NOT TACTIC. Different tactics can pursue one objective across separate stages;
    an index keyed only on tactic cannot reconnect them."""
    rows = []
    for led in ledgers:
        for r in _load(led):
            if "_unparseable" not in r:
                rows.append(dict(r, _ledger=led))

    by_goal: dict = collections.defaultdict(list)
    for r in rows:
        if r.get("goal"):
            by_goal[r["goal"]].append(r)

    current_identity = _identity_now(rows, body_sha_now, layout_sig, identity_now)
    stale = []
    for row in rows:
        negative = (
            (row.get("schema_version") == JOURNAL_SCHEMA_VERSION
             and (row.get("disposition") or {}).get("outcome") in ("revert", "inconclusive"))
            or row.get("verdict") in ("reverted", "null")
        )
        if not negative or not _is_stale(row, identity_now=current_identity):
            continue
        stale.append({
            "round": row.get("round"),
            "record_id": row.get("record_id"),
            "work_kind": row.get("work_kind"),
            "goal": row.get("goal"),
            "verdict": row.get("verdict") or (row.get("disposition") or {}).get("outcome"),
            "arm_id": row.get("arm_id"),
            "stale_dimensions": _stale_reasons(row, identity_now=current_identity),
            "ledger": row["_ledger"],
        })

    untried_modes = []
    if mode_ledger and os.path.isfile(mode_ledger):
        with open(mode_ledger) as fh:
            md = json.load(fh)
        untried_modes = [m for m, v in (md.get("modes") or {}).items()
                         if v.get("state") == "untried"]

    same_goal = [{"round": r.get("round"), "lever": r.get("lever"),
                  "verdict": r.get("verdict"), "arm_id": r.get("arm_id"),
                  "stale": _is_stale(r, identity_now=current_identity), "refs": _row_refs(r)}
                 for r in by_goal.get(goal, [])] if goal else None
    if same_goal is not None:
        same_goal = same_goal[:limit]
    return {
        "rows_indexed": len(rows),
        # 1. What has already been tried against THIS goal, from every arm, so a second tactic
        #    against a goal a sibling already closed is visible before it is spent.
        "same_goal": same_goal,
        "same_goal_truncated": max(0, len(by_goal.get(goal, [])) - limit) if goal else 0,
        # Compatibility discovery only: this list is bounded and contains no rows.
        "goals_seen": sorted(by_goal)[:limit],
        "goals_seen_truncated": max(0, len(by_goal) - limit),
        # 2. Verdicts measured against a body or layout that no longer exists. Retrying one of
        #    these is a QUERY RESULT, not an act of memory -- which is the whole point, because
        #    "when the structure changes, levers you previously rejected need re-testing" was
        #    advisory prose with no carrier, and one layout change inverted six of them at once.
        "stale_negatives": stale[:limit],
        "stale_negatives_truncated": max(0, len(stale) - limit),
        # 3. Which search modes were never entered (P4).
        "untried_modes": untried_modes,
        "summary": summary(ledgers, body_sha_now, layout_sig, limit, current_identity),
    }


def _recall_stdout_view(result: dict, out: str | None, receipt_out: str | None) -> dict:
    """The stdout form of a recall that has already been written to disk.

    A recall answers three questions -- what this goal already tried, which negatives went stale,
    which modes were never entered -- and the caller has to READ that answer before forming the
    next hypothesis. Printing the whole document alongside `--out` put those three fields behind
    `summary`, whose bucket counts are the largest part of the payload and are recoverable from
    the file. Hosts that cap tool output then truncate from the end, so the decision fields are
    what got cut while the recoverable part survived.

    So when the answer is on disk, stdout keeps the three fields verbatim and drops only what the
    file already holds. This is a rendering choice, not a schema change: `--out` and
    `--receipt-out` are unchanged, and a recall with neither still prints in full.
    """
    view = {
        "recall_written_to": out,
        "receipt_written_to": receipt_out,
        "rows_indexed": result.get("rows_indexed"),
        "same_goal": result.get("same_goal"),
        "same_goal_truncated": result.get("same_goal_truncated"),
        "stale_negatives": result.get("stale_negatives"),
        "stale_negatives_truncated": result.get("stale_negatives_truncated"),
        "untried_modes": result.get("untried_modes"),
        "summary_in_file_only": sorted(result.get("summary") or {}),
    }
    return {key: value for key, value in view.items() if value is not None}


def query(ledger, goal=None, layout_sig=None, body_sha_now=None, stale_only=False, mode=None,
          work_kind=None, identity_now=None, limit=64):
    """The read side. This is what RECALL calls, and it is a JSON query rather than a document read
    because the packs correctly forbid reading another direction's transcripts and dirs -- so the
    memory has to be a structured index or it cannot exist at all.

    `stale_only` is the point of the whole schema: a verdict whose `body_sha` or `layout_sig` no
    longer matches the current tree was measured against a kernel that no longer exists. Retrying it
    stops being an act of memory and becomes the result of a query."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
        raise ValueError("query limit must be an integer in [1, 64]")
    rows = [r for r in _load(ledger) if "_unparseable" not in r]
    current_identity = _identity_now(rows, body_sha_now, layout_sig, identity_now)
    hits = []
    for r in rows:
        if goal and r.get("goal") != goal:
            continue
        if mode and r.get("mode") != mode:
            continue
        if work_kind and r.get("work_kind") != work_kind:
            continue
        if layout_sig and r.get("layout_sig") != layout_sig:
            continue
        stale = _is_stale(r, identity_now=current_identity)
        if stale_only and not stale:
            continue
        hits.append(dict(r, _stale=stale,
                         _stale_dimensions=_stale_reasons(r, identity_now=current_identity)))
    return {
        "ledger": ledger,
        "matched": len(hits),
        "rows": hits[:limit],
        "truncated": max(0, len(hits) - limit),
    }


def decision_log_ref(round_number, summary_ref):
    """The only permitted shape for an explanatory decision-log pointer."""
    if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 1:
        raise ValueError("round must be a positive integer")
    if not isinstance(summary_ref, str) or not summary_ref.strip() or len(summary_ref) > 256:
        raise ValueError("summary_ref must be a non-empty artifact reference of at most 256 chars")
    return {"schema": DECISION_LOG_REF_SCHEMA, "round": round_number,
            "summary_ref": summary_ref.strip()}


def _atomic_json(path, doc):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".round_record.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def describe():
    return {
        "tool": "round_record",
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "journal_schema": JOURNAL_SCHEMA,
        "work_kinds": list(WORK_KINDS),
        "identity_kinds": list(IDENTITY_KINDS),
        "structured_record_fields": sorted(STRUCTURED_FIELDS),
        "journal_record_fields": sorted(JOURNAL_FIELDS),
        "summary_schema": ROUND_SUMMARY_SCHEMA,
        "summary_buckets": list(SUMMARY_BUCKETS),
        "decision_log_schema": DECISION_LOG_REF_SCHEMA,
        "bounds": {"summary_default_entries_per_bucket": SUMMARY_DEFAULT_LIMIT,
                   "summary_max_entries_per_bucket": 64,
                   "artifact_refs_per_structured_record": 16},
        "compatibility": "schema_version 1/2/3 rows remain readable; work_kind writes use v4",
    }


# ---------------------------------------------------------------- cli

def _add_append_args(ap):
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--work-kind", choices=WORK_KINDS,
                    help="emit canonical v4 journal envelope; absent preserves legacy v3 append")
    ap.add_argument("--work-id")
    ap.add_argument("--run-id")
    ap.add_argument("--generation", type=int)
    ap.add_argument("--stage")
    ap.add_argument("--role")
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--prediction", required=True,
                    help="the QUANTITATIVE expectation, written before the edit: which bucket "
                         "moves, by how much, which reading shows it")
    ap.add_argument("--change",
                    help="legacy v3 write-gate description; v4 uses --change-ref")
    ap.add_argument("--change-ref", help="bounded source/patch/result reference for v4")
    ap.add_argument("--alternative", action="append",
                    help="repeatable bounded alternative considered; no classifier vocabulary")
    ap.add_argument("--next-action")
    ap.add_argument("--oracle-ref")
    ap.add_argument("--identity-hash", action="append", metavar="KIND=SHA256",
                    help="repeatable full identity digest for body/layout/comparator/measurement/"
                         "environment/toolchain/skill_build/policy")
    ap.add_argument("--recall-receipt")
    ap.add_argument("--branch-reentry", action="store_true")
    ap.add_argument("--measurement-event")
    ap.add_argument("--measurement-events", default=DEFAULT_MEASUREMENT_EVENTS)
    ap.add_argument("--recall-events", default=DEFAULT_RECALL_EVENTS)
    ap.add_argument("--defer", action="store_true")
    ap.add_argument("--lever", help="optional fixed lever key retained in schema_version=3; "
                                    "--change remains a write-gate input but is not serialized")
    ap.add_argument("--comparator", required=True, choices=COMPARATORS)
    ap.add_argument("--speedup", type=float, help="ratio vs --comparator (1.0 == parity)")
    ap.add_argument("--latency-ms", type=float, dest="latency_ms")
    ap.add_argument("--noise-band", type=float, dest="noise_band",
                    help="percent; the measured spread of THIS window's control arm. Required for "
                         "--verdict kept")
    ap.add_argument("--correctness", default="ok")
    ap.add_argument("--verdict", required=True, choices=VERDICTS)
    ap.add_argument("--confidence", choices=CONFIDENCE)
    ap.add_argument("--enabling-step", action="store_true", dest="enabling_step",
                    help="provisionally kept toward a coupled combination; its net is confirmed "
                         "against the checkpoint later")
    ap.add_argument("--evidence", default="")
    ap.add_argument("--evidence-layer", action="append", dest="evidence_layer",
                    help="repeatable; a TOKEN (lowercase slug), not a sentence -- the prose goes in "
                         "--evidence. A layer lit for the FIRST time is one of the two signals that "
                         "says this mode is still building, so a per-round-unique string turns that "
                         "trigger off. Canonical: " + " / ".join(EVIDENCE_LAYERS))
    ap.add_argument("--artifact", action="append", help="repeatable; a dereferenceable path")
    ap.add_argument("--summary-ref",
                    help="bounded summary artifact; also retained in artifacts for recall")
    ap.add_argument("--body", action="append", help="repeatable; source file(s) this number "
                                                    "describes -> body_sha")
    ap.add_argument("--tensor", action="append", help="repeatable; name:dtype:stride_pattern "
                                                      "-> layout_sig")
    ap.add_argument("--goal", help=f"what this round was TRYING to achieve, not the tactic, as a "
                                   f"reusable KEY of at most {GOAL_MAX_CHARS} chars. Two tactics "
                                   f"against one goal are what a tactic-keyed index cannot see "
                                   f"connect: fourteen rounds went into an in-kernel coalesced "
                                   f"read while "
                                   f"the winning launcher-side transpose had the same goal, and "
                                   f"nothing linked them. A sentence-length goal reproduces that "
                                   f"miss by matching only itself")
    ap.add_argument("--mode", choices=MODES,
                    help="active search modes: sweep / branch / climb; legacy values are accepted "
                         "for old ledgers; diagnose / verify / converge are round kinds")
    ap.add_argument("--arm-id", dest="arm_id")
    ap.add_argument("--shape", action="append",
                    help="repeatable shape id for the canonical measurement identity")
    ap.add_argument("--aggregation", default="unspecified",
                    help="measurement aggregation (for example geomean); explicit even when legacy")
    ap.add_argument("--boundary", default="unknown",
                    help="timing boundary; unknown is retained for legacy measurements")
    ap.add_argument("--measurement-source", choices=(
        "device_timing", "host_timing", "derived", "static_model",
        "legacy_latency", "legacy_ratio", "unknown"),
                    help="provenance for the canonical measurement; omitted uses an honest legacy source")
    ap.add_argument("--measurement-scope", choices=("device", "host", "derived", "unknown"),
                    help="measurement scope; device_timing requires --measurement-scope device")
    ap.add_argument("--sample-count", type=int,
                    help="observed sample count; required with --measurement-source device_timing")
    ap.add_argument("--baseline-ref",
                    help="baseline artifact/id; required for a fully attributable device measurement")
    ap.add_argument("--comparator-ref",
                    help="comparator artifact/id; required for a fully attributable device measurement")


def main(argv=None) -> int:
    # `--selftest` is the same thing as the `selftest` subcommand, and both spellings exist for a
    # reason that is not cosmetic: the repo's selftest registry (`validate.py`) invokes every tool it
    # guards as `<tool> --selftest`, so a tool that only answers the subcommand form cannot be
    # registered -- and an unregistered selftest is the unrun selftest that registry exists to
    # prevent. This file's selftest is one half of a cross-file drift guard, so it has to be
    # reachable that way.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return _selftest()
    if argv == ["--describe", "json"] or argv == ["--describe", "--format=json"]:
        print(json.dumps(describe(), ensure_ascii=False, sort_keys=True))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_append_args(sub.add_parser("append", help="append one validated round"))
    v = sub.add_parser("verify", help="ledger-wide checks")
    v.add_argument("--ledger", required=True)
    v.add_argument("--strict", action="store_true", help="exit 1 on an error-severity finding")
    rc = sub.add_parser("recall", help="the RECALL step: what was already tried against this "
                                       "goal, which negatives are now stale, which modes untried")
    rc.add_argument("--ledger", action="append", required=True,
                    help="repeatable -- pass every arm's ledger to get the cross-arm view")
    rc.add_argument("--goal")
    rc.add_argument("--body-sha", dest="body_sha_now")
    rc.add_argument("--layout-sig", dest="layout_sig")
    rc.add_argument("--mode-ledger")
    rc.add_argument("--limit", type=int, default=SUMMARY_DEFAULT_LIMIT,
                    help="bounded entries per recall/summary bucket (1..64)")
    rc.add_argument("--out", help="write the complete bounded recall JSON")
    rc.add_argument("--receipt-out", help="write a hashed auditable recall receipt")
    rc.add_argument("--trigger", action="append", choices=RECALL_TRIGGERS)
    rc.add_argument("--identity-hash", action="append", metavar="KIND=SHA256")
    rc.add_argument("--recall-events", default=DEFAULT_RECALL_EVENTS)
    q = sub.add_parser("query", help="the raw row filter under recall")
    q.add_argument("--ledger", required=True)
    q.add_argument("--goal")
    q.add_argument("--mode")
    q.add_argument("--layout-sig", dest="layout_sig")
    q.add_argument("--body-sha", dest="body_sha_now")
    q.add_argument("--stale-only", action="store_true", dest="stale_only")
    q.add_argument("--work-kind", choices=WORK_KINDS)
    q.add_argument("--identity-hash", action="append", metavar="KIND=SHA256")
    q.add_argument("--limit", type=int, default=64)
    s = sub.add_parser("summary", help="bounded valid/stale/unfinished/recheck recall handoff")
    s.add_argument("--ledger", action="append", required=True)
    s.add_argument("--body-sha", dest="body_sha_now")
    s.add_argument("--layout-sig", dest="layout_sig")
    s.add_argument("--limit", type=int, default=SUMMARY_DEFAULT_LIMIT)
    s.add_argument("--out", help="write the complete bounded summary JSON")
    dl = sub.add_parser("decision-log", help="write a decision-log pointer; no explanatory prose")
    dl.add_argument("--round", type=int, required=True)
    dl.add_argument("--summary-ref", required=True)
    dl.add_argument("--out", required=True)
    mark = sub.add_parser("mark-measurement",
                          help="declare one valid measurement that must be journaled before transition")
    mark.add_argument("--events", default=DEFAULT_MEASUREMENT_EVENTS)
    mark.add_argument("--measurement-id", required=True)
    mark.add_argument("--work-kind", required=True, choices=WORK_KINDS)
    mark.add_argument("--work-id", required=True)
    mark.add_argument("--evidence-ref", required=True)
    ab = sub.add_parser("abandon-recall",
                        help="settle a recall receipt whose work never happened, with a reason")
    ab.add_argument("--receipt", required=True)
    ab.add_argument("--reason", required=True)
    ab.add_argument("--recall-events", default=DEFAULT_RECALL_EVENTS)
    life = sub.add_parser("lifecycle-check",
                          help="verify declared valid measurements and recall receipts are closed")
    life.add_argument("--ledger", default=DEFAULT_JOURNAL)
    life.add_argument("--measurement-events", default=DEFAULT_MEASUREMENT_EVENTS)
    life.add_argument("--recall-events", default=DEFAULT_RECALL_EVENTS)
    life.add_argument("--strict", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "verify":
        res = verify(a.ledger)
        print(json.dumps(res, indent=2))
        return 1 if (a.strict and not res["ok"]) else 0
    if a.cmd == "recall":
        try:
            identity_now = _parse_named_hashes(a.identity_hash)
            if a.receipt_out:
                issue_recall_receipt(
                    a.ledger, a.receipt_out, a.trigger or ["manual"],
                    body_sha_now=a.body_sha_now, layout_sig=a.layout_sig,
                    identity_now=identity_now, goal=a.goal, mode_ledger=a.mode_ledger,
                    limit=a.limit, event_log=a.recall_events,
                )
            # The receipt is a hash-signed attestation, not the answer: it is squeezed to two
            # entries per bucket and carries no `stale_negatives` or `untried_modes` at all.
            # Those are the buckets the next hypothesis is formed against, and `--receipt-out`
            # is required on exactly the path that forms one, so `--out` and stdout carry the
            # complete index in both cases.
            result = recall(
                a.ledger, a.body_sha_now, a.layout_sig, a.goal, a.mode_ledger, a.limit,
                identity_now=identity_now,
            )
        except (ValueError, Refused) as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        if a.out:
            _atomic_json(a.out, result)
        if a.out or a.receipt_out:
            print(json.dumps(_recall_stdout_view(result, a.out, a.receipt_out),
                             ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "query":
        try:
            identity_now = _parse_named_hashes(a.identity_hash)
            result = query(a.ledger, a.goal, a.layout_sig, a.body_sha_now, a.stale_only,
                           a.mode, a.work_kind, identity_now, a.limit)
        except (Refused, ValueError) as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if a.cmd == "summary":
        try:
            result = summary(a.ledger, a.body_sha_now, a.layout_sig, a.limit)
        except ValueError as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        if a.out:
            _atomic_json(a.out, result)
        print(json.dumps({"schema": ROUND_SUMMARY_SCHEMA, "counts": result["counts"],
                          "out": a.out or None}, ensure_ascii=False, sort_keys=True))
        return 0
    if a.cmd == "decision-log":
        try:
            result = decision_log_ref(a.round, a.summary_ref)
        except ValueError as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        _atomic_json(a.out, result)
        print(json.dumps({"round": result["round"], "summary_ref": result["summary_ref"],
                          "out": a.out}, ensure_ascii=False, sort_keys=True))
        return 0
    if a.cmd == "abandon-recall":
        try:
            result = abandon_recall_receipt(a.recall_events, a.receipt, a.reason)
        except (ValueError, Refused) as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if a.cmd == "mark-measurement":
        try:
            result = mark_measurement(
                a.events, a.measurement_id, a.work_kind, a.work_id, a.evidence_ref
            )
        except ValueError as exc:
            print(f"[round_record] ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if a.cmd == "lifecycle-check":
        result = lifecycle_check(a.ledger, a.measurement_events, a.recall_events)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if a.strict and not result["ok"] else 0

    try:
        rec = build(a)
    except Refused as exc:
        print(f"[round_record] REFUSED: {exc}", file=sys.stderr)
        return 2
    _append_jsonl(a.ledger, rec)
    if rec.get("schema_version") == JOURNAL_SCHEMA_VERSION:
        if rec.get("measurement_event_id"):
            _record_measurement_event(a.measurement_events, rec["measurement_event_id"], rec)
        if rec.get("recall_receipt_ref"):
            receipt = _load_recall_receipt(rec["recall_receipt_ref"])
            _consume_recall_event(a.recall_events, receipt, rec)
        print(f"[round_record] appended {rec['work_kind']} {rec['work_id']} "
              f"sequence={rec['sequence']} -> {a.ledger}")
    else:
        print(f"[round_record] appended round {rec['round']} verdict={rec['verdict']} "
              f"vs {rec['comparator']} -> {a.ledger}")
    return 0


def _selftest() -> int:
    import tempfile

    class A:  # a namespace the same shape argparse produces
        def __init__(self, **kw):
            d = dict(ledger=None, round=1, hypothesis="h", prediction="p", change="c",
                     lever=None, summary_ref=None,
                     comparator="in_batch_control", speedup=None, latency_ms=None, noise_band=None,
                     correctness="ok", verdict="null", confidence=None, enabling_step=False,
                     evidence="e", evidence_layer=None, artifact=None, body=None, tensor=None,
                     goal=None, mode=None, arm_id=None, shape=None, aggregation="unspecified",
                     boundary="unknown", measurement_source=None, measurement_scope=None,
                     sample_count=None, baseline_ref=None, comparator_ref=None)
            d.update(kw)
            self.__dict__.update(d)

    def refused(**kw):
        try:
            build(A(**kw))
        except Refused as exc:
            return str(exc)
        raise AssertionError(f"should have been refused: {kw}")

    # Refusal 1: an unnamed or out-of-enum comparator, and an absent prediction.
    assert "outside" in refused(comparator="whatever")
    assert "--prediction is required" in refused(prediction="")
    # Refusal 2: two units in one row.
    assert "two metric fields" in refused(speedup=1.1, latency_ms=2.0)
    # Refusal 3: a verdict outside the four states -- every one of the nine observed spellings.
    for spelling in ("win", "win-confirmed", "win-candidate", "carried", "rejected", "negative"):
        assert "outside the enum" in refused(verdict=spelling), spelling
    # Refusal 4: kept inside the band, and kept with no band at all.
    assert "without --noise-band" in refused(verdict="kept", speedup=1.02)
    msg = refused(verdict="kept", speedup=1.004, noise_band=0.84)
    assert "+0.400%" in msg and "0.840%" in msg, msg
    # A kept/reverted row with no number at all is the same refusal from the other side.
    assert "with no metric" in refused(verdict="kept")
    assert "with no metric" in refused(verdict="reverted")
    # ...but a `null` round legitimately has none: refuting a hypothesis is a result.
    assert build(A(verdict="null"))["verdict"] == "null"
    legacy_latency = build(A(verdict="null", latency_ms=1.23))["measurement"]
    assert legacy_latency["identity"]["scope"] == "unknown"
    assert legacy_latency["identity"]["source"] == "legacy_latency"
    forged_latency = json.loads(json.dumps(legacy_latency))
    forged_latency["identity"]["source"] = "device_timing"
    assert "device_scope" in {x["code"] for x in validate_measurement(forged_latency)}
    device_measurement = build(A(verdict="null", speedup=1.0, shape=["s1"],
                                 aggregation="geomean", boundary="kernel",
                                 measurement_source="device_timing",
                                 measurement_scope="device", sample_count=20,
                                 baseline_ref="baseline.json",
                                 comparator_ref="champion.json"))["measurement"]
    assert device_measurement["identity"]["source"] == "device_timing"

    # The same round clears once it is honestly labelled, three ways.
    for kw in (dict(verdict="null", speedup=1.004),
               dict(verdict="kept", speedup=1.004, noise_band=0.84, enabling_step=True),
               dict(verdict="kept", speedup=1.05, noise_band=0.84)):
        rec = build(A(**kw))
        assert rec["comparator"] == "in_batch_control"
        assert "latency_ms" not in rec

    # confidence is orthogonal: kept+candidate and kept+confirmed are both legal and distinct.
    assert build(A(verdict="kept", speedup=1.05, noise_band=0.84,
                   confidence="candidate"))["confidence"] == "candidate"
    # ...and 'high' is the spelling the measured bypass used. It is not one of the two states.
    assert "outside" in refused(confidence="high")

    # New writes use the three converged search modes plus non-search round kinds.
    for m in SEARCH_MODES + ROUND_KINDS:
        assert build(A(mode=m))["mode"] == m
    # Historical modes remain readable, but cannot silently become new behaviour.
    assert any(x["rule"] == "legacy_mode"
               for x in row_violations(dict(build(A()), mode="layout")))
    assert "remains readable" in refused(mode="layout")
    assert "outside" in refused(mode="tinker")

    # goal is a key, so it is length-bounded; the reasoning belongs in --hypothesis.
    long_goal = ("raise the matrix-unit fill of this kernel so that the WAIT-dominated "
                 "inter-MFMA bubble is amortised over more matrix work")
    assert "is the KEY" in refused(goal=long_goal)
    assert build(A(goal="raise matrix-unit fill"))["goal"] == "raise matrix-unit fill"

    # evidence_layers takes tokens. A sentence here is what silently disables the mode trigger.
    assert "is not a token" in refused(evidence_layer=["A1 next_free_vgpr 370->400"])
    assert "is not a token" in refused(evidence_layer=["ISA_Census"])
    assert build(A(evidence_layer=["isa_census", "pmc_tcc"]))["evidence_layers"] == \
        ["isa_census", "pmc_tcc"]

    # layout_sig moves without body_sha moving -- the transpose case, which is the whole point.
    d = tempfile.mkdtemp(prefix="round_record_selftest_")
    src = os.path.join(d, "k.py")
    with open(src, "w") as fh:
        fh.write("# kernel\n")
    row = build(A(body=[src], tensor=["a_scale:fp32:row_major"]))
    row2 = build(A(body=[src], tensor=["a_scale:fp32:col_major"]))
    assert row["body_sha"] == row2["body_sha"], "the kernel bytes did not change"
    assert row["layout_sig"] != row2["layout_sig"], "but the layout did, and a verdict dies with it"

    # verify() sees what a per-row writer cannot.
    led = os.path.join(d, "rounds.jsonl")
    with open(led, "w") as fh:
        fh.write(json.dumps(build(A(round=1, ledger=led))) + "\n")
        fh.write(json.dumps(build(A(round=1, ledger=led))) + "\n")   # collision
        fh.write(json.dumps({"round": 5, "verdict": "win"}) + "\n")  # unversioned + drift + gap
    res = verify(led)
    kinds = {f["kind"] for f in res["findings"]}
    assert {"round_number_collision", "unversioned_rows", "verdict_vocabulary_drift",
            "comparator_missing", "round_number_gaps"} <= kinds, kinds
    assert res["ok"] is False

    # THE MEASURED BYPASS, as a regression test. A row that copies schema_version and then breaks a
    # refusal used to read as clean, because the version field was the only thing checked.
    forged = os.path.join(d, "forged.jsonl")
    with open(forged, "w") as fh:
        fh.write(json.dumps(build(A(round=19, ledger=forged, body=[src],
                                    tensor=["k:bf16:row_major"]))) + "\n")
        fh.write(json.dumps({"schema_version": SCHEMA_VERSION, "round": 20, "verdict": "kept",
                             "comparator": "in_batch_control", "prediction": "p",
                             "speedup_vs_comparator": 1.02, "noise_band_pct": 0.6,
                             "confidence": "high"}) + "\n")
    res = verify(forged)
    kinds = {f["kind"] for f in res["findings"]}
    assert "append_refusals_violated" in kinds, kinds
    assert res["ok"] is False, "a forged row must not read as clean"
    assert {x["round"] for x in res["integrity_violations"]} == {20}
    assert {"confidence_enum", "structured_record_extra_fields"} <= {
        x["rule"] for x in res["integrity_violations"]}
    assert res["integrity_violations"][0]["claimed_schema_version"] == SCHEMA_VERSION
    assert "unversioned_rows" not in kinds, "the version was present -- that is the whole point"
    # ...and the same ledger reports what it cannot attribute: round 20 carries a number, no body.
    assert "body_sha_missing" in kinds, kinds

    # A newly-tightened field vocabulary is a SHAPE finding, not a forgery, and it does not fail the
    # ledger. This is the split that keeps the seven forged rows visible instead of buried under the
    # twenty-five rows whose only sin is a goal written before the bound existed.
    old = os.path.join(d, "old.jsonl")
    with open(old, "w") as fh:
        row = build(A(round=1, ledger=old, body=[src], speedup=1.0,
                      tensor=["k:bf16:row_major"], goal="close B2"))
        row["goal"] = "close the B2 lgkmcnt bubble by staging K through LDS one buffer deeper"
        row["mode"] = "diagnose_v2"
        row["evidence_layers"] = ["B2 lgkmcnt 41% -> 12%"]
        fh.write(json.dumps(row) + "\n")
    res = verify(old)
    kinds = {f["kind"] for f in res["findings"]}
    assert "record_shape_violations" in kinds, kinds
    assert "append_refusals_violated" not in kinds, kinds
    assert res["ok"] is True, "a shape finding reconciles a record; it does not void a claim"
    assert set(res["shape_violations"]) == {"mode_enum", "evidence_layer_not_token"}, \
        res["shape_violations"]

    # The three ledger-shape audits, each on the distribution that produced it.
    shape = os.path.join(d, "shape.jsonl")
    with open(shape, "w") as fh:
        for i in range(1, 6):
            fh.write(json.dumps(build(A(round=i, ledger=shape, body=[src], speedup=1.0,
                                        goal=f"close bucket B{i}", mode="climb",
                                        evidence_layer=[f"probe_{i}"]))) + "\n")
    res = verify(shape)
    kinds = {f["kind"] for f in res["findings"]}
    # five once-only tokens over five rounds: every round reads as "lit a new layer" to search_mode.
    assert "evidence_layer_drift" in kinds, kinds
    # five distinct goals over five rounds: recall --goal can join nothing.
    assert "goal_key_inflation" in kinds, kinds
    # nothing declared a layout, so stale_negatives returning 0 means "unknown", not "none".
    assert "layout_sig_absent" in kinds, kinds
    assert res["census"]["search_modes_never_used"] == sorted(set(SEARCH_MODES) - {"climb"})
    assert res["ok"] is True, "all three are warnings -- they shape the record, not the claim"

    # One goal, three spellings: separate keys to `recall`, which is the observed miss.
    dup = os.path.join(d, "dup.jsonl")
    with open(dup, "w") as fh:
        for i, tail in enumerate(("", " per wave", ", measured"), 1):
            fh.write(json.dumps(build(A(round=i, ledger=dup, body=[src], mode="climb",
                                        goal=f"amortise the MFMA bubble over more matrix work"
                                             f"{tail}"))) + "\n")
    near = [x for x in verify(dup)["findings"] if x["kind"] == "goal_near_duplicates"]
    assert near and "amortise the mfma bubble" in near[0]["detail"], near
    # ...while two genuinely different goals that merely open the same way are left alone.
    diff = os.path.join(d, "diff.jsonl")
    with open(diff, "w") as fh:
        for i, g in enumerate(("close the B2 lgkmcnt bubble", "close the C3 dram byte gap"), 1):
            fh.write(json.dumps(build(A(round=i, ledger=diff, body=[src], mode="climb",
                                        goal=g))) + "\n")
    assert not [x for x in verify(diff)["findings"] if x["kind"] == "goal_near_duplicates"]

    # query(): a verdict measured on another body comes back flagged, so retry is a query result.
    hits = query(led, body_sha_now="deadbeef", stale_only=False)
    assert hits["matched"] == 3
    with open(led, "w") as fh:
        fh.write(json.dumps(build(A(round=1, ledger=led, body=[src], goal="coalesce a_scale"))) + "\n")
    assert query(led, goal="coalesce a_scale")["matched"] == 1
    assert query(led, goal="something else")["matched"] == 0
    stale = query(led, body_sha_now="0" * 16, stale_only=True)
    assert stale["matched"] == 1 and stale["rows"][0]["_stale"] is True

    # RECALL, on that observed miss: two tactics, one goal, two different arms' ledgers. An
    # index keyed on the TACTIC shows nothing; keyed on the GOAL they see each other.
    la = os.path.join(d, "armA.jsonl")
    lb = os.path.join(d, "armB.jsonl")
    with open(la, "w") as fh:
        fh.write(json.dumps(build(A(round=44, ledger=la, arm_id="A", verdict="reverted",
                                    speedup=0.99, goal="coalesce the a_scale read",
                                    change="in-kernel tl.reshape + tl.split ladder",
                                    body=[src], tensor=["a_scale:fp32:row_major"]))) + "\n")
    with open(lb, "w") as fh:
        fh.write(json.dumps(build(A(round=63, ledger=lb, arm_id="B", verdict="null",
                                    goal="coalesce the a_scale read",
                                    change="chunked coalesced load, second formulation",
                                    body=[src], tensor=["a_scale:fp32:row_major"]))) + "\n")
    mp = os.path.join(d, "modes.json")
    with open(mp, "w") as fh:
        json.dump({"modes": {"climb": {"state": "open"}, "layout": {"state": "untried"},
                             "branch": {"state": "untried"}}}, fh)

    rc = recall([la, lb], goal="coalesce the a_scale read", mode_ledger=mp)
    assert rc["rows_indexed"] == 2
    assert {h["arm_id"] for h in rc["same_goal"]} == {"A", "B"}, rc["same_goal"]
    assert set(rc["untried_modes"]) == {"layout", "branch"}, rc["untried_modes"]

    # When the answer is on disk, stdout drops only what the file already holds. The three
    # fields the next hypothesis is formed against must survive verbatim, because a host that
    # caps tool output truncates from the end and would otherwise cut exactly those.
    view = _recall_stdout_view(rc, "recall.json", None)
    assert view["same_goal"] == rc["same_goal"], view
    assert view["stale_negatives"] == rc["stale_negatives"], view
    assert view["untried_modes"] == rc["untried_modes"], view
    assert view["recall_written_to"] == "recall.json"
    assert "summary" not in view, "the summary is recoverable from the file"
    assert view["summary_in_file_only"] == sorted(rc["summary"]), view
    assert (len(json.dumps(view, ensure_ascii=False, indent=2))
            < len(json.dumps(rc, ensure_ascii=False, indent=2)))
    # With neither --out nor --receipt-out there is no file to point at, so nothing is dropped.
    assert _recall_stdout_view(rc, None, None)["same_goal"] == rc["same_goal"]

    # A receipt issued for work that then does not happen must have an honest way out. Without an
    # abandon event the only route to a clean close is a forged `consumed`, which is what four of
    # four owners were driven to in one campaign.
    with tempfile.TemporaryDirectory(prefix="round_record_abandon_") as tmp:
        receipt_path = os.path.join(tmp, "receipt.json")
        events = os.path.join(tmp, "recall_events.jsonl")
        issue_recall_receipt([la], receipt_path, ["new_climb_hypothesis"], event_log=events)
        assert lifecycle_check(la, os.path.join(tmp, "m.jsonl"), events)["findings"], \
            "an issued receipt with no outcome must be a finding"
        for bad in ("", "dropped"):
            try:
                abandon_recall_receipt(events, receipt_path, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"abandon accepted reason {bad!r}")
        abandon_recall_receipt(events, receipt_path,
                               "hypothesis dropped before its round; stage closed on budget")
        after = lifecycle_check(la, os.path.join(tmp, "m.jsonl"), events)
        assert not [f for f in after["findings"] if f["kind"] == "recall_receipt_unconsumed"], after
        # The issue is still on the timeline: settling a receipt is not erasing it.
        kinds = [row["event"] for row in _valid_events(events, RECALL_EVENT_SCHEMA)]
        assert kinds == ["issued", "abandoned"], kinds

    # `--receipt-out` must not downgrade what `--out` and stdout carry. The receipt is capped at
    # two entries per bucket and has no stale/untried buckets, so returning it as the answer
    # silently empties the two fields the next hypothesis is formed against.
    with tempfile.TemporaryDirectory(prefix="round_record_recall_") as tmp:
        out_path = os.path.join(tmp, "recall.json")
        receipt_path = os.path.join(tmp, "receipt.json")
        rc_argv = ["recall", "--ledger", la, "--ledger", lb,
                   "--goal", "coalesce the a_scale read", "--trigger", "new_climb_hypothesis",
                   "--out", out_path, "--receipt-out", receipt_path,
                   "--recall-events", os.path.join(tmp, "events.jsonl")]
        assert main(rc_argv) == 0
        with open(out_path) as fh:
            out_doc = json.load(fh)
        with open(receipt_path) as fh:
            receipt_doc = json.load(fh)
        assert receipt_doc["schema"] == RECALL_RECEIPT_SCHEMA, receipt_doc.get("schema")
        assert out_doc.get("schema") != RECALL_RECEIPT_SCHEMA, "--out must not receive the receipt"
        for bucket in ("rows_indexed", "same_goal", "stale_negatives", "untried_modes"):
            assert bucket in out_doc, f"--out lost {bucket} when a receipt was requested"
        assert receipt_doc["entry_limit"] == 2, receipt_doc["entry_limit"]

    # Now the launcher transposes a_scale. The kernel bytes are UNCHANGED, and both prior
    # negatives are surfaced as stale -- so retrying them is a query result, not an act of memory.
    new_layout = parse_layout_sig(["a_scale:fp32:col_major"])
    rc2 = recall([la, lb], body_sha_now=body_sha([src]), layout_sig=new_layout)
    assert len(rc2["stale_negatives"]) == 2, rc2["stale_negatives"]
    assert {s["round"] for s in rc2["stale_negatives"]} == {44, 63}
    # A kept row is not resurfaced as a stale NEGATIVE -- only things that were ruled out are.
    with open(lb, "a") as fh:
        fh.write(json.dumps(build(A(round=64, ledger=lb, arm_id="B", verdict="kept",
                                    speedup=1.05, noise_band=0.8, goal="coalesce the a_scale read",
                                    body=[src], tensor=["a_scale:fp32:row_major"]))) + "\n")
    assert len(recall([la, lb], layout_sig=new_layout)["stale_negatives"]) == 2

    # Schema v3 is a structured result record: write-gate prose is accepted
    # before the edit but cannot leak into the ledger afterwards.
    structured = build(A(round=65, ledger=lb, lever="coalesce_access",
                         summary_ref="summaries/r65.json", artifact=["profile/r65.json"]))
    assert "prediction" not in structured and "hypothesis" not in structured and "change" not in structured
    assert structured["summary_ref"] in structured["artifacts"]
    assert not [x for x in row_violations(structured)
                if x["rule"] == "structured_record_extra_fields"]
    injected = dict(structured, hypothesis="must not be here")
    assert any(x["rule"] == "structured_record_extra_fields" for x in row_violations(injected))
    with open(lb, "a") as fh:
        fh.write(json.dumps(structured) + "\n")
    handoff = summary([la, lb], body_sha_now=body_sha([src]), layout_sig=new_layout, limit=1)
    assert set(handoff["counts"]) == set(SUMMARY_BUCKETS)
    assert handoff["counts"]["stale"] == 3 and handoff["truncated"]["stale"] == 2, handoff
    pointer = decision_log_ref(65, "summaries/r65.json")
    assert set(pointer) == {"schema", "round", "summary_ref"}, pointer

    import shutil
    shutil.rmtree(d, ignore_errors=True)
    print("[round_record] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
