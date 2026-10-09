#!/usr/bin/env python3
"""Check a `final_report.json` against the shape its READERS actually read.

WHY THIS EXISTS. `final_report.json` is the one artifact that crosses every pack boundary: it is
what a fleet collects, what a dashboard tabulates, and in the captain case the ONLY thing a
supervisor is allowed to read (walking the owner's worker dirs is forbidden by both orchestration
skills). Every other cross-boundary artifact in this repo has a schema behind it. This one had
prose in seven copies of `references/method/records.md` and nothing that compared a written report against
any of them, so two reports produced by the same skill on the same day could share about half their
keys, spell the same verdict in different cases, and both omit numbers a reader is documented to
read -- with the data sitting in the run's own manifest, just never lifted into the report.

WHY IT IS NOT IN `schema.py`. That file is vendor/amd's JSON round contract and ships only in the
three GATED packs. Five packs write a `final_report.json` and four of those ship no `schema.py` at
all, so a schema hosted there would have covered the reports least at risk. This lives in core/ and
therefore ships, byte-identical, in all seven.

WHAT IT IS NOT. It is not a gate. Default exit is 0 and always will be: a report that is thin is
still evidence, and refusing to read it is worse than reading it with its gaps named. `--strict`
exits 1 on error-severity findings, for a caller that has decided to enforce.

WHAT IT CHECKS. Only fields a documented reader reads, in three groups:

  core      every report: the verdict (under either accepted key), status, caveats, deferred
  close     the closure contract: `close_audit` + `verified_by`, `skeptic_verdict` + `skeptic_items`
  captain   a report that names a champion additionally owes the numbers the fleet's accept step
            reads out of it (`kernel-opt-fleet` reference.md 8.B)

IN GEAK this is optional bookkeeping a deep_engineer may use for its records, not a run mode.
Role ids such as `captain` / `deep` / `skeptic` are upstream record-schema identifiers (`deep` = the
deep_engineer, pass `--role deep`); nothing spawns them, and final arbitration in GEAK is Director's.

Pack-specific keys are NOT checked. Each pack legitimately adds its own, and a shared linter that
insisted on one pack's spelling would be wrong in the other six.

Usage:
  report_lint.py [--report final_report.json] [--owner auto|captain|worker|canonical-base]
  report_lint.py --report r.json --json lint.json --strict
  report_lint.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from canonical_record import (
    ARM_RESULT_SCHEMA,
    ARBITRATIONS as CANONICAL_ARBITRATIONS,
    FINAL_REPORT_SCHEMA,
    OUTCOMES as CANONICAL_OUTCOMES,
    RUN_STATUSES as CANONICAL_STATUSES,
    project_final_report,
    validate_arm_result,
    validate_final_report,
)

# TWO FIELDS THAT LOOK LIKE ONE. `result` is what the optimization achieved; `verdict` is whether
# the closing party accepted the run. They answer different questions and neither substitutes for
# the other -- a run can be accepted and have achieved nothing, and a real win can be rejected for a
# broken bundle. They get conflated because both are one lower-case word at the top of the same
# file, and a report carrying only `verdict: accept` looks complete while stating no outcome at all.
#
# One superset enum over the two negative spellings. The Triton family closes a failed deep dig as
# `negative_revert_plain` (there is a plain tier to revert TO); the packs with no plain tier close it
# as `negative_keep_baseline`. Both mean "measured, did not beat the comparator, comparator kept",
# which is a fully recorded outcome and not a failure to run.
VERDICTS = CANONICAL_OUTCOMES
ARBITRATIONS = CANONICAL_ARBITRATIONS

STATUSES = CANONICAL_STATUSES
UNWAIVABLE_KINDS = frozenset({
    "canonical_entry_missing", "final_report_missing", "comparator_missing", "profile_missing",
    "resweep_missing", "sweep_unresolved", "arm_local_sweep_missing",
    "post_merge_sweep_missing", "handoff_unresolved", "arm_gate_missing", "identity_mismatch",
    "forbidden_deep_branch", "unauthorized_resweep",
    "comparator_not_pinned", "deferred_never_closed", "arm_roster_incomplete",
    "roster_undeclared", "path_coverage_missing", "anchor_unverified",
})

# Was the closure checked by the party that produced it, or by an independent one? A self-check and
# an independent check leave identical artifacts, so without this field a reader cannot tell a
# cross-examined result from an unexamined one -- the same failure `stay_plain_basis`
# (configured|measured) already fixes for ceiling claims.
VERIFIED_BY = ("self", "fleet")

# `not_run` is a legitimate value and deliberately distinct from an absent field: a run that never
# called the skeptic differs from a run that called it and dropped the answer.
SKEPTIC_VERDICTS = ("credible", "weak", "contradicted", "not_run")

# What the author DID about a challenge. A skeptic verdict with no disposition is where an
# advisory review quietly becomes a formality: `contradicted` and `credible` look the same in a
# report that records only that a skeptic ran.
DISPOSITIONS = ("accepted", "rebutted")

# Read by kernel-opt-fleet reference.md 8.B out of a captain owner's report, and by nothing else --
# so a captain report missing one of these is not thin, it is unreadable by its own consumer.
CAPTAIN_NUMERIC = ("default_ms", "champion_ms", "best_ms", "vs_champion")

# `measured_at_parity` is the third value, and it exists because the enum was too small to describe
# a real close: a deep-dig arm that RAN, cleared its gates, and returned parity while the binding
# resource was one the deep tier does not control (register pressure, where the tier's exclusive
# control is over LDS). That argument is language-independent, so parity settles the question rather
# than inviting a retry -- but with only `configured|measured` available, the run that hit it had to
# file a `stay_plain_basis` that contradicted its own `winning_stage`.
STAY_PLAIN_BASIS = ("configured", "measured", "measured_at_parity")

# Where a retired claim CAME FROM, because the repair differs. `brief` means the dispatch generator
# is still emitting it and will emit it again to the next worker; `arm` and `own` are local.
# `skeptic` is the fourth because a refuted skeptic RECOMMENDATION has its own repair: nothing has to
# stop emitting it, but the closure argument that cited it has to be revisited.
KNOWN_WRONG_SOURCES = ("brief", "arm", "own", "skeptic")

# WHO retired it, which is the ORTHOGONAL axis `source` was carrying by force. This is the same
# defect `round_record.py` fixed by splitting `verdict` from `confidence`, and it shows up the same
# way -- as spellings. Measured on two reports from one fleet run:
#
#   fwd_attn      19 entries, ALL in-enum. One worker, one captain, one skeptic: there is only one
#                 "own", so three values are enough.
#   gemm_triton   20 entries, 16 out-of-enum -- and every one of the 16 is `own` with a parenthetical
#                 naming an ACTOR: `own (captain)` x3, `own (captain, first read)`,
#                 `own (front-end worker)` x2, `own (gluon worker)` x4, `own (arm <arm-name>)`,
#                 plus a bare `skeptic`. That run was two-stage, so "own" had five referents.
#
# The actors are not invented here: they are this pack's role registry -- the run captain, the
# front-end worker, its best-of-N arms, the deep-dig worker, and the closure skeptic.
KNOWN_WRONG_RETIRED_BY = ("captain", "frontend", "arm", "deepdig", "skeptic")

# Three of those 16 said something `source` genuinely cannot: `own (captain, written into the
# worker's brief)`, `own (captain, carried from the front end's skeptic into the gluon brief)`,
# `own (front-end profile, carried into the gluon brief by the captain)`. A claim's ORIGIN and
# whether it PROPAGATED into a brief are independent, and it is the propagation that triggers the
# expensive repair -- so the writer who owns the brief cannot express it by choosing between `own`
# and `brief`. `emitted_in_brief` carries it.
_OWN_WITH_ACTOR = re.compile(r"^\s*(brief|arm|own|skeptic)\s*\(([^)]*)\)\s*$", re.I)

# What this file says it is. Checked at note severity rather than error because a report is readable
# without it and refusing one over a label would be the wrong trade -- but two runs of the same skill
# on the same day have already spelled this two ways, and a consumer that has to guess which spelling
# it is holding is one spelling away from reading the wrong keys out of the right file.
SCHEMA_ID = FINAL_REPORT_SCHEMA
# The spellings a real close has written for the SAME schema. `kernel_opt_run.final_report` is what
# `kernel-opt-run`'s own captains produce -- it names the skill that wrote the file, which is a
# reasonable thing to think the field is for -- and treating it as unknown put a finding on the
# reports that were otherwise the most complete. Accepted here so the note fires on a genuinely
# unrecognised name rather than on a synonym.
SCHEMA_ALIASES = frozenset({SCHEMA_ID, "kernel_opt.final_report/1", "kernel_opt.final_report/2",
                            "kernel_opt_run.final_report",
                            "kernel_opt_run.final_report/1", "kernel_opt.final_report",
                            "kernel_opt.final_report/1"})

# ---------------------------------------------------------------- arm_result.json
#
# The BRANCH arm roll-up. It had NO validator at all -- this file only ever read `final_report.json`
# -- while three `*-bestof` agents each specify it in prose. The measured cost of that, on one run:
#
#   `gate_stage`  specified as exactly `A` / `B` / `C`, and three arms wrote `branch_arm_closed`,
#                 `closed_at_measured_wall`, `branch_arm_complete`. Three free-text values, none in
#                 the enum, so "was this arm rankable" is unanswerable by query -- which is the one
#                 question the field exists for, because an arm that never cleared Gate A may not
#                 be ranked.
#   `verdict`     specified NOWHERE across all three agent files, while each one's Return block
#                 carries `result: supported|not_supported|inconclusive` and the kill rule refers to
#                 `negative`/`revert`. Three vocabularies for one question, in one file. The run then
#                 wrote `kept` / `supported` / `supported`.
#
# So the enums live HERE, and the agent files point at this validator instead of restating them --
# the same ownership rule the mode vocabulary and `known_wrong` got, for the same reason: a spec
# duplicated in prose drifts, and the copy a worker reads is the one that wins.
ARM_VERDICTS = ("supported", "not_supported", "inconclusive")
ARM_COMPLETION = ("complete", "truncated", "crashed")
ARM_DIRECTION_CLASS = ("structural", "knob")

# `A|B|C` is how far the arm got. Not rankable below B, because Gate A is build + own sweep + one
# re-profile on its own body.
ARM_GATE_STAGES = ("a", "b", "c")

# Gate B promotes on the MECHANISM, not the clock. `implementation-bound` promotes: it is the state
# the biggest mechanism of one recorded run was in when its first round scored null at +0.09%.
ARM_MECHANISM_STATUS = ("refuted", "implementation-bound", "confirmed")

ARM_CONFIG_BASIS_MODES = ("own_sweep", "closed_axis", "partial", "inherited")

# Every `*-bestof` agent declares these five. Everything else is per-pack and is checked only when
# present -- a gate that fires on a field the worker's own instructions never mention is a gate
# landing where its justifying evidence cannot reach it.
ARM_REQUIRED = ("lever", "direction_class", "iters_used", "verdict", "completion")

# The comparator's name differs by pack and the difference is real: the Triton family scores against
# tuned plain Triton, the others against their anchor. Either spelling satisfies "this arm reported
# a number".
ARM_GEOMEAN_KEYS = ("best_geomean_vs_plain", "best_geomean_vs_anchor",
                    "geomean_vs_plain", "geomean_vs_anchor")


def _add(out, severity, field, detail):
    out.append({"severity": severity, "field": field, "detail": detail})


def _scalar(doc, key, default=""):
    """The VALUE of an enum-ish field, whether it was written bare or as `{value, ...}`.

    Both spellings are in use and the object form carries more (the reason travels with the
    arbitration), so every reader of these fields goes through here rather than indexing the dict."""
    v = doc.get(key, default)
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    return v if isinstance(v, str) else default


# The headline numbers, and the SPELLINGS a real close has written for each of them.
#
# WHY THIS TABLE IS HERE. Four owners on one track produced four structurally different reports and
# all four carried the required content -- a headline as a scalar in one and a per-case mapping in
# another, a budget nested under its own key in one and flat in the next, a served table under two
# names, review items under two more. A supervisor then probed those files with GUESSED key names,
# got nulls, and read them as reports missing required fields; it nearly rejected clean work over
# its own bad probe, twice. That is the expensive failure mode, worse than a thin report, because it
# throws away work that was done.
#
# So the spellings are OWNED here and every consumer goes through the readers below. The point is
# not to bless any one spelling -- it is that "the report does not carry this" and "I looked under
# the wrong name" must stop producing the same answer.
HEADLINE_ALIASES = {
    "default_ms": ("default_ms", "baseline_ms", "comparator_ms"),
    "champion_ms": ("champion_ms", "current_ms"),
    "best_ms": ("best_ms", "final_ms"),
    "vs_champion": ("vs_champion", "result_vs_champion_ms", "final_vs_champion"),
}
SERVED_ROWS_ALIASES = ("served_range", "served_range_table", "served_shape_table", "served_envelope")
# A headline may sit at the top level or inside the block that states the headline. Both are in
# use and neither is wrong; what is wrong is a consumer that only knows one of them.
HEADLINE_CONTAINERS = ("headline", "result", "measurement", "verdict")
BUDGET_CONTAINERS = ("budget", "rounds", "rounds_spent", "budget_and_ceiling", "time")
BUDGET_ALIASES = {
    "rounds_used": ("rounds_used", "spent", "rounds_spent", "used"),
    "round_budget": ("round_budget", "budget", "rounds", "cap"),
}


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _headline_scopes(doc):
    return [("", doc)] + [(f"{c}.", doc[c]) for c in HEADLINE_CONTAINERS
                          if isinstance(doc.get(c), dict)]


def declared_primary_case(doc):
    """Which served case the headline is quoted against, when the report says so.

    Two routes, both from DECLARED values and neither from prose. An explicit `primary_case` wins.
    Failing that, the canonical measurement record carries the one ranked number in `value`; if that
    number matches exactly one entry of the per-case headline mapping, then the report has named its
    primary case by arithmetic -- two of its own fields agreeing -- and that is a derivation rather
    than a guess. Ambiguity (no match, or several) resolves to None, so this fails closed.
    """
    for _prefix, scope in _headline_scopes(doc):
        p = _scalar(scope, "primary_case")
        if isinstance(p, str) and p:
            return p
    measurement = doc.get("measurement")
    ranked = _num(measurement.get("value")) if isinstance(measurement, dict) else None
    if ranked is None:
        return None
    for _prefix, scope in _headline_scopes(doc):
        for name in HEADLINE_ALIASES["champion_ms"]:
            cand = scope.get(name)
            if not isinstance(cand, dict):
                continue
            hits = [k for k, x in cand.items() if _num(x) == ranked]
            if len(hits) == 1:
                return hits[0]
    return None


def read_headline_number(doc, key):
    """One headline number, however this report spelled it. Returns (value, how, problem).

    `how` names the route taken so a caller can say WHERE it read the number, and `problem` is a
    sentence for the caller to quote when the number is genuinely not resolvable. A caller that
    treats an unresolvable number as 0 or null is the defect this exists to remove: it makes an
    unreadable report indistinguishable from a report of zero.
    """
    names = HEADLINE_ALIASES.get(key, (key,))
    scopes = _headline_scopes(doc)
    primary_declared = declared_primary_case(doc)
    problem = None
    for prefix, scope in scopes:
        for name in names:
            if name not in scope:
                continue
            v, where = scope[name], f"{prefix}{name}"
            direct = _num(v)
            if direct is not None:
                return direct, where, None
            if isinstance(v, dict):
                inner = _num(v.get("value"))
                if inner is not None:
                    return inner, f"{where}.value", None
                # A PER-CASE MAPPING is a richer spelling, not a broken one -- but only if the
                # report says which case the headline is quoted against. `primary_case` is looked
                # for beside the number, beside the block holding it, and at the top level, because
                # all three are where an author naturally puts it. Unresolved it stays a problem:
                # picking one would invent the headline.
                cases = {k: _num(x) for k, x in v.items() if _num(x) is not None}
                primary = next((p for p in (v.get("primary_case"), scope.get("primary_case"),
                                            primary_declared)
                                if isinstance(p, str) and p in cases), None)
                if primary:
                    return cases[primary], f"{where}[{primary}] (primary_case)", None
                if cases:
                    problem = problem or (
                        f"`{where}` carries {len(cases)} per-case values ({sorted(cases)[:6]}) and "
                        f"which one the headline is quoted against is unstated: no `primary_case` "
                        f"beside it, beside its block or at the top level, and the canonical "
                        f"`measurement.value` does not single one out either. Put the headline in "
                        f"`value` or declare `primary_case`")
                    continue
            problem = problem or f"`{where}` is present but is not a number ({type(v).__name__})"
    return None, None, problem or (f"none of {list(names)} is present at the top level or under "
                                   f"{list(HEADLINE_CONTAINERS)}; the headline is quoted against "
                                   f"this")


def read_served_rows(doc):
    """The served-range rows, under whichever of the accepted names this report used."""
    for name in SERVED_ROWS_ALIASES:
        v = doc.get(name)
        if isinstance(v, list) and v:
            return v, name
        if isinstance(v, dict):
            rows = v.get("rows") or v.get("served_range") or v.get("cases")
            if isinstance(rows, list) and rows:
                return rows, f"{name}.rows"
            # A TABLE KEYED BY CASE is the third spelling in use: one entry per served
            # shape, the shape's own label as the key. The case label is injected so a caller can
            # iterate rows uniformly without knowing how this report was written -- and without
            # this file knowing anything about what a case is called on any given harness.
            keyed = [dict(row, case=row.get("case", label))
                     for label, row in v.items() if isinstance(row, dict)]
            if keyed:
                return keyed, f"{name}{{by case}}"
    return None, None


def read_budget(doc):
    """(rounds_used, round_budget), looked for flat and inside each accepted container."""
    scopes = [doc] + [doc[c] for c in BUDGET_CONTAINERS if isinstance(doc.get(c), dict)]
    out = {}
    for field, names in BUDGET_ALIASES.items():
        for scope in scopes:
            hit = next((_num(scope[n]) for n in names if _num(scope.get(n)) is not None), None)
            if hit is not None:
                out[field] = hit
                break
    return out.get("rounds_used"), out.get("round_budget")


def _enum(out, doc, key, allowed, severity="error", required=True):
    if key not in doc:
        if required:
            _add(out, severity, key, f"absent; expected one of {list(allowed)}")
        return None
    val = doc[key]
    # A RICH OBJECT IS A LEGAL SPELLING OF AN ENUM FIELD, and refusing it cost a whole report. One
    # measured close wrote `result: {"value": "win", "default_ms": ..., "champion_ms": ...}` and
    # `verdict: {"value": "accept", "reason": "..."}` -- strictly MORE information than the bare
    # string, since the reason travels with the arbitration instead of in a sibling key nobody has to
    # fill. This validator called both "expected a string, found dict", and the report that had
    # written the most careful version of the field was the one that failed. So: read `.value` and
    # check that; anything else in the object is the author's, and none of this validator's business.
    if isinstance(val, dict) and "value" in val:
        val = val["value"]
    if not isinstance(val, str):
        _add(out, "error", key,
             f"expected a string, or an object carrying `value`, found {type(val).__name__}")
        return None
    if val.lower() != val:
        # Case is not cosmetic here: every documented reader compares against a lowercase
        # literal, so an upper-cased verdict silently matches nothing and reads as absent.
        _add(out, "error", key, f"{val!r} is not lower-case; readers compare against "
                                f"lower-case literals, so this matches no branch")
    if val.lower() not in allowed:
        _add(out, severity, key, f"{val!r} is not one of {list(allowed)}")
    return val.lower()


def _list_of_objects(out, doc, key, required_fields=(), required=True, severity="error"):
    if key not in doc:
        if required:
            _add(out, severity, key, "absent; expected a list (empty is a valid, and different, answer)")
        return
    val = doc[key]
    if not isinstance(val, list):
        _add(out, severity, key, f"expected a list, found {type(val).__name__}")
        return
    for i, item in enumerate(val):
        if not isinstance(item, dict):
            _add(out, severity, f"{key}[{i}]", f"expected an object, found {type(item).__name__}")
            continue
        for f in required_fields:
            if f not in item or item[f] in (None, ""):
                _add(out, severity, f"{key}[{i}].{f}", "absent or empty")


def _is_arm_shape(doc) -> bool:
    """An arm roll-up is a different document, not a thin final_report.

    Detected from content for the same reason the captain shape is: the caller is often the party
    being checked. `lever` plus any of the arm-only fields is unambiguous -- no final_report carries
    `lever`."""
    return "lever" in doc and any(k in doc for k in
                                  ("direction_class", "iters_used", "completion", "gate_stage",
                                   *ARM_GEOMEAN_KEYS))


def _lint_arm(out, doc):
    """The BRANCH arm roll-up. Enum drift first, then the cross-field rules that carry the gates."""
    for k in ARM_REQUIRED:
        if k not in doc or doc[k] in (None, ""):
            _add(out, "error", k, f"absent; every *-bestof agent declares it in the roll-up")
    _enum(out, doc, "direction_class", ARM_DIRECTION_CLASS, required=False)
    verdict = _enum(out, doc, "verdict", ARM_VERDICTS, required=False)
    completion = _enum(out, doc, "completion", ARM_COMPLETION, required=False)
    # `gate_stage` is checked here rather than through `_enum`, because it is the one field in this
    # tree whose documented form is UPPER-case: it is a single-letter stage label, not a lower-case
    # word literal, so `_enum`'s case rule would reject the spelling the agents actually specify.
    gate = None
    if "gate_stage" in doc:
        raw = doc["gate_stage"]
        if not isinstance(raw, str):
            _add(out, "error", "gate_stage", f"expected a string, found {type(raw).__name__}")
        elif raw.strip().lower() not in ARM_GATE_STAGES:
            _add(out, "error", "gate_stage",
                 f"{raw!r} is not one of ['A', 'B', 'C']. It is a LETTER, not a sentence: one "
                 f"measured run wrote 'branch_arm_closed' / 'closed_at_measured_wall' / "
                 f"'branch_arm_complete' across three arms, so 'was this arm rankable' -- the one "
                 f"question the field exists for -- became unanswerable by query")
        else:
            gate = raw.strip().lower()
    mech = _enum(out, doc, "mechanism_status", ARM_MECHANISM_STATUS, required=False)

    if isinstance(doc.get("iters_used"), bool) or not isinstance(
            doc.get("iters_used", 0), (int, float)):
        _add(out, "error", "iters_used", f"expected a number, found "
                                         f"{type(doc.get('iters_used')).__name__}")

    geo_key = next((k for k in ARM_GEOMEAN_KEYS if doc.get(k) is not None), None)

    # `completion` says whether the arm FINISHED; `verdict` says what a finished arm FOUND. A
    # truncated arm that also reports a number is read by the parent as a measured no-win, and the
    # fan-out then compares one fewer arm than it reports.
    if completion and completion != "complete" and geo_key:
        _add(out, "error", geo_key,
             f"completion={completion!r} with a number in `{geo_key}`. An arm that did not finish "
             f"must leave the geomean null: only `complete` licenses the parent reading it as a "
             f"result, and a truncated arm carrying a number is indistinguishable from one that "
             f"finished and found nothing")

    # Gate A is what makes an arm RANKABLE, and `config_basis` is the declared half of it.
    cb = doc.get("config_basis")
    if gate in ("b", "c") and not isinstance(cb, dict):
        _add(out, "error", "config_basis",
             f"gate_stage={gate!r} claims Gate A is cleared, and Gate A is build + this arm's OWN "
             f"config sweep + one re-profile. Without `config_basis` there is nothing on disk "
             f"separating an arm that re-swept from one that reused the parent's pin, and the two "
             f"report the same single number")
    if isinstance(cb, dict):
        _enum(out, cb, "mode", ARM_CONFIG_BASIS_MODES, required=False)
        if not str(cb.get("objective") or "").strip():
            _add(out, "note", "config_basis.objective",
                 "absent; it must be the PARENT's objective -- a grid ranked on one case and quoted "
                 "against a multi-case geomean ranks a different question")
        if str(cb.get("mode") or "").lower() == "own_sweep" and not str(
                cb.get("sweep_ref") or "").strip():
            _add(out, "error", "config_basis.sweep_ref",
                 "mode=own_sweep owes a sweep_ref inside this arm's own dir; pointing at the "
                 "parent's record is not this arm's sweep")
    if gate == "a" and geo_key:
        _add(out, "note", geo_key,
             f"gate_stage='a' means this arm is NOT rankable, so the parent may not use "
             f"`{geo_key}` in the CONVERGE geomean; it is a reading, not a ranking")

    # A kill owes a mechanism. A slower number might be measuring the config, not the structure.
    kills = mech == "refuted" or verdict == "not_supported"
    if kills and not doc.get("kill_evidence"):
        _add(out, "error", "kill_evidence",
             f"this arm kills its own direction (mechanism_status={mech!r}, verdict={verdict!r}) "
             f"and owes kill_evidence from its OWN measurements. Eleven points plus a scaling "
             f"ladder is a real kill; two hundred points and no mechanism is not")
    # `mechanism_status` is about the HYPOTHESIS; `verdict` is about what the arm DELIVERED. Refuted
    # plus supported is a legal and informative pair -- the mechanism was not there and the arm won
    # on something else its profile found -- and it is worth naming, because the one run that hit it
    # had nowhere to put the distinction and wrote a 76-character sentence into `verdict`:
    # 'mechanism_refuted_but_arm_wins_on_a_different_lever_found_by_the_same_profile'.
    if mech == "refuted" and verdict == "supported" and not doc.get("lever_stack"):
        _add(out, "note", "lever_stack",
             "mechanism_status=refuted with verdict=supported says the hypothesis was wrong and the "
             "arm delivered anyway -- a real and useful pair, but the lever that DID pay has to be "
             "named here, or the parent inherits a win it cannot attribute")

    if doc.get("not_applicable") and not isinstance(doc.get("na_evidence"), dict):
        _add(out, "error", "na_evidence",
             "not_applicable=true owes na_evidence.{profile_ref|bytes_model|metric_delta} from THIS "
             "arm's own profile -- a reasoned N/A is never a sibling-arm inference")

    # Gate C: the depth default, and the one rule that gives it teeth.
    db = doc.get("depth_basis")
    if db is not None and not isinstance(db, dict):
        _add(out, "error", "depth_basis", f"expected an object, found {type(db).__name__}")
    elif isinstance(db, dict):
        used, prot = db.get("used"), db.get("protected_to")
        nums = {k: v for k, v in (("allocated", db.get("allocated")), ("used", used),
                                  ("protected_to", prot)) if v is not None}
        for k, v in nums.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                _add(out, "error", f"depth_basis.{k}",
                     f"expected a number, found {type(v).__name__}")
        ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nums.values())
        if ok and used is not None and prot is not None:
            if used > prot and not str(db.get("extension_granted_by") or "").strip():
                _add(out, "error", "depth_basis.extension_granted_by",
                     f"used={used} is past protected_to={prot}, and depth past the protection floor "
                     f"is the PARENT's grant rather than the arm's choice. Name who granted it")
            # The protection floor exists so a promoted arm that is still moving its own price
            # estimate cannot be cut. Measured: the only arm that produced +11.5% re-priced itself
            # 15x at ITS ROUND 12.
            if (used < prot and completion == "complete" and mech != "refuted"
                    and doc.get("reprice_events")):
                _add(out, "error", "depth_basis.used",
                     f"used={used} is below protected_to={prot} on an arm that was NOT refuted at "
                     f"Gate B and still has reprice_events. That is the cut the protection floor "
                     f"exists to prevent: an arm that is winless and still re-pricing is working. "
                     f"Either record the Gate-B refutation or restore the depth")
    elif isinstance(doc.get("iters_used"), (int, float)) and not isinstance(
            doc.get("iters_used"), bool):
        _add(out, "note", "depth_basis",
             "absent; iters_used says how many rounds happened and depth_basis says what was "
             "ALLOWED. Without the second, an arm cut early and an arm that finished early leave "
             "the same record, and the Gate C floor cannot be checked at all")


def _is_captain_shape(doc) -> bool:
    """A report that names a champion is read by the captain-owner branch of the accept step.

    Detected from the report rather than from a flag the caller passes: the caller is often the
    party being checked, and a shape assertion it supplies is the one input a thin report would
    get wrong in exactly the direction that hides the gap."""
    return bool(doc.get("champion_ref") or doc.get("winning_stage") or doc.get("served_range"))


def _lint_waivers(out, doc, ca):
    """Every high finding the audit raised must be answered by a WAIVER, one for one.

    THE MEASURED FAILURE. On one campaign four kernels closed. `close_audit --strict` exited 1 on
    three of them -- 6, 2 and 1 high findings -- and all three were accepted anyway, each with a paragraph
    in the report saying the findings had been investigated and were tool false positives. Two of
    those judgements were correct. None of them was CHECKABLE: the prose named no finding, cited no
    file, and no consumer could distinguish a disposed finding from an ignored one. The exit code had
    already been earned and spent.

    A waiver is that same claim in a shape a reader can act on: which finding (check + kind), why, and
    where the evidence is. It does not make the finding go away -- it stays in the audit output -- and
    it is not a rubber stamp: `close_audit --waivers` only excludes an entry that carries BOTH a
    reason and an evidence path.

    Deliberately not an `error` when `high_severity_count` is 0 and `waivers` is absent: a clean close
    owes nothing here, and the common case must stay quiet.
    """
    n_high = ca.get("high_severity_count")
    if not isinstance(n_high, int) or isinstance(n_high, bool) or n_high <= 0:
        return
    wv = doc.get("waivers")
    if wv is None:
        _add(out, "error", "waivers",
             f"close_audit reported {n_high} HIGH finding(s) and this report carries no `waivers[]`. "
             f"A high finding contradicts something this run concluded, so accepting the run means "
             f"answering each one: `waivers: [{{check, kind, why, evidence_ref}}]`. Prose saying the "
             f"findings were investigated is what this replaces -- it names no finding and cites no "
             f"file, so nothing downstream can check it or re-open it")
        return
    if not isinstance(wv, list):
        _add(out, "error", "waivers", f"expected a list, found {type(wv).__name__}")
        return
    ok = 0
    for i, w in enumerate(wv):
        if not isinstance(w, dict):
            _add(out, "error", f"waivers[{i}]", f"expected an object, found {type(w).__name__}")
            continue
        missing = [f for f in ("check", "kind", "why", "evidence_ref")
                   if not str(w.get(f) or "").strip()]
        if missing:
            _add(out, "error", f"waivers[{i}]",
                 f"missing {missing}; a waiver without a `check`/`kind` matches no finding, and one "
                 f"without an `evidence_ref` is the paragraph again in a field. "
                 f"close_audit --waivers ignores it")
            continue
        if str(w.get("kind") or "") in UNWAIVABLE_KINDS:
            _add(out, "error", f"waivers[{i}]",
                 f"{w['kind']!r} is a non-waivable finalization failure; produce the missing "
                 "canonical entry, comparator, profile, resweep, or arm-gate evidence instead")
            continue
        ok += 1
    if ok < n_high:
        _add(out, "error", "waivers",
             f"{ok} usable waiver(s) for {n_high} HIGH finding(s) -- {n_high - ok} unanswered. Each "
             f"high finding needs its own entry naming the finding and pointing at the evidence; a "
             f"single blanket waiver does not cover a set")


def lint(doc, owner="auto", require_canonical=False):
    out: list = []
    if not isinstance(doc, dict):
        _add(out, "error", "<root>", f"expected a JSON object, found {type(doc).__name__}")
        return {"findings": out, "owner_shape": None, "verdict": None}

    # An arm roll-up is a different document with a different owner, so it branches before any of
    # the final_report checks -- running those against it would report a dozen absent fields it was
    # never supposed to carry, which is how a validator gets ignored.
    if owner == "arm" or (owner == "auto" and _is_arm_shape(doc)):
        if doc.get("schema") == ARM_RESULT_SCHEMA:
            for finding in validate_arm_result(doc):
                _add(out, finding["severity"], finding["field"], finding["detail"])
        else:
            _add(out, "error", "schema",
                 f"legacy arm roll-ups are not rankable inputs. Adapt them with "
                 f"`canonical_record.py write-arm-result` before return/convergence; "
                 f"the only ArmResult schema is {ARM_RESULT_SCHEMA!r}.")
        out.sort(key=lambda f: (f["severity"] != "error", f["field"], f["detail"]))
        return {"findings": out, "owner_shape": "arm", "result": doc.get("verdict"),
                "verdict": None, "errors": sum(1 for f in out if f["severity"] == "error")}

    # Every final-report reader sees exactly one shape.  Historical spelling tolerance belongs in
    # canonical_record's projector, not independently in each consumer.
    doc = project_final_report(doc)
    integrity = doc.get("integrity") or {}
    source_schema = integrity.get("source_schema")
    for finding in integrity.get("findings") or []:
        _add(out, finding.get("severity", "error"), finding.get("field", "integrity"),
             finding.get("detail", finding.get("code", "projection failed")))
    if integrity.get("legacy_unverified"):
        _add(out, "note", "integrity.legacy_unverified",
             "historical report paths were normalized to typed references but remain unverified")
    if str(source_schema or "") not in SCHEMA_ALIASES:
        _add(out, "note", "schema",
             f"{source_schema!r}; the name this schema goes by is {SCHEMA_ID!r} (also accepted: "
             f"{sorted(SCHEMA_ALIASES - {SCHEMA_ID})}), and a consumer selecting behaviour on this "
             f"string will not recognise another spelling")

    # `canonical-base` is the portable producer contract: canonical_record.py owns identity and
    # lifecycle fields, while its caller may add report-specific prose/reader fields through
    # --extra-fields.  A base is usable evidence, but must not pretend its absent prose was emitted.
    canonical_base = owner == "canonical-base"

    # --- core: final reports use four independent domains.  Legacy result/verdict
    # records remain readable unless callers choose strict canonical enforcement.
    canonical_fields = source_schema == FINAL_REPORT_SCHEMA
    if canonical_fields:
        shape_hint = owner == "captain" or (owner == "auto" and _is_captain_shape(doc))
        for finding in validate_final_report(doc, captain=shape_hint):
            _add(out, finding["severity"], finding["field"], finding["detail"])
        result = _scalar(doc, "outcome").lower()
        arb = _scalar(doc, "arbitration").lower()
        if "result" in doc and _scalar(doc, "result").lower() != result:
            _add(out, "error", "result", "legacy result conflicts with canonical outcome")
        if "verdict" in doc and _scalar(doc, "verdict").lower() != arb:
            _add(out, "error", "verdict", "legacy verdict conflicts with canonical arbitration")
    else:
        if require_canonical:
            _add(out, "error", "canonical_ref",
                 "strict lint requires canonical final-report fields and canonical_ref")
        result = _enum(out, doc, "result", VERDICTS)
        arb = _enum(out, doc, "verdict", ARBITRATIONS, required=False)
    # The confusion itself, caught in both directions: a `verdict` holding an outcome means the run
    # never recorded whether it was accepted, and a `result` holding an arbitration means it never
    # recorded what it achieved. Either way the missing one reads as answered.
    if not canonical_fields and _scalar(doc, "verdict").lower() in VERDICTS:
        _add(out, "error", "verdict",
             f"{_scalar(doc, 'verdict')!r} is an OUTCOME, and `verdict` is the accept/reject "
             f"arbitration; put it in `result` or this run states no arbitration")
    if not canonical_fields and _scalar(doc, "result").lower() in ARBITRATIONS:
        _add(out, "error", "result",
             f"{_scalar(doc, 'result')!r} is an ARBITRATION, and `result` is what the optimization "
             f"achieved; put it in `verdict` or this run states no outcome")

    _enum(out, doc, "status", STATUSES, required=False)
    extension_severity = "note" if canonical_base else "error"
    _list_of_objects(out, doc, "deferred", ("task_id", "resume_entry"),
                     severity=extension_severity)
    if "caveats" not in doc:
        _add(out, extension_severity, "caveats",
             "absent; an unavailable acceptance artifact is recorded here, so an empty list and a "
             "missing key are different claims and only one of them is auditable")
    elif not isinstance(doc["caveats"], list):
        _add(out, extension_severity, "caveats",
             f"expected a list, found {type(doc['caveats']).__name__}")

    # --- close: who checked the closure, and what became of the challenges
    ca = doc.get("close_audit")
    if ca is None:
        _add(out, extension_severity, "close_audit",
             "absent; a closure that was never audited and one whose audit found nothing leave the "
             "same report, and the difference is what a supervisor needs")
    elif not isinstance(ca, dict):
        _add(out, extension_severity, "close_audit",
             f"expected an object, found {type(ca).__name__}")
    else:
        _enum(out, ca, "verified_by", VERIFIED_BY)
        if "findings" not in ca:
            _add(out, "error", "close_audit.findings",
                 "absent; expected the count the audit reported (0 is an answer, absence is not)")
        elif not isinstance(ca["findings"], int) or isinstance(ca["findings"], bool):
            _add(out, "error", "close_audit.findings",
                 f"expected an integer count, found {type(ca['findings']).__name__}")
        if not ca.get("audit_ref"):
            _add(out, "note", "close_audit.audit_ref",
                 "no path to the audit output; the count is then unre-derivable")
        if "high_severity_count" not in ca:
            # A high finding contradicts a conclusion the run already banked, so it is not
            # caveat-eligible the way the rest are; folded into one total it reads as one more
            # thing that was noted.
            _add(out, "note", "close_audit.high_severity_count",
                 "absent; the audit separates findings that contradict a banked conclusion from "
                 "findings that record an unfollowed contract, and only the total was carried")
        if canonical_fields and require_canonical:
            lifecycle = ca.get("sweep_lifecycle")
            if not isinstance(lifecycle, dict):
                _add(out, "error", "close_audit.sweep_lifecycle",
                     "strict canonical reports must carry the typed sweep lifecycle audit summary")
            elif lifecycle.get("checked") is not True:
                _add(out, "error", "close_audit.sweep_lifecycle.checked",
                     "must be true after close_audit checked arm_local, post_merge, and handoff")
            elif not str(lifecycle.get("audit_ref") or ca.get("audit_ref") or "").strip():
                _add(out, "error", "close_audit.sweep_lifecycle.audit_ref",
                     "must retain a reference to the close audit that checked the lifecycle")
        _lint_waivers(out, doc, ca)

    # --- known_wrong: the most-repeated cross-round obligation in the whole tree, which until now
    # had no filename, no field name, no schema and no validator. Both orchestration skills demand
    # a "Known-wrong block" at every close and at every re-seed, with the right reason ("a refuted
    # claim survives on disk in the brief; if the packet does not retire it, the next session
    # re-reads it as fact and re-spends rounds on a dead lever") -- and an obligation nothing can
    # check is one that was, measurably, not met: 9.9% of one campaign's rounds ran on stale
    # guidance, and one of them opens "TWO LEADS FROM MY DISPATCH BRIEF ARE ALREADY DEAD IN MY OWN
    # LEDGER AND MUST NOT BE RE-SPENT". `source` is what makes it actionable: a claim refuted from
    # `brief` is one the brief GENERATOR has to stop emitting, which is a different repair from a
    # claim an arm refuted about itself.
    kw = doc.get("known_wrong")
    if kw is None:
        _add(out, "error" if (owner == "captain" or
                               (owner == "auto" and _is_captain_shape(doc))) else "note",
             "known_wrong",
             "absent; an empty list is the explicit claim that nothing was retired")
    elif not isinstance(kw, list):
        _add(out, "error", "known_wrong", f"expected a list, found {type(kw).__name__}")
    else:
        for i, item in enumerate(kw):
            if not isinstance(item, dict):
                _add(out, "error", f"known_wrong[{i}]",
                     f"expected an object, found {type(item).__name__}")
                continue
            for k in ("claim", "refuted_by"):
                if not str(item.get(k) or "").strip():
                    _add(out, "error", f"known_wrong[{i}].{k}",
                         "absent; a retirement without the claim and what refuted it cannot stop "
                         "the claim being re-read as fact")
            src = item.get("source")
            if src not in KNOWN_WRONG_SOURCES:
                # Recognise the drift instead of only rejecting it. `own (gluon worker)` is not
                # carelessness -- it is an author restoring an axis the schema deleted, and telling
                # them WHICH field the parenthetical belongs in is what actually stops it recurring.
                m = _OWN_WITH_ACTOR.match(str(src or ""))
                if m:
                    _add(out, "error", f"known_wrong[{i}].source",
                         f"{src!r} folds two axes into one field. `source` is the ORIGIN "
                         f"{list(KNOWN_WRONG_SOURCES)}; the parenthetical {m.group(2)!r} names WHO "
                         f"retired it, which belongs in `retired_by` "
                         f"{list(KNOWN_WRONG_RETIRED_BY)} (plus `arm_id` when it was an arm). If "
                         f"the parenthetical says the claim reached a dispatch brief, that is "
                         f"`emitted_in_brief: true` -- origin and propagation are independent, and "
                         f"it is the propagation that makes the brief generator the thing to repair")
                else:
                    _add(out, "error", f"known_wrong[{i}].source",
                         f"{src!r} is not one of {list(KNOWN_WRONG_SOURCES)}; a claim "
                         f"refuted from `brief` needs the brief generator repaired, which is a "
                         f"different action from retiring an arm's own claim")
            rb = item.get("retired_by")
            if rb is not None and rb not in KNOWN_WRONG_RETIRED_BY:
                _add(out, "error", f"known_wrong[{i}].retired_by",
                     f"{rb!r} is not one of {list(KNOWN_WRONG_RETIRED_BY)} -- this pack's role "
                     f"registry, which is what makes 'who found this out' answerable across a "
                     f"two-stage run where `own` has five referents")
            if rb == "arm" and not str(item.get("arm_id") or "").strip():
                _add(out, "note", f"known_wrong[{i}].arm_id",
                     "retired_by=arm without arm_id; reports have written the arm's name into the "
                     "source string because there was nowhere else to put it")
            eib = item.get("emitted_in_brief")
            if eib is not None and not isinstance(eib, bool):
                _add(out, "error", f"known_wrong[{i}].emitted_in_brief",
                     f"expected a bool, found {type(eib).__name__}")
            if src == "brief" and eib is False:
                _add(out, "error", f"known_wrong[{i}].emitted_in_brief",
                     "source=brief with emitted_in_brief=false contradicts itself: `brief` IS the "
                     "statement that a dispatch brief carried this claim")

    sv = _enum(out, doc, "skeptic_verdict", SKEPTIC_VERDICTS,
               severity=extension_severity)
    if sv and sv != "not_run":
        _list_of_objects(out, doc, "skeptic_items",
                         ("claim", "verdict", "disposition", "evidence_ref"))
        for i, item in enumerate(doc.get("skeptic_items") or []):
            if isinstance(item, dict) and item.get("disposition") not in DISPOSITIONS + (None,):
                _add(out, "error", f"skeptic_items[{i}].disposition",
                     f"{item['disposition']!r} is not one of {list(DISPOSITIONS)}")
        if sv == "contradicted" and not (doc.get("skeptic_items") or []):
            _add(out, "error", "skeptic_items",
                 "the skeptic contradicted this result and the report itemizes nothing; a "
                 "contradiction with no recorded disposition reads exactly like agreement")

    # --- captain: the numbers the fleet's accept step reads out of this file and nowhere else
    shape = ("canonical-base" if canonical_base else
             "captain" if (owner == "captain" or (owner == "auto" and _is_captain_shape(doc)))
             else "worker")
    if shape == "captain":
        # The captain's job IS the arbitration -- it spans two skills, so no worker can make it --
        # and a captain report that does not carry one has not done the job it exists for.
        if "verdict" not in doc:
            _add(out, "error", "verdict",
                 f"absent; a captain report owes the accept/reject arbitration "
                 f"({list(ARBITRATIONS)}) as well as the outcome in `result`")
        if not doc.get("champion_ref"):
            _add(out, "error", "champion_ref",
                 "absent; the accept step re-asserts the named bundle, and cannot re-assert one "
                 "the report does not name")
        # Read through the shared readers, so this validator and the fleet's accept step agree
        # about where a number lives. A report that spells the headline as a per-case mapping and
        # names its primary case is complete; one that spells it that way and names no primary case
        # is genuinely unstated, and only the second is an error.
        for k in CAPTAIN_NUMERIC:
            value, how, problem = read_headline_number(doc, k)
            if problem:
                _add(out, "error", k, problem)
            elif how != k:
                _add(out, "note", k, f"read as {how}; a consumer probing `{k}` directly finds "
                                     f"nothing, so it goes through report_lint's reader")
            del value
        rows, how = read_served_rows(doc)
        if not rows:
            _add(out, "error", "served_range",
                 f"absent or empty under any of {list(SERVED_ROWS_ALIASES)}; a single-shape result "
                 f"cannot be weighed by the served mix")
        elif how != "served_range":
            _add(out, "note", "served_range", f"read as {how}")
        _enum(out, doc, "stay_plain_basis", STAY_PLAIN_BASIS, required=False)

    out.sort(key=lambda f: (f["severity"] != "error", f["field"], f["detail"]))
    return {"findings": out, "owner_shape": shape, "result": result, "verdict": arb,
            "errors": sum(1 for f in out if f["severity"] == "error")}


def _selftest():
    good = {
        "result": "win", "status": "closed", "caveats": [], "deferred": [],
        "close_audit": {"verified_by": "self", "findings": 0, "audit_ref": "close_audit.json"},
        "skeptic_verdict": "credible",
        "skeptic_items": [{"claim": "c", "verdict": "weak", "disposition": "rebutted",
                           "evidence_ref": "exp/x.json"}],
    }
    r = lint(good)
    assert r["errors"] == 0, r["findings"]
    assert r["owner_shape"] == "worker" and r["result"] == "win" and r["verdict"] is None

    # the schema label is a note, never an error: it names the file without gatekeeping it
    assert any(f["field"] == "schema" and f["severity"] == "note" for f in r["findings"])
    assert not any(f["field"] == "schema"
                   for f in lint(dict(good, schema=SCHEMA_ID))["findings"])

    # a captain-shaped report is detected from its own content, and then owes more
    cap = dict(good, champion_ref="champ/plain_champion.json", known_wrong=[])
    r = lint(cap)
    assert r["owner_shape"] == "captain"
    fields = {f["field"] for f in r["findings"]}
    assert {"default_ms", "champion_ms", "best_ms", "vs_champion", "served_range",
            "verdict"} <= fields, fields
    full = dict(cap, verdict="accept", default_ms=1.0, champion_ms=0.9, best_ms=0.8,
                vs_champion=1.125, served_range=[{"shape": "s", "ms": 0.8}],
                stay_plain_basis="measured")
    assert lint(full)["errors"] == 0, lint(full)["findings"]
    assert lint(full)["verdict"] == "accept"
    # ...and --owner may force the stricter shape on a report that hid it
    assert lint(good, owner="captain")["errors"] > 0

    # THE SPELLINGS. Four owners on one track wrote four shapes and a supervisor probing guessed
    # names read the clean ones as incomplete. Each shape below is complete and must lint clean;
    # what stays an error is a headline whose basis the report never names.
    percase = dict(full, champion_ms={"small": 0.9, "large": 1.8}, primary_case="large")
    assert lint(percase, owner="captain")["errors"] == 0, lint(percase, owner="captain")["findings"]
    assert read_headline_number(percase, "champion_ms")[0] == 1.8
    unnamed = dict(full, champion_ms={"small": 0.9, "large": 1.8})
    assert any(f["field"] == "champion_ms" and f["severity"] == "error"
               for f in lint(unnamed, owner="captain")["findings"])
    assert read_headline_number(unnamed, "champion_ms")[2], "an unstated basis owes a problem line"
    # A per-case headline with NO explicit primary_case is still resolvable when the canonical
    # measurement's ranked value singles one case out: that is two of the report's own declared
    # fields agreeing, not a guess. It fails closed when they do not.
    derived = dict(full, champion_ms={"small": 0.9, "large": 1.8},
                   default_ms={"small": 2.0, "large": 4.0},
                   measurement={"schema": "kernel_opt.measurement/1", "value": 1.8})
    assert declared_primary_case(derived) == "large"
    assert read_headline_number(derived, "champion_ms")[0] == 1.8
    assert read_headline_number(derived, "default_ms")[0] == 4.0
    assert lint(derived, owner="captain")["errors"] == 0, lint(derived, owner="captain")["findings"]
    ambiguous = dict(derived, measurement={"schema": "kernel_opt.measurement/1", "value": 0.5})
    assert declared_primary_case(ambiguous) is None
    assert read_headline_number(ambiguous, "champion_ms")[2], "an unmatched ranked value is not a basis"
    # a headline inside the block that STATES the headline is the other spelling in use
    nested = {k: v for k, v in full.items() if k not in ("champion_ms", "default_ms")}
    nested["headline"] = {"champion_ms": 0.9, "default_ms": 1.0}
    assert lint(nested, owner="captain")["errors"] == 0, lint(nested, owner="captain")["findings"]
    assert read_headline_number(nested, "champion_ms")[1] == "headline.champion_ms"

    tabled = {k: v for k, v in full.items() if k != "served_range"}
    tabled["served_range_table"] = {"rows": [{"shape": "s", "ms": 0.8}]}
    assert lint(tabled, owner="captain")["errors"] == 0, lint(tabled, owner="captain")["findings"]
    assert read_served_rows(tabled)[1] == "served_range_table.rows"
    aliased = {k: v for k, v in full.items() if k != "default_ms"}
    aliased["baseline_ms"] = 1.0
    assert lint(aliased, owner="captain")["errors"] == 0, lint(aliased, owner="captain")["findings"]
    # ...and a genuinely absent number is still an error, so this reader cannot hide a thin report
    thin = {k: v for k, v in full.items() if k != "champion_ms"}
    assert any(f["field"] == "champion_ms" and f["severity"] == "error"
               for f in lint(thin, owner="captain")["findings"])
    # the budget reader is shape-tolerant in the same way
    assert read_budget({"budget": {"rounds_used": 38, "round_budget": 200}}) == (38, 200)
    assert read_budget({"rounds": {"spent": 13, "budget": 100}}) == (13, 100)
    assert read_budget({"rounds_used": 9, "round_budget": 200}) == (9, 200)
    assert read_budget({}) == (None, None)

    # result and verdict are DIFFERENT questions, and swapping them is caught both ways
    assert any(f["field"] == "verdict" and "OUTCOME" in f["detail"]
               for f in lint(dict(good, verdict="win"))["findings"])
    assert any(f["field"] == "result" and "ARBITRATION" in f["detail"]
               for f in lint(dict(good, result="accept"))["findings"])
    assert lint(dict(good, verdict="reject"))["errors"] == 0

    # case is load-bearing, because every reader compares against a lower-case literal
    assert any(f["field"] == "result" and "lower-case" in f["detail"]
               for f in lint(dict(good, result="WIN"))["findings"])
    assert any(f["field"] == "verdict" and "lower-case" in f["detail"]
               for f in lint(dict(good, verdict="ACCEPT"))["findings"])
    # both negative spellings are legal; an invented verdict is not
    for v in ("negative_keep_baseline", "negative_revert_plain"):
        assert lint(dict(good, result=v))["errors"] == 0, v
    assert lint(dict(good, result="negative"))["errors"] > 0

    # absent vs empty are different claims, and only one of them is auditable
    for drop in ("caveats", "deferred", "close_audit", "skeptic_verdict"):
        d = dict(good)
        del d[drop]
        assert any(f["field"].startswith(drop) for f in lint(d)["findings"]), drop
    assert lint(dict(good, caveats=["profiler degraded: <reason>"]))["errors"] == 0

    # --- known_wrong. The validator shipped with no selftest coverage at all, which is how its
    #     enum stayed one axis too small through a whole fleet run.
    kwg = dict(good, known_wrong=[
        {"claim": "c1", "source": "brief", "refuted_by": "r", "evidence_ref": "e"},
        {"claim": "c2", "source": "own", "refuted_by": "r", "retired_by": "deepdig"},
        {"claim": "c3", "source": "skeptic", "refuted_by": "r", "retired_by": "captain"},
        {"claim": "c4", "source": "arm", "refuted_by": "r", "retired_by": "arm",
         "arm_id": "arm_b", "emitted_in_brief": False},
    ])
    assert lint(kwg)["errors"] == 0, lint(kwg)["findings"]

    # THE MEASURED DRIFT, verbatim from a real report: `own` with an actor in a parenthetical. The
    # error has to name the field the parenthetical belongs in, or the same string comes back.
    for spelling, actor in (("own (gluon worker)", "gluon worker"),
                            ("own (captain)", "captain"),
                            ("own (arm arm_b)", "arm arm_b"),
                            ("own (captain, written into the worker's brief)",
                             "captain, written into the worker's brief")):
        f = [x for x in lint(dict(good, known_wrong=[
            {"claim": "c", "source": spelling, "refuted_by": "r"}]))["findings"]
            if x["field"] == "known_wrong[0].source"]
        assert f, spelling
        assert "retired_by" in f[0]["detail"] and actor in f[0]["detail"], f[0]["detail"]
        assert "emitted_in_brief" in f[0]["detail"], f[0]["detail"]
    # A source that is not the parenthetical shape still gets the plain enum message.
    f = [x for x in lint(dict(good, known_wrong=[
        {"claim": "c", "source": "somewhere", "refuted_by": "r"}]))["findings"]
        if x["field"] == "known_wrong[0].source"]
    assert f and "retired_by" not in f[0]["detail"], f[0]["detail"]

    # retired_by is optional but enum-checked, arm_id is a note, and origin/propagation may not
    # contradict each other.
    bad_rb = lint(dict(good, known_wrong=[
        {"claim": "c", "source": "own", "refuted_by": "r", "retired_by": "gluon worker"}]))
    assert any(f["field"] == "known_wrong[0].retired_by" for f in bad_rb["findings"]), bad_rb
    no_arm = lint(dict(good, known_wrong=[
        {"claim": "c", "source": "arm", "refuted_by": "r", "retired_by": "arm"}]))
    assert no_arm["errors"] == 0 and any(f["field"] == "known_wrong[0].arm_id"
                                        for f in no_arm["findings"]), no_arm["findings"]
    contra_brief = lint(dict(good, known_wrong=[
        {"claim": "c", "source": "brief", "refuted_by": "r", "emitted_in_brief": False}]))
    assert any(f["field"] == "known_wrong[0].emitted_in_brief"
               for f in contra_brief["findings"]), contra_brief["findings"]

    # --- arm_result.json. Return/converge accepts one canonical schema only. Older roll-ups
    # remain evidence but must pass through the explicit writer/adapter before a captain ranks them.
    arm_measurement = {
        "schema": "kernel_opt.measurement/1", "measurement_id": "arm-m1", "value": 1.32,
        "collected_at": "2026-09-01T00:00:00Z",
        "identity": {
            "shape_set": ["s1"], "aggregation": "geomean", "unit": "ratio",
            "boundary": "kernel", "source": "device_timing", "scope": "device",
            "sample": {"count": 20, "method": "interleaved"},
            "baseline_ref": "baseline.json", "comparator_ref": "champion.json",
        },
    }
    arm = {"schema": ARM_RESULT_SCHEMA, "arm_id": "arm-a", "lever": "split_kv",
           "direction_class": "structural", "iters_used": 15, "verdict": "supported",
           "completion": "complete", "gate_stage": "c", "measurement": arm_measurement}
    r = lint(arm)
    assert r["owner_shape"] == "arm" and r["errors"] == 0, r["findings"]
    legacy_arm = dict(arm)
    legacy_arm.pop("schema")
    assert any(f["field"] == "schema" and "write-arm-result" in f["detail"]
               for f in lint(legacy_arm)["findings"])
    assert any(f["field"].endswith("verdict")
               for f in lint(dict(arm, verdict="kept"))["findings"])
    truncated = dict(arm, completion="truncated", verdict="inconclusive")
    truncated.pop("measurement")
    assert lint(truncated)["errors"] == 0, lint(truncated)["findings"]
    assert any(f["field"].endswith("measurement")
               for f in lint(dict(arm, measurement=None))["findings"])

    # a contradiction that itemizes nothing reads exactly like agreement -> error
    contra = dict(good, skeptic_verdict="contradicted", skeptic_items=[])
    assert any(f["field"] == "skeptic_items" for f in lint(contra)["findings"])
    # ...and a disposition outside the enum is caught per item
    bad_item = dict(good, skeptic_items=[{"claim": "c", "verdict": "weak",
                                          "disposition": "noted", "evidence_ref": "x"}])
    assert any(f["field"] == "skeptic_items[0].disposition" for f in lint(bad_item)["findings"])
    # not_run is a real answer and does not drag the itemization in behind it
    assert lint({k: v for k, v in dict(good, skeptic_verdict="not_run").items()
                 if k != "skeptic_items"})["errors"] == 0

    # a self-audited closure is legal and must SAY so; an invented auditor is not
    assert lint(dict(good, close_audit={"verified_by": "fleet", "findings": 2,
                                        "audit_ref": "a.json"}))["errors"] == 0
    assert lint(dict(good, close_audit={"verified_by": "nobody", "findings": 0}))["errors"] > 0
    assert lint(dict(good, close_audit={"verified_by": "self", "findings": True}))["errors"] > 0

    # ---- a RICH enum field is more information, not a type error ---------------------------
    # Verbatim from a measured close: the report that wrote the most careful version of these two
    # fields -- reason travelling with the arbitration instead of in a sibling key -- was the one
    # this validator failed, on both of them, as "expected a string, found dict".
    rich = dict(good,
                result={"value": "win", "default_ms": 0.0502, "champion_ms": 0.0258},
                verdict={"value": "accept", "reason": "champion_gate PASS 8/8, re-run by the captain"})
    assert lint(rich)["errors"] == 0, lint(rich)["findings"]
    assert lint(rich)["result"] == "win" and lint(rich)["verdict"] == "accept"
    # ...and the two-directions check still reads through the object form
    swapped = dict(good, result={"value": "accept"}, verdict={"value": "win"})
    fs = {f["field"] for f in lint(swapped)["findings"] if f["severity"] == "error"}
    assert {"result", "verdict"} <= fs, fs
    # an object with no `value` is still a type error, and says what it wanted
    nov = lint(dict(good, result={"outcome": "win"}))
    assert any(f["field"] == "result" and "carrying `value`" in f["detail"]
               for f in nov["findings"]), nov["findings"]
    # a NUMERIC headline follows the same convention -- several bases are more information, and
    # exactly one of them is the number the speedup is quoted against
    if "champion_ms" in good:
        multi = dict(good, champion_ms={"geomean_basis": 0.7646, "c64_only_basis": 2.5966})
        assert any(f["field"] == "champion_ms" and "no `value`" in f["detail"]
                   for f in lint(multi)["findings"]), lint(multi)["findings"]
        named = dict(good, champion_ms={"value": 0.7646, "geomean_basis": 0.7646,
                                        "c64_only_basis": 2.5966})
        assert not [f for f in lint(named)["findings"] if f["field"] == "champion_ms"], \
            lint(named)["findings"]
    # the schema name a kernel-opt-run captain actually writes is a synonym, not an unknown
    assert not [f for f in lint(dict(good, schema="kernel_opt_run.final_report"))["findings"]
                if f["field"] == "schema"]
    assert [f for f in lint(dict(good, schema="something.else/9"))["findings"]
            if f["field"] == "schema"]

    # Canonical final reports separate lifecycle stage, measured outcome, acceptance
    # arbitration, and run status.  Strict mode requires this form and its reference.
    measurement = {
        "schema": "kernel_opt.measurement/1", "measurement_id": "m1", "value": 1.1,
        "collected_at": "2026-09-01T00:00:00Z",
        "identity": {
            "shape_set": ["s1"], "aggregation": "geomean", "unit": "ratio",
            "boundary": "kernel", "source": "device_timing", "scope": "device",
            "sample": {"count": 20, "method": "interleaved"},
            "baseline_ref": "baseline.json", "comparator_ref": "champion.json",
        },
    }
    digest = "a" * 64
    typed_ref = {"schema": "kernel_opt.artifact_ref/1", "uri": "work:/fixture.json",
                 "sha256": digest}
    def producer_receipt(role):
        return {
            "schema": "kernel_opt.producer_receipt/1", "receipt_id": f"fixture-{role}",
            "artifact_role": role, "produced_at": "2026-09-01T00:00:00Z",
            "timing": "process_time",
            "producer": {"tool": "fixture", "executable_sha256": digest, "mode": "primary"},
            "artifact": {"ref": f"{role}.json", "schema": "fixture/1", "sha256": digest},
            "invariant_validator": {"tool": "canonical_record.py",
                                    "executable_sha256": digest,
                                    "schema": "fixture/1", "passed": True},
        }
    canonical = {
        "schema": FINAL_REPORT_SCHEMA, "run_id": "run-a", "generation": 1,
        "lifecycle_profile": "direct-owner",
        "stage": "finalization", "outcome": "win", "arbitration": "accept",
        "status": "closed", "measurement": measurement, "canonical_ref": "canonical_record.json",
        "champion_ref": "champion.json", "champion_gate_ref": "champion_gate.json",
        "refs": {"canonical": typed_ref, "close_audit": typed_ref,
                 "champion": typed_ref, "champion_gate": typed_ref},
        "anchor": {"type": "pinned_comparator", "ref": typed_ref},
        "close_audit_ref": "close_audit.json", "scope": {"authorization": "plain-only"},
        "deep_arbitration": {"state": "not_authorized", "champion_ms": 1.1},
        "default_ms": 1.2, "champion_ms": 1.1, "best_ms": 1.1,
        "served_range": [{"case": "s1", "default_ms": 1.2, "champion_ms": 1.1}],
        "verdict": "accept", "vs_champion": {"value": 1.0},
        "caveats": [], "deferred": [], "known_wrong": [],
        "producer": {"role": "direct_owner"},
        "audit": {"strict_pass": True, "high_severity_count": 0, "blocked_count": 0},
        "provenance": {"receipts": [producer_receipt("audit"),
                                    producer_receipt("final_report")]},
        "integrity": {"source_schema": FINAL_REPORT_SCHEMA, "normalization_status": "canonical",
                      "legacy_unverified": False, "eligible_clean": True},
        "close_audit": {
            "verified_by": "self", "findings": 0, "high_severity_count": 0,
            "audit_ref": "close_audit.json",
            "sweep_lifecycle": {"checked": True, "audit_ref": "close_audit.json"},
        },
        "skeptic_verdict": "not_run",
    }
    assert lint(canonical, require_canonical=True)["errors"] == 0, \
        lint(canonical, require_canonical=True)["findings"]
    canonical_base = {
        key: canonical[key] for key in (
            "schema", "run_id", "generation", "stage", "outcome", "arbitration", "status",
            "measurement", "canonical_ref", "champion_ref", "champion_gate_ref", "anchor",
            "close_audit_ref", "scope", "deferred", "deep_arbitration", "known_wrong", "producer",
            "lifecycle_profile", "refs", "audit", "provenance", "integrity",
        )
    }
    base_lint = lint(canonical_base, owner="canonical-base", require_canonical=True)
    assert base_lint["errors"] == 0 and base_lint["owner_shape"] == "canonical-base", \
        base_lint["findings"]
    assert {"caveats", "close_audit", "skeptic_verdict"} <= {
        item["field"] for item in base_lint["findings"] if item["severity"] == "note"
    }, base_lint["findings"]
    assert lint(dict(canonical, known_wrong=None), owner="captain",
                require_canonical=True)["errors"] > 0
    nonwaivable = dict(canonical, close_audit={
        "verified_by": "self", "findings": 1, "high_severity_count": 1,
        "audit_ref": "close_audit.json"},
        waivers=[{"check": "canonical_finalization", "kind": "resweep_missing",
                  "why": "skip", "evidence_ref": "x"}])
    assert any(f["field"] == "waivers[0]" and "non-waivable" in f["detail"]
               for f in lint(nonwaivable)["findings"])
    policy_nonwaivable = dict(canonical, close_audit={
        "verified_by": "self", "findings": 1, "high_severity_count": 1,
        "audit_ref": "close_audit.json"},
        waivers=[{"check": "role_policy", "kind": "unauthorized_resweep",
                  "why": "ignore", "evidence_ref": "x"}])
    assert any(f["field"] == "waivers[0]" and "non-waivable" in f["detail"]
               for f in lint(policy_nonwaivable)["findings"])

    # ---- waivers: a HIGH finding must be answered one for one -------------------------------
    # The measured failure this replaces: three closes exited 1 under --strict and were accepted with
    # a paragraph saying the findings were tool false positives. Two of those calls were right; none
    # was checkable.
    hi = dict(good, close_audit={"verified_by": "self", "findings": 23,
                                 "high_severity_count": 2, "audit_ref": "close_audit.json"})
    assert any(f["field"] == "waivers" and "no `waivers[]`" in f["detail"]
               for f in lint(hi)["findings"]), lint(hi)["findings"]
    # one waiver for two high findings does not cover the set
    one = dict(hi, waivers=[{"check": "branch_waves", "kind": "single_wave_close",
                             "why": "remaining structure bounded at 2.2% geomean", "evidence_ref": "s.log"}])
    assert any(f["field"] == "waivers" and "unanswered" in f["detail"]
               for f in lint(one)["findings"]), lint(one)["findings"]
    # two complete waivers clear it
    two = dict(hi, waivers=one["waivers"] + [
        {"check": "claim_citations", "kind": "citation_gap",
         "why": "the cited artifact is attached", "evidence_ref": "exp/r22.json"}])
    assert lint(two)["errors"] == 0, lint(two)["findings"]
    # a waiver with no evidence path is the paragraph again, in a field
    noev = dict(hi, waivers=[dict(w, evidence_ref="") for w in two["waivers"]])
    assert any(f["field"].startswith("waivers[") and "evidence_ref" in f["detail"]
               for f in lint(noev)["findings"]), lint(noev)["findings"]
    # a clean close owes nothing here and must stay quiet
    assert not [f for f in lint(dict(good, close_audit={
        "verified_by": "self", "findings": 3, "high_severity_count": 0,
        "audit_ref": "a.json"}))["findings"] if f["field"].startswith("waivers")]

    assert lint([])["findings"][0]["field"] == "<root>"
    print("[report_lint] SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="final_report.json",
                    help="final_report.json, or a BRANCH arm's arm_result.json (detected from "
                         "content; force with --owner arm)")
    ap.add_argument("--owner", default="auto",
                    choices=("auto", "captain", "worker", "arm", "canonical-base"))
    ap.add_argument("--json", dest="out_json")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when an error-severity finding is present (default: always 0 -- a "
                         "thin report is still evidence, and refusing to read it is worse)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    try:
        with open(a.report) as f:
            doc = json.load(f)
    except FileNotFoundError:
        print(f"[report_lint] no report at {a.report}")
        return 1 if a.strict else 0
    except json.JSONDecodeError as exc:
        print(f"[report_lint] {a.report} is not valid JSON: {exc}")
        return 1 if a.strict else 0

    res = lint(doc, a.owner, require_canonical=a.strict)
    print(f"[report_lint] {a.report}: shape={res['owner_shape']} result={res['result']} "
          f"verdict={res['verdict']} errors={res['errors']} findings={len(res['findings'])}")
    for f in res["findings"]:
        print(f"  [{f['severity']}] {f['field']}: {f['detail']}")
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[report_lint] wrote {a.out_json}")
    return 1 if (a.strict and res["errors"]) else 0


if __name__ == "__main__":
    sys.exit(main())
