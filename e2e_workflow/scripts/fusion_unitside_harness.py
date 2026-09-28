#!/usr/bin/env python3
"""Phase 3.0: the 单侧 (isolated) gate for kernel-fusion candidates.

Deterministic validator of the per-candidate microbench VERDICTS produced by the
fusion_unit_validator role. It does NOT run kernels and claims no perf of its own —
it VALIDATES that each verdict is trustworthy (well-formed + provenance-consistent
with the Phase 2.1 candidate it claims to test) and derives the gate:

Every verdict is scored on TWO INDEPENDENT AXES, because collapsing them is how a
correct measurement of the wrong quantity gets closed as a refutation:

  correctness_status ∈ {pass, fail, not_engaged}   — did it compute the right answer?
  perf_status        ∈ {win, no_win, undefined}    — is it faster?

  * `perf_status` is `undefined` whenever `correctness_status != pass`, and
    `isolated_speedup` is nulled with it. A kernel that was fed wrong takes a wrong
    branch and a wrong tile config, so its `cand_ms` timed something that was never
    the candidate. "parity failed AND it was slow" is ONE fault counted twice.

  failure_kind ∈ {none, performance, functional, unmeasured, not_engaged} — and this
  is the field a human should read first, because it names WHICH KIND of failure:
  * performance — 功能失败? no. It computed the right answer and simply is not
    faster. This is the only verdict that may be closed on the spot.
  * functional  — it diverged AND the author recorded a `parity_diagnosis`, i.e. the
    mundane feeding causes were eliminated and the divergence is believed real.
  * unmeasured  — it diverged and nobody looked. NOT a result. The gate fails.
  * not_engaged — a collective whose size guard fell back to the split path; there
    is no fused execution to judge at this shape.

`unit_side_status` ∈ {pass, fail, blocked, needs_diagnosis} is DERIVED from the pair
and kept for the board/apply-back consumers:
  (pass, win) → pass · (pass, no_win) → fail · (fail, undefined) → needs_diagnosis ·
  (not_engaged, undefined) → blocked
Only `pass` is eligible for apply-back.

A verdict that is malformed, references an unknown candidate, tests a DIFFERENT shape
than the candidate captured, or names a fused_fn that is not the candidate's existing
API is an ERROR (untrustworthy → the harness FAILS so it is re-run) -- never silently a
pass. This is the anti-cheat that keeps 单侧 honest, mirroring the provenance checks in
fusion_candidate_harness.py.

COVERAGE (recall) gate
----------------------
The checks above are all *precision*: they ask whether a SUBMITTED verdict is
trustworthy. They cannot see a candidate for which no verdict was ever submitted,
because the validator loops over verdicts, not over candidates. A run that
microbenched 16 of 42 candidates therefore rendered as
"总计 16 条 verdict：pass 16 / fail 0 ... status=pass" -- a 38%-coverage run that is
indistinguishable from a complete one. Observed on DSR1 2026-08-26: the highest
roofline decode candidate (kv_write_cluster) and all 20 prefill candidates silently
never reached the microbench, and nothing anywhere turned red.

So the denominator is now explicit. Every candidate that CITES an existing fused API
(i.e. is 单侧-testable at all) must end up in exactly one of:

  * a real verdict (pass / fail / blocked), or
  * `--equivalent <candidate_id>=<representative_candidate_id>`: the same
    recipe/cohort was measured once on its declared representative. The sibling
    inherits an auditable `equivalent_pass`, `equivalent_fail`,
    `equivalent_blocked`, or `equivalent_needs_diagnosis` result without
    repeating an identical unit test, or
  * `--subsumed <candidate_id>=<ladder_top>` (status `subsumed_pass`): its
    ladder-top superset was benched and PASSED, and that microbench exercised
    every row this rung would remove. Covered, and cheap -- this is what lets a
    10-slot budget cover a board of overlapping ladders, or
  * an explicit `--waive <candidate_id>=<reason>` (status `waived`), or
  * `--budget-skipped <candidate_id>=<reason>` (status `budget_skipped`) →
    NOT covered, the harness FAILS. Budget exhaustion used to be passed in as
    `--waive`, which rendered as `waived`, dropped out of the unvalidated set,
    and let coverage report complete over a board where 17 of 27 in-scope
    candidates had never been measured (DSR1 2026-09-03) -- including the one
    carrying a ★★★ prior with two measured e2e confirms. Running out of budget
    is a fact about us, not a judgment about the candidate, or
  * `not_validated` → the harness FAILS.

tier-C candidates (`implementation_class: new_helper_kernel`, i.e. no existing kernel
to bench) are reported as `deferred_author` and are legitimately out of scope -- they
are counted and shown, never silently dropped. Pass `--allow-partial-coverage` to
report the gap without failing (the gap is still printed loudly either way).

Usage:
  python3 fusion_unitside_harness.py --candidates fusion_candidates.json \
      --topk fusion_topk.json \
      --verdicts <dir-of-*.json | combined.json> \
      --out-md FUSION_UNITSIDE.md --out-json fusion_unitside.json \
      [--min-speedup 1.0] [--allow-partial-coverage] \
      [--waive <candidate_id>=<reason> ...]
"""
import argparse
import glob
import json
import os
import sys


REQUIRED_VERDICT_FIELDS = (
    "candidate_id", "parity", "isolated_speedup", "tested_shape", "fused_fn")

# Fields added by the 2026-09-01 generality pass. They exist to force an EXPLICIT
# answer to two questions the old schema let a verdict stay silent about -- silence
# that produced two of the three misses in the DSR1 blind test:
#
#   ref_ops               — what the SPLIT REFERENCE leg actually executed. Without
#                           it, `ref_ms` is an unattributed number and nobody can see
#                           that the reference was narrower than the thing the fusion
#                           replaces.
#   outside_work_removed  — ops the fused kernel makes unnecessary that are NOT
#                           members of the captured region. `removable_row_ids` is
#                           constrained to be a subset of `members` (candidate harness
#                           :943), and `members` is a slice of ONE trace region -- so
#                           per-step work that lives outside that slice (a weight
#                           dequant hoisted into another operator, a copy the caller
#                           does) is structurally invisible to the reference. The
#                           pipeline cannot discover such work on its own; it CAN
#                           refuse to accept a verdict that never looked. `[]` is a
#                           legal answer -- an unanswered one is not.
SCOPE_VERDICT_FIELDS = ("ref_ops", "outside_work_removed")

# Required only when parity != pass: the mundane-cause elimination the ROLE already
# demands in prose (fusion_unit_validator.md:137). Prose lost to the gate, because the
# gate rendered `fail` and the board consumed `fail` as terminal.
DIAGNOSIS_FIELD = "parity_diagnosis"


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _load_verdicts(path):
    """A directory of *.json (one verdict each) OR a combined json ({verdicts:[...]}
    or a bare list)."""
    if os.path.isdir(path):
        out = []
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            out.append(_load(fp))
        return out
    data = _load(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("verdicts") or [data]
    return []


def _norm(text):
    """Normalize an API/fn name for a robust contains-match: lowercase, keep only
    alnum + underscore runs (drops the ' (--flag)' suffixes, '::', 'aiter ' prefix)."""
    text = str(text or "").lower()
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch == "_") else " ")
    return [tok for tok in "".join(keep).split() if tok]


def _fn_matches_api(fused_fn, existing_apis):
    """The verdict's fused_fn must be one of the candidate's declared existing APIs
    (the microbench cannot claim a win for a kernel the candidate never cited)."""
    fn_toks = set(_norm(fused_fn))
    if not fn_toks:
        return False
    for api in existing_apis or []:
        api_toks = set(_norm(api.get("name")))
        if not api_toks:
            continue
        # the fused_fn's distinctive tokens (drop the generic 'aiter') are a subset of
        # the API name's tokens, or vice-versa — either direction is a match.
        core = fn_toks - {"aiter"}
        acore = api_toks - {"aiter"}
        if core and (core <= api_toks or acore <= fn_toks):
            return True
    return False


def _candidate_shapes(candidate):
    """The set of captured member input-dim rows (as tuples) the candidate legitimately
    operates on. A verdict must have tested ONE of these."""
    shapes = set()
    for member in candidate.get("members", []) or []:
        dims = ((member.get("shape") or {}).get("input_dims")) or []
        for row in dims:
            if isinstance(row, list) and row:
                shapes.add(tuple(int(x) for x in row))
    return shapes


def _tested_shape_tuple(tested_shape):
    """Accept [tokens, hidden] or [[..],[..]]; return the primary 2-D row as a tuple,
    or None if unparseable."""
    if not isinstance(tested_shape, list) or not tested_shape:
        return None
    first = tested_shape[0]
    row = first if isinstance(first, list) else tested_shape
    try:
        return tuple(int(x) for x in row)
    except (TypeError, ValueError):
        return None


# tier-C: the candidate has no existing fused kernel, so there is nothing to
# microbench against the split reference. Legitimately out of 单侧 scope -- but it is
# REPORTED (deferred_author), never dropped from the denominator silently.
DEFERRED_CLASSES = ("new_helper_kernel",)

ROW_FIELDS = ("candidate_id", "family", "phase", "tier_hint", "unit_side_status",
              "correctness_status", "perf_status", "failure_kind",
              "reason", "parity", "isolated_speedup", "speedup_valid", "ref_ms",
              "cand_ms", "engaged", "tested_shape", "fused_fn", "tp", "tol",
              "ref_ops", "outside_work_removed", "parity_diagnosis")


def _in_scope(candidate):
    """True iff this candidate is 单侧-testable: it cites an existing fused API.

    tier-C (author a new kernel) has nothing to bench -> out of scope (deferred),
    not missing. Everything else MUST produce a verdict or an explicit waiver."""
    if str(candidate.get("implementation_class") or "") in DEFERRED_CLASSES:
        return False
    return bool(candidate.get("existing_apis"))


def _member_ops(candidate):
    """Every op name the captured region runs, as the reference must reproduce it.

    Both spellings are collected because members carry `parent_operator` (the aten/
    python-level op) and `kernel` (the device kernel); a reference leg may legitimately
    be described in either vocabulary, so a match on EITHER counts."""
    ops = set()
    for member in candidate.get("members", []) or []:
        for key in ("parent_operator", "kernel"):
            value = member.get(key)
            if value:
                ops.add(str(value))
    return ops


def _op_covered(needle, haystack):
    """Loose containment: trace op names carry template/overload suffixes that a
    verdict's prose name will not repeat verbatim. Substring in either direction, on
    the lowercased short name, is the honest match -- a stricter test would reject
    correct verdicts and train authors to copy trace strings instead of thinking."""
    def norm(value):
        # Strip everything but alphanumerics: a verdict author writes "rmsnorm" while
        # the trace says "aten::rms_norm", and a gate that rejects that pairing is a
        # gate that gets bypassed on its first real use. Punctuation carries no
        # meaning here, so it must not carry a veto.
        return "".join(ch for ch in str(value).lower().split("(")[0]
                       if ch.isalnum())
    n = norm(needle)
    if not n:
        return True
    for hay in haystack:
        h = norm(hay)
        if not h:
            continue
        if n in h or h in n:
            return True
    return False


def _row(cid, candidate, status, reason, **extra):
    """One result row with a uniform schema, so coverage rows (which have no
    measurement) render alongside verdict rows without KeyErrors."""
    row = {f: None for f in ROW_FIELDS}
    row.update({
        "candidate_id": cid,
        "family": candidate.get("family") if candidate else None,
        "phase": candidate.get("phase") if candidate else None,
        "tier_hint": candidate.get("implementation_class") if candidate else None,
        "unit_side_status": status,
        "reason": reason,
    })
    row.update(extra)
    return row


def _phase_generalization(results, candidate_list, phase_waivers):
    """A kernel that WINS in one phase must be answered for in every other phase.

    The single worst miss of the DSR1 blind test was not a kernel nobody knew about.
    `fused_qk_rope_concat_and_cache_mla` was in the pool, was benched, and PASSED at
    2.76x parity-exact -- in prefill, on [7238,512], because the candidate that claimed
    it was a prefill candidate and the shape-provenance gate (correctly) pins a verdict
    to its candidate's captured shapes. Prefill was 2.48% of wall clock, so both rows
    were deferred by a correct rule applied to the wrong object. The same kernel in
    decode is worth +20.61% e2e. Nothing anywhere went red, because coverage is
    computed over CANDIDATES and every candidate had a verdict.

    So this closes a second denominator: over (winning kernel x phase). It is
    workload-independent -- it needs no knowledge of which kernels matter, only the
    observation that a kernel proven good somewhere deserves an explicit answer
    everywhere. `--phase-waiver <fused_fn>@<phase>=<reason>` is the answer when the
    seam genuinely does not exist in that phase."""
    tested = {}
    winners = {}
    for row in results:
        fn = row.get("fused_fn")
        if not fn:
            continue
        short = str(fn).split(".")[-1]
        tested.setdefault(short, set()).add(str(row.get("phase")))
        if row.get("unit_side_status") == "pass":
            winners.setdefault(short, []).append(row.get("candidate_id"))
    gaps = []
    for fn, cids in sorted(winners.items()):
        # Generalize only across phases where THIS fused function has a concrete
        # candidate seam. The old global phase set required every winning decode
        # kernel to be benched in prefill merely because some unrelated prefill
        # candidate existed, making valid runs fail without a possible test.
        applicable_phases = sorted({
            str(candidate.get("phase"))
            for candidate in candidate_list
            if candidate.get("phase") and
            _fn_matches_api(fn, candidate.get("existing_apis"))
        })
        for phase in applicable_phases:
            if phase in tested.get(fn, set()):
                continue
            key = "%s@%s" % (fn, phase)
            gaps.append({
                "fused_fn": fn, "phase": phase,
                "won_in": sorted(tested.get(fn, set())),
                "won_as": sorted(set(cids)),
                "waiver": phase_waivers.get(key),
                "open": key not in phase_waivers,
            })
    return {
        "phases": sorted({str(c.get("phase")) for c in candidate_list
                           if c.get("phase")}),
        "winning_kernels": sorted(winners),
        "gaps": gaps,
        "open_gaps": [g for g in gaps if g["open"]],
    }


def validate(candidates_path, verdicts_path, min_speedup=1.0,
             require_coverage=True, waivers=None, legacy_schema_ok=False,
             phase_waivers=None, require_phase_generalization=True,
             allow_shapeless_candidate="", topk_path=None,
             subsumed=None, equivalent=None, budget_skipped=None):
    payload = _load(candidates_path)
    candidate_list = payload.get("candidates", [])
    candidates = {c["candidate_id"]: c for c in candidate_list}
    topk_ids = None
    candidate_exec = {}
    if topk_path:
        topk = _load(topk_path)
        topk_ids = {
            cid
            for entry in (topk.get("execution_list") or [])
            for cid in (entry.get("candidate_ids") or [])
        }
        candidate_exec = {
            cid: entry.get("exec_id")
            for entry in (topk.get("execution_list") or [])
            for cid in (entry.get("candidate_ids") or [])
        }
    verdicts = _load_verdicts(verdicts_path)
    waivers = dict(waivers or {})
    phase_waivers = dict(phase_waivers or {})
    # Two dispositions that used to be laundered through --waive, split apart
    # because they are opposites:
    #   subsumed  = COVERED. Its ladder-top superset was benched and PASSED;
    #               that microbench exercised every row this rung would remove.
    #   budget_skipped = NOT COVERED. We ran out of unit-side slots. On DSR1
    #               2026-09-03 seventeen such rows were passed in as --waive,
    #               rendered `waived`, dropped out of `unvalidated`, and the
    #               gate reported complete:true over a board where the ★★★
    #               oproj candidate had never been measured. 「没测」不得渲染
    #               成「测过了」— a skipped row now counts against coverage.
    subsumed = dict(subsumed or {})
    equivalent = dict(equivalent or {})
    budget_skipped = dict(budget_skipped or {})

    errors = []
    results = []
    # Every candidate_id a verdict was SUBMITTED for -- including ones whose
    # verdict turned out untrustworthy. An untrustworthy verdict is a loud error
    # already; it must not ALSO be reported as a silent coverage gap.
    submitted = set()
    for index, verdict in enumerate(verdicts):
        vp = "verdict[%d]" % index
        cid = verdict.get("candidate_id")
        if cid:
            submitted.add(cid)
        # 1. well-formed
        missing = [f for f in REQUIRED_VERDICT_FIELDS if verdict.get(f) is None]
        if missing:
            errors.append("%s missing fields %s" % (vp, missing))
            continue
        vp = "verdict[%s]" % cid
        # 2. known candidate (provenance)
        candidate = candidates.get(cid)
        if candidate is None:
            errors.append("%s references unknown candidate_id" % vp)
            continue
        family = str(candidate.get("family") or "")
        # 3. fused_fn is one of the candidate's declared APIs (anti-cheat)
        if not _fn_matches_api(verdict.get("fused_fn"),
                               candidate.get("existing_apis")):
            errors.append(
                "%s fused_fn '%s' is not one of the candidate's existing_apis %s"
                % (vp, verdict.get("fused_fn"),
                   [a.get("name") for a in candidate.get("existing_apis", [])]))
            continue
        # 4. shape provenance — the microbench must have tested the shape the candidate
        #    actually ran on (a pass on a different shape is meaningless). Two cases:
        #    (a) members carry exact captured input_dims (e.g. prefill kernel_exact):
        #        tested_shape MUST be one of them (strict).
        #    (b) members carry no dims (e.g. decode runtime_probe_wrapper): fall back to
        #        the selected_bucket token count — tested_shape's leading (token) dim
        #        MUST equal batch_size (decode) / input_tokens (prefill). This still
        #        stops a pass faked on a different token count.
        tested = _tested_shape_tuple(verdict.get("tested_shape"))
        cand_shapes = _candidate_shapes(candidate)
        if tested is None:
            errors.append("%s tested_shape %s is unparseable"
                          % (vp, verdict.get("tested_shape")))
            continue
        if cand_shapes:
            if tested not in cand_shapes:
                errors.append(
                    "%s tested_shape %s is not among the candidate's captured member "
                    "shapes %s (cannot trust a 单侧 pass on a different shape)"
                    % (vp, verdict.get("tested_shape"), sorted(cand_shapes)))
                continue
        elif not allow_shapeless_candidate:
            # 2026-09-01. This branch used to fall through to the loose token-count
            # check below, which means: when the members carried NO shape, the shape
            # provenance gate silently became no gate. That is not a hypothetical --
            # it is the 08-31 run. With nothing to check against, a microbench built
            # on [4,512] (= input_tokens x kv_lora_rank, no batch axis, no rope
            # columns) sailed through, the kernel took a degenerate path, and the
            # parity fail plus distorted timing blocked two fusions worth +20.6% and
            # +14.0% e2e. A gate that disappears exactly when its input is missing is
            # worse than no gate, because it reports success.
            errors.append(
                "%s the candidate carries no captured member shape, so tested_shape "
                "%s cannot be checked against anything and this verdict is not "
                "trustworthy at any speedup. Re-run fusion_candidate_harness.py "
                "against the real semantic table (it grafts member shapes by "
                "row_id), or pass --allow-shapeless-candidate REASON to accept the "
                "weaker token-count-only check knowingly."
                % (vp, verdict.get("tested_shape")))
            continue
        else:
            bucket = candidate.get("selected_bucket") or {}
            expect_tok = (bucket.get("batch_size")
                          if candidate.get("phase") == "decode"
                          else bucket.get("input_tokens"))
            if expect_tok and int(tested[0]) != int(expect_tok):
                errors.append(
                    "%s tested_shape leading dim %d != selected_bucket token count %d "
                    "(cannot trust a 单侧 pass on a different token count)"
                    % (vp, tested[0], int(expect_tok)))
                continue
        # 5. REFERENCE SCOPE (added 2026-09-01). `isolated_speedup` is a ratio whose
        #    numerator the verdict chose. If the reference leg is narrower than what
        #    the fused kernel actually replaces, the ratio is not "the speedup, a bit
        #    pessimistic" -- it is the speedup of a different question. Observed on
        #    DSR1: a V-absorb fusion measured 0.698 against a reference that omitted
        #    the per-step weight dequant the fusion deletes; the KB records the same
        #    fusion at +10.34% e2e. A correctly-measured number of the wrong quantity
        #    reads exactly like a real loss, and terminates the candidate.
        ref_ops = verdict.get("ref_ops")
        outside = verdict.get("outside_work_removed")
        if not legacy_schema_ok:
            scope_missing = [f for f in SCOPE_VERDICT_FIELDS
                             if verdict.get(f) is None]
            if scope_missing:
                errors.append(
                    "%s missing %s — the split reference is unattributed. State which "
                    "ops the ref leg ran, and which work the fusion removes from "
                    "OUTSIDE the captured member set ([] if none). See "
                    "fusion_unit_validator.md step 2b." % (vp, scope_missing))
                continue
        if isinstance(ref_ops, list) and ref_ops:
            member_ops = _member_ops(candidate)
            uncovered = sorted(op for op in member_ops
                               if not _op_covered(op, ref_ops))
            # Only removable members are load-bearing: a member the fusion does NOT
            # remove may legitimately sit outside the timed reference.
            removable = set(candidate.get("removable_row_ids") or [])
            if removable:
                removable_ops = set()
                for member in candidate.get("members", []) or []:
                    if member.get("row_id") in removable:
                        for key in ("parent_operator", "kernel"):
                            if member.get(key):
                                removable_ops.add(str(member[key]))
                uncovered = sorted(op for op in removable_ops
                                   if not _op_covered(op, ref_ops))
            if uncovered:
                errors.append(
                    "%s ref_ops %s does not cover the removable member ops %s — the "
                    "reference is narrower than the region the fusion replaces, so "
                    "isolated_speedup understates by an unknown factor"
                    % (vp, ref_ops, uncovered))
                continue
        if isinstance(outside, list) and outside:
            unmeasured = sorted(op for op in outside
                                if not _op_covered(op, ref_ops or []))
            if unmeasured:
                errors.append(
                    "%s declares the fusion removes out-of-region work %s, but the "
                    "reference leg (ref_ops %s) does not run it — either time it in "
                    "the ref or explain in outside_work_removed why it is not "
                    "attributable" % (vp, unmeasured, ref_ops))
                continue

        # ---- verdict is TRUSTWORTHY; derive the gate from its content ------------
        try:
            speedup = float(verdict.get("isolated_speedup"))
        except (TypeError, ValueError):
            errors.append("%s isolated_speedup is not a number" % vp)
            continue
        parity = str(verdict.get("parity")).lower()
        is_collective = family.startswith("collective")
        engaged = verdict.get("engaged")
        status, reason = "pass", ""
        correctness, perf, failure_kind = "pass", "win", "none"
        speedup_valid = True
        if is_collective and engaged is False:
            status = "blocked"
            correctness, perf, failure_kind = "not_engaged", "undefined", "not_engaged"
            reason = ("fused collective path did not engage at this shape "
                      "(size-guard fallback to split) — nothing to apply here")
        elif parity != "pass":
            # 2026-09-01: this used to be `fail`, and `fail` is terminal on the board.
            # But a fused kernel whose output diverges was, with high probability, FED
            # WRONG (layout, scale orientation, group_size, fnuz-vs-fn, a dtype the
            # kernel silently reinterprets). A mis-fed kernel also takes a wrong path
            # and a wrong tile config -- so its `cand_ms` is the time of something that
            # was never the candidate. Reporting "parity failed AND it was slow" as two
            # independent strikes double-counts ONE fault. Observed on DSR1: a rope+KV
            # fusion recorded 0.4849 with parity fail and was closed; the KB records the
            # same kernel, same shape, same arch at 1.11--1.12x parity-exact.
            #
            # So: parity != pass makes isolated_speedup UNDEFINED, not bad. The
            # candidate is not refuted, it is unmeasured, and the harness FAILS until
            # someone either fixes the feeding or records the elimination.
            status = "needs_diagnosis"
            correctness, perf = "fail", "undefined"
            # `functional` only once the mundane causes are on the record. Until then
            # nobody has earned the right to call this a functional failure -- it is
            # simply unmeasured, and that distinction is the entire lesson of 08-31.
            failure_kind = ("functional" if verdict.get(DIAGNOSIS_FIELD)
                            else "unmeasured")
            speedup_valid = False
            reason = ("parity != pass — isolated_speedup %.3f is UNDEFINED, not slow "
                      "(a mis-fed kernel takes a wrong path and mis-times). Diagnose "
                      "the feeding before this number is allowed to mean anything."
                      % speedup)
            if not verdict.get(DIAGNOSIS_FIELD):
                errors.append(
                    "%s parity != pass with no %s — list the mundane causes you "
                    "eliminated (weight contiguity/layout, scale orientation, "
                    "group_size, fnuz-vs-fn fp8, two independent fp8 quantizations, "
                    "wrong q_out_dtype). fusion_unit_validator.md:137 requires this; "
                    "it is now enforced." % (vp, DIAGNOSIS_FIELD))
        elif speedup <= min_speedup:
            status = "fail"
            # 功能正确，纯性能不划算 — the one failure that is safe to close on the spot.
            correctness, perf, failure_kind = "pass", "no_win", "performance"
            reason = ("PERF failure (功能正确): parity pass, isolated_speedup %.3f "
                      "<= %.3f — measured on the candidate's own captured shape, so "
                      "the number means what it says. Do not apply back."
                      % (speedup, min_speedup))
        else:
            reason = ("parity pass, isolated_speedup %.3fx%s"
                      % (speedup, ", engaged" if is_collective else ""))
        results.append({
            "candidate_id": cid,
            "family": family,
            "phase": candidate.get("phase"),
            "tier_hint": candidate.get("implementation_class"),
            "unit_side_status": status,
            "correctness_status": correctness,
            "perf_status": perf,
            "failure_kind": failure_kind,
            "reason": reason,
            "parity": parity,
            # Nulled, not hidden: the raw ratio stays under `isolated_speedup_raw` so
            # the trail is auditable, but nothing downstream can read an undefined
            # number out of the field it normally reads.
            "isolated_speedup": round(speedup, 4) if speedup_valid else None,
            "isolated_speedup_raw": round(speedup, 4),
            "speedup_valid": speedup_valid,
            "ref_ops": verdict.get("ref_ops"),
            "outside_work_removed": verdict.get("outside_work_removed"),
            "parity_diagnosis": verdict.get(DIAGNOSIS_FIELD),
            "ref_ms": verdict.get("ref_ms"),
            "cand_ms": verdict.get("cand_ms"),
            "engaged": engaged,
            "tested_shape": verdict.get("tested_shape"),
            "fused_fn": verdict.get("fused_fn"),
            "tp": verdict.get("tp"),
            "tol": verdict.get("tol"),
        })

    # ---- COVERAGE (recall): close the loop over CANDIDATES, not verdicts ------
    # Up to here the loop was verdict-driven, so a candidate nobody benched was
    # invisible. Walk the candidate list and give every in-scope candidate an
    # explicit disposition.
    coverage_rows = []
    unvalidated = []
    waived_unknown = sorted(set(waivers) - set(candidates))
    for candidate in candidate_list:
        cid = candidate.get("candidate_id")
        if not cid or cid in submitted:
            continue
        if topk_ids is not None and cid not in topk_ids:
            coverage_rows.append(_row(
                cid, candidate, "deferred_rank_budget",
                "candidate is outside fusion_topk.execution_list; not in the "
                "unitside coverage denominator"))
            continue
        if not _in_scope(candidate):
            coverage_rows.append(_row(
                cid, candidate, "deferred_author",
                "no existing fused kernel (tier-C, 需自写) — out of 单侧 scope, "
                "counted but not benched"))
            continue
        if cid in equivalent:
            representative = equivalent[cid]
            representative_row = next((
                row for row in results
                if row.get("candidate_id") == representative), None)
            representative_status = (
                representative_row.get("unit_side_status")
                if representative_row else None)
            if representative_status not in {
                    "pass", "fail", "blocked", "needs_diagnosis"}:
                coverage_rows.append(_row(
                    cid, candidate, "not_validated",
                    "declared equivalent to %s, but that representative did not "
                    "produce a trustworthy verdict" % representative))
                unvalidated.append(cid)
            else:
                equivalent_status = "equivalent_%s" % representative_status
                coverage_rows.append(_row(
                    cid, candidate, equivalent_status,
                    "same recipe/cohort inherits representative %s result: %s"
                    % (representative_status, representative)))
            continue
        if cid in subsumed:
            coverage_rows.append(_row(
                cid, candidate, "subsumed_pass",
                "covered by its ladder top: %s" % subsumed[cid]))
            continue
        if cid in budget_skipped:
            coverage_rows.append(_row(
                cid, candidate, "budget_skipped",
                "单侧预算耗尽，NEVER MEASURED（不是豁免）：%s" % budget_skipped[cid]))
            unvalidated.append(cid)
            continue
        if cid in waivers:
            coverage_rows.append(_row(
                cid, candidate, "waived",
                "explicitly waived: %s" % waivers[cid]))
            continue
        coverage_rows.append(_row(
            cid, candidate, "not_validated",
            "NO verdict submitted — this candidate never reached the microbench"))
        unvalidated.append(cid)
    results.extend(coverage_rows)
    status_by_id = {
        row.get("candidate_id"): row.get("unit_side_status")
        for row in results if row.get("candidate_id")}
    for cid in waived_unknown:
        errors.append("--waive references unknown candidate_id '%s'" % cid)
    for flag, mapping in (("--equivalent", equivalent),
                          ("--subsumed", subsumed),
                          ("--budget-skipped", budget_skipped)):
        for cid in sorted(set(mapping) - set(candidates)):
            errors.append("%s references unknown candidate_id '%s'" % (flag, cid))
    for cid, representative in sorted(equivalent.items()):
        if representative not in candidates:
            errors.append(
                "--equivalent for '%s' references unknown representative '%s'"
                % (cid, representative))
        elif candidate_exec and candidate_exec.get(cid) != candidate_exec.get(representative):
            errors.append(
                "--equivalent requires candidate and representative in the same "
                "execution_list row: %s=%s" % (cid, representative))
    # A row cannot be both covered and never measured, or claim two coverage modes.
    for cid in sorted((set(subsumed) | set(equivalent)) & set(budget_skipped)):
        errors.append(
            "'%s' is marked covered and --budget-skipped; it is either covered "
            "by a passing test or it was not measured" % cid)
    for cid in sorted(set(subsumed) & set(equivalent)):
        errors.append(
            "'%s' is marked BOTH --subsumed and --equivalent" % cid)

    in_scope = [
        c for c in candidate_list
        if _in_scope(c) and
        (topk_ids is None or c.get("candidate_id") in topk_ids)
    ]
    by_phase = {}
    for candidate in in_scope:
        ph = str(candidate.get("phase") or "?")
        slot = by_phase.setdefault(ph, {"in_scope": 0, "validated": 0,
                                        "waived": 0, "subsumed_pass": 0,
                                        "equivalent_covered": 0,
                                        "equivalent_pass": 0,
                                        "budget_skipped": 0,
                                        "not_validated": 0})
        slot["in_scope"] += 1
        cid = candidate.get("candidate_id")
        status = status_by_id.get(cid)
        if cid in submitted:
            slot["validated"] += 1
        elif str(status or "").startswith("equivalent_"):
            slot["equivalent_covered"] += 1
            if status == "equivalent_pass":
                slot["equivalent_pass"] += 1
        elif status == "subsumed_pass":
            slot["subsumed_pass"] += 1
        elif status == "budget_skipped":
            slot["budget_skipped"] += 1
        elif status == "waived":
            slot["waived"] += 1
        else:
            slot["not_validated"] += 1
    coverage = {
        "required": require_coverage,
        "in_scope": len(in_scope),
        "validated": sum(1 for c in in_scope if c.get("candidate_id") in submitted),
        "waived": sum(1 for c in in_scope
                      if status_by_id.get(c.get("candidate_id")) == "waived"),
        # Covered without its own microbench: the ladder top above it passed.
        "subsumed_pass": sum(1 for c in in_scope
                             if status_by_id.get(c.get("candidate_id"))
                             == "subsumed_pass"),
        "equivalent_pass": sum(1 for c in in_scope
                                if status_by_id.get(c.get("candidate_id"))
                                == "equivalent_pass"),
        "equivalent_covered": sum(
            1 for c in in_scope
            if str(status_by_id.get(c.get("candidate_id")) or "").startswith(
                "equivalent_")),
        # NOT covered. Counted inside not_validated as well, on purpose.
        "budget_skipped": sum(1 for c in in_scope
                              if status_by_id.get(c.get("candidate_id"))
                              == "budget_skipped"),
        "not_validated": len(unvalidated),
        "not_validated_ids": unvalidated,
        "deferred_author": sum(1 for c in candidate_list if not _in_scope(c)),
        "candidates_total": len(candidate_list),
        "topk_candidate_ids": sorted(topk_ids) if topk_ids is not None else None,
        "deferred_rank_budget": sum(
            1 for c in candidate_list
            if topk_ids is not None and c.get("candidate_id") not in topk_ids),
        "by_phase": by_phase,
        "complete": not unvalidated,
    }

    counts = {"pass": 0, "fail": 0, "blocked": 0, "needs_diagnosis": 0,
              "not_validated": 0, "waived": 0, "subsumed_pass": 0,
              "equivalent_pass": 0, "equivalent_fail": 0,
              "equivalent_blocked": 0,
              "equivalent_needs_diagnosis": 0,
              "budget_skipped": 0, "deferred_author": 0,
              "deferred_rank_budget": 0}
    for r in results:
        counts[r["unit_side_status"]] = counts.get(r["unit_side_status"], 0) + 1

    coverage_fail = bool(require_coverage and unvalidated)
    phase_candidates = [
        c for c in candidate_list
        if topk_ids is None or c.get("candidate_id") in topk_ids
    ]
    phase_gen = _phase_generalization(results, phase_candidates, phase_waivers)
    phase_fail = bool(require_phase_generalization and phase_gen["open_gaps"])
    for gap in (phase_gen["open_gaps"] if require_phase_generalization else []):
        errors.append(
            "phase-generalization gap: %s PASSED 单侧 in %s (as %s) but was never "
            "benched in phase '%s'. A kernel proven good in one phase must be answered "
            "for in every phase present in the candidate pool — bench it there, or "
            "--phase-waiver %s@%s='<why no such seam in %s>'."
            % (gap["fused_fn"], "/".join(gap["won_in"]), ",".join(gap["won_as"]),
               gap["phase"], gap["fused_fn"], gap["phase"], gap["phase"]))
    # An UNDIAGNOSED divergence fails the gate. A DIAGNOSED one does not: the demand
    # is the elimination record, not a successful fusion. The row stays
    # `needs_diagnosis` either way -- it is never eligible for apply-back, and it is
    # never reported as a `fail`, because a kernel that was fed wrong was not refuted.
    needs_diag = [r["candidate_id"] for r in results
                  if r.get("unit_side_status") == "needs_diagnosis"
                  and not r.get("parity_diagnosis")]
    for cid in needs_diag:
        errors.append(
            "%s is needs_diagnosis with no elimination record (parity failed → its "
            "timing is undefined). It is NOT a fail and must not be closed as one: "
            "re-feed the kernel, or record what you eliminated in parity_diagnosis."
            % cid)
    result = {
        "schema_version": 2,
        "phase": "unitside_gate",
        "status": "fail" if (errors or coverage_fail or phase_fail) else "pass",
        "min_speedup": min_speedup,
        "candidates_json": os.path.abspath(candidates_path),
        "errors": errors,
        "coverage": coverage,
        "coverage_ok": not coverage_fail,
        "phase_generalization": phase_gen,
        "phase_generalization_ok": not phase_fail,
        "counts": counts,
        "verdict_count": len(verdicts),
        "results": results,
    }
    return result


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


def render_markdown(result):
    lines = ["# Kernel Fusion 单侧 Gate (Phase 3.0)", ""]
    lines.append(
        "隔离验证每个 fusion 的 **fused kernel vs split 参考**：正确性(parity) + "
        "isolated speedup。**`pass` 才可进 apply-back**；`blocked` = 该 shape 下 "
        "fused 路径未生效(size-guard 回退)，非失败；`fail` = 无收益/不正确。")
    lines.append("")
    c = result["counts"]
    cov = result.get("coverage") or {}
    pg = result.get("phase_generalization") or {}
    if c.get("needs_diagnosis"):
        lines.append(
            "> 🔴 **%d 条 `needs_diagnosis`**：parity 未通过 ⇒ 该条的 "
            "`isolated_speedup` 是**未定义**，不是「慢」。喂错的 kernel 会走错分支、"
            "选错 tile，测出来的时间不属于这个候选。这些候选**没有被证伪**，只是"
            "**没测成**，不得当作 `fail` 结案。" % c["needs_diagnosis"])
        lines.append("")
    if pg.get("open_gaps"):
        lines.append(
            "> 🔴 **(获胜 kernel × phase) 缺口 %d 个**：%s。某个 kernel 在一个 phase "
            "上已经证明是赢的，就必须在候选池出现过的每个 phase 上给出答复——要么去测，"
            "要么写明那个 phase 没有这个缝。"
            % (len(pg["open_gaps"]),
               "、".join("`%s` 缺 %s（已在 %s 通过）"
                        % (g["fused_fn"], g["phase"], "/".join(g["won_in"]))
                        for g in pg["open_gaps"])))
        lines.append("")
    # Coverage FIRST: the denominator is the headline. A report that leads with
    # "16 verdict, 16 pass" reads as complete at 38% coverage -- that is exactly the
    # failure this section exists to make impossible.
    if cov:
        lines.append(
            "**覆盖率**：在范围内候选 **%d** / 已验证 **%d** / 等价代测 **%d**（其中通过 %d）/ 阶梯代测 **%d** / "
            "未验证 **%d**（其中预算跳过 %d）/ 豁免 %d"
            "；tier-C 待自写 %d（不计入范围）；Top-K 外延后 %d；候选总数 %d。"
            % (cov.get("in_scope", 0), cov.get("validated", 0),
               cov.get("equivalent_covered", 0),
               cov.get("equivalent_pass", 0),
               cov.get("subsumed_pass", 0),
               cov.get("not_validated", 0), cov.get("budget_skipped", 0),
               cov.get("waived", 0),
               cov.get("deferred_author", 0),
               cov.get("deferred_rank_budget", 0),
               cov.get("candidates_total", 0)))
        by_phase = cov.get("by_phase") or {}
        if by_phase:
            lines.append("")
            lines.append("分阶段覆盖：" + "；".join(
                "%s **%d/%d**" % (ph, v.get("validated", 0) + v.get("waived", 0)
                                  + v.get("equivalent_covered", 0)
                                  + v.get("subsumed_pass", 0),
                                  v.get("in_scope", 0))
                for ph, v in sorted(by_phase.items())) + "。")
        lines.append("")
        if cov.get("budget_skipped"):
            lines.append(
                "> 🔴 **%d 条因单侧预算耗尽从未测过**。这不是豁免，也不是"
                "「阶梯代测」：没有任何 microbench 碰过它们。要么提高 "
                "`fusion_unitside_budget`，要么把它们挂到某条已 pass 的阶梯顶端"
                "（`--subsumed`），不得用 `--waive` 洗成「已交代」。"
                % cov.get("budget_skipped", 0))
            lines.append("")
        if cov.get("not_validated"):
            lines.append(
                "> 🔴 **覆盖率不完整**：%d 条在范围内的候选从未提交 verdict。"
                "单侧结论对它们**没有任何结论**——不是 pass，也不是 fail，是没测。"
                "补测或用 `--waive <id>=<理由>` 显式豁免。"
                % cov.get("not_validated", 0))
            lines.append("")
    lines.append("总计 %d 条 verdict：pass %d / fail %d / blocked %d；harness status=**%s**（错误 %d）。"
                 % (result["verdict_count"], c.get("pass", 0), c.get("fail", 0),
                    c.get("blocked", 0), result["status"], len(result["errors"])))
    lines.append("")
    kinds = {}
    for r in result["results"]:
        k = r.get("failure_kind")
        if k and k != "none":
            kinds[k] = kinds.get(k, 0) + 1
    if kinds:
        lines.append("失败按**种类**拆开：%s。"
                     "只有 `performance` 是**功能正确、纯粹不够快**，可以就地结案；"
                     "`functional` 是排除了喂法问题之后仍然算错；"
                     "`unmeasured` 不是结论，是还没人看过——它让 harness 失败。"
                     % "、".join("**%s** %d" % (k, v) for k, v in sorted(kinds.items())))
        lines.append("")
    lines.append("| 候选 | 阶段 | family | 单侧结论 | correctness | perf | 失败种类 | parity | isolated speedup | engaged | tested shape | fused fn | 说明 |")
    lines.append("|---|:--:|---|:--:|:--:|:--:|:--:|:--:|---:|:--:|---|---|---|")
    for r in result["results"]:
        sp = ("%.3fx" % r["isolated_speedup"]) if r["isolated_speedup"] is not None else "-"
        fk = r.get("failure_kind") or "—"
        lines.append("| `%s` | %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | `%s` | %s |" % (
            _esc(r["candidate_id"]), _esc(r["phase"]), _esc(r["family"]),
            r["unit_side_status"], _esc(r.get("correctness_status") or "—"),
            _esc(r.get("perf_status") or "—"), _esc(fk),
            _esc(r["parity"]), sp, _esc(r["engaged"]),
            _esc(r["tested_shape"]), _esc(r["fused_fn"]), _esc(r["reason"])))
    lines.append("")
    gaps = [r for r in result["results"]
            if r.get("unit_side_status") in ("not_validated", "budget_skipped")]
    if gaps:
        lines.append("## 覆盖率缺口：未验证候选（必须补 verdict 或显式豁免）")
        lines.append("")
        lines.append("| 候选 | 阶段 | family | tier | 说明 |")
        lines.append("|---|:--:|---|---|---|")
        for r in gaps:
            lines.append("| `%s` | %s | %s | %s | %s |" % (
                _esc(r["candidate_id"]), _esc(r["phase"]), _esc(r["family"]),
                _esc(r["tier_hint"]), _esc(r["reason"])))
        lines.append("")
    if result["errors"]:
        lines.append("## 不可信 verdict（harness 错误，需按报错重跑，不得当作 pass）")
        lines.append("")
        for e in result["errors"]:
            lines.append("- %s" % e)
        lines.append("")
    lines.append(
        "说明：本 gate 只校验 microbench verdict 的**可信度**（字段完整 + 候选存在 + "
        "tested_shape 属于候选抓到的 member shape + fused_fn 属于候选 existing_apis + "
        "collective 是否真生效）并据此判 pass/fail/blocked；它不跑 kernel、不产生自己的 "
        "perf 数字。isolated speedup / parity 由 fusion_unit_validator 的隔离 microbench 实测。"
        "**并且**校验覆盖率：每条在范围内的候选都必须有 verdict 或显式豁免，"
        "否则 `not_validated` 且 harness fail——「没测」不得渲染成「测过了」。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_waivers(pairs, flag="--waive"):
    """--waive dc_kv=needs paged-KV state -> {"dc_kv": "needs paged-KV state"}.
    A waiver without a reason is rejected: "skipped" is not a reason."""
    out = {}
    for item in pairs or []:
        cid, sep, reason = str(item).partition("=")
        cid, reason = cid.strip(), reason.strip()
        if not sep or not cid or not reason:
            raise SystemExit(
                "%s must be <candidate_id>=<reason> with a non-empty reason; "
                "got %r" % (flag, item))
        out[cid] = reason
    return out


def run(candidates_path, verdicts_path, out_md, out_json, min_speedup=1.0,
        require_coverage=True, waivers=None, legacy_schema_ok=False,
        phase_waivers=None, require_phase_generalization=True,
        allow_shapeless_candidate="", topk_path=None,
        subsumed=None, equivalent=None, budget_skipped=None):
    result = validate(candidates_path, verdicts_path, min_speedup,
                      require_coverage=require_coverage, waivers=waivers,
                      legacy_schema_ok=legacy_schema_ok,
                      phase_waivers=phase_waivers,
                      require_phase_generalization=require_phase_generalization,
                      allow_shapeless_candidate=allow_shapeless_candidate,
                      topk_path=topk_path, subsumed=subsumed,
                      equivalent=equivalent,
                      budget_skipped=budget_skipped)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--topk", default=None,
        help="fusion_topk.json; when set, coverage is limited to concrete "
             "candidate_ids in execution_list")
    parser.add_argument("--verdicts", required=True,
                        help="dir of per-candidate *.json verdicts OR a combined json")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="isolated_speedup must exceed this for a pass (default 1.0)")
    parser.add_argument("--allow-partial-coverage", action="store_true",
                        help="report the coverage gap without failing the gate "
                             "(the gap is printed loudly either way)")
    parser.add_argument("--waive", action="append", metavar="ID=REASON", default=[],
                        help="explicitly waive one candidate from the coverage "
                             "requirement, with a reason (repeatable)")
    parser.add_argument("--subsumed", action="append", metavar="ID=LADDER_TOP",
                        default=[],
                        help="this candidate's ladder-top superset was benched "
                             "and PASSED, so its removable rows are already "
                             "covered; counts as covered (repeatable)")
    parser.add_argument("--equivalent", action="append",
                        metavar="ID=REPRESENTATIVE_CANDIDATE", default=[],
                        help="same recipe/cohort as a passing representative; "
                             "counts as covered without another unit test")
    parser.add_argument("--budget-skipped", action="append",
                        metavar="ID=REASON", default=[],
                        help="this candidate never reached the microbench "
                             "because the unit-side budget ran out. Counts as "
                             "NOT covered and fails the gate — do not launder "
                             "it through --waive (repeatable)")
    parser.add_argument("--phase-waiver", action="append", default=[],
                        metavar="FUSED_FN@PHASE=REASON",
                        help="a kernel that passed 单侧 in one phase has no seam in "
                             "PHASE — state why (repeatable)")
    parser.add_argument("--allow-phase-gap", action="store_true",
                        help="report un-answered (winning kernel x phase) gaps "
                             "without failing the gate")
    parser.add_argument("--allow-legacy-verdict-schema", action="store_true",
                        help="accept verdicts written before ref_ops / "
                             "outside_work_removed existed (pre-2026-09-01 rounds). "
                             "The reference-scope blind spot is UNCHECKED when set")
    parser.add_argument("--allow-shapeless-candidate", metavar="REASON", default="",
                        help="accept verdicts whose candidate carries no captured "
                             "member shape, falling back to a token-count-only "
                             "check. Requires a REASON. This re-opens the exact "
                             "hole that cost 08-31 two fusions worth +20.6%% and "
                             "+14.0%% e2e — fix the semantic capture instead")
    args = parser.parse_args()
    if args.allow_shapeless_candidate and not args.allow_shapeless_candidate.strip():
        raise SystemExit("--allow-shapeless-candidate requires a non-empty reason")
    result = run(args.candidates, args.verdicts, args.out_md, args.out_json,
                 args.min_speedup,
                 require_coverage=not args.allow_partial_coverage,
                 waivers=_parse_waivers(args.waive),
                 legacy_schema_ok=args.allow_legacy_verdict_schema,
                 phase_waivers=_parse_waivers(args.phase_waiver),
                 require_phase_generalization=not args.allow_phase_gap,
                 allow_shapeless_candidate=args.allow_shapeless_candidate,
                 topk_path=args.topk,
                 subsumed=_parse_waivers(args.subsumed, "--subsumed"),
                 equivalent=_parse_waivers(args.equivalent, "--equivalent"),
                 budget_skipped=_parse_waivers(
                     args.budget_skipped, "--budget-skipped"))
    cov = result.get("coverage") or {}
    pg = result.get("phase_generalization") or {}
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "errors": len(result["errors"]),
                      "phase_generalization": {
                          "winning_kernels": len(pg.get("winning_kernels") or []),
                          "open_gaps": ["%s@%s" % (g["fused_fn"], g["phase"])
                                        for g in (pg.get("open_gaps") or [])]},
                      "coverage": {k: cov.get(k) for k in
                                   ("in_scope", "validated", "equivalent_covered",
                                    "equivalent_pass",
                                    "subsumed_pass",
                                    "budget_skipped", "not_validated",
                                    "waived", "deferred_author", "complete")}},
                     indent=2))
    if result["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
