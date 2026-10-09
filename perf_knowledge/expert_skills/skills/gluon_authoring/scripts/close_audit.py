#!/usr/bin/env python3
"""Surface the FACTS about a finished run that nobody is positioned to see.

WHY THIS EXISTS. Two rules in the orchestration layer are individually right and together leave a
hole. The captain is told to check artifacts by name and never walk a tree, because anything it
reads rides on every later turn's cache-read. The fleet is told, for a captain owner, to read that
owner's report and stop, because re-deriving a conclusion it already paid for is the most expensive
thing it can do. So a failure that is only visible by walking the tree and comparing timestamps --
an arm that finished after it was ranked, a tool nobody in the direction ever ran, the same shape
carrying two different numbers in two files -- is invisible to BOTH roles by construction.

The way out is not to relax either rule. It is that a script changes the economics: the reason for
"never walk a tree" is that what you read rides on your context, and a script's walk does not. This
returns a small JSON, so a supervisor can have the tree walked without paying to read the tree.

WHAT IT IS NOT. It is not a gate and it makes no judgments. By default it never exits non-zero on a
finding, and its output has no verdict field. Every check here answers a question of arithmetic or
of file order -- "is this record newer than the moment it was ranked", "does this number appear
twice with two values" -- and stops. Whether a given fact should block a close is policy, and
policy belongs with the reader who knows what was dispatched. Some findings carry a `severity`, and
that is still not a verdict: it separates a fact that CONTRADICTS a conclusion the run already
banked from one that records a contract not yet followed, because a reader who gets both as one
number reads the count and not the finding.

`--strict` DOES NOT MOVE THAT LINE, it lets the reader express its policy in an exit code. The
paragraph above stayed true and nothing ever failed: measured over one eight-kernel campaign the
single `close_audit.json` produced recorded a detected contract violation at `high_severity_count:
0` and every high finding was accepted with a sentence. So the policy still belongs to the reader
-- `kernel-opt-run` Phase 4 is the reader, and it now states its policy by passing `--strict`
rather than by reading a count it is free to ignore. Run bare and this is a report, exactly as
before.

IN GEAK this is optional bookkeeping a deep_engineer may use for its records, not a run mode.
Role ids such as `captain` / `deep` / `skeptic` are upstream record-schema identifiers (`deep` = the
deep_engineer, pass `--role deep`); nothing spawns them, and final arbitration in GEAK is Director's.

BLOCKED IS NOT NOT-APPLICABLE (guardrail R9). A check whose input is absent used to return
`applicable: False`, which reads identically to a check that correctly does not apply to this pack
-- so a missing artifact DISABLED its check instead of failing it. The two are now distinct:
`applicable: False` means this pack never declared the input (an NVIDIA pack has no
rocprof-compute), while `blocked: True` means the pack DOES declare the producing tool in
`pack_facts.artifacts.produces` and the artifact is not on disk. Blocked is a failure state and is
counted with the high findings under `--strict`.

HOW IT STAYS PORTABLE. It ships from core/, byte-identical in all seven packs, and reads
`scripts/pack_facts.json` (stamped by compose from the pack's manifest) for everything that differs
between them: which round contract this pack runs, what its artifacts are called, which tool fills
which evidence role, and whether it declares an entry gate or an authoring gap. Nothing in here
branches on a DSL name, and an unfilled evidence role reports as a stated tool gap rather than as a
tool that was skipped -- the two are opposite findings and the difference is the pack's, not the
run's.

Usage:
  close_audit.py --work-root <dir> [--as self|fleet] [--json close_audit.json]
  close_audit.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fnmatch
import glob
import json
import os
import re
import sys

from canonical_record import (
    ARM_RESULT_SCHEMA,
    BRANCH_CONVERGE_SCHEMA,
    BRANCH_REQUEST_SCHEMA,
    FINAL_REPORT_SCHEMA,
    RESWEEP_REQUEST_SCHEMA,
    RESWEEP_RESULT_SCHEMA,
    RUN_DECISION_SCHEMA,
    SWEEP_REQUEST_SCHEMA,
    SWEEP_RESULT_SCHEMA,
    WORKER_RESULT_SCHEMA,
    same_identity,
    project_final_report,
    validate_arm_result,
    validate_branch_request,
    validate_final_report,
    validate_resweep_request,
    validate_resweep_result,
    validate_run_decision,
    write_producer_receipt,
    validate_sweep_pair,
    validate_sweep_request,
    validate_sweep_result,
    validate_worker_result,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Roll-up artifacts that mark the moment the parent COLLECTED its children's work. An arm record
# written after the earliest of these was written was not in the comparison the parent made, however
# complete it looks now. Names come from both regimes because the audit does not know which one it
# is reading until it has read pack_facts.json -- and a run may carry a stray file from either.
COLLECTION_MARKERS = ("final_report.json", "worker_result.json", "t0_merge.json", "merge.json",
                      "decision_log.md")

# Where a fan-out arm's roll-up lands. Both regimes spell the parent dir differently (`exp/t0/` for
# the gated engine, `exp/branch/<stage>/` for the tool-first packs) and both may be nested, so the
# arm record is found by its OWN NAME rather than by the path above it. This used to also require
# an `arm_*` parent directory, which contradicted that: measured across a 9-kernel campaign, arms
# living in `arms/s2/`, `dir_arm1/` and `armC_atomicsplit/` were invisible to every check below --
# including the roster check whose whole job is noticing a missing arm.
ARM_GLOB = "**/arm_result.json"

# The headline number an arm is ranked by, under each of its spellings. A pack that renames this
# is a pack whose arms stop being comparable, which is worth finding out here rather than later.
ARM_HEADLINE_KEYS = ("best_geomean_vs_anchor", "best_geomean_vs_plain", "geomean_vs_anchor",
                     "geomean_vs_plain", "best_ms")

# Language that asserts a search is OVER -- not merely that something is hard. These are the claims
# that stop work, so they are the ones worth knowing the citation status of. Two deliberate
# narrowings, both learned from running this against real trees: a bare "cannot" matches ordinary
# prose and profiler chatter far more often than it matches a claim, and the scan runs over PROSE
# only (see _text_records(prose_only=True)) because a JSON field holding an enum has nowhere to put
# a citation -- flagging `"escalation": "stay_plain"` for lacking one is a category error, and a
# check that mostly fires on category errors is a check nobody reads.
CLAIM_RE = re.compile(
    r"(?:\bimpossible\b|\binexpressible\b|not\s+expressible|cannot\s+be\s+expressed|"
    r"\bat\s+(?:the\s+)?ceiling\b|\bat\s+the\s+wall\b|\bhard\s+wall\b|already\s+optimal|"
    r"nothing\s+left\b|no\s+headroom|(?:is|are|was|were)\s+exhausted|"
    r"\bstay[\s_-]plain\b|\bkeep[\s_-]baseline\b)", re.I)
# A citation marker: a file path, a line reference, or a number carrying a unit. Prose alone is not
# one. This is the same shape of test the packs already apply to a lever card's evidence.
CITE_RE = re.compile(r"(?:\b[\w./-]+\.(?:json|md|py|s|ll|ttgir|amdgcn|sass|cu|cpp|hip)\b"
                     r"|:\d+\b|\b\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|%|x|GB/s|TB/s|TFLOP|TFLOPS|cyc)\b)",
                     re.I)

# Per-shape number tables, and the keys a row uses to say which shape it is and how long it took.
# Only these container keys are read: scanning every list in every JSON would pair up unrelated
# numbers and report contradictions that are not.
POINT_CONTAINERS = ("served_range", "per_case", "per_shape", "points", "cases")
LABEL_KEYS = ("shape", "name", "case", "label", "shape_id", "config")
MS_KEYS = ("ms", "latency_ms", "time_ms", "current_ms", "best_ms", "mean_ms", "median_ms",
           "champion_ms", "baseline_ms", "default_ms", "anchor_ms", "plain_ms", "torch_ms")

# A verdict that RETIRES a lever. These are the ones that go stale: a lever measured neutral against
# one kernel body is not measured against the body that body became.
NEGATIVE_VERDICTS = ("revert", "negative", "neutral", "not_supported", "no_win", "rejected")

# How an arm says which config its number was measured at, and what each mode owes. The mode alone
# is not the finding -- an arm that reuses the parent's pin is often right to. What matters is
# whether the mode it declares is consistent with the verdict it reached. See phases/tune.md §4.
CONFIG_BASIS_MODES = ("own_sweep", "closed_axis", "partial", "inherited")
# A mode that scores the arm at a config tuned for a DIFFERENT structure. This biases the arm slow,
# so it can only invalidate a kill, never a win -- the asymmetry the checks below turn on.
BORROWED_MODES = ("inherited", "partial")
KILL_EVIDENCE_KEYS = ("profile_ref", "isa_ref", "pmc_delta", "scaling_evidence")

# Keys an arm uses to record a batch of configs as prose inside its roll-up instead of as a sweep
# envelope. Not wrong in itself -- but a search recorded only here is a search that cannot be
# counted, and the reader who is not allowed to take the arm's word for it has nothing to open.
FREEFORM_SEARCH_KEYS = ("rejected", "ablation", "levers_tried_and_dropped", "landings_measured",
                        "rejected_with_numbers", "config_plateau", "tried")

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", "checkpoint"}


# ------------------------------------------------------------------ small readers

def _load_json(path):
    try:
        with open(path) as f:
            value = json.load(f)
        if os.path.basename(path) == "final_report.json" and isinstance(value, dict):
            return project_final_report(
                value, work_root=os.path.dirname(path), source_path=os.path.abspath(path))
        return value
    except (OSError, json.JSONDecodeError):
        return None


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _rel(root, path):
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _walk(root, pattern):
    """Glob under root, skipping dirs that hold copies rather than records."""
    out = []
    for p in glob.glob(os.path.join(root, pattern), recursive=True):
        parts = set(os.path.relpath(p, root).split(os.sep))
        if parts & SKIP_DIRS or os.path.basename(p) == SELF_OUTPUT:
            continue
        out.append(p)
    return sorted(out)


_PACK_STEMS: set = set()


def _pack_own_stems():
    """`<parent>/<name>` for every file the pack containing this script ships.

    A run hands its worker the guidance on disk, so the work tree contains a COPY of this pack's own
    prose. Reading that back would let the audit answer its questions out of the documentation
    instead of out of the run: the guidance names every tool, so tool-reach passes without a tool
    having run, and the guidance discusses ceilings at length, so the claim scan fires on hundreds of
    lines nobody in this run wrote. Matching on `<parent>/<name>` rather than on content is
    deliberate -- the copy is usually of an older revision, so byte equality would stop recognising
    it exactly as the pack drifts."""
    if _PACK_STEMS:
        return _PACK_STEMS
    stems = _PACK_STEMS
    pack_root = os.path.dirname(HERE)
    for dirpath, dirnames, filenames in os.walk(pack_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            stems.add(os.path.join(os.path.basename(dirpath), fn))
    return stems


# Names a run WRITES. Never treated as copied pack material however the stems fall, so a pack that
# someday ships a file under one of these names cannot blind the audit to the run's own records.
RUN_ARTIFACTS = frozenset({
    "final_report.json", "worker_result.json", "arm_result.json", "decision_log.md",
    "run_manifest.json", "skeptic_review.md", "resolve.json",
    "context.json", "metrics.json", "decision.json", "verdict.json", "record.json",
    "sweep_request.json", "sweep_result.json"})

# This tool's OWN output, skipped on every read. The fleet re-runs the audit and diffs its findings
# against the ones the report quotes, and that diff is only meaningful if the audit is a function of
# the run alone: an audit that reads its last answer back in gives the second caller a different
# count than the first and turns every re-check into a spurious mismatch.
SELF_OUTPUT = "close_audit.json"

# A directory holding one of these is a copied pack's root, not run output. `SKILL.md` is the
# upstream entry name; GEAK's pack dir `gluon_authoring/` has no SKILL.md -- its entry is `skill.md`
# (the upstream entry folded into GEAK's expert-skill entry), so a copy of the GEAK pack must be
# recognised by that name. Case matters on Linux: the two names are two files.
PACK_ENTRY_NAMES = frozenset({"SKILL.md", "skill.md"})


def _text_records(root, prose_only=False):
    """Every record THIS RUN wrote, as (path, text), with the pack's own copied guidance excluded.

    `prose_only` narrows to `.md`: a claim lives in prose, and a JSON field holding an enum has
    nowhere to put a citation, so asking it for one only manufactures findings. The wide form keeps
    JSON and logs, because a command log is honest evidence that a tool ran."""
    own = _pack_own_stems()
    suffixes = (".md",) if prose_only else (".md", ".json", ".txt", ".log")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIRS]
        if dirpath != root and PACK_ENTRY_NAMES.intersection(filenames):
            dirnames[:] = []        # a whole pack was copied in here; none of it is this run's
            continue
        for fn in sorted(filenames):
            if not fn.endswith(suffixes) or fn == SELF_OUTPUT:
                continue
            if fn not in RUN_ARTIFACTS and os.path.join(os.path.basename(dirpath), fn) in own:
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) > 4 << 20:
                    continue        # a dump, not a record
                with open(p, errors="ignore") as f:
                    out.append((p, f.read()))
            except OSError:
                continue
    return out


def load_pack_facts(explicit=None):
    """The per-pack vocabulary, stamped by compose from the manifest.

    Absent is not fatal and not defaulted: the audit runs with every pack-specific check reported
    as undeterminable, which is the honest answer when the file that says what this pack is called
    is missing. Guessing a regime here would produce findings about artifacts that never existed."""
    path = explicit or os.path.join(HERE, "pack_facts.json")
    return _load_json(path) or {}


# ------------------------------------------------------------------ the checks

def _finding(out, check, kind, detail, refs=(), severity=None):
    """Append one fact. `severity` is optional and carried only when a check sets it.

    It is not a verdict -- the tool still takes no position on whether a run should be accepted.
    It answers a narrower question that the reader cannot get from the count alone: whether this
    fact contradicts a conclusion the run has already banked, or merely records that a contract is
    not yet being followed. A check that fires on every pack in the fleet needs that split, or the
    one finding that overturns a result arrives indistinguishable from a hundred that do not."""
    f = {"check": check, "kind": kind, "detail": detail, "refs": sorted(refs)}
    if severity is not None:
        f["severity"] = severity
    out.append(f)


# An objective is a SENTENCE in practice, not a token: `geomean(c2,c32,c64)` and "unweighted
# geomean of execution_time_ms over c2, c32, c64 (the parent's objective)" are the same ranking said
# two ways. String equality called them different questions and raised three HIGH findings on one
# kernel -- each one saying an arm had ranked a different question than the trunk, on arms that had
# copied the trunk's objective and expanded it into prose. A high finding that is wrong is worse than
# no finding: the captain spent the close disposing of it, and the channel reserved for overturned
# conclusions carried three non-conclusions.
#
# So compare the ranking's two load-bearing parts and nothing else: the AGGREGATION and the CASE SET.
# Anything a reader would accept as a restatement -- units, "unweighted", "the parent's objective",
# the metric's own field name -- is dropped. When neither side yields a case set the comparison falls
# back to the normalised text, which is strictly better than the raw string and no worse than before.
_OBJ_NOISE = re.compile(
    r"\b(the|a|an|of|over|on|for|in|at|and|its|own|parent'?s?|objective|unweighted|weighted|equal|"
    r"equally|mean|average|avg|metric|latency|execution|exec|time|_?ms\b|milliseconds?|us|seconds?|"
    r"across|three|served|cases?|case|shapes?|shape|value|values|per|kernel|graded|harness)\b",
    re.I)
_OBJ_AGGS = ("geomean", "geometric", "harmonic", "arithmetic", "sum", "total", "max", "min",
             "median", "p50", "p95", "p99")


def _objective_signature(text):
    """`(aggregation, frozenset(case_ids))` -- what a ranking actually IS, stripped of phrasing."""
    s = str(text or "").lower()
    agg = next((a for a in _OBJ_AGGS if a in s), None)
    if agg in ("geometric",):
        agg = "geomean"
    # Case ids as reports write them: c2 / c32 / c64, shape1, s1, m=2048. Bare integers are NOT
    # case ids -- "over 3 shapes" must not become a case set.
    cases = frozenset(re.findall(r"\b(?:c|s|shape|case)[_-]?(\d+)\b", s))
    return agg, cases


def _same_objective(a, b):
    """Do these two strings name the SAME ranking? See `_OBJ_NOISE` for why this is not `==`."""
    if str(a).strip() == str(b).strip():
        return True
    sa, sb = _objective_signature(a), _objective_signature(b)
    if sa[1] and sb[1]:
        # A case set on both sides is the strong form: same aggregation over the same cases is the
        # same question however it is phrased. A missing aggregation on one side is not a difference
        # (prose often leaves it implicit), but two DIFFERENT ones are.
        if sa[1] != sb[1]:
            return False
        return not (sa[0] and sb[0] and sa[0] != sb[0])
    # No case set to compare: fall back to the de-phrased text.
    na = " ".join(sorted(_OBJ_NOISE.sub(" ", str(a).lower()).split()))
    nb = " ".join(sorted(_OBJ_NOISE.sub(" ", str(b).lower()).split()))
    na = re.sub(r"[^a-z0-9=]+", " ", na).strip()
    nb = re.sub(r"[^a-z0-9=]+", " ", nb).strip()
    return na == nb


def _blocked(facts, out, check, artifact, why):
    """Guardrail R9: a check that cannot run is BLOCKED, which is a failure state, never a pass.

    Only call this when the pack itself declares the artifact -- `pack_facts.artifacts.produces`
    is the test. A pack that never promised the input is `applicable: False` and that is the
    correct, quiet answer; a pack that promised it and did not produce it is the case that used to
    disable its own check and read as clean. Returns the summary dict so a check can `return
    _blocked(...)` in the same place it used to return `{"applicable": False}`."""
    produces = (facts.get("artifacts") or {}).get("produces") or {}
    if artifact not in produces:
        return {"applicable": False, "reason": why}
    _finding(out, check, "check_blocked",
             f"this pack declares {artifact!r} (produced by {produces[artifact]}) and no such file "
             f"is under this root, so the check that reads it could not run. A check that cannot "
             f"run is BLOCKED, not passed -- {why}",
             severity="high")
    return {"applicable": True, "blocked": True, "reason": why, "declared_producer": produces[artifact]}


def _declares(facts, artifact):
    artifacts = facts.get("artifacts") or {}
    return artifact in (artifacts.get("produces") or {}) or artifact in (artifacts.get("consumes") or {})


def _stringify(value):
    """Flatten a field that ships as either a sentence or an object into searchable text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return " ".join(_stringify(v) for v in value.values() if v is not None).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value if v is not None).strip()
    if value is None or isinstance(value, bool):
        return ""
    return str(value)


# The words a run uses when it puts a block of measured cost beyond what it is allowed to change.
# Vendor- and language-neutral on purpose: this file ships byte-identical in every pack.
_UNREACHABILITY = (
    "unreachable", "not reachable", "out of reach", "outside the editable",
    "outside my editable", "outside our editable", "beyond the editable",
    "not in the editable", "cannot be reached", "can not be reached", "not addressable",
    "not body-addressable", "caller-side only", "harness-owned", "frozen by the caller",
    "outside the editable surface",
)


def _claims_unreachable(text):
    low = (text or "").lower()
    return any(token in low for token in _UNREACHABILITY)


def _measured_bound(decl):
    """True when an unreachability price names a quantity, an artifact that RAN, and a method.

    Deliberately shape-tolerant about WHERE the fields sit (a bound may be nested under its own key
    or spelled flat beside the statement) and strict about WHAT they are, because the failure being
    caught is a missing measurement, not a missing convention.
    """
    if not isinstance(decl, dict):
        return False
    scopes = [decl] + [v for v in decl.values() if isinstance(v, dict)]
    for scope in scopes:
        pct = next((scope.get(k) for k in ("pct", "remaining_prize_pct", "bound_pct",
                                           "bounded_remaining_prize_pct")
                    if isinstance(scope.get(k), (int, float))
                    and not isinstance(scope.get(k), bool)), None)
        ref = scope.get("evidence_ref")
        kind = ref.get("kind") if isinstance(ref, dict) else None
        method = scope.get("method")
        if (pct is not None and kind in ("probe", "measurement")
                and isinstance(method, str) and method.strip()):
            return True
    return False


def _editable_surface(root):
    """The run's DECLARED writable surface, or None when nothing declared one.

    Read rather than assumed. Two judgements in this file turn on it -- whether a hardcoded axis is
    unfinished work or a travelling caveat, and whether an out-of-reach cost is a contract fact --
    and both were previously decided by whichever default the check happened to encode. A pack that
    declares nothing keeps the conservative reading, so this cannot silently harden an old tree.
    """
    for name in ("**/resolve.json", "**/editable_surface.json", "**/dispatch.json"):
        for path in _walk(root, name):
            doc = _load_json(path)
            surface = (doc or {}).get("editable_surface")
            if isinstance(surface, dict) and isinstance(surface.get("paths"), list):
                return {"paths": [str(p) for p in surface["paths"]],
                        "declared_by": surface.get("declared_by"),
                        "ref": _rel(root, path)}
    return None


def _arm_headline(record):
    """The canonical schema carries one ranked measurement; legacy files are only audit evidence."""
    if record.get("schema") == ARM_RESULT_SCHEMA:
        measurement = record.get("measurement")
        return measurement.get("value") if isinstance(measurement, dict) else None
    return next((record[key] for key in ARM_HEADLINE_KEYS if record.get(key) is not None), None)


def check_arm_roster(root, facts, out):
    """1. Every dispatched arm's record exists, says it finished, and carries its headline number
    -- and none of them was written after the parent collected.

    The timestamp half is the part no role can do without walking the tree. An arm still writing
    when the parent ranked is absent from that ranking, and absent is unfinished, not zero: scored
    as a no-win it drops out of the comparison while still being counted in it."""
    arms = _walk(root, ARM_GLOB)
    collected = [(_mtime(p), p) for name in COLLECTION_MARKERS
                 for p in _walk(root, f"**/{name}") if _mtime(p)]
    collect_at = min(collected)[0] if collected else None
    marker = _rel(root, min(collected)[1]) if collected else None
    for path in arms:
        ref, rec = _rel(root, path), _load_json(path)
        if rec is None:
            _finding(out, "arm_roster", "unreadable", "arm record is missing or not valid JSON",
                     [ref])
            continue
        if rec.get("schema") == ARM_RESULT_SCHEMA:
            for finding in validate_arm_result(rec):
                _finding(out, "arm_roster", "canonical_arm_invalid", finding["detail"], [ref],
                         severity="high")
        elif _declares(facts, "arm_result.json"):
            _finding(out, "arm_roster", "legacy_arm_requires_adapter",
                     "this pack declares canonical arm_result.json, but this record has no canonical "
                     "ArmResult schema. It is evidence only until adapted by canonical_record.py "
                     "write-arm-result; convergence may not rank mixed formats.",
                     [ref], severity="high")
        comp = rec.get("completion")
        headline = _arm_headline(rec)
        if comp is None:
            _finding(out, "arm_roster", "completion_absent",
                     "arm does not say whether it finished; a crash, a budget cut-off and a "
                     "measured null are indistinguishable in this record", [ref])
        elif comp != "complete":
            _finding(out, "arm_roster", f"completion_{comp}",
                     f"arm reports completion={comp!r}: its numbers are partial by its own account",
                     [ref])
        if headline is None and not rec.get("not_applicable"):
            _finding(out, "arm_roster", "headline_absent",
                     f"no headline number under any of {list(ARM_HEADLINE_KEYS)}; the parent has "
                     f"nothing to rank this arm by", [ref])
        if comp == "complete" and headline is None and not rec.get("not_applicable"):
            _finding(out, "arm_roster", "complete_without_number",
                     "arm says complete and carries no headline number -- one of the two is wrong",
                     [ref])
        mt = _mtime(path)
        if collect_at is not None and mt is not None and mt > collect_at:
            _finding(out, "arm_roster", "written_after_collection",
                     f"arm record is {mt - collect_at:.0f}s NEWER than {marker}, the earliest "
                     f"roll-up in this tree, so whatever it says now was not what the parent "
                     f"ranked", [ref, marker])
    # The marker is named and dated in the summary because this is the one check whose answer moves
    # when a roll-up is later rewritten: editing the report raises its mtime, the baseline slides
    # forward, and arms that were late stop looking late. A fleet re-run that disagrees with the
    # report's count should compare these two fields before it calls the difference a mismatch.
    return {"arms_found": len(arms), "collection_marker": marker, "collected_at": collect_at,
            "baseline_note": "earliest mtime among " + ", ".join(COLLECTION_MARKERS)}


def check_arm_count(root, facts, out):
    """2. As many arms came back as went out.

    The only detection anywhere in the system for a fan-out that silently launched fewer arms than
    it asked for. Skipped where the pack declares it does not fan out, since there the expected
    count is zero by construction and a roster would be the surprising thing."""
    if facts.get("breadth_enabled") is False:
        return {"expected": 0, "found": 0, "basis": "pack declares breadth_enabled: false"}
    found = len(_walk(root, ARM_GLOB))
    expected, basis = None, None
    # `branch_plan.json` is the artifact the pack's own tool writes to DECLARE the roster, and it
    # was missing from this list -- so a run that did exactly what the contract asks, and validated
    # a plan through branch_plan.py, still audited as having declared nothing. Measured over 19
    # kernel/phase runs: 12 carried a branch_plan.json and 15 were reported `roster_undeclared`.
    # `candidate_roster` is its roster; `fanout` is the count that roster must equal.
    for name, keys in (("branch_plan.json", ("candidate_roster", "arms")),
                       ("branch_request.json", ("arms", "candidates", "prescan_candidates")),
                       ("t0_merge.json", ("arms", "roster")),
                       ("census.json", ("tactics",)),
                       ("decision.json", ("prescan_candidates",))):
        for p in _walk(root, f"**/{name}"):
            doc = _load_json(p) or {}
            for k in keys:
                v = doc.get(k)
                if isinstance(v, list) and v:
                    expected, basis = len(v), f"{_rel(root, p)}:{k}"
                    break
            if expected is not None:
                break
        if expected is not None:
            break
    if expected is None:
        # High only once arms exist. With no arms there is no fan-out to have lost one from, and
        # the undeclared roster is a shape the run never entered. With arms on disk it is the
        # difference between a complete search and an unknown one: breadth completeness is the
        # thing BRANCH exists to guarantee, and without a declared roster no reader -- and no
        # later check -- can tell a roster that covered the census from one that dropped half of it
        # at spawn. This is the finding that makes "was the search complete?" unanswerable.
        _finding(out, "arm_count", "roster_undeclared",
                 f"{found} arm record(s) on disk and nothing declaring how many were dispatched; a "
                 f"fan-out that lost an arm at spawn is indistinguishable from one that asked for "
                 f"this many. Emit the plan through scripts/branch_plan.py so the roster it covers "
                 f"is on disk before the arms run",
                 severity="high" if found else None)
    elif expected != found:
        # A declared roster that does not match what came back is the loss this check was written
        # for, and it is strictly more actionable than the undeclared case.
        _finding(out, "arm_count", "count_mismatch",
                 f"{basis} declares {expected} arm(s); {found} record(s) on disk",
                 severity="high")
    return {"expected": expected, "found": found, "basis": basis}


def _arm_territory(root, record_path):
    """The outermost `arm*` directory above an arm record -- everything that arm owns.

    Not the record's own directory: the packs nest the roll-up (`<arm>/exp/branch/plain/<arm>/`)
    while the arm's sweeps land further up, at `<arm>/exp/`. Taking the innermost dir would call an
    arm's own sweep foreign. Taking the outermost is what the containment question actually means --
    is this file the arm's, or is it the trunk's?"""
    rel = os.path.relpath(os.path.dirname(record_path), root)
    parts = [] if rel == "." else rel.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("arm"):
            return os.path.join(root, *parts[:i + 1])
    return os.path.dirname(record_path)


def _resolve_ref(root, base, ref):
    """A recorded path, tried as the record meant it: beside the record, then from the work root."""
    if not isinstance(ref, str) or not ref.strip():
        return None
    for cand in (os.path.join(base, ref), os.path.join(root, ref), ref):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def _inside(parent, path):
    parent = os.path.abspath(parent)
    return os.path.abspath(path).startswith(parent + os.sep) or os.path.abspath(path) == parent


def _ref_path(value):
    """Read either portable string refs or the typed refs emitted by canonical_record.py."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("path", "artifact"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return None


def _canonical_findings(out, check, kind, document, validator, ref):
    for finding in validator(document):
        _finding(out, check, kind, finding["detail"], [ref], severity="high")


def _role_policy_findings(out, document, validator, ref, *, policy_kind):
    """Surface executable role-policy violations under their declared audit kinds."""
    findings = validator(document)
    for finding in findings:
        _finding(out, "role_policy",
                 policy_kind if finding["code"] in {"forbidden_deep_branch", "unauthorized_resweep"}
                 else f"{policy_kind}_invalid",
                 finding["detail"], [ref], severity="high")
    return findings


def check_role_policy(root, facts, out):
    """Consume the two role-boundary signals declared in audit_policy.

    Cursor cannot technically stop a prompt from trying a forbidden action.  The durable,
    detectable boundary is therefore the artifact it would have to write: a deep worker cannot
    submit a branch request or execute a resweep, and a resweep request must point back to its
    actual deep-worker result.  This audit deliberately does not infer roles from directory names.
    """
    result = {"branch_requests": 0, "worker_results": 0, "resweep_requests": 0,
              "resweep_results": 0, "violations": 0}
    for path in _walk(root, "**/branch_request.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        result["branch_requests"] += 1
        ref = _rel(root, path)
        if doc.get("schema") == BRANCH_REQUEST_SCHEMA:
            _canonical_findings(out, "role_policy", "branch_request_invalid", doc,
                                validate_branch_request, ref)
        if doc.get("requester_role") == "deep":
            result["violations"] += 1
            _finding(out, "role_policy", "forbidden_deep_branch",
                     "a deep worker emitted branch_request.json. Deep workers may request a later "
                     "structure review, but only a direction worker may request captain-owned BRANCH.",
                     [ref], severity="high")

    for path in _walk(root, "**/worker_result.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict) or doc.get("schema") != WORKER_RESULT_SCHEMA:
            continue
        result["worker_results"] += 1
        ref = _rel(root, path)
        findings = _role_policy_findings(out, doc, validate_worker_result, ref,
                                         policy_kind="forbidden_deep_branch")
        if any(finding["code"] == "forbidden_deep_branch" for finding in findings):
            result["violations"] += 1

    for path in _walk(root, "**/resweep_request.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        result["resweep_requests"] += 1
        ref = _rel(root, path)
        if doc.get("schema") != RESWEEP_REQUEST_SCHEMA:
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "resweep_request.json does not use the deep-request schema and cannot establish "
                     "that execution remained with a captain/direct owner.", [ref], severity="high")
            continue
        findings = _role_policy_findings(out, doc, validate_resweep_request, ref,
                                         policy_kind="unauthorized_resweep")
        if any(finding["code"] == "unauthorized_resweep" for finding in findings):
            result["violations"] += 1
        parent_ref = _ref_path(doc.get("parent_result_ref"))
        parent_path = _resolve_ref(root, os.path.dirname(path), parent_ref)
        parent = _load_json(parent_path) if parent_path else None
        requester = doc.get("requester_role")
        if not isinstance(parent, dict) or parent.get("schema") not in (
                WORKER_RESULT_SCHEMA, BRANCH_CONVERGE_SCHEMA):
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "a resweep request has no resolvable canonical worker-result or branch-converge parent.",
                     [ref], severity="high")
            continue
        if requester == "deep" and parent.get("producer_role") != "deep":
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "a deep resweep request must name its own deep worker-result as parent.",
                     [ref, _rel(root, parent_path)], severity="high")
        elif parent.get("schema") == WORKER_RESULT_SCHEMA and (
                parent.get("run_id") != doc.get("run_id")
                or parent.get("generation") != doc.get("generation")):
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "resweep request and worker-result parent do not prove the same run and generation.",
                     [ref, _rel(root, parent_path)], severity="high")

    for path in _walk(root, "**/resweep_result.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        result["resweep_results"] += 1
        ref = _rel(root, path)
        if doc.get("schema") != RESWEEP_RESULT_SCHEMA:
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "resweep_result.json has no captain/direct-owner execution schema.", [ref],
                     severity="high")
            continue
        _canonical_findings(out, "role_policy", "resweep_result_invalid", doc,
                            validate_resweep_result, ref)
        if doc.get("executor_role") not in ("captain", "direct_owner"):
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "a deep worker may request but never execute a resweep.", [ref], severity="high")
            continue
        request_ref = _ref_path(doc.get("request_ref"))
        request_path = _resolve_ref(root, os.path.dirname(path), request_ref)
        request = _load_json(request_path) if request_path else None
        if not isinstance(request, dict) or request.get("schema") != RESWEEP_REQUEST_SCHEMA:
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "a resweep result must consume a resolvable canonical resweep request.",
                     [ref], severity="high")
        elif (request.get("run_id") != doc.get("run_id") or
              request.get("generation") != doc.get("generation")):
            result["violations"] += 1
            _finding(out, "role_policy", "unauthorized_resweep",
                     "resweep result and request do not prove the same run and generation.",
                     [ref, _rel(root, request_path)], severity="high")
    return result


def _pin_artifacts(root, facts):
    """The pin records this pack produces (champion / best-config), newest first."""
    produced = ((facts.get("artifacts") or {}).get("produces") or {})
    names = {n for n in produced if any(fnmatch.fnmatch(n, pat) for pat in
                                        ("*champion*.json", "*best_config*.json", "*best*.json"))}
    names |= {"plain_best_config.json"}     # the family-wide spelling, produced or not
    hits = []
    for name in sorted(names):
        hits.extend(_walk(root, f"**/{name}"))
    return sorted(set(hits))


def check_arm_config_basis(root, facts, out, depth_ratio=4.0):
    """3. Which config each arm's number was measured at, and whether that supports its verdict.

    An arm that re-swept on its own structure and an arm that reused the parent's pin file the same
    record: one geomean, one verdict. The difference only becomes visible if the arm says which it
    did, so `config_basis.mode` is a declared field and its absence is itself reportable.

    The rule this check exists for is asymmetric, and the asymmetry is not a nicety. Scoring a new
    structure at a config tuned for the old one biases it SLOW. So a borrowed config cannot
    manufacture a win -- an arm that wins anyway is a conservative lower bound -- but it can very
    easily manufacture a loss, and a loss is what retires a direction permanently. Same for the
    mechanism: an arm that kills itself on a slower number alone may have measured its config.

    Severity splits on exactly that: `high` marks a fact that contradicts a conclusion already
    banked (a kill that may not be one, a ranking across incomparable objectives); `low` marks a
    contract not yet followed, which is worth fixing and has overturned nothing.

    An arm that declares NO basis gets the low finding and nothing else. Every `high` here asks
    whether a DECLARED basis supports the verdict, and a record written before the contract existed
    cannot answer that question -- marking all of them high would fill the channel reserved for
    overturned conclusions with the entire back catalogue, which is how a severity split stops
    being read at all."""
    arms = _walk(root, ARM_GLOB)
    trunk_objective, obj_ref = None, None
    for p in _pin_artifacts(root, facts):
        doc = _load_json(p) or {}
        if isinstance(doc, dict) and doc.get("objective"):
            trunk_objective, obj_ref = str(doc["objective"]), _rel(root, p)
        # A comparator the sweep itself refused to call tuned discounts every arm hanging off it.
        if isinstance(doc, dict) and (doc.get("trust_level") == "provisional"
                                      or doc.get("partially_sampled") is True):
            _finding(out, "arm_config_basis", "comparator_provisional",
                     f"the comparator every arm is quoted against is {doc.get('trust_level')!r} "
                     f"(partially_sampled={doc.get('partially_sampled')!r}); each arm's speedup "
                     f"carries that discount whether or not the arm mentions it",
                     [_rel(root, p)], severity="low")

    groups: dict = {}
    seen = {"own_sweep": 0, "closed_axis": 0, "partial": 0, "inherited": 0, "absent": 0}
    for path in arms:
        ref, rec = _rel(root, path), _load_json(path)
        if not isinstance(rec, dict):
            continue                        # check_arm_roster already reported the unreadable ones
        if str(rec.get("direction_class") or "").strip().lower() != "structural":
            continue                        # a knob arm owes no structure-specific sweep
        territory = _arm_territory(root, path)
        verdict = str(rec.get("verdict") or rec.get("result") or "").strip().lower()
        kills = verdict in NEGATIVE_VERDICTS and not rec.get("not_applicable")
        cb = rec.get("config_basis")
        cb = cb if isinstance(cb, dict) else {}
        mode = str(cb.get("mode") or "").strip().lower()
        parent = os.path.dirname(os.path.dirname(territory.rstrip(os.sep))) or root
        groups.setdefault(_rel(root, parent), []).append(
            {"arm": _rel(root, territory), "mode": mode or None,
             "n_measured": cb.get("n_measured"), "verdict": verdict or None,
             "headline": _arm_headline(rec)})

        # -- a search recorded only as prose cannot be counted -----------------------------
        # Asked of every arm, declared basis or not: the carrier is the question, and a batch of
        # configs living in a free-form key is uncountable whichever contract the arm followed.
        if not str(cb.get("sweep_ref") or "").strip():
            prose = [k for k in FREEFORM_SEARCH_KEYS if rec.get(k)]
            if prose:
                _finding(out, "arm_config_basis", "search_record_not_in_envelope",
                         f"the arm's search appears only as free-form keys ({', '.join(prose)}) "
                         f"with no sweep_ref, so how many configs it covered is not countable",
                         [ref], severity="low")

        if not cb:
            seen["absent"] += 1
            _finding(out, "arm_config_basis", "config_basis_absent",
                     "structural arm does not say which config its number was measured at, so an "
                     "arm that re-swept and an arm that reused the parent's pin are the same "
                     "record here", [ref], severity="low")
            # Everything below asks whether a DECLARED basis supports the verdict, and a record
            # written before the contract existed cannot answer it. Reporting those as `high` would
            # put every legacy arm in the channel reserved for facts that overturn a conclusion,
            # which is how a severity split stops being read. The kill still gets its own line --
            # the retirement is unverifiable, and that is worth saying once, plainly.
            if kills:
                _finding(out, "arm_config_basis", "kill_basis_undeclared",
                         f"verdict {verdict!r} retires this direction and the record declares "
                         f"neither the config it was measured at nor a kill_evidence mechanism, so "
                         f"whether the structure or its borrowed config lost is not recorded",
                         [ref], severity="low")
            continue
        if mode not in CONFIG_BASIS_MODES:
            _finding(out, "arm_config_basis", "mode_unrecognised",
                     f"config_basis.mode is {cb.get('mode')!r}, not one of "
                     f"{list(CONFIG_BASIS_MODES)}; nothing below can be checked against it",
                     [ref], severity="low")
        else:
            seen[mode] += 1

        # -- what each declared mode owes -------------------------------------------------
        if mode == "own_sweep":
            sref = cb.get("sweep_ref")
            resolved = _resolve_ref(root, os.path.dirname(path), sref)
            if not str(sref or "").strip():
                _finding(out, "arm_config_basis", "sweep_ref_absent",
                         "mode is own_sweep and no sweep_ref names the envelope, so the sweep is "
                         "asserted rather than checkable", [ref], severity="high")
            elif resolved is None:
                _finding(out, "arm_config_basis", "sweep_ref_missing",
                         f"sweep_ref {sref!r} is not on disk under this root", [ref],
                         severity="high")
            elif not _inside(territory, resolved):
                _finding(out, "arm_config_basis", "sweep_ref_outside_arm",
                         f"sweep_ref {sref!r} resolves outside {_rel(root, territory)!r}; pointing "
                         f"at another record is not this arm having swept", [ref], severity="high")
            n_meas = cb.get("n_measured")
            if not (isinstance(n_meas, int) and not isinstance(n_meas, bool) and n_meas > 0):
                _finding(out, "arm_config_basis", "n_measured_absent",
                         f"mode is own_sweep and n_measured is {n_meas!r}; a sweep with no count "
                         f"cannot be compared with a sibling's", [ref], severity="low")
        elif mode == "closed_axis":
            if not str(cb.get("closure_ref") or "").strip():
                _finding(out, "arm_config_basis", "closure_ref_absent",
                         "mode is closed_axis and no closure_ref names the sweep that closed it; "
                         "declining to re-sweep is a citation, not an assertion", [ref],
                         severity="high")
            rpm = cb.get("resource_profile_match")
            rpm = rpm if isinstance(rpm, dict) else {}
            moved = []
            for field in ("lds_bytes_per_wg", "vgpr", "spill"):
                pair = rpm.get(field)
                if (isinstance(pair, (list, tuple)) and len(pair) == 2
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                for x in pair) and pair[0] != pair[1]):
                    moved.append(f"{field} {pair[0]}->{pair[1]}")
            if moved and not str(rpm.get("recheck_ref") or "").strip():
                _finding(out, "arm_config_basis", "closure_premise_moved",
                         f"the cited closure was measured at a different resource profile "
                         f"({'; '.join(moved)}) and no recheck_ref re-confirms it; a closed axis "
                         f"re-opened by this arm's own edit is worth one measurement", [ref],
                         severity="high")
        elif mode == "partial" and not (cb.get("frozen_axes") or []):
            _finding(out, "arm_config_basis", "frozen_axes_unnamed",
                     "mode is partial and frozen_axes is empty, so which trunk axes went "
                     "un-reswept on the new structure is not recorded", [ref], severity="low")
        elif mode == "inherited" and not (str(cb.get("inherit_rationale") or "").strip()
                                          or cb.get("scored_path_byte_equivalent")):
            _finding(out, "arm_config_basis", "inherit_unreasoned",
                     "mode is inherited with neither an inherit_rationale nor "
                     "scored_path_byte_equivalent", [ref], severity="low")

        # -- a kill owes a mechanism, and owes its own config ------------------------------
        if kills:
            kev = rec.get("kill_evidence")
            kev = kev if isinstance(kev, dict) else {}
            if not any(str(kev.get(k) or "").strip() for k in KILL_EVIDENCE_KEYS):
                _finding(out, "arm_config_basis", "kill_without_mechanism",
                         f"verdict {verdict!r} retires this direction and kill_evidence names no "
                         f"mechanism ({list(KILL_EVIDENCE_KEYS)}); a slower number alone may be "
                         f"measuring the config rather than the structure", [ref], severity="high")
            if mode in BORROWED_MODES and not cb.get("scored_path_byte_equivalent"):
                _finding(out, "arm_config_basis", "kill_at_borrowed_config",
                         f"verdict {verdict!r} was reached at config_basis.mode={mode!r}, a config "
                         f"tuned for a different structure, which biases this arm slow -- so the "
                         f"loss is not yet shown. A win under the same mode would need no remedy",
                         [ref], severity="high")

        # -- the arm and the trunk must be answering the same question ---------------------
        arm_obj = str(cb.get("objective") or "").strip()
        if arm_obj and trunk_objective and not _same_objective(arm_obj, trunk_objective):
            _finding(out, "arm_config_basis", "objective_mismatch",
                     f"this arm ranked its configs on {arm_obj!r} while the comparator is "
                     f"{trunk_objective!r}; that is not a weaker ranking, it is a ranking of a "
                     f"different question", [ref, obj_ref], severity="high")

    # -- depth dispersion: reported with the margin, never against a threshold of its own -----
    # Depth is a bad proxy for rigour in both directions -- the widest grid in a run has killed an
    # arm correctly, and eleven points plus a scaling ladder has too. What makes a dispersion worth
    # a second look is the MARGIN it sits next to, and that comparison is the reader's.
    for parent in sorted(groups):
        rows = groups[parent]
        depths = [r["n_measured"] for r in rows
                  if isinstance(r["n_measured"], int) and not isinstance(r["n_measured"], bool)
                  and r["n_measured"] > 0]
        if len(depths) < 2 or min(depths) <= 0 or max(depths) / min(depths) < depth_ratio:
            continue
        heads = [r["headline"] for r in rows if isinstance(r["headline"], (int, float))
                 and not isinstance(r["headline"], bool)]
        margin = (f"{(max(heads) - min(heads)) / min(heads) * 100:.1f}% apart"
                  if len(heads) >= 2 and min(heads) > 0 else "headlines not comparable")
        _finding(out, "arm_config_basis", "depth_dispersion",
                 f"arms in {parent!r} were swept at {sorted(depths)} points "
                 f"({max(depths) / min(depths):.1f}x apart, ratio {depth_ratio:g}) and their "
                 f"headlines are {margin}; whether that margin survives the depth difference is a "
                 f"judgment about the mechanism, not about the counts", severity="low")

    # A missing basis on ONE arm stays low above: a pre-contract record cannot answer a question
    # the contract had not asked yet. A roster where a winner was picked and NOT ONE arm declared
    # what config its number came from is the other case, and it is the one the low severity was
    # never meant to cover -- the ranking itself is unsupported. It overturns a conclusion rather
    # than annotating one, and it has been measured doing so: a run that ranked its arms on the
    # parent's pin shipped NG=32, and the post-merge re-sweep on the merged body put NG=8 ahead at
    # every served case, voiding the certificate the arm phase had issued.
    # WHOSE declaration settles it. This used to skip the group as soon as ANY arm carried a mode,
    # which reads the question as "did the roster adopt the contract" when the question is "is the
    # ordering this close rests on auditable". A losing arm declaring `inherited` says nothing about
    # the configuration the WINNER's number came from, and it silenced the finding entirely -- so
    # the cheapest way past a high finding about the ranking was a field on an arm that lost.
    for parent, rows in sorted(groups.items()):
        structural = [r for r in rows if r.get("verdict")]
        if len(structural) < 2:
            continue
        winners = [r for r in structural
                   if r["verdict"] not in NEGATIVE_VERDICTS]
        if not winners:
            continue                        # an all-negative roster selected nothing to defend
        bare = [r for r in winners if not r.get("mode")]
        if not bare:
            continue                        # every carried-forward arm says where its number came from
        # Pre-contract stays pre-contract: a roster where NOBODY declared anything may be a record
        # written before the field existed, and that case is already the low `basis_absent` note per
        # arm. What is not exempt is a roster that HAS the contract -- somebody filled it in -- and
        # still cannot say what the winner was measured at.
        adopted = any(r.get("mode") for r in structural)
        _finding(out, "arm_config_basis", "winner_ranked_without_declared_basis",
                 f"{len(structural)} structural arms in {parent!r} were ranked and "
                 f"{len(winners)} carried forward; {len(bare)} of those carried-forward arm(s) "
                 f"declare no configuration their number was measured at"
                 + (f", while {sum(1 for r in structural if r.get('mode'))} other arm(s) in the "
                    f"same roster do -- so this is the contract being followed everywhere except "
                    f"on the arm the close depends on" if adopted else
                    ", and not one arm in the roster declares one")
                 + ". The winning config differs per structure, so a ranking taken at a shared pin "
                   "can inverse the deep ranking -- which is the ordering this close rests on",
                 [r["arm"] for r in bare[:4]], severity="high")

    return {"arms_structural": sum(seen.values()), "by_mode": seen,
            "trunk_objective": trunk_objective, "depth_ratio": depth_ratio,
            "groups": {k: sorted(v, key=lambda r: r["arm"]) for k, v in sorted(groups.items())}}


def check_post_merge_resweep(root, facts, out):
    """4. A fan-out that carried a winner into the trunk owes the trunk a re-sweep, or a reason.

    The merged body's bottleneck stack is a different stack, so the pin the trunk was holding was
    measured against a kernel that no longer exists. This does not say the pin moved -- in the runs
    where it was checked it often did not, and a re-sweep in which every axis moves and every axis
    loses is exactly the point: it closes the config with numbers instead of by inheritance."""
    checked = 0
    for name in ("t0_merge.json", "branch_merge.json", "merge.json"):
        for p in _walk(root, f"**/{name}"):
            doc = _load_json(p)
            if not isinstance(doc, dict):
                continue
            winner = str(doc.get("winner") or "").strip().lower()
            if not winner or winner in ("anchor", "stay-plain", "stay_plain", "baseline", "none"):
                continue
            checked += 1
            if (str(doc.get("post_merge_resweep_ref") or "").strip()
                    or str(doc.get("resweep_waived_reason") or "").strip()):
                continue
            _finding(out, "post_merge_resweep", "resweep_unrecorded",
                 f"winner {winner!r} was carried into the trunk and the merge names neither a "
                 f"post_merge_resweep_ref nor a resweep_waived_reason; the trunk's pin was "
                 f"measured against the pre-merge body. Every number the run quotes against that "
                 f"pin is quoted against a kernel that no longer exists, and one run's largest "
                 f"single gain came from precisely this re-sweep: a structural change inverted "
                 f"six standing config rejections, three of which were one decision",
                 [_rel(root, p)],
                 # Raised from low. A low finding says a contract is not yet followed; this one says
                 # a banked comparator may not hold, which is the definition of high here. It also
                 # had NO OWNER in the role division until `triton-direction` was assigned it, so
                 # the only thing that ever noticed was this line -- fired at close, after the
                 # budget was spent.
                 severity="high")
    return {"merges_with_a_carried_winner": checked}


def check_sweep_point_accounting(root, facts, out, tolerance=0.10):
    """5. A claimed sweep size against the rows actually on disk.

    The cheapest possible check and it has caught real drift both ways: a report saying 285 configs
    over a table holding 428, and one saying 347 over a table holding 379. Neither is fraud and
    neither was explained -- which is the problem, because the count is what a reader uses to judge
    whether an axis was actually covered. Reported only when both numbers exist; a run with no
    countable envelope is not accused of a mismatch with nothing."""
    claim_keys = ("config_sweep_points_evaluated", "sweep_points_evaluated", "sweep_points",
                  "configs_swept", "n_configs_swept")
    claimed, claim_ref = None, None
    for name in ("final_report.json", "worker_result.json"):
        for p in _walk(root, f"**/{name}"):
            doc = _load_json(p)
            if not isinstance(doc, dict):
                continue
            stack = [doc]
            while stack and claimed is None:
                node = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        if (k in claim_keys and isinstance(v, (int, float))
                                and not isinstance(v, bool) and v > 0):
                            claimed, claim_ref = int(v), _rel(root, p)
                            break
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(node, list):
                    stack.extend(x for x in node if isinstance(x, (dict, list)))
    if claimed is None:
        return {"claimed": None, "on_disk": None, "basis": None}

    on_disk, tables = 0, []
    for p in _walk(root, "**/*.json"):
        base = os.path.basename(p)
        stem = base[:-5] if base.endswith(".json") else base
        # `res` only as a whole stem: blockscale's `sweep/<label>/res.json` is the tightest real
        # spelling, and matching it as a substring would pull in every unrelated file with those
        # three letters in its name.
        if base in RUN_ARTIFACTS or not (stem == "res" or re.search(
                r"sweep|screen|autotune|best_config|tune|cfg", stem, re.I)):
            continue
        doc = _load_json(p)
        rows = None
        if isinstance(doc, list):
            rows = len(doc)
        elif isinstance(doc, dict):
            for k in ("measured", "results", "rows", "points", "configs", "ranked"):
                if isinstance(doc.get(k), list):
                    rows = len(doc[k])
                    break
        if rows:
            on_disk += rows
            tables.append(_rel(root, p))
    if not tables:
        return {"claimed": claimed, "on_disk": None, "basis": claim_ref}
    if abs(on_disk - claimed) > max(1, tolerance * max(on_disk, claimed)):
        _finding(out, "sweep_point_accounting", "count_mismatch",
                 f"{claim_ref} claims {claimed} swept config(s); {len(tables)} envelope(s) under "
                 f"this root hold {on_disk} row(s). One of the two is describing a different set "
                 f"and the record does not say which", [claim_ref], severity="low")
    return {"claimed": claimed, "on_disk": on_disk, "basis": claim_ref, "tables": len(tables)}


def check_evidence_floor(root, facts, out):
    """6. An evidence layer that CRASHED, against whether the report says so.

    There is no tolerance to set here: this is a presence test, not a comparison.

    What it is for. A profiler layer fails in a way that leaves a plausible artifact behind -- a
    file of the right name, in the right place, holding a traceback instead of a report. Every
    machine reader downstream sees a null and cannot tell it from a layer nobody asked for, so the
    run proceeds on the remaining layers and the gap survives into the report as prose, if at all.
    Measured over one nine-kernel campaign: 27 of 30 collections died on the same fixable
    permission error, all nine reports mentioned it somewhere, and three put it in `caveats`.

    The check reads the artifact, not the null. `collected_unparsed` is called out separately
    because it is the expensive case: the numbers WERE collected and are sitting on disk, and only
    the last hop to a machine-readable file is missing.

    Severity follows the precedent set for arm records: a run whose captures declare `sol_state`
    has adopted the contract and is held to it; a pre-contract run gets a low note instead, so
    re-auditing an archive does not bury the one finding that overturns something.
    """
    caps = [p for p in _walk(root, "**/capture.json")]
    arts = _walk(root, "**/rc_analyze*")
    if not caps and not arts:
        # No AMD SOL layer in this pack at all (the NVIDIA spellings ship no rocprof-compute) ->
        # "not applicable" is the answer, and a finding here would be an accusation about a tool
        # that was never part of this contract. But if the pack DECLARES capture.json and none is
        # on disk, the same return value used to disable this check and read as clean -- that is
        # R9, and _blocked() separates the two.
        return _blocked(facts, out, "evidence_floor", "capture.json",
                        "every conclusion in this run was reached with group C dark, and nothing "
                        "downstream can tell that from a run that had no use for it")

    states, on_contract, refs = collections.Counter(), False, {}
    for p in caps:
        doc = _load_json(p)
        if not isinstance(doc, dict):
            continue
        st = doc.get("sol_state")
        if st:
            on_contract = True
        else:
            # Pre-contract capture: derive the same three-way answer from the artifacts beside it.
            st = "parsed" if doc.get("rc_metrics") else None
            if st is None:
                d = os.path.dirname(doc.get("rc_analyze") or "")
                sibs = sorted(glob.glob(os.path.join(d, "rc_analyze*"))) if d else []
                st = "not_attempted"
                for s in sibs:
                    try:
                        with open(s, errors="replace") as f:
                            txt = f.read()
                    except OSError:
                        continue
                    if re.search(r"Speed-of-Light|System Speed|Dependency-Wait", txt, re.I):
                        st = "collected_unparsed"
                        break
                    if "PermissionError" in txt or "Traceback (most recent call last)" in txt:
                        st = "crashed"
        states[st] += 1
        refs.setdefault(st, _rel(root, p))

    broken = states["crashed"] + states["collected_unparsed"]
    if not broken:
        return {"applicable": True, "captures": len(caps), "states": dict(states),
                "declared": None, "on_contract": on_contract}

    # Does the run say so where a reader is allowed to look? `caveats` is the slot; the wider doc
    # is accepted as a weaker yes, because a truthful sentence in the wrong field is a filing
    # problem, not a silence.
    declared, declared_in = False, None
    for name in ("final_report.json", "worker_result.json"):
        for p in _walk(root, f"**/{name}"):
            doc = _load_json(p)
            if not isinstance(doc, dict):
                continue
            cav = json.dumps(doc.get("caveats") or doc.get("gaps") or [], ensure_ascii=False)
            if re.search(r"rocprof-compute|rc_analyze|rc_metrics|\bSOL\b", cav, re.I):
                declared, declared_in = True, _rel(root, p) + ":caveats"
                break
        if declared:
            break

    sev = "high" if on_contract else "low"
    if states["crashed"] and not declared:
        _finding(out, "evidence_floor", "sol_crashed_undeclared",
                 f"{states['crashed']} capture(s) record a CRASHED SOL collection (e.g. "
                 f"{refs.get('crashed')}) and no caveat in this run's roll-up mentions it. Every "
                 f"conclusion here was reached with group C dark, and a reader of the report "
                 f"cannot tell that from a run that had no use for it",
                 [refs.get("crashed")] if refs.get("crashed") else (), severity=sev)
    if states["collected_unparsed"]:
        _finding(out, "evidence_floor", "sol_collected_but_unparsed",
                 f"{states['collected_unparsed']} capture(s) have the SOL report on disk and no "
                 f"rc_metrics.json built from it (e.g. {refs.get('collected_unparsed')}): the "
                 f"expensive part succeeded and only the parse is missing, so every machine "
                 f"reader still sees null. Re-run parse_rc.py against that file",
                 [refs.get("collected_unparsed")] if refs.get("collected_unparsed") else (),
                 severity="low")
    return {"applicable": True, "captures": len(caps), "states": dict(states),
            "declared": declared_in, "on_contract": on_contract}


def check_preflight_closure(root, facts, out):
    """7. Entry-gate caveats that named an owner, against whether that owner ever answered.

    The gate already does the hard half. It separates "checked and fine" from "nobody checked",
    and for the second it names who owns the answer -- `owner="worker round-1 capture.sh"`. What
    was missing is the other end: nothing read the handoff back to ask whether the named owner
    closed it. So a deferred check stayed deferred for the length of a run, the roll-up said
    `closed`, and the two facts never met.

    Closure is read off artifacts, not off a promise. `[EVIDENCE]` is closed by a capture that
    states a definite `sol_state` -- the owner looked and said what they found, whichever way it
    went. Anything else is closed by the roll-up naming the check in its caveats, which is the
    weaker but honest form: carried, not silently dropped.
    """
    pfs = _walk(root, "**/preflight.json")
    if not pfs:
        return {"applicable": False, "reason": "no preflight.json under this root"}
    typed_closures = {}
    for path in _walk(root, "**/debt_closure.json"):
        doc = _load_json(path)
        if isinstance(doc, dict) and doc.get("schema") == "kernel_opt.debt_closure/1":
            typed_closures[doc.get("debt_id")] = (doc, _rel(root, path))

    # What the run's captures ended up saying about the evidence layer, if anything.
    sol_answered = any(isinstance(c, dict) and c.get("sol_state")
                       for c in (_load_json(p) for p in _walk(root, "**/capture.json")))
    declared = ""
    for name in ("final_report.json", "worker_result.json"):
        for p in _walk(root, f"**/{name}"):
            doc = _load_json(p)
            if isinstance(doc, dict):
                declared += json.dumps(doc.get("caveats") or [], ensure_ascii=False)

    open_items, seen = [], 0
    for p in pfs:
        doc = _load_json(p)
        if not isinstance(doc, dict):
            continue
        typed_debts = doc.get("debts")
        if isinstance(typed_debts, list):
            for debt in typed_debts:
                if not isinstance(debt, dict) or not debt.get("debt_id"):
                    open_items.append(("DEBT", "invalid", "preflight", _rel(root, p)))
                    continue
                seen += 1
                closure_item = typed_closures.get(debt["debt_id"])
                if closure_item:
                    closure, closure_ref = closure_item
                    same_identity = (
                        closure.get("status") == "closed"
                        and closure.get("source_sha256") == debt.get("source_sha256")
                        and closure.get("boundary_id") == debt.get("boundary_id")
                        and closure.get("owner") == debt.get("owner")
                        and closure.get("evidence_ref"))
                    if same_identity:
                        continue
                    open_items.append((debt.get("check", "DEBT"), "identity_mismatch",
                                       debt.get("owner"), closure_ref))
                else:
                    open_items.append((debt.get("check", "DEBT"), debt.get("status"),
                                       debt.get("owner"), _rel(root, p)))
            continue
        for c in doc.get("caveats") or []:
            if not isinstance(c, dict) or not c.get("owner"):
                continue
            seen += 1
            name = c.get("check", "?")
            if name == "EVIDENCE" and sol_answered:
                continue
            if re.search(rf"\b{re.escape(name)}\b", declared, re.I):
                continue
            open_items.append((name, c.get("status"), c.get("owner"), _rel(root, p)))

    # Did this run actually rank configurations? A deferral about the configuration surface costs
    # nothing in a run that never swept, and costs everything in one that did.
    swept = bool(_walk(root, "**/plain_best_config.json") or _walk(root, "**/cxx_best_config.json"))
    for name, status, owner, ref in open_items:
        # EVIDENCE is the one that is high. An unanswered [EVIDENCE] means the whole counter layer
        # stayed dark for the run, and it is not a hypothetical: over one eight-kernel campaign
        # this check was SKIPPED or DECLARED_NOT_EXERCISED in 4 of the 4 kernels that left a
        # preflight, every time as a budget decision -- after which a kernel's own skeptic found
        # "no L2-hit or coalescing counter anywhere on disk. Three rounds were fired blind at a
        # symptom." The other deferrals cost a fact; this one costs a bound class.
        #
        # [KNOBS] and [INBODY] are the same shape of debt one level down, and they were low for as
        # long as nothing read what they cost. They name the LIVE AXIS SET -- which knobs this
        # kernel actually reads. Defer them and the sweep still runs, but over a space nobody
        # established: the comparator it pins, every arm's config basis, and the fixed config a
        # climb spends ten rounds at are all quoted against that space. Measured over 19
        # kernel/phase runs, `deferred_never_closed` was the single most frequent finding at 42
        # occurrences, alongside 12 non-pinned comparators and 10 arms that could not say which
        # config their number came from -- the same debt, arriving downstream under other names.
        # A run that never swept is untouched by this: the axis set it did not establish is one it
        # also never used.
        config_surface = name.upper() in ("KNOBS", "INBODY")
        sev = "high" if name.upper() == "EVIDENCE" or (config_surface and swept) else "low"
        why = ""
        if name.upper() == "EVIDENCE":
            why = (". This is an evidence DEBT, not a skipped nicety: every bound-class claim in "
                   "this run was made with group C dark")
        elif config_surface and swept:
            why = (". This run PUBLISHED a swept comparator, so the debt is not hypothetical: the "
                   "configuration space that comparator was ranked over was never established, and "
                   "every arm basis and pinned climb config downstream inherits it")
        _finding(out, "preflight_closure", "deferred_never_closed",
                 f"the entry gate left [{name}] {status} and named '{owner}' as its owner; nothing "
                 f"in this run answers it and no caveat in the roll-up carries it forward. A check "
                 f"nobody ran reads downstream exactly like one that passed" + why,
                 [ref], severity=sev)
    return {"applicable": True, "owned_caveats": seen, "still_open": len(open_items),
            "open": [o[0] for o in open_items]}


def check_claim_citations(root, facts, out):
    """6. Search-ending claims either carry a citation marker or do not.

    Reports presence, never sufficiency. A claim that something is impossible or already optimal is
    what stops a search, so it is the one claim worth knowing the evidence status of -- and judging
    whether the cited evidence actually supports it is a reading task, not an arithmetic one."""
    scanned = uncited = 0
    for path, text in _text_records(root, prose_only=True):
        for i, line in enumerate(text.splitlines(), 1):
            if not CLAIM_RE.search(line):
                continue
            scanned += 1
            if not CITE_RE.search(line):
                uncited += 1
                _finding(out, "claim_citations", "no_citation_marker",
                         f"a search-ending claim with no path, line reference or united number on "
                         f"the same line: {line.strip()[:120]}", [f"{_rel(root, path)}:{i}"])
    return {"claims_seen": scanned, "without_marker": uncited}


def check_tool_reach(root, facts, out):
    """7. Was each evidence role's tool actually reached INSIDE this work root?

    A tool run once by a supervisor at the top of a campaign is not the same as a tool run by the
    direction whose numbers are being reported, and the distinction is invisible in a report. What
    counts as reached here is the tool's name appearing in a record under this root -- which is
    weak evidence, and is reported as the weak evidence it is.

    An UNFILLED role reports as a tool gap, not as a skipped tool. Those are opposite findings, and
    which one applies is a fact about the pack (stamped from its manifest), never about the run."""
    tools = (facts.get("evidence_tools") or {})
    gaps = (facts.get("tool_gaps") or {})
    if not tools:
        _finding(out, "tool_reach", "pack_facts_absent",
                 "no pack_facts.json beside this script, so which tools this pack expects is "
                 "unknown; every reach question below is undeterminable rather than answered")
        return {"reached": [], "not_reached": [], "gaps": sorted(gaps)}
    blob = "\n".join(t for _, t in _text_records(root))
    reached, missing = [], []
    for role, path in sorted(tools.items()):
        if not path:
            _finding(out, "tool_reach", "tool_gap",
                     gaps.get(role) or f"this pack ships no tool for role {role!r}")
            continue
        name = os.path.basename(path)
        if name in blob:
            reached.append(role)
        else:
            missing.append(role)
            _finding(out, "tool_reach", "not_reached",
                     f"role {role!r} is filled by {name} in this pack and that name appears in no "
                     f"record under this root")
    return {"reached": reached, "not_reached": missing, "gaps": sorted(gaps)}


def _collect_points(root):
    """((label, metric), ms, source) for every per-shape row in every record under root.

    A row carries several times at once -- what it was, what it became, what torch does -- so the
    metric name is half of the identity. Reconciling on the shape alone would compare one file's
    baseline against another's champion and call the run's whole speedup a contradiction."""
    pts = []
    for path in _walk(root, "**/*.json"):
        doc = _load_json(path)
        stack = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            for k, v in node.items():
                if isinstance(v, (dict, list)) and k not in POINT_CONTAINERS:
                    stack.append(v)
                if k not in POINT_CONTAINERS or not isinstance(v, list):
                    continue
                for row in v:
                    if not isinstance(row, dict):
                        continue
                    label = next((str(row[x]) for x in LABEL_KEYS if row.get(x) not in (None, "")),
                                 None)
                    if not label:
                        continue
                    for metric in MS_KEYS:
                        ms = row.get(metric)
                        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or ms <= 0:
                            continue
                        pts.append(((label, metric), float(ms), path))
    return pts


def check_duplicate_readings(root, facts, out, band=0.02):
    """8. The same shape carrying two different numbers in two files.

    Timing noise makes two readings of one shape differ a little; the band is what "a little" means
    here and is a parameter, not a constant of nature. What this reports is the pair, not which of
    the two is right -- that is exactly the question the run has to go back and answer, and the
    reason a spread wider than the band matters is that every downstream ratio silently picked one."""
    by_label: dict = {}
    for key, ms, path in _collect_points(root):
        by_label.setdefault(key, []).append((ms, path))
    wide = 0
    for key in sorted(by_label):
        label, metric = key
        readings = by_label[key]
        if len({p for _, p in readings}) < 2:
            continue            # one file quoting itself is not two readings
        vals = [m for m, _ in readings]
        lo, hi = min(vals), max(vals)
        if lo <= 0 or (hi - lo) / lo <= band:
            continue
        wide += 1
        _finding(out, "duplicate_readings", "spread_over_band",
                 f"shape {label!r} has {metric} recorded at {lo:g} and {hi:g} "
                 f"({(hi - lo) / lo * 100:.1f}% apart, band {band * 100:.0f}%); a ratio quoted off "
                 f"one of them is not reproducible from the other",
                 {_rel(root, p) for _, p in readings})
    return {"labels": len(by_label), "over_band": wide, "band": band}


def check_served_regressions(root, facts, out):
    """9. Rows of the served table that came out SLOWER, and whether anything explains them.

    A headline is a weighted average, so a point that regressed can sit inside a reported win and
    never surface. Reporting the row is the whole job here: whether the regression is acceptable
    depends on the served mix, which the report's own weights answer and this does not."""
    rows_seen = regressed = 0
    for p in _walk(root, "**/final_report.json"):
        doc = _load_json(p) or {}
        table = doc.get("served_range")
        if not isinstance(table, list):
            continue
        for row in table:
            if not isinstance(row, dict):
                continue
            rows_seen += 1
            ratio = next((row[k] for k in ("vs_default", "vs_champion", "speedup", "vs_baseline")
                          if isinstance(row.get(k), (int, float))
                          and not isinstance(row.get(k), bool)), None)
            if ratio is None or ratio >= 1.0:
                continue
            regressed += 1
            note = " ".join(str(row.get(k, "")) for k in ("note", "reason", "caveat", "why"))
            _finding(out, "served_regressions",
                     "regressed_with_note" if note.strip() else "regressed_unexplained",
                     f"served row {row.get('shape') or row.get('name')!r} is {ratio:g}x"
                     + ("" if note.strip() else " and the row carries no note saying why"),
                     [_rel(root, p)])
    return {"rows": rows_seen, "regressed": regressed}


def check_ceiling_provenance(root, facts, out):
    """9c. A percent-of-ceiling quoted against a ceiling nobody probed.

    `hw_budget.py` already does the hard half: when the only ceiling available is the SKU's
    datasheet peak it REFUSES to compute a percentage or a gap, sets `ceiling_refused`, and names
    the probe that would settle it. That refusal works -- measured across 19 kernel/phase runs, 11
    of the 13 that never probed also quoted no percentage, which is the honest outcome and the
    reason this check does not simply ask whether the probe ran.

    The two that did quote one are what this reads for. A datasheet peak is a number no real access
    shape reaches, so a fraction of it is not a fraction of anything the kernel could have got: it
    understates the gap, and a headline built on it survives review precisely because it looks
    conservative.
    """
    refused, quoted = [], []
    for p in _walk(root, "**/hw_budget*.json"):
        doc = _load_json(p)
        if not isinstance(doc, dict):
            continue
        w = doc.get("workload") if isinstance(doc.get("workload"), dict) else doc
        if w.get("ceiling_refused") and str(w.get("hbm_ceiling_source") or "") == "datasheet":
            refused.append(_rel(root, p))
    if not refused:
        return {"applicable": False, "reason": "no budget artifact refused a ceiling"}
    pat = re.compile(r"pct_of_hbm_ceiling|%[- ]of[- ](?:the )?(?:hbm )?ceiling|hbm_gap", re.I)
    for name in ("final_report.json", "worker_result.json", "plain_champion.json"):
        for p in _walk(root, f"**/{name}"):
            try:
                with open(p, errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            if pat.search(text):
                quoted.append(_rel(root, p))
    for ref in quoted:
        _finding(out, "ceiling_provenance", "quoted_an_unprobed_ceiling",
                 "this run quotes a percent-of-ceiling or a ceiling gap, and its budget artifact "
                 "refused to compute one because the only ceiling available was the SKU datasheet "
                 "peak -- a number no real access shape reaches. Probe the in-shape ceiling "
                 "(mem_bw_probe.py at this run's length / stride / read:write mix / program count) "
                 "and pass --measured-hbm-tb-s, or drop the fraction and quote the measured time",
                 [ref] + refused[:1], severity="high")
    return {"applicable": True, "refused": len(refused), "quoted_anyway": len(quoted)}


def check_served_split(root, facts, out):
    """9b. Whether the served cases agreed on a winner, and whether anyone looked.

    A sweep runs the driver once per config across every served case and keeps one latency per
    case, then ranks on their aggregate. When the cases disagree -- one config wins decode, another
    wins prefill -- the aggregate winner can be nobody's best, and everything downstream (the
    profile, the census, ten climb rounds) is spent on a config chosen for a shape mix rather than
    for a shape. Nothing in the run says so unless this is read.

    Both outcomes are reportable and neither is a failure. `split` says a bucketed comparator would
    describe this kernel better than the single pin it got. `no_split` is the more useful negative:
    it is the evidence that a per-bucket track would have bought nothing here, which is exactly the
    question a reader deciding whether to build one needs answered. `unlabelled` means the sweep
    never declared its case order, so the readings are on disk and unreadable as shapes.
    """
    split = no_split = unlabelled = 0
    for p in _pin_artifacts(root, facts):
        doc = _load_json(p) or {}
        pc = doc.get("per_case") if isinstance(doc, dict) else None
        if not isinstance(pc, dict):
            continue
        ref = _rel(root, p)
        if not pc.get("winner_per_case"):
            unlabelled += 1
            _finding(out, "served_split", "unlabelled",
                     "the sweep kept a latency per served case but declared no case order, so the "
                     "readings cannot be read as shapes; pass --case (or name them in --objective) "
                     "to make the per-shape winners legible", [ref])
            continue
        if pc.get("split"):
            split += 1
            _finding(out, "served_split", "split",
                     f"the served cases do not share a winner: {', '.join(pc.get('split_cases') or [])} "
                     f"prefer a different config from the aggregate winner this run pinned and "
                     f"climbed, so the pinned comparator is not the best config for every case it "
                     f"is quoted over", [ref])
        else:
            no_split += 1
            _finding(out, "served_split", "no_split",
                     f"one config wins every served case ({len(pc.get('winner_per_case') or {})} "
                     f"cases labelled), so a single pinned comparator describes this range and a "
                     f"per-bucket split would have bought nothing", [ref])
    return {"split": split, "no_split": no_split, "unlabelled": unlabelled}


def check_stale_lever_verdicts(root, facts, out):
    """10. Levers retired BEFORE the body they were measured against reached its final form.

    A negative verdict is a measurement of one lever against one kernel body. Every later accepted
    change makes a new body, and nothing re-opens the verdict -- so a lever ruled out early stays
    ruled out against a kernel that no longer exists. This lists the ones whose record predates the
    moment the winning configuration was pinned. It does not say any of them should be re-run:
    which retirements a body change actually invalidates is a judgment about mechanism."""
    pinned, pin_ref = None, None
    produced = ((facts.get("artifacts") or {}).get("produces") or {})
    # The pin artifact is whatever THIS pack calls its champion/best-config -- taken from the
    # manifest rather than from a list of names, because that list is different in every pack.
    for name in sorted(produced):
        if not any(fnmatch.fnmatch(name, pat) for pat in
                   ("*champion*.json", "*best_config*.json", "*best*.json")):
            continue
        for p in _walk(root, f"**/{name}"):
            mt = _mtime(p)
            if mt is not None and (pinned is None or mt > pinned):
                pinned, pin_ref = mt, _rel(root, p)
    if pinned is None:
        return {"pin_ref": None, "stale": 0,
                "note": "no champion / best-config artifact found, so there is no pin moment to "
                        "compare a verdict's age against"}
    stale = []
    for path in _walk(root, "**/*.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        verdict = str(doc.get("verdict") or doc.get("result") or "").strip().lower()
        if verdict not in NEGATIVE_VERDICTS:
            continue
        mt = _mtime(path)
        if mt is not None and mt < pinned:
            lever = doc.get("lever") or doc.get("tactic_id") or doc.get("direction") or "?"
            stale.append((str(lever), _rel(root, path)))
    for lever, ref in sorted(stale):
        _finding(out, "stale_lever_verdicts", "predates_pin",
                 f"lever {lever!r} was ruled out against a kernel body that predates the pinned "
                 f"configuration ({pin_ref})", [ref])
    return {"pin_ref": pin_ref, "stale": len(stale)}


def check_entry_gate(root, facts, out):
    """0. A pack that fail-closes on entry, or that ships unverified guidance, owes a word about it.

    Generic on purpose. `entry_gate` and `authoring_gap` are manifest facts, so a pack that has
    neither is silently skipped here and a pack that grows one is covered the same day. The reason
    this runs FIRST is that a run which never cleared its entry gate produces artifacts that look
    exactly like measured ones, and every check below would then be auditing paper."""
    gate = facts.get("entry_gate") or {}
    result = {"entry_gate": None, "authoring_gap": facts.get("authoring_gap")}
    if gate.get("artifact"):
        hits = _walk(root, f"**/{gate['artifact']}")
        if not hits:
            _finding(out, "entry_gate", "record_absent",
                     f"this pack declares a fail-closed entry gate ({gate.get('tool')}) and no "
                     f"{gate['artifact']} is on disk; nothing here shows it was ever cleared")
        else:
            doc = _load_json(hits[0]) or {}
            result["entry_gate"] = bool(doc.get("ok"))
            if not doc.get("ok"):
                _finding(out, "entry_gate", "refused",
                         f"{gate['artifact']} records ok=false"
                         + (f": {'; '.join(doc.get('gaps') or [])}" if doc.get("gaps") else "")
                         + " -- every number produced after this point was produced without the "
                           "capability the gate was checking for", [_rel(root, hits[0])])
    if facts.get("authoring_gap"):
        named = any(("unverified_cards_relied_on" in t) for _, t in _text_records(root))
        if not named:
            _finding(out, "entry_gate", "unverified_reliance_unnamed",
                     f"this pack carries authoring_gap={facts['authoring_gap']!r} -- its guidance "
                     f"is documented, not measured -- and no record under this root names which "
                     f"unverified guidance the result leaned on")
    return result


def check_branch_waves(root, facts, out):
    """11. BRANCH closed after ONE wave, with nothing priced to say why that was enough.

    The band is 2-3 waves and `search_mode.py check` has always reported a single-wave close -- as a
    `warn`, which `--strict` does not read, so a one-wave close cost nothing. Measured on one
    campaign: four kernels closed, four ran exactly one wave, `reopened` empty on all four. The
    yield this forfeits is not marginal: hit rate against campaign position is U-shaped and the 70-80% rebound
    to 21.3% comes almost entirely from fan-out and re-sweep, with one kernel going 0-for-140 and
    then 50% over rounds 201-220.

    The exemption is a PRICE, not a sentence. `worker_result.branch.waves_declined` naming a bounded
    remaining prize ("all remaining structure is <= 2.2% geomean, measured against a minimal-work
    kernel") is a reason to stop; "no further axes looked promising" is the thing this check exists
    to stop being free. So the finding is raised unless the declination carries a number.

    A NUMBER IS NOT YET A PRICE. The test used to be `re.search(r"\d", ...)`: any digit anywhere in
    the declination bought the downgrade. Two audited closes then priced a declined wave by
    attributing most of the remaining time to a block declared beyond the run's reach, with no
    measurement of that block -- and a later wave, entered from the same boundary, recovered part of
    one of them with paired measurement. So a declination that rests on unreachability owes the same
    thing the census owes for the same claim: an artifact that RAN, and the statement that the
    figure is an upper bound. A price whose denominator was never measured is an assertion wearing a
    number, and it is the cheapest way to end a search.
    """
    result = {"ledgers": 0, "single_wave": [], "run_closed": False}
    # IS THE RUN OVER. Mid-run this check must stay quiet -- an open BRANCH may still re-enter, and a
    # finding on every intermediate audit is a finding nobody reads. Once the run has closed, an open
    # BRANCH is not "may still re-enter": it is a mode left open at the close, which is the same
    # forfeit as an exhausted one and was the shape of three of four measured kernels.
    closed_refs = [_rel(root, p) for p in _walk(root, "**/final_report.json")]
    for wr in _walk(root, "**/worker_result.json"):
        if str((_load_json(wr) or {}).get("status") or "").strip().lower() in ("done", "closed"):
            closed_refs.append(_rel(root, wr))
    run_closed = bool(closed_refs)
    result["run_closed"] = run_closed

    priced, unmeasured = None, None
    for wr in _walk(root, "**/worker_result.json"):
        w = _load_json(wr) or {}
        b = w.get("branch") if isinstance(w.get("branch"), dict) else {}
        raw = b.get("waves_declined")
        if raw is None:
            raw = w.get("waves_declined")
        decl = raw if isinstance(raw, dict) else {"statement": str(raw or "").strip()}
        text = _stringify(decl)
        if not (text and re.search(r"\d", text)):
            continue
        if _claims_unreachable(text) and not _measured_bound(decl):
            unmeasured = (text, _rel(root, wr))
            continue
        priced = (text, _rel(root, wr))
        break

    ledgers = _walk(root, "**/search_mode_ledger.json")
    result["ledgers"] = len(ledgers)
    if not ledgers and run_closed and facts.get("breadth_enabled"):
        _finding(out, "branch_waves", "no_mode_ledger",
                 "the run is closed and no search_mode_ledger.json exists under this root. The mode "
                 "ledger is what makes BRANCH re-entrant -- it is the thing that says a mode was "
                 "entered, exhausted with evidence, and may be re-entered from a NEW bottleneck "
                 "stack. Without it `search_mode.py check` cannot run at all, so neither the wave "
                 "count nor the never-entered modes were ever visible", severity="high")
        return result
    for path in ledgers:
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        br = ((doc.get("modes") or {}).get("branch") or {})
        if not isinstance(br, dict) or br.get("state") not in ("open", "exhausted"):
            continue
        waves = max(len(br.get("entered_rounds") or []), 1 + int(br.get("reopened") or 0))
        if waves >= 2 or not run_closed:
            continue
        ref = _rel(root, path)
        result["single_wave"].append(ref)
        if priced:
            _finding(out, "branch_waves", "single_wave_declined_with_a_price",
                     f"BRANCH ran 1 wave against a band of 2-3 and stopped on a PRICED declination "
                     f"({priced[0][:160]}) -- recorded, not contested", [ref, priced[1]],
                     severity="low")
            continue
        if unmeasured:
            _finding(out, "branch_waves", "declination_price_unmeasured",
                     f"BRANCH ran 1 wave against a band of 2-3 and the declination puts the "
                     f"remaining cost beyond this run's reach ({unmeasured[0][:160]}). That claim "
                     f"ends the search, so it owes an artifact that RAN and an explicit upper "
                     f"bound: declare `waves_declined` as an object carrying "
                     f"{{pct|remaining_prize_pct, evidence_ref{{kind: probe|measurement, "
                     f"artifact}}, method, is_upper_bound: true}}. A digit in the sentence is not a "
                     f"price -- the denominator has to have been measured",
                     [ref, unmeasured[1]], severity="high")
            continue
        left_open = br.get("state") == "open"
        _finding(out, "branch_waves", "single_wave_close",
                 f"BRANCH ran ONE wave and the band is 2-3"
                 + (", and it is still `open` at the close -- the mode was entered once and never "
                    "exhausted with evidence, so nothing recorded what the search had run out of"
                    if left_open else ", and it is exhausted") +
                 ". Wave two enumerates from a DIFFERENT source than wave one -- the current "
                 "bottleneck stack plus RECALL's stale_negatives, not a replay of CENSUS-0 -- so it "
                 "is strictly new information rather than another sample of the same enumeration. If "
                 "one wave was genuinely enough, say so with a NUMBER in "
                 "worker_result.branch.waves_declined (a bounded remaining prize); a single-wave "
                 "close with no price is the measured failure mode, not the norm",
                 [ref] + closed_refs[:1], severity="high")
    return result


_CLIMB_ROUND_FLOOR = 10

_BUDGET_CONTAINERS = ("budget", "rounds", "rounds_spent", "budget_and_ceiling", "time")
_USED_KEYS = ("rounds_used", "spent", "rounds_spent", "used")
_CAP_KEYS = ("round_budget", "budget", "rounds", "cap")


def _read_round_budget(doc):
    """(rounds_used, round_budget) from a report, flat or inside the containers reports use."""
    scopes = [doc] + [v for k, v in doc.items()
                      if k in _BUDGET_CONTAINERS and isinstance(v, dict)]
    used = cap = None
    for scope in scopes:
        for k in _USED_KEYS:
            v = scope.get(k)
            if used is None and isinstance(v, int) and not isinstance(v, bool):
                used = v
        for k in _CAP_KEYS:
            v = scope.get(k)
            if cap is None and isinstance(v, int) and not isinstance(v, bool) and v > 0:
                cap = v
    return used, cap


def _wall_clock_left(state):
    """Fraction of the declared task time limit still unspent when the state last moved."""
    d = state.get("deadline")
    if not isinstance(d, dict):
        return None
    limit = d.get("time_limit_s")
    if not isinstance(limit, int) or limit <= 0:
        return None
    try:
        start = _parse_ts(d.get("started_at"))
        end = _parse_ts(state.get("updated_at"))
    except (TypeError, ValueError):
        return None
    if start is None or end is None:
        return None
    used = (end - start).total_seconds()
    if used < 0:
        return None
    return max(0.0, 1.0 - used / limit)


def _parse_ts(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def check_climb_depth(root, facts, out):
    """11b. The run closed with budget in hand and a line it had barely climbed.

    The counterpart to `check_branch_waves`, and it exists because that check had no counterpart.
    BRANCH width was counted against a declared band; CLIMB depth was counted by nothing, so the
    audit could only ever be wrong in one direction -- it priced stopping the fan-out early and gave
    away stopping the climb early for free. Measured across four kernels on one campaign: every one
    of them closed one to three climb rounds deep, two of them finalized with a third to a half of
    their wall clock unspent, and all four passed a strict close audit. Two of the four wrote that
    the binding constraint was the clock; their own timestamps say the clock was not binding.

    WHAT THE FLOOR IS. Not a quota. The pack declares reconsideration after ten contiguous
    comparable non-improving CLIMB rounds -- that is the only stopping condition it names for this
    mode, so a climb that ends before ten rounds ended before its own rule could be evaluated. The
    number is therefore read off the contract, not chosen here.

    WHAT MAKES IT HIGH. Shallow on its own is a fact about a run; the run may have had nothing left
    to spend. Shallow WITH the budget visibly unspent contradicts something the close already
    banked: `exhausted`, `at_ceiling`, or a time limit named as the binding constraint. And as with
    a declined BRANCH wave, the way out is a price -- a bounded remaining prize with an artifact
    that ran -- not a sentence saying the line looked done.
    """
    result = {"climb_rounds": None, "run_closed": False, "unspent": [], "priced": None}
    closed_refs = [_rel(root, p) for p in _walk(root, "**/final_report.json")]
    for wr in _walk(root, "**/worker_result.json"):
        if str((_load_json(wr) or {}).get("status") or "").strip().lower() in ("done", "closed"):
            closed_refs.append(_rel(root, wr))
    if not closed_refs:
        return result                      # mid-run this check says nothing, exactly as BRANCH does
    result["run_closed"] = True

    # A run that stopped for a reason OTHER than deciding it was finished obviously has budget left,
    # and saying so would be noise on top of the blocker the reader is already looking at. Only an
    # optimization close is answerable for its depth: the question is whether the search stopped
    # early, and a blocked, refused or user-deferred run did not choose to stop at all.
    for path in _walk(root, "**/final_report.json") + _walk(root, "**/run_state.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        for field in ("status", "state", "result", "terminal_status"):
            v = str(doc.get(field) or "").strip().lower()
            if v.startswith(("blocked", "refus", "deferred", "needs_user")):
                result["not_an_optimization_close"] = v
                return result

    # Depth, counted every way it is recorded and reconciled to the largest. `--round` is optional
    # on the mode ledger and the bundle's own count may be absent, so no single source is complete
    # and the smallest would let an incomplete record read as a shallow climb.
    rounds, ref = 0, None
    for path in _walk(root, "**/search_mode_ledger.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        links = doc.get("round_links") if isinstance(doc.get("round_links"), dict) else {}
        entered = ((doc.get("modes") or {}).get("climb") or {}).get("entered_rounds") or []
        n = max(sum(1 for v in links.values() if v == "climb"), len(entered))
        if n > rounds:
            rounds, ref = n, _rel(root, path)
    # A ledger row's `mode` is OPTIONAL in the round writer, so counting only `mode == "climb"` --
    # which is all this did -- reads every journal written without `--mode` as zero rounds deep.
    # Measured on one campaign: not one of the four journals carried a `mode` key, so on 75-,
    # 80- and 16-row ledgers this term was 0 every time and the bundle's asserted integer was the
    # only number left standing. Count the rounds the ledger actually distinguishes as well, and
    # take whichever reading is larger: a row per round is a record of a round whether or not the
    # writer was told which mode it belonged to.
    for pattern in ("**/optimization_journal.jsonl", "**/rounds.jsonl"):
        for path in _walk(root, pattern):
            tagged, seen, untagged = 0, set(), 0
            try:
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            tagged += int(row.get("mode") == "climb")
                            if isinstance(row.get("round"), int) and not isinstance(
                                    row.get("round"), bool):
                                seen.add(row["round"])
                            else:
                                untagged += 1
                        except (ValueError, AttributeError):
                            pass
            except OSError:
                continue
            n = max(tagged, len(seen) + untagged)
            if n > rounds:
                rounds, ref = n, _rel(root, path)
    # Everything counted so far came from a PER-ROUND record: a link per round in the ledger, a line
    # per round in the journal. Keep that total apart from the bundle's own `climb.rounds`, which is
    # one integer a writer asserted. Both are legitimate readings of depth and only one of them can
    # be checked, so reconciling them to a single max -- which is what this did -- lets the
    # unverifiable one satisfy the floor on its own.
    witnessed, witness_ref = rounds, ref
    declined, declared, declared_ref = None, 0, None
    # Both spellings, because real reports use both and reading one of them was the same
    # defect as reading one journal field: `climb.rounds` in the champion and worker bundles,
    # `climb_rounds` at the report's top level. A declaration this check cannot see is a
    # declaration it cannot hold to a record.
    for path in (_walk(root, "**/plain_champion.json") + _walk(root, "**/worker_result.json")
                 + _walk(root, "**/final_report.json")):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        climb = doc.get("climb") if isinstance(doc.get("climb"), dict) else {}
        for n in (climb.get("rounds"), doc.get("climb_rounds")):
            if isinstance(n, int) and not isinstance(n, bool) and n > declared:
                declared, declared_ref = n, _rel(root, path)
        if declined is None and doc.get("climb_declined") is not None:
            declined = (doc["climb_declined"], _rel(root, path))
    if declared > rounds:
        rounds, ref = declared, declared_ref
    result["climb_rounds"] = rounds
    result["climb_rounds_witnessed"] = witnessed
    result["climb_rounds_declared"] = declared

    # THE DEPTH THAT CLEARS THE FLOOR HAS TO BE THE ONE SOMETHING RECORDED. Measured on one
    # campaign, on the run that closed as its only clean acceptance: `climb.rounds: 86`
    # in `plain_champion.d/plain_champion.json`, `optimization_journal.jsonl` zero bytes, and a
    # `search_mode_ledger.json` whose `round_links` held exactly one entry -- so the audit read 86,
    # returned at the floor, and never looked at a round. The 86 rounds were real; they were logged
    # to `.direction/climb_log.jsonl`, which is not in the contract and which nothing reads. That is
    # the defect: a run that genuinely climbed 86 rounds and a run that typed 86 were, to this
    # check, the same run. Naming it is also the cheapest way to get the history somewhere a
    # reviewer can find it.
    if declared >= _CLIMB_ROUND_FLOOR and witnessed < _CLIMB_ROUND_FLOOR:
        _finding(out, "climb_depth", "climb_depth_unwitnessed",
                 f"the close reports {declared} CLIMB round(s) as a single asserted integer, and the "
                 f"per-round records under this root witness {witnessed}. A depth that clears the "
                 f"floor of {_CLIMB_ROUND_FLOOR} on an unverifiable reading is not a checked depth: "
                 f"this check cannot tell a run that climbed {declared} rounds from one that wrote "
                 f"the number. Depth is counted from a link per round in `search_mode_ledger.json` "
                 f"or a line per round in `optimization_journal.jsonl` -- a private per-round log "
                 f"under `.direction/` or any other off-contract path is not read by this or any "
                 f"other check. Emit the rounds the run actually ran through the round writer, or "
                 f"correct `climb.rounds` to the depth the record supports",
                 [r for r in (declared_ref, witness_ref) if r] + closed_refs[:1], severity="high")
        return result
    if rounds >= _CLIMB_ROUND_FLOOR:
        return result

    # What the close still had. Both readings are the run's own declarations, not inferred from
    # file times: the round count comes from the report that closed, and the wall clock from the
    # interval between the deadline the run set at entry and the last move its state recorded.
    unspent, under, refs = [], [], []
    for path in _walk(root, "**/final_report.json") + _walk(root, "**/worker_result.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        used, cap = _read_round_budget(doc)
        if used is None or not cap or used > cap:
            continue
        (unspent if (cap - used) / cap >= 0.5 else under).append(f"{used} of {cap} rounds spent")
        refs.append(_rel(root, path))
        break
    for path in _walk(root, "**/run_state.json"):
        state = _load_json(path)
        if not isinstance(state, dict):
            continue
        left = _wall_clock_left(state)
        if left is None:
            continue
        (unspent if left >= 0.25 else under).append(
            f"{left * 100:.0f}% of the declared wall clock unspent")
        refs.append(_rel(root, path))
        break
    result["unspent"], result["under_threshold"] = unspent, under

    if declined is not None and _measured_bound(
            declined[0] if isinstance(declined[0], dict) else {"statement": str(declined[0])}):
        result["priced"] = declined[1]
        _finding(out, "climb_depth", "shallow_climb_declined_with_a_price",
                 f"CLIMB ran {rounds} round(s) against a floor of {_CLIMB_ROUND_FLOOR} and stopped "
                 f"on a PRICED declination ({_stringify(declined[0])[:160]}) -- recorded, not "
                 f"contested", ([ref] if ref else []) + [declined[1]], severity="low")
        return result
    if unspent:
        _finding(out, "climb_depth", "shallow_climb_with_budget_left",
                 f"the run closed after {rounds} CLIMB round(s) with {' and '.join(unspent)}. The "
                 f"floor is {_CLIMB_ROUND_FLOOR} because that is where the pack's own stopping rule "
                 f"for this mode -- reconsideration after ten contiguous non-improving rounds -- "
                 f"first becomes answerable, so this climb ended before anything could say the line "
                 f"was done. Whatever the close banked (`exhausted`, `at_ceiling`, or a clock named "
                 f"as the binding constraint) is contradicted by the run's own accounting. Either "
                 f"the remaining budget goes back into the climb, or price what is left: declare "
                 f"`climb_declined` as an object carrying {{remaining_prize_pct, evidence_ref{{kind: "
                 f"probe|measurement, artifact}}, method, is_upper_bound: true}}",
                 ([ref] if ref else []) + refs + closed_refs[:1], severity="high")
        return result
    # Two different LOW answers, and saying the wrong one misleads the reader about what to fix. A
    # run whose budget is recorded and merely sits under the threshold is not a gap in the record --
    # most of its budget went somewhere, and where it went (usually BRANCH) is the thing to look at.
    if under:
        _finding(out, "climb_depth", "shallow_climb_inside_the_budget",
                 f"the run closed after {rounds} CLIMB round(s), below the floor of "
                 f"{_CLIMB_ROUND_FLOOR}, with {' and '.join(under)} -- recorded, and under the "
                 f"threshold this check calls unspent. So the budget WAS consumed; it was consumed "
                 f"somewhere other than the climb. That is a legitimate shape (a wide BRANCH is "
                 f"expensive) and it is also how a shallow climb stops being visible, so the "
                 f"reader is owed where the rounds went",
                 ([ref] if ref else []) + refs + closed_refs[:1], severity="low")
        return result
    _finding(out, "climb_depth", "climb_depth_unaccounted",
             f"the run closed after {rounds} CLIMB round(s), below the floor of "
             f"{_CLIMB_ROUND_FLOOR}, and nothing in this tree records what it had left to spend -- "
             f"no round budget in the report, no task deadline in run_state.json. Depth cannot be "
             f"read against a budget nobody wrote down, so this is a gap in the record rather than "
             f"a contradiction in it",
             ([ref] if ref else []) + closed_refs[:1], severity="low")
    return result


# The memory side of the round contract, named two ways. `pmc_tcc` is the ledger's token for it;
# the counter spellings are what a captured tool log holds when the read happened outside the
# ledger. Both vendors' spellings are here because this file ships byte-identical in every pack, and
# a run that read the right thing must not be accused for having written it somewhere else.
_DIAL_C_TOKEN = "pmc_tcc"
_DIAL_C_COUNTERS = re.compile(
    r"\b(?:TCC_HIT_sum|TCC_MISS_sum|TCC_EA0?_(?:RD|WR)REQ|FETCH_SIZE|WRITE_SIZE|L2CacheHit"
    r"|dram__bytes|lts__t_sector|l1tex__t_sector)\b", re.I)

# "Which side" is not "which level". A roll-up that says memory names the side; the lever depends on
# the level, and these are the sentences that use the side as if it were the level.
_MEMORY_BOUND_CLAIM = re.compile(
    r"\b(?:memory|bandwidth|dram|hbm|l2|cache|byte)[\s_-]*(?:bound|bounded|limited|constrained"
    r"|saturated|the\s+bottleneck)\b|\bbound\s+is\s+(?:memory|bandwidth|dram|hbm|l2|cache)\b", re.I)


def check_dial_c(root, facts, out):
    """11c. A memory bound the run steered by and never counted.

    NOT a per-round tax, deliberately. The round contract names four readings and this check asks
    about exactly one of them, only when the run's own prose puts weight on it. Requiring C every
    round would be both wrong (most rounds do not need it) and expensive in the one currency that
    matters -- rounds spent satisfying the audit are rounds not spent climbing, and a 274-record /
    16-sweep campaign is what that trade looks like when it goes badly.

    So the trigger is the CLAIM, not the round. A SOL roll-up says which SIDE the kernel is on;
    group C says which LEVEL, and the levers are different per level: an L2-resident working set
    wants reuse and access shape, a fabric-saturated one wants fewer bytes, and a latency-bound one
    that reads as "memory" wants neither. A run that asserts a memory bound with C dark chose its
    levers against a distinction nothing in the tree measured. Measured across one campaign: C was
    absent from every recorded round in all four kernels, and every close then argued about a bound
    nothing had counted.

    HIGH only when the assertion sits on a line that also ENDS the search -- there the uncounted
    bound is load-bearing for the close. Elsewhere it is a LOW note about how the round was steered.
    """
    result = {"dial_c_read": False, "memory_claims": 0, "refs": []}
    for path in _walk(root, "**/*.jsonl"):
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    if _DIAL_C_TOKEN in line:
                        result["dial_c_read"] = True
                        result["refs"].append(_rel(root, path))
                        break
        except OSError:
            continue
        if result["dial_c_read"]:
            break
    if not result["dial_c_read"]:
        for path, text in _text_records(root):
            if _DIAL_C_TOKEN in text or _DIAL_C_COUNTERS.search(text):
                result["dial_c_read"] = True
                result["refs"].append(_rel(root, path))
                break
    if result["dial_c_read"]:
        return result

    ending, noting = [], []
    for path, text in _text_records(root, prose_only=True):
        for i, line in enumerate(text.splitlines(), 1):
            if not _MEMORY_BOUND_CLAIM.search(line):
                continue
            result["memory_claims"] += 1
            (ending if CLAIM_RE.search(line) else noting).append(
                (f"{_rel(root, path)}:{i}", line.strip()[:120]))
    if ending:
        _finding(out, "dial_c", "search_ended_on_an_uncounted_bound",
                 f"the search ends on a memory bound and no group-C read exists anywhere in this "
                 f"tree: no round carries the `{_DIAL_C_TOKEN}` evidence layer and no record holds "
                 f"the counters behind a roll-up. A SOL table names the SIDE; which LEVEL is "
                 f"saturated is what selects the lever, and the three answers (L2-resident reuse, "
                 f"fabric bytes, latency wearing a memory shape) call for different and mutually "
                 f"exclusive moves. {ending[0][1]}",
                 [r for r, _ in ending[:4]], severity="high")
    elif noting:
        _finding(out, "dial_c", "memory_bound_asserted_without_group_c",
                 f"{len(noting)} statement(s) put the bound on the memory side and nothing in this "
                 f"tree read the counters behind it. Not a contradiction -- the close does not rest "
                 f"on it -- but the levers chosen in those rounds were chosen against a level "
                 f"nobody measured. {noting[0][1]}",
                 [r for r, _ in noting[:4]], severity="low")
    return result


def check_comparator_trust(root, facts, out):
    """12. The whole run was quoted against a comparator its own tool refused to pin.

    `plain_autotune` emits `trust_level` in three states and says what the winner may be USED as:
    `pinned` is a comparator, `provisional` is a number. Measured: every captain-run sweep
    on four kernels came back `provisional` -- unaudited axes, an unattested meter, seconds-per-read
    two to twenty-two times the ceiling -- and all four were nonetheless carried as the run's
    baseline for every later claim. `champion_gate`'s SAMPLING check only fires on
    `partially_sampled=True`, which none of them set, so nothing stood between the two.

    This does not say the run is wrong. It says the speedup carries the comparator's discount, and
    that a reader is entitled to see the blockers that stopped the pin next to the headline.
    """
    result = {"envelopes": 0, "provisional": []}
    seen = {}
    for path in _walk(root, "**/plain_best_config.json") + _walk(root, "**/plain_champion.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict) or "trust_level" not in doc:
            continue
        result["envelopes"] += 1
        trust = str(doc.get("trust_level") or "").strip().lower()
        if trust == "pinned":
            continue
        blockers = [b.get("code") for b in (doc.get("pin_blockers") or [])
                    if isinstance(b, dict) and b.get("code")]
        seen[_rel(root, path)] = (trust, blockers)
    for ref, (trust, blockers) in sorted(seen.items()):
        result["provisional"].append(ref)
        # `comparator_not_pinned`, not `comparator_provisional`: `arm_config_basis` already emits the
        # second name for a different fact (the arms' shared comparator), and two checks answering to
        # one kind is how a reader ends up disposing of the wrong one.
        _finding(out, "comparator_trust", "comparator_not_pinned",
                 f"the comparator this run is quoted against is trust_level={trust!r}, not `pinned`"
                 + (f" -- blocked by {', '.join(sorted(set(blockers)))}" if blockers else "")
                 + ". Every speedup downstream carries that discount, and the tool's own ladder says "
                   "a non-pinned winner is a number rather than a comparator. Either clear the "
                   "blockers or state the discount next to the headline",
                 [ref], severity="high")
    return result


def check_axis_audit_closed(root, facts, out):
    """13. A comparator produced over an axis surface that was never closed.

    `axis_audit` leaves every axis `swept` / `derived` / `pinned` / `classified` / `needs_edit` /
    `unaudited` / `unproven`, and an ABSENT axis leaves the same envelope as a REJECTED one -- which
    is the whole reason the audit exists. Measured: one kernel's captain sweep carried
    three `unaudited` axes (NUM_KSPLIT, SPLITK_BLOCK_SIZE, cache_modifier) and five `needs_edit`, and
    that envelope was the comparator for all 25 trunk rounds and all three BRANCH arms.

    `unproven` is separated out because it is a different repair: the axis WAS probed and the probe
    could not read it (fewer than two comparable artifacts), so the action is the compile cmd's IR
    destination rather than a sweep or a classification.

    A COMPARATOR WITH NO `axis_audit` AT ALL used to `continue` -- so the whole check was skipped by
    the one route that most needs it. Measured on one fleet: three of four comparators were emitted
    on the no-external-injection route, none of them carried the block, and the axis surface was
    therefore never audited on any of them; the fourth, which did carry it, audited clean. That is
    the same defect this check's own docstring names one paragraph up: an ABSENT record and a CLEAN
    one leave the same envelope, and here the absent one was the record of the audit itself.
    """
    result = {"envelopes": 0, "open_axes": {}}
    surface = _editable_surface(root)
    for path in _walk(root, "**/plain_best_config.json"):
        doc = _load_json(path)
        audit_d = (doc or {}).get("axis_audit")
        ref = _rel(root, path)
        if not isinstance(audit_d, dict):
            _finding(out, "axis_audit_closed", "comparator_without_axis_audit",
                     "this comparator carries no axis_audit block, so which axes were swept, "
                     "derived, pinned, classified or never looked at is unrecorded -- and an "
                     "unaudited surface leaves the same envelope as an audited-and-closed one. "
                     "Emit the block from the sweep that produced this comparator; if the "
                     "comparator is a default anchor rather than a sweep winner, the audit is what "
                     "shows the surface was measured empty rather than never read",
                     [ref], severity="high")
            continue
        result["envelopes"] += 1
        unaudited = [a for a in (audit_d.get("unaudited") or [])]
        unproven = [a for a in (audit_d.get("unproven") or [])]
        needs_edit = [a for a in (audit_d.get("needs_edit") or [])]
        if unaudited or unproven:
            result["open_axes"][ref] = {"unaudited": unaudited, "unproven": unproven}
        if unaudited:
            _finding(out, "axis_audit_closed", "unaudited_axes_in_the_comparator",
                     f"{', '.join(sorted(unaudited))} -- a discovery source names {'it' if len(unaudited) == 1 else 'them'} "
                     f"and this sweep neither swept nor classified {'it' if len(unaudited) == 1 else 'them'}, yet the "
                     f"envelope was used as a comparator. An absent axis leaves the same record as a "
                     f"rejected one, so the config surface is not shown to be closed. Sweep them, or "
                     f"record knobs.classify with a reason", [ref], severity="high")
        if unproven:
            _finding(out, "axis_audit_closed", "unproven_axes_in_the_comparator",
                     f"{', '.join(sorted(unproven))} -- knob_probe compiled {'it' if len(unproven) == 1 else 'them'} and "
                     f"recovered fewer than two comparable artifacts, so whether {'it is' if len(unproven) == 1 else 'they are'} "
                     f"live is UNMEASURED. The space searched is the derivation intersected with a "
                     f"partially-READ knob set. Thread {{IR_OUT}} through the probe's --compile-cmd "
                     f"and re-probe", [ref], severity="high")
        if needs_edit:
            # SEVERITY BY DECLARATION, not by assumption. "Hardcoded in the kernel" is only a
            # caveat if the kernel is not writable here. When the run's declared editable surface
            # covers the source, the same line is an axis the run was allowed to reach and did not
            # -- and on one measured kernel that axis was the entire difference between closing
            # near parity and closing at a large multiple. Both readings were previously in use, in
            # opposite directions, decided by whichever default a check encoded; now the run says
            # which it is and this check reads the declaration. A pack that declares nothing keeps
            # the conservative `low`.
            editable = bool(surface and surface["paths"])
            _finding(out, "axis_audit_closed", "needs_edit_axes",
                     f"{', '.join(sorted(needs_edit))} {'is' if len(needs_edit) == 1 else 'are'} "
                     f"hardcoded in the kernel, so this comparator is the best config reachable "
                     f"WITHOUT editing it"
                     + (f" -- and the run's declared editable surface ({surface['ref']}) DOES "
                        f"cover its source, so this is an axis the contract allowed and the search "
                        f"did not take, not a caveat. Promote it and sweep the reachable axis, or "
                        f"record the reading that says promoting it does not pay"
                        if editable else
                        " -- a qualifier that has to travel with the number rather than sit beside "
                        "it. No editable-surface declaration was found under this root, so the "
                        "conservative reading is kept; declare one and this becomes decidable"),
                     [ref] + ([surface["ref"]] if editable else []),
                     severity="high" if editable else "low")
    return result


# WHAT ADMISSION IS FOR, and why a shape mismatch earns a HIGH. Every artifact below has a writer
# with a controlled vocabulary -- `structure_census.py` refuses a disposition outside three values
# and a state outside five, `branch_plan.py` refuses a roster under the fan-out floor -- and every
# consumer downstream finds the file by NAME and then reads it with `.get()`. So a file that never
# went through its writer is read by every consumer as an answer, and the consumers that cannot find
# their key skip it in silence.
#
# Measured on one campaign, on the kernel that lost the most to a premature disposition: its
# `structure_census.json` carried schema `kernel_opt.structure_census/1` where the writer emits
# `structure_census/3`, had no `questions` key at all, and described nine candidates with a
# hand-rolled `status: ACTIVE|DEFERRED-CONDITIONAL|DISPOSED` instead of the `disposition`/`state`
# pair the writer validates. `structure_census.py check --strict` raises KeyError on it and
# `check_census_provenance` skipped it, so the hand-written census was SILENT while a MISSING one is
# a high finding. Its `branch_plan.json` was forged the same way -- schema `kernel_opt.branch_plan/1`
# against the writer's `branch_plan/1`, and no `candidate_roster`. Three structural candidates left
# the roster through a status the contract does not have, and the fan-out never saw them again.
#
# That is the incentive backwards, and closing it is the whole point: after this check, not writing
# the artifact and writing one that nothing produced cost the same.
#
# The ledger is deliberately NOT here. `search_mode_ledger.json` was well-formed in the same
# campaign and still under-reported its depth by 85 rounds -- a conforming file carrying a wrong
# number is `check_climb_depth`'s question, not this one. Admission answers only whether the writer
# ran; it never reads a value.
ADMITTED_ARTIFACTS = (
    ("**/structure_census.json", "structure_census/", (("questions", dict),),
     "scripts/structure_census.py"),
    ("**/branch_plan.json", "branch_plan/", (("candidate_roster", list),),
     "scripts/branch_plan.py"),
)


def check_artifact_admission(root, facts, out):
    """14a. An artifact its own writer would have refused, read downstream as an answer.

    The gate in front of every check that resolves an artifact by filename. It asks one question --
    did this file come through the tool that owns it -- and answers it from the schema the writer
    stamps plus the keys the writer always emits. It reads no values and judges no decision.

    A hand-forged artifact is not a paperwork defect. The census is what BRANCH enumerates from, so
    a candidate carrying a status the vocabulary does not contain is a candidate that silently
    leaves the roster: the contract's only way to park an unmeasured candidate is `deferred`, which
    `set_q` refuses without `--lower-bound-pct` and which `deferred_never_closed` then reopens at
    the close. Bypass the writer and both of those protections are gone at once.
    """
    result = {"checked": 0, "rejected": []}
    for pattern, prefix, required, writer in ADMITTED_ARTIFACTS:
        for path in _walk(root, pattern):
            ref = _rel(root, path)
            result["checked"] += 1
            doc = _load_json(path)
            if not isinstance(doc, dict):
                result["rejected"].append(ref)
                _finding(out, "artifact_admission", "not_tool_produced",
                         f"{ref} is unreadable or is not a JSON object, so {writer} did not write "
                         f"it. Every consumer resolves this artifact by filename and then reads it "
                         f"with `.get()`, so an unreadable file is not skipped downstream -- it is "
                         f"read as an artifact with no content and therefore no objections",
                         [ref], severity="high")
                continue
            schema = str(doc.get("schema") or "")
            if not schema.startswith(prefix):
                result["rejected"].append(ref)
                _finding(out, "artifact_admission", "not_tool_produced",
                         f"{ref} carries schema {schema or '<absent>'!r}, and {writer} stamps "
                         f"{prefix}*. A file this tool did not write did not pass its refusals "
                         f"either, so the vocabularies that make this artifact checkable -- the "
                         f"closed disposition and state sets, the priced deferral, the roster floor "
                         f"-- were never applied to it. Re-emit it through {writer}; do not rename "
                         f"the schema of a hand-written file to match",
                         [ref], severity="high")
                continue
            missing = sorted(key for key, kind in required if not isinstance(doc.get(key), kind))
            if missing:
                result["rejected"].append(ref)
                _finding(out, "artifact_admission", "not_tool_produced",
                         f"{ref} claims schema {schema!r} but is missing {missing} in the shape "
                         f"{writer} always emits. The schema string is the one part of an artifact a "
                         f"hand-written file can copy, so it is not on its own evidence the writer "
                         f"ran; the keys every consumer reads are. A consumer that cannot find its "
                         f"key skips this file silently and reports nothing",
                         [ref], severity="high")
    return result


def check_census_provenance(root, facts, out):
    """14. CENSUS-0 either did not run, or its record did not go through the tool that refuses.

    Two separate defects and one check, because they have the same consequence: the six L0 questions
    are the input BRANCH enumerates from, and an axis absent from the census leaves a record
    identical to one considered and killed.

    (a) NO CENSUS AT ALL. Measured: three of eight kernels have no `structure_census.json`
    anywhere under their root. On one of them the six questions were answered in a markdown decision
    log instead, which is a fine thing to write and not a thing `SKILL.md 4.1` can read.

    (b) A DEFERRAL WITH NO PRICE. `structure_census.py set_q` REFUSES `--disposition deferred`
    without `--lower-bound-pct`, so a deferral with a null price cannot have come through the tool --
    the file was hand-written. That refusal is not a formality: the most expensive single miss
    measured was an unpriced deferral, a lead its own report valued at "c2 1.53 -> 2.8-3.1x" that
    was recorded, never started, and then shipped by a competing system from the other side.
    """
    result = {"census_files": 0, "unpriced_deferrals": [], "missing": False}
    files = _walk(root, "**/structure_census.json")
    result["census_files"] = len(files)
    if not files:
        # Only a finding for a pack that runs a census at all -- breadth_enabled is the test, since a
        # deep-dig pack has no CENSUS-0 phase and must not be told it skipped one.
        if facts.get("breadth_enabled"):
            result["missing"] = True
            _finding(out, "census_provenance", "no_census",
                     "no structure_census.json under this root, and this pack runs CENSUS-0. The six "
                     "L0 questions are what BRANCH enumerates from, so with no census the fan-out was "
                     "enumerated from the profile alone -- which cannot see the launcher, the call "
                     "contract or the data layout. Answering them in prose elsewhere does not make "
                     "them readable by the phase that consumes them",
                     severity="high")
        return result
    for path in files:
        doc = _load_json(path)
        qs = (doc or {}).get("questions")
        if not isinstance(qs, dict):
            # A census with no questions block is not a priced-deferral question at all -- there is
            # nothing here to price. `check_artifact_admission` owns that file and reports it HIGH;
            # this skip must stay a skip so one forged census does not read as two defects. Before
            # admission existed the skip WAS the defect: it made a hand-written census cheaper than
            # a missing one.
            continue
        ref = _rel(root, path)
        for name, q in sorted(qs.items()):
            if not isinstance(q, dict):
                continue
            if str(q.get("disposition") or "").strip().lower() != "deferred":
                continue
            if q.get("lower_bound_pct") in (None, ""):
                result["unpriced_deferrals"].append(f"{ref}#{name}")
                _finding(out, "census_provenance", "deferral_without_a_price",
                         f"{name} is `deferred` with lower_bound_pct null. `structure_census.py "
                         f"set_q` REFUSES that combination, so this record did not come through the "
                         f"tool -- it was written by hand and the refusal was bypassed rather than "
                         f"answered. A deferral owes what you believe it is worth, because that is "
                         f"what lets a later round rank the lead against what it is doing instead; "
                         f"the most expensive single miss measured was exactly this",
                         [ref], severity="high")
    return result


def check_knob_probe_reproducible(root, facts, out):
    """15. The same knob, on the same kernel, classified two different ways.

    A knob is INERT or LIVE as a property of the kernel, not of the directory the probe ran in. Two
    `knobs.resolved.json` under one root disagreeing about one knob therefore means the INSTRUMENT
    moved, and the sweep space of at least one of them is wrong.

    Measured twice on one campaign: `BT`/`BK`/`Hg` read LIVE from the captain's
    `exp_plain/` and INERT ("artifact byte-identical") from the worker's `dir_*/exp/` on the same
    kernel; and `BLOCK`/`GROUP`/`NG` read LIVE in one directory and INERT in another. The cause was a
    global IR fallback that handed every value the same directory when the per-value dirs came up
    empty -- so the disagreement is the only externally visible symptom, and nothing was reading it.
    """
    result = {"files": 0, "conflicts": []}
    # Keyed on the kernel FILE, with the source sha kept alongside. Keying on the sha alone hides the
    # case entirely: the body is edited between the captain's probe and the worker's, so the two
    # records never share a key and a contradiction becomes invisible. Keying on the file alone
    # over-reports: after a real body change a knob's verdict is allowed to move. So both -- same sha
    # is a contradiction (high), different sha is a re-read of a changed body worth a look (low).
    by_knob = {}
    for path in _walk(root, "**/knobs.resolved.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        result["files"] += 1
        kfile = str(doc.get("kernel") or "?")
        sha = str(doc.get("kernel_sha256") or "")
        for name, rec in (doc.get("knobs") or {}).items():
            if not isinstance(rec, dict):
                continue
            v = str(rec.get("verdict") or "").strip().upper()
            if not v:
                continue
            by_knob.setdefault((kfile, name), []).append((v, sha, _rel(root, path)))
    for (kfile, name), rows in sorted(by_knob.items()):
        # LIVE vs LIVE? is the same answer at two confidence levels (the second is "no oracle ran"),
        # so it is not a conflict. UNPROVEN vs anything IS one: it means the probe read the artifact
        # in one place and could not in another.
        def _cls(v):
            return "LIVE" if v.startswith("LIVE") else v
        classes = {_cls(v) for v, _s, _r in rows}
        if len(classes) < 2:
            continue
        same_body = len({s for _v, s, _r in rows}) == 1
        refs = sorted({r for _v, _s, r in rows})
        where = " and ".join(f"{_cls(v)} in {r}" for v, _s, r in sorted(rows, key=lambda x: x[2]))
        result["conflicts"].append(f"{name}: {sorted(classes)} same_body={same_body}")
        if same_body:
            _finding(out, "knob_probe_reproducible", "verdict_conflict",
                     f"knob {name!r} on {kfile} is classified {where} -- at the SAME source sha. A "
                     f"knob is live or inert as a property of the kernel, so at one body there is "
                     f"only one right answer: the instrument moved between the two probes and at "
                     f"least one sweep space is wrong. The usual cause is the probe recovering its "
                     f"IR from a directory it does not own -- check {{IR_OUT}} is threaded through "
                     f"--compile-cmd in both", refs, severity="high")
        else:
            _finding(out, "knob_probe_reproducible", "verdict_moved_with_the_body",
                     f"knob {name!r} on {kfile} is classified {where}, at DIFFERENT source shas. A "
                     f"body edit may legitimately move a verdict, so this is not a contradiction -- "
                     f"but INERT-vs-LIVE is a large move for one edit, and the probe reads INERT "
                     f"whenever it recovers no artifact to compare. Worth confirming the later "
                     f"record's per-value artifacts are real before treating the axis as dead",
                     refs, severity="low")
    return result


def check_measurement_arbitration(root, facts, out):
    """16. A broker was up and none of the readings held a window on it.

    `exclusive_window` per reading and `exclusive_window_rate` in the aggregate answer the one
    question a lease cannot answer for itself: did the lease land on the pool that owns THIS run's
    lanes. A rate of 0 with a broker running is the signature of the wrong socket -- measured once,
    a GPU-0 fleet adopted a broker serving lanes 6,7, every acquire came back "no such
    lane(s)", and the whole sweep ran unarbitrated while its log read "Broker ready".

    `lane_mismatch` is the same fact stated by the broker itself, and it is unambiguous: the daemon
    we are talking to owns none of our cards, so it is not our pool.
    """
    result = {"envelopes": 0, "unarbitrated": []}
    for path in _walk(root, "**/plain_best_config.json"):
        doc = _load_json(path)
        rb = (doc or {}).get("resident_backend")
        if not isinstance(rb, dict):
            continue
        win = rb.get("windows") if isinstance(rb.get("windows"), dict) else {}
        if not win.get("readings"):
            continue
        result["envelopes"] += 1
        ref = _rel(root, path)
        rate = win.get("exclusive_window_rate")
        if win.get("lane_mismatch"):
            result["unarbitrated"].append(ref)
            _finding(out, "measurement_arbitration", "lane_mismatch",
                     f"{win['lane_mismatch']} of {win['readings']} readings were refused with 'no "
                     f"such lane': the broker these leases went to owns none of this run's lanes, so "
                     f"it is not this run's pool and those readings were NOT arbitrated. Point "
                     f".geak_gpu_sock / GEAK_GPU_BROKER_SOCK at the broker that owns them",
                     [ref], severity="high")
        elif rate == 0:
            result["unarbitrated"].append(ref)
            _finding(out, "measurement_arbitration", "no_exclusive_window",
                     f"a resident backend took {win['readings']} readings and held an exclusive "
                     f"window on NONE of them. Residency parks VRAM on the card between readings "
                     f"without holding a file lock, so a sibling's timed benchmark can land on the "
                     f"same card -- these numbers are not shown to be uncontended",
                     [ref], severity="high")
        elif isinstance(rate, (int, float)) and rate < 1.0:
            _finding(out, "measurement_arbitration", "partial_exclusive_window",
                     f"{round(rate * 100, 1)}% of {win['readings']} readings held an exclusive "
                     f"window; the rest shared their lane with whatever else the box was running",
                     [ref], severity="low")
    return result


# ------------------------------------------------------------------ waivers

# WHAT A WAIVER IS FOR, and what it replaces. Measured on one campaign: four kernels closed,
# `--strict` exited 1 on three of them (6, 2 and 1 high findings), and all three were accepted with a
# paragraph in `final_report` saying the findings had been investigated and were tool false
# positives. Two of those dispositions were right. None of them was checkable: the paragraph named
# no finding, pointed at no file, and nothing downstream could tell a disposed finding from an
# ignored one.
#
# So the escape stays -- a wrong high finding must be arguable, and one of them WAS wrong (see
# `_same_objective`) -- but it becomes a record: which finding, why, and where the evidence is. That
# is the whole difference. `--strict` then fails only on the UNWAIVED ones, and a waiver that matches
# nothing is reported rather than silently satisfied.
WAIVER_FIELDS = ("check", "kind", "why", "evidence_ref")
UNWAIVABLE_KINDS = frozenset({
    "canonical_entry_missing", "canonical_entry_invalid", "final_report_missing",
    "comparator_missing", "profile_missing", "resweep_missing", "arm_gate_missing",
    "identity_mismatch", "forbidden_deep_branch", "unauthorized_resweep",
    # These are not disputed observations: waiving them would turn an incomplete search or
    # untrusted denominator into a successful close by prose alone.
    # A declined wave priced against a block nobody measured is an incomplete search with a number
    # attached. Waiving it would let the prose stand as the evidence.
    "declination_price_unmeasured",
    "comparator_not_pinned", "deferred_never_closed", "arm_roster_incomplete",
    "roster_undeclared", "path_coverage_missing", "anchor_unverified",
    # An artifact its writer never produced cannot be argued into having been produced. The waiver
    # exists so a WRONG finding can be contested with evidence, and the evidence that would settle
    # this one is the artifact itself re-emitted through its tool -- which RESOLVES the finding
    # rather than waiving it. Left waivable, the cheapest route past every vocabulary in the pack
    # would be to hand-write the file and waive the one check that noticed.
    "not_tool_produced",
})


def load_waivers(path):
    """Waivers from a bare list, a `{"waivers": [...]}` file, or a `final_report.json` carrying them.

    Three shapes because the report is where they belong (a reader opening the report should see what
    was waived) and a separate file is what a record owner writes first."""
    doc = _load_json(path)
    if isinstance(doc, list):
        return [w for w in doc if isinstance(w, dict)]
    if isinstance(doc, dict):
        return [w for w in (doc.get("waivers") or []) if isinstance(w, dict)]
    return []


def apply_waivers(findings, waivers):
    """`(unwaived_high, waived_high, unused_waivers)`.

    Matching is on (check, kind) and never on the detail text: the detail carries measured numbers
    that move between runs, so keying on it would make every waiver single-use in a way that looks
    like the waiver working. A waiver with no `why` or no `evidence_ref` does not match anything --
    it is the paragraph again, in a field."""
    highs = [f for f in findings if f.get("severity") == "high"]
    used, waived, unwaived = set(), [], []
    for f in highs:
        if f.get("kind") in UNWAIVABLE_KINDS:
            unwaived.append(f)
            continue
        hit = None
        for i, w in enumerate(waivers):
            if str(w.get("check")) != f["check"] or str(w.get("kind")) != f["kind"]:
                continue
            if not str(w.get("why") or "").strip() or not str(w.get("evidence_ref") or "").strip():
                continue
            hit = i
            break
        if hit is None:
            unwaived.append(f)
        else:
            used.add(hit)
            waived.append(dict(f, _waiver=waivers[hit]))
    unused = [w for i, w in enumerate(waivers) if i not in used]
    return unwaived, waived, unused


def _canonical_kind(finding):
    """Map canonical-contract failures to non-waivable audit categories."""
    code, field = finding.get("code"), finding.get("field", "")
    if code in UNWAIVABLE_KINDS:
        return code
    if field.endswith(("comparator_ref", "baseline_ref")):
        return "comparator_missing"
    if field.endswith("profile_ref"):
        return "profile_missing"
    if code == "identity_mismatch":
        return "identity_mismatch"
    return "canonical_entry_invalid"


def check_sweep_lifecycle(root, facts, out):
    """Reconcile typed sweep requests/results at the canonical close boundary.

    Legacy ``resweep_*`` documents remain accepted by their existing validators.  They are not,
    however, evidence for a new typed sweep lifecycle: the purpose and immutable attempt lineage
    did not exist in that format and cannot be reconstructed from a filename.
    """
    requests, results, result_by_path = {}, {}, {}
    for path in _walk(root, "**/sweep_request.json"):
        doc = _load_json(path)
        ref = _rel(root, path)
        if not isinstance(doc, dict) or doc.get("schema") != SWEEP_REQUEST_SCHEMA:
            _finding(out, "sweep_lifecycle", "sweep_request_invalid",
                     "typed sweep request is unreadable or uses the wrong schema", [ref],
                     severity="high")
            continue
        for finding in validate_sweep_request(doc):
            _finding(out, "sweep_lifecycle", "sweep_request_invalid", finding["detail"], [ref],
                     severity="high")
        requests[os.path.realpath(path)] = (path, doc)

    for path in _walk(root, "**/sweep_result.json"):
        doc = _load_json(path)
        ref = _rel(root, path)
        if not isinstance(doc, dict) or doc.get("schema") != SWEEP_RESULT_SCHEMA:
            _finding(out, "sweep_lifecycle", "sweep_result_invalid",
                     "typed sweep result is unreadable or uses the wrong schema", [ref],
                     severity="high")
            continue
        for finding in validate_sweep_result(doc):
            _finding(out, "sweep_lifecycle", "sweep_result_invalid", finding["detail"], [ref],
                     severity="high")
        request_path = _resolve_ref(root, os.path.dirname(path), _ref_path(doc.get("request_ref")))
        if not request_path or os.path.realpath(request_path) not in requests:
            _finding(out, "sweep_lifecycle", "sweep_request_missing",
                     "sweep result does not resolve to a typed sweep request", [ref], severity="high")
            continue
        request = requests[os.path.realpath(request_path)][1]
        for finding in validate_sweep_pair(request, doc):
            _finding(out, "sweep_lifecycle", "immutable_attempt_lineage",
                     finding["detail"], [_rel(root, request_path), ref], severity="high")
        results[os.path.realpath(request_path)] = (path, doc)
        result_by_path[os.path.realpath(path)] = (path, doc)

    for real_request, (path, request) in requests.items():
        if real_request not in results:
            kind = "handoff_unresolved" if request.get("purpose") == "handoff" else "sweep_unresolved"
            _finding(out, "sweep_lifecycle", kind,
                     f"typed {request.get('purpose')!r} sweep request has no captain/direct-owner "
                     "result; requests are not execution evidence", [_rel(root, path)], severity="high")

    # An arm claiming an own sweep must cite the typed arm-local execution.  This is evaluated only
    # under the canonical gate so historical arm roll-ups still remain readable as legacy evidence.
    arm_local = 0
    for path in _walk(root, ARM_GLOB):
        arm = _load_json(path)
        if not isinstance(arm, dict) or arm.get("schema") != ARM_RESULT_SCHEMA:
            continue
        basis = arm.get("config_basis")
        if not isinstance(basis, dict) or basis.get("mode") != "own_sweep":
            continue
        arm_local += 1
        ref = _ref_path(basis.get("sweep_ref"))
        resolved = _resolve_ref(root, os.path.dirname(path), ref)
        item = result_by_path.get(os.path.realpath(resolved)) if resolved else None
        if item is None:
            _finding(out, "sweep_lifecycle", "arm_local_sweep_missing",
                     "canonical structural arm declares own_sweep without a resolved typed "
                     "arm_local sweep result", [_rel(root, path)], severity="high")
        elif item[1].get("purpose") != "arm_local":
            _finding(out, "sweep_lifecycle", "arm_local_purpose_mismatch",
                     "an arm own_sweep must resolve a purpose=arm_local sweep result",
                     [_rel(root, path), _rel(root, item[0])], severity="high")

    post_merge = 0
    for path in _walk(root, "**/branch_converge.json"):
        converge = _load_json(path)
        if not isinstance(converge, dict) or converge.get("schema") != BRANCH_CONVERGE_SCHEMA:
            continue
        if not converge.get("winner"):
            continue
        post_merge += 1
        sweep = converge.get("post_merge_resweep")
        ref = _ref_path(sweep.get("result_ref")) if isinstance(sweep, dict) else None
        resolved = _resolve_ref(root, os.path.dirname(path), ref)
        item = result_by_path.get(os.path.realpath(resolved)) if resolved else None
        if item is None:
            _finding(out, "sweep_lifecycle", "post_merge_sweep_missing",
                     "a canonical branch winner requires a typed purpose=post_merge sweep result; "
                     "legacy resweep_result is compatibility evidence only", [_rel(root, path)],
                     severity="high")
        elif item[1].get("purpose") != "post_merge":
            _finding(out, "sweep_lifecycle", "post_merge_purpose_mismatch",
                     "the branch winner's result must have purpose=post_merge",
                     [_rel(root, path), _rel(root, item[0])], severity="high")
    return {"requests": len(requests), "results": len(results), "arm_local": arm_local,
            "post_merge": post_merge, "handoff": sum(1 for _, doc in requests.values()
                                                     if doc.get("purpose") == "handoff")}


def check_canonical_finalization(root, facts, out):
    """Require a canonical entry and its matching final report in strict close mode."""
    canonical_paths = _walk(root, "**/canonical_record.json")
    reports = _walk(root, "**/final_report.json")
    state_paths = _walk(root, "**/run_state.json")
    if not canonical_paths:
        _finding(out, "canonical_finalization", "canonical_entry_missing",
                 "strict finalization requires canonical_record.json", severity="high")
    if not reports:
        _finding(out, "canonical_finalization", "final_report_missing",
                 "strict finalization requires final_report.json", severity="high")
    canonical_docs = []
    for path in canonical_paths:
        doc = _load_json(path)
        if not isinstance(doc, dict) or doc.get("schema") != RUN_DECISION_SCHEMA:
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "canonical record is unreadable or uses the wrong schema", [_rel(root, path)],
                     severity="high")
            continue
        canonical_docs.append((path, doc))
        for finding in validate_run_decision(doc):
            _finding(out, "canonical_finalization", _canonical_kind(finding), finding["detail"],
                     [_rel(root, path)], severity="high")
    state_docs = []
    try:
        from run_state import validate_state
    except ImportError:
        validate_state = None
    for path in state_paths:
        doc = _load_json(path)
        if not isinstance(doc, dict) or validate_state is None:
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "run state is unreadable or its validator is unavailable", [_rel(root, path)],
                     severity="high")
            continue
        for finding in validate_state(doc):
            _finding(out, "canonical_finalization", "canonical_entry_invalid", finding["detail"],
                     [_rel(root, path)], severity="high")
        state_docs.append((path, doc))
    for path in reports:
        doc = _load_json(path)
        if not isinstance(doc, dict) or doc.get("schema") != FINAL_REPORT_SCHEMA:
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "strict finalization requires the canonical final-report schema",
                     [_rel(root, path)], severity="high")
            continue
        integrity = doc.get("integrity") or {}
        for finding in integrity.get("findings") or []:
            _finding(out, "canonical_finalization", _canonical_kind(finding),
                     finding.get("detail", finding.get("code", "projection integrity failure")),
                     [_rel(root, path)], severity="high")
        if integrity.get("legacy_unverified"):
            _finding(out, "canonical_finalization", "legacy_unverified",
                     "legacy paths remain typed but unverified and cannot close a strict run",
                     [_rel(root, path)], severity="high")
        if integrity.get("late_reconstruction"):
            _finding(out, "canonical_finalization", "late_reconstruction",
                     "captain-time receipt reconstruction is evidence but not process-time provenance",
                     [_rel(root, path)], severity="high")
        for finding in validate_final_report(doc, captain=any(
                k in doc for k in ("champion_ref", "winning_stage", "served_range"))):
            _finding(out, "canonical_finalization", _canonical_kind(finding), finding["detail"],
                     [_rel(root, path)], severity="high")
        matches = [(cp, cd) for cp, cd in canonical_docs
                   if cd.get("run_id") == doc.get("run_id")]
        if len(matches) != 1:
            _finding(out, "canonical_finalization", "canonical_entry_missing",
                     "report run_id must resolve to exactly one canonical record", [_rel(root, path)],
                     severity="high")
            continue
        canonical_path, canonical = matches[0]
        report_canonical_ref = _ref_path(doc.get("canonical_ref"))
        resolved_report_ref = _resolve_ref(root, os.path.dirname(path), report_canonical_ref)
        if not resolved_report_ref or os.path.realpath(resolved_report_ref) != os.path.realpath(canonical_path):
            _finding(out, "canonical_finalization", "canonical_entry_missing",
                     "report canonical_ref does not resolve to its matching canonical record",
                     [_rel(root, canonical_path), _rel(root, path)], severity="high")
        if canonical.get("generation") != doc.get("generation"):
            _finding(out, "canonical_finalization", "identity_mismatch",
                     "canonical and report generations differ", [_rel(root, canonical_path), _rel(root, path)],
                     severity="high")
        if not same_identity(canonical.get("measurement"), doc.get("measurement")):
            _finding(out, "canonical_finalization", "identity_mismatch",
                     "canonical and report measurement identities differ",
                     [_rel(root, canonical_path), _rel(root, path)], severity="high")
        states = [(sp, sd) for sp, sd in state_docs if sd.get("run_id") == doc.get("run_id")]
        if len(states) != 1:
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "report run_id must resolve to exactly one run_state record",
                     [_rel(root, path)], severity="high")
            continue
        state_path, state = states[0]
        if state.get("generation") != doc.get("generation"):
            _finding(out, "canonical_finalization", "identity_mismatch",
                     "run state and report generations differ",
                     [_rel(root, state_path), _rel(root, path)], severity="high")
        if state.get("state") not in ("finalizing", "closed"):
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "strict close requires run_state=finalizing or closed",
                     [_rel(root, state_path)], severity="high")
        if any(item.get("status") == "open" for item in state.get("obligations", [])
               if isinstance(item, dict)):
            _finding(out, "canonical_finalization", "canonical_entry_invalid",
                     "strict close cannot retain open run-state obligations",
                     [_rel(root, state_path)], severity="high")
        report_ref = _ref_path((canonical.get("final") or {}).get("report_ref"))
        resolved_final_ref = _resolve_ref(root, os.path.dirname(canonical_path), report_ref)
        if not resolved_final_ref or os.path.realpath(resolved_final_ref) != os.path.realpath(path):
            _finding(out, "canonical_finalization", "final_report_missing",
                     "canonical final.report_ref does not resolve to this final report",
                     [_rel(root, canonical_path), _rel(root, path)], severity="high")
    return {"canonical_records": len(canonical_paths), "reports": len(reports),
            "run_states": len(state_paths)}


def check_journal_lifecycle(root, facts, out):
    """Close only declared measurement/recall obligations; never inspect optimization choices."""
    try:
        import round_record
    except ImportError as exc:
        _finding(out, "journal_lifecycle", "journal_validator_missing",
                 f"journal lifecycle validator is unavailable: {exc}", severity="high")
        return {"applicable": True, "error": str(exc)}
    ledger = os.path.join(root, round_record.DEFAULT_JOURNAL)
    measurements = os.path.join(root, round_record.DEFAULT_MEASUREMENT_EVENTS)
    recalls = os.path.join(root, round_record.DEFAULT_RECALL_EVENTS)
    applicable = any(os.path.isfile(path) for path in (ledger, measurements, recalls))
    if not applicable:
        return {"applicable": False, "reason": "no canonical journal lifecycle artifacts declared"}
    result = round_record.lifecycle_check(ledger, measurements, recalls)
    for finding in result["findings"]:
        refs = [finding["ref"]] if finding.get("ref") else []
        _finding(out, "journal_lifecycle", finding["kind"], finding["detail"], refs,
                 severity="high")
    # AN EMPTY LEDGER IS NOT AN ABSENT ONE, and `lifecycle_check` answers a different question about
    # it: with no rows there are no unsettled obligations, so it returns clean. Measured on one
    # campaign: `optimization_journal.jsonl` at zero bytes on two of eight runs, one of them the
    # only clean acceptance, and both passed here in silence. The file existing is an affirmative
    # act -- the round writer creates it -- so at a close, zero rows means every round's lever,
    # prediction, reading and keep/revert went somewhere this pack cannot read. That is the record a
    # later round recalls from and a reviewer reconstructs the search from, and it is the one thing a
    # rerun cannot recover.
    if os.path.isfile(ledger) and not result["journal_rows"] and _walk(root, "**/final_report.json"):
        _finding(out, "journal_lifecycle", "journal_empty_at_close",
                 f"{round_record.DEFAULT_JOURNAL} exists and holds zero rounds at a closed run. "
                 f"`lifecycle_check` reads it for unsettled obligations and an empty ledger has "
                 f"none, so this closes clean on every other reading -- but the per-round history "
                 f"is gone: which lever each round tried, the prediction it wrote before measuring, "
                 f"the reading that decided keep or revert. `recall` has nothing to return, so the "
                 f"next round can re-run a lever already ruled out and read it as new evidence. "
                 f"Append the rounds through the round writer as they happen; a private log under "
                 f"`.direction/` or a decision log in markdown is not read by this or any check",
                 [round_record.DEFAULT_JOURNAL], severity="high")
    return {
        "applicable": True,
        "journal_rows": result["journal_rows"],
        "pending_measurements": result["pending_measurements"],
        "pending_recall_receipts": result["pending_recall_receipts"],
    }


# ------------------------------------------------------------------ driver

CHECKS = (check_entry_gate, check_arm_roster, check_arm_count, check_arm_config_basis,
          check_post_merge_resweep, check_sweep_point_accounting, check_evidence_floor,
          check_preflight_closure, check_claim_citations, check_tool_reach,
          check_duplicate_readings, check_served_regressions, check_served_split,
          check_ceiling_provenance, check_stale_lever_verdicts,
          check_branch_waves, check_climb_depth, check_dial_c,
          check_comparator_trust, check_axis_audit_closed,
          check_artifact_admission, check_census_provenance,
          check_knob_probe_reproducible, check_measurement_arbitration,
          check_role_policy, check_journal_lifecycle)


def audit(root, facts, verified_by="self", band=0.02, depth_ratio=4.0, count_tolerance=0.10,
          require_canonical=False):
    findings: list = []
    summary = {}
    # Every tolerance a check compares against is a caller's parameter, never a constant baked into
    # the check: what counts as noise, as a wide depth gap, or as a reconcilable count is a property
    # of the box and the run, and a check that decided it privately could not be argued with.
    tunables = {check_duplicate_readings: {"band": band},
                check_arm_config_basis: {"depth_ratio": depth_ratio},
                check_sweep_point_accounting: {"tolerance": count_tolerance}}
    checks = CHECKS + ((check_canonical_finalization, check_sweep_lifecycle)
                       if require_canonical else ())
    for fn in checks:
        name = fn.__name__.replace("check_", "")
        kwargs = tunables.get(fn, {})
        try:
            summary[name] = fn(root, facts, findings, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # A check that cannot run reports that it could not run. Dropping it would let a
            # crashed check read as a clean one, which is the single worst thing an audit can do.
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
            _finding(findings, name, "check_failed", f"this check did not run: {exc}")
    # Stable order so a second party's run can be diffed against the first's line by line.
    findings.sort(key=lambda f: (f["check"], f["kind"], f["detail"], tuple(f["refs"])))
    return {
        "schema": "close_audit",
        "_what_this_is": "facts only -- no verdict, no pass/fail. Every entry is arithmetic or "
                         "file order. What any of it should mean for accepting this run is the "
                         "reader's call.",
        "work_root": os.path.abspath(root),
        "pack": {k: facts.get(k) for k in ("dsl", "vendor", "round_contract", "breadth_enabled")},
        "verified_by": verified_by,
        "findings": findings,
        "finding_count": len(findings),
        # Split out because a reader deciding whether to accept a run needs the high count first:
        # those are the facts that contradict something the run already concluded. Both are still
        # facts and neither is a verdict.
        "high_severity_count": sum(1 for f in findings if f.get("severity") == "high"),
        # R9. Reported separately from the high count even though blocked findings are also high,
        # because the two ask different questions of the reader: a high finding says a banked
        # conclusion is contradicted, a blocked check says nobody knows either way. A reader that
        # sees only the first number reads an unrun check as a clean one.
        "blocked_checks": sorted(k for k, v in summary.items()
                                 if isinstance(v, dict) and v.get("blocked")),
        "summary": summary,
    }


def _selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="close_audit_selftest_")
    try:
        facts = {"dsl": "x", "vendor": "v", "round_contract": "tool_first", "breadth_enabled": True,
                 "artifacts": {"produces": {"plain_champion.json": "scripts/e.py"}, "consumes": {}},
                 "evidence_tools": {"ab_interleave": "scripts/ab_bench.py", "budget": None},
                 "tool_gaps": {"budget": "no budget tool ships here"}}

        def w(rel, obj, text=None):
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(text if text is not None else json.dumps(obj))
            return p

        # a clean-ish tree: two arms, both complete, collected after both
        w("exp/branch/plain/arm_1/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.1, "lever": "a"})
        w("exp/branch/plain/arm_2/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.9, "lever": "b"})
        w("branch_request.json", {"arms": ["a", "b"]})
        champ = w("plain_champion.json", {"x": 1})
        report = w("final_report.json", {"served_range": [{"shape": "s1", "ms": 1.0,
                                                           "vs_default": 1.2}]})
        w("decision_log.md", None, "## r1\nran scripts/ab_bench.py and kept it\n")
        os.utime(champ, (1_700_000_000, 1_700_000_000))
        os.utime(report, (1_700_000_100, 1_700_000_100))
        for a in ("arm_1", "arm_2"):
            os.utime(os.path.join(root, "exp/branch/plain", a, "arm_result.json"),
                     (1_699_999_000, 1_699_999_000))
        os.utime(os.path.join(root, "decision_log.md"), (1_700_000_100, 1_700_000_100))

        r = audit(root, facts)
        kinds = {f["kind"] for f in r["findings"]}
        assert "written_after_collection" not in kinds, r["findings"]
        assert "count_mismatch" not in kinds, r["findings"]
        assert r["summary"]["tool_reach"]["reached"] == ["ab_interleave"], r["summary"]["tool_reach"]
        assert "tool_gap" in kinds, kinds            # the unfilled role is STATED, not silent
        assert r["verified_by"] == "self"

        # an arm that lands after the roll-up is the finding this whole tool exists for
        late = os.path.join(root, "exp/branch/plain/arm_2/arm_result.json")
        os.utime(late, (1_700_000_500, 1_700_000_500))
        r2 = audit(root, facts)
        assert any(f["kind"] == "written_after_collection" for f in r2["findings"]), r2["findings"]

        # a lost arm, and an arm that claims to have finished with nothing to show
        os.remove(late)
        w("exp/branch/plain/arm_3/arm_result.json", {"completion": "complete", "lever": "c"})
        r3 = audit(root, facts)
        k3 = {f["kind"] for f in r3["findings"]}
        assert "complete_without_number" in k3 and "headline_absent" in k3, k3

        # two readings of one shape, past the band
        w("exp/second_reading.json", {"per_case": [{"shape": "s1", "ms": 1.5}]})
        r4 = audit(root, facts)
        assert any(f["kind"] == "spread_over_band" for f in r4["findings"]), r4["findings"]
        # ...and a band wide enough to cover them makes it a non-finding, as a band should
        assert not any(f["kind"] == "spread_over_band"
                       for f in audit(root, facts, band=1.0)["findings"])

        # a served row that went backwards, with and without a reason attached
        w("final_report.json", {"served_range": [{"shape": "s2", "ms": 2.0, "vs_default": 0.8}]})
        assert any(f["kind"] == "regressed_unexplained" for f in audit(root, facts)["findings"])
        w("final_report.json", {"served_range": [{"shape": "s2", "ms": 2.0, "vs_default": 0.8,
                                                  "note": "dispatch-guarded below this size"}]})
        assert any(f["kind"] == "regressed_with_note" for f in audit(root, facts)["findings"])

        # a search-ending claim, with and without something to check it against
        w("notes.md", None, "the layout is inexpressible here\n")
        assert any(f["kind"] == "no_citation_marker" for f in audit(root, facts)["findings"])
        w("notes.md", None, "the layout is inexpressible here (see ir/dump.ttgir:41)\n")
        assert not any(f["kind"] == "no_citation_marker" for f in audit(root, facts)["findings"])

        # guidance the run was HANDED is not guidance the run wrote: a copied pack asserts ceilings
        # on every other page and names every tool, and reading it back would answer both the claim
        # check and the reach check out of the documentation instead of out of the run
        w("pack/SKILL.md", None, "the tile is inexpressible on this target\nrun scripts/ab_bench.py\n")
        rc = audit(root, facts)
        assert not any(f["kind"] == "no_citation_marker" for f in rc["findings"]), rc["findings"]
        w("geak_pack/skill.md", None,
          "the tile is inexpressible on this target\nrun scripts/ab_bench.py\n")
        rc = audit(root, facts)
        assert not any(f["kind"] == "no_citation_marker" for f in rc["findings"]), rc["findings"]
        copied = sorted(s for s in _pack_own_stems() if s.endswith(".md"))
        if copied:
            w(os.path.join("tools_skill", copied[0]), None, "already optimal, nothing left\n")
            assert not any(f["kind"] == "no_citation_marker"
                           for f in audit(root, facts)["findings"])

        # ---- an arm's config basis --------------------------------------------------------
        # a structural arm that says nothing about which config it was measured at
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.0, "lever": "d",
           "direction_class": "structural", "verdict": "keep"})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "config_basis_absent" in kcb, kcb

        # ...and an UNDECLARED basis under a kill stays low: the checks below ask whether a
        # declared basis supports the verdict, and a pre-contract record cannot answer that. One
        # plain line, not a high finding on every arm ever written.
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.9, "lever": "d",
           "direction_class": "structural", "verdict": "not_supported"})
        fu = [f for f in audit(root, facts)["findings"] if f["check"] == "arm_config_basis"]
        assert {f["kind"] for f in fu} == {"config_basis_absent", "kill_basis_undeclared"}, fu
        assert all(f["severity"] == "low" for f in fu), fu

        # ...but a roster that RANKED arms and carried a winner, with no arm declaring a basis,
        # is the ranking itself being unsupported. Four owners in one campaign closed this way.
        w("exp/branch/plain/arm_5/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.2, "lever": "e",
           "direction_class": "structural", "verdict": "keep"})
        ranked = [f for f in audit(root, facts)["findings"]
                  if f["kind"] == "winner_ranked_without_declared_basis"]
        assert len(ranked) == 1 and ranked[0]["severity"] == "high", ranked
        # WHOSE declaration settles it. A LOSING arm declaring a basis used to silence the whole
        # group, which made a field on an arm that lost the cheapest way past a high finding about
        # the ranking. It says nothing about the configuration the winner's number came from.
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.9, "lever": "d",
           "direction_class": "structural", "verdict": "not_supported",
           "config_basis": {"mode": "inherited", "inherit_rationale": "same tile, same pin"}})
        ranked = [f for f in audit(root, facts)["findings"]
                  if f["kind"] == "winner_ranked_without_declared_basis"]
        assert len(ranked) == 1 and ranked[0]["severity"] == "high", ranked
        assert "except on the arm the close depends on" in ranked[0]["detail"], ranked[0]
        assert ranked[0]["refs"] == ["exp/branch/plain/arm_5"], ranked[0]["refs"]
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.9, "lever": "d",
           "direction_class": "structural", "verdict": "not_supported"})
        # The WINNER declaring its basis is what makes the ranking answerable again.
        w("exp/branch/plain/arm_5/sweep.json", {"schema": "sweep", "points": 24})
        w("exp/branch/plain/arm_5/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.2, "lever": "e",
           "direction_class": "structural", "verdict": "keep",
           "config_basis": {"mode": "own_sweep", "sweep_ref": "exp/branch/plain/arm_5/sweep.json",
                            "n_measured": 24}})
        assert not [f for f in audit(root, facts)["findings"]
                    if f["kind"] == "winner_ranked_without_declared_basis"]
        # The later assertions read this same tree, so the probe arm goes back out.
        shutil.rmtree(os.path.join(root, "exp", "branch", "plain", "arm_5"))

        # a kill on a slower number alone, at a config tuned for a different structure: two
        # separate high findings, because either one alone would already unmake the kill
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.9, "lever": "d",
           "direction_class": "structural", "verdict": "negative",
           "config_basis": {"mode": "inherited", "inherit_rationale": "parent pin"}})
        fcb = audit(root, facts)["findings"]
        kcb = {f["kind"] for f in fcb}
        assert {"kill_without_mechanism", "kill_at_borrowed_config"} <= kcb, kcb
        assert all(f["severity"] == "high" for f in fcb
                   if f["kind"] in ("kill_without_mechanism", "kill_at_borrowed_config"))

        # ...the same borrowed config under a WIN is not a finding: it biases the arm slow, so a
        # win measured through it is a conservative lower bound and needs no remedy
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.2, "lever": "d",
           "direction_class": "structural", "verdict": "win",
           "config_basis": {"mode": "inherited", "inherit_rationale": "parent pin"}})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert not ({"kill_without_mechanism", "kill_at_borrowed_config"} & kcb), kcb

        # a shallow kill WITH a mechanism is a clean kill -- depth is not what makes it one
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.96, "lever": "d",
           "direction_class": "structural", "verdict": "negative",
           "config_basis": {"mode": "own_sweep", "sweep_ref": "sweep.json", "n_measured": 11},
           "kill_evidence": {"scaling_evidence": "per-CU 3.58 -> 2.34 TFLOP/s across the ladder"}})
        w("exp/branch/plain/arm_4/sweep.json", {"measured": [{"cfg": 1}]})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "kill_without_mechanism" not in kcb and "sweep_ref_missing" not in kcb, kcb

        # a sweep_ref that points at the trunk's record is not this arm having swept
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 0.96, "lever": "d",
           "direction_class": "structural", "verdict": "negative",
           "config_basis": {"mode": "own_sweep", "sweep_ref": "plain_champion.json",
                            "n_measured": 11},
           "kill_evidence": {"isa_ref": "asm/diff.s:41"}})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "sweep_ref_outside_arm" in kcb, kcb

        # a closed axis whose premise the arm itself moved, with and without the recheck
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.05, "lever": "d",
           "direction_class": "structural", "verdict": "keep",
           "config_basis": {"mode": "closed_axis", "closure_ref": "exp/r3.json",
                            "closure_n_measured": 99,
                            "resource_profile_match": {"vgpr": [219, 145]}}})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "closure_premise_moved" in kcb, kcb
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.05, "lever": "d",
           "direction_class": "structural", "verdict": "keep",
           "config_basis": {"mode": "closed_axis", "closure_ref": "exp/r3.json",
                            "closure_n_measured": 99,
                            "resource_profile_match": {"vgpr": [219, 145],
                                                       "recheck_ref": "exp/r22.json"}}})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "closure_premise_moved" not in kcb, kcb

        # an arm ranked on a different question than the comparator it is quoted against
        w("plain_best_config.json", {"objective": "geomean(c2,c32,c64)", "trust_level": "pinned"})
        w("exp/branch/plain/arm_4/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.05, "lever": "d",
           "direction_class": "structural", "verdict": "keep",
           "config_basis": {"mode": "own_sweep", "sweep_ref": "sweep.json", "n_measured": 20,
                            "objective": "c2 only"}})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "objective_mismatch" in kcb, kcb

        # a comparator the sweep itself would not call tuned discounts every arm under it
        w("plain_best_config.json", {"objective": "geomean(c2,c32,c64)",
                                     "trust_level": "provisional", "partially_sampled": True})
        kcb = {f["kind"] for f in audit(root, facts)["findings"]}
        assert "comparator_provisional" in kcb, kcb
        w("plain_best_config.json", {"objective": "geomean(c2,c32,c64)", "trust_level": "pinned"})

        # a knob arm is not asked for a structure's sweep
        w("exp/branch/plain/arm_5/arm_result.json",
          {"completion": "complete", "best_geomean_vs_plain": 1.0, "lever": "e",
           "direction_class": "knob", "verdict": "negative"})
        assert not any(f["refs"] == ["exp/branch/plain/arm_5/arm_result.json"]
                       and f["check"] == "arm_config_basis"
                       for f in audit(root, facts)["findings"])
        os.remove(os.path.join(root, "exp/branch/plain/arm_5/arm_result.json"))

        # a winner carried into the trunk without a re-sweep or a reason for not doing one
        w("t0_merge.json", {"winner": "arm_1"})
        assert any(f["kind"] == "resweep_unrecorded" for f in audit(root, facts)["findings"])
        w("t0_merge.json", {"winner": "arm_1", "post_merge_resweep_ref": "exp/r12_sweep.json"})
        assert not any(f["kind"] == "resweep_unrecorded" for f in audit(root, facts)["findings"])
        w("t0_merge.json", {"winner": "anchor"})       # nothing merged, nothing owed

        # a claimed sweep size the tables on disk do not add up to
        w("exp/sweep_big.json", {"measured": [{"c": i} for i in range(40)]})
        w("final_report.json", {"served_range": [{"shape": "s2", "ms": 2.0, "vs_default": 0.8,
                                                  "note": "guarded"}],
                                "config_sweep_points_evaluated": 400})
        assert any(f["kind"] == "count_mismatch" and f["check"] == "sweep_point_accounting"
                   for f in audit(root, facts)["findings"])
        # ...and like every other tolerance here, a caller who says the gap is expected gets to say
        # so, rather than arguing with a number the check picked privately
        assert not any(f["check"] == "sweep_point_accounting"
                       for f in audit(root, facts, count_tolerance=20.0)["findings"])
        w("final_report.json", {"served_range": [{"shape": "s2", "ms": 2.0, "vs_default": 0.8,
                                                  "note": "guarded"}],
                                "config_sweep_points_evaluated": 41})
        assert not any(f["check"] == "sweep_point_accounting"
                       for f in audit(root, facts)["findings"])

        # ---- the evidence floor ------------------------------------------------------------
        # A tree with no SOL layer at all is not accused of losing one.
        assert audit(root, facts)["summary"]["evidence_floor"]["applicable"] is False
        assert audit(root, facts)["blocked_checks"] == []

        # R9. The SAME empty tree, for a pack that DECLARES capture.json, is BLOCKED rather than
        # not-applicable -- the two returned the identical value before and a missing artifact
        # therefore disabled its own check. Blocked is high and is listed separately, because
        # "nobody knows" and "a banked conclusion is contradicted" are different questions.
        declaring = dict(facts, artifacts=dict(facts.get("artifacts") or {},
                                               produces={"capture.json": "scripts/capture.sh"}))
        res_b = audit(root, declaring)
        assert res_b["summary"]["evidence_floor"] == {
            "applicable": True, "blocked": True,
            "reason": "every conclusion in this run was reached with group C dark, and nothing "
                      "downstream can tell that from a run that had no use for it",
            "declared_producer": "scripts/capture.sh"}, res_b["summary"]["evidence_floor"]
        assert res_b["blocked_checks"] == ["evidence_floor"], res_b["blocked_checks"]
        assert res_b["high_severity_count"] >= 1
        assert [f for f in res_b["findings"] if f["kind"] == "check_blocked"], res_b["findings"]

        # A capture that says the collection crashed, with nothing in the roll-up admitting it.
        # High, because this capture DECLARES sol_state: it is on the contract and answerable to it.
        w("exp/cap_r1/rc/rc_analyze.txt", None,
          "Traceback (most recent call last):\nPermissionError: [Errno 13] pmc_dispatch_info.csv\n")
        w("exp/cap_r1/capture.json",
          {"att_mfma_eff": "att/mfma_eff.txt", "asm_audit": "ir/asm_audit.txt",
           "rc_analyze": os.path.join(root, "exp/cap_r1/rc/rc_analyze.txt"),
           "rc_metrics": None, "sol_state": "crashed"})
        fe = [f for f in audit(root, facts)["findings"] if f["check"] == "evidence_floor"]
        assert [f["kind"] for f in fe] == ["sol_crashed_undeclared"], fe
        assert fe[0]["severity"] == "high", fe

        # The same crash, declared in the slot a reader is allowed to look at, is not a finding.
        # The check is about a gap nobody can see, not about the gap.
        rep = _load_json(os.path.join(root, "final_report.json"))
        w("final_report.json", dict(rep, caveats=["rocprof-compute analyze failed on this box; "
                                                  "every round below was read without SOL"]))
        assert not [f for f in audit(root, facts)["findings"]
                    if f["kind"] == "sol_crashed_undeclared"]
        w("final_report.json", rep)

        # A pre-contract capture (no sol_state) is read off its artifacts all the same, but stays
        # low: it never agreed to the field, so re-auditing an archive cannot bury a real finding.
        w("exp/cap_r1/capture.json",
          {"att_mfma_eff": "att/mfma_eff.txt", "asm_audit": "ir/asm_audit.txt",
           "rc_analyze": os.path.join(root, "exp/cap_r1/rc/rc_analyze.txt"), "rc_metrics": None})
        fe = [f for f in audit(root, facts)["findings"] if f["kind"] == "sol_crashed_undeclared"]
        assert fe and fe[0]["severity"] == "low", fe

        # The expensive case: the report IS on disk, under the name a by-hand recovery lands on,
        # and no metrics were built from it. Reported separately -- only the last hop is missing.
        w("exp/cap_r1/rc/rc_analyze_fixed.txt", None,
          "System Speed-of-Light\nDependency-Wait 47\n")
        ke = {f["kind"] for f in audit(root, facts)["findings"] if f["check"] == "evidence_floor"}
        assert ke == {"sol_collected_but_unparsed"}, ke

        # ...and once the parse lands, the layer is simply live.
        w("exp/cap_r1/rc/rc_metrics.json", {"sol": {"mfma_pct": 5.1}})
        w("exp/cap_r1/capture.json",
          {"att_mfma_eff": "att/mfma_eff.txt", "asm_audit": "ir/asm_audit.txt",
           "rc_analyze": os.path.join(root, "exp/cap_r1/rc/rc_analyze.txt"),
           "rc_metrics": os.path.join(root, "exp/cap_r1/rc/rc_metrics.json"),
           "sol_state": "parsed"})
        assert not [f for f in audit(root, facts)["findings"] if f["check"] == "evidence_floor"]

        # ---- the entry gate's own handoff --------------------------------------------------
        # A run with no entry gate on disk is not asked whether it closed one.
        assert audit(root, facts)["summary"]["preflight_closure"]["applicable"] is False

        # A gate that deferred a check to a named owner, and a run that never answered it. The
        # capture above already carries sol_state, so use a different check to isolate this one.
        w("preflight.json", {"pass": True, "caveats": [
            {"check": "KNOBS", "status": "deferred", "msg": "not run here",
             "owner": "SWEEP phase knob_probe.py"}]})
        fp = [f for f in audit(root, facts)["findings"] if f["check"] == "preflight_closure"]
        assert [f["kind"] for f in fp] == ["deferred_never_closed"], fp
        # This root published a swept comparator, so an unanswered [KNOBS] is not a missing nicety:
        # the space that comparator was ranked over was never established.
        assert fp[0]["severity"] == "high", fp
        assert "PUBLISHED a swept comparator" in fp[0]["detail"], fp

        # ...and the same debt in a run that never swept stays low. The axis set it did not
        # establish is one it also never used, so nothing downstream inherits the gap.
        nosweep = tempfile.mkdtemp(prefix="close_audit_nosweep_")
        with open(os.path.join(nosweep, "preflight.json"), "w") as f:
            json.dump({"pass": True, "caveats": [
                {"check": "KNOBS", "status": "deferred", "msg": "not run here",
                 "owner": "SWEEP phase knob_probe.py"}]}, f)
        fns = [f for f in audit(nosweep, facts)["findings"] if f["check"] == "preflight_closure"]
        assert [f["severity"] for f in fns] == ["low"], fns

        # Carrying it into the roll-up's caveats closes it. Not as strong as answering it, but it
        # is the difference between a gap a reader can see and one they cannot.
        rep = _load_json(os.path.join(root, "final_report.json"))
        w("final_report.json", dict(rep, caveats=["KNOBS was never probed; the sweep ran on the "
                                                  "manifest's declared knobs alone"]))
        assert not [f for f in audit(root, facts)["findings"]
                    if f["check"] == "preflight_closure"]
        w("final_report.json", rep)

        # EVIDENCE is closed by the owner ANSWERING, whichever way: a capture that states a
        # definite sol_state is the worker having looked, and that is what was asked of it.
        w("preflight.json", {"pass": True, "caveats": [
            {"check": "EVIDENCE", "status": "deferred", "msg": "no capture in preflight",
             "owner": "worker round-1 capture.sh"}]})
        assert not [f for f in audit(root, facts)["findings"]
                    if f["check"] == "preflight_closure"]
        os.remove(os.path.join(root, "preflight.json"))

        # a verdict older than the pinned config
        stale = w("exp/round_1/verdict.json", {"verdict": "negative", "lever": "old_lever"})
        os.utime(stale, (1_699_000_000, 1_699_000_000))
        assert any(f["kind"] == "predates_pin" for f in audit(root, facts)["findings"])

        # the audit is a function of the run, not of its own last answer: the fleet's re-run has to
        # land on the same findings the report quotes, or every re-check reads as a mismatch
        first = audit(root, facts)
        w(SELF_OUTPUT, first)
        assert audit(root, facts)["findings"] == first["findings"]

        # a pack that does not fan out is not asked about arms it never had
        noarms = audit(root, dict(facts, breadth_enabled=False))
        assert noarms["summary"]["arm_count"]["expected"] == 0

        # ---- the roster the run declared, and whether it can be read ------------------------
        # branch_plan.py's own artifact is the contract's way of declaring a roster, so the audit
        # must read it; before it did, a run that followed the contract still audited as having
        # declared nothing.
        rr = tempfile.mkdtemp(prefix="close_audit_roster_")

        def _wr(rel, doc):
            p = os.path.join(rr, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(doc, f)

        _wr("arms/a1/arm_result.json", {"schema": "kernel_opt.arm_result/1"})
        _wr("arms/a2/arm_result.json", {"schema": "kernel_opt.arm_result/1"})
        ac = [f for f in audit(rr, facts)["findings"] if f["check"] == "arm_count"]
        assert [f["kind"] for f in ac] == ["roster_undeclared"], ac
        assert ac[0]["severity"] == "high", ("arms with no declared roster make breadth "
                                             "completeness unanswerable", ac)
        # a validated plan declares it, and the counts agree -> silence
        _wr("branch_plan.json", {"schema": "kernel_opt.branch_plan/1", "fanout": 2,
                                 "candidate_roster": [{"id": "a1"}, {"id": "a2"}]})
        assert not [f for f in audit(rr, facts)["findings"] if f["check"] == "arm_count"]
        # ...and a plan that declares more than came back is the arm-loss this check exists for
        _wr("branch_plan.json", {"schema": "kernel_opt.branch_plan/1", "fanout": 4,
                                 "candidate_roster": [{"id": f"a{i}"} for i in range(4)]})
        ac2 = [f for f in audit(rr, facts)["findings"] if f["check"] == "arm_count"]
        assert [(f["kind"], f["severity"]) for f in ac2] == [("count_mismatch", "high")], ac2

        # an entry gate that refused, and unverified guidance nobody named
        gated = dict(facts, entry_gate={"tool": "scripts/preflight_x.sh",
                                        "artifact": "preflight_x.json"},
                     authoring_gap="unverified_no_hw")
        rg = audit(root, gated)
        kg = {f["kind"] for f in rg["findings"]}
        assert "record_absent" in kg and "unverified_reliance_unnamed" in kg, kg
        w("preflight/preflight_x.json", {"ok": False, "gaps": ["no compiler"]})
        assert any(f["kind"] == "refused" for f in audit(root, gated)["findings"])
        w("preflight/preflight_x.json", {"ok": True, "gaps": []})
        assert not any(f["kind"] in ("refused", "record_absent")
                       for f in audit(root, gated)["findings"])

        # no pack_facts at all: undeterminable, and it says so rather than inventing an answer
        blind = audit(root, {})
        assert any(f["kind"] == "pack_facts_absent" for f in blind["findings"])

        # ---- an objective is a SENTENCE, and string equality raised three wrong HIGH findings ---
        # Every string below is verbatim from one measured campaign, where all three arms had copied
        # the trunk's objective and expanded it into prose.
        _trunk_obj = "geomean(c2,c32,c64)"
        for _same in ("geomean(c2,c32,c64)",
                      "unweighted geomean of c2/c32/c64 latency (the parent's objective)",
                      "unweighted geomean of execution_time_ms over c2, c32, c64 "
                      "(the parent's objective)",
                      "unweighted geomean of execution_time_ms over the three served cases "
                      "c2/c32/c64 (M=2048/32768/65536, N=5120, K=2880, bf16) -- the parent's "
                      "objective",
                      "equal-weight geomean over the served shapes c2/c32/c64 "
                      "(the parent's objective)",
                      "geomean(c2,c32,c64) ms at the production grid cdiv(M,128)*cdiv(N,128)"):
            assert _same_objective(_same, _trunk_obj), _same
        # ...and a genuinely different ranking is still different: the case SET and the AGGREGATION
        # are the two things that make it one
        for _diff in ("geomean(c2,c32)", "geomean(c2,c32,c64,c128)", "sum of c2,c32,c64",
                      "max(c2,c32,c64)", "harmonic mean of c2,c32,c64"):
            assert not _same_objective(_diff, _trunk_obj), _diff
        # with no case set on either side the de-phrased text is the comparison -- better than the raw
        # string, and no worse than what it replaced
        assert _same_objective("total wall time (ms)", "total wall time")
        assert not _same_objective("total wall time", "peak memory")
        # a shape count is not a case set: "three shapes" must not parse as case 3
        assert _objective_signature("geomean over three shapes")[1] == frozenset(), \
            _objective_signature("geomean over three shapes")

        # ---- 11-16. the six checks that one measured campaign needed and did not have ---------
        # Each one is asserted against the shape the measured run actually wrote, because that shape
        # is what the check has to catch and a synthetic worst case would not have found it.
        nb = tempfile.mkdtemp(prefix="close_audit_new_")

        def wn(rel, obj):
            p = os.path.join(nb, rel)
            os.makedirs(os.path.dirname(p) or nb, exist_ok=True)
            with open(p, "w") as f:
                json.dump(obj, f)
            return p

        # 11. BRANCH: one wave, mode left OPEN, run closed. Three of four measured kernels.
        wn("final_report.json", {"result": "win"})
        wn("dir_01/search_mode_ledger.json",
           {"modes": {"branch": {"state": "open", "entered_rounds": [6]}}})
        k11 = {f["kind"] for f in audit(nb, facts)["findings"]}
        assert "single_wave_close" in k11, k11
        # ...and mid-run it is SILENT: an open BRANCH may still re-enter
        os.unlink(os.path.join(nb, "final_report.json"))
        assert "single_wave_close" not in {f["kind"] for f in audit(nb, facts)["findings"]}
        wn("final_report.json", {"result": "win"})
        # a PRICED declination is the exemption, and prose is not a price
        wn("dir_01/worker_result.json",
           {"status": "done", "branch": {"waves_declined": "nothing looked promising"}})
        assert "single_wave_close" in {f["kind"] for f in audit(nb, facts)["findings"]}
        wn("dir_01/worker_result.json",
           {"status": "done",
            "branch": {"waves_declined": "all remaining structure bounded at 2.2% geomean"}})
        k11b = {f["kind"] for f in audit(nb, facts)["findings"]}
        assert "single_wave_close" not in k11b and "single_wave_declined_with_a_price" in k11b, k11b
        # A NUMBER IS NOT A PRICE when the number is attributed to a block nobody measured. This
        # sentence has a digit and would have bought the low-severity downgrade.
        wn("dir_01/worker_result.json",
           {"status": "done", "branch": {"waves_declined":
            "11.6 of the remaining points sit outside the editable surface"}})
        k11c = {f["kind"] for f in audit(nb, facts)["findings"]}
        assert "declination_price_unmeasured" in k11c, k11c
        assert "single_wave_declined_with_a_price" not in k11c, k11c
        assert "declination_price_unmeasured" in UNWAIVABLE_KINDS, \
            "an unmeasured price may not be written off in prose"
        # ...and the same claim WITH a measured bound is the exemption again
        wn("dir_01/worker_result.json",
           {"status": "done", "branch": {"waves_declined": {
               "statement": "the remaining cost is outside the editable surface",
               "pct": 11.6, "is_upper_bound": True,
               "method": "ablation removed the block, paired against a same-window control",
               "evidence_ref": {"kind": "measurement", "artifact": "exp/ablation.json"}}}})
        k11d = {f["kind"] for f in audit(nb, facts)["findings"]}
        assert "single_wave_declined_with_a_price" in k11d, k11d
        assert "declination_price_unmeasured" not in k11d, k11d
        wn("dir_01/worker_result.json",
           {"status": "done",
            "branch": {"waves_declined": "all remaining structure bounded at 2.2% geomean"}})
        # two waves is the band and is not a finding at all
        wn("dir_01/search_mode_ledger.json",
           {"modes": {"branch": {"state": "exhausted", "entered_rounds": [6, 31]}}})
        assert not [f for f in audit(nb, facts)["findings"] if f["check"] == "branch_waves"]
        # no ledger at all, at a closed run, is worse than a single wave: nothing could be checked
        os.unlink(os.path.join(nb, "dir_01/search_mode_ledger.json"))
        assert "no_mode_ledger" in {f["kind"] for f in audit(nb, facts)["findings"]}
        wn("dir_01/search_mode_ledger.json",
           {"modes": {"branch": {"state": "exhausted", "entered_rounds": [6, 31]}}})

        # 11b. CLIMB depth, the counterpart BRANCH width had and this did not. Same tree, and the
        # depth is read off the ledger the width check already reads.
        cd = tempfile.mkdtemp(prefix="close_audit_climb_")

        def wc(rel, obj):
            p = os.path.join(cd, rel)
            os.makedirs(os.path.dirname(p) or cd, exist_ok=True)
            with open(p, "w") as f:
                json.dump(obj, f)
            return p

        def kinds(check="climb_depth"):
            return {f["kind"] for f in audit(cd, facts)["findings"] if f["check"] == check}

        def sev(kind):
            return next(f.get("severity") for f in audit(cd, facts)["findings"]
                        if f["kind"] == kind)

        # Mid-run this is silent, for the same reason the wave check is.
        wc("dir_01/search_mode_ledger.json",
           {"round_links": {"7": "climb", "8": "climb"},
            "modes": {"branch": {"state": "exhausted", "entered_rounds": [1, 4]},
                      "climb": {"state": "exhausted", "entered_rounds": [7]}}})
        assert not kinds(), kinds()
        # Closed, three rounds deep, and NOTHING says what was left: a gap, not a contradiction.
        wc("final_report.json", {"result": "win"})
        assert kinds() == {"climb_depth_unaccounted"}, kinds()
        assert sev("climb_depth_unaccounted") == "low"
        # Recorded but under the threshold is a DIFFERENT low, and the two must not share wording:
        # the budget was consumed, just not by the climb, and telling this reader "nothing records
        # what it had left" sends them looking for a missing field instead of at the BRANCH.
        wc("final_report.json", {"result": "win", "rounds_used": 38, "round_budget": 40})
        assert kinds() == {"shallow_climb_inside_the_budget"}, kinds()
        assert sev("shallow_climb_inside_the_budget") == "low"
        assert "consumed somewhere other than the climb" in next(
            f["detail"] for f in audit(cd, facts)["findings"]
            if f["kind"] == "shallow_climb_inside_the_budget")
        # The measured shape: 38 of 200 rounds spent, climb two deep. This is the one that was
        # passing a strict audit clean on four kernels.
        wc("final_report.json", {"result": "win", "budget": {"rounds_used": 38, "round_budget": 200}})
        assert kinds() == {"shallow_climb_with_budget_left"}, kinds()
        assert sev("shallow_climb_with_budget_left") == "high"
        # Wall clock says the same thing independently -- a run that finalized 1.5h into its 4h and
        # then wrote that the clock was the binding constraint.
        wc("final_report.json", {"result": "win", "budget": {"rounds_used": 195, "round_budget": 200}})
        assert kinds() == {"shallow_climb_inside_the_budget"}, kinds()
        wc("run_state.json",
           {"updated_at": "2026-09-06T17:37:00Z", "state": "closed",
            "deadline": {"started_at": "2026-09-06T16:07:00Z", "time_limit_s": 14400,
                         "deadline_at": "2026-09-06T20:07:00Z", "close_reserve_s": 2400}})
        k = kinds()
        assert k == {"shallow_climb_with_budget_left"}, k
        assert "wall clock unspent" in next(
            f["detail"] for f in audit(cd, facts)["findings"]
            if f["kind"] == "shallow_climb_with_budget_left")
        # A run that genuinely used its clock is not accused of stopping early on that ground. This
        # is also the BRANCH-heavy shape: 195/200 rounds and 97% of the clock consumed, two climb
        # rounds. Not a HIGH -- the budget really is gone -- but it does not read as clean either.
        wc("run_state.json",
           {"updated_at": "2026-09-06T20:00:00Z", "state": "closed",
            "deadline": {"started_at": "2026-09-06T16:07:00Z", "time_limit_s": 14400,
                         "deadline_at": "2026-09-06T20:07:00Z", "close_reserve_s": 2400}})
        assert kinds() == {"shallow_climb_inside_the_budget"}, kinds()
        # A PRICED declination is the exemption, and it clears BOTH the shallow finding and the
        # unaccounted one -- same bar as a declined wave, and prose is still not a price.
        wc("final_report.json", {"result": "win", "budget": {"rounds_used": 38, "round_budget": 200}})
        wc("dir_01/worker_result.json",
           {"status": "done", "climb_declined": "the line looked done after two rounds"})
        assert kinds() == {"shallow_climb_with_budget_left"}, kinds()
        wc("dir_01/worker_result.json",
           {"status": "done", "climb_declined": {
               "statement": "the remaining prize on this line is bounded",
               "remaining_prize_pct": 1.4, "is_upper_bound": True,
               "method": "paired A/B at the pin against a same-window control",
               "evidence_ref": {"kind": "probe", "artifact": "exp/climb_probe.json"}}})
        assert kinds() == {"shallow_climb_declined_with_a_price"}, kinds()
        assert sev("shallow_climb_declined_with_a_price") == "low"
        # At the floor the check is silent when a PER-ROUND record carries the depth -- the mode
        # ledger or the round journal. Depth is still reconciled to the largest of them, because
        # `--round` is optional and the smallest would read an incomplete record as shallow.
        wc("dir_01/worker_result.json", {"status": "done"})
        with open(os.path.join(cd, "dir_01", "optimization_journal.jsonl"), "w") as f:
            for i in range(1, 11):
                f.write(json.dumps({"round": i, "mode": "climb"}) + "\n")
        assert not kinds(), kinds()
        os.unlink(os.path.join(cd, "dir_01", "optimization_journal.jsonl"))
        assert kinds() == {"shallow_climb_with_budget_left"}, kinds()
        # The bundle's own `climb.rounds` is NOT one of those records: it is one integer a writer
        # asserted, and before this it satisfied the floor by itself. One measured campaign's only clean
        # acceptance closed exactly here -- declared 86, ledger witnessing one round, journal empty
        # -- so the run that climbed and the run that typed the number were indistinguishable.
        wc("dir_01/plain_champion.json", {"climb": {"rounds": 12, "kept": 2}})
        assert kinds() == {"climb_depth_unwitnessed"}, kinds()
        assert sev("climb_depth_unwitnessed") == "high"
        assert "witness 2" in next(f["detail"] for f in audit(cd, facts)["findings"]
                                   if f["kind"] == "climb_depth_unwitnessed")
        # ...and it goes quiet the moment the rounds it claims are actually on the record.
        with open(os.path.join(cd, "dir_01", "optimization_journal.jsonl"), "w") as f:
            for i in range(1, 13):
                f.write(json.dumps({"round": i, "mode": "climb"}) + "\n")
        assert not kinds(), kinds()
        os.unlink(os.path.join(cd, "dir_01", "optimization_journal.jsonl"))
        # A run that did not CHOOSE to stop is not answerable for its depth. Blocked, refused and
        # user-deferred closes obviously have budget left, and a finding saying so would sit on top
        # of the blocker the reader came for.
        os.unlink(os.path.join(cd, "dir_01", "plain_champion.json"))
        wc("final_report.json", {"result": "win", "budget": {"rounds_used": 38, "round_budget": 200}})
        assert kinds() == {"shallow_climb_with_budget_left"}, kinds()
        for stopped in ({"status": "blocked_environment"}, {"status": "deferred_needs_user"},
                        {"result": "refused_entry_gate"}):
            wc("final_report.json", dict(stopped, budget={"rounds_used": 38, "round_budget": 200}))
            assert not kinds(), (stopped, kinds())
        shutil.rmtree(cd, ignore_errors=True)

        # 11b-2. An EMPTY ledger at a close, which is not the same thing as an absent one.
        # `lifecycle_check` reads the ledger for unsettled obligations and zero rows carry none, so
        # this closed clean on every reading -- measured at zero bytes on two of eight runs.
        jl = tempfile.mkdtemp(prefix="close_audit_journal_")

        def jkinds():
            return {f["kind"] for f in audit(jl, facts)["findings"]
                    if f["check"] == "journal_lifecycle"}

        open(os.path.join(jl, "optimization_journal.jsonl"), "w").close()
        assert not jkinds(), jkinds()          # mid-run an empty ledger has said nothing yet
        with open(os.path.join(jl, "final_report.json"), "w") as f:
            json.dump({"result": "win"}, f)
        assert jkinds() == {"journal_empty_at_close"}, jkinds()
        assert next(f["severity"] for f in audit(jl, facts)["findings"]
                    if f["kind"] == "journal_empty_at_close") == "high"
        with open(os.path.join(jl, "optimization_journal.jsonl"), "w") as f:
            f.write(json.dumps({"round": 1, "mode": "climb"}) + "\n")
        assert "journal_empty_at_close" not in jkinds(), jkinds()
        shutil.rmtree(jl, ignore_errors=True)

        # 11c. Dial C. Triggered by the CLAIM, never by the round -- a per-round requirement would
        # cost the run the one currency the climb floor just bought it.
        dc = tempfile.mkdtemp(prefix="close_audit_dialc_")

        def wd(rel, text):
            p = os.path.join(dc, rel)
            os.makedirs(os.path.dirname(p) or dc, exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
            return p

        def dkinds():
            return {f["kind"] for f in audit(dc, facts)["findings"] if f["check"] == "dial_c"}

        # No memory claim, no C: silent. Never reading the counters is fine when nothing rests on
        # them, and a finding here would be the tax this check exists to avoid.
        wd("report.md", "# close\n\nthe loop is issue limited and the tile shape is what moved it\n")
        assert not dkinds(), dkinds()
        # A memory bound asserted in passing, with C dark: LOW. The close does not rest on it, but
        # the levers in those rounds were chosen against a level nobody measured.
        wd("report.md", "# close\n\nround 4 read as memory bound so we widened the vector load\n")
        assert dkinds() == {"memory_bound_asserted_without_group_c"}, dkinds()
        # The same bound ENDING the search: HIGH. This is the measured shape -- every close argued
        # about a bound nothing had counted.
        wd("report.md", "# close\n\nthe kernel is memory bound at 94% of SOL and is at the ceiling\n")
        assert dkinds() == {"search_ended_on_an_uncounted_bound"}, dkinds()
        # Any group-C read anywhere clears it, in the ledger's vocabulary...
        with open(os.path.join(dc, "rounds.jsonl"), "w") as f:
            f.write(json.dumps({"round": 4, "evidence_layers": ["pmc_tcc"]}) + "\n")
        assert not dkinds(), dkinds()
        os.unlink(os.path.join(dc, "rounds.jsonl"))
        assert dkinds() == {"search_ended_on_an_uncounted_bound"}, dkinds()
        # ...or as the counters themselves in a captured tool log, because a run that read the right
        # thing must not be accused for having written it somewhere other than the ledger.
        wd("pmc.log", "TCC_HIT_sum 1.2e9  TCC_MISS_sum 3.1e8  FETCH_SIZE 41 GB\n")
        assert not dkinds(), dkinds()
        os.unlink(os.path.join(dc, "pmc.log"))
        # The NVIDIA spelling counts too: this file ships byte-identical in every pack.
        wd("ncu.txt", "dram__bytes.sum 41230000000\n")
        assert not dkinds(), dkinds()
        shutil.rmtree(dc, ignore_errors=True)

        # 12-13. the comparator's own trust ladder, and the axis surface behind it
        wn("exp_plain/plain_best_config.json",
           {"trust_level": "provisional",
            "pin_blockers": [{"code": "axis-audit"}, {"code": "read-cost"}],
            "axis_audit": {"unaudited": ["NUM_KSPLIT", "cache_modifier"], "unproven": ["BT"],
                           "needs_edit": ["SPLIT_K"], "swept": ["BLOCK_M"]}})
        f13 = audit(nb, facts)["findings"]
        k13 = {(f["check"], f["kind"]) for f in f13}
        assert ("comparator_trust", "comparator_not_pinned") in k13, sorted(k13)
        assert ("axis_audit_closed", "unaudited_axes_in_the_comparator") in k13, sorted(k13)
        assert ("axis_audit_closed", "unproven_axes_in_the_comparator") in k13, sorted(k13)
        assert ("axis_audit_closed", "needs_edit_axes") in k13, sorted(k13)
        _want_high = {("comparator_trust", "comparator_not_pinned"),
                      ("axis_audit_closed", "unaudited_axes_in_the_comparator"),
                      ("axis_audit_closed", "unproven_axes_in_the_comparator")}
        assert all(f.get("severity") == "high" for f in f13
                   if (f["check"], f["kind"]) in _want_high), \
            [f for f in f13 if (f["check"], f["kind"]) in _want_high]
        # `arm_config_basis` already owns the name `comparator_provisional` for a DIFFERENT fact, and
        # two checks answering to one kind is how a reader disposes of the wrong one
        assert ("comparator_trust", "comparator_provisional") not in k13
        # `needs_edit` SEVERITY IS A FUNCTION OF THE DECLARED EDITABLE SURFACE. With no declaration
        # under the root the conservative reading stands: a caveat that travels with the number.
        assert all(f.get("severity") == "low" for f in f13
                   if f["kind"] == "needs_edit_axes"), \
            [f for f in f13 if f["kind"] == "needs_edit_axes"]
        # Declare a surface that covers the source and the same axis is work the contract allowed
        # and the search did not take.
        wn("resolve.json", {"editable_surface": {"declared_by": "resolve_manifest",
                                                 "paths": ["k.py"]}})
        f13b = audit(nb, facts)["findings"]
        assert all(f.get("severity") == "high" for f in f13b
                   if f["kind"] == "needs_edit_axes"), \
            [f for f in f13b if f["kind"] == "needs_edit_axes"]
        os.unlink(os.path.join(nb, "resolve.json"))

        # A comparator with NO axis_audit block used to skip this check entirely, which is the same
        # absent-record-reads-as-clean defect the check exists to catch, one level up.
        wn("exp_plain/plain_best_config.json", {"trust_level": "pinned", "pin_blockers": []})
        f13c = audit(nb, facts)["findings"]
        assert ("axis_audit_closed", "comparator_without_axis_audit") in {
            (f["check"], f["kind"]) for f in f13c}, sorted(
                (f["check"], f["kind"]) for f in f13c)
        assert all(f.get("severity") == "high" for f in f13c
                   if f["kind"] == "comparator_without_axis_audit")

        # a PINNED comparator with a closed surface is what this looks like when it is right
        wn("exp_plain/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "axis_audit": {"unaudited": [], "unproven": [], "needs_edit": [], "swept": ["BLOCK_M"]}})
        assert not [f for f in audit(nb, facts)["findings"]
                    if f["check"] in ("comparator_trust", "axis_audit_closed")]

        # 14. CENSUS-0: absent entirely, then present with an unpriced deferral
        assert "no_census" in {f["kind"] for f in audit(nb, facts)["findings"]}
        # a pack that runs no census must NOT be told it skipped one
        assert "no_census" not in {f["kind"] for f in
                                   audit(nb, dict(facts, breadth_enabled=False))["findings"]}
        # The fixtures carry the writer's schema because that is what the writer stamps; a census
        # without one is a different defect and 14a owns it.
        wn("dir_01/structure_census.json",
           {"schema": "structure_census/3",
            "questions": {"device_fill": {"disposition": "fanned", "why": "8 WGs on 304 CUs"},
                          "input_layout": {"disposition": "deferred", "why": "later",
                                           "lower_bound_pct": None}}})
        k14 = {f["kind"] for f in audit(nb, facts)["findings"]}
        assert "no_census" not in k14 and "deferral_without_a_price" in k14, k14
        wn("dir_01/structure_census.json",
           {"schema": "structure_census/3",
            "questions": {"input_layout": {"disposition": "deferred", "why": "later",
                                           "lower_bound_pct": 25.0}}})
        assert "deferral_without_a_price" not in {f["kind"] for f in audit(nb, facts)["findings"]}

        # 14a. ADMISSION, asserted against the exact shape one campaign's most expensive miss
        # wrote: the writer's schema replaced by a `kernel_opt.`-prefixed invention, no `questions`,
        # and a candidate carrying a status the vocabulary does not contain.
        def adm():
            return {f["kind"] for f in audit(nb, facts)["findings"]
                    if f["check"] == "artifact_admission"}

        assert not adm(), adm()                    # the conforming census above admits cleanly
        wn("dir_01/structure_census.json",
           {"schema": "kernel_opt.structure_census/1",
            "candidates": [{"id": "S5", "status": "DISPOSED",
                            "disposition_evidence": "outside editable_surface"}]})
        assert adm() == {"not_tool_produced"}, adm()
        # ...and a forged census must not ALSO read as a priced-deferral defect: one forged file is
        # one finding, or the reader cannot tell how many things are actually wrong.
        assert "deferral_without_a_price" not in {f["kind"] for f in audit(nb, facts)["findings"]}
        # Copying the schema string is not evidence the writer ran -- the keys consumers read are.
        wn("dir_01/structure_census.json", {"schema": "structure_census/3", "candidates": []})
        assert adm() == {"not_tool_produced"}, adm()
        assert "questions" in next(f["detail"] for f in audit(nb, facts)["findings"]
                                   if f["check"] == "artifact_admission")
        wn("dir_01/structure_census.json",
           {"schema": "structure_census/3",
            "questions": {"input_layout": {"disposition": "deferred", "why": "later",
                                           "lower_bound_pct": 25.0}}})
        assert not adm(), adm()
        # The plan was forged the same way, and the roster is the key its consumers read.
        wn("dir_01/branch_plan.json", {"schema": "kernel_opt.branch_plan/1",
                                       "candidates_enumerated": 9, "written_by": "hand"})
        assert adm() == {"not_tool_produced"}, adm()
        wn("dir_01/branch_plan.json", {"schema": "branch_plan/1", "candidate_roster": []})
        assert not adm(), adm()
        assert "not_tool_produced" in UNWAIVABLE_KINDS, \
            "a file nothing produced may not be argued into existence"

        # 15. the same knob classified two ways. At ONE body sha that is a contradiction; across a
        # body edit it is a re-read worth looking at, and keying on the sha alone hid BOTH.
        wn("exp_plain/knobs.resolved.json",
           {"kernel": "k.py", "kernel_sha256": "aaa", "knobs": {"BT": {"verdict": "LIVE"}}})
        wn("dir_01/exp/knobs.resolved.json",
           {"kernel": "k.py", "kernel_sha256": "aaa", "knobs": {"BT": {"verdict": "INERT"}}})
        f15 = [f for f in audit(nb, facts)["findings"] if f["check"] == "knob_probe_reproducible"]
        assert [f["kind"] for f in f15] == ["verdict_conflict"], f15
        assert f15[0]["severity"] == "high" and "SAME source sha" in f15[0]["detail"]
        wn("dir_01/exp/knobs.resolved.json",
           {"kernel": "k.py", "kernel_sha256": "bbb", "knobs": {"BT": {"verdict": "INERT"}}})
        f15b = [f for f in audit(nb, facts)["findings"] if f["check"] == "knob_probe_reproducible"]
        assert [f["kind"] for f in f15b] == ["verdict_moved_with_the_body"], f15b
        assert f15b[0]["severity"] == "low"
        # LIVE vs LIVE? is one answer at two confidence levels, not a conflict
        wn("dir_01/exp/knobs.resolved.json",
           {"kernel": "k.py", "kernel_sha256": "aaa", "knobs": {"BT": {"verdict": "LIVE?"}}})
        assert not [f for f in audit(nb, facts)["findings"]
                    if f["check"] == "knob_probe_reproducible"]

        # 16. a resident backend that never held a window is unarbitrated, whatever the log said
        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "resident_backend": {"width": 2, "windows": {"readings": 60, "exclusive": 0,
                                                         "exclusive_window_rate": 0.0,
                                                         "lane_mismatch": 0}}})
        f16 = [f for f in audit(nb, facts)["findings"] if f["check"] == "measurement_arbitration"]
        assert [f["kind"] for f in f16] == ["no_exclusive_window"], f16
        assert f16[0]["severity"] == "high"
        # the broker's OWN answer -- "no such lane" -- is the unambiguous form of the same fact
        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "resident_backend": {"windows": {"readings": 60, "exclusive": 0,
                                             "exclusive_window_rate": 0.0, "lane_mismatch": 60}}})
        assert [f["kind"] for f in audit(nb, facts)["findings"]
                if f["check"] == "measurement_arbitration"] == ["lane_mismatch"]
        # every reading arbitrated is the clean case
        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "resident_backend": {"windows": {"readings": 60, "exclusive": 60,
                                             "exclusive_window_rate": 1.0, "lane_mismatch": 0}}})
        assert not [f for f in audit(nb, facts)["findings"]
                    if f["check"] == "measurement_arbitration"]

        # 17. the served cases, and whether they agreed. Both answers are reportable: a split says
        # the pinned comparator is not every case's best; no_split is the evidence that a
        # per-bucket track would have bought nothing here.
        def _served_kinds():
            return [f["kind"] for f in audit(nb, facts)["findings"] if f["check"] == "served_split"]

        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "per_case": {"cases": ["c2", "c32"], "split": True, "split_cases": ["c2"],
                         "winner_per_case": {"c2": {"ms": 1.0, "config": {"BLOCK_M": 32}},
                                             "c32": {"ms": 3.4, "config": {"BLOCK_M": 128}}}}})
        assert _served_kinds() == ["split"], _served_kinds()
        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "per_case": {"cases": ["c2", "c32"], "split": False, "split_cases": [],
                         "winner_per_case": {"c2": {"ms": 2.0, "config": {"BLOCK_M": 128}},
                                             "c32": {"ms": 3.4, "config": {"BLOCK_M": 128}}}}})
        assert _served_kinds() == ["no_split"], _served_kinds()
        # readings kept but never labelled -> the shapes are on disk and unreadable
        wn("exp2/plain_best_config.json",
           {"trust_level": "pinned", "pin_blockers": [],
            "per_case": {"cases": ["c2", "c32"], "winner_per_case": {}, "split": False}})
        assert _served_kinds() == ["unlabelled"], _served_kinds()
        # a sweep from before this contract says nothing, and is not made to
        wn("exp2/plain_best_config.json", {"trust_level": "pinned", "pin_blockers": []})
        assert _served_kinds() == [], _served_kinds()

        # 18. a percent-of-ceiling quoted against a ceiling the budget tool refused to compute.
        cr = tempfile.mkdtemp(prefix="close_audit_ceiling_")

        def _wc(rel, doc):
            p = os.path.join(cr, rel)
            os.makedirs(os.path.dirname(p) or cr, exist_ok=True)
            with open(p, "w") as f:
                json.dump(doc, f)

        _wc("hw_budget.json", {"workload": {"hbm_ceiling_source": "datasheet",
                                            "ceiling_refused": "datasheet peak only"}})
        # Refusing and then quoting nothing is the honest outcome, and the common one.
        _wc("final_report.json", {"status": "closed", "best_ms": 1.0})
        assert not [f for f in audit(cr, facts)["findings"] if f["check"] == "ceiling_provenance"]
        # Quoting one anyway is the finding.
        _wc("final_report.json", {"status": "closed", "notes": "reached 61% of the HBM ceiling"})
        fc = [f for f in audit(cr, facts)["findings"] if f["check"] == "ceiling_provenance"]
        assert [(f["kind"], f["severity"]) for f in fc] == [("quoted_an_unprobed_ceiling", "high")], fc
        # A probed ceiling means the budget never refused, so the check does not apply.
        _wc("hw_budget.json", {"workload": {"hbm_ceiling_source": "measured",
                                            "pct_of_hbm_ceiling": 61.0}})
        assert not [f for f in audit(cr, facts)["findings"] if f["check"] == "ceiling_provenance"]

        # Role boundaries are consumed at close: deep workers may request a resweep, never branch
        # or execute it, and the request must resolve to the actual deep worker result.
        role_root = tempfile.mkdtemp(prefix="close_audit_role_")
        role_measurement = {
            "schema": "kernel_opt.measurement/1", "measurement_id": "m", "value": 1.0,
            "collected_at": "2026-09-01T00:00:00Z",
            "identity": {"shape_set": ["s"], "aggregation": "geomean", "unit": "ratio",
                         "boundary": "kernel", "source": "device_timing", "scope": "device",
                         "sample": {"count": 2, "method": "interleaved"},
                         "baseline_ref": "baseline.json", "comparator_ref": "champion.json"},
        }
        role_worker = {
            "schema": WORKER_RESULT_SCHEMA, "run_id": "r", "generation": 1,
            "producer_role": "deep", "stage": "resweep", "status": "completed",
            "produced_at": "2026-09-01T00:00:00Z", "measurement": role_measurement,
            "work_kinds": ["climb", "resweep_request"], "requests": [],
        }
        def wn_role(rel, doc):
            path = os.path.join(role_root, rel)
            os.makedirs(os.path.dirname(path) or role_root, exist_ok=True)
            with open(path, "w") as handle:
                json.dump(doc, handle)
            return path
        worker_ref = wn_role("worker_result.json", role_worker)
        wn_role("resweep_request.json", {
            "schema": RESWEEP_REQUEST_SCHEMA, "request_id": "rq", "run_id": "r", "generation": 1,
            "requester_role": "deep", "requested_at": "2026-09-01T00:00:00Z",
            "parent_result_ref": worker_ref, "evidence_ref": "profile.json", "state": "requested",
        })
        clean_role_findings = []
        check_role_policy(role_root, facts, clean_role_findings)
        assert not clean_role_findings, clean_role_findings
        plain_worker_ref = wn_role("plain_worker_result.json", dict(
            role_worker, producer_role="direction", stage="converge",
            work_kinds=["climb"]))
        wn_role("captain_resweep_request.json", {
            "schema": RESWEEP_REQUEST_SCHEMA, "request_id": "captain-rq", "run_id": "r",
            "generation": 1, "requester_role": "captain", "requested_at": "2026-09-01T00:00:00Z",
            "parent_result_ref": plain_worker_ref, "evidence_ref": "profile.json", "state": "requested",
        })
        wn_role("direct_owner_resweep_request.json", {
            "schema": RESWEEP_REQUEST_SCHEMA, "request_id": "owner-rq", "run_id": "r",
            "generation": 1, "requester_role": "direct_owner",
            "requested_at": "2026-09-01T00:00:00Z", "parent_result_ref": plain_worker_ref,
            "evidence_ref": "profile.json", "state": "requested",
        })
        plain_role_findings = []
        check_role_policy(role_root, facts, plain_role_findings)
        assert not plain_role_findings, plain_role_findings
        direct_request_ref = os.path.join(role_root, "direct_owner_resweep_request.json")
        wn_role("direct_owner_resweep_result.json", {
            "schema": RESWEEP_RESULT_SCHEMA, "request_ref": direct_request_ref, "run_id": "r",
            "generation": 1, "executor_role": "direct_owner",
            "completed_at": "2026-09-01T00:00:00Z", "measurement": role_measurement,
        })
        consumed_role_findings = []
        check_role_policy(role_root, facts, consumed_role_findings)
        assert not consumed_role_findings, consumed_role_findings
        bad_request_path = wn_role("deep_execution/resweep_request.json", dict(
            _load_json(os.path.join(role_root, "resweep_request.json")),
            request_id="deep-execution", measurement=role_measurement))
        execution_findings = []
        check_role_policy(role_root, facts, execution_findings)
        assert any(f["kind"] == "unauthorized_resweep" for f in execution_findings), \
            execution_findings
        os.unlink(bad_request_path)
        wn_role("branch_request.json", {
            "schema": BRANCH_REQUEST_SCHEMA, "request_id": "b", "run_id": "r", "generation": 1,
            "requester_role": "deep", "requested_at": "2026-09-01T00:00:00Z",
            "candidates": ["census.json"], "evidence_ref": "profile.json",
        })
        role_findings = []
        check_role_policy(role_root, facts, role_findings)
        assert any(f["kind"] == "forbidden_deep_branch" for f in role_findings), role_findings
        # Typed sweeps retain an immutable attempt/lineage from request to the owner-executed
        # result.  Arms may request; only the captain/direct owner result validates here.
        typed_request = {
            "schema": SWEEP_REQUEST_SCHEMA, "request_id": "typed-rq", "run_id": "r",
            "generation": 1, "purpose": "handoff", "requester_role": "arm",
            "requested_at": "2026-09-01T00:00:00Z", "source": "profile.json",
            "context": {"objective": "geomean"}, "parent": "arm_result.json",
            "attempt": {"attempt_id": "a1"},
            "lineage": {"lineage_id": "l1", "root_request_id": "typed-rq"},
            "state": "requested", "work_kind": "sweep", "change_scope": "config",
            "axis_kind": "config", "realization_kind": "request",
        }
        typed_request_path = wn_role("typed/sweep_request.json", typed_request)
        typed_result = {
            "schema": SWEEP_RESULT_SCHEMA, "request_ref": typed_request_path, "run_id": "r",
            "generation": 1, "purpose": "handoff", "executor_role": "captain",
            "completed_at": "2026-09-01T00:00:00Z", "measurement": role_measurement,
            "attempt": {"attempt_id": "a1"},
            "lineage": {"lineage_id": "l1", "root_request_id": "typed-rq"},
            "work_kind": "sweep", "change_scope": "config", "axis_kind": "config",
            "realization_kind": "execution",
        }
        typed_result_path = wn_role("typed/sweep_result.json", typed_result)
        typed_findings = []
        typed_summary = check_sweep_lifecycle(role_root, facts, typed_findings)
        assert not typed_findings and typed_summary["handoff"] == 1, typed_findings
        wn_role("typed/sweep_result.json", dict(typed_result, attempt={"attempt_id": "a2"}))
        typed_findings = []
        check_sweep_lifecycle(role_root, facts, typed_findings)
        assert any(f["kind"] == "immutable_attempt_lineage" for f in typed_findings), typed_findings
        wn_role("typed/sweep_result.json", typed_result)
        wn_role("arms/a/arm_result.json", {
            "schema": ARM_RESULT_SCHEMA, "arm_id": "a", "lever": "split_k",
            "direction_class": "structural", "iters_used": 1, "completion": "complete",
            "verdict": "supported", "gate_stage": "b", "measurement": role_measurement,
            "config_basis": {"mode": "own_sweep", "sweep_ref": "missing_sweep_result.json"},
        })
        wn_role("branch_converge.json", {
            "schema": BRANCH_CONVERGE_SCHEMA, "winner": "a",
            "post_merge_resweep": {"status": "completed", "result_ref": "missing_sweep_result.json"},
        })
        typed_findings = []
        check_sweep_lifecycle(role_root, facts, typed_findings)
        typed_kinds = {f["kind"] for f in typed_findings}
        assert {"arm_local_sweep_missing", "post_merge_sweep_missing"} <= typed_kinds, typed_findings
        shutil.rmtree(role_root, ignore_errors=True)
        # ---- waivers: the escape stays, but it becomes a RECORD ------------------------------
        wn("exp_plain/plain_best_config.json",
           {"trust_level": "provisional", "pin_blockers": [{"code": "read-cost"}],
            "axis_audit": {"unaudited": ["NUM_KSPLIT"], "unproven": [], "needs_edit": [],
                           "swept": ["BLOCK_M"]}})
        _f = audit(nb, facts)["findings"]
        _highs = [f for f in _f if f.get("severity") == "high"]
        assert _highs, "the fixture must produce high findings for the waiver test"
        # Comparator trust is deliberately no longer waivable. Exercise the generic waiver
        # mechanics with a disputed-but-auditable observation instead.
        _waivable = [{"check": "test", "kind": "auditable_dispute", "severity": "high",
                      "detail": "fixture"}]
        # a waiver naming the finding AND its evidence excludes it from the exit code
        _wv = [{"check": f["check"], "kind": f["kind"], "why": "investigated: tool false positive",
                "evidence_ref": "dir_01/exp/r12.json"} for f in _waivable]
        _un, _wd, _unused = apply_waivers(_waivable, _wv)
        assert not _un and len(_wd) == len(_waivable) and not _unused, (_un, _unused)
        # a waiver with no evidence_ref is the paragraph again, in a field: it matches nothing
        _un2, _wd2, _ = apply_waivers(_waivable, [dict(w, evidence_ref="") for w in _wv])
        assert len(_un2) == len(_waivable) and not _wd2, (_un2, _wd2)
        _un3, _wd3, _ = apply_waivers(_waivable, [dict(w, why="   ") for w in _wv])
        assert len(_un3) == len(_waivable) and not _wd3
        # a waiver for a finding that is not there is REPORTED, not silently satisfied
        _un4, _wd4, _unused4 = apply_waivers(
            _waivable, _wv + [{"check": "nope", "kind": "nope", "why": "w", "evidence_ref": "e"}])
        assert not _un4 and [u["check"] for u in _unused4] == ["nope"], _unused4
        # matching is on (check, kind) and NOT on the detail, which carries measured numbers that
        # move between runs -- keying on it makes every waiver single-use in a way that looks fine
        _moved = [dict(f, detail=f["detail"] + " (numbers changed)") for f in _waivable]
        assert not apply_waivers(_moved, _wv)[0]
        # low findings are not waivable and not in scope: they never reached the exit code
        assert all(f.get("severity") == "high" for f in _wd)
        # the three accepted file shapes
        wn("waivers_bare.json", _wv)
        wn("report_with_waivers.json", {"result": "win", "waivers": _wv})
        wn("waivers_wrapped.json", {"waivers": _wv})
        for _p in ("waivers_bare.json", "report_with_waivers.json", "waivers_wrapped.json"):
            assert len(load_waivers(os.path.join(nb, _p))) == len(_wv), _p
        assert load_waivers(os.path.join(nb, "no_such_file.json")) == []
        shutil.rmtree(nb, ignore_errors=True)

        # findings are stably ordered, so an independent re-run diffs line by line
        a1, a2 = audit(root, facts), audit(root, facts)
        assert a1["findings"] == a2["findings"]
        assert audit(root, facts, verified_by="fleet")["verified_by"] == "fleet"
        # Canonical checks are enabled by strict finalization, not retroactively
        # imposed on legacy audit-only runs.
        strict_missing = audit(root, facts, require_canonical=True)
        assert any(f["kind"] == "canonical_entry_missing" for f in strict_missing["findings"])
        unwaived, waived, _unused = apply_waivers(
            [{"check": "canonical_finalization", "kind": "resweep_missing", "severity": "high"}],
            [{"check": "canonical_finalization", "kind": "resweep_missing",
              "why": "not applicable", "evidence_ref": "none"}])
        assert len(unwaived) == 1 and not waived
        unwaived, waived, _unused = apply_waivers(
            [{"check": "role_policy", "kind": "unauthorized_resweep", "severity": "high"}],
            [{"check": "role_policy", "kind": "unauthorized_resweep",
              "why": "ignore", "evidence_ref": "none"}])
        assert len(unwaived) == 1 and not waived
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("[close_audit] SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default=".")
    ap.add_argument("--as", dest="verified_by", default="self", choices=("self", "fleet"),
                    help="who is running this. A self-check and an independent check leave the "
                         "same artifacts, so the report has to say which one it was.")
    ap.add_argument("--pack-facts", help="override the stamped pack_facts.json beside this script")
    ap.add_argument("--band", type=float, default=0.02,
                    help="the fraction two readings of one shape may differ by before they are "
                         "reported as a spread (default 0.02). This is a property of the box and "
                         "the harness, not a constant -- set it to the noise floor you measured.")
    ap.add_argument("--depth-ratio", type=float, default=4.0,
                    help="how far apart two sibling arms' sweep depths may be before the "
                         "dispersion is reported alongside their margin (default 4.0). Depth is "
                         "not rigour; this only puts the two numbers next to each other.")
    ap.add_argument("--count-tolerance", type=float, default=0.10,
                    help="the fraction a reported sweep size may differ from the rows on disk "
                         "before the two are reported as describing different sets (default 0.10). "
                         "Loose by design: a report legitimately counts a different subset than "
                         "every envelope under the root, and only a gap wider than that is a fact.")
    ap.add_argument("--json", dest="out_json")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when a high-severity finding or a BLOCKED check is present. This "
                         "does not move the facts/policy line -- it lets the reader that already "
                         "owns the policy state it in an exit code instead of in a count it is "
                         "free to ignore. `kernel-opt-run` Phase 4 passes it; run bare and this "
                         "is a report, exactly as before.")
    ap.add_argument("--waivers", metavar="FILE",
                    help="a JSON file, or a final_report.json carrying `waivers[]`, whose entries "
                         "{check, kind, why, evidence_ref} EXCLUDE matching high findings from the "
                         "--strict exit code. The findings stay in the report either way. This "
                         "exists because the alternative is what already happens: a close that "
                         "exits 1 gets accepted anyway with a paragraph saying the findings were "
                         "tool false positives, and a paragraph is not addressable. A waiver names "
                         "the finding and points at the evidence, so a reader can check it.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not os.path.isdir(a.work_root):
        print(f"[close_audit] no such work root: {a.work_root}")
        return 2

    res = audit(a.work_root, load_pack_facts(a.pack_facts), a.verified_by, a.band, a.depth_ratio,
                a.count_tolerance, require_canonical=a.strict)
    blocked = res["blocked_checks"]
    waivers = load_waivers(a.waivers) if (a.strict and a.waivers) else []
    unwaived, waived, unused = apply_waivers(res["findings"], waivers) if a.strict else (
        [], [], [])
    res["strict_pass"] = bool(a.strict and not unwaived and not blocked)
    res["strict_exit_code"] = 0 if res["strict_pass"] else (1 if a.strict else None)
    print(f"[close_audit] {res['work_root']} ({res['pack'].get('dsl') or 'unknown pack'}, "
          f"{res['pack'].get('round_contract') or 'unknown regime'}) "
          f"verified_by={res['verified_by']} findings={res['finding_count']} "
          f"(high={res['high_severity_count']}, blocked={len(blocked)})")
    for f in res["findings"]:
        refs = f" [{', '.join(f['refs'])}]" if f["refs"] else ""
        sev = f"[{f['severity']}] " if f.get("severity") else ""
        print(f"  {sev}{f['check']}/{f['kind']}: {f['detail']}{refs}")
    if blocked:
        print(f"[close_audit] BLOCKED checks (R9 -- these did not run, which is not a pass): "
              f"{', '.join(blocked)}")
    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
        write_producer_receipt(a.out_json, "audit", tool_path=__file__)
        print(f"[close_audit] wrote {a.out_json}")
    # Bare: ALWAYS 0. This reports; it does not arbitrate, and a gate whose criteria are "facts a
    # reader should weigh" fails the wrong runs. Under --strict the reader has declared its policy
    # and the exit code carries it -- see the module docstring for why that is not the same move.
    if not a.strict:
        return 0
    for w in waived:
        print(f"  WAIVED {w['check']}/{w['kind']}: {w['_waiver'].get('why', '')[:160]} "
              f"[{w['_waiver'].get('evidence_ref', 'NO EVIDENCE REF')}]")
    for u in unused:
        # A waiver matching nothing is not harmless: it is a claim about a finding that is not there,
        # so either the audit moved or the waiver was copied from another run.
        print(f"  [close_audit] waiver for {u.get('check')}/{u.get('kind')} matched NO finding -- "
              f"stale or copied from another run")
    if unwaived or blocked:
        print(f"[close_audit] --strict: refusing the close on {len(unwaived)} unwaived high "
              f"finding(s)" + (f" and {len(blocked)} blocked check(s)" if blocked else "")
              + (f"; {len(waived)} waived" if waived else ""))
        return 1
    if waived:
        print(f"[close_audit] --strict: every high finding is WAIVED ({len(waived)}), each naming "
              f"its evidence. The findings remain in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
