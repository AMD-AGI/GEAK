#!/usr/bin/env python3
"""Warm start and write-back for the e2e serving lane, over the same two planes as the kernel lane.

    e2e_store.py identity --model Qwen3-397B --gfx gfx950 --framework vllm \
        --framework-version 0.26.0 --precision mxfp8 --tp 8 --isl 1024 --osl 1024 --conc 64
    e2e_store.py resolve  ... --plane remote --refs-dir REFS --cache-dir CACHE
    e2e_store.py write    ... --plane both --store DIR --result run.json --apply

`resolve` answers the question the e2e Director asks at Setup — "has anyone already tuned this
deployment, and what did they land on" — and `write` is what makes the next run's answer non-empty.
Addresses come from `kb.identity.e2e_canonical_ids`, never from string formatting here, because a
reader and a writer that disagree by one segment do not raise: the run just cold starts.

WHAT AN E2E RECORD IS FOR, and why it is not a kernel record with different fields. A kernel entry
offers a patch to apply. An e2e entry mostly offers a CONFIG — env flags, server args, a launch
script — plus a list of kernels that turned out to be worth overlaying. The patch is optional and
often absent (a config-only win is a real win), so `resolve` never treats a missing artifact as a
broken record the way the kernel lane does.

THE THREE RUNGS ANSWER THREE DIFFERENT QUESTIONS. They are not a fallback chain that happens to have
three links; each one is the correct page for a different ask, which is why all three are always
written:

    ...:mxfp8:tp_8:isl_1024:osl_1024:conc_64   "tune THIS benchmark point"
    ...:mxfp8:tp_8                             "given TP=8, how do I configure the server"
    ...:mxfp8                                  "how many ways should I shard this model at all"

Only the last one can compare TP4 against TP8, because that comparison needs them filed together.

READS AND WRITES RANK ON DIFFERENT METRICS, deliberately, and the split is the one thing to keep
straight in this file:

  * READING (`resolve`) orders every rung by absolute `throughput_tok_s`, high to low. That is what
    a Director asking "what should I run" wants on every page, including the coarse ones: it is
    choosing a config to spend a server launch on, and the fastest observed deployment is the
    honest first offer. The coarse rungs' numbers were measured at other workload points and are
    NOT comparable to this run's baseline — `_render_reference` says so in as many words — but a
    speedup ordering there is not more comparable, it just hides the incomparability behind a
    ratio. `--sort-by speedup` restores the old ordering for a caller that wants it.
  * WRITING (`write`) keeps the per-rung champion metric from `rung_metric()`: throughput on the
    exact rung, speedup on the coarser ones. The champion pointer is a promotion gate, not a
    display order, and a coarse page promoting on absolute tokens/sec would crown whichever
    workload point happens to be cheapest rather than whichever run improved anything.

Both scalars are written flat at the top of every document (the service's `sessions/top?metric=`
reads a top-level scalar and rejects a nested path), which is what lets a read rank on one while
the champion is kept on the other without a second write.

No delete exists on the service. Every `--apply` is permanent.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time

# The shared KB plane lives at the repo root as the `kb` package, not beside this file. Executed as
# a CLI from an arbitrary cwd, so the root is derived from __file__ and never from the environment.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb import identity as kbid                                             # noqa: E402
from kb.attest import (OUTCOMES, attest_session, attestation_ok,           # noqa: E402
                       attestations_of, carry_attestations, retire_hint)
from kb.curate import collapse_by_direction                                 # noqa: E402
from kb.ladder import publish                                               # noqa: E402
from kb.plane import open_plane                                             # noqa: E402
from kb.retract import is_retired, retract_session, retraction_ok           # noqa: E402
from kb.store_local import KBStoreError, finite_speedup                     # noqa: E402

SCHEMA = "geak.e2e.v1"
THROUGHPUT_METRIC = "throughput_tok_s"      # ranks the exact-workload rung, and every read
SPEEDUP_METRIC = "speedup"                  # champion metric on every coarser rung
DEFAULT_TOP_N = 3
DEFAULT_SCAN = 25
# What `resolve` orders a page by, by name on the CLI. See the module docstring for why a read and
# a write do not agree on this.
SORT_METRICS = {"throughput": THROUGHPUT_METRIC, "speedup": SPEEDUP_METRIC}
DEFAULT_SORT_BY = "throughput"

# Which metric ranks which rung, indexed the same way e2e_canonical_ids() returns them. A ladder
# shorter than three (the run recorded no tp, or no workload shape) drops rungs from the FRONT, so
# the table is applied by counting back from the end rather than by position.
_COARSE = (SPEEDUP_METRIC, 1.0)
_EXACT = (THROUGHPUT_METRIC, 0.0)


def rung_metric(index: int, total: int):
    """(metric, promote_floor) for rung `index` of `total`, most specific first."""
    is_exact_workload = (index == 0 and total == 3)
    return _EXACT if is_exact_workload else _COARSE


# -- identity ------------------------------------------------------------------------------------


def identity_of(a) -> dict:
    return kbid.e2e_identity(a.model, a.gfx, a.framework, a.framework_version, a.precision,
                             tp=a.tp, isl=a.isl, osl=a.osl, conc=a.conc)


def ladder_of(a):
    """[(canonical_id, tier, metric, floor)], most specific first.

    Tiers are named after what was DROPPED, not after how good the match is, so a caller logging
    `workload_any` can see at a glance that the numbers it is looking at were measured on some other
    benchmark point and must not be quoted as this deployment's throughput.
    """
    cids = kbid.e2e_canonical_ids(identity_of(a))
    tiers = {3: ("exact", "workload_any", "tp_any"), 2: ("exact", "tp_any"), 1: ("exact",)}[len(cids)]
    return [(cid, tier) + rung_metric(i, len(cids))
            for i, (cid, tier) in enumerate(zip(cids, tiers))]


# -- planes --------------------------------------------------------------------------------------


# -- read ----------------------------------------------------------------------------------------


def _view(candidate, cid: str, tier: str, metric: str, champion_metric: str = "") -> dict:
    """One offered record, flattened to what a Director prompt actually needs."""
    knowledge = candidate.knowledge if isinstance(candidate.knowledge, dict) else {}
    value = knowledge.get("value") if isinstance(knowledge.get("value"), dict) else {}
    workload = value.get("workload") if isinstance(value.get("workload"), dict) else {}
    ledger = attestations_of(value)
    return {
        "session_id": candidate.session_id,
        "canonical_id": cid,
        "match_tier": tier,
        "ranked_by": metric,
        # Which metric the CHAMPION on this page was promoted under, which on a coarse rung is not
        # the one the offer is ordered by. Spelled out so `is_champion` cannot be misread as "the
        # top of this list" — on a coarse rung the champion is the best speedup and the first row
        # is the best throughput, and those are routinely different records.
        "champion_metric": champion_metric or metric,
        "score": candidate.speedup,
        "throughput_tok_s": finite_speedup(knowledge.get(THROUGHPUT_METRIC)),
        "speedup": finite_speedup(knowledge.get(SPEEDUP_METRIC)),
        "baseline_throughput_tok_s": finite_speedup(value.get("baseline_throughput_tok_s")),
        "direction": str(value.get("direction") or ""),
        "workload": workload,
        "accepted_config": value.get("accepted_config") if isinstance(
            value.get("accepted_config"), dict) else {},
        "accepted_kernels": [k for k in (value.get("accepted_kernels") or [])
                             if isinstance(k, dict)][:32],
        "validation_status": str(value.get("validation_status") or ""),
        # How much to believe the number, as the writer judged it. Surfaced rather than used as a
        # filter: an unvalidated record is still a lead worth benching, and dropping it would make
        # the warm start narrower exactly where it is cheapest to be broad. Only a RETRACTED record
        # is filtered, because that one has been positively declared false.
        "validated": bool(value.get("validated")),
        "validation_basis": str(value.get("validation_basis") or "unverified"),
        "parity": str(value.get("parity") or ""),
        "lifecycle": str(value.get("lifecycle") or ""),
        "upstream": value.get("upstream") if isinstance(value.get("upstream"), dict) else {},
        # What this record has DONE SINCE it was written, as opposed to what its writer claimed
        # about it. `validated` above is one box's judgement of its own run; these are everyone
        # else's. A reader deciding whether to spend a server launch wants both, and they
        # disagree often enough that collapsing them would be a lie in one direction or the other.
        "validations": ledger["validations"],
        "recalls": ledger["recalls"],
        "not_reproduced": ledger["not_reproduced"],
        # Kept apart from `not_reproduced` all the way out to the reader. "the flag is gone
        # upstream" and "this box's baseline already pins that knob" look identical in a single
        # count and mean opposite things about whether the record is worth benching here.
        "inapplicable": ledger["inapplicable"],
        "last_outcome": ledger["last_outcome"],
        "retire_hint": retire_hint(value),
        # How to actually run this again. Empty for records written before the field existed —
        # those are the ones a reader has to reconstruct from `accepted_config` by hand.
        "repro": value.get("repro") if isinstance(value.get("repro"), dict) else {},
        "is_champion": bool(candidate.is_champion),
    }


def read_metric(a) -> str:
    """The metric a READ orders every rung by. `throughput_tok_s` unless the caller says otherwise.

    Applied to the store itself and not only to the list it hands back, which matters on the remote
    plane: `candidates()` pages `sessions/top?metric=` and hydrates the first `--scan` rows, so
    fetching a page ordered by speedup and then re-sorting it locally by throughput would rank a
    biased sample and quietly drop the fast-but-modest-ratio records that never made the cut.
    """
    return SORT_METRICS.get(str(getattr(a, "sort_by", "") or DEFAULT_SORT_BY).strip().lower(),
                            THROUGHPUT_METRIC)


def _sort_key(metric: str):
    """Rank order for a view list: the chosen metric first, the other as tie-break, id last.

    A record missing the metric sorts last rather than raising — a write cannot produce one (final
    throughput is required), but a document written by hand or by an older lane can.
    """
    other = SPEEDUP_METRIC if metric == THROUGHPUT_METRIC else THROUGHPUT_METRIC
    low = float("-inf")
    return lambda v: (-(v.get(metric) if v.get(metric) is not None else low),
                      -(v.get(other) if v.get(other) is not None else low),
                      v["session_id"])


def cmd_resolve(a) -> dict:
    ladder = ladder_of(a)
    metric = read_metric(a)
    # Echo the plane back. A caller that tries the service and falls back to disk otherwise cannot
    # tell from the output which one answered — the ladder, the ranking and the shapes are identical
    # — and "where did this candidate come from" is the first question asked when one turns out to
    # be wrong. `dict(out, ...)` carries it onto every return path below.
    out = {"tried": [c for c, _t, _m, _f in ladder], "canonical_id": ladder[0][0],
           "match_tier": "", "ranked_by": "", "sorted_by": metric, "champion_metric": "",
           "candidates": [], "read_reason": "",
           "plane": str(getattr(a, "plane", "local") or "local"), "curation": {}}
    # Leave the ADDRESS behind on disk, not just the answer. The writer at the end of the run is a
    # different process, and when the workflow dies mid-flight it is a different program entirely
    # (run_e2e.py salvaging the run from its artifacts) — one that has the measurement but not the
    # dimensions the Director established at preflight, and so could not address the KB at all. Its
    # write was silently lost for every run that did not finish cleanly. Written HERE, from the same
    # argv that formed the read, because two places formatting these dims independently is how a
    # reader and a writer drift onto different pages. Best-effort: a read must not fail over it.
    if getattr(a, "identity_out", ""):
        try:
            with open(a.identity_out, "w") as handle:
                # `dims` is the raw argv, one key per --flag of _identity_args, because the reader
                # of this file writes with those same flags: it hands them straight back. `identity`
                # is the derived form (gfx -> gpu, tp -> "tp_2"), kept for a human reading the file
                # and for anyone matching it against a canonical id — reconstructing argv FROM it
                # would mean a second, inverse copy of e2e_identity() living in run_e2e.py.
                json.dump({"identity": identity_of(a),
                           "dims": {"model": str(a.model or ""), "gfx": str(a.gfx or ""),
                                    "framework": str(a.framework or ""),
                                    "framework-version": str(a.framework_version or ""),
                                    "precision": str(a.precision or ""),
                                    "rocm-version": str(a.rocm_version or ""),
                                    "tp": a.tp, "isl": a.isl, "osl": a.osl, "conc": a.conc},
                           "store": str(getattr(a, "store", "") or ""),
                           # The plane the RUN writes on, which is not this read's plane: a `both`
                           # run reads remote-first (see kbResolveScript) and would otherwise leave
                           # behind "remote", talking the salvage writer out of the local mirror the
                           # run was configured to keep. Falls back to the read's own plane when the
                           # caller says nothing.
                           "plane": str(getattr(a, "identity_plane", "")
                                        or getattr(a, "plane", "") or "remote"),
                           "canonical_id": ladder[0][0]}, handle, indent=2, sort_keys=True)
        except OSError:
            pass
    last_why = ""
    for cid, tier, champion_metric, floor in ladder:
        # The rung's own metric opens nothing here: a read ranks every rung the same way (see
        # read_metric), and the floor only ever gates a promotion, which a read never performs.
        store, _mirror, why = open_plane(a, metric, floor)
        if store is None:
            last_why = why
            continue
        try:
            found = store.candidates(cid, limit=max(1, int(a.scan)))
        except Exception as e:
            last_why = "read_failed: %s: %s" % (type(e).__name__, str(e)[:120])
            continue
        # Retracted records are dropped BEFORE anything else looks at them, and before the
        # direction collapse in particular: a retracted entry that happens to rank first for its
        # direction would otherwise evict the surviving alternatives for that same direction, so a
        # single false record could hide every good one behind it. Done client-side because it has
        # to be — retraction zeroes the ranking scalar and re-points the champion, but the service
        # still serves the session, and nothing in the scheme lets us ask it not to.
        kept = [c for c in found if not is_retired(c.value)]
        curation = {"scanned": len(found), "retired": len(found) - len(kept),
                    "sorted_by": metric}
        # Re-sorted here even though the store already ordered by this metric, because the two
        # planes order by slightly different things: the local one ranks on the document scalar,
        # the remote one on the score the service computed and falls back to the document only
        # when that is absent. Sorting the hydrated views is the one place both planes are
        # guaranteed to agree, and collapse_by_direction's contract is that its input is already
        # in rank order — it keeps the FIRST entry per direction, so a wrong order here silently
        # offers the wrong member of every group.
        ordered = sorted([_view(c, cid, tier, metric, champion_metric) for c in kept],
                         key=_sort_key(metric))
        # top_n=len(kept): e2e filters min_speedup AFTER collapse and slices to top_n below, so
        # collapse must not pre-slice. It consumes only the per-idea best, not the alternates.
        views, _alternates, collapsed = collapse_by_direction(
            ordered, lambda v: v["direction"], lambda v: v["session_id"], len(kept))
        curation["same_direction_collapsed"] = collapsed
        if a.min_speedup:
            # Applied to `speedup` on every rung, including the throughput-ranked one: the floor
            # asks "did this run actually improve anything", which is a question about the ratio no
            # matter what the page is sorted by. A record with no speedup recorded is kept — it may
            # still be a usable config — rather than silently failing an unanswerable test.
            before = len(views)
            views = [v for v in views
                     if v["speedup"] is None or v["speedup"] >= float(a.min_speedup)]
            curation["below_min_speedup"] = before - len(views)
        curation["min_speedup"] = float(a.min_speedup or 0.0)
        if not views:
            # A rung whose every candidate was curated away is NOT an empty page, and the next rung
            # down is about to be tried as if it were. Carry the counts forward so the caller can
            # tell "nobody has recorded this" from "everything recorded here was retracted" —
            # identical read_reasons otherwise, opposite meanings.
            out["curation"] = dict(curation, canonical_id=cid, tier=tier)
            continue
        views = views[: max(1, int(a.top_n))]
        if a.cache_dir:
            for view in views:
                view["bundle"] = _materialize(store, cid, found, view, a.cache_dir)
        if a.refs_dir:
            _render_reference(a.refs_dir, cid, tier, views)
        return dict(out, canonical_id=cid, match_tier=tier, ranked_by=metric,
                    champion_metric=champion_metric, candidates=views, read_reason="read",
                    curation=dict(curation, canonical_id=cid, tier=tier))
    return dict(out, read_reason=last_why or "e2e_page_not_found")


def _materialize(store, cid: str, found, view: dict, cache_dir: str) -> dict:
    """Pull one record's artifacts down, reporting a failure instead of raising.

    An e2e record is usable without its files — the config lives in the knowledge document — so a
    download problem degrades the offer rather than dropping it.
    """
    candidate = next((c for c in found if c.session_id == view["session_id"]), None)
    if candidate is None:
        return {"error": "candidate vanished between ranking and download"}
    try:
        path = store.materialize(cid, candidate, cache_dir)
    except (KBStoreError, OSError) as e:
        return {"error": str(e)[:200]}
    # Walked, not listdir'd: artifacts nest (`kernels/<name>.patch`), and a flat listing reports the
    # directory itself as if it were the file, which is exactly the name a caller would then fail to
    # open.
    files_root = os.path.join(path, "files")
    names = sorted(os.path.relpath(os.path.join(root, f), files_root).replace(os.sep, "/")
                   for root, _dirs, found in os.walk(files_root) for f in found)
    return {"path": path, "files": names}


def _kernel_line(kernels) -> str:
    """`moe_stage1 (ck, 1.84x, kernels/moe_stage1.patch)` — one readable line per kernel.

    Spells out the patch path because the Director reads this prose and then has to go open the
    file; a name alone sends it back to the store to ask a question this page already answered.
    """
    parts = []
    for k in kernels:
        bits = [b for b in (k.get("language"),
                            "%sx" % k["isolated_speedup"] if k.get("isolated_speedup") else "",
                            k.get("patch") or k.get("kernel_canonical_id") or "") if b]
        parts.append("%s (%s)" % (k.get("name") or "?", ", ".join(bits)) if bits
                     else str(k.get("name") or "?"))
    return "; ".join(parts)


def _track_record_line(view: dict) -> str:
    """`benched 3x since: 1 reproduced, 1 no-win, 1 could not run (last: failed)`, or a plain miss."""
    if not view.get("recalls"):
        return "never benched by anyone since it was recorded"
    parts = ["%d reproduced a win" % view["validations"] if view["validations"] else "",
             "%d could not be run at all" % view["not_reproduced"]
             if view.get("not_reproduced") else "",
             "%d did not fit that box's baseline (no verdict on the record)" % view["inapplicable"]
             if view.get("inapplicable") else ""]
    detail = ", ".join(p for p in parts if p) or "none reproduced a win"
    hint = view.get("retire_hint") or ""
    return "benched %dx since it was recorded — %s%s" % (
        view["recalls"], detail, " (**%s**)" % hint if hint else "")


def _repro_line(view: dict) -> str:
    """Where the launch script and the kernel patches for this record actually are.

    Spelled as paths rather than as "see the bundle" because the reader is an agent that is about
    to run something, and the next thing it does after this line is open a file. A record written
    before `repro` existed says so plainly instead of pointing at a path that is not there.
    """
    repro = view.get("repro") or {}
    root = ((view.get("bundle") or {}).get("path") or "").rstrip("/")
    where = (lambda name: "%s/files/%s" % (root, name) if root else name)
    launch = str(repro.get("launch") or "")
    if not launch:
        return ("no launch script recorded (pre-`repro` record) — rebuild it from the config below")
    bits = ["`%s`%s" % (where(launch),
                        " (SYNTHESIZED from the config, never executed as-is)"
                        if repro.get("launch_origin") == "synthesized" else " (captured from the run)")]
    patches = [k.get("patch") for k in (repro.get("kernels") or []) if isinstance(k, dict)
               and k.get("patch")]
    if patches:
        bits.append("kernel patches: " + ", ".join("`%s`" % where(p) for p in patches))
    missing = int(repro.get("kernels_without_patch") or 0)
    if missing:
        bits.append("%d accepted kernel(s) carry NO patch here — that part of the win cannot be "
                    "reproduced from this record" % missing)
    return "; ".join(bits)


def _render_reference(refs_dir: str, cid: str, tier: str, views) -> str:
    """Mirror the offer into prose the Director can read, or return "" and let the read stand."""
    try:
        os.makedirs(refs_dir, exist_ok=True)
        key = hashlib.sha256(("|".join(v["session_id"] for v in views)).encode()).hexdigest()[:7]
        path = os.path.join(refs_dir, "e2e_reference_%s.md" % key)
        lines = ["# e2e warm start — `%s`" % cid, "",
                 "Match tier `%s`, ordered by `%s` (highest first)."
                 % (tier, views[0]["ranked_by"]), ""]
        if tier != "exact":
            lines += ["> These were measured on a DIFFERENT workload point than the one requested. "
                      "Treat the configs as candidates and the numbers as non-comparable — do not "
                      "quote them as this deployment's throughput. The ordering above is by "
                      "absolute throughput, so it says which deployment was fastest AT ITS OWN "
                      "workload point, not which one would be fastest here.", ""]
        for rank, v in enumerate(views, start=1):
            lines += [
                "## %d. %s%s" % (rank, v["direction"] or "unlabeled",
                                 " (champion)" if v["is_champion"] else ""),
                "- throughput: %s tok/s (baseline %s), speedup %s" % (
                    v["throughput_tok_s"], v["baseline_throughput_tok_s"], v["speedup"]),
                "- workload: %s" % (json.dumps(v["workload"], sort_keys=True) or "{}"),
                # Spelled out rather than reduced to a word: the Director's next decision is
                # whether to spend a server launch on this, and "unverified, parity n/a" is a very
                # different prompt from "validated, hot A/B, parity pass" even at the same speedup.
                "- validation: %s (%s, basis %s, parity %s)" % (
                    "VALIDATED" if v["validated"] else "unvalidated",
                    v["validation_status"] or "unrecorded",
                    v["validation_basis"] or "unverified", v["parity"] or "unrecorded"),
                # The claim above is the writer's own; this line is everybody else's experience of
                # it since. A record benched three times and never reproduced is a very different
                # bet from an untried one at the same speedup, and only this line says which it is.
                "- track record: %s" % _track_record_line(v),
                "- accepted kernels: %s" % (_kernel_line(v["accepted_kernels"]) or "none"),
                "- reproduce: %s" % _repro_line(v),
                "- config:", "```json",
                json.dumps(v["accepted_config"], indent=2, sort_keys=True), "```", "",
            ]
        with open(path, "w") as handle:
            handle.write("\n".join(lines))
        return path
    except OSError:
        return ""


# -- write ---------------------------------------------------------------------------------------


def _record_state(a, result: dict) -> dict:
    """How much this record's number should be believed, recorded AT WRITE TIME.

    A reader cannot recover this later. `validation_status` alone does not answer it — a record can
    say `validated_win` and still be a number nobody checked for output parity, and the two are
    indistinguishable once the run's logs are gone. So the judgement is made here, once, by the
    process that still has the evidence, and stored as a first-class field:

      * `validated` — did this number clear a real gate on the box that produced it. False is a
        perfectly good answer and does NOT mean the record is worthless; it means a reader should
        re-measure before quoting it. A parity of `n/a` is false, not true: "we did not check" is
        not "we checked and it was fine".
      * `validation_basis` — WHICH gate. `hot_ab` is the Director's same-session interleaved A/B
        (the strong one). `cold_gate` is a fresh-server before/after (weaker: it carries the drift
        between two launches). `unverified` is a number that was reported but never independently
        reproduced. These are not comparable, so a reader that mixes them must at least know it is.
      * `parity` — pass | fail | n/a, verbatim. A faster server that answers differently is a
        regression, and this is the only field that says whether anyone looked.
      * `lifecycle` — `active` (believed) or `candidate` (recorded, unproven). The third value,
        `retracted`, is written only by kb/retract.py and never by a fresh write.
      * `retained` — the curation flag both lanes' readers filter on. True at birth; retraction
        flips it. Written explicitly rather than left absent so `retained is False` stays a
        three-state test (true / false / never stated) instead of degrading to a falsy check.

    Every field is overridable from the CLI because the caller sometimes knows better than the
    result JSON — a backfill from an old run has evidence this function cannot see, and forcing it
    to fake a `validation_status` to get the right state would corrupt the field that means
    something else.
    """
    status = str(result.get("validation_status") or "")
    parity = str(getattr(a, "parity", "") or result.get("output_parity") or "").strip().lower()
    basis = str(getattr(a, "validation_basis", "") or "").strip().lower()
    if not basis or basis == "auto":
        # A `validation_status` is only ever set by the Director's validate phase, and that phase is
        # the same-session A/B by construction. No status means nothing re-measured this run.
        basis = "hot_ab" if status else "unverified"
    validated = str(getattr(a, "validated", "") or "auto").strip().lower()
    if validated in ("", "auto"):
        decided = status == "validated_win" and parity == "pass"
    else:
        decided = validated in ("1", "true", "yes", "on")
    return {"validated": decided, "validation_basis": basis, "parity": parity or "n/a",
            "lifecycle": "active" if decided else "candidate", "retained": True}


def build_record(a, result: dict, workdir=None) -> dict:
    """One run's knowledge document, identical at every rung.

    Both ranking scalars sit flat at the top level because that is the only shape
    `sessions/top?metric=` can read. Everything else lives under `value`, including the dimensions
    that are already in the canonical id — a record that cannot say what it is once detached from
    its address is not auditable.

    `workdir` is a scratch directory whose lifetime must outlast the store write: the synthesized
    launch script is created in it and uploaded from it. Passing None asks for the document only,
    which is what `retract` and `attest` want when they recompute the content digest — they must
    not synthesize files, and must not refuse over a reproducibility rule that only governs new
    writes.
    """
    identity = identity_of(a)
    final = finite_speedup(result.get("final_throughput_tok_s"))
    baseline = finite_speedup(result.get("baseline_throughput_tok_s"))
    speedup = finite_speedup(result.get("throughput_speedup"))
    if speedup is None and final is not None and baseline:
        speedup = round(final / baseline, 6)
    if final is None:
        raise SystemExit("result has no final_throughput_tok_s; refusing to record a run with no "
                         "measurement — an unranked record is invisible on every page")
    kernels, kernel_files = _accepted_kernels(a, result)
    state = _record_state(a, result)
    value = {
        "model": identity["model"], "gpu": identity["gpu"],
        "framework": identity["framework"], "framework_version": identity["framework_version"],
        "precision": identity["precision"],
        # Ints, not the raw argv strings: this is the only copy of the shape once a record is read
        # back off a coarse rung, and "1024" sorts and compares differently from 1024 in every
        # consumer that touches it. A value that will not parse is dropped, matching counted().
        "workload": {k: int(str(getattr(a, k)).strip())
                     for k in ("tp", "isl", "osl", "conc")
                     if str(getattr(a, k, "") or "").strip().lstrip("-").isdigit()},
        "baseline_throughput_tok_s": baseline,
        "final_throughput_tok_s": final,
        "direction": str(a.direction or result.get("direction") or ""),
        "accepted_config": result.get("accepted_config") if isinstance(
            result.get("accepted_config"), dict) else {},
        "accepted_kernels": kernels,
        "validation_status": str(result.get("validation_status") or ""),
        "upstream": result.get("upstream") if isinstance(result.get("upstream"), dict) else {},
        # GEAK's own comparability keys (schema v2): what basis the pair was measured on, which
        # client took the number, which workload points were validated. A stored speedup is only
        # meaningful against these, so they ride WITH the number rather than being rediscovered.
        "comparability": result.get("comparability") if isinstance(
            result.get("comparability"), dict) else {},
        "measured_by": str(a.measured_by or ""),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    value.update(state)
    files = _artifact_files(a, result)
    files.update(kernel_files)
    value["artifacts"] = {k: v[0] for k, v in files.items()}
    # After `artifacts` (it reads the captured launch/overlay names from there) and before the
    # final rebuild (it may ADD the synthesized script and fetched patches to `files`).
    value["repro"] = _repro(a, result, value, kernels, files, workdir)
    if files:
        value["artifacts"] = {k: v[0] for k, v in files.items()}
    else:
        value.pop("artifacts", None)
    document = {"schema": SCHEMA, THROUGHPUT_METRIC: final, "value": value}
    if speedup is not None:
        document[SPEEDUP_METRIC] = speedup
    # Keyed by STORED NAME, not by role: `value.artifacts` maps role -> stored name, and
    # materialize() checks every one of those names against what the bundle actually holds. Keying
    # the upload by role instead writes `files/patch` while the document promises `final.patch`,
    # and the record fails its own integrity check on the way back out.
    return {"knowledge": document, "files": {v[0]: v[1] for v in files.values()},
            "speedup": speedup, "throughput": final}


def _kernel_records(result: dict):
    """Every kernel this run kept, from BOTH tracks, deduped by name.

    e2e_workflow.js banks head-op rewrites into `accepted_heads` and milestone rewrites into
    `accepted_kernels`, and its own final view of a run is the union of the two. Reading only
    `accepted_kernels` here dropped an entire track — both e2e records already in the remote KB came
    from runs whose whole win lived in `accepted_heads`, so they recorded no kernels at all. Union,
    not fallback: a run that used both tracks must lose neither.

    MERGE rather than first-wins on a name collision. One op optimized on both tracks is one kernel,
    and the two entries know different things about it (a head carries target_callable, a milestone
    carries source_path_in_sglang). Dropping the second would reintroduce exactly the silent field
    loss this function exists to prevent.

    `short_name` is accepted as a name spelling because that is what the workflow pushed for years
    before it also emitted `name`; a record written by an older lane still reads correctly here.
    """
    merged, order = {}, []
    for source, raw_list in (("accepted_kernels", result.get("accepted_kernels")),
                             ("accepted_heads", result.get("accepted_heads"))):
        for raw in (raw_list or []):
            item = {"name": str(raw)} if not isinstance(raw, dict) else dict(raw)
            name = str(item.get("name") or item.get("kernel_name")
                       or item.get("short_name") or "").strip()
            if not name:
                continue
            item["name"] = name
            if source == "accepted_heads":
                item.setdefault("from_accepted_heads", True)
            if name not in merged:
                merged[name] = item
                order.append(name)
                continue
            for key, value in item.items():
                if value in (None, "", 0):
                    continue
                if merged[name].get(key) in (None, "", 0):
                    merged[name][key] = value
    return [merged[name] for name in order]


def _accepted_kernels(a, result: dict):
    """(entries, {role: (stored_name, local_path)}) for the kernels this run kept.

    A bare list of names was useless to a reader: it said a kernel mattered without saying how to
    obtain it, so the next run had to rediscover the same rewrite. Each entry now carries BOTH ways
    to get the patch, because each fails differently:

      * `kernel_canonical_id` addresses the kernel lane's own record — the source of truth, with
        that kernel's full history, its own champion and its own measurements. Preferred. It can
        miss: the kernel page is only there if the kernel lane wrote it, and it is keyed on the
        ROCm version this run has to declare.
      * `patch` is the diff itself, copied into THIS record under `kernels/<name>.patch`. It costs
        duplicated bytes and it goes stale, but it is the only copy that cannot 404 — an e2e record
        whose kernel references have rotted still reproduces the configuration it is claiming.

    Accepts either shape from the workflow: plain strings (what FINALIZE_SCHEMA's accepted_kernels
    emits today) or objects carrying language/patch/speedup. Strings degrade to a name and a
    canonical id, never to an error, so this stays readable by the unmodified e2e_workflow.js.

    Both tracks are read — see _kernel_records for why reading one of them lost half of every run.
    """
    entries, files = [], {}
    gfx = kbid.segment(a.gfx, kbid.UNKNOWN)
    rocm = str(getattr(a, "rocm_version", "") or
               (result.get("upstream") or {}).get("rocm_version") or "")
    for item in _kernel_records(result):
        name = item["name"]
        language = str(item.get("language") or item.get("backend") or "").strip()
        # Start from what the producer sent and normalize over it, rather than copying a fixed set
        # of keys into a fresh dict. A whitelist silently discards everything the workflow knows
        # that this function has not been taught about — `kind`, `op_kind`, `e2e_delta_pct`,
        # `routed_to` all vanished on the first real run — and the loss is invisible until someone
        # reads the record back looking for a field they are sure they wrote.
        entry = dict(item)
        entry.update({"name": name, "language": language,
                      # `isolated` is the spelling the workflow used at every push site before
                      # bankAccepted also emitted `isolated_speedup`; without it an older
                      # result.json ranks as having no measurement at all.
                      "isolated_speedup": finite_speedup(item.get("isolated_speedup")
                                                         or item.get("speedup")
                                                         or item.get("isolated")),
                      "pct_gpu_time": finite_speedup(item.get("pct_gpu_time"))})
        entry.pop("patch_path", None)     # replaced below by the STORED name, if it exists
        # Only address the kernel lane when the language is known. Guessing it would mint an id
        # that looks authoritative and resolves to nothing, which on a scheme with no search is
        # indistinguishable from the kernel never having been optimized.
        #
        # An env/flag win is NOT a kernel rewrite — it routes the op to an existing implementation
        # (`routed_to: aiter`), and the e2e workflow stores that routing target in the same `backend`
        # field a real rewrite uses for its LANGUAGE. The kernel lane never wrote a page for it, so
        # addressing one here fabricates a permanent dead reference on a store with no delete.
        kind = str(item.get("winner_kind") or item.get("kind") or "").strip().lower()
        if language and kind not in ("env", "flag"):
            entry["kernel_canonical_id"] = kbid.kernel_canonical_ids(
                kbid.kernel_identity(gfx, name, language, rocm))[0]
        if item.get("session_id"):
            entry["kernel_session_id"] = str(item["session_id"])
        patch = str(item.get("patch") or item.get("patch_path") or "")
        if patch and os.path.isfile(patch):
            stored = "kernels/%s.patch" % kbid.segment(name, "kernel")
            files["kernel:" + name] = (stored, patch)
            entry["patch"] = stored
        entries.append(entry)
    return entries, files


_ARTIFACT_KEYS = (("patch", "final_patch", "final.patch"),
                  ("launch", "final_launch_script", "launch.sh"),
                  ("report", "report_path", "report.md"),
                  ("overlay", "final_overlay", "overlay.py"))

OVERLAY_MANIFEST = "_overlay_manifest.json"
OVERLAY_TARBALL = "overlay.tar.gz"
# Build/interpreter caches. Neither survives being moved to another reader's box, and both are
# larger than the source they were derived from.
OVERLAY_SKIP_DIRS = ("__pycache__", ".torch_ext")
# Holds the TemporaryDirectory objects alive for the life of the process. The tarball has to still
# be on disk when the store uploads it, which happens long after _artifact_files() returns, and a
# TemporaryDirectory that goes out of scope deletes its tree immediately.
_PACKED = []


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)\s+import\b|import\s+([A-Za-z_]\w*))",
                        re.MULTILINE)


def _sibling_imports(path: str) -> list:
    """Top-level module names `path` imports, by source text rather than by importing it.

    Reading the source is the only option here: importing an overlay module off the write path
    would run its top-level code — which for an authored kernel means compiling Triton against
    whatever GPU the writer happens to be on. The regex takes plain `import X` / `from X import`
    only; a dotted or relative form is somebody else's package, not an overlay sibling.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError:
        return []
    return [a or b for a, b in _IMPORT_RE.findall(source)]


def _overlay_modules(manifest_path: str) -> list:
    """Every overlay-root module the mechanism needs, transitively, deduplicated.

    Two roots, because the overlay has two entry points. The manifest names the module each
    rebind binds TO, and sitecustomize.py is the code that installs them — an overlay that
    captures shapes or traces a seam does that work in siblings sitecustomize.py imports
    directly, with no manifest entry naming them at all.

    Neither root is the whole story on its own, because what they name is regularly a shim:
    GLM-5.2's `dsa_engage_c0_triton` is 1.6 KB of `from dsa_authored_c0_triton import
    tilelang_sparse_fwd` wrapping a 23 KB authored Triton kernel, and packing the manifest's
    name alone shipped a tarball that raises ImportError the moment sitecustomize.py runs it.
    So each packed module's own imports are followed, and any that resolves to a sibling at the
    overlay root is packed too. Names that resolve to nothing at the root are just ordinary
    third-party imports and are left alone; the closure therefore terminates at the overlay
    boundary.

    A malformed manifest still yields sitecustomize.py's side of the closure rather than
    raising: the caller is packing a best-effort artifact, and `_repro()` is the thing that
    decides whether what came out is enough.
    """
    root = os.path.dirname(manifest_path)
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    seen, out = set(), []
    queue = [str((entry or {}).get("impl_module") or "") for entry in (manifest.get("rebinds") or [])]
    queue.extend(_sibling_imports(os.path.join(root, "sitecustomize.py")))
    while queue:
        # A submodule rebind (`geak_authored.gemm_flydsl`) is addressed by its top-level package,
        # which is what sits at the overlay root and what has to be packed. An absolute or
        # separator-bearing name is not a module name at all and is refused rather than resolved
        # into somebody's filesystem.
        module = queue.pop(0).split(".")[0]
        if not module or module in seen or os.path.isabs(module) or os.sep in module:
            continue
        as_file, as_pkg = os.path.join(root, module + ".py"), os.path.join(root, module)
        if not os.path.isfile(as_file) and not os.path.isdir(as_pkg):
            continue
        seen.add(module)
        out.append(module)
        if os.path.isfile(as_file):
            queue.extend(_sibling_imports(as_file))
        else:
            for base, dirs, names in os.walk(as_pkg):
                dirs[:] = [d for d in dirs if d not in OVERLAY_SKIP_DIRS]
                for name in names:
                    if name.endswith(".py"):
                        queue.extend(_sibling_imports(os.path.join(base, name)))
    return out


def _pack_overlay(dirpath: str) -> str:
    """`final_overlay`'s directory -> a .tar.gz of the overlay mechanism, or "" if it is not one.

    `final_overlay` names a DIRECTORY, always: the mechanism is a sitecustomize.py plus the modules
    it swaps in, and no single file carries it. _artifact_files() used to take only
    `os.path.isfile`, so that directory was dropped without a word and no e2e record ever carried
    an overlay — the runs that cleared _repro()'s reproducibility gate were the ones that also
    happened to emit a kernel patch, and a pure-overlay win could not be recorded at all.

    Only the mechanism goes in: the manifest, the sitecustomize.py that installs it, and the code
    the manifest names. The manifest names it two different ways and BOTH have to be packed. A
    `modules` entry replaces a whole upstream module and points at a file under `_patched/`. A
    `rebinds` entry leaves the module alone and swaps one symbol for `impl_module`'s — a sibling
    package or .py file at the overlay's top level, NOT under `_patched/`. Packing only `_patched/`
    silently produced a tarball with a manifest that rebinds to an import that is not in it: the
    gpt-oss-120b `_fwd_kernel` win is a `rebinds` overlay whose entire HIP kernel lives in
    `geak_hip_extend/`, and the record would have promised a reproducible run and shipped nothing
    to reproduce it with.

    An accepted-candidate directory also holds the A/B evidence it was judged on (`cand/`, `ref/`,
    server logs, profiles) — that is how the number was arrived at, not how it is reproduced, and
    it is several times the size of what a reader needs. `__pycache__` is skipped: a .pyc is stale
    the moment the reader's interpreter differs. `.torch_ext/` is skipped for the same reason one
    step further along: it is a torch cpp_extension BUILD cache (ninja logs, .o, a gfx-specific
    .so) that the reader's own load() rebuilds from the .hip source that is packed.

    Everything is packed under a single `overlay/` top level so the reader can untar into the
    bundle root and get one predictable directory to point PYTHONPATH at (see _launch_text).

    No manifest means this is not an overlay directory. Returning "" then is deliberate — the gate
    refuses the record rather than have it promise a tarball of something nobody can install.
    """
    manifest_path = os.path.join(dirpath, OVERLAY_MANIFEST)
    if not os.path.isfile(manifest_path):
        return ""
    holder = tempfile.TemporaryDirectory(prefix="e2e_overlay_")
    _PACKED.append(holder)
    out = os.path.join(holder.name, OVERLAY_TARBALL)

    def _add_tree(tar, root_dir):
        for root, dirs, names in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in OVERLAY_SKIP_DIRS]
            for name in names:
                if name.endswith(".pyc"):
                    continue
                src = os.path.join(root, name)
                tar.add(src, arcname="overlay/" + os.path.relpath(src, dirpath))

    with tarfile.open(out, "w:gz") as tar:
        for name in (OVERLAY_MANIFEST, "sitecustomize.py"):
            src = os.path.join(dirpath, name)
            if os.path.isfile(src):
                tar.add(src, arcname="overlay/" + name)
        _add_tree(tar, os.path.join(dirpath, "_patched"))
        for module in _overlay_modules(manifest_path):
            pkg = os.path.join(dirpath, module)
            if os.path.isdir(pkg):
                _add_tree(tar, pkg)
            elif os.path.isfile(pkg + ".py"):
                tar.add(pkg + ".py", arcname="overlay/" + module + ".py")
    return out


TUNING_PREFIX = "tuning/"
# Bounds on what the tuning phase can drag into a record. Tuned tables are small (the a8w8 blockscale
# CSVs run 0.5-1 MB), so a path far outside that is a deploy bundle or a built .so that got listed by
# mistake, and a record is the wrong place to discover it. Both caps report what they dropped rather
# than truncating quietly — a record that silently carries half its lever is the failure being fixed.
TUNING_FILE_MAX = 24
TUNING_FILE_MAX_BYTES = 64 * 1024 * 1024


def _tuning_files(result: dict, dropped: list) -> dict:
    """{role: (stored_name, local_path)} for the tuning phase's deployable artifacts.

    A tuned table is applied through an env var and deployed INTO the installed package
    (`site-packages/aiter/configs/...`), not into the framework's git tree. It is therefore
    STRUCTURALLY absent from `final.patch` — no amount of diffing the source tree will pick it up —
    and a record carrying only the patch reproduces the launch command but not the win. The
    DeepSeek-V4-Pro 20260823 run wrote files [final.patch, launch.sh, report.md] while its actual
    lever, a 56-row a8w8 blockscale table measured at 3.29x isolated, stayed on the box and was lost
    with it. `live_tree_files` exists precisely because these paths sit outside the patchable tree,
    so it is read here alongside `artifacts`.

    Only an ACCEPTED tuning contributes. A withdrawn or unproven one has no artifact worth carrying,
    and copying one in would let a reader mistake a rejected search residue for a banked lever.
    """
    tuning = result.get("tuning_skillset") or {}
    if not isinstance(tuning, dict) or str(tuning.get("gate") or "") != "accepted":
        return {}
    found, seen = {}, set()
    for path in (list(tuning.get("artifacts") or []) + list(tuning.get("live_tree_files") or [])):
        path = str(path or "")
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            dropped.append("%s (missing)" % path)
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            dropped.append("%s (unreadable)" % path)
            continue
        if size > TUNING_FILE_MAX_BYTES:
            dropped.append("%s (%d bytes > cap)" % (path, size))
            continue
        if len(found) >= TUNING_FILE_MAX:
            dropped.append("%s (over %d-file cap)" % (path, TUNING_FILE_MAX))
            continue
        # Index-prefixed: two tuned tables can share a basename across trees (the deploy bundle's copy
        # and the installed one), and a bare basename would silently drop one of them.
        stored = "%s%02d_%s" % (TUNING_PREFIX, len(found), os.path.basename(path))
        found["tuning:" + path] = (stored, path)
    return found


def _artifact_files(a, result: dict) -> dict:
    """{role: (stored_name, local_path)} for the run outputs that actually exist on disk.

    A path the result names but the filesystem does not have is dropped here rather than at upload
    time, so `value.artifacts` never promises a file the record does not carry — the kernel lane's
    materialize() now treats that promise as a hard error, and it should.
    """
    found = {}
    for role, field, stored in _ARTIFACT_KEYS:
        path = str(result.get(field) or "")
        if not path:
            continue
        if os.path.isfile(path):
            found[role] = (stored, path)
        elif role == "overlay" and os.path.isdir(path):
            packed = _pack_overlay(path)
            if packed:
                found[role] = (OVERLAY_TARBALL, packed)
    dropped = []
    found.update(_tuning_files(result, dropped))
    for why in dropped:
        sys.stderr.write("e2e_store: tuning artifact NOT carried into the record: %s\n" % why)
    for extra in (getattr(a, "file", None) or []):   # retract recomputes a record but takes no --file
        path = str(extra)
        if os.path.isfile(path):
            found["file:" + os.path.basename(path)] = (os.path.basename(path), path)
    return found


def _env_pairs(env: str) -> dict:
    """`"A=1 B=2"` -> `{"A": "1", "B": "2"}`, which is the shape a reader can act on.

    The workflow carries server env as one flat string because that is what it hands
    `bench_e2e.sh`'s EXTRA_ENV, and a reader who wants to know whether a record set
    VLLM_USE_AITER should not have to write a parser to find out. Anything that is not a
    `KEY=VALUE` token is skipped rather than guessed at; the verbatim string is kept alongside
    this dict, so nothing is lost by being strict here.
    """
    pairs = {}
    for token in str(env or "").split():
        key, sep, val = token.partition("=")
        if sep and key and not key[0].isdigit():
            pairs[key] = val
    return pairs


def _fetch_kernel_patch(root: str, canonical_id: str, name: str, into: str) -> str:
    """Pull one kernel's patch out of the kernel lane's local store, or return "".

    The e2e record names the kernel lane's canonical id, but the bytes only ride along if the
    workflow happened to still have the patch file on disk at write time — and by the time the
    e2e run finalizes, the kernel workflow's scratch is usually gone. This walks over to the
    kernel lane's own store and copies its champion's patch in, so the e2e record is complete
    without the e2e run having had to hoard files it did not produce.

    Best-effort by construction: a kernel page that does not exist, a bundle that fails its
    integrity check, an unreadable root — all of them mean "no patch here", which is a state the
    record already knows how to describe (`kernels_without_patch`). Never raises.
    """
    try:
        from kb.store_local import LocalKBStore
        store = LocalKBStore(root, metric=SPEEDUP_METRIC)
        found = store.candidates(canonical_id, limit=1)
        if not found:
            return ""
        bundle = store.materialize(canonical_id, found[0].session_id, into)
        for candidate in ("patch.diff", "final.patch", "patch"):
            path = os.path.join(bundle, "files", candidate)
            if os.path.isfile(path):
                return path
    except Exception:
        return ""
    return ""


def _kernel_patches(a, kernels, files: dict, workdir: str) -> int:
    """Fill in the patches the run did not carry, and return how many are STILL missing.

    Mutates `kernels` (setting `patch` on entries it manages to fetch) and `files` (adding the
    bytes to upload) in place, because both are already the write path's accumulators and a
    third copy would just be a chance for them to disagree.
    """
    root = str(getattr(a, "kernel_store", "") or "")
    missing = 0
    for entry in kernels:
        if entry.get("patch"):
            continue
        cid = str(entry.get("kernel_canonical_id") or "")
        name = str(entry.get("name") or "kernel")
        fetched = ""
        if root and cid and workdir:
            fetched = _fetch_kernel_patch(root, cid, name,
                                          os.path.join(workdir, "kernel_%s" % kbid.segment(name,
                                                                                           "k")))
        if fetched:
            stored = "kernels/%s.patch" % kbid.segment(name, "kernel")
            files["kernel:" + name] = (stored, fetched)
            entry["patch"] = stored
            entry["patch_origin"] = "kernel_store"
            continue
        # Said out loud in the record itself. A reader that sees three accepted kernels and two
        # patches has no way to tell whether the third was a no-op or whether its bytes were
        # simply lost, and those two readings lead to opposite decisions about whether the
        # configuration below is worth trying.
        entry["patch_missing"] = True
        missing += 1
    return missing


def _launch_text(a, result: dict, value: dict, kernels, overlay: str) -> str:
    """A runnable `launch.sh` built from the config, for a run that captured no script.

    Written against `e2e_workflow/scripts/bench_e2e.sh`'s env contract, because that script is
    how this pipeline actually measured the number being recorded — reproducing the record means
    re-entering it at the same door, not inventing a second launcher whose defaults nobody has
    compared. Every knob it reads that we know is emitted; the ones we cannot know (MODEL's path,
    HOST/PORT) are left as environment overrides with loud defaults.

    The header says SYNTHESIZED in as many words. A reader must be able to tell a script that was
    executed and produced the number above from one that was reconstructed afterwards and has
    never been run, and no amount of correctness in the body substitutes for saying which it is.
    """
    identity = identity_of(a)
    workload = value.get("workload") or {}
    config = value.get("accepted_config") or {}
    flags = str(config.get("flags") or "")
    env = str(config.get("env") or "")
    model_path = str(result.get("model_path") or (result.get("upstream") or {}).get("model_path")
                     or "")
    lines = [
        "#!/usr/bin/env bash",
        "# SYNTHESIZED by e2e_store.py from this record's accepted_config — NOT the script that",
        "# produced the number below. It has never been executed as written. Read it before you",
        "# run it: the paths it cannot know (MODEL, GEAK_SCRIPTS) are yours to supply.",
        "#",
        "#   identity     : %s" % " | ".join(
            [identity["model"], identity["gpu"], identity["framework"],
             identity["framework_version"], identity["precision"]]),
        "#   workload     : %s" % (json.dumps(workload, sort_keys=True) or "{}"),
        "#   direction    : %s" % (value.get("direction") or "unlabeled"),
        "#   measured     : %s tok/s vs baseline %s" % (value.get("final_throughput_tok_s"),
                                                        value.get("baseline_throughput_tok_s")),
        "#   recorded_at  : %s by %s" % (value.get("recorded_at") or "",
                                         value.get("measured_by") or "unknown"),
        "",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        '# Where GEAK\'s e2e_workflow/scripts lives on THIS box. bench_e2e.sh is the entry point',
        '# the recorded number was measured through.',
        ': "${GEAK_SCRIPTS:?set GEAK_SCRIPTS to GEAK/e2e_workflow/scripts}"',
        ': "${MODEL:=%s}"' % (model_path or ""),
        'if [ -z "${MODEL}" ]; then',
        # The caller's spelling, not the canonical segment: `qwen3-397b` is an address, and the
        # reader has to type a path or an HF id, which is case-sensitive.
        '  echo "set MODEL to the path or HF id of %s (bench_e2e.sh requires it)" >&2; exit 4'
        % (str(getattr(a, "model", "") or "") or identity["model"]),
        "fi",
        "",
    ]
    if kernels:
        patched = [k for k in kernels if k.get("patch")]
        lines += ["# Kernel rewrites this configuration depends on. SRC must point at the source",
                  "# tree the patches were cut against (the serving stack's checkout)."]
        if patched:
            lines += ['SRC="${SRC:-}"',
                      'if [ -n "${SRC}" ]; then',
                      '  for p in %s; do' % " ".join('"$HERE/%s"' % k["patch"] for k in patched),
                      '    git -C "$SRC" apply --check "$p" && git -C "$SRC" apply "$p"',
                      "  done",
                      "else",
                      '  echo "SRC unset: skipping %d kernel patch(es) — the recorded speedup will '
                      'NOT reproduce without them" >&2' % len(patched),
                      "fi"]
        absent = [k for k in kernels if not k.get("patch")]
        if absent:
            lines += ["# NO PATCH IN THIS RECORD for: %s" % ", ".join(
                str(k.get("name") or "?") for k in absent),
                "# Fetch them from the kernel lane (kernel_canonical_id in value.accepted_kernels)",
                "# or the number below will not reproduce."]
        lines.append("")
    if overlay.endswith(".tar.gz"):
        # Packed by _pack_overlay under a single `overlay/` top level. Extracted rather than shipped
        # loose so the bundle keeps one file per artifact role; guarded so a reader who already
        # unpacked (or edited) it does not get their copy overwritten on the next run.
        lines += ['# Python-level overlay this run served with.',
                  '[ -d "$HERE/overlay" ] || tar -xzf "$HERE/%s" -C "$HERE"' % overlay,
                  'OVERLAY_PYTHONPATH="${OVERLAY_PYTHONPATH:-$HERE/overlay}"', ""]
    elif overlay:
        lines += ['# Python-level overlay this run served with.',
                  'OVERLAY_PYTHONPATH="${OVERLAY_PYTHONPATH:-$HERE/%s}"' % overlay, ""]
    pairs = _env_pairs(env)
    if pairs:
        lines.append("# Server environment, as recorded.")
        lines += ["export %s=%s" % (k, _sh_quote(v)) for k, v in sorted(pairs.items())]
        lines.append("")
    lines += ["exec env \\"]
    for key, val in (("BACKEND", identity["framework"]),
                     ("MODEL", "${MODEL}"),
                     ("TP", workload.get("tp")), ("ISL", workload.get("isl")),
                     ("OSL", workload.get("osl")), ("CONC", workload.get("conc")),
                     ("GPU", "${GPU:-0}"), ("OUT_DIR", "${OUT_DIR:-$PWD/repro_out}")):
        if val in (None, "", kbid.UNKNOWN):
            continue
        lines.append("  %s=%s \\" % (key, _sh_quote(str(val))))
    if flags:
        lines.append("  EXTRA_SERVER_ARGS=%s \\" % _sh_quote(flags))
    if env:
        lines.append("  EXTRA_ENV=%s \\" % _sh_quote(env))
    if overlay:
        lines.append('  OVERLAY_PYTHONPATH="${OVERLAY_PYTHONPATH}" \\')
    lines += ['  bash "${GEAK_SCRIPTS}/bench_e2e.sh"', ""]
    return "\n".join(lines)


def _sh_quote(text: str) -> str:
    """Single-quote for /bin/sh, leaving `${VAR}` references alone.

    Not shlex.quote: several of the values above are deliberately shell expansions the reader is
    meant to be able to override from the environment, and quoting those into literals would turn
    a script you can point at your own model into one that tries to open a file called `${MODEL}`.
    """
    text = str(text)
    if text.startswith("${") and text.endswith("}"):
        return '"%s"' % text
    return "'%s'" % text.replace("'", "'\"'\"'")


def _repro(a, result: dict, value: dict, kernels, files: dict, workdir) -> dict:
    """`value.repro`: everything needed to run this configuration again, said twice.

    Once as a script (`launch`) for a reader that wants to run it, and once as structured fields
    for a reader that wants to reason about it without parsing shell. Both, not either: the
    script is the only artifact that is complete, and the fields are the only form a curation
    pass or another agent can compare across records.

    `workdir` is None when the caller only wants the document's shape — `retract`/`attest`
    recompute a record purely to re-derive its content digest, and neither should be synthesizing
    files or refusing to run because a record it is taking BACK was never reproducible.
    """
    artifacts = value.get("artifacts") or {}
    captured = str(artifacts.get("launch") or "")
    overlay = str(artifacts.get("overlay") or "")
    config = value.get("accepted_config") or {}
    flags, env = str(config.get("flags") or ""), str(config.get("env") or "")
    missing = _kernel_patches(a, kernels, files, workdir) if workdir else sum(
        1 for k in kernels if not k.get("patch"))
    if workdir:
        # `value.artifacts` is rebuilt by the caller from `files` after this returns, so adding to
        # `files` here is enough to make the synthesized script a first-class artifact of the
        # record rather than a loose file nobody indexed.
        have_patch = any(k.get("patch") for k in kernels)
        if not captured and not flags and not env and not have_patch and not overlay:
            raise SystemExit(
                "result carries no launch script, no accepted_config flags or env, no kernel "
                "patch and no overlay; refusing to record a run nobody can reproduce — a record "
                "that only says a number was once achieved cannot be acted on, and this store has "
                "no delete to take it back with")
        if not captured:
            path = os.path.join(workdir, "launch.sh")
            with open(path, "w") as handle:
                handle.write(_launch_text(a, result, value, kernels, overlay))
            os.chmod(path, 0o755)
            files["launch"] = ("launch.sh", path)
            captured, origin = "launch.sh", "synthesized"
        else:
            origin = "captured"
    else:
        origin = "captured" if captured else ""
    return {
        "launch": captured,
        "launch_origin": origin,
        "entry_point": "e2e_workflow/scripts/bench_e2e.sh",
        "server_args": flags,
        "env": env,
        "env_pairs": _env_pairs(env),
        "model": str(result.get("model_path")
                     or (result.get("upstream") or {}).get("model_path") or ""),
        "backend": identity_of(a)["framework"],
        "workload": dict(value.get("workload") or {}),
        "overlay": overlay,
        "kernels": [{"name": k.get("name") or "", "patch": k.get("patch") or "",
                     "kernel_canonical_id": k.get("kernel_canonical_id") or ""}
                    for k in kernels],
        "kernels_without_patch": missing,
        # The one field a reader can branch on: is what follows enough to re-run, or is it a lead.
        "complete": bool(captured) and not missing,
    }


# The verdicts that mean "this run bought nothing", whatever the ratio rounded to. `startsWith`
# semantics (not equality) because the Director spells its fallbacks with suffixes —
# `flagged_no_number_used_carried_ab`, `recovered_no_gain_*` — and a new suffix must not silently
# reopen the gate.
NO_WIN_VERDICTS = ("validated_no_win", "recovered_no_gain", "flagged_")


def win_gate(result: dict) -> str:
    """Why this result must NOT be recorded, or '' when it may be. ONE implementation.

    Two callers ask this question — e2e_workflow.js at the end of a live run, and run_e2e.py when it
    salvages a run whose workflow died before it got there — and they must answer it identically.
    They previously could not: only the JS had a gate at all, so every salvaged run was unwritable
    and every gate fix reached exactly half the write paths.

    The Director's VERDICT decides, not the raw ratio. A declared no-win can still carry a ratio
    above 1.0 when a low outlier in the base leg depresses its median: the 20260822 gemma-4-26B run
    read 1.0215x same-session while measuring 0.9453x against its provided baseline, and keying the
    write on the ratio alone minted that below-baseline number as the exact-id champion. A KB write
    is PERMANENT (the service exposes no DELETE), so a wrong record cannot be cleaned up, only
    outranked — hence a verdict the Director already computed to mean "no win" never writes.
    """
    speedup = finite_speedup(result.get("throughput_speedup"))
    final = finite_speedup(result.get("final_throughput_tok_s"))
    status = str(result.get("validation_status") or "")
    if not final:
        return "no final throughput measured"
    if speedup is None or speedup <= 1.0:
        return "no win to record (%sx)" % speedup
    if any(status.startswith(s) for s in NO_WIN_VERDICTS):
        return ("Director declared no win (%s) — the %sx same-session ratio is box-drift, "
                "not a gain" % (status, speedup))
    return ""


def cmd_write(a) -> dict:
    workdir = tempfile.mkdtemp(prefix="e2e_store_write_")
    try:
        return _write(a, workdir)
    finally:
        # Only after every rung has been published: the synthesized launch script and any patches
        # fetched from the kernel lane live here, and both planes read them at write time.
        shutil.rmtree(workdir, ignore_errors=True)


def _write(a, workdir: str) -> dict:
    try:
        with open(a.result, "r", errors="replace") as handle:
            result = json.load(handle)
    except (OSError, ValueError) as e:
        raise SystemExit("cannot read --result %s: %s" % (a.result, e))
    if not isinstance(result, dict):
        raise SystemExit("--result must be a JSON object")

    # Opt-in rather than always-on: `write` is also the backfill path, where a human has evidence
    # this result JSON cannot carry (a framework-layer win the sub-run self-judged no-win), and a
    # gate that could not be declined would make that correction unexpressible. Automated callers
    # pass it; a human deciding to override omits it, deliberately and visibly.
    if getattr(a, "require_win", False):
        why = win_gate(result)
        if why:
            return {"ok": True, "applied": False, "skipped": True, "why": why,
                    "session_id": "", "files": [], "rungs": []}

    record = build_record(a, result, workdir)
    ladder = ladder_of(a)
    sid = kbid.session_id(ladder[0][0], identity_of(a)["model"],
                          _content_digest(record["knowledge"]))
    out = {"applied": bool(a.apply), "session_id": sid, "speedup": record["speedup"],
           "throughput_tok_s": record["throughput"],
           "files": sorted(record["files"]), "rungs": []}
    # A rung ranks on its own metric (throughput on the exact rung, speedup on the coarser ones), so
    # each opens its own per-metric store; `publish` writes that one rung, all-or-none. All-or-none
    # runs at THIS loop level too: a rung we cannot open or write stops the ladder before a partial
    # one is published, and because the exact rung is written first, what lands is never a coarse
    # page that outranks the specific one it was meant to summarize.
    for cid, tier, metric, floor in ladder:
        rung = {"canonical_id": cid, "tier": tier, "metric": metric, "written": False,
                "promoted": False, "error": ""}
        if not a.apply:
            out["rungs"].append(rung)
            continue
        store, mirror, why = open_plane(a, metric, floor, create=True)
        if store is None:
            rung["error"] = why
            out["rungs"].append(rung)
            break
        # A re-bench of the same configuration lands on the SAME session id by design (see
        # _content_digest) and replaces the document wholesale. Without this, every re-measurement
        # would silently reset that record's validation history to zero — and it would look
        # perfectly healthy afterwards, because the new document is well-formed and simply says
        # nobody has ever tried this.
        knowledge = _carrying_ledger(store, cid, sid, record["knowledge"])
        rec = {"canonical_id": cid, "session_id": sid, "knowledge": knowledge}
        score_of = lambda r, m=metric: r["knowledge"].get(m)
        may_promote = not getattr(a, "no_promote", False)
        written, promoted, err = publish(store, [rec], record["files"], score_of,
                                         promote=may_promote)
        rung["written"] = bool(written)
        rung["promoted"] = bool(promoted)
        rung["error"] = err or why   # `both` with an unreachable service: recorded, not fatal
        if mirror is not None:
            # The mirror never gates the primary; its own failure is reported, not raised. It gets
            # its own ledger lookup because the two planes drift: a record attested on the remote
            # and re-written from a box that only ever wrote locally must not have the remote's
            # count overwritten by the local one's.
            mrec = dict(rec, knowledge=_carrying_ledger(mirror, cid, sid, record["knowledge"]))
            _mw, _mp, merr = publish(mirror, [mrec], record["files"], score_of,
                                     promote=may_promote)
            if merr and not rung["error"]:
                rung["error"] = merr
        out["rungs"].append(rung)
        if err:
            break
    out["ok"] = all(r["written"] for r in out["rungs"]) if a.apply else True
    return out


def _carrying_ledger(store, cid: str, sid: str, knowledge: dict) -> dict:
    """`knowledge` with any attestation ledger the store already holds for this session moved in.

    A store that cannot answer is treated as a store that holds nothing: failing the write over a
    lookup would turn a transient service blip into a lost measurement, and the worst case of
    guessing wrong here is a reset counter, not a wrong number.
    """
    try:
        previous = store.get_session(cid, sid)
    except Exception:
        previous = None
    fresh = dict(knowledge)
    fresh["value"] = carry_attestations(
        previous.get("value") if isinstance(previous, dict) else None, dict(fresh["value"]))
    return fresh


def _as_number(value):
    """A measurement off the command line, or None. Argparse hands these over as strings, and the
    caller is usually a shell line built by the workflow, so a stray unit or an empty flag must
    drop the evidence rather than abort an attestation that is otherwise perfectly recordable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return finite_speedup(float(value))
    except (TypeError, ValueError):
        return None


def cmd_attest(a) -> dict:
    """Count one attempt to actually RUN a stored record, at every rung it was written to.

    This is the other half of the read path. A resolve offers records; something downstream takes
    one to a box and finds out whether it still holds. Until now that finding-out evaporated, so
    the tenth reader of a record that has failed nine times saw exactly what the first reader saw.

    Applied to the whole ladder for the same reason retraction is: all three rungs share one
    session id, and counting only on the exact rung leaves the two coarse pages — the ones a
    reader on a DIFFERENT workload reads, which is most readers — quoting a stale ledger.

    Deliberately does not touch the ranking scalars or the champion. See kb/attest.py: one failure
    on one box is evidence, not a verdict, and burying a record on the strength of it would make
    this command too dangerous to run automatically, which would mean it never ran at all.
    """
    session_id = str(getattr(a, "session_id", "") or "").strip()
    if not session_id:
        raise SystemExit("attest needs --session-id (the id the write printed); there is nothing "
                         "to recompute it from, because the outcome being recorded is not in any "
                         "result JSON")
    evidence = {k: v for k, v in (
        ("measured_tok_s", _as_number(getattr(a, "measured_tok_s", None))),
        ("baseline_tok_s", _as_number(getattr(a, "baseline_tok_s", None))),
        ("parity", str(getattr(a, "parity", "") or "").strip()),
        ("note", str(getattr(a, "note", "") or "").strip()),
        ("workload", {k: str(getattr(a, k) or "") for k in ("tp", "isl", "osl", "conc")
                      if getattr(a, k, None)}),
    ) if v not in (None, "", {})}
    if evidence.get("measured_tok_s") and evidence.get("baseline_tok_s"):
        evidence["delta_pct"] = round(
            (evidence["measured_tok_s"] / evidence["baseline_tok_s"] - 1.0) * 100.0, 3)
    out = {"applied": bool(a.apply), "session_id": session_id, "outcome": a.outcome, "rungs": []}
    for cid, tier, metric, floor in ladder_of(a):
        store, mirror, why = open_plane(a, metric, floor)
        planes = [p for p in (store, mirror) if p is not None]
        if not planes:
            out["rungs"].append({"canonical_id": cid, "tier": tier, "error": why, "found": False})
            continue
        for plane in planes:
            report = attest_session(plane, cid, session_id, a.outcome,
                                    actor=str(a.measured_by or ""), evidence=evidence,
                                    apply=bool(a.apply))
            report.update({"tier": tier, "plane_note": why})
            out["rungs"].append(report)
    out["ok"] = attestation_ok(out["rungs"], a.apply)
    # Hoisted out of the per-rung reports because every rung holds the identical document, and a
    # caller deciding whether to open a curation ticket should not have to notice that.
    hints = [r.get("retire_hint") for r in out["rungs"] if r.get("retire_hint")]
    out["retire_hint"] = hints[0] if hints else ""
    return out


def cmd_retract(a) -> dict:
    """Take back one already-written record, at every rung it was written to.

    The session id is the SAME at all three rungs — `cmd_write` computes it once from the content
    digest and reuses it — so one identity plus one session id addresses the whole ladder, which is
    what makes a retraction possible at all without having kept a record of where things landed.

    Two ways to name the session, and the second exists because the first usually is not available:
    `--session-id` when the write's output was kept, or `--result` to recompute the digest from the
    same JSON the write was fed. The recompute is exact — the digest keys on config, kernel names,
    workload and direction, none of which a re-read changes — but it does require the SAME
    `--direction`, which is easy to forget and would silently address a session that does not exist.
    That case reports `found: false` per rung rather than inventing one.
    """
    session_id = str(getattr(a, "session_id", "") or "").strip()
    if not session_id:
        if not getattr(a, "result", ""):
            raise SystemExit("retract needs --session-id, or --result to recompute it")
        try:
            with open(a.result, "r", errors="replace") as handle:
                result = json.load(handle)
        except (OSError, ValueError) as e:
            raise SystemExit("cannot read --result %s: %s" % (a.result, e))
        record = build_record(a, result)
        session_id = kbid.session_id(ladder_of(a)[0][0], identity_of(a)["model"],
                                     _content_digest(record["knowledge"]))
    out = {"applied": bool(a.apply), "session_id": session_id, "reason": a.reason, "rungs": []}
    for cid, tier, metric, floor in ladder_of(a):
        store, mirror, why = open_plane(a, metric, floor)
        planes = [p for p in (store, mirror) if p is not None]
        if not planes:
            out["rungs"].append({"canonical_id": cid, "tier": tier, "error": why, "found": False})
            continue
        for plane in planes:
            # Both scalars are zeroed on every rung, not just the one this rung ranks on. The
            # document is identical at all three rungs by construction, so a rewrite that zeroed
            # only the local metric would leave the record still ranked on the other two.
            report = retract_session(plane, cid, session_id, a.reason, metric,
                                     extra_metrics=(THROUGHPUT_METRIC, SPEEDUP_METRIC),
                                     actor=str(a.measured_by or ""), scan=int(a.scan),
                                     apply=bool(a.apply))
            report.update({"tier": tier, "metric": metric, "plane_note": why})
            out["rungs"].append(report)
    out["ok"] = retraction_ok(out["rungs"], a.apply)
    return out


def _content_digest(knowledge: dict) -> str:
    """Dedup key: the CONFIG, not the measurement.

    Re-benchmarking one config must land on the same session id so `mode="replace"` updates that
    record instead of accumulating a page full of near-identical entries that all outrank each
    other by noise. So the throughput numbers and the timestamp are deliberately excluded — two
    runs of the same config ARE the same candidate, and the later one wins.
    """
    value = knowledge.get("value") or {}
    payload = json.dumps({"config": value.get("accepted_config") or {},
                          "kernels": sorted(str(k.get("name") or "") for k in
                                            (value.get("accepted_kernels") or [])
                                            if isinstance(k, dict)),
                          "workload": value.get("workload") or {},
                          "direction": value.get("direction") or ""},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- cli -----------------------------------------------------------------------------------------


def _identity_args(p):
    p.add_argument("--model", required=True, help="model name, e.g. Qwen3-397B")
    p.add_argument("--gfx", default="", help="gfx target, e.g. gfx950")
    p.add_argument("--framework", default="", help="serving stack: vllm | sglang")
    p.add_argument("--framework-version", default="", help="serving stack version, e.g. 0.26.0")
    p.add_argument("--precision", default="", help="e.g. mxfp8, fp8, bf16")
    p.add_argument("--rocm-version", default="",
                   help="ROCm of the container, used to address the accepted kernels' own records")
    p.add_argument("--tp", default=None, help="tensor parallel degree")
    p.add_argument("--isl", default=None, help="input sequence length")
    p.add_argument("--osl", default=None, help="output sequence length")
    p.add_argument("--conc", default=None, help="concurrency")


def _state_args(p):
    """Overrides for what _record_state would otherwise derive. Shared with `retract` because it
    recomputes the content digest, and the digest is computed off a full build_record()."""
    p.add_argument("--validated", default="auto", choices=("auto", "true", "false"),
                   help="auto = validated_win AND parity pass; override when you know better")
    p.add_argument("--validation-basis", default="auto",
                   choices=("auto", "hot_ab", "cold_gate", "unverified"),
                   help="which gate produced the number (auto: hot_ab if a status was recorded)")
    p.add_argument("--parity", default="", help="pass | fail | n/a; defaults to result.output_parity")


def _plane_args(p):
    p.add_argument("--plane", choices=("local", "remote", "both"), default="local")
    p.add_argument("--store", default="", help="on-disk store root (plane local|both)")
    p.add_argument("--scan", type=int, default=DEFAULT_SCAN,
                   help="candidates hydrated per rung before curation")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("identity", help="print the ladder this deployment reads and writes")
    _identity_args(q)

    q = sub.add_parser("resolve", help="offer prior runs for this deployment")
    _identity_args(q)
    _plane_args(q)
    q.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    q.add_argument("--min-speedup", type=float, default=0.0)
    q.add_argument("--sort-by", choices=tuple(SORT_METRICS), default=DEFAULT_SORT_BY,
                   help="how to order the offer on EVERY rung (default: absolute throughput, "
                        "high to low). The champion metric per rung is unaffected.")
    q.add_argument("--refs-dir", default="", help="write prose references here")
    q.add_argument("--cache-dir", default="", help="materialize artifact bundles here")
    q.add_argument("--identity-out", default="",
                   help="also dump the resolved dimensions here, so a later writer (including "
                        "run_e2e.py salvaging a workflow that died) can address the same pages")
    q.add_argument("--identity-plane", default="",
                   help="the plane to record in --identity-out (the RUN's write plane, which for a "
                        "`both` run differs from this read's remote-first plane)")

    q = sub.add_parser("write", help="record one run at every rung")
    _identity_args(q)
    _plane_args(q)
    q.add_argument("--result", required=True, help="JSON from the workflow's report/validate step")
    q.add_argument("--direction", default="", help="what this run DID, for the shortlist collapse")
    q.add_argument("--measured-by", default="", help="who/what produced the number")
    q.add_argument("--file", action="append", default=[], help="extra artifact to attach")
    q.add_argument("--kernel-store", default="",
                   help="kernel lane's on-disk store root; when given, patches this run no longer "
                        "has on disk are fetched from there by kernel_canonical_id so the record "
                        "stays reproducible. Off by default: it is real I/O on the write path.")
    _state_args(q)
    q.add_argument("--require-win", action="store_true",
                   help="refuse to write a result the Director declared a no-win, whatever its "
                        "raw ratio (see win_gate). Every automated caller passes this; a human "
                        "backfilling a win the run itself mis-judged omits it on purpose.")
    q.add_argument("--no-promote", action="store_true",
                   help="record the measurement but leave the champion pointer alone. For numbers "
                        "the writer knows are provisional — a run salvaged from disk artifacts "
                        "with no final Validate behind it.")
    q.add_argument("--apply", action="store_true", help="actually write; default is a dry run")

    q = sub.add_parser("attest", help="count one attempt to RUN a stored record: validated | "
                                      "failed | not_reproduced | inapplicable. Moves no score, "
                                      "no champion.")
    _identity_args(q)
    _plane_args(q)
    q.add_argument("--session-id", required=True, help="the session that was tried")
    q.add_argument("--outcome", required=True, choices=OUTCOMES,
                   help="validated = reproduced a win; failed = ran but did not win; "
                        "not_reproduced = could not be made to run at all; "
                        "inapplicable = could not be applied to THIS box's baseline (a knob the "
                        "record pins is already pinned to something else here) — counted, but "
                        "kept out of the retire arithmetic, because it judges the pairing and "
                        "not the record")
    q.add_argument("--measured-tok-s", default=None, help="what it did here, for the history entry")
    q.add_argument("--baseline-tok-s", default=None, help="what this box does without it")
    q.add_argument("--parity", default="", help="pass | fail | n/a on this box")
    q.add_argument("--note", default="", help="one line a future reader can act on")
    q.add_argument("--measured-by", default="", help="who tried it")
    q.add_argument("--apply", action="store_true", help="actually record it; default is a dry run")

    q = sub.add_parser("retract", help="take back a written record (rewrite, since there is no "
                                       "delete): retained=false, scores zeroed, champion re-pointed")
    _identity_args(q)
    _plane_args(q)
    q.add_argument("--session-id", default="", help="the session to retract, from the write output")
    q.add_argument("--result", default="", help="recompute the session id from the SAME JSON and "
                                                "--direction the write was given")
    q.add_argument("--direction", default="", help="must match the write, or the id will not match")
    q.add_argument("--reason", required=True,
                   help="why this record is wrong; it is all a future reader has to judge by")
    q.add_argument("--measured-by", default="", help="who is retracting it")
    _state_args(q)
    q.add_argument("--apply", action="store_true", help="actually rewrite; default is a dry run")

    a = p.parse_args(argv)
    if a.command == "identity":
        result = {"identity": identity_of(a),
                  "ladder": [{"canonical_id": c, "tier": t, "ranked_by": m, "promote_floor": f}
                             for c, t, m, f in ladder_of(a)]}
    elif a.command == "resolve":
        result = cmd_resolve(a)
    elif a.command == "retract":
        result = cmd_retract(a)
    elif a.command == "attest":
        result = cmd_attest(a)
    else:
        result = cmd_write(a)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
