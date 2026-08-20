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

RANKING METRIC DIFFERS BY RUNG, and getting this wrong is silent. On the exact rung the workload is
identical by construction, so the honest ranking is absolute `throughput_tok_s` — ranking it by
speedup would put a run that started from a badly configured baseline above a run that was already
fast and got faster. On the coarser rungs the workloads differ, absolute numbers are not comparable
at all, and `speedup` is the only thing that means anything. Both scalars are written flat at the
top of every document (the service's `sessions/top?metric=` reads a top-level scalar and rejects a
nested path), so each rung can rank on whichever it needs without a second write.

No delete exists on the service. Every `--apply` is permanent.
"""

import argparse
import hashlib
import json
import os
import sys
import time

# The shared KB plane lives at the repo root as the `kb` package, not beside this file. Executed as
# a CLI from an arbitrary cwd, so the root is derived from __file__ and never from the environment.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb import identity as kbid                                             # noqa: E402
from kb.curate import collapse_by_direction                                 # noqa: E402
from kb.ladder import publish                                               # noqa: E402
from kb.plane import open_plane                                             # noqa: E402
from kb.retract import is_retired, retract_session, retraction_ok           # noqa: E402
from kb.store_local import KBStoreError, finite_speedup                     # noqa: E402

SCHEMA = "geak.e2e.v1"
THROUGHPUT_METRIC = "throughput_tok_s"      # ranks the exact-workload rung
SPEEDUP_METRIC = "speedup"                  # ranks every coarser rung
DEFAULT_TOP_N = 3
DEFAULT_SCAN = 25

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


def _view(candidate, cid: str, tier: str, metric: str) -> dict:
    """One offered record, flattened to what a Director prompt actually needs."""
    knowledge = candidate.knowledge if isinstance(candidate.knowledge, dict) else {}
    value = knowledge.get("value") if isinstance(knowledge.get("value"), dict) else {}
    workload = value.get("workload") if isinstance(value.get("workload"), dict) else {}
    return {
        "session_id": candidate.session_id,
        "canonical_id": cid,
        "match_tier": tier,
        "ranked_by": metric,
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
        "is_champion": bool(candidate.is_champion),
    }


def cmd_resolve(a) -> dict:
    ladder = ladder_of(a)
    # Echo the plane back. A caller that tries the service and falls back to disk otherwise cannot
    # tell from the output which one answered — the ladder, the ranking and the shapes are identical
    # — and "where did this candidate come from" is the first question asked when one turns out to
    # be wrong. `dict(out, ...)` carries it onto every return path below.
    out = {"tried": [c for c, _t, _m, _f in ladder], "canonical_id": ladder[0][0],
           "match_tier": "", "ranked_by": "", "candidates": [], "read_reason": "",
           "plane": str(getattr(a, "plane", "local") or "local"), "curation": {}}
    last_why = ""
    for cid, tier, metric, floor in ladder:
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
        curation = {"scanned": len(found), "retired": len(found) - len(kept)}
        # top_n=len(kept): e2e filters min_speedup AFTER collapse and slices to top_n below, so
        # collapse must not pre-slice. It consumes only the per-idea best, not the alternates.
        views, _alternates, collapsed = collapse_by_direction(
            [_view(c, cid, tier, metric) for c in kept],
            lambda v: v["direction"], lambda v: v["session_id"], len(kept))
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
                    candidates=views, read_reason="read",
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


def _render_reference(refs_dir: str, cid: str, tier: str, views) -> str:
    """Mirror the offer into prose the Director can read, or return "" and let the read stand."""
    try:
        os.makedirs(refs_dir, exist_ok=True)
        key = hashlib.sha1(("|".join(v["session_id"] for v in views)).encode()).hexdigest()[:7]
        path = os.path.join(refs_dir, "e2e_reference_%s.md" % key)
        lines = ["# e2e warm start — `%s`" % cid, "",
                 "Match tier `%s`, ranked by `%s`." % (tier, views[0]["ranked_by"]), ""]
        if tier != "exact":
            lines += ["> These were measured on a DIFFERENT workload point than the one requested. "
                      "Treat the configs as candidates and the numbers as non-comparable — do not "
                      "quote them as this deployment's throughput.", ""]
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
                "- accepted kernels: %s" % (_kernel_line(v["accepted_kernels"]) or "none"),
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


def build_record(a, result: dict) -> dict:
    """One run's knowledge document, identical at every rung.

    Both ranking scalars sit flat at the top level because that is the only shape
    `sessions/top?metric=` can read. Everything else lives under `value`, including the dimensions
    that are already in the canonical id — a record that cannot say what it is once detached from
    its address is not auditable.
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
    if files:
        value["artifacts"] = {k: v[0] for k, v in files.items()}
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


def _artifact_files(a, result: dict) -> dict:
    """{role: (stored_name, local_path)} for the run outputs that actually exist on disk.

    A path the result names but the filesystem does not have is dropped here rather than at upload
    time, so `value.artifacts` never promises a file the record does not carry — the kernel lane's
    materialize() now treats that promise as a hard error, and it should.
    """
    found = {}
    for role, field, stored in _ARTIFACT_KEYS:
        path = str(result.get(field) or "")
        if path and os.path.isfile(path):
            found[role] = (stored, path)
    for extra in (getattr(a, "file", None) or []):   # retract recomputes a record but takes no --file
        path = str(extra)
        if os.path.isfile(path):
            found["file:" + os.path.basename(path)] = (os.path.basename(path), path)
    return found


def cmd_write(a) -> dict:
    try:
        with open(a.result, "r", errors="replace") as handle:
            result = json.load(handle)
    except (OSError, ValueError) as e:
        raise SystemExit("cannot read --result %s: %s" % (a.result, e))
    if not isinstance(result, dict):
        raise SystemExit("--result must be a JSON object")

    record = build_record(a, result)
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
        rec = {"canonical_id": cid, "session_id": sid, "knowledge": record["knowledge"]}
        score_of = lambda r, m=metric: r["knowledge"].get(m)
        written, promoted, err = publish(store, [rec], record["files"], score_of)
        rung["written"] = bool(written)
        rung["promoted"] = bool(promoted)
        rung["error"] = err or why   # `both` with an unreachable service: recorded, not fatal
        if mirror is not None:
            # The mirror never gates the primary; its own failure is reported, not raised.
            _mw, _mp, merr = publish(mirror, [rec], record["files"], score_of)
            if merr and not rung["error"]:
                rung["error"] = merr
        out["rungs"].append(rung)
        if err:
            break
    out["ok"] = all(r["written"] for r in out["rungs"]) if a.apply else True
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
    q.add_argument("--refs-dir", default="", help="write prose references here")
    q.add_argument("--cache-dir", default="", help="materialize artifact bundles here")

    q = sub.add_parser("write", help="record one run at every rung")
    _identity_args(q)
    _plane_args(q)
    q.add_argument("--result", required=True, help="JSON from the workflow's report/validate step")
    q.add_argument("--direction", default="", help="what this run DID, for the shortlist collapse")
    q.add_argument("--measured-by", default="", help="who/what produced the number")
    q.add_argument("--file", action="append", default=[], help="extra artifact to attach")
    _state_args(q)
    q.add_argument("--apply", action="store_true", help="actually write; default is a dry run")

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
    else:
        result = cmd_write(a)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
