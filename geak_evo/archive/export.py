"""Deterministic, browsable program records derived only from the research journal.

The journal stays the truth; this tree is a rebuildable view of it. Output bytes
depend only on the journal (sorted keys, no export-time clock), so exporting the
same journal twice is byte-identical. Export each journal snapshot to a new
directory; an identical existing tree is verified and reused. The view includes
saved strategy fields, current research notes and linked claim records, but never
copies conversations or bulk reports. These remain attributed agent statements.

    programs/<seq>-<id12>/{program.json, source/}   ideas/<slug>-<sha8>.json
    attempts/<hash>.json   index.json   index.csv   measurements.csv   lineage.json   timeline.json   RESUME.json
    EXPORT.json  (written last; journal head + sha256 of every other file)
"""

import argparse
import csv
import io
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .._util import sha256_hex
from .claims import ClaimLog
from .records import ArchiveError, read_record
from .store import ResearchStore, StateError, _read
from .trials import trial_history

SCHEMA = "geak-evo-program-records-v1"
IDEA_FIELDS = ("idea_id", "statement", "status", "next_test", "applicability", "competing_explanations")
EVAL_FIELDS = ("evaluation_id", "settled", "independent", "correctness_observed", "correct", "eligible",
               "provisional", "timing_stable", "physical_devices", "times_ms", "relative_spreads")
OBSERVATION_ARTIFACTS = ("native_evaluation.json", "paired_feedback.json", "native/job.json",
                         "native/source.json", "native/prepared.json", "native/operator_task.py")
IDEA_BASIS = "engineer idea card: a model-authored claim, not a measurement; measured data is in programs/"
TIMELINE_BASIS = ("the host's recorded recommendation at each journal sequence; not recomputed from timings. "
                  "checkpoint_journal_time = journal commit; accepted_monotonic_ns = host monotonic acceptance")
TIME_BASIS = "journal commit time when first recorded: an upper bound on creation, not discovery or measurement time"
SCOPE = ("rebuildable view of journal-reachable programs and retained trial inventories; "
         "uncommitted objects and unsettled episodes not imported by the host are outside this view. "
         "Exact source closure is portable; historical external references and evaluator qualification are not.")


def _enc(value):
    return (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _refs(value):
    """Follow portable evidence references, never historical absolute file paths."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if isinstance(item.get("evidence_id"), str) and isinstance(item.get("artifact"), str):
                yield item
            else:
                pending.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            pending.extend(reversed(item))


class _Journal:
    """Hash-checked, cached reads of the replayed journal and its immutable objects."""

    def __init__(self, store, frozen_cases=None):
        self.store, self.cache = store, {}
        if frozen_cases is not None and (not isinstance(frozen_cases, dict) or any(
                not isinstance(job, str) or not isinstance(cases, list) or
                any(not isinstance(case, str) or not case for case in cases)
                for job, cases in frozen_cases.items())):
            raise ArchiveError("frozen_cases must map job IDs to lists of registered case IDs")
        self.frozen_cases = frozen_cases
        with store._lock():
            index = store._replay()  # Validates the chain, bindings and anchors.
            paths = sorted((store.root / "records").glob("*.json"))[:index["sequence"]]
            raw = [_read(path.parent, path.name)[0] for path in paths]
        self.records = [json.loads(data)["state"] for data in raw]
        self.digests = [sha256_hex(data) for data in raw]
        if not raw or self.digests[-1] != index["head"]:
            raise StateError("export requires a nonempty committed journal")

    def obj(self, identity, kind):
        if (identity, kind) not in self.cache:
            if kind == "candidate":
                value = self.store.get_candidate(identity)
            elif kind == "source":
                value = self.store.source(identity)
            elif kind == "evidence":
                value = self.store._bound_evidence(identity)
            else:
                value = self.store._object(identity, kind)
            self.cache[identity, kind] = value
        return self.cache[identity, kind]

    def artifact(self, evidence_id, name):
        blob = self.obj(evidence_id, "evidence")["artifacts"].get(name)
        return None if blob is None else json.loads(self.store._bytes(blob))

    def ref(self, ref):
        return self.artifact(ref["evidence_id"], ref["artifact"]) if ref else None


def _evaluation(journal, evidence_id):
    """(program id, summary) of a native evaluation receipt, or None."""
    native = journal.artifact(evidence_id, "native_evaluation.json")
    if native is None:
        return None
    envelope = journal.obj(evidence_id, "evidence")
    binding = native["binding"]
    if (binding.get("candidate_id") != envelope["candidate_id"] or
            any(binding.get(key) != value for key, value in envelope["binding"].items())):
        raise StateError("native evaluation binding differs from archived candidate")
    row = {key: native.get(key) for key in EVAL_FIELDS}
    row.update(evidence_id=evidence_id, binding=binding, kind="native_evaluation",
               measurement_epoch_id=binding["measurement_epoch_id"],
               device=((journal.artifact(evidence_id, "native/job.json") or {}).get("workload") or {}).get("device"),
               errors=sorted({str(error) for op in native.get("operations", [])
                              for error in (op.get("assessment") or {}).get("errors") or []}))
    # Keep the existing comparison address without deriving or endorsing a ratio.
    if journal.artifact(evidence_id, "paired_feedback.json") is not None:
        row["paired_ref"] = {"evidence_id": evidence_id, "artifact": "paired_feedback.json"}
    return native["binding"]["candidate_id"], row


def _attempt(journal, row, superseded):
    packet = journal.ref(row["packet_ref"])
    proposal, decision = packet["proposal"], packet["decision"]
    candidate = journal.obj(row["candidate_id"], "candidate") if row.get("candidate_id") else None
    parent = journal.obj(proposal["parent_candidate_id"], "candidate")
    collection = row.get("collection_ref")
    inventory = journal.artifact(collection["evidence_id"], "native_trial_sources.json") if collection else None
    roles = (journal.artifact(collection["evidence_id"], "native_source_roles.json") or {}) if collection else {}
    harvest_ref = None
    if collection:
        artifacts = journal.obj(collection["evidence_id"], "evidence")["artifacts"]
        name = next((name for name in ("native_idea_harvest.json", "archive/manifest.json") if name in artifacts), None)
        if name:
            harvest_ref = {"evidence_id": collection["evidence_id"], "artifact": name}
    harvest = journal.ref(harvest_ref) or {}
    explanation = harvest.get("explanation") or {}
    strategy = explanation.get("strategy")
    has_strategy = isinstance(strategy, str) and bool(strategy.strip())
    return {"attempt_id": row["attempt_id"], "settled": row["settled"], "superseded": superseded,
            "parent_id": proposal["parent_candidate_id"], "program_id": row["candidate_id"],
            "source_roles": roles,
            "new_source": candidate is not None and candidate["source_id"] != parent["source_id"],
            "retained_trial_candidate_ids": sorted({trial["candidate_id"] for trial in (inventory or {}).get("trials", [])
                                                     if trial.get("candidate_id")}),
            **{key: proposal[key] for key in ("direction_id", "island_id", "engineer_id")},
            **{key: decision.get(key) for key in ("advice_id", "comparison", "selection")},
            "transport_complete": (journal.ref(row.get("transport_ref")) or {}).get("transport_complete"),
            "question": proposal.get("question"),
            "implemented_idea": {
                "strategy": strategy[:4096] if has_strategy else None,
                "strategy_truncated": has_strategy and len(strategy) > 4096,
                "missing_reason": None if has_strategy else "No saved strategy in this attempt's harvest.",
                "ref": harvest_ref, "field": "explanation.strategy",
                "reported_source_id": explanation.get("source_id"),
                "selected_source_id": (harvest.get("selected") or {}).get("source_id"),
                **{key: explanation.get(key) for key in ("binding_valid", "investigation_complete", "explanation_status", "errors")},
                "basis": "Saved engineer statement for the reported source; not a description of every trial or proof of the mechanism."},
            **{key: row.get(key) for key in ("packet_ref", "transport_ref", "collection_ref")}}


def _slug(text):
    return sha256_hex(text.encode())


def _research_files(journal, directions, evo):
    """Project only current journal notes and the exact committed claim chain."""
    files, direction_files, claim_files = {}, {}, {}
    for direction_id, direction in sorted(directions.items()):
        name = direction_files[direction_id] = "directions/%s.json" % _slug(direction_id)
        sequence = next(i for i in range(len(journal.records), 0, -1)
                        if journal.records[i - 1]["direction_id"] == direction_id)
        note = direction["note"]
        files[name] = _enc({"direction_id": direction_id, "revision": direction["revision"],
            "program_id": direction["tip"], "source_id": journal.obj(direction["tip"], "candidate")["source_id"],
            "attempt_id": note.get("evolution_operation_id"),
            "journal_sequence": sequence, "journal_record_sha256": journal.digests[sequence - 1],
            "research_status": note.get("research_status"),
            "note": {key: note.get(key) for key in ("claim", "scope", "next_test", "evidence_uptake")},
            "refs": {key: note.get(key) for key in ("archive_card_ref", "research_update_ref", "latest_transport_ref")},
            "basis": "Current saved direction review; interpretation, not validation. It describes this direction/tip, not every earlier trial."})
    head = evo.get("claim_head")
    rows = []
    # Read from the frozen journal's head, never from the store's current head.
    # Keep exact events, including corrections; do not reclassify quantities.
    for ref, event in reversed(ClaimLog(journal.store, head)._read_events()):
        name = "claims/%s/%s" % (ref["evidence_id"], ref["artifact"])
        _, files[name] = journal.store.evidence_artifact(ref["evidence_id"], ref["artifact"])
        row = {"ref": ref, "file": name, "candidate_id": event["candidate_id"], "kind": event["kind"]}
        rows.append(row)
        claim_files.setdefault(event["candidate_id"], []).append(name)
    files["claims.json"] = _enc({"head": head, "events": rows,
        "missing_reason": None if head else "No ClaimLog head saved in this campaign snapshot.",
        "basis": "Exact saved claims and corrections in chain order; producer authority and scientific truth are not re-evaluated by this export."})
    return files, direction_files, claim_files


def _scan(journal):
    """Programs in first-seen journal order (ancestors first), evidence, ideas and directions."""
    programs, evidence, ideas, directions, evo, seed, count = {}, {}, {}, {}, {"history": []}, None, [0]
    trials = {}

    def visit(identity, sequence):  # Journal order keeps the recursion shallow.
        if identity in programs:
            return
        value = programs[identity] = journal.obj(identity, "candidate")
        for ancestor in value["parents"] + value["donors"]:
            visit(ancestor, sequence)
        count[0] += 1
        programs[identity] = {"program_id": identity, "ordinal": count[0], "first_journal_sequence": sequence,
            "first_recorded_journal_time": journal.records[sequence - 1].get("updated_at"), "time_basis": TIME_BASIS,
            **{key: value[key] for key in ("source_id", "job_id", "measurement_epoch_id", "parents", "donors", "contribution_claims")},
            "directions": [], "idea_ids": [], "evaluations": [], "not_launched": [],
            "authors": [], "produced_by_attempts": [], "attempts_from_here": []}

    for sequence, state in enumerate(journal.records, 1):
        found = [state["tip"]] + ([state["anchor"]["candidate_id"]] if state["anchor"] else [])
        refs = list(state["evidence"]) + ([state["anchor"]["evidence_id"]] if state["anchor"] else [])
        pending, seen = list(_refs(state)), set()
        while pending:
            ref = pending.pop(0)
            key = (ref["evidence_id"], ref["artifact"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref["evidence_id"])
            envelope = journal.obj(ref["evidence_id"], "evidence")
            blob = envelope["artifacts"].get(ref["artifact"])
            if blob is None:
                raise StateError("referenced evidence artifact is absent")
            raw = journal.store._bytes(blob)
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(value, dict) and value.get("schema") == "geak-member-recommendation-v1":
                selected = (value.get("choice") or {}).get("candidate_id")
                if selected:
                    found.append(selected)
            pending.extend(_refs(value))
        if "evolution" in state["note"]:
            evo = state["note"]["evolution"]
            rows = evo["history"] + evo.get("superseded_attempts", [])
            found += [row["candidate_id"] for row in rows if row.get("candidate_id")]
            refs += [row[key]["evidence_id"] for row in rows for key in ("packet_ref", "collection_ref", "transport_ref") if row.get(key)]
            seed = state["tip"]
        else:
            directions[state["direction_id"]] = dict(state, idea_id=None)
        card_ref = state["note"].get("archive_card_ref")
        if card_ref and "evolution" not in state["note"]:
            card = read_record(journal.store, card_ref, frozen_cases=journal.frozen_cases)
            idea = card["idea"]
            entry = ideas.setdefault(idea["idea_id"], {"revisions": []})
            entry.update({key: idea[key] for key in IDEA_FIELDS if key in idea})
            entry["revisions"].append({"journal_sequence": sequence, "direction_id": state["direction_id"],
                "program_id": state["tip"], "status": idea.get("status"), "card_ref": card_ref,
                "author": card.get("author"),
                "islands": sorted({(journal.ref(ref) or {}).get("island_id") or "unassigned"
                                   for ref in card.get("membership_refs", [])})})
            found += [card["last_faithful"]["candidate_id"]] if card.get("last_faithful") else []
            directions[state["direction_id"]]["idea_id"] = idea["idea_id"]
        refs = list(dict.fromkeys(refs))
        trial_groups = {}
        for evidence_id in refs:
            envelope = journal.obj(evidence_id, "evidence")
            inventory_blob = envelope["artifacts"].get("native_trial_sources.json")
            for link in journal.artifact(evidence_id, "native_evidence_index.json") or []:
                raw_id = link["evidence_ref"]["evidence_id"]
                if raw_id not in refs:
                    refs.append(raw_id)
                if inventory_blob and journal.obj(raw_id, "evidence")["artifacts"].get("native_trial_sources.json") == inventory_blob:
                    trial_groups[evidence_id] = raw_id
        for evidence_id in refs:  # Grows: retained raw evidence and paired baselines.
            envelope = journal.obj(evidence_id, "evidence")
            found.append(envelope["candidate_id"])
            roles = journal.artifact(evidence_id, "native_source_roles.json") or {}
            found += [roles[role] for role in sorted(roles)]
            inventory = journal.artifact(evidence_id, "native_trial_sources.json") or {}
            harvest_name = next((name for name in ("native_idea_harvest.json", "archive/manifest.json")
                                 if name in envelope["artifacts"]), None)
            harvest = journal.artifact(evidence_id, harvest_name) if harvest_name else None
            for trial in inventory.get("trials", []):
                if trial.get("candidate_id"):
                    candidate = journal.obj(trial["candidate_id"], "candidate")
                    if (candidate["source_id"] != trial["source_id"] or any(
                            candidate[key] != envelope["binding"][key] for key in ("job_id", "measurement_epoch_id"))):
                        raise StateError("trial source binding differs from candidate")
                    found.append(trial["candidate_id"])
                group = trial_groups.get(evidence_id, evidence_id)
                key = (group, sha256_hex(_enc(trial)))
                retained = trials.setdefault(key, dict(trial, evidence_id=evidence_id, investigation_evidence_id=group,
                    evidence_ids=[], first_journal_sequence=sequence, job_id=envelope["binding"]["job_id"],
                    measurement_epoch_id=envelope["binding"]["measurement_epoch_id"]))
                if evidence_id not in retained["evidence_ids"]:
                    retained["evidence_ids"].append(evidence_id)
                if harvest is not None:
                    raw = harvest["trials"][trial["trial_index"]]
                    if raw.get("archive_ref") != trial.get("feedback_ref"):
                        raise StateError("trial inventory differs from retained harvest")
                    outcome = {"error": raw.get("error"), "status": raw.get("status"),
                        "source_archive_error": raw.get("source_archive_error"),
                        "verified_status": (raw.get("verified") or {}).get("status"),
                        "diagnostic_status": (raw.get("diagnostic") or {}).get("status")}
                    if retained.get("outcome", outcome) != outcome:
                        raise StateError("conflicting retained trial outcomes")
                    retained["outcome"] = outcome
                    retained["raw_trial_ref"] = {"evidence_id": evidence_id, "artifact": harvest_name}
            for ref in journal.artifact(evidence_id, "native_evidence_index.json") or []:
                identity = ref["evidence_ref"]["evidence_id"]
                if identity not in refs:
                    refs.append(identity)
            baseline = (journal.artifact(evidence_id, "paired_feedback.json") or {}).get("baseline_evidence_ref")
            if baseline and baseline["evidence_id"] not in refs:
                refs.append(baseline["evidence_id"])
                found.append(journal.obj(baseline["evidence_id"], "evidence")["candidate_id"])
        for identity in found:
            visit(identity, sequence)
        evidence.update({e: sequence for e in refs if e not in evidence})
    journal.evidence_sequences = evidence
    return programs, list(evidence), ideas, directions, evo, seed, list(trials.values())


def _status(row):
    if row.get("kind") == "native_trial":
        return row["status"]
    if row.get("settled") is not True or row.get("correctness_observed") is not True:
        return "incomplete"
    if row.get("correct") is False:
        return "failed"
    if row.get("correct") is not True:
        return "incomplete"
    return "eligible" if row.get("eligible") is True else "correct_not_eligible"


def _trial_evaluations(journal, evidence):
    history = trial_history(journal.store, evidence)
    for row in history["trials"]:
        row["first_journal_sequence"] = min(journal.evidence_sequences[identity]
                                            for identity in row["evidence_ids"])
    return history


def build(store, *, frozen_cases=None):
    """Return the journal, ordered programs and {relative path: bytes} except sources and EXPORT.json."""
    journal = _Journal(store, frozen_cases)
    programs, evidence, ideas, directions, evo, seed, trials = _scan(journal)
    history = _trial_evaluations(journal, evidence)
    trial_evaluations = history["trials"]
    for evidence_id in evidence:
        result = _evaluation(journal, evidence_id)
        if result and result[0] in programs and result[1] not in programs[result[0]]["evaluations"]:
            result[1]["first_journal_sequence"] = journal.evidence_sequences[evidence_id]
            programs[result[0]]["evaluations"].append(result[1])
        reason = (journal.artifact(evidence_id, "native_not_launched.json") or {}).get("reason")
        owner = programs.get(journal.obj(evidence_id, "evidence")["candidate_id"])
        if reason and owner and reason not in owner["not_launched"]:
            owner["not_launched"].append(reason)
    for trial in trial_evaluations:
        if trial.get("candidate_id") in programs:
            programs[trial["candidate_id"]]["evaluations"].append(trial)
    attempts, planner_rows = {}, 0
    for superseded, rows in ((False, evo["history"]), (True, evo.get("superseded_attempts", []))):
        for row in rows:
            if not row.get("attempt_id") or not row.get("packet_ref"):
                planner_rows += 1  # Failed/rejected plans and reviews produce no program.
                continue
            key = json.dumps([row["attempt_id"], superseded])
            attempt = attempts[key] = _attempt(journal, row, superseded)
            produced = attempt["retained_trial_candidate_ids"] + ([attempt["program_id"]] if attempt["new_source"] else [])
            for identity in dict.fromkeys(produced):
                if identity in programs:
                    programs[identity]["produced_by_attempts"].append(row["attempt_id"])
                    if attempt["direction_id"] not in programs[identity]["directions"]:
                        programs[identity]["directions"].append(attempt["direction_id"])
                    programs[identity]["authors"].append({"engineer_id": attempt["engineer_id"],
                        "attempt_id": attempt["attempt_id"], "attempt_ref": attempt["transport_ref"]})
            if attempt["parent_id"] in programs:
                programs[attempt["parent_id"]]["attempts_from_here"].append(row["attempt_id"])
    by_attempt = {row["attempt_id"]: row for row in attempts.values()}
    for row in trial_evaluations:
        attempt = by_attempt.get(row["attempt_id"], {})
        row.update({key: attempt.get(key) for key in ("direction_id", "engineer_id")})
        row["idea_id"] = directions.get(row["direction_id"], {}).get("idea_id")
    for idea_id, entry in ideas.items():
        idea_directions = {revision["direction_id"] for revision in entry["revisions"]}
        for row in programs.values():
            if idea_directions.intersection(row["directions"]) and idea_id not in row["idea_ids"]:
                row["idea_ids"].append(idea_id)
        for revision in entry["revisions"]:
            row = programs[revision["program_id"]]
            if revision.get("author") and revision["author"] not in row["authors"]:
                row["authors"].append(revision["author"])
            for key, value in (("directions", revision["direction_id"]), ("idea_ids", idea_id)):
                row[key] += [] if value in row[key] else [value]
    files, direction_files, claim_files = _research_files(journal, directions, evo)
    attempt_files = {key: "attempts/%s.json" % _slug(key) for key in attempts}
    order = sorted(programs, key=lambda identity: programs[identity]["ordinal"])
    anchors = {state["anchor"]["candidate_id"]: state["anchor"] for state in directions.values() if state["anchor"]}
    index, edges = [], []
    for identity in order:
        row = programs[identity]
        row["evaluations"].sort(key=lambda evaluation: (evaluation["first_journal_sequence"], evaluation["evidence_id"]))
        row["folder"] = folder = "programs/%06d-%s" % (row["ordinal"], identity[:12])
        sequence = max((e["first_journal_sequence"] for e in row["evaluations"]), default=None)
        latest = [e for e in row["evaluations"] if e["first_journal_sequence"] == sequence]
        if latest:
            outcomes = sorted({_status(e) for e in latest})
            row["status"] = {"value": outcomes[0] if len(outcomes) == 1 else "mixed", "outcomes": outcomes,
                "evaluations": [{"evaluation_id": e.get("evaluation_id"), "evidence_id": e["evidence_id"],
                                 "kind": e["kind"], "status": _status(e)} for e in latest],
                "basis": "all evaluations in the last referenced checkpoint; acquisition order is unknown"}
        else:
            row["status"] = {"value": "not_evaluated", "reasons": row["not_launched"]}
        if identity in anchors:
            row["accepted_anchor"] = anchors[identity]
        row["source"] = journal.obj(row["source_id"], "source")["files"]
        row["candidate_id"] = identity
        related = [(key, attempt) for key, attempt in attempts.items()
                   if attempt["attempt_id"] in row["produced_by_attempts"] or identity in attempt["source_roles"].values()]
        related_directions = set(row["directions"]) | {attempt["direction_id"] for _, attempt in related}
        row["research_files"] = {
            "directions": [direction_files[d] for d in sorted(related_directions) if d in direction_files],
            "attempts": [attempt_files[key] for key, _ in related],
            "claims": claim_files.get(identity, []), "claim_history": "claims.json"}
        row["parent_folders"] = [programs[p]["folder"] for p in row["parents"]]
        files[folder + "/program.json"] = _enc(dict(row, schema=SCHEMA, kind="program"))
        files[folder + "/candidate.json"] = journal.store._bytes(identity)
        files[folder + "/source.json"] = journal.store._bytes(row["source_id"])
        edges += [{"child": identity, "parent": p, "role": role} for role in ("parent", "donor") for p in row[role + "s"]]
        index.append({"ordinal": row["ordinal"], "program_id": identity, "folder": folder, "source_id": row["source_id"],
                      "parents": row["parents"], "status": row["status"]["value"], "directions": row["directions"],
                      "idea_ids": row["idea_ids"], "kind": ";".join(sorted({e["kind"] for e in latest})),
                      "times_ms": latest[0].get("times_ms") if len(latest) == 1 else None})
    for idea_id, entry in sorted(ideas.items()):
        idea_directions = {revision["direction_id"] for revision in entry["revisions"]}
        files["ideas/%s.json" % _slug(idea_id)] = _enc(dict(entry, schema=SCHEMA, kind="idea", basis=IDEA_BASIS,
            direction_files=[direction_files[d] for d in sorted(idea_directions) if d in direction_files],
            investigation_files=[attempt_files[key] for key, attempt in attempts.items()
                                 if attempt["direction_id"] in idea_directions],
            program_folders=sorted(row["folder"] for row in programs.values() if idea_id in row["idea_ids"])))
    for key, value in attempts.items():
        idea = value["implemented_idea"]
        strategy = ((journal.ref(idea["ref"]) or {}).get("explanation") or {}).get("strategy")
        idea.update(strategy_file=None, strategy_sha256=None)
        if isinstance(strategy, str) and strategy.strip():
            name = attempt_files[key].removesuffix(".json") + ".strategy.txt"
            files[name] = strategy.encode("utf-8")
            idea.update(strategy_file=name, strategy_sha256=sha256_hex(files[name]))
        files[attempt_files[key]] = _enc(dict(value, schema=SCHEMA, kind="attempt"))
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    columns = ("ordinal", "program_id", "status", "kind", "directions", "idea_ids", "times_ms", "folder")
    writer.writerow(columns)
    for row in index:
        writer.writerow([";".join(row[key]) if key in ("directions", "idea_ids") else
                         json.dumps(row[key], sort_keys=True) if key == "times_ms" else row[key] for key in columns])
    timeline, last = [], None  # Host selection history; this is not a measured-best curve.
    for sequence, state in enumerate(journal.records, 1):
        head = (state["note"].get("evolution") or {}).get("recommendation_head")
        if head and head != last:
            last, choice = head, journal.ref(head) or {}
            program = (choice.get("choice") or {}).get("candidate_id")
            timeline.append({"journal_sequence": sequence, "checkpoint_journal_time": state.get("updated_at"), "head": head,
                             "disposition": choice.get("disposition"), "program_id": program,
                             "accepted_monotonic_ns": choice.get("accepted_monotonic_ns"),
                             "folder": programs[program]["folder"] if program in programs else None})
    recommendation = journal.ref(evo.get("recommendation_head")) or {}
    resume = {"schema": SCHEMA, "kind": "resume", "journal_head": journal.digests[-1],
        "journal_sequence": len(journal.digests), "seed_program_id": seed, "planner_rows_without_program": planner_rows,
        "unrecoverable_trials": [{key: trial.get(key) for key in
            ("trial_index", "source_id", "evidence_id", "source_retention_error")} for trial in trials if not trial.get("candidate_id")],
        "campaign": {key: evo.get(key) for key in ("phase", "binding", "deadline", "limits")},
        "recommendation": {"head": evo.get("recommendation_head"), "disposition": recommendation.get("disposition"),
                           "program_id": (recommendation.get("choice") or {}).get("candidate_id")},
        "directions": {d: {"tip": s["tip"], "tip_folder": programs[s["tip"]]["folder"], "revision": s["revision"],
                           "status": s["status"], "anchor": s["anchor"], "idea_id": s["idea_id"],
                           "research_status": s["note"].get("research_status")} for d, s in sorted(directions.items())},
        "restart": {"materialize": "ResearchStore(root, ...).materialize(program.source_id, new_dir) "
                                   "or copy programs/<folder>/source/",
                    "continue": "resume through the host using an admitted parent_candidate_id; exported history is not a live campaign",
                    "limitations": "Historical file refs retain their original absolute paths. This view does not include the full "
                        "runtime, all raw evidence artifacts or external job dependencies. Keep the original ResearchStore and "
                        "host job/runtime inputs; reconstruct local packets and revalidate on a relocated host. "
                        "Revalidation is new evidence, never a rewrite of discovery-time observations."}}
    measurements = []
    for identity in order:
        for evaluation in programs[identity]["evaluations"]:
            measurements.append(dict(evaluation, candidate_id=identity,
                source_id=programs[identity]["source_id"], job_id=programs[identity]["job_id"],
                measurement_epoch_id=programs[identity]["measurement_epoch_id"]))
    measurements.extend(trial for trial in trial_evaluations if not trial.get("candidate_id"))
    measured_table = io.StringIO()
    fields = ("candidate_id", "source_id", "job_id", "measurement_epoch_id", "kind", "evaluation_id", "evidence_id",
              "settled", "correctness_observed", "correct", "eligible", "physical_devices", "times_ms",
              "relative_spreads", "source_retention_error", "feedback_error", "outcome")
    writer = csv.writer(measured_table, lineterminator="\n")
    writer.writerow(fields)
    for row in measurements:
        writer.writerow([json.dumps(row.get(key), sort_keys=True, allow_nan=False) for key in fields])
    files["measurements.json"] = _enc(measurements)
    files["measurements.csv"] = measured_table.getvalue().encode()
    files["trials.json"] = _enc(trials)
    files["trial-history.json"] = _enc(history)
    files["evidence.json"] = _enc({identity: journal.obj(identity, "evidence") for identity in evidence})
    # Copy bounded measurement/context payloads verbatim, without computing or
    # requalifying their numbers. Other artifacts remain addressed in evidence.json.
    for identity in evidence:
        artifacts = journal.obj(identity, "evidence")["artifacts"]
        for name in OBSERVATION_ARTIFACTS:
            if name in artifacts:
                files["observations/%s/%s" % (identity, name)] = journal.store._bytes(artifacts[name])
    files.update({"index.json": _enc(index), "index.csv": table.getvalue().encode(), "RESUME.json": _enc(resume),
                  "timeline.json": _enc({"kind": "host_selected", "basis": TIMELINE_BASIS, "entries": timeline,
                                         "measured_best": "not computed; measurements retain their original contexts"}),
                  "lineage.json": _enc({"nodes": [{key: row[key] for key in ("program_id", "folder", "status")}
                                                  for row in index], "edges": edges})})
    return journal, [programs[identity] for identity in order], files


def _inventory(root):
    """Hash every regular file and its source-identity executable bits."""
    result = {".": {"kind": "directory", "mode": stat.S_IMODE(root.lstat().st_mode)}}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise StateError("export contains a symlink or nonregular file")
        result[path.relative_to(root).as_posix()] = ({"sha256": sha256_hex(path.read_bytes()), "executable": mode & 0o111,
            "mode": stat.S_IMODE(mode), "kind": "file"} if stat.S_ISREG(mode) else
            {"kind": "directory", "mode": stat.S_IMODE(mode)})
    return result


def export_programs(store, out_dir, *, frozen_cases=None):
    """Publish one immutable snapshot, or verify an identical existing tree.

    Missing/corrupt CAS objects fail before publication. Explicit trial-retention
    failures remain rows; they are not silently promoted to recoverable sources.
    A newer journal needs a new output directory. Diagnostic cards require the
    host's registered {job_id: [case_id, ...]} mapping as frozen_cases.
    """
    out = Path(os.path.abspath(out_dir))
    if out.resolve() != out or any(path.is_symlink() for path in (out, *out.parents)):
        raise StateError("export destination must not traverse symlinks")
    if out == store.root.resolve() or store.root.resolve() in out.parents:
        raise StateError("export destination must be outside the research store")
    if not out.parent.is_dir():
        raise StateError("export destination parent must exist")
    journal, programs, files = build(store, frozen_cases=frozen_cases)
    staging = Path(tempfile.mkdtemp(prefix=".program-export-", dir=out.parent))
    try:
        for row in programs:
            source = staging / row["folder"] / "source"
            source.parent.mkdir(parents=True, exist_ok=True)
            store.materialize(row["source_id"], source)
        for name, data in sorted(files.items()):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o644)
        for path in [staging, *(p for p in staging.rglob("*") if p.is_dir())]:
            path.chmod(0o755)
        inventory = _inventory(staging)
        missing = sum(row.get("candidate_id") is None for row in json.loads(files["trials.json"]))
        manifest = {"schema": SCHEMA, "journal_head": journal.digests[-1], "journal_sequence": len(journal.digests),
            "programs": len(programs), "scope": SCOPE, "status": "retention_gaps" if missing else "complete_view",
            "source_retention_failures": missing,
            "files_sha256": {name: item["sha256"] for name, item in inventory.items() if item["kind"] == "file"},
            "files_executable": {name: item["executable"] for name, item in inventory.items() if item["kind"] == "file"},
            "entry_modes": {name: item["mode"] for name, item in inventory.items()}}
        (staging / "EXPORT.json").write_bytes(_enc(manifest))
        (staging / "EXPORT.json").chmod(0o644)
        if out.exists():
            if not out.is_dir() or _inventory(out) != _inventory(staging):
                raise StateError("existing export differs; use a new output directory")
        else:
            os.rename(staging, out)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export program records from a GEAK-EVO research store.")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frozen-cases", type=Path, help="host registry JSON: {job_id: [case_id, ...]}; required for diagnostic cards")
    args = parser.parse_args(argv)
    if not (args.store / "records").is_dir():
        parser.error("not a research store")
    # Read-only viewer: allow exactly the job/epoch pairs the journal records.
    states = [json.loads(_read(p.parent, p.name)[0])["state"] for p in sorted((args.store / "records").glob("*.json"))]
    pairs = sorted({(state["job_id"], state["measurement_epoch_id"]) for state in states})
    try:
        frozen_cases = json.loads(args.frozen_cases.read_bytes()) if args.frozen_cases else None
    except (OSError, ValueError) as exc:
        parser.error("cannot read --frozen-cases registry: " + str(exc))
    try:
        manifest = export_programs(ResearchStore(args.store, allowed_epochs=pairs, source_paths={}), args.out,
                                   frozen_cases=frozen_cases)
    except ArchiveError as exc:
        parser.error(str(exc) + "; diagnostic cards need --frozen-cases with the host registry")
    print(json.dumps({key: manifest[key] for key in ("journal_head", "journal_sequence", "programs")}, sort_keys=True))


if __name__ == "__main__":
    main()
