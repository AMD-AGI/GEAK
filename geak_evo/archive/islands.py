"""Island groups: campaigns of one job that share their reviewed code with each other.

A group is K ordinary serial EVO campaigns of one study and one job, each with its
own device and measurement epoch. Each campaign publishes every reviewed role it
produced (all of them, never a selection) into a shared registry directory. A
publication is a pure function of one committed prefix of the origin's
hash-chained journal.

Importing a sibling's publication replays that prefix read-only, requires the
identical document, and requires that the exact source bytes reproduce the
published source identity. The code then becomes a local immigrant candidate with
no parents and no local measurement, and the origin evidence its outcomes name is
copied as ordinary local evidence. Origin reviews, costs and evidence are
foreign-epoch observations only; only local measurements rate an immigrant.
Importing the same publication again writes nothing new.

Where a member campaign keeps its store is a ``store_root`` callable (default:
``<campaign_root>/store``). The claim-context projection comes from the planner and
is a ``claim_context`` callable (default: ``geak_evo.planner.context``, imported
when first needed).
"""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from .._util import require
from .claims import ClaimLog
from .store import ResearchStore, _digest, _json, _read, _write
from .succession import episode_review, parent_outcomes

SCHEMA = "geak-island-publication-v2"
# Host-written collection artifacts whose evidence references are copied with an outcome.
REFERENCE_ARTIFACTS = ("native_evidence_index.json",)
MEASUREMENT_SCOPE = ("Origin review, costs and evidence were produced on another device and measurement "
                     "epoch. They are not local fitness, validated claims or promotion evidence here. "
                     "origin_evidence_map maps origin evidence IDs to their exact local copies. "
                     "origin_claim_context is the claim state at that verified publication prefix; "
                     "is_latest_imported_for_origin identifies the newest imported prefix for that origin. "
                     "Older publications remain historical; no origin quantity is locally authorized.")
# Default layout: a member campaign keeps its ResearchStore in <campaign_root>/store.
STORE_DIR = "store"



def _need(condition, message):
    require(condition, message, ValueError)


def member_store(campaign_root):
    """Default location of a member campaign's store."""
    return Path(campaign_root) / STORE_DIR


def _planner_claim_context(view, *, binding=None):
    from ..planner.context import current_claim_context
    return current_claim_context(view, binding=binding)


def group_for_policy(policy, campaign_root, job_id, *, store_root=member_store, claim_context=None):
    if "island_group" not in policy:
        return None
    return IslandGroup(policy, campaign_root, job_id, store_root=store_root, claim_context=claim_context)


def _references(value):
    """Every evidence_id named inside one decoded host JSON artifact."""
    if type(value) is dict:
        found = [value["evidence_id"]] if type(value.get("evidence_id")) is str else []
        return found + [ref for child in value.values() for ref in _references(child)]
    if type(value) is list:
        return [ref for child in value for ref in _references(child)]
    return []


def _content(doc):
    """Republish changed source-relevant claims, not an unrelated claim-head advance."""
    context = dict(doc["origin_claim_context"], claim_head=None)
    return _json({"outcomes": doc["outcomes"], "origin_claim_context": context})


class _OriginStore(ResearchStore):
    """Read-only view of a store, cut at one verified journal prefix.

    It never creates directories, locks or writes. Old direction revisions inside
    the prefix stay readable, so a publication can be recomputed exactly.
    """

    def __init__(self, root, job_id, epoch):
        self.root, self.verify_evidence, self.commit_frontier = Path(root), None, None
        self.allowed_epochs, self.source_paths = frozenset({(job_id, epoch)}), {}
        self._prefix_index, self._prefix_revisions = None, None

    def _lock(self):
        raise ValueError("origin prefix store cannot open a lock")

    def _put(self, data):
        raise ValueError("origin prefix store cannot write objects")

    def _commit(self, index, state):
        raise ValueError("origin prefix store cannot commit records")

    def _replay(self, through=None):
        """Verify the journal exactly up to ``through``; ignore any cached index."""
        self._prefix_index, self._prefix_revisions = None, None
        _need(type(through) is int and through > 0, "positive publication prefix sequence required")
        index, revisions = {"head": None, "sequence": 0, "directions": {}}, {}
        for path in sorted((self.root / "records").glob("*.json")):
            if index["sequence"] == through:
                break
            data, _ = _read(path.parent, path.name)
            record = json.loads(data)
            sequence = index["sequence"] + 1
            _need(path.name == "%08d-%s.json" % (sequence, _digest(data))
                  and record["previous"] == index["head"], "origin journal hash/sequence mismatch")
            state = record["state"]
            previous = index["directions"].get(state["direction_id"])
            _need(state["revision"] == (previous["revision"] + 1 if previous else 1),
                  "origin direction revision mismatch")
            self._validate(state)
            revisions[(state["direction_id"], state["revision"])] = state
            index["directions"][state["direction_id"]] = state
            index.update(head=_digest(data), sequence=sequence)
        _need(index["sequence"] == through, "origin journal prefix is absent")
        self._prefix_index, self._prefix_revisions = index, revisions
        return copy.deepcopy(index)

    def get(self, direction_id=None):
        _need(self._prefix_index is not None, "origin publication prefix is not verified")
        directions = self._prefix_index["directions"]
        return copy.deepcopy(directions if direction_id is None else directions[direction_id])

    def get_revision(self, direction_id, revision):
        _need(type(revision) is int and revision > 0, "positive historical revision required")
        _need(self._prefix_revisions is not None, "origin publication prefix is not verified")
        state = self._prefix_revisions.get((direction_id, revision))
        _need(state is not None, "historical direction revision is outside the verified publication prefix")
        return copy.deepcopy(state)


def _reviewed_publication_outcomes(store, snapshot, candidate_id):
    """None while any review is pending; an inconsistent committed review fails."""
    outcomes = parent_outcomes(store, snapshot, candidate_id, require_review=False)
    if not outcomes:
        return None
    history_rows = snapshot["state"]["history"]
    for outcome in outcomes:
        histories = [row for row in history_rows
                     if row.get("attempt_id") == outcome["attempt_id"]
                     and all(row.get(key) == outcome[key] for key in ("transport_ref", "collection_ref"))]
        _need(len(histories) == 1, "publication outcome lacks one exact native history")
        history = histories[0]
        reviews = [row for row in history_rows
                   if row.get("review_kind") == "research_review"
                   and row["proposal"]["direction_id"] == outcome["direction_id"]
                   and all(row["proposal"][key] == history[key] for key in ("transport_ref", "collection_ref"))]
        if not reviews:
            return None
        outcome["review"] = episode_review(store, snapshot, history, outcome["direction_id"])
    return outcomes


class IslandGroup:
    def __init__(self, policy, campaign_root, job_id, *, store_root=member_store, claim_context=None):
        group = policy["island_group"]  # The policy validator has checked shape and membership.
        self.job_id = job_id
        self.group_id, self.registry = group["group_id"], Path(group["registry_root"])
        self.members = [Path(root) for root in group["members"]]
        self.campaign_root = Path(campaign_root)
        self.store_root = store_root
        self.claim_context = claim_context or _planner_claim_context
        _need(self.campaign_root in self.members, "campaign is not a declared group member")

    def _document(self, store, index, campaign_id, origin_root, candidate_id):
        """One role's publication from one verified prefix; None until all its outcomes are reviewed."""
        directions = index["directions"]
        _need(type(directions.get(campaign_id, {}).get("note", {}).get("evolution")) is dict,
              "publication origin is not an evolution campaign")
        snapshot = {"campaign_id": campaign_id, "state": directions[campaign_id]["note"]["evolution"],
                    "direction_notes": {name: {"revision": row["revision"], "note": row["note"]}
                                        for name, row in directions.items() if name != campaign_id}}
        outcomes = _reviewed_publication_outcomes(store, snapshot, candidate_id)
        if outcomes is None:
            return None
        candidate = store.get_candidate(candidate_id)
        # Read the origin's committed claim state without importing its quantity
        # authority. The recipient receives corrections and declared dependency
        # states even when the original episode review has not changed.
        claim_view = ClaimLog(store, snapshot["state"].get("claim_head")).view()
        claim_context = self.claim_context(claim_view, binding={
            key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")})
        published = [{key: outcome[key] for key in ("direction_id", "attempt_id", "roles",
                                                    "transport_ref", "collection_ref", "collection_costs")}
                     | {"review_note": outcome["review"]["note"]} for outcome in outcomes]
        return _json({"schema": SCHEMA, "group_id": self.group_id,
                      "origin_campaign_root": str(origin_root), "origin_campaign_id": campaign_id,
                      "origin_journal": {"sequence": index["sequence"], "head": index["head"]},
                      "job_id": candidate["job_id"], "measurement_epoch_id": candidate["measurement_epoch_id"],
                      "candidate_id": candidate_id, "source_id": candidate["source_id"],
                      "origin_claim_context": claim_context, "outcomes": published})

    def _own_roles(self, store, index, campaign_id):
        roles = set()
        for row in index["directions"][campaign_id]["note"]["evolution"]["history"]:
            if row.get("settled") is True and row.get("collection_ref"):
                _, artifacts = store.evidence(row["collection_ref"]["evidence_id"])
                if "native_source_roles.json" in artifacts:
                    roles.update(json.loads(artifacts["native_source_roles.json"]).values())
        return roles

    def publish(self, store, campaign_id):
        """Publish each own role once per distinct reviewed content; pending reviews wait."""
        with store._lock():
            index = store._replay()
        origin = _OriginStore(store.root, self.job_id,
                              index["directions"][campaign_id]["measurement_epoch_id"])
        prefix = origin._replay(through=index["sequence"])
        _need(prefix["head"] == index["head"], "own publication prefix changed")
        index = prefix
        roles = self._own_roles(store, index, campaign_id)
        published = {(doc["candidate_id"], _content(doc)) for _, _, doc in self._publications()
                     if Path(doc["origin_campaign_root"]) == self.campaign_root}
        for candidate_id in sorted(roles):
            raw = self._document(origin, index, campaign_id, self.campaign_root, candidate_id)
            if raw is None or (candidate_id, _content(json.loads(raw))) in published:
                continue
            try:
                _write(self.registry / _digest(raw), raw, immutable=True)
            except FileExistsError:
                _need(_read(self.registry, _digest(raw))[0] == raw, "registry publication hash mismatch")

    def _publications(self):
        for name in sorted(os.listdir(self.registry)):
            if name.startswith("."):
                continue
            raw, _ = _read(self.registry, name)
            _need(_digest(raw) == name, "registry publication hash mismatch: " + name)
            doc = json.loads(raw)
            _need(type(doc) is dict and doc.get("schema") == SCHEMA and doc.get("group_id") == self.group_id
                  and Path(doc.get("origin_campaign_root", "")) in self.members
                  and doc.get("job_id") == self.job_id, "publication outside the declared group or job: " + name)
            yield name, raw, doc

    def _copies(self, origin, doc):
        """Exact origin evidence of every published outcome: {origin evidence_id: artifacts}."""
        named = []
        for outcome in doc["outcomes"]:
            named += [outcome["transport_ref"]["evidence_id"], outcome["collection_ref"]["evidence_id"]]
            _, artifacts = origin.evidence(outcome["collection_ref"]["evidence_id"])
            for name in REFERENCE_ARTIFACTS + (outcome["collection_ref"]["artifact"],):
                if name in artifacts:
                    named += _references(json.loads(artifacts[name]))
        named += _references(doc["origin_claim_context"])
        copies = {}
        while named:
            identity = named.pop()
            if identity in copies:
                continue
            copies[identity] = origin.evidence(identity)[1]
            # Follow claim history and exact scalar provenance, without parsing
            # unrelated large profiler payloads or treating their prose as refs.
            for name in ("claim.json", "correction.json", "quantity_provenance.json"):
                if name in copies[identity]:
                    named += _references(json.loads(copies[identity][name]))
        return copies

    def _import(self, store, epoch, raw, doc, copies, *, write):
        """Register the immigrant and its local evidence, or only compute their IDs."""
        value = {"kind": "candidate", "job_id": self.job_id, "measurement_epoch_id": epoch,
                 "source_id": doc["source_id"], "parents": [], "donors": [], "contribution_claims": []}
        identity = store.candidate(self.job_id, epoch, doc["source_id"]) if write else _digest(_json(value))
        binding = {"job_id": self.job_id, "measurement_epoch_id": epoch, "candidate_source_id": doc["source_id"]}

        def archive(artifacts):
            if write:
                return store.archive_evidence(identity, artifacts)
            return _digest(_json({"kind": "evidence", "candidate_id": identity, "binding": binding,
                                  "artifacts": {name: _digest(data) for name, data in artifacts.items()}}))

        mapping = {origin: archive(artifacts) for origin, artifacts in copies.items()}
        evidence = archive({"island_publication.json": raw, "origin_evidence_map.json": _json(mapping)})
        return identity, evidence, mapping

    def _verify(self, store, origin, raw, doc):
        """Replay the origin prefix; require the identical document and the exact source."""
        journal = doc["origin_journal"]
        index = origin._replay(through=journal["sequence"])
        _need(index["head"] == journal["head"] and raw == self._document(
            origin, index, doc["origin_campaign_id"], doc["origin_campaign_root"], doc["candidate_id"]),
            "publication differs from its origin journal")
        source = origin.source(doc["source_id"])
        scratch = Path(tempfile.mkdtemp(prefix=".immigrant-", dir=self.campaign_root))
        try:
            for row in source["files"]:
                if row["blob_id"]:
                    path = scratch / row["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(origin._bytes(row["blob_id"]))
                    path.chmod(0o644 | row["executable"])
            captured = store.capture_source(scratch, [row["path"] for row in source["files"]])
        finally:
            shutil.rmtree(scratch)
        _need(captured == doc["source_id"], "imported source differs from its origin identity")

    def sync(self, store, snapshot):
        """Publish own reviewed roles; verify and import every new sibling publication.

        Returns {immigrant candidate_id: [record]}. A sibling source identical to
        this campaign's seed is never imported. On any failure the call fails and
        the reason is saved under ``<campaign_root>/island-rejections``.
        """
        try:
            self.publish(store, snapshot["campaign_id"])
            return self._records(store, snapshot["state"]["binding"]["measurement_epoch_id"],
                                 store.get(snapshot["campaign_id"])["tip"], write=True)
        except Exception as exc:
            record = _json({"error": type(exc).__name__, "message": str(exc)})
            rejected = self.campaign_root / "island-rejections"
            rejected.mkdir(mode=0o700, exist_ok=True)
            try:
                _write(rejected / (_digest(record) + ".json"), record, immutable=True)
            except FileExistsError:
                pass
            raise

    def lookup(self, store, epoch, seed_candidate_id):
        """Read-only recomputation of earlier verified imports."""
        return self._records(store, epoch, seed_candidate_id, write=False)

    def _records(self, store, epoch, seed_candidate_id, *, write):
        seed = store.get_candidate(seed_candidate_id)["source_id"]
        result = {}
        for name, raw, doc in self._publications():
            if Path(doc["origin_campaign_root"]) == self.campaign_root or doc["source_id"] == seed:
                continue
            origin = _OriginStore(self.store_root(Path(doc["origin_campaign_root"])), self.job_id,
                                  doc["measurement_epoch_id"])
            copies = self._copies(origin, doc)
            identity, evidence, mapping = self._import(store, epoch, raw, doc, copies, write=False)
            # The import evidence is written last, only after its publication was
            # verified, so its presence means this publication is already imported.
            if not (store.root / "objects" / evidence).exists():
                if not write:
                    continue  # Published after this campaign's latest import.
                try:
                    self._verify(store, origin, raw, doc)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise ValueError("island publication " + name + " rejected: " + str(exc)) from exc
                _need(self._import(store, epoch, raw, doc, copies, write=True) == (identity, evidence, mapping),
                      "import identity differs from its recomputation")
            for local in [evidence, *mapping.values()]:
                store.evidence(local)
            result.setdefault(identity, []).append({
                "evidence_id": evidence, "origin_evidence_map": mapping,
                "origin_campaign_id": doc["origin_campaign_id"], "origin_candidate_id": doc["candidate_id"],
                "origin_measurement_epoch_id": doc["measurement_epoch_id"], "origin_journal": doc["origin_journal"],
                "origin_claim_context": doc["origin_claim_context"],
                "outcomes": doc["outcomes"], "measurement_scope": MEASUREMENT_SCOPE})
        for records in result.values():
            _mark_latest(records)
        return result


def _mark_latest(records):
    """Flag the newest imported prefix per (origin campaign, origin candidate); sort by origin."""
    newest = {}
    for record in records:
        key = (record["origin_campaign_id"], record["origin_candidate_id"])
        newest[key] = max(newest.get(key, 0), record["origin_journal"]["sequence"])
    for record in records:
        key = (record["origin_campaign_id"], record["origin_candidate_id"])
        record["is_latest_imported_for_origin"] = record["origin_journal"]["sequence"] == newest[key]
    records.sort(key=lambda record: (record["origin_campaign_id"], record["origin_journal"]["sequence"]))
