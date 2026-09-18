#!/usr/bin/env python3
"""Identity-first reconciliation for the GEAK execution tracker.

Every retention bug the tracker has had came from the same mistake: merging a
DERIVED summary (a list length, a total byte count, an endpoint pair, a
lexicographic score) after the information that distinguished the records had
already been discarded. Aggregates cannot tell "this block shrank" from "a
different block grew", and a resolved edge cannot remember that it was later
contradicted.

So reconciliation happens here, on the records themselves, keyed by stable
identity, BEFORE anything is derived from them:

  call        (agent_id, call_id)
  input block (call_id, source_uuid, kind, position)
  tool action tool_use_id
  attempt     (spawn_event_id, attempt_id)
  edge        (type, from, to, event_id | spawn_event_id)

Each reconciled item carries an explicit state:

  ``observed``   present in the latest read of its source
  ``retained``   absent from the latest read, kept from an earlier capture
  ``replayed``   seen again, byte-identical: idempotent
  ``conflicted`` seen again with contradictory identity-defining fields

``conflicted`` is sticky. A claim that has been contradicted does not become
true again because a later pass happens not to see the contradiction, so a
conflicted identity can never be resurrected as proven; the disputed evidence is
kept for inspection instead. Callers derive graphs, counters and coverage from
this store, so every view agrees and none of them re-invents a merge rule.
"""

OBSERVED = "observed"
RETAINED = "retained"
REPLAYED = "replayed"
CONFLICTED = "conflicted"


class Reconciled:
    """An identity-keyed store of captured records and their states."""

    def __init__(self):
        self.items = {}       # key -> value
        self.states = {}      # key -> state
        self.conflicts = {}   # key -> list of conflicting field reports

    # -- core ---------------------------------------------------------------
    def absorb(self, key, value, identity_fields=(), merge=None, seen_now=True):
        """Fold one record in under ``key``.

        ``identity_fields`` are the fields whose disagreement means the two
        records describe DIFFERENT things under the same identity -- a genuine
        contradiction, not an update. ``merge`` reconciles the non-identity
        payload of two captures of the same record.
        """
        prior = self.items.get(key)
        if prior is None:
            self.items[key] = value
            self.states[key] = OBSERVED if seen_now else RETAINED
            return self.states[key]

        clash = [f for f in identity_fields if prior.get(f) != value.get(f)]
        if clash:
            self.states[key] = CONFLICTED
            self.conflicts.setdefault(key, []).append(
                {"fields": clash,
                 "previous": {f: prior.get(f) for f in clash},
                 "incoming": {f: value.get(f) for f in clash}})
            # Keep the earlier record as the disputed evidence; it is no longer
            # something we are entitled to present as established.
            return CONFLICTED

        if self.states.get(key) == CONFLICTED:
            return CONFLICTED  # sticky: never un-conflict

        self.items[key] = merge(prior, value) if merge else value
        self.states[key] = REPLAYED if prior == value else OBSERVED
        return self.states[key]

    def retain_missing(self, keys_seen_now):
        """Mark everything not seen in the latest read as retained history."""
        retained = 0
        for key in self.items:
            if key in keys_seen_now:
                continue
            if self.states.get(key) in (CONFLICTED,):
                continue
            if self.states.get(key) != RETAINED:
                self.states[key] = RETAINED
                retained += 1
        return retained

    # -- queries -------------------------------------------------------------
    def usable(self):
        """Items safe to present: everything except contradicted identities."""
        return [(k, v) for k, v in self.items.items()
                if self.states.get(k) != CONFLICTED]

    def conflicted_keys(self):
        return [k for k, st in self.states.items() if st == CONFLICTED]

    def count(self, state):
        return sum(1 for st in self.states.values() if st == state)

    def summary(self):
        return {"total": len(self.items), "observed": self.count(OBSERVED),
                "retained": self.count(RETAINED), "replayed": self.count(REPLAYED),
                "conflicted": self.count(CONFLICTED)}


# --------------------------------------------------------------------------- #
# Identity functions and field-wise merges for the record kinds the tracker has
# --------------------------------------------------------------------------- #
def block_key(call_id, block, position):
    """Input blocks are identified by their SOURCE record, not their position
    in a list: a list index shifts when a sibling changes, and comparing total
    bytes lets one block's growth mask another block's loss."""
    # Position WITHIN the source record, never the flattened list index: the
    # flattened index shifts when an earlier source record goes missing, which
    # made the same block look like a new one and left duplicates behind.
    within = block.get("source_pos")
    if within is None:
        # An unresolved legacy block: keep it on its OWN identity rather than
        # claiming a raw position it may not have had.
        if block.get("legacy_position_unresolved"):
            return (call_id, block.get("source_uuid"), block.get("kind"),
                    "#legacy-unresolved-%s" % position)
        within = position
    source = block.get("source_uuid")
    if source is None:
        # No source identity at all. Distinct records must NOT collapse into one,
        # so fall back to the flattened position, which is unique per capture.
        return (call_id, "#nosource", block.get("kind"), position)
    return (call_id, source, block.get("kind"),
            block.get("tool_use_id") or within)


def merge_block(prev, new):
    """Keep whichever capture actually holds more of the block's text."""
    if len(prev.get("text") or "") > len(new.get("text") or ""):
        out = dict(prev)
        out["retained_from_earlier_capture"] = True
        return out
    return new


def merge_action(prev, new):
    """Tool action: never lose a recorded result, nor let its payload shrink."""
    out = dict(new)
    pr, nr = (prev.get("result") or {}), (out.get("result") or {})
    if nr.get("status") == "missing" and pr.get("status") in ("ok", "error"):
        out["result"] = pr
        out["retained_from_earlier_capture"] = True
    elif len(pr.get("preview") or "") > len(nr.get("preview") or ""):
        out["result"] = pr
        out["retained_from_earlier_capture"] = True
    if len(prev.get("args_preview") or "") > len(out.get("args_preview") or ""):
        out["args_preview"] = prev.get("args_preview")
        out["args_truncated"] = prev.get("args_truncated")
        out["args_bytes_total"] = prev.get("args_bytes_total")
        out["retained_from_earlier_capture"] = True
    return out


def attempt_key(spawn_event_id, attempt):
    return (spawn_event_id, attempt.get("attempt_id"))


#: Relationship types that carry an explicit recorded event id. ONLY these may
#: be keyed by that id; the baseline journal edges have none.
LINKAGE_TYPES = ("result_supplied_to_dispatch", "agent_spawn", "agent_spawn_return")


def edge_key(edge):
    """Identity of a relationship.

    For the explicit linkage contracts, identity is the recorded event id and
    endpoints are compared fields -- so the same event pointing somewhere else
    CONTRADICTS rather than becoming a second claim.

    The baseline journal edges (workflow -> agent, agent -> workflow) carry no
    event id, so they must keep their per-invocation identity. Keying them by id
    collapsed every orchestration edge onto (orchestration, None) and every
    return onto (return, None), making different agents' endpoints look like
    contradictions and deleting the whole graph on the next pass.
    """
    kind = edge.get("type")
    eid = edge.get("event_id") or edge.get("spawn_event_id")
    if kind in LINKAGE_TYPES and eid:
        return (kind, eid)
    return (kind, edge.get("from"), edge.get("to"))


def normalize_invalidation_key(item):
    """Accept persisted keys from older formats and return the current shape.

    An earlier format wrote [type, from, to, event_id]; comparing that against
    the current key silently un-invalidated a relationship on upgrade.
    """
    if isinstance(item, (list, tuple)):
        parts = list(item)
        if len(parts) == 4:
            kind, frm, to, eid = parts
            if kind in LINKAGE_TYPES and eid:
                return (kind, eid)
            return (kind, frm, to)
        return tuple(parts)
    return None


def migrate_blocks(blocks, current=None):
    """Resolve legacy blocks that predate recorded source positions.

    ``source_pos`` is the block's index in the source record's ORIGINAL content
    array. A legacy capture only tells us which blocks SURVIVED filtering, and
    those are not the same whenever the source contained an omitted entry (an
    image, an empty text block): counting survivors invents a position that can
    name a different block, which duplicates it against its own re-read.

    So no position is manufactured. A legacy block is matched against the
    current capture as evidence, and only where the correspondence is
    unambiguous -- exactly one current block from the same source record and of
    the same kind. Otherwise it keeps a distinct, explicitly unresolved identity
    so it is retained rather than silently merged onto the wrong block.
    """
    if not blocks:
        return []
    by_source = {}
    for blk in (current or []):
        if isinstance(blk, dict) and blk.get("source_uuid") is not None:
            by_source.setdefault((blk.get("source_uuid"), blk.get("kind")), []).append(blk)

    # Correspondence must be one-to-one across BOTH sets. A single candidate on
    # the current side is not a match if several legacy blocks compete for it:
    # assigning them all the same position collapses distinct captured blocks
    # into one and silently drops the others.
    legacy_counts = {}
    for blk in blocks:
        if isinstance(blk, dict) and blk.get("source_pos") is None \
                and blk.get("source_uuid") is not None:
            key = (blk.get("source_uuid"), blk.get("kind"))
            legacy_counts[key] = legacy_counts.get(key, 0) + 1

    out = []
    for blk in blocks:
        if not isinstance(blk, dict) or blk.get("source_pos") is not None \
                or blk.get("source_uuid") is None:
            out.append(blk)
            continue
        key = (blk.get("source_uuid"), blk.get("kind"))
        candidates = by_source.get(key) or []
        blk = dict(blk)
        if (len(candidates) == 1 and legacy_counts.get(key) == 1
                and candidates[0].get("source_pos") is not None):
            blk["source_pos"] = candidates[0]["source_pos"]
            blk["legacy_position_resolved"] = True
        else:
            # Ambiguous on either side, or absent: do NOT guess a raw position.
            blk["legacy_position_unresolved"] = True
        out.append(blk)
    return out


#: Fields whose disagreement contradicts the relationship an edge asserts.
EDGE_IDENTITY_FIELDS = ("from", "to", "producer_result_ref", "consumer_input_ref",
                        "forwarding", "transformation", "spawn_tool_call_id")

#: Fields whose disagreement contradicts what an attempt records.
ATTEMPT_IDENTITY_FIELDS = ("status", "result_ref", "error")


def reconcile_attempts(prev_attempts, new_attempts, spawn_event_id):
    """Reconcile spawn attempts by attempt_id.

    Preserving old attempts only when the new list is EMPTY loses history on a
    partial re-read; and picking an outcome by sorting attempt ids treats an
    identifier as a chronology. Attempts are merged individually, and the
    aggregate outcome is only stated when the records order themselves.
    """
    store = Reconciled()
    seen = set()
    for att in new_attempts or []:
        key = attempt_key(spawn_event_id, att)
        seen.add(key)
        store.absorb(key, att, identity_fields=ATTEMPT_IDENTITY_FIELDS)
    for att in prev_attempts or []:
        key = attempt_key(spawn_event_id, att)
        if key in store.items:
            store.absorb(key, att, identity_fields=ATTEMPT_IDENTITY_FIELDS)
        else:
            store.absorb(key, att, seen_now=False)
    store.retain_missing(seen)

    merged = [v for _k, v in store.usable()]
    ordered = [a for a in merged if isinstance(a.get("seq"), int)]
    seqs = [a["seq"] for a in ordered]
    if ordered and len(ordered) == len(merged) and len(set(seqs)) == len(seqs):
        # A unique final position is required. A TIE provides no evidence of
        # which attempt was last, and sorting it would silently inherit the
        # attempt-id ordering -- the guess this rule exists to prevent.
        merged.sort(key=lambda a: a["seq"])
        outcome = merged[-1].get("status")
    elif ordered and len(set(seqs)) != len(seqs):
        outcome = "unknown"
    elif len(merged) == 1:
        outcome = merged[0].get("status")
    else:
        # No authoritative ordering: an attempt id is an identity, not a
        # sequence, so the final outcome is unknown rather than guessed.
        outcome = "unknown"
    if store.conflicted_keys():
        outcome = "conflicted"
    return merged, outcome, store.summary()
