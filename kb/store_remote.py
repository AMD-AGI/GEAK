#!/usr/bin/env python3
"""The HTTP plane, wearing LocalKBStore's interface.

`experience_store.py` and `e2e_store.py` read and write through one small surface — candidates(),
get_session(), materialize(), write(), maybe_promote() — so pointing a lane at the service instead
of at a directory is a constructor change and nothing else. Everything that decides WHAT gets
offered (the bench-key comparability filter, the direction collapse, the min-speedup floor) stays
in the caller, because the service ranks on nothing but the `speedup` number a producer declared
and would happily order a `b:` measurement against a `b2:` one.

Three places where the service is not a directory, and the adapter has to say so rather than
pretend:

  * `identities()` returns nothing. There is no search on the `geak` scheme
    (`POST /v1/kb/search` -> `search_unsupported`), so the pre-ladder near-miss scan that
    LocalKBStore supports is simply unavailable here. The ladder itself still works — it constructs
    addresses rather than discovering them, which is exactly why it was built that way.
  * ranking costs one request per candidate. `GET /sessions/top` returns ids and scores only, no
    knowledge, and the caller needs `direction` and `bench_key` to curate. So the top page is
    fetched once and then hydrated, bounded by `scan` — a page holds up to 200 sessions and pulling
    all of them to offer three would be absurd.
  * the ladder is NOT cheap on the wire. Artifacts land under `kb/<producer>/<session_id>/`, so the
    rungs of one write share an S3 prefix — but the file MANIFEST is per (canonical_id, session_id),
    and a rung that never called put_files reports file_count 0 and downloads nothing. Committing
    once and expecting the coarse page to inherit it produces a bundle with no patch.diff, which is
    how this was found. So every rung uploads, and transfer scales with ladder depth even though
    storage does not.

There is no delete. Every write here is permanent — the service exposes no DELETE for `/v1/kb/*`,
so a wrong canonical id is not a mistake that can be cleaned up afterwards, only one that can be
outranked.
"""

import json
import os

from kb.store_local import (CHAMPION_METRIC, KBStoreError, Candidate, finite_speedup,
                            safe_rel_path, validate_session_id)

DEFAULT_SCAN = 25          # candidates hydrated before curation; well under the 200 rollup cap
ARTIFACT_KIND = "rewrite"


class RemoteKBStore(object):
    """One producer's candidates under a canonical identity, over HTTP."""

    def __init__(self, client, scan: int = DEFAULT_SCAN, metric: str = CHAMPION_METRIC,
                 promote_floor: float = 1.0):
        self._client = client
        self._scan = max(1, int(scan))
        # See LocalKBStore.__init__ for why this is a parameter. Remotely it is also a hard
        # constraint rather than a preference: `sessions/top?metric=` reads one flat top-level
        # `knowledge.<name>` scalar and nothing else, so the ranking name has to be decided by
        # whoever writes the document, not discovered afterwards.
        self.metric = str(metric or CHAMPION_METRIC)
        self.promote_floor = float(promote_floor)
        self.root = str(getattr(client, "base_url", "") or "kb-store")

    @classmethod
    def from_env(cls, scan: int = DEFAULT_SCAN, metric: str = CHAMPION_METRIC,
                 promote_floor: float = 1.0):
        """Build from KB_STORE_URL / KB_STORE_TOKEN, or return (None, reason)."""
        try:
            from kb.store_client import KBStoreClient, kb_store_token, kb_store_url
        except ImportError as e:
            return None, "store_unavailable: " + str(e)[:120]
        if not kb_store_url() or not kb_store_token():
            return None, ("no_credentials: (GEAK_)KB_STORE_URL / (GEAK_)KB_STORE_TOKEN "
                          "are not both set")
        try:
            return cls(KBStoreClient.from_env(), scan, metric, promote_floor), ""
        except Exception as e:
            return None, "unusable_store: %s: %s" % (type(e).__name__, str(e)[:120])

    # -- read ----------------------------------------------------------------------------

    def identities(self):
        """Always empty: the scheme supports exact lookup only. See the module docstring."""
        return []

    def candidates(self, canonical_id: str, limit: int = 3):
        """Rank this identity's candidates, hydrating each one's knowledge document.

        A 404 is an empty page, not an error: on a scheme with no search that is the ONLY signal
        distinguishing "nothing recorded" from "recorded somewhere else", and the caller's ladder is
        what turns it into a next attempt.
        """
        try:
            top = self._client.get_top_sessions(canonical_id, metric=self.metric,
                                                limit=self._scan, offset=0)
        except Exception:
            return []
        rows = (top or {}).get("sessions") or []
        found = []
        for row in rows:
            sid = str(row.get("session_id") or "")
            if not sid:
                continue
            knowledge = self.get_session(canonical_id, sid)
            if not isinstance(knowledge, dict):
                continue
            # Prefer the score the service computed; fall back to the document so a candidate is
            # never dropped just because it was indexed under a different metric name.
            speedup = finite_speedup(row.get("score"))
            if speedup is None:
                speedup = finite_speedup(knowledge.get(self.metric))
            found.append(Candidate(sid, knowledge, speedup, bool(row.get("is_champion"))))
        found.sort(key=lambda c: (-(c.speedup if c.speedup is not None else float("-inf")),
                                  c.session_id))
        return found[: max(0, int(limit))] if limit else found

    def get_session(self, canonical_id: str, session_id: str):
        try:
            record = self._client.get_session(canonical_id, validate_session_id(session_id))
        except Exception:
            return None
        if not isinstance(record, dict):
            return None
        # The envelope carries the producer's document under `knowledge`; a record fetched by id
        # is already that document on some routes. Accept either rather than guess.
        knowledge = record.get("knowledge")
        return knowledge if isinstance(knowledge, dict) else record

    def champion(self, canonical_id: str):
        """The identity's PROMOTED session, or {} when nothing has been promoted.

        Read from the rollup's `champion` field, NOT from get_best_record(). That endpoint answers
        "the record to act on" — it falls back to the most recently selected record when no champion
        exists — so using it here reports a brand-new page as already having a champion whose score
        is the write that just landed. maybe_promote() then compares a candidate against itself,
        never beats it, and no first champion is ever set. LocalKBStore returns {} for an
        unpromoted identity and this plane has to mean the same thing by it.
        """
        try:
            rollup = self._client.get_rollup(canonical_id)
        except Exception:
            return {}
        champion = (rollup or {}).get("champion")
        if not isinstance(champion, dict) or not champion.get("session_id"):
            return {}
        return {"session_id": str(champion.get("session_id") or ""),
                "metric": str(champion.get("metric") or self.metric),
                "value": champion.get("value")}

    def champion_speedup(self, canonical_id: str):
        champion = self.champion(canonical_id)
        # Same guard as the local plane: an incumbent recorded under another metric is not a lower
        # bar, it is an incomparable number, and ranking tokens-per-second against a ratio would
        # either block every promotion or wave through every one.
        if str(champion.get("metric") or "") != self.metric:
            return None
        return finite_speedup(champion.get("value"))

    def materialize(self, canonical_id: str, candidate, destination: str) -> str:
        """Download one selected candidate as the standard bundle: recipe.json + files/.

        Same layout LocalKBStore.materialize() produces, so `_render_references` and the lane's
        adopt step cannot tell the two planes apart.
        """
        session_id = candidate.session_id if isinstance(candidate, Candidate) else str(candidate)
        knowledge = (candidate.knowledge if isinstance(candidate, Candidate)
                     else self.get_session(canonical_id, session_id))
        if not isinstance(knowledge, dict):
            raise KBStoreError("candidate knowledge is unreadable: " + session_id)
        bundle = os.path.join(destination, validate_session_id(session_id))
        # download_session() lays out `<dest>/values.json` + `<dest>/files/<path>` itself, so the
        # destination is the BUNDLE, not the bundle's files/ directory. Handing it files_root nests
        # a second files/ inside and every patch path the caller renders resolves to nothing.
        files_root = os.path.join(bundle, "files")
        os.makedirs(bundle, exist_ok=True)
        try:
            self._client.download_session(canonical_id, session_id, bundle)
        except Exception as e:
            # A knowledge document that names artifacts we could not fetch is worse than a loud
            # failure: the lane would hand an agent a patch path that resolves to nothing.
            raise KBStoreError("artifact download failed for %s: %s" % (session_id, str(e)[:160]))
        # A successful download is not proof the bundle is usable. An identity whose manifest was
        # never committed answers with values.json and nothing else, and the caller would only find
        # out when an agent opened an empty patch path. Check what the document itself promised.
        os.makedirs(files_root, exist_ok=True)   # a record with no artifacts still gets the layout
        promised = (knowledge.get("value") or {}).get("artifacts")
        for name in sorted(set((promised or {}).values())) if isinstance(promised, dict) else []:
            if not os.path.isfile(os.path.join(files_root, name)):
                raise KBStoreError(
                    "%s declares %s but %s holds no manifest for this session — the record was "
                    "written without committing its files" % (session_id, name, canonical_id))
        recipe = dict(knowledge)
        recipe.update({"canonical_id": canonical_id, "session_id": session_id,
                       "is_champion": bool(getattr(candidate, "is_champion", False)),
                       "champion": bool(getattr(candidate, "is_champion", False))})
        recipe.setdefault(self.metric, getattr(candidate, "speedup", None))
        with open(os.path.join(bundle, "recipe.json"), "w") as handle:
            json.dump(recipe, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return bundle

    # -- write ---------------------------------------------------------------------------

    def write(self, canonical_id: str, session_id: str, knowledge, files=None) -> str:
        """Record one candidate under an identity, uploading its artifacts at most once.

        `mode="replace"` so re-measuring the same patch updates that one session rather than
        merging two documents into a chimera — the session id is a digest of the patch, so a repeat
        genuinely IS the same candidate.
        """
        if not isinstance(knowledge, dict):
            raise KBStoreError("knowledge is not an object")
        session_id = validate_session_id(session_id)
        named = {safe_rel_path(rel): src for rel, src in (files or {}).items()}
        try:
            self._client.put_knowledge(canonical_id, knowledge, session_id=session_id,
                                       mode="replace")
            if named:
                self._client.put_files(canonical_id, session_id, [
                    (rel, named[rel], ARTIFACT_KIND, {}) for rel in sorted(named)])
        except Exception as e:
            raise KBStoreError("%s: %s" % (type(e).__name__, str(e)[:160]))
        return "%s/%s" % (canonical_id, session_id)

    def promote(self, canonical_id: str, session_id: str, speedup: float) -> None:
        self._client.set_champion(canonical_id, validate_session_id(session_id),
                                  metric=self.metric, value=float(speedup))

    def maybe_promote(self, canonical_id: str, session_id: str, speedup) -> bool:
        """Upstream's gate, verbatim: only a real win, and only over the incumbent."""
        speedup = finite_speedup(speedup)
        if speedup is None or speedup <= self.promote_floor:
            return False
        incumbent = self.champion_speedup(canonical_id)
        if incumbent is not None and speedup <= incumbent:
            return False
        try:
            self.promote(canonical_id, session_id, speedup)
        except Exception:
            return False
        return True


__all__ = ["ARTIFACT_KIND", "DEFAULT_SCAN", "RemoteKBStore"]
