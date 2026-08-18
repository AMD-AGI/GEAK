#!/usr/bin/env python3
"""Push `experience_store.py export-remote` output to a KernelForge KB Store.

    experience_store.py export-remote --root kb_artifacts --out /tmp/kb.jsonl
    kb_remote_upload.py --records /tmp/kb.jsonl                 # dry run: says what it would do
    kb_remote_upload.py --records /tmp/kb.jsonl --local <dir> --apply   # to the on-disk plane
    kb_remote_upload.py --records /tmp/kb.jsonl --apply         # to the service

The two planes take the SAME records, byte for byte. That is the whole point of --local: the
read/apply/optimize/write-back loop can be proven offline, and what proves it is that the service
would receive exactly what the local store received.

Split from the exporter on purpose. The exporter is pure and offline, so the whole mapping can be
reviewed and diffed before anything leaves the machine; this half is the only code that talks to
the network, and it does nothing until --apply.

Order per candidate is artifacts, then knowledge, then the champion pointer. That way a record is
never visible referencing bytes the store does not hold yet, and a run interrupted halfway leaves
uploaded-but-unreferenced blobs rather than a record pointing at nothing.

Needs the upstream client on PYTHONPATH (KernelForge `src/`, or the single vendored
kb_store_client.py) plus KB_STORE_URL / KB_STORE_TOKEN. The token is read from the environment and
never printed: --apply logs the canonical id and session id only.
"""

import argparse
import json
import os
import sys


def _load_client():
    """Import the upstream client, or explain precisely what is missing."""
    try:
        from kernel_agents.knowledge.remote_exp.kb_store_client import KBStoreClient, KBStoreError
        return KBStoreClient, KBStoreError
    except ImportError:
        pass
    try:  # a vendored copy of the single file, sitting next to this script or on PYTHONPATH
        from kb_store_client import KBStoreClient, KBStoreError  # type: ignore
        return KBStoreClient, KBStoreError
    except ImportError as e:
        raise SystemExit(
            "cannot import KBStoreClient: " + str(e) + "\n"
            "  put KernelForge's src/ on PYTHONPATH, or vendor "
            "kernel_agents/knowledge/remote_exp/kb_store_client.py next to this script"
        )


class _LocalBackend:
    """The on-disk store behind the client's write surface.

    Same three calls in the same order as the service path, so `upload_one` below does not know
    which plane it is writing to. The local store lands a session as one atomic unit, so the
    artifacts-then-knowledge ordering is belt and braces here — it matters upstream, where they
    are two round trips.
    """

    def __init__(self, root: str):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kb_store_local import LocalKBStore  # noqa: PLC0415 - optional, only for --local
        self.store = LocalKBStore(root)
        self.root = self.store.root
        self._staged = {}

    def put_files(self, cid, sid, entries):
        self._staged[(cid, sid)] = {rel: local for (rel, local, _kind, _meta) in entries}

    def put_knowledge(self, cid, knowledge, session_id="", mode="merge"):
        files = self._staged.pop((cid, session_id), {})
        return {"session_id": session_id,
                "path": self.store.write(cid, session_id, knowledge, files)}

    def set_champion(self, cid, sid, metric="speedup", value=0.0):
        self.store.promote(cid, sid, value)
        return {"session_id": sid, "metric": metric, "value": value}


def read_records(path: str):
    out = []
    with open(path, "r", errors="replace") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{n}: not JSON: {e}")
            # A summary line from a stdout capture is not a record; skip it rather than fail, so
            # `export-remote ... | tee` output works as an input without hand-editing.
            if not isinstance(rec, dict) or not rec.get("canonical_id"):
                continue
            out.append(rec)
    return out


def upload_one(store, rec: dict, *, apply: bool, quiet: bool) -> dict:
    cid, sid = rec["canonical_id"], rec["session_id"]
    files = rec.get("files") or []
    missing = [f["local_path"] for f in files if not os.path.isfile(f.get("local_path") or "")]
    if missing:
        return {"canonical_id": cid, "session_id": sid, "ok": False,
                "reason": "missing_local_file: " + missing[0]}
    plan = {"canonical_id": cid, "session_id": sid, "files": [f["path"] for f in files],
            "bytes": sum(int(f.get("size") or 0) for f in files),
            "speedup": (rec.get("knowledge") or {}).get("speedup"),
            "champion": bool(rec.get("champion"))}
    if not apply:
        return dict(plan, ok=True, applied=False)

    if files:
        store.put_files(cid, sid, [
            (f["path"], f["local_path"], f.get("kind") or "rewrite", {}) for f in files
        ])
    # replace, not merge: this exporter emits the entry's complete current state every time, so a
    # merge would keep fields a later curation pass deliberately removed.
    store.put_knowledge(cid, rec["knowledge"], session_id=sid, mode="replace")
    if rec.get("champion") and rec.get("champion_eligible"):
        store.set_champion(cid, sid, metric="speedup", value=float(plan["speedup"] or 0.0))
    if not quiet:
        print(json.dumps(dict(plan, ok=True, applied=True), ensure_ascii=False))
    return dict(plan, ok=True, applied=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True, help="JSON lines from export-remote")
    p.add_argument("--local", default="", metavar="DIR",
                   help="write to an on-disk KB store at DIR instead of the service")
    p.add_argument("--apply", action="store_true", help="actually upload; default is a dry run")
    p.add_argument("--limit", type=int, default=0, help="stop after N candidates (smoke tests)")
    p.add_argument("--quiet", action="store_true", help="summary line only")
    a = p.parse_args(argv)

    records = read_records(a.records)
    if a.limit > 0:
        records = records[: a.limit]

    store = None
    if a.apply and a.local:
        store = _LocalBackend(a.local)
    elif a.apply:
        KBStoreClient, _KBStoreError = _load_client()
        if not (os.environ.get("KB_STORE_URL") or "").strip():
            raise SystemExit("KB_STORE_URL is not set; refusing to --apply")
        store = KBStoreClient.from_env()

    ok = failed = 0
    failures = []
    for rec in records:
        try:
            result = upload_one(store, rec, apply=a.apply, quiet=a.quiet)
        except Exception as e:  # one bad candidate must not abandon the rest of the backlog
            result = {"canonical_id": rec.get("canonical_id"), "session_id": rec.get("session_id"),
                      "ok": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        if result.get("ok"):
            ok += 1
            if not a.apply and not a.quiet:
                print(json.dumps(result, ensure_ascii=False))
        else:
            failed += 1
            failures.append(result)
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)

    print(json.dumps({"applied": bool(a.apply),
                      "plane": "local" if a.local else "service",
                      "root": getattr(store, "root", "") if a.local else "",
                      "candidates": len(records), "ok": ok, "failed": failed,
                      "champions": sum(1 for r in records if r.get("champion")),
                      "identities": len({r["canonical_id"] for r in records}),
                      "bytes": sum(int(f.get("size") or 0) for r in records
                                   for f in (r.get("files") or []))}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
