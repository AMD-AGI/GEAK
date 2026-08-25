"""Opening a KB plane — the one place both lanes turn a CLI namespace into a store to read or write.

`open_plane(a, metric, floor, create)` returns `(primary, mirror_or_None, why)` for one plane:

    plane=local   -> (LocalKBStore, None, "")            reads/writes the on-disk store at a.store
    plane=remote  -> (RemoteKBStore, None, "")           reads/writes the service
    plane=both    -> (LocalKBStore, RemoteKBStore, why)  local is the source of truth; the mirror
                                                         is reported, never fatal

The kernel lane ranks on one metric (`speedup`) and passes `(CHAMPION_METRIC, 1.0)`; the e2e lane
ranks each rung on its own metric (throughput on the exact rung, speedup on the coarser ones) and
passes the rung's `(metric, floor)`. That per-metric parameter is the ONLY thing that used to make
these two openers different functions.

`--plane both` writes locally FIRST and remotely second, and a remote failure surfaces as `why`
(`remote_unavailable: ...`) rather than as a refusal: the run already spent GPU hours producing the
measurement, and a network blip must not discard it. Without that field an unreachable service
would look exactly like a successful write.
"""

import os


def open_plane(a, metric, floor, create=False):
    """(primary, mirror_or_None, why). See module docstring.

    A missing store is a hard miss when reading — a typo'd path must not read as an empty store and
    quietly cold-start a run that had experience waiting. Writing creates it, because the first
    write into a fresh store is the normal case, not an error.
    """
    plane = str(getattr(a, "plane", "local") or "local")

    local = None
    if plane in ("local", "both"):
        local, why = _open_local(str(getattr(a, "store", "") or ""), metric, floor, create)
        if plane == "local":
            return local, None, why
        if local is None:
            return None, None, why

    # plane in ("remote", "both")
    try:
        from kb.store_remote import RemoteKBStore
    except ImportError as e:
        return None, None, "store_unavailable: " + str(e)[:120]
    scan = getattr(a, "scan", 25)
    remote, remote_why = RemoteKBStore.from_env(scan, metric, floor)
    if plane == "remote":
        return remote, None, remote_why
    return local, remote, ("remote_unavailable: " + remote_why if remote is None else "")


def _open_local(root, metric, floor, create):
    """The on-disk store, or (None, reason). Imported lazily so a box that only has this file can
    still open a remote plane."""
    try:
        from kb.store_local import LocalKBStore
    except ImportError as e:
        return None, "store_unavailable: " + str(e)[:120]
    if not root or not os.path.isdir(root):
        if not create:
            return None, "no_such_store: " + root
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as e:
            return None, "unusable_store: " + str(e)[:120]
    return LocalKBStore(root, metric=metric, promote_floor=floor), ""
