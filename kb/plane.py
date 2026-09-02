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

That ordering is a WRITE contract. A read wants the opposite preference and gets it from
`read_planes()` below; see its docstring.
"""

import copy
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


def read_planes(a, metric):
    """The planes a READ should try, in order: [(store, plane_name)], plus why one is missing.

    A read takes ONE plane at a time. Merging two rankings would need a cross-plane comparability
    rule that nothing here has, so `both` has to CHOOSE — and `open_plane`'s choice is the wrong one
    for a read. It hands back the local store as the primary because that is the write path's source
    of truth, which means `--plane both` on a read silently answered from disk and let a stale mirror
    shadow the service. So the read order is the service FIRST, the mirror only when the service has
    no answer, and the caller reports which one spoke (`read_plane` in the resolve output).

    "No answer" means no candidates, not merely an error: on a scheme with no search an empty remote
    page and a 404 are the same response, and the mirror may hold a hand-curated history that a thin
    remote page must not shadow. Deciding that is the caller's job — it is the one that knows what a
    candidate is — so this returns both planes and lets it stop at the first that answers.

    No `floor` parameter, unlike `open_plane`: a floor only ever gates a promotion, and a read never
    performs one.

    Both lanes' JS used to spell this branch out in emitted bash, once each, testing $KB_STORE_TOKEN
    and re-running the whole resolve. Those copies could not fix the CLI, which had no branch at all.
    """
    plane = str(getattr(a, "plane", "local") or "local")
    if plane != "both":
        store, _mirror, why = open_plane(a, metric, 1.0)
        return ([(store, plane)] if store is not None else []), why
    # NOT open_plane's `both` branch: that one refuses outright when the local store is missing,
    # which is right for a write (it was asked to file in two places) and wrong for a read, where it
    # turns "this box keeps no mirror" into a cold start against a service that had the answer.
    out, why = [], ""
    for name in ("remote", "local"):
        store, _mirror, one_why = open_plane(_with_plane(a, name), metric, 1.0)
        if store is not None:
            out.append((store, name))
        elif not why:
            why = one_why
    return out, ("" if out else why)


def _with_plane(a, plane):
    """`a` with one field overridden. A shallow copy because argparse namespaces are flat and the
    caller must not see its own namespace mutated by a read."""
    clone = copy.copy(a)
    clone.plane = plane
    return clone


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
