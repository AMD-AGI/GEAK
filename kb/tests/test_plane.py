#!/usr/bin/env python3
"""Which store a read opens, and in which order.

`open_plane` orders `both` as (local, remote) because that is the WRITE contract — file locally
first, mirror second, and never lose a measurement to a network blip. A read that reuses that order
answers from disk, which means a stale local mirror silently shadows the shared service. That is
what `read_planes` exists to correct, so these tests pin the order itself rather than any one
caller's use of it.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb import plane as plane_mod                                               # noqa: E402
from kb.plane import open_plane, read_planes                                    # noqa: E402


def _args(**kw):
    kw.setdefault("scan", 25)
    return types.SimpleNamespace(**kw)


class _FakeRemote:
    """Stands in for the service. `from_env` is the only entry point read_planes touches."""

    def __init__(self, ok=True, why=""):
        self.ok, self.why = ok, why

    def install(self, monkeypatch):
        remote = self

        class _Store:
            pass

        module = types.ModuleType("kb.store_remote")

        class RemoteKBStore:
            @staticmethod
            def from_env(scan, metric, floor):
                return (_Store() if remote.ok else None), remote.why

        module.RemoteKBStore = RemoteKBStore
        monkeypatch.setitem(sys.modules, "kb.store_remote", module)


def _names(planes):
    return [name for _store, name in planes]


# -- a single plane is passed straight through ------------------------------------------------


def test_local_only_opens_the_local_store(tmp_path, monkeypatch):
    _FakeRemote(ok=False, why="no_token").install(monkeypatch)
    planes, why = read_planes(_args(plane="local", store=str(tmp_path)), "speedup")
    assert _names(planes) == ["local"] and why == ""


def test_a_missing_local_store_is_a_miss_not_an_empty_read(tmp_path, monkeypatch):
    """A typo'd path must not read as a cold start. Same rule as open_plane's."""
    _FakeRemote(ok=False).install(monkeypatch)
    planes, why = read_planes(_args(plane="local", store=str(tmp_path / "nope")), "speedup")
    assert planes == [] and why.startswith("no_such_store")


def test_remote_only_never_touches_disk(tmp_path, monkeypatch):
    _FakeRemote().install(monkeypatch)
    planes, why = read_planes(_args(plane="remote", store=str(tmp_path / "nope")), "speedup")
    assert _names(planes) == ["remote"] and why == ""


# -- `both` is where a read and a write disagree ------------------------------------------------


def test_both_puts_the_service_first_which_is_the_opposite_of_a_write(tmp_path, monkeypatch):
    """The whole point. open_plane hands back local as the primary; a read must try remote first."""
    _FakeRemote().install(monkeypatch)
    a = _args(plane="both", store=str(tmp_path))
    primary, mirror, _why = open_plane(a, "speedup", 1.0)
    assert primary is not None and mirror is not None      # write order: local, then remote
    assert _names(read_planes(a, "speedup")[0]) == ["remote", "local"]


def test_both_still_offers_the_mirror_so_a_dead_service_is_not_a_cold_start(tmp_path, monkeypatch):
    _FakeRemote(ok=False, why="no_token").install(monkeypatch)
    planes, why = read_planes(_args(plane="both", store=str(tmp_path)), "speedup")
    assert _names(planes) == ["local"] and why == ""


def test_both_reads_the_service_even_when_this_box_keeps_no_mirror(tmp_path, monkeypatch):
    """open_plane REFUSES here — right for a write, which was asked to file in two places, and wrong
    for a read, where it turns "no local store" into a cold start against a service that had it."""
    _FakeRemote().install(monkeypatch)
    a = _args(plane="both", store=str(tmp_path / "nope"))
    assert open_plane(a, "speedup", 1.0)[0] is None
    assert _names(read_planes(a, "speedup")[0]) == ["remote"]


def test_both_with_neither_plane_reports_why(tmp_path, monkeypatch):
    _FakeRemote(ok=False, why="no_token").install(monkeypatch)
    planes, why = read_planes(_args(plane="both", store=str(tmp_path / "nope")), "speedup")
    assert planes == [] and why


def test_the_callers_namespace_is_not_mutated_by_a_read(tmp_path, monkeypatch):
    """`both` is served by re-opening under a per-plane clone. If that clone were the caller's own
    namespace, one read would silently rewrite the plane every later write uses."""
    _FakeRemote().install(monkeypatch)
    a = _args(plane="both", store=str(tmp_path))
    read_planes(a, "speedup")
    assert a.plane == "both"


def test_a_read_never_carries_a_promotion_floor(tmp_path, monkeypatch):
    """A floor gates a promotion; a read performs none. Pinned because passing the rung's floor
    through would make a coarse rung's 1.0x bar quietly filter the offer as well."""
    seen = []

    class _Recorder(_FakeRemote):
        def install(self, monkeypatch):
            module = types.ModuleType("kb.store_remote")

            class RemoteKBStore:
                @staticmethod
                def from_env(scan, metric, floor):
                    seen.append(floor)
                    return object(), ""

            module.RemoteKBStore = RemoteKBStore
            monkeypatch.setitem(sys.modules, "kb.store_remote", module)

    _Recorder().install(monkeypatch)
    read_planes(_args(plane="remote", store=str(tmp_path)), "speedup")
    assert seen == [1.0]
    assert "floor" not in plane_mod.read_planes.__code__.co_varnames
