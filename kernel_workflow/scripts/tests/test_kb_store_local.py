"""Tests for the on-disk KB Store (kernel_workflow/scripts/kb_store_local.py).

This plane exists to be swapped for the KernelForge service without changing behaviour, so what is
pinned here is the contract the service defines, not this implementation's conveniences:
  - the address: the canonical id, split on ':', IS the directory path;
  - the ranking: `speedup` descending and nothing else, with anything unrankable read as absent;
  - the two write outcomes: a repeated session id updates one candidate, a new one appends;
  - the champion gate: only a real win, and only over the incumbent;
  - the cost model: `candidates()` reads knowledge documents and no artifact bytes.

The last case cross-checks the tree against upstream's own reader when a KernelForge checkout is
importable, and skips otherwise — the point is to catch drift, not to add a dependency.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_store_local import KBStoreError, LocalKBStore  # noqa: E402

CID = "kernel:geak:fused_moe_kernel:rocm:7.2:triton:mi355x"


def knowledge(speedup=2.0, direction="tile-retune", name="fused_moe_kernel"):
    """The four-key document upstream writes; `value` is the producer's own and opaque here."""
    return {"producer": "geak", "speedup": speedup,
            "identity": {"producer": "geak", "kernel_name": name, "framework": "rocm",
                         "framework_version": "7.2", "backend": "triton", "gpu": "mi355x"},
            "value": {"direction": direction, "kernel_name": name}}


def artifacts(tmp_path, tag="a", text="patch body\n"):
    patch = tmp_path / f"{tag}.diff"
    patch.write_text(text)
    report = tmp_path / f"{tag}.md"
    report.write_text(f"# report {tag}\n")
    return {"patch.diff": str(patch), "report.md": str(report)}


# --------------------------------------------------------------------------- addressing

def test_the_canonical_id_is_the_path(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "geak-fused_moe_kernel-aaaa-bbbb", knowledge(), artifacts(tmp_path))
    session = (tmp_path / "store" / "kernel" / "geak" / "fused_moe_kernel" / "rocm" / "7.2"
               / "triton" / "mi355x" / "sessions" / "geak-fused_moe_kernel-aaaa-bbbb")
    assert (session / "knowledge.json").is_file()
    assert sorted(os.listdir(session / "files")) == ["patch.diff", "report.md"]
    document = json.loads((session / "knowledge.json").read_text())
    assert sorted(document) == ["identity", "producer", "speedup", "value"]
    assert store.identities() == [CID]


@pytest.mark.parametrize("bad", ["", "kernel", "kernel:GEAK:k", "kernel:geak:k:../etc", "kernel::k"])
def test_an_unusable_identity_is_refused_not_normalized(tmp_path, bad):
    with pytest.raises(KBStoreError):
        LocalKBStore(tmp_path).identity_dir(bad)


@pytest.mark.parametrize("bad", ["../escape.diff", "/abs.diff", "a/../../b.diff", "a:b.diff", ""])
def test_an_artifact_cannot_escape_its_session(tmp_path, bad):
    source = tmp_path / "src.diff"
    source.write_text("x")
    with pytest.raises(KBStoreError):
        LocalKBStore(tmp_path / "store").write(CID, "sid-1", knowledge(), {bad: str(source)})


@pytest.mark.parametrize("bad", ["", ".hidden", "has space", "a" * 129, "sid/../x"])
def test_an_unusable_session_id_is_refused(tmp_path, bad):
    with pytest.raises(KBStoreError):
        LocalKBStore(tmp_path / "store").write(CID, bad, knowledge(), {})


# --------------------------------------------------------------------------- ranking

def test_candidates_rank_on_speedup_alone(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    for sid, speedup in [("sid-low", 1.2), ("sid-high", 4.5), ("sid-mid", 2.0)]:
        store.write(CID, sid, knowledge(speedup=speedup), {})
    assert [c.session_id for c in store.candidates(CID, limit=0)] == ["sid-high", "sid-mid", "sid-low"]
    assert [c.session_id for c in store.candidates(CID, limit=2)] == ["sid-high", "sid-mid"]


@pytest.mark.parametrize("unrankable", [None, True, "1.5", {"speedup": 9}, float("nan")])
def test_a_speedup_the_store_cannot_rank_reads_as_absent(tmp_path, unrankable):
    """`True` would sort as 1.0 and "1.5" would raise; both mean the producer wrote something else."""
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-real", knowledge(speedup=1.1), {})
    store.write(CID, "sid-odd", knowledge(speedup=unrankable), {})
    ranked = store.candidates(CID, limit=0)
    assert [c.session_id for c in ranked] == ["sid-real", "sid-odd"]
    assert ranked[1].speedup is None


def test_the_store_does_not_know_what_a_bench_key_is(tmp_path):
    """Comparability is the caller's job — `resolve-remote` filters it, the store must not."""
    store = LocalKBStore(tmp_path / "store")
    slow = knowledge(speedup=1.5)
    slow["value"]["metric"] = {"bench_key": "b:imported"}
    fast = knowledge(speedup=40.0)
    fast["value"]["metric"] = {"bench_key": "b2:this-box"}
    store.write(CID, "sid-imported", slow, {})
    store.write(CID, "sid-onbox", fast, {})
    assert [c.session_id for c in store.candidates(CID, limit=0)] == ["sid-onbox", "sid-imported"]


def test_ties_rank_the_same_way_on_every_read(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    for sid in ("sid-c", "sid-a", "sid-b"):
        store.write(CID, sid, knowledge(speedup=2.0), {})
    assert [c.session_id for c in store.candidates(CID, limit=0)] == ["sid-a", "sid-b", "sid-c"]


def test_a_cold_identity_is_empty_not_an_error(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    assert store.candidates(CID, limit=3) == []
    assert store.champion(CID) == {}
    assert store.champion_speedup(CID) is None
    assert store.get_session(CID, "sid-missing") is None
    assert store.read_bytes(CID, "sid-missing", "patch.diff") == b""


def test_a_half_written_document_is_a_miss_not_a_crash(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-ok", knowledge(speedup=3.0), {})
    store.write(CID, "sid-broken", knowledge(speedup=9.0), {})
    (tmp_path / "store" / "kernel" / "geak" / "fused_moe_kernel" / "rocm" / "7.2" / "triton"
     / "mi355x" / "sessions" / "sid-broken" / "knowledge.json").write_text("{not json")
    assert [c.session_id for c in store.candidates(CID, limit=0)] == ["sid-ok"]


def test_ranking_reads_no_artifact_bytes(tmp_path, monkeypatch):
    """A 240KB patch must not be paid for until a candidate is actually selected."""
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(speedup=2.0), artifacts(tmp_path, "a"))
    store.write(CID, "sid-2", knowledge(speedup=3.0), artifacts(tmp_path, "b"))

    real_open = open

    def guard(path, *args, **kw):
        assert "/files/" not in str(path), f"candidates() read an artifact: {path}"
        return real_open(path, *args, **kw)

    monkeypatch.setattr("builtins.open", guard)
    assert len(store.candidates(CID, limit=0)) == 2


# --------------------------------------------------------------------------- write semantics

def test_the_same_session_id_updates_one_candidate(tmp_path):
    """Session ids are content-addressed upstream, so a remeasure must not grow the store."""
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(speedup=2.0), artifacts(tmp_path, "a", "first\n"))
    store.write(CID, "sid-1", knowledge(speedup=2.4), artifacts(tmp_path, "a2", "second\n"))
    ranked = store.candidates(CID, limit=0)
    assert len(ranked) == 1 and ranked[0].speedup == 2.4
    assert store.read_bytes(CID, "sid-1", "patch.diff") == b"second\n"


def test_a_different_session_id_appends_under_the_same_key(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(speedup=2.0), {})
    store.write(CID, "sid-2", knowledge(speedup=2.1), {})
    assert len(store.candidates(CID, limit=0)) == 2
    assert store.identities() == [CID]


def test_a_rewrite_drops_artifacts_the_new_record_no_longer_carries(tmp_path):
    """The session lands as one unit, so a stale file must not survive underneath it."""
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(), artifacts(tmp_path, "a"))
    only_patch = {"patch.diff": artifacts(tmp_path, "b")["patch.diff"]}
    store.write(CID, "sid-1", knowledge(), only_patch)
    assert store.session_files(CID, "sid-1") == ["patch.diff"]


def test_knowledge_must_be_a_document(tmp_path):
    with pytest.raises(KBStoreError):
        LocalKBStore(tmp_path / "store").write(CID, "sid-1", ["not", "a", "document"], {})


# --------------------------------------------------------------------------- champion

def test_the_champion_moves_only_on_a_real_win_over_the_incumbent(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    for sid, speedup in [("sid-loss", 0.9), ("sid-tie", 1.0), ("sid-win", 1.8), ("sid-worse", 1.4)]:
        store.write(CID, sid, knowledge(speedup=speedup), {})
    assert store.maybe_promote(CID, "sid-loss", 0.9) is False    # not faster than its own baseline
    assert store.maybe_promote(CID, "sid-tie", 1.0) is False     # a tie is not a win
    assert store.champion(CID) == {}
    assert store.maybe_promote(CID, "sid-win", 1.8) is True
    assert store.maybe_promote(CID, "sid-worse", 1.4) is False   # loses to the incumbent
    assert store.champion_speedup(CID) == 1.8
    assert [c.session_id for c in store.candidates(CID, limit=0) if c.is_champion] == ["sid-win"]


@pytest.mark.parametrize("unrankable", [None, True, "2.0"])
def test_an_unrankable_speedup_never_takes_the_champion_pointer(tmp_path, unrankable):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(speedup=unrankable), {})
    assert store.maybe_promote(CID, "sid-1", unrankable) is False
    assert store.champion(CID) == {}


def test_a_champion_written_under_another_metric_is_not_read_as_a_speedup(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(), {})
    store.promote(CID, "sid-1", 3.0)
    path = os.path.join(store.identity_dir(CID), "champion.json")
    with open(path, "w") as handle:
        json.dump({"session_id": "sid-1", "metric": "latency_ms", "value": 3.0}, handle)
    assert store.champion_speedup(CID) is None


# --------------------------------------------------------------------------- materialize

def test_materialize_lays_out_the_bundle_a_caller_can_apply(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(speedup=2.5), artifacts(tmp_path, "a", "the patch\n"))
    store.promote(CID, "sid-1", 2.5)
    bundle = store.materialize(CID, store.candidates(CID, limit=1)[0], str(tmp_path / "cache"))
    assert sorted(os.listdir(bundle)) == ["files", "recipe.json"]
    assert open(os.path.join(bundle, "files", "patch.diff")).read() == "the patch\n"
    recipe = json.loads(open(os.path.join(bundle, "recipe.json")).read())
    assert recipe["canonical_id"] == CID and recipe["session_id"] == "sid-1"
    assert recipe["is_champion"] is True and recipe["speedup"] == 2.5


def test_materialize_accepts_a_bare_session_id(tmp_path):
    """The read path has a session id long before it has a Candidate object."""
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(), artifacts(tmp_path, "a"))
    bundle = store.materialize(CID, "sid-1", str(tmp_path / "cache"))
    assert os.path.isfile(os.path.join(bundle, "files", "patch.diff"))


def test_materialize_refuses_a_candidate_that_is_not_there(tmp_path):
    with pytest.raises(KBStoreError):
        LocalKBStore(tmp_path / "store").materialize(CID, "sid-missing", str(tmp_path / "cache"))


def test_materializing_twice_replaces_rather_than_merges(tmp_path):
    store = LocalKBStore(tmp_path / "store")
    store.write(CID, "sid-1", knowledge(), artifacts(tmp_path, "a"))
    dest = str(tmp_path / "cache")
    store.materialize(CID, "sid-1", dest)
    store.write(CID, "sid-1", knowledge(), {"patch.diff": artifacts(tmp_path, "b")["patch.diff"]})
    bundle = store.materialize(CID, "sid-1", dest)
    assert os.listdir(os.path.join(bundle, "files")) == ["patch.diff"]
    assert not [n for n in os.listdir(dest) if n.startswith(".")], "staging dirs must not be left behind"


# --------------------------------------------------------------------------- upstream parity

def test_upstream_reads_the_tree_we_write():
    """Guards the one thing this file cannot prove on its own: that the shape is still theirs."""
    for candidate_src in ("/tmp/KernelForge/src",):
        if os.path.isdir(candidate_src) and candidate_src not in sys.path:
            sys.path.insert(0, candidate_src)
    record_store = pytest.importorskip("kernel_agents.rewrite_by_flydsl.record_store")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ours = LocalKBStore(os.path.join(tmp, "store"))
        patch = os.path.join(tmp, "p.diff")
        with open(patch, "w") as handle:
            handle.write("the patch\n")
        ours.write(CID, "sid-low", knowledge(speedup=1.5), {"patch.diff": patch})
        ours.write(CID, "sid-high", knowledge(speedup=3.5), {"patch.diff": patch})
        ours.promote(CID, "sid-high", 3.5)

        theirs = record_store.LocalRewriteRecords(os.path.join(tmp, "store"))
        ranked = theirs.candidates(CID, limit=3)
        assert [(c.session_id, c.speedup, c.is_champion) for c in ranked] == [
            ("sid-high", 3.5, True), ("sid-low", 1.5, False)]
        assert theirs.champion_speedup(CID) == 3.5
        bundle = theirs.materialize(CID, ranked[0], os.path.join(tmp, "cache"))
        assert (bundle / "files" / "patch.diff").read_text() == "the patch\n"
