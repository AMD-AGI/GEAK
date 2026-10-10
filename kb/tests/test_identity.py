#!/usr/bin/env python3
"""The address arithmetic, and specifically the version cut that decides which page a run lands on.

An identity function has no failure mode that looks like one: a wrong dimension is not an error, it
is a cold start at an address nobody else writes to. So what is pinned here is the two directions the
`framework_version` cut can go wrong. Too fine (the build string) and every rebuilt wheel opens a
fresh empty page — the miss #438 was about. Too coarse (`0.5`) and two SGLang releases with different
kernels share a page, which is worse than the miss because the reader cannot tell.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb import identity as kbid                                                 # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    # The shape that motivated the cut: a dev wheel built from a tagged release.
    ("0.5.15.post1.dev20260723+g6c9fd0adc5", "0.5.15"),
    ("0.5.15", "0.5.15"),
    ("v0.5.15", "0.5.15"),                  # a `v` prefix is decoration, not a different release
    (" 0.5.15 ", "0.5.15"),
    ("0.5.15rc1", "0.5.15"),                # a release candidate is that release, for addressing
    # Fewer than three components is what the version HAS, not something to pad: 0.5 and 0.5.0 are
    # different strings upstream and inventing the third would file at an address nobody writes.
    ("0.5", "0.5"),
    ("2", "2"),
])
def test_the_release_is_three_components_at_most(raw, expected):
    assert kbid._release_version(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_an_unobserved_version_is_named_never_guessed(raw):
    """`unspecified` is a real page — the one an entry whose stack was never recorded lands on. A
    default of "0.0.0" or "" would instead file it among records that state a version."""
    assert kbid._release_version(raw) == kbid.UNKNOWN_VERSION


def test_a_version_that_does_not_parse_is_kept_not_dropped():
    """A stack that spells its version some way this regex does not model still gets ONE stable page
    of its own, which is the property that matters — sharing `unspecified` with the unobserved ones
    would merge two states that mean opposite things."""
    assert kbid._release_version("nightly-main") == "nightly-main"


def test_two_releases_do_not_share_a_page():
    """The reason the cut keeps three components and not two."""
    assert kbid._release_version("0.5.15") != kbid._release_version("0.5.17")


def test_the_cut_is_idempotent():
    """Re-addressing an already-cut version must not move it: read and write call this at different
    times over the same string, and a second application that changed anything would split the page."""
    once = kbid._release_version("0.5.15.post1.dev20260723+g6c9fd0adc5")
    assert kbid._release_version(once) == once


def test_the_e2e_address_carries_the_cut_version_and_the_record_keeps_the_rest():
    """The bargain the docstring states: only the ADDRESS is coarse. Every rung is built from the cut
    value, and no rung drops it — which is why the read side needs a legacy rung for what was filed
    before the cut (e2e_store.legacy_version_ladder)."""
    identity = kbid.e2e_identity("Kimi-K3", "gfx950", "sglang",
                                 "0.5.15.post1.dev20260723+g6c9fd0adc5", "mxfp4",
                                 tp=8, isl=1024, osl=1024, conc=64)
    assert identity["framework_version"] == "0.5.15"
    cids = kbid.e2e_canonical_ids(identity)
    assert len(cids) == 3 and all(":0.5.15:" in c for c in cids)
    assert not any("dev20260723" in c for c in cids)


# --- the agentx workload segment and the ep dimension ---------------------------------------
#
# On a trace replay the sequence lengths are an OBSERVED statistic of the run, not a knob it was
# given. Addressing on them means the address tracks measurement noise, and the first rung — the
# one `session_id` fingerprints — then mints a new page per run: the reader stops there and sees a
# cold start, and the rebench-replaces-the-record rule in e2e_store never fires.

_AGENTX = dict(model="M", gfx="gfx950", framework="vllm", framework_version="0.26.0",
               precision="fp8", tp=8, ep=1, conc=10,
               workload_kind=kbid.WORKLOAD_KIND_AGENTX)


def _agentx(isl, osl):
    """The same run, measured twice — only the observed shape differs."""
    return kbid.e2e_identity(isl=isl, osl=osl, **_AGENTX)


def test_an_agentx_address_survives_two_different_observed_shapes():
    """The headline. Two replays of one corpus observe different mean lengths; they are one page."""
    assert kbid.e2e_canonical_ids(_agentx(146713, 1109)) == \
           kbid.e2e_canonical_ids(_agentx(89000, 900))


def test_the_agentx_session_fingerprint_is_stable_across_runs():
    """What the stable address buys, and the more expensive half of the bug it fixes: the session id
    is taken over the most specific rung, so a drifting rung silently resets the attestation ledger
    (e2e_store._content_digest never finds the record it is supposed to replace)."""
    a, b = _agentx(146713, 1109), _agentx(89000, 900)
    assert kbid.session_id(kbid.e2e_canonical_ids(a)[0], "model-x", "digest-x") == \
           kbid.session_id(kbid.e2e_canonical_ids(b)[0], "model-x", "digest-x")


def test_the_agentx_workload_segment_names_the_kind_and_the_concurrency():
    cids = kbid.e2e_canonical_ids(_agentx(146713, 1109))
    assert cids[0].endswith("tp_8:ep_1:wl_agentx:conc_10")
    assert not any("isl_" in c or "osl_" in c for c in cids)


def test_the_synthetic_address_is_byte_identical_to_the_pre_ep_scheme():
    """A synthetic run states its shape, so it keeps addressing on it — and a caller that names no
    ep gets the address it got before ep existed. Every record already on disk was written this way."""
    identity = kbid.e2e_identity("M", "gfx950", "vllm", "0.26.0", "fp8",
                                 tp=8, isl=1024, osl=1024, conc=64)
    assert kbid.e2e_canonical_ids(identity)[0] == \
        "geak:e2e:m:gfx950:vllm:0.26.0:fp8:tp_8:isl_1024:osl_1024:conc_64"


def test_ep_rides_the_tp_rung_and_is_never_dropped_alone():
    """ep folds into the tp rung rather than adding a fourth. A standalone `...:tp_8` page would
    hold ep=1 and ep=8 runs with nothing in the address to tell them apart — the exact confusion the
    module docstring says the scheme cannot represent. The coarse rung already means "any parallelism"."""
    cids = kbid.e2e_canonical_ids(_agentx(146713, 1109))
    assert len(cids) == 3
    assert cids[1].endswith("tp_8:ep_1")
    assert not any(c.endswith(":tp_8") for c in cids)


def test_ep_without_tp_creates_no_rung():
    """ep qualifies a tp split; on its own it has nothing to qualify."""
    cids = kbid.e2e_canonical_ids(
        kbid.e2e_identity("M", "gfx950", "vllm", "0.26.0", "fp8", ep=8))
    assert len(cids) == 1


def test_an_agentx_run_with_no_conc_drops_only_the_workload_rung():
    cids = kbid.e2e_canonical_ids(kbid.e2e_identity(
        "M", "gfx950", "vllm", "0.26.0", "fp8", tp=8, ep=1,
        workload_kind=kbid.WORKLOAD_KIND_AGENTX))
    assert len(cids) == 2 and cids[0].endswith("tp_8:ep_1")
