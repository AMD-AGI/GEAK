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
