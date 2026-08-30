#!/usr/bin/env python3
"""Ranked candidates -> one offer per idea, with suspect records out of the way.

`collapse_by_direction` keeps only the FIRST entry of each direction group, which is what makes the
ordering handed to it consequential in a way a plain ranking is not: a record that merely outranks
its group does not lead it, it DELETES it. `demote_hinted` is the fix, and these tests pin the two
together — the GLM-5.2-MXFP4 page had a suspect record and a validated one both filed under
`kernels`, and the suspect one's higher raw number evicted the validated one from every read.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb.curate import collapse_by_direction, demote_hinted                      # noqa: E402


def _item(name, direction="", hint=""):
    return {"id": name, "direction": direction, "hint": hint}


def _hinted(i):
    return i["hint"]


def _ids(items):
    return [i["id"] for i in items]


# -- demotion -------------------------------------------------------------------------------------


def test_hinted_records_go_behind_every_clean_one():
    out = demote_hinted([_item("a", hint="suspect"), _item("b"), _item("c", hint="suspect"),
                         _item("d")], _hinted)
    assert _ids(out) == ["b", "d", "a", "c"]


def test_relative_order_survives_inside_both_partitions():
    """A stable partition, not a sort: whatever the caller ranked on still decides the order within
    each half, so neither lane has to agree with the other about the scalar."""
    items = [_item(str(n), hint="x" if n % 2 else "") for n in range(6)]
    assert _ids(demote_hinted(items, _hinted)) == ["0", "2", "4", "1", "3", "5"]


def test_all_hinted_or_none_hinted_is_the_identity():
    clean = [_item("a"), _item("b")]
    assert _ids(demote_hinted(clean, _hinted)) == ["a", "b"]
    dirty = [_item("a", hint="x"), _item("b", hint="x")]
    assert _ids(demote_hinted(dirty, _hinted)) == ["a", "b"]


def test_the_input_list_is_not_mutated():
    items = [_item("a", hint="x"), _item("b")]
    demote_hinted(items, _hinted)
    assert _ids(items) == ["a", "b"]


# -- the reason it exists -------------------------------------------------------------------------


def test_a_suspect_record_no_longer_deletes_its_direction_group():
    """The GLM page, in miniature. Ranked on the raw number the suspect record leads `kernels` and
    the validated one is never offered; demoted first, the group is represented by the good one and
    the suspect one rides along as its alternate — still readable, no longer occupying the slot."""
    ranked = [_item("suspect", "kernels", hint="3 attempts could not reproduce it"),
              _item("validated", "kernels")]

    top, alts, collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                 lambda i: i["id"], 3)
    assert _ids(top) == ["suspect"] and _ids(alts[0]) == ["validated"] and collapsed == 1

    top, alts, collapsed = collapse_by_direction(demote_hinted(ranked, _hinted),
                                                 lambda i: i["direction"], lambda i: i["id"], 3)
    assert _ids(top) == ["validated"] and _ids(alts[0]) == ["suspect"] and collapsed == 1


def test_demotion_costs_a_hinted_record_its_slot_only_when_something_contests_it():
    """A hinted record alone on its direction is still the best answer anyone has. Demotion is not
    filtering; only retraction hides a record."""
    ranked = demote_hinted([_item("a", "kernels", hint="x"), _item("b", "flags")], _hinted)
    top, _alts, _collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                   lambda i: i["id"], 3)
    assert sorted(_ids(top)) == ["a", "b"]


def test_demotion_can_push_a_direction_out_of_top_n_entirely():
    """The one real cost: with more ideas than slots, a hinted group loses to a clean one. That is
    the trade — a slot is a full on-box verify, and spending it on suspect evidence is the thing
    being avoided."""
    ranked = demote_hinted([_item("a", "one", hint="x"), _item("b", "two"), _item("c", "three")],
                           _hinted)
    top, _alts, _collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                   lambda i: i["id"], 2)
    assert _ids(top) == ["b", "c"]


# -- grouping -------------------------------------------------------------------------------------


def test_an_undirected_entry_is_its_own_group():
    ranked = [_item("a"), _item("b"), _item("c", "kernels"), _item("d", "kernels")]
    top, alts, collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                 lambda i: i["id"], 5)
    assert _ids(top) == ["a", "b", "c"] and _ids(alts[2]) == ["d"] and collapsed == 1


def test_direction_matching_ignores_case_and_padding():
    ranked = [_item("a", "Kernels"), _item("b", " kernels ")]
    top, alts, _collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                  lambda i: i["id"], 5)
    assert _ids(top) == ["a"] and _ids(alts[0]) == ["b"]


def test_top_n_counts_ideas_and_falls_back_rather_than_to_nothing():
    """0/None mean "unset" and take the default 3, not "offer nothing" — a caller that forgot the
    flag should get the usual page, and there is no way to ask for an empty offer."""
    ranked = [_item(str(n), "d%d" % n) for n in range(4)]
    for n, expect in ((0, 3), (None, 3), (1, 1), (3, 3), (99, 4)):
        top, _alts, _collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                       lambda i: i["id"], n)
        assert len(top) == expect, n


def test_collapsed_counts_only_what_the_chosen_groups_swallowed():
    """A group that never made top-N was not collapsed, it was not offered — reporting it as a
    re-discovery would tell a reader their page is denser than it is."""
    ranked = [_item("a", "one"), _item("b", "one"), _item("c", "two"), _item("d", "two")]
    _top, _alts, collapsed = collapse_by_direction(ranked, lambda i: i["direction"],
                                                   lambda i: i["id"], 1)
    assert collapsed == 1
