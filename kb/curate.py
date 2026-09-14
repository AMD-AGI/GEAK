"""Curation shared by both lanes' readers: collapsing a ranked candidate list to one rank per idea.

The store ranks on a scalar; the reader wants breadth. Three implementations of one `direction`
are one idea offered three times, and every candidate a lane adopts costs a full on-box verify, so
the runners-up ride along as `alternates` instead of each taking a slot. An entry with no direction
is its own group — unlabeled is honest, not a group.
"""


def demote_hinted(ordered, hint_of):
    """`ordered` with every hinted item moved behind every unhinted one, order preserved.

    Applied between the ranking and `collapse_by_direction`, because the collapse keeps only the
    FIRST entry per direction: a hinted record that outranks its group on the raw scalar does not
    merely lead the group, it deletes the rest of it. A stable partition rather than a sort key so
    both lanes can share it whatever they rank on.
    """
    ordered = list(ordered)
    return ([i for i in ordered if not hint_of(i)] + [i for i in ordered if hint_of(i)])


def collapse_by_direction(ordered, direction_of, unique_of, top_n):
    """One rank per IDEA, best first. Input must already be in rank order.

    `direction_of(item)` -> the idea label (case/space-insensitive; empty => its own group).
    `unique_of(item)`    -> a stable per-item key, used to name an undirected group.
    Returns (chosen, alternates-per-chosen, how many were collapsed).
    """
    groups, order = {}, []
    for item in ordered:
        key = str(direction_of(item) or "").strip().lower() or "__undirected__" + unique_of(item)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    chosen = order[: max(1, int(top_n or 3))]
    return ([groups[k][0] for k in chosen], [groups[k][1:] for k in chosen],
            sum(len(groups[k]) - 1 for k in chosen))
