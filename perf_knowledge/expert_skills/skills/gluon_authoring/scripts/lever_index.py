#!/usr/bin/env python3
"""lever_index.py - what the experience index has to say about THIS bound class, on THIS chip.

`lever-cards.json` was never meant to be read front to back. Its index was the bound-class
decision tree: `classify.py` picked `lever-cards[bound_class]`, filtered by arch availability, and
handed the survivors to the round as `suggested_levers`. The tool-first packs (triton, gluon) drop
`classify.py` on purpose -- the agent reads the profile and names the bound itself -- and dropping
it took the index with it. The cards stayed; nothing indexed them. Measured over 19 kernel/phase
runs of one campaign: `lever-cards.json` appears in zero of them, while the arms re-derived from
first principles, at full measurement cost, facts these cards already hold.

This restores the index and nothing else. It does not classify, does not decide, and does not rank
by anything it has not been told.

WHAT IS HARD HERE, AND WHAT IS NOT. The distinction is the whole point of the output shape:

  * `arch_availability` is measured fact. A lever the chip does not have is excluded, and that
    exclusion is safe to act on without checking.
  * A `gating_law` is a causal rule with a named source -- "latency-removing levers pay only at low
    occupancy" is not a heuristic about what usually helps, it is a statement about when a lever
    CANNOT pay. Reported as a warning against the specific candidates it governs.
  * Everything else is EXPERIENCE, and experience is not always right. Each card carries the `gate`
    that made it apply in the past and the `verify` that settled it. Those two fields are the
    handles: the gate is a condition to check against your own profile, not a claim that it holds,
    and the output says so in `_contract`. A candidate here is a thing worth measuring, never a
    thing to apply.

So the ordering below is by the card's own recorded `priority`, not by any judgement of this tool
about your kernel. It has not seen your kernel.

Usage:
  lever_index.py --bound memory [--sub bandwidth] [--arch gfx942]
  lever_index.py --bound latency --occupancy-waves 4      # flags the occupancy-gated candidates
  lever_index.py --list-bounds
  lever_index.py --selftest
"""

import argparse
import json
import os
import sys

_HW = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "references", "hardware"))
_PRIORITY_ORDER = {"high": 0, "medium": 1, "med": 1, "low": 2}

# Premises that belong to no single card because they hold for ANY source edit, returned with every
# query. Both are compile-time -- no GPU, no benchmark window, no comparator -- and both were the
# most frequently recorded negatives across nine campaigns of arm records. A gating law says when a
# named lever cannot pay; these say when a round cannot have measured anything at all.
_UNIVERSAL_PREMISES = [
    {
        "id": "edit_must_reach_the_isa",
        "premise": "the edit is visible in what the compiler emits",
        "check": "stripped-asm hash, before vs after",
        "why": "The most frequent recorded negative across those campaigns was a source change "
               "whose emitted code was byte-identical to the baseline -- the compiler was already "
               "doing it, or the construct was overridden elsewhere (one run found "
               "`tl.range(num_stages=1)` overriding the kernel-level count, leaving three variants "
               "byte-identical to plain). Identical code cannot have a timing difference, and any "
               "delta measured against it is the instrument.",
    },
    {
        "id": "freed_resource_must_buy_a_wave",
        "premise": "the resource the edit frees actually raises occupancy",
        "check": "waves/SIMD as built, both sides",
        "why": "Recorded repeatedly: VGPR 238->152 and LDS 32768->24576 with `Occupancy: 2` "
               "unchanged; split_k at ks1/ks2/ks4 with occupancy and shared bytes unchanged at "
               "every split. Occupancy moves in steps, so a large saving that does not cross a "
               "step buys nothing while the lever still pays its own cost.",
    },
]


# What a negative has to carry to be worth anything to the next round. This is the other half of
# the index: a candidate retired without these cannot be re-examined later, so it is neither a
# usable negative nor a re-openable question -- it just disappears. Every field maps to a check
# that already reads for it, which is why the list is this short and not a wish list.
_RETIREMENT_SCOPE = {
    "why": "A negative is a measurement of ONE lever against ONE body at ONE config. Record the "
           "scope with it or the next round cannot tell whether it still holds. Measured across "
           "nine campaigns: 24 lever verdicts were recorded against a body that predates the "
           "pinned configuration, and 10 arms could not say which config their number came from.",
    "record": [
        "config_basis -- the config the number was measured at, and whether the arm swept its own "
        "or borrowed the parent's pin (`arm_config_basis` reads this; a borrowed config biases a "
        "new structure SLOW, so it can manufacture a loss but not a win)",
        "kill_evidence -- the mechanism that decided it, not just the slower time",
        "source identity of the body it was measured against (`stale_lever_verdicts` compares this "
        "to the pin; a verdict older than the winning body may already be void)",
        "which axis/decomposition was tried, when the lever has several -- a lever killed on "
        "one decomposition is not thereby killed on the others",
    ],
    "not_a_permanent_rejection": "These cards are preconditions. If a later source, layout or "
                                 "launch change invalidates the premise a negative rested on, it "
                                 "re-enters the roster through the resweep change record.",
}


def _load_cards(path=None):
    target = path or os.path.join(_HW, "lever-cards.json")
    try:
        with open(target) as fh:
            return json.load(fh), target
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[lever_index] cannot read lever cards at {target}: {exc}")


def _arch_state(card, arch):
    """`yes` / `no` / a version caveat / unknown. Absent means the card never spoke to this arch."""
    avail = card.get("arch_availability")
    if not isinstance(avail, dict) or not arch:
        return None
    return avail.get(arch)


def _matches(card, bound, sub):
    """Every `applies_to` row of this card that answers for (bound, sub).

    A row whose `sub` is null answers for the whole bound class; a row with a sub answers only for
    that sub. Asking without a sub returns both, because "memory-bound, sub not yet resolved" is a
    real state of a round and the cards for the subs are exactly what would resolve it.
    """
    out = []
    for row in card.get("applies_to") or []:
        if not isinstance(row, dict) or row.get("bound") != bound:
            continue
        if sub and row.get("sub") not in (None, sub):
            continue
        out.append(row)
    return out


def index(bound, sub=None, arch=None, occupancy_waves=None, cards_path=None):
    doc, source = _load_cards(cards_path)
    levers = doc.get("levers")
    if not isinstance(levers, dict):
        raise SystemExit(f"[lever_index] {source} has no `levers` object")
    laws = {law.get("id"): law for law in (doc.get("gating_laws") or [])
            if isinstance(law, dict) and law.get("id")}

    candidates, excluded, gated = [], [], []
    for name, card in sorted(levers.items()):
        if not isinstance(card, dict):
            continue
        rows = _matches(card, bound, sub)
        if not rows:
            continue
        state = _arch_state(card, arch)
        if isinstance(state, str) and state.strip().lower() in ("no", "false", "absent"):
            excluded.append({"lever": name, "reason": f"arch_availability[{arch}] = {state!r}",
                             "hardness": "measured fact -- safe to act on without checking"})
            continue
        row = min(rows, key=lambda r: _PRIORITY_ORDER.get(str(r.get("priority") or "").lower(), 3))
        entry = {
            "lever": name,
            "priority": row.get("priority"),
            "layer": card.get("layer"),
            "sub": row.get("sub"),
            # The two fields that make this a referral and not an instruction.
            "gate_to_check": row.get("gate"),
            "verify": card.get("verify"),
            "expected_move": row.get("expected_move") or row.get("direction"),
            "cost": card.get("cost"),
            "direction_class": card.get("direction_class"),
            "native_group": card.get("native_group"),
            "expressible_in": card.get("expressible_in"),
            "authoring_ref": card.get("authoring_ref"),
            "note": card.get("note"),
        }
        if arch:
            entry["arch_availability"] = state if state is not None else "not stated for this arch"
        law_id = card.get("gating_law")
        if law_id:
            entry["gating_law"] = law_id
            law = laws.get(law_id)
            if law:
                entry["gating_law_effect"] = law.get("effect")
        # The one law this tool can evaluate rather than merely quote, because the caller can hand
        # it the number: at 2+ resident waves a latency-removing lever is already paid for.
        if (occupancy_waves is not None and bound == "latency"
                and float(occupancy_waves) >= 2 and law_id != "cut_idle_unit_neutral"):
            gated.append({
                "lever": name, "law": "occupancy_gates_latency",
                "why": f"{occupancy_waves} resident waves/SIMD: latency-removing levers become "
                       f"neutral-to-negative; they pay only at low occupancy",
                "hardness": "causal rule with a named source -- bound the win before spending a round",
            })
        candidates.append(entry)

    candidates.sort(key=lambda e: (_PRIORITY_ORDER.get(str(e.get("priority") or "").lower(), 3),
                                   e["lever"]))
    return {
        "schema": "amd.lever_index/1",
        "source": os.path.relpath(source, os.path.dirname(_HW)),
        "query": {"bound": bound, "sub": sub, "arch": arch,
                  "occupancy_waves": occupancy_waves},
        "candidates": candidates,
        "excluded_on_arch": excluded,
        "gated_by_law": gated,
        "gating_laws": [laws[k] for k in sorted(laws)],
        "universal_premises": _UNIVERSAL_PREMISES,
        "if_you_retire_one": _RETIREMENT_SCOPE,
        "_contract": (
            "An INDEX, not a decision. `excluded_on_arch` is measured fact and `gated_by_law` is a "
            "causal rule -- both are safe to act on. Every entry in `candidates` is prior "
            "EXPERIENCE and may be wrong for this kernel: `gate_to_check` is the condition that "
            "made it apply before, to be checked against YOUR profile rather than assumed, and "
            "`verify` is how the question was settled. Ordering is the card's own recorded "
            "priority; this tool has not seen your kernel. `universal_premises` hold for whichever "
            "candidate you pick and cost a compile, not a measurement window."
        ),
    }


def _selftest():
    # The cards are overlaid from a lang/family layer, so they exist in a COMPOSED pack and not in
    # the unmerged source tree. Skipping there is correct: there is nothing to index yet, and
    # failing would only report the layering.
    default = os.path.join(_HW, "lever-cards.json")
    if not os.path.isfile(default):
        print(f"[lever_index] SELFTEST SKIP -- no lever-cards.json at {default} "
              f"(uncomposed source tree; the cards arrive from a lang/family layer)")
        return 0
    doc, path = _load_cards()
    levers = doc.get("levers") or {}
    assert levers, f"no levers in {path}"

    bounds = {row.get("bound") for c in levers.values() if isinstance(c, dict)
              for row in (c.get("applies_to") or []) if isinstance(row, dict)}
    assert bounds, "no applies_to rows carry a bound"

    # The index returns only cards that answer for the asked bound, and every candidate carries the
    # two fields that keep it a referral rather than an instruction.
    for bound in sorted(b for b in bounds if b):
        res = index(bound)
        for entry in res["candidates"]:
            card = levers[entry["lever"]]
            assert any(r.get("bound") == bound for r in card.get("applies_to") or []), \
                (bound, entry["lever"])
            assert "gate_to_check" in entry and "verify" in entry, entry
        assert "EXPERIENCE" in res["_contract"]

    # A sub narrows, never widens.
    subs = {row.get("sub") for c in levers.values() if isinstance(c, dict)
            for row in (c.get("applies_to") or [])
            if isinstance(row, dict) and row.get("bound") == "memory" and row.get("sub")}
    if subs:
        sub = sorted(subs)[0]
        wide = {e["lever"] for e in index("memory")["candidates"]}
        narrow = {e["lever"] for e in index("memory", sub=sub)["candidates"]}
        assert narrow <= wide, (sub, narrow - wide)

    # An arch the cards call `no` is EXCLUDED, not merely deprioritised -- that half is fact.
    for name, card in levers.items():
        avail = card.get("arch_availability")
        if not isinstance(avail, dict):
            continue
        for arch, state in avail.items():
            if str(state).strip().lower() == "no":
                for row in card.get("applies_to") or []:
                    if not isinstance(row, dict) or not row.get("bound"):
                        continue
                    res = index(row["bound"], arch=arch)
                    assert name not in {e["lever"] for e in res["candidates"]}, (name, arch)
                    assert name in {e["lever"] for e in res["excluded_on_arch"]}, (name, arch)
                break

    # Occupancy is the one law this tool evaluates instead of quoting.
    hi = index("latency", occupancy_waves=4)
    lo = index("latency", occupancy_waves=1)
    assert not lo["gated_by_law"], lo["gated_by_law"]
    if hi["candidates"]:
        assert hi["gated_by_law"], "2+ waves must flag the latency candidates as already paid for"

    # The universal premises ride every query, including one that matches no card at all -- they
    # are about the round, not about the lever, so an empty candidate list must still carry them.
    for res in (index("memory"), index("compute"), index("no_such_bound")):
        ids = [p["id"] for p in res["universal_premises"]]
        assert ids == ["edit_must_reach_the_isa", "freed_resource_must_buy_a_wave"], ids
        assert all(p.get("check") and p.get("why") for p in res["universal_premises"])
        # Querying the index and recording its outcome are one loop: whoever asks what to try is
        # the one who will retire something, and the scope to record travels with the question.
        scope = res["if_you_retire_one"]
        assert scope["record"] and scope["not_a_permanent_rejection"], scope
        joined = " ".join(scope["record"])
        for field in ("config_basis", "kill_evidence", "source identity"):
            assert field in joined, field

    print(f"[lever_index] SELFTEST PASS -- {len(levers)} cards, bounds {sorted(b for b in bounds if b)}; "
          f"arch `no` excludes, sub narrows, every candidate carries its gate and verify")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bound", help="bound class named by YOUR profile "
                                    "(memory | compute | latency | register | balanced)")
    ap.add_argument("--sub", help="sub-resource, when the profile resolved one")
    ap.add_argument("--arch", help="gfx942 | gfx950 | ... -- excludes levers this chip lacks")
    ap.add_argument("--occupancy-waves", type=float, metavar="N",
                    help="resident waves/SIMD as BUILT; at 2+ the occupancy_gates_latency law fires")
    ap.add_argument("--cards", help="path to lever-cards.json (default: this pack's)")
    ap.add_argument("--list-bounds", action="store_true", help="which bound classes the cards answer for")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.list_bounds:
        doc, _ = _load_cards(a.cards)
        pairs = sorted({(row.get("bound"), row.get("sub"))
                        for c in (doc.get("levers") or {}).values() if isinstance(c, dict)
                        for row in (c.get("applies_to") or []) if isinstance(row, dict)})
        for bound, sub in pairs:
            if bound:
                print(f"{bound}" + (f" --sub {sub}" if sub else ""))
        return 0
    if not a.bound:
        ap.error("--bound is required (or --list-bounds / --selftest)")
    print(json.dumps(index(a.bound, a.sub, a.arch, a.occupancy_waves, a.cards), indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
