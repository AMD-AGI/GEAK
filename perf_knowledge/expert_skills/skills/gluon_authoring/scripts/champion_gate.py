#!/usr/bin/env python3
"""champion_gate.py - the entry assertion for a DEEP-DIG skill (gluon / flydsl).

The deep-dig skills do not tune plain source themselves: they start from a `plain_champion` bundle
produced by the broad-search front end (tile-programming-triton) and climb from there. That handoff
is the single point where the whole run's honesty is decided, because every later speedup is quoted
against it. This gate refuses to start on a bundle that cannot support such a claim.

It generalizes the one surviving mechanical plain->escalated check ([PLAIN-UNTUNED] in gate.py), which
only fired inside the gated per-round engine and only compared 7 hardcoded knob names. Here the check
is a standalone tool, keys on whatever the config actually contains, and additionally pins the SOURCE
(a bundle whose champion source has been edited since it was measured is not the thing that was
measured) and the COMPARATOR (a champion slower than the kernel's own default is a strawman inverted).

Checks (HARD unless marked soft):
  [SCHEMA]     the bundle loads, `schema: plain_champion`, required fields present
  [SOURCE]     `source_ref` resolves and its sha256 matches `source_sha`
  [CONFIG]     the pinned `.ttgir` was dumped AT `config` -- cross-checked against the TTGIR's own
               `ttg.num-warps` and against a `<ttgir>.config.json` sidecar when either is available
               (soft when neither exists: unverifiable, not contradicted)
  [COMPARATOR] `champion_ms` <= `default_ms` and <= `sweep_winner_ms` (soft when `default_ms` is null,
               which is itself reported -- the "not a default strawman" claim is then unprovable)
  [GATED]      `trust_level` is `pinned`. A provisional or ungated comparator is exploration
               only and requires an explicit flag; it cannot support final acceptance.
  [SAMPLING]   the winner's neighbourhood was MEASURED. Either the feasible grid was sampled whole
               (`partially_sampled` false), or the sweep returned a certified order-2 INTERIOR local
               optimum -- no single-axis and no paired move improved it at the measured noise band,
               and it sits on no ladder end. A derived space is deliberately never enumerated, so
               enumeration cannot be the test; the test is the strength of the claim about the WINNER.
               Neither present -> FAIL unless --allow-provisional (then a declared soft degrade: the
               comparator is the best of a SUBSET, so downstream speedups are inflated by whatever the
               unmeasured configs would have gained)
  [RANGE]      `served_range` is non-empty (soft here; the close requires it)
  [CLIMB]      the plain line was actually CLIMBED before it was published. Either `climb.rounds` is
               at or above the floor, or `climb_declined` prices what was left -- a bounded remaining
               prize with an artifact that ran. Neither -> FAIL unless --allow-shallow-climb
  [LOCUS]      the execution locus is recorded, so the deep-dig runs where the champion was measured
  [TOOLCHAIN]  the toolchain is pinned (soft) -- an unpinned bundle cannot be re-measured later

Usage:
  champion_gate.py --champion <work>/plain_champion.json [--allow-provisional] [--allow-ungated]
      [--allow-shallow-climb] [--json]
  champion_gate.py --selftest
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# The keys whose mismatch means the TTGIR cannot reach the champion's performance. NOT a fixed list of
# knob names: this is only the fallback ORDER for reporting. The comparison itself runs over every key
# the two configs share, because a kernel that spells its tile `BLOCK_SIZE_M` (aiter) or `BLK_M` is
# just as config-pinned as one that spells it `BLOCK_M` -- keying on names is how the old sweep prune
# silently blocked 6 of 7 kernel families.
_REPORT_FIRST = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                 "num_warps", "num_stages", "matrix_instr_nonkdim", "GROUP_SIZE_M")

_REQUIRED = ("kernel", "source_ref", "source_sha", "config", "champion_ms", "ttgir")

_NUM_WARPS_RE = re.compile(r'ttg\.num-warps["\s]*[=:]\s*(\d+)')


def _adopt_climb_floor() -> tuple:
    """The CLIMB depth floor has ONE owner, `search_mode.py`, and this adopts it.

    Same reason that file adopts `round_record.py`'s vocabulary: two literals with no cross-check
    drift, and the drift is invisible because both sides keep passing their own tests. In a composed
    pack both files land in the same scripts/ dir and this import resolves; where it does not (a
    deep-dig pack that ships the gate without the front end's mode ledger) the fallback applies and
    the check says so, rather than pretending the number was verified.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import search_mode                                        # noqa: PLC0415
        return int(search_mode.CLIMB_ROUND_FLOOR), True
    except (ImportError, AttributeError, ValueError):
        return 10, False
    finally:
        sys.path.pop(0)


CLIMB_ROUND_FLOOR, CLIMB_FLOOR_FROM_OWNER = _adopt_climb_floor()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(base: Path, ref) -> Path | None:
    """A bundle path is relative to the bundle dir (so the whole bundle is movable) or absolute."""
    if not ref:
        return None
    p = Path(str(ref))
    return p if p.is_absolute() else (base / p)


def _shared_key_mismatch(a: dict, b: dict) -> list:
    """Keys present in BOTH with different values, tile/warp knobs reported first."""
    mism = [k for k in set(a) & set(b) if a[k] != b[k]]
    return sorted(mism, key=lambda k: (_REPORT_FIRST.index(k) if k in _REPORT_FIRST else 99, k))


def _climb_price(base: Path, decl) -> str | None:
    """The declination's price, or None when it is an assertion wearing a number.

    Deliberately the same bar the declined BRANCH wave has to clear in `close_audit.py`: a digit in
    a sentence is not a price. The claim "the rest of this line is not worth climbing" ENDS the
    search, so it owes a bound that something measured -- an artifact that ran, and the statement
    that the figure is an upper bound rather than a guess that happens to be numeric.
    """
    if not isinstance(decl, dict):
        return None
    pct = decl.get("remaining_prize_pct", decl.get("pct"))
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return None
    if decl.get("is_upper_bound") is not True:
        return None
    ev = decl.get("evidence_ref")
    if not isinstance(ev, dict) or ev.get("kind") not in ("probe", "measurement"):
        return None
    art = _resolve(base, ev.get("artifact"))
    if art is None or not art.is_file():
        return None
    method = str(decl.get("method") or "").strip()
    if not method:
        return None
    return f"<= {pct}% remaining, {ev['kind']} {Path(str(ev['artifact'])).name}, {method[:80]}"


def gate(champ: dict, base: Path, allow_provisional: bool = False, allow_ungated: bool = False,
         allow_default_anchor: bool = False, allow_shallow_climb: bool = False):
    """Returns (results, hard_fail). `base` is the dir the bundle's relative paths resolve against."""
    results, hard_fail = [], False

    def rec(check, ok, msg, hard=True):
        nonlocal hard_fail
        results.append({"check": check, "ok": bool(ok), "hard": bool(hard), "msg": msg})
        if not ok and hard:
            hard_fail = True

    # [SCHEMA]
    if not isinstance(champ, dict) or champ.get("schema") != "plain_champion":
        rec("SCHEMA", False, f"not a plain_champion bundle (schema={champ.get('schema') if isinstance(champ, dict) else type(champ).__name__!r}). "
                             "The broad-search front end (tile-programming-triton) writes it at close.")
        return results, hard_fail          # nothing below is meaningful without the schema
    missing = [k for k in _REQUIRED if champ.get(k) in (None, "")]
    if missing:
        rec("SCHEMA", False, f"missing required field(s): {missing}")
        return results, hard_fail
    rec("SCHEMA", True, f"plain_champion for {champ['kernel']}")

    # [SOURCE] -- the bundle must still describe the file that was measured.
    src = _resolve(base, champ.get("source_ref"))
    if src is None or not src.is_file():
        rec("SOURCE", False, f"source_ref does not resolve to a file: {champ.get('source_ref')!r} "
                             f"(resolved against {base})")
    else:
        got = _sha256(src)
        if got != champ["source_sha"]:
            rec("SOURCE", False, f"{src} has changed since it was measured "
                                 f"(sha256 {got[:12]} != recorded {str(champ['source_sha'])[:12]}). "
                                 "Re-measure the champion, or transcribe the recorded source -- a "
                                 "deep-dig anchored on an edited source is measured against nothing.")
        else:
            rec("SOURCE", True, f"{src.name} matches its recorded sha256")

    # [LIVE] -- the file the RUN loads must still be the file the gate just validated.
    #
    # [SOURCE] above hashes `source_ref`, which is by design a frozen copy -- so it
    # matches essentially forever and cannot detect the failure that actually happens.
    # Everything downstream loads `kernel`, and nothing was checking it. Found on a real
    # bundle: `kernel` -> task/kernel_jit.py had been overwritten by a later Gluon track
    # (its sha is that track's installed winner), while `source_ref` -> champion/
    # kernel_jit.py was pristine. The gate reported `[PASS] SOURCE ... matches its
    # recorded sha256` and cleared a run whose plain arm would have been GLUON. One agent
    # noticed on its own; the next one would not have, and would have reported a ratio
    # near 1.0 against the wrong denominator.
    live = _resolve(base, champ.get("kernel"))
    if live is None or not live.is_file():
        rec("LIVE", False, f"`kernel` does not resolve to a file: {champ.get('kernel')!r}. "
                           "That is the path the run loads, so nothing downstream is anchored.")
    else:
        got_live = _sha256(live)
        if got_live != champ["source_sha"]:
            same_name = (src is not None and src.is_file()
                         and _sha256(src) == champ["source_sha"])
            hint = (f"Restore it from source_ref ({champ.get('source_ref')}), which does still "
                    "match, or re-measure." if same_name else
                    "Both `kernel` and `source_ref` are off the recorded sha; re-measure.")
            rec("LIVE", False,
                f"`kernel` ({live}) is NOT the measured source: sha256 {got_live[:12]} != "
                f"recorded {str(champ['source_sha'])[:12]}. Whatever the run benchmarks as "
                f"'plain' is not what champion_ms describes. {hint}")
        else:
            rec("LIVE", True, f"`kernel` ({live.name}) is byte-identical to the measured source")

    # [CONFIG] -- the TTGIR the anchor is recovered from must be the tuned config's TTGIR.
    cfg = champ.get("config")
    ttgir = _resolve(base, champ.get("ttgir"))
    if not isinstance(cfg, dict):
        rec("CONFIG", False, f"config is not an object: {type(cfg).__name__}")
    elif ttgir is None or not ttgir.is_file():
        rec("CONFIG", False, f"ttgir does not resolve to a file: {champ.get('ttgir')!r}. The anchor is "
                             "recovered from the champion's own TTGIR; without it there is nothing to "
                             "recover from (dump_ir.sh at the pinned config).")
    else:
        text = ttgir.read_text(errors="ignore")
        evidence, contradiction = [], []
        m = _NUM_WARPS_RE.search(text)
        if m and "num_warps" in cfg:
            if int(m.group(1)) == int(cfg["num_warps"]):
                evidence.append(f"ttg.num-warps={m.group(1)}")
            else:
                contradiction.append(f"ttg.num-warps={m.group(1)} but config num_warps={cfg['num_warps']}")
        side = Path(str(ttgir) + ".config.json")
        if side.is_file():
            try:
                sc = json.loads(side.read_text())
            except (ValueError, OSError) as e:
                contradiction.append(f"{side.name} unreadable ({e})")
            else:
                mism = _shared_key_mismatch(sc if isinstance(sc, dict) else {}, cfg)
                if mism:
                    contradiction.append(f"{side.name} disagrees on {mism}")
                else:
                    evidence.append(f"{side.name} agrees on {len(set(sc) & set(cfg))} shared key(s)")
        if contradiction:
            rec("CONFIG", False, "the TTGIR was NOT dumped at the pinned config: "
                                 + "; ".join(contradiction) +
                                 ". Re-dump from the winning config, else the anchor starts below "
                                 "plain-best and every later delta is measured from the wrong floor.")
        elif evidence:
            rec("CONFIG", True, "TTGIR is consistent with the pinned config (" + ", ".join(evidence) + ")")
        else:
            rec("CONFIG", True, f"{ttgir.name} exists but carries no cross-checkable config signal "
                                f"(no ttg.num-warps, no {side.name}) -- UNVERIFIED, not contradicted. "
                                "Have dump_ir.sh write the sidecar to make this checkable.", hard=False)

    # [COMPARATOR] -- champion must beat the default and the config-only winner.
    ch = champ.get("champion_ms")
    dm, sw = champ.get("default_ms"), champ.get("sweep_winner_ms")
    if not isinstance(ch, (int, float)):
        rec("COMPARATOR", False, f"champion_ms is not a number: {ch!r}")
    elif isinstance(dm, (int, float)) and ch > dm:
        rec("COMPARATOR", False, f"champion_ms {ch:.4f} is SLOWER than the kernel's own default "
                                 f"{dm:.4f} ms. That is a strawman inverted: every downstream speedup "
                                 "quoted against it is measured from a floor the shipped kernel beats.")
    elif isinstance(sw, (int, float)) and ch > sw:
        rec("COMPARATOR", False, f"champion_ms {ch:.4f} is SLOWER than the config sweep's own winner "
                                 f"{sw:.4f} ms -- the source-level work regressed the tuned baseline.")
    elif not isinstance(dm, (int, float)):
        rec("COMPARATOR", True, f"champion_ms {ch:.4f} recorded, but default_ms is null -- the "
                                "'vs tuned plain, never the default strawman' claim is UNPROVABLE "
                                "until the front end publishes it.", hard=False)
    else:
        gain = f", {dm / ch:.3f}x vs default" if ch else ""
        rec("COMPARATOR", True, f"champion_ms {ch:.4f} <= default {dm:.4f}{gain}")

    # [GATED] -- an UNGATED sweep never checked its winner, so the bundle's timings describe a
    # program nobody proved computes the right thing. That outranks [SAMPLING]: a capped sweep is an
    # incomplete measurement, an ungated one may not be a measurement of the kernel at all.
    trust = champ.get("trust_level")
    anchor = champ.get("anchor") or {}
    if anchor.get("type") == "production_default_anchor":
        # A DECLARABLE EXCEPTION, like the two below it, rather than an unconditional refusal.
        #
        # What this check actually refuses is starting a DEEP DIG from a default anchor, and that is
        # right: the anchor is not a pinned configuration comparator, so a deep result measured
        # against it is measured against an untuned denominator. But a run whose scope never
        # reaches a deep dig was being failed for a transition it does not attempt, and the same
        # bundle's [GATED] line two branches down already says "anchor remains plain-only" -- two
        # checks, one fact, opposite verdicts, and the disagreement was being resolved by hand.
        # So the caller declares which question it is asking, and the exception is recorded.
        rec("ANCHOR", bool(allow_default_anchor),
            "default anchor accepted via --allow-default-anchor: this bundle may NOT start a deep "
            "dig, and the plain-only limitation travels in caveats[] with the number"
            if allow_default_anchor else
            "production_default_anchor permits plain in-body structural work only; it is not a "
            "pinned configuration comparator and cannot start a formal deep dig. If this close is "
            "plain-only and never asks for one, say so with --allow-default-anchor instead of "
            "waiving the finding by hand",
            hard=not allow_default_anchor)
    elif anchor.get("type") != "pinned_comparator":
        rec("ANCHOR", False, "bundle must declare a pinned_comparator anchor for deep work")
    else:
        rec("ANCHOR", True, "pinned comparator anchor declared")
    if trust == "ungated":
        rec("GATED", bool(allow_ungated),
            "the champion's config sweep ran UNGATED (no oracle), so its winner was never checked "
            "for correctness and may be wrong-and-fast. Declare correctness.cmd and re-sweep, or "
            "pass --allow-ungated to proceed with it DECLARED in caveats[]."
            if not allow_ungated else
            "ungated sweep accepted via --allow-ungated -- record it in caveats[]: every timing in "
            "this bundle is unverified, and the deep dig inherits that.",
            hard=not allow_ungated)
    elif trust == "pinned":
        rec("GATED", True, "the config sweep is oracle-gated and pinned")
    elif trust == "provisional":
        rec("GATED", bool(allow_provisional),
            "the comparator is provisional and may be used only for explicitly experimental "
            "exploration; it cannot support final acceptance"
            if allow_provisional else
            "the comparator is provisional, not pinned; close its evidence or pass "
            "--allow-provisional only for exploration (never final acceptance)",
            hard=not allow_provisional)
    elif trust == "production_default_anchor":
        rec("GATED", True,
            "default anchor correctness is carried by preflight; anchor remains plain-only")
    else:
        rec("GATED", True, "the sweep predates trust_level, so whether an oracle ran is UNKNOWN -- "
                           "re-emit the bundle from a current plain_autotune.py to make it checkable",
            hard=False)

    # [SAMPLING] -- a provisional comparator inflates everything downstream.
    #
    # The LOCAL OPTIMUM certificate is read alongside coverage because the two support different
    # claims and this gate needs both. Coverage is about the grid; the certificate is about the
    # WINNER -- order 1 means every +/-1 single-axis neighbour was measured and none was faster,
    # order 2 adds the sensitive-pair diagonals. That is precisely the spot-check the PASS branch
    # below used to ask the operator to perform by hand, so when a sweep already did it mechanically
    # there is no reason to ask again; and when a sweep is capped, it is the difference between "the
    # best of an arbitrary 7% slice" and "a point with no faster neighbour in any direction we
    # measured", which is the strongest thing a capped sweep can honestly say.
    lo = champ.get("local_optimum") or {}
    # Prefer the PAIRWISE number when the sweep reports one. A derived space has an astronomical cross
    # product, so its measured fraction rounds to 0.0% however well the search ran, and quoting it here
    # reads as an indictment of a pin the certificate actually supports. What a reader can act on is how
    # many axis PAIRS were ever co-sampled -- that is what "a pair never co-sampled" below refers to.
    pair, cov = lo.get("pair_coverage_pct"), champ.get("grid_coverage_pct")
    cov_s = (f" ({pair:.1f}% of axis pairs ever co-sampled)" if isinstance(pair, (int, float))
             else f" (grid coverage {cov:.1f}%)" if isinstance(cov, (int, float)) else "")
    cert_s = ""
    if lo.get("certified"):
        cert_s = (f" The winner IS a certified order-{lo.get('order')} local optimum: every "
                  f"{'single-axis and sensitive-pair-diagonal' if lo.get('order', 0) >= 2 else 'single-axis'}"
                  f" neighbour was measured and none was faster.")
    elif lo.get("unprobed"):
        cert_s = (f" And the winner is NOT a certified local optimum -- "
                  f"{len(lo['unprobed'])} neighbour(s)/shell(s) were never measured "
                  f"({', '.join(str(u) for u in lo['unprobed'][:4])}"
                  f"{', ...' if len(lo['unprobed']) > 4 else ''}); re-run the sweep with a larger "
                  f"--budget-points to close them.")
    # A SEARCHED space is never enumerated -- that is the point of deriving it rather than crossing it
    # -- so `partially_sampled` alone cannot decide this check. An order-2 certificate whose winner is
    # interior is the stronger claim of the two: coverage says which points were visited, the
    # certificate says nothing adjacent to the ANSWER was faster. Requiring interiority as well is what
    # keeps this from being a loophole: an optimum on a ladder end is not evidence the range was wide
    # enough, and that is exactly the case where an unvisited point outside the range wins.
    certified_interior = bool(lo.get("certified") and (lo.get("order") or 0) >= 2
                              and lo.get("interior"))
    if champ.get("partially_sampled") and certified_interior:
        rec("SAMPLING", True,
            f"the config space was SEARCHED rather than enumerated{cov_s}, and the winner is a "
            f"certified order-2 INTERIOR local optimum.{cert_s} That is the strongest claim a derived "
            f"space supports, and it is stronger than exhausting a narrower range -- but carry the "
            f"certificate's own `unprobed` list into caveats[], because a pair never co-sampled has "
            f"not been ruled out.")
    elif champ.get("partially_sampled"):
        rec("SAMPLING", bool(allow_provisional),
            "the champion's config sweep neither sampled its space whole NOR certified its winner as "
            "an order-2 interior local optimum, so it is a provisional comparator, not a tuned one. "
            "Close the certificate (raise --budget-points; if the winner sits on a ladder END, extend "
            "that axis and re-run -- an optimum on the boundary is not evidence the range was wide "
            "enough), or pass --allow-provisional to proceed with the inflation DECLARED in caveats[]."
            f"{cov_s}{cert_s}"
            if not allow_provisional else
            "partially_sampled accepted via --allow-provisional -- record it in caveats[]: downstream "
            f"speedups are inflated by whatever the unmeasured configs would have gained.{cov_s}{cert_s}",
            hard=not allow_provisional)
    elif lo.get("certified"):
        # Both claims present and consistent: the grid was covered AND the winner's neighbourhood was
        # measured. This is the one case where the hand spot-check below is genuinely redundant.
        rec("SAMPLING", True,
            f"the feasible config grid is reported fully sampled{cov_s}, and the pinned config's "
            f"+/-1 neighbourhood was MEASURED rather than assumed.{cert_s} The manual spot-check is "
            f"already done.")
    else:
        # This branch can only read the bundle's OWN claim that the grid was covered, and a sweep
        # reporting that its winner survived says nothing about points it never tested. Observed: a
        # 6.1% plain win one grid step outside the swept range, on a kernel whose tier log recorded a
        # completed re-sweep -- inherited by the port as a fake escalation gain. So the PASS is
        # reported as unfalsified rather than verified, with the cheap check that would falsify it.
        rec("SAMPLING", True,
            "the bundle reports the feasible config grid as fully sampled. NOT VERIFIABLE HERE -- "
            "this check can only read the claim. Before the first round, spot-check the pinned "
            "config at +/-1 grid step on EACH swept axis: a sweep's own report that its pin survived "
            "is not evidence about points it did not test, and a port that starts one grid point "
            f"short of the real champion inherits that gap as a fake escalation gain.{cert_s}")

    # [MEASUREMENT] -- a contaminated reading in the sweep is a fact about the BOX, and it travels
    # with the bundle because every margin quoted downstream is quoted against these numbers. Soft:
    # the sweep already kept the faster reading of each re-verified point, so the pin itself is
    # sound; what needs saying is that narrow margins in this run's table are not resolvable.
    n_contam = champ.get("n_contaminated")
    if isinstance(n_contam, int) and n_contam > 0:
        rec("MEASUREMENT", True,
            f"{n_contam} point(s) in the champion's sweep moved by more than the drift tolerance when "
            f"re-read, i.e. their first reading was measuring interference rather than the config. "
            f"The sweep kept the faster reading, so the pin stands -- but treat any margin in this "
            f"bundle narrower than the re-verification spread as unresolved.", hard=False)

    # [RANGE] -- soft here, hard at CLOSE.
    rng = champ.get("served_range")
    rec("RANGE", bool(rng) and isinstance(rng, list),
        f"served_range has {len(rng)} row(s)" if isinstance(rng, list) and rng else
        "served_range is empty -- a single-anchor win that regresses at other shapes is a bucketed "
        "dispatch, not a clean win. The close requires this table; fill it before closing.",
        hard=False)

    # [CLIMB] -- was the line this bundle publishes ever actually climbed.
    #
    # Every other check here asks whether the number is TRUE. This one asks whether the search that
    # produced it happened, and it is here because publishing the champion is the moment the plain
    # run stops: the gate is a hard handoff boundary and the run moves to arbitration, so
    # a bundle emitted after two climb rounds converts the remaining wall clock into paperwork. That
    # is not hypothetical -- four measured kernels closed one to three climb rounds deep, two of
    # them with a third to a half of their budget unspent, and every one of them passed a strict
    # close audit, because BRANCH width was counted and CLIMB depth was not.
    #
    # The floor is not a quota to spend. It is the depth at which the pack's OWN stopping rule
    # (reconsideration after ten contiguous non-improving rounds) becomes answerable; below it,
    # "the line is done" is a statement no artifact in the bundle can support.
    climb = champ.get("climb") if isinstance(champ.get("climb"), dict) else None
    rounds = climb.get("rounds") if climb else None
    priced = _climb_price(base, champ.get("climb_declined"))
    floor_src = "" if CLIMB_FLOOR_FROM_OWNER else " (floor is this file's fallback copy: search_mode.py did not import)"
    if isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= CLIMB_ROUND_FLOOR:
        kept = climb.get("kept")
        rec("CLIMB", True, f"{rounds} climb round(s) at or above the floor of {CLIMB_ROUND_FLOOR}"
                           + (f", {kept} kept" if isinstance(kept, int) else "") + floor_src)
    elif priced:
        rec("CLIMB", True,
            f"climb stopped at {rounds if rounds is not None else 'an unrecorded'} round(s), below "
            f"the floor of {CLIMB_ROUND_FLOOR}, on a PRICED declination ({priced}) -- recorded, not "
            f"contested. Carry it into caveats[]: the deep track's speedup is quoted against a line "
            f"whose own remainder was bounded rather than exhausted.{floor_src}")
    else:
        detail = ("bundle declares no `climb` at all, so nothing says the published line was ever "
                  "climbed" if climb is None else
                  f"`climb.rounds` is {rounds!r}" if not isinstance(rounds, int) or isinstance(rounds, bool)
                  else f"climb stopped after {rounds} round(s) and the floor is {CLIMB_ROUND_FLOOR}")
        rec("CLIMB", bool(allow_shallow_climb),
            f"{detail}. Publishing here ends the plain run, so it owes one of two things: reach the "
            f"floor, or declare `climb_declined` as an object carrying {{remaining_prize_pct, "
            f"evidence_ref{{kind: probe|measurement, artifact}}, method, is_upper_bound: true}} -- "
            f"the same bar a declined BRANCH wave clears, because a digit in a sentence is not a "
            f"price. Pass --allow-shallow-climb to proceed with the shortfall DECLARED in "
            f"caveats[]: every deep-track gain is then measured from a floor the plain run did not "
            f"finish looking for.{floor_src}"
            if not allow_shallow_climb else
            f"{detail}; accepted via --allow-shallow-climb -- record it in caveats[], because the "
            f"deep track's baseline is a line the plain run stopped climbing early.{floor_src}",
            hard=not allow_shallow_climb)

    # [PATH_COVERAGE] -- a served-range row is not proof that a new guarded path ran.
    coverage = champ.get("path_coverage")
    if not isinstance(coverage, list):
        rec("PATH_COVERAGE", False,
            "bundle lacks explicit path_coverage; every new guard/fast path/fallback/masked tail "
            "must declare its trigger cases and correctness artifact")
    else:
        invalid = []
        for index, item in enumerate(coverage):
            if not isinstance(item, dict):
                invalid.append(str(index))
                continue
            if (not isinstance(item.get("path_id"), str) or not item["path_id"].strip()
                    or not isinstance(item.get("trigger_cases"), list)
                    or not item["trigger_cases"]
                    or not isinstance(item.get("correctness_ref"), str)
                    or not item["correctness_ref"].strip()):
                invalid.append(str(index))
                continue
            ref = _resolve(base, item["correctness_ref"])
            if ref is None or not ref.is_file():
                invalid.append(str(index))
        rec("PATH_COVERAGE", not invalid,
            "no new guarded control paths declared" if not coverage else
            f"all {len(coverage)} new control path(s) have trigger cases and correctness evidence"
            if not invalid else f"invalid/unresolved path coverage entries: {', '.join(invalid)}")

    # [LOCUS] -- run the deep dig where the champion was measured.
    loc = champ.get("locus") or {}
    if loc.get("docker"):
        rec("LOCUS", True, f"container {str(loc['docker'])[:12]}, gpu {loc.get('gpu', '?')}, "
                           f"pythonpath {'set' if loc.get('pythonpath') else 'UNSET'}",
            hard=False)
    else:
        rec("LOCUS", True, "host locus (no container recorded) -- confirm this box is the one the "
                           "champion was measured on, or the baseline is not comparable", hard=False)

    # [TOOLCHAIN]
    tc = champ.get("toolchain") or {}
    rec("TOOLCHAIN", bool(tc), f"pinned: {', '.join(f'{k}={v}' for k, v in sorted(tc.items()))}" if tc
        else "toolchain not pinned -- this bundle cannot be re-measured after a ROCm/Triton bump",
        hard=False)
    return results, hard_fail


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        srcf = base / "champion" / "k.py"
        srcf.parent.mkdir(parents=True)
        srcf.write_text("# champion kernel\n")
        sha = _sha256(srcf)
        # The LIVE check needs `kernel` to be a real file. Same bytes as the frozen copy,
        # different path -- which is exactly the shape of a healthy bundle.
        livef = base / "task" / "k.py"
        livef.parent.mkdir(parents=True)
        livef.write_text("# champion kernel\n")
        irf = base / "ir" / "champion.ttgir"
        irf.parent.mkdir(parents=True)
        irf.write_text('module attributes {"ttg.num-warps" = 8 : i32} {\n}\n')
        good = {"schema": "plain_champion", "kernel": "task/k.py",
                "source_ref": "champion/k.py", "source_sha": sha,
                "config": {"num_warps": 8, "BLOCK_SIZE_M": 128},
                "default_ms": 11.16, "sweep_winner_ms": 10.0, "champion_ms": 9.31,
                "ttgir": "ir/champion.ttgir", "served_range": [{"shape": "m=1", "ms": 9.31}],
                "partially_sampled": False, "trust_level": "pinned",
                "anchor": {"type": "pinned_comparator", "ref": "plain_best_config.json"},
                "path_coverage": [],
                "climb": {"rounds": 14, "kept": 3, "ledger_ref": "exp/rounds.jsonl"},
                "locus": {"docker": "abc123", "gpu": "4"}, "toolchain": {"arch": "gfx942"}}

        def run(mut=None, **kw):
            c = json.loads(json.dumps(good))
            if mut:
                mut(c)
            return gate(c, base, **kw)

        res, hard = run()
        assert not hard, [r for r in res if not r["ok"]]
        by = {r["check"]: r for r in res}
        assert by["CONFIG"]["ok"] and "ttg.num-warps=8" in by["CONFIG"]["msg"]
        assert by["COMPARATOR"]["ok"] and by["SAMPLING"]["ok"] and by["SOURCE"]["ok"]
        assert by["LIVE"]["ok"], by["LIVE"]

        # [LIVE] the failure [SOURCE] structurally cannot see: the frozen copy is pristine
        # while the file the RUN loads has been overwritten. Found on a real bundle where
        # the overwrite was a later Gluon track's installed winner, i.e. the run's "plain"
        # arm would have been Gluon and the gate said PASS.
        livef.write_text("# overwritten by a later track\n")
        res_l, hard_l = run()
        by_l = {r["check"]: r for r in res_l}
        assert by_l["SOURCE"]["ok"], "the frozen copy is untouched, so SOURCE must still pass"
        assert not by_l["LIVE"]["ok"], "LIVE must catch an overwritten `kernel`"
        assert hard_l, "an unanchored run must be a hard fail"
        assert "source_ref" in by_l["LIVE"]["msg"], "say how to restore it"
        livef.write_text("# champion kernel\n")
        assert {r["check"]: r for r in run()[0]}["LIVE"]["ok"], "restoring must clear [LIVE]"

        # a `kernel` that does not resolve is also unanchored
        res_m, hard_m = run(lambda c: c.update(kernel="task/gone.py"))
        assert not {r["check"]: r for r in res_m}["LIVE"]["ok"] and hard_m

        # a wrong schema short-circuits and never claims the other checks passed
        res, hard = gate({"schema": "plain_best_config"}, base)
        assert hard and len(res) == 1 and res[0]["check"] == "SCHEMA"
        # a missing required field is a hard SCHEMA fail, not a later confusing failure
        res, hard = run(lambda c: c.pop("ttgir"))
        assert hard and res[-1]["check"] == "SCHEMA" and "ttgir" in res[-1]["msg"]

        # THE CORE PROPERTY: an edited champion source can never pass. This is the defect the whole
        # bundle exists to prevent -- transcribing something other than what was measured.
        srcf.write_text("# champion kernel, edited after measurement\n")
        res, hard = run()
        assert hard and not {r["check"]: r for r in res}["SOURCE"]["ok"]
        srcf.write_text("# champion kernel\n")
        assert not run()[1], "restoring the source must clear [SOURCE]"

        # config mismatch is caught from the TTGIR itself...
        res, hard = run(lambda c: c["config"].__setitem__("num_warps", 4))
        assert hard and "ttg.num-warps=8" in {r["check"]: r for r in res}["CONFIG"]["msg"]
        # ...and from a sidecar, keyed on whatever names the kernel actually uses (BLOCK_SIZE_M here,
        # not the GEMM-canonical BLOCK_M -- the naming assumption that blocked 6 of 7 families).
        side = Path(str(irf) + ".config.json")
        side.write_text(json.dumps({"num_warps": 8, "BLOCK_SIZE_M": 64}))
        res, hard = run()
        assert hard and "BLOCK_SIZE_M" in {r["check"]: r for r in res}["CONFIG"]["msg"]
        side.write_text(json.dumps({"num_warps": 8, "BLOCK_SIZE_M": 128}))
        assert not run()[1], "an agreeing sidecar must pass"
        side.unlink()
        # no cross-checkable signal at all -> soft UNVERIFIED, not a silent pass and not a hard fail
        irf.write_text("module {\n}\n")
        res, hard = run()
        cc = {r["check"]: r for r in res}["CONFIG"]
        assert not hard and cc["ok"] and not cc["hard"] and "UNVERIFIED" in cc["msg"]
        irf.write_text('module attributes {"ttg.num-warps" = 8 : i32} {\n}\n')

        # comparator inversions are hard: slower than the default, and slower than the sweep winner
        assert run(lambda c: c.update(champion_ms=12.0))[1]
        assert run(lambda c: c.update(champion_ms=10.5, sweep_winner_ms=10.0))[1]
        # a null default_ms is soft but reported as unprovable, never silently fine
        res, hard = run(lambda c: c.update(default_ms=None))
        cmp_r = {r["check"]: r for r in res}["COMPARATOR"]
        assert not hard and cmp_r["ok"] and not cmp_r["hard"] and "UNPROVABLE" in cmp_r["msg"]

        # partial sampling blocks by default and only degrades when the caller declares it
        assert run(lambda c: c.update(partially_sampled=True))[1]
        res, hard = run(lambda c: c.update(partially_sampled=True), allow_provisional=True)
        assert not hard and "caveats[]" in {r["check"]: r for r in res}["SAMPLING"]["msg"]

        # [CLIMB]. A bundle that never says the line was climbed is the measured shape, and it is a
        # hard fail: this is the gate that ends the plain run, so silence here is the run choosing
        # to stop without saying it stopped.
        res, hard = run(lambda c: c.pop("climb"))
        cl = {r["check"]: r for r in res}["CLIMB"]
        assert hard and not cl["ok"] and "no `climb` at all" in cl["msg"], cl
        # Two rounds and nothing else is the same failure with a number attached.
        res, hard = run(lambda c: c.update(climb={"rounds": 2}))
        assert hard and "after 2 round(s)" in {r["check"]: r for r in res}["CLIMB"]["msg"]
        # The declared escape degrades rather than lies, exactly like --allow-provisional.
        res, hard = run(lambda c: c.update(climb={"rounds": 2}), allow_shallow_climb=True)
        cl = {r["check"]: r for r in res}["CLIMB"]
        assert not hard and cl["ok"] and "caveats[]" in cl["msg"], cl
        # A PRICED declination clears it without the flag -- and the price has to be a price. A bare
        # number, a missing upper-bound claim, or an artifact that does not exist all still fail,
        # because "an assertion wearing a number" is the cheapest way to end a search.
        probe = base / "exp" / "climb_probe.json"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(json.dumps({"measured_remaining_pct": 1.4}))
        full = {"remaining_prize_pct": 1.4, "is_upper_bound": True, "method": "paired A/B at the pin",
                "evidence_ref": {"kind": "probe", "artifact": "exp/climb_probe.json"}}
        res, hard = run(lambda c: (c.update(climb={"rounds": 2}), c.update(climb_declined=full)))
        cl = {r["check"]: r for r in res}["CLIMB"]
        assert not hard and cl["ok"] and "PRICED" in cl["msg"], cl
        for broken in ({**full, "is_upper_bound": False},
                       {**full, "evidence_ref": {"kind": "probe", "artifact": "exp/absent.json"}},
                       {**full, "method": ""},
                       {"remaining_prize_pct": "small", "is_upper_bound": True,
                        "method": "eyeball", "evidence_ref": full["evidence_ref"]},
                       "the rest is under 2%"):
            assert run(lambda c, b=broken: (c.update(climb={"rounds": 2}),
                                            c.update(climb_declined=b)))[1], broken
        # The floor is adopted from its owner, not re-declared here. Where the import resolves the
        # two must agree; where it does not, the message says the number is unverified.
        if CLIMB_FLOOR_FROM_OWNER:
            import search_mode                                    # noqa: PLC0415
            assert CLIMB_ROUND_FLOOR == search_mode.CLIMB_ROUND_FLOOR
        else:
            assert "fallback copy" in {r["check"]: r for r in run()[0]}["CLIMB"]["msg"]
        # At the floor exactly, it passes: the floor is a depth, not a quota to exceed.
        assert not run(lambda c: c.update(climb={"rounds": CLIMB_ROUND_FLOOR}))[1]

        # A SEARCHED space passes on the strength of its certificate, with no --allow-provisional: a
        # derived space is never enumerated, so demanding enumeration would block every sweep that
        # derived its space instead of typing one.
        def _cert(order=2, interior=True):
            return lambda c: c.update(partially_sampled=True,
                                      local_optimum={"certified": True, "order": order,
                                                     "interior": interior, "unprobed": []})
        res, hard = run(_cert())
        s = {r["check"]: r for r in res}["SAMPLING"]
        assert not hard and s["ok"] and "SEARCHED" in s["msg"], s
        # ...and it is quoted with the PAIRWISE number when the sweep has one, because the measured
        # fraction of a derived cross product rounds to 0% no matter how good the search was.
        res, _ = run(lambda c: (c.update(partially_sampled=True, grid_coverage_pct=0.0),
                                c.update(local_optimum={"certified": True, "order": 2, "interior": True,
                                                        "pair_coverage_pct": 39.4, "unprobed": []})))
        s = {r["check"]: r for r in res}["SAMPLING"]
        assert "39.4% of axis pairs" in s["msg"] and "0.0%" not in s["msg"], s
        # ...but only at order 2 AND in the interior. An order-1 certificate never tried a paired move,
        # and an optimum on a ladder END is the case where the unmeasured winner sits just outside.
        assert run(_cert(order=1))[1], "an order-1 certificate is not a substitute for coverage"
        assert run(_cert(interior=False))[1], "an optimum on the boundary does not certify the range"

        # an ungated sweep blocks by default and only degrades when the caller declares it. This is
        # the stronger of the two caveats: partial sampling means "incomplete measurement", ungated
        # means "the winner was never checked to be correct at all".
        assert run(lambda c: c.update(trust_level="ungated"))[1]
        res, hard = run(lambda c: c.update(trust_level="ungated"), allow_ungated=True)
        assert not hard and "caveats[]" in {r["check"]: r for r in res}["GATED"]["msg"]
        # --allow-provisional must NOT wave through an ungated bundle: different caveat, different flag
        assert run(lambda c: c.update(trust_level="ungated"), allow_provisional=True)[1]
        # a bundle predating trust_level is UNKNOWN, not assumed gated
        res, hard = run(lambda c: c.pop("trust_level", None))
        g = {r["check"]: r for r in res}["GATED"]
        assert not hard and not g["hard"] and "UNKNOWN" in g["msg"], g

        # A default anchor still refuses to START A DEEP DIG by default...
        _anchor = lambda c: c.update(anchor={"type": "production_default_anchor"},
                                     trust_level="production_default_anchor")
        assert run(_anchor)[1], "a default anchor may not silently authorize deep work"
        # ...and is DECLARED, not waived by hand, when the close is plain-only and asks for none.
        res, hard = run(_anchor, allow_default_anchor=True)
        an = {r["check"]: r for r in res}["ANCHOR"]
        assert not hard and an["ok"] and "may NOT start a deep dig" in an["msg"], an
        # The other two flags are different questions and must not stand in for this one.
        assert run(_anchor, allow_provisional=True)[1]
        assert run(_anchor, allow_ungated=True)[1]
        # ...and [ANCHOR] and [GATED] must now AGREE about the same bundle, which is the disagreement
        # that was being settled by hand.
        assert {r["check"]: r for r in res}["GATED"]["ok"], res

        # an empty served_range warns but does not block the deep dig from starting
        res, hard = run(lambda c: c.update(served_range=[]))
        assert not hard and not {r["check"]: r for r in res}["RANGE"]["ok"]
    print("[champion_gate] SELFTEST PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", required=True, help="plain_champion.json from the broad-search skill")
    ap.add_argument("--allow-provisional", action="store_true",
                    help="proceed on a partially-sampled comparator, with the inflation declared")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="proceed on a champion whose sweep had no oracle, with it declared")
    ap.add_argument("--allow-default-anchor", action="store_true",
                    help="the close is plain-only and never starts a deep dig, so a "
                         "production_default_anchor is the correct comparator for it. The bundle "
                         "still may not begin deep work; the limitation is declared, not removed")
    ap.add_argument("--allow-shallow-climb", action="store_true",
                    help="publish a champion whose plain line was climbed below the floor and whose "
                         "remainder was not priced, with the shortfall declared. Every deep-track "
                         "gain is then quoted against a line the plain run stopped climbing early")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    path = Path(a.champion).resolve()
    try:
        champ = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"[champion_gate] cannot read {path}: {e}", file=sys.stderr)
        return 2
    results, hard_fail = gate(champ, path.parent, allow_provisional=a.allow_provisional,
                              allow_ungated=a.allow_ungated,
                              allow_default_anchor=a.allow_default_anchor,
                              allow_shallow_climb=a.allow_shallow_climb)
    if a.json:
        print(json.dumps({"pass": not hard_fail, "checks": results}, indent=2))
    else:
        for r in results:
            tag = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "WARN")
            print(f"  [{tag}] {r['check']:11s} {r['msg']}")
        print("\n" + ("CHAMPION GATE PASS -> may start the deep dig"
                      if not hard_fail else
                      "CHAMPION GATE FAIL -> do NOT start (fix the FAILs above)"))
    return 0 if not hard_fail else 2


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
