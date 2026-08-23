#!/usr/bin/env python3
"""Summarize on-contract results: within-instance spread and restart-to-restart spread, separately.

Result dirs are named <arm>_s<session>_r<run>_<timestamp>. Runs sharing a session label came from
one server instance; means across session labels give the restart-to-restart spread. Directories
with a probe_/profile_/sweep_/_invalid prefix are off-contract and skipped.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SKIP = ("probe_", "profile_", "sweep_")
PAT = re.compile(r"^(?P<arm>.+?)_(?P<sess>s\d+)_r(?P<run>\d+)_\d{8}_\d{6}$")


def load():
    arms = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name.startswith(SKIP) or "_invalid" in d.name:
            continue
        m = PAT.match(d.name)
        f = d / "inferencex_result.json"
        if not m or not f.exists():
            continue
        j = json.loads(f.read_text())
        # Re-check the workload contract on every point we are about to average.
        assert j["num_prompts"] == 192 and j["max_concurrency"] == 64, d.name
        arms.setdefault(m["arm"], {}).setdefault(m["sess"], []).append(
            (int(m["run"]), j["output_throughput"])
        )
    return arms


def p2p(xs):
    return 100.0 * (max(xs) - min(xs)) / st.mean(xs) if len(xs) > 1 else float("nan")


def main():
    arms = load()
    summary = {}
    for arm, sessions in sorted(arms.items()):
        print(f"\n=== arm: {arm} ===")
        means, allruns = [], []
        for sess, runs in sorted(sessions.items()):
            runs.sort()
            vals = [v for _, v in runs]
            allruns += vals
            means.append(st.mean(vals))
            print(
                f"  {sess}: "
                + " / ".join(f"{v:8.2f}" for v in vals)
                + f"   mean {st.mean(vals):8.2f}  within-instance p2p {p2p(vals):.3f}%"
            )
        print(f"  pooled mean over {len(allruns)} runs, {len(means)} sessions: {st.mean(allruns):.3f}")
        print(f"  RESTART-TO-RESTART  (spread of session means): {p2p(means):.3f}%")
        print(f"  run-level p2p       (all runs, all sessions)  : {p2p(allruns):.3f}%")
        summary[arm] = dict(
            pooled=st.mean(allruns), means=means, allruns=allruns,
            restart_p2p=p2p(means), run_p2p=p2p(allruns),
        )

    BASE = sys.argv[1] if len(sys.argv) > 1 else "r1ref"
    if len(summary) >= 2 and BASE in summary:
        base = summary[BASE]["pooled"]
        print(f"\n=== deltas vs {BASE} (pooled {base:.3f}) ===")
        for arm, s in sorted(summary.items()):
            if arm == BASE:
                continue
            d = 100.0 * (s["pooled"] - base) / base
            floor = max(summary[BASE]["restart_p2p"], s["restart_p2p"])
            print(f"  {arm}: {s['pooled']:.3f}  delta {d:+.3f}%   "
                  f"(restart floor on this boot {floor:.3f}%; round-1 floor 0.39%)")
            lo_fs, hi_r1 = min(s["allruns"]), max(summary[BASE]["allruns"])
            print(f"      run level    : worst {arm} {lo_fs:.2f} vs best {BASE} {hi_r1:.2f} -> "
                  f"{'SEPARATED' if lo_fs > hi_r1 else 'OVERLAP'}")

            # Session means are the independent unit: runs inside one server instance are
            # correlated, so treating all 9 as independent would overstate confidence.
            a, b = summary[BASE]["means"], s["means"]
            lo_m, hi_m = min(b), max(a)
            print(f"      session level: worst {arm} {lo_m:.2f} vs best {BASE} {hi_m:.2f} -> "
                  f"{'SEPARATED' if lo_m > hi_m else 'OVERLAP'}")
            if len(a) > 1 and len(b) > 1:
                va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
                se = (va + vb) ** 0.5
                t = (st.mean(b) - st.mean(a)) / se if se else float("inf")
                df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se else 0
                print(f"      Welch on session means: t={t:.2f}, df={df:.1f}, "
                      f"diff={st.mean(b)-st.mean(a):+.2f} tok/s, SE={se:.2f}")
            print(f"      VERDICT: delta {d:+.3f}% is "
                  f"{'ABOVE' if d > s['run_p2p'] else 'BELOW'} the run-level p2p floor "
                  f"({s['run_p2p']:.3f}%) -> "
                  f"{'claimable' if d > s['run_p2p'] else 'NOT claimed as a throughput win'}")


if __name__ == "__main__":
    sys.exit(main())
