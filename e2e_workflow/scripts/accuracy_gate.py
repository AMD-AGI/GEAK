#!/usr/bin/env python3
"""Deterministic gsm8k accuracy gate: baseline vs candidate on the SAME questions.

Two rules were in use and each failed one way. A fixed `cand >= base - tol`
rejects on sampling noise: at n=200 one question is 0.5pt and the SE is ~1.8pt,
and a real +1.3% win was dropped on a z~1.0 "drop". A noise-only test passes
anything it cannot detect: on Qwen3.5-35B-A3B-FP8 a bf16 fusion measured
0.895 -> 0.870 at n=200 (p=0.44) and would have shipped unexamined.

So the verdict has three states:

* pass         -- the observed drop is within ACCURACY_TOL (no material drop);
* fail         -- the drop exceeds ACCURACY_TOL AND is significant under a
                  one-sided exact McNemar test on the paired per-question
                  outcomes (both legs answer the same questions, greedy);
* inconclusive -- the drop exceeds ACCURACY_TOL but n cannot resolve it.
                  Re-run both legs on more questions (up to the full test set)
                  and gate again. At the full set it becomes pass with a
                  warning: a drop that large would have been detected.

Pairing matters: discordant questions carry all the information, so the test
is far sharper than comparing two independent proportions.
"""
import argparse
import json
import math
import sys

GSM8K_TEST_SIZE = 1319


def _load(path):
    with open(path) as fh:
        doc = json.load(fh)
    return {int(item["idx"]): bool(item["ok"]) for item in doc.get("results", [])}


def _binom_tail(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / float(2 ** n) if n else 1.0


def gate(base, cand, tol, alpha=0.05, full_size=GSM8K_TEST_SIZE):
    if set(base) != set(cand):
        raise ValueError(
            "legs answered different questions (%d vs %d, %d shared); re-run both "
            "with the same --limit and --seed" % (
                len(base), len(cand), len(set(base) & set(cand))))
    n = len(base)
    if not n:
        raise ValueError("no per-question results")
    lost = sum(1 for i in base if base[i] and not cand[i])    # base right, cand wrong
    gained = sum(1 for i in base if cand[i] and not base[i])  # cand right, base wrong
    base_acc = sum(base.values()) / float(n)
    cand_acc = sum(cand.values()) / float(n)
    drop = base_acc - cand_acc
    # One-sided exact McNemar: among discordant questions, is `lost` improbably large?
    p_value = _binom_tail(lost, lost + gained)
    if drop <= tol:
        verdict, reason = "pass", "drop %.4f within tolerance %.4f" % (drop, tol)
    elif p_value < alpha:
        verdict, reason = "fail", (
            "drop %.4f exceeds tolerance %.4f and is significant (McNemar p=%.4f < %.2f)"
            % (drop, tol, p_value, alpha))
    elif n >= full_size:
        verdict, reason = "pass", (
            "drop %.4f exceeds tolerance %.4f but is not significant on the full test "
            "set (McNemar p=%.4f); recorded as a warning" % (drop, tol, p_value))
    else:
        verdict, reason = "inconclusive", (
            "drop %.4f exceeds tolerance %.4f but n=%d cannot resolve it (McNemar "
            "p=%.4f); re-run both legs with a larger --limit (full set: %d)"
            % (drop, tol, n, p_value, full_size))
    return {
        "verdict": verdict,
        "reason": reason,
        "n": n,
        "base_accuracy": round(base_acc, 6),
        "cand_accuracy": round(cand_acc, 6),
        "drop": round(drop, 6),
        "tolerance": tol,
        "alpha": alpha,
        "base_right_cand_wrong": lost,
        "cand_right_base_wrong": gained,
        "mcnemar_one_sided_p": round(p_value, 6),
        "warning": verdict == "pass" and drop > tol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="gsm8k_eval.py --out of the baseline leg")
    parser.add_argument("--cand", required=True, help="gsm8k_eval.py --out of the candidate leg")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="largest drop that is not material (ACCURACY_TOL)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = gate(_load(args.base), _load(args.cand), args.tol, args.alpha)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return {"pass": 0, "fail": 1, "inconclusive": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
