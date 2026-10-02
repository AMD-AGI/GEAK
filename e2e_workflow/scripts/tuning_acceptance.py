"""Shared acceptance predicate for the tuning phase's own complete pre/post pair."""
import math


def complete_tuning_pair(tuning):
    if not isinstance(tuning, dict) or tuning.get("ab_complete") is not True:
        return False
    try:
        return all(type(value) in (int, float) and math.isfinite(value) and value > 0
                   for value in (tuning.get("pre_tune_throughput_tok_s"),
                                 tuning.get("post_tune_throughput_tok_s")))
    except OverflowError:
        return False


def tuning_accepted(tuning, accuracy_gate=None):
    if not complete_tuning_pair(tuning):
        return False
    requested = accuracy_gate if accuracy_gate is not None else tuning.get("accuracy_gate", "none")
    correctness = tuning.get("correctness_gate")
    correctness_ok = correctness == "pass" or (
        requested == "none" and correctness in ("none", "skipped")
    )
    return (tuning.get("gate") == "accepted" and tuning.get("ran") is not False
            and tuning.get("engagement_verified") is True and correctness_ok
            and tuning["post_tune_throughput_tok_s"] > tuning["pre_tune_throughput_tok_s"])
