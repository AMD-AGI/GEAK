"""Map a horizon seal or a validated final recommendation to the run outcome.

``final_outcome`` reads the selection only:

* ``selected``: the recommendation is ``selected``; ``best`` is that choice.
  A choice has passed the native correctness check, but nothing here says it
  is faster than ORIGINAL (see ``choices``).
* ``no_improvement``: the recommendation is ``missing`` because the planner's
  last applied record is an explicit ``clear`` (keep ORIGINAL); ``best`` is None.
* ``error``: anything else (unresolved seal, no campaign revision before the
  horizon, no applied recommendation, unknown status); ``best`` is None.

``measured_outcome`` then takes the caller's matched measurement of the
selected program against ORIGINAL (speedup = ORIGINAL time / selected time)
and reports ``improved`` only when that speedup is above 1. A slower or equal
selected program gives ``no_improvement`` with ``best`` None; the choice is kept
under ``selected`` so the archive entry is not lost.

No other program is ever substituted.
"""

import math

SEAL_SCHEMA = "geak-committed-horizon-endpoint-v1"


def final_outcome(result):
    """Return ``{"outcome", "best", "detail"}`` for a seal or a validated recommendation."""
    if type(result) is not dict:
        return {"outcome": "error", "best": None, "detail": "no_recommendation"}
    if result.get("schema") == SEAL_SCHEMA:
        if result.get("status") not in ("selected", "missing"):
            return {"outcome": "error", "best": None, "detail": result.get("error_reason") or "unresolved"}
        if result.get("recommendation") is None:
            return {"outcome": "error", "best": None, "detail": result.get("missing_reason") or "no_recommendation"}
        result = result["recommendation"]
    status, record = result.get("status"), result.get("record")
    if status == "selected" and type(result.get("choice")) is dict:
        return {"outcome": "selected", "best": result["choice"], "detail": "selected"}
    if status == "missing" and type(record) is dict and record.get("disposition") == "clear":
        return {"outcome": "no_improvement", "best": None, "detail": "clear"}
    return {"outcome": "error", "best": None, "detail": "no_applied_recommendation" if status == "missing"
            else "unknown_status"}


def measured_outcome(outcome, speedup):
    """Turn a ``selected`` outcome into ``improved`` or ``no_improvement``.

    ``speedup`` is the caller's matched ORIGINAL/selected measurement. Other
    outcomes are returned unchanged and need no measurement.
    """
    if outcome["outcome"] != "selected":
        return outcome
    if type(speedup) not in (int, float) or not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("a finite positive measured speedup is required for a selected program")
    if speedup > 1:
        return {"outcome": "improved", "best": outcome["best"], "detail": "selected", "speedup": speedup}
    return {"outcome": "no_improvement", "best": None, "detail": "selected_not_faster",
            "selected": outcome["best"], "speedup": speedup}
