"""Explicit workflow scheduling limits for the normal Hyperloom handoff.

These controls allocate the caller's existing time budget. They never change
benchmark durations, serving configuration, or the caller's hard deadline.
"""
import os


class SchedulingError(ValueError):
    """The requested schedule cannot fit the configured measurement protocol."""


# Handoff values win over environment defaults, including an explicit zero.
LIMITS = {
    "budget": ("GEAK_KERNEL_TASK_BUDGET", 0),
    "min_kernel_tasks": ("GEAK_MIN_KERNEL_TASKS", 0),
    "kernel_budget": ("GEAK_KERNEL_ROUND_BUDGET", 1),
    "head_budget": ("GEAK_HEAD_BUDGET", 0),
    "head_author_max": ("GEAK_HEAD_AUTHOR_MAX", 0),
    "head_author_tries": ("GEAK_HEAD_AUTHOR_TRIES", 1),
    "head_corrective_max": ("GEAK_HEAD_CORRECTIVE_MAX", 0),
    "ab_finish_retries": ("GEAK_AB_FINISH_RETRIES", 0),
    "baseline_extract_retries": ("GEAK_BASELINE_EXTRACT_RETRIES", 0),
    "search_deadline_s": ("GEAK_SEARCH_DEADLINE_S", 1),
}
SWITCHES = {
    "tuning_skillset": "GEAK_TUNING_SKILLSET",
    "use_learned_kb": "GEAK_USE_LEARNED_KB",
    "fast_mode": "GEAK_FAST_MODE",
    "deep_mode": "GEAK_DEEP_MODE",
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SchedulingError(f"{name} must be an integer >= {minimum}")
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal() or int(text) < minimum:
        raise SchedulingError(f"{name} must be an integer >= {minimum}")
    result = int(text)
    if result > 2**53 - 1:
        raise SchedulingError(f"{name} exceeds the workflow's safe integer range")
    return result


def _requested(handoff, env, name, env_name):
    if handoff.get(name) is not None:
        return handoff[name]
    value = env.get(env_name)
    return value if value is not None and str(value).strip() else None


def apply_schedule(handoff, args, env=None):
    """Apply an allowlist and return the effective finite schedule for dry-run.

    No supplied scheduling controls means no change to the historical defaults.
    AgentX admission checks are duration lower bounds, not runtime guarantees:
    callers must still budget for server boots, internal warmup and quality.
    """
    env = os.environ if env is None else env
    requested = {}
    for name, (env_name, minimum) in LIMITS.items():
        value = _requested(handoff, env, name, env_name)
        if value is not None:
            requested[name] = _integer(value, name, minimum)
    for name, env_name in SWITCHES.items():
        if name == "tuning_skillset" and handoff.get(name) is not None:
            # This pre-existing handoff switch is already normalized by
            # run_e2e. It must not opt an old caller into a new timing contract.
            continue
        value = _requested(handoff, env, name, env_name)
        if value is not None:
            text = str(value).strip().lower()
            if text not in {"true", "false", "1", "0", "on", "off"}:
                raise SchedulingError(f"{name} must be true or false")
            requested[name] = "true" if text in {"true", "1", "on"} else "false"
    warm_start = _requested(handoff, env, "warm_start", "GEAK_WARM_START")
    if warm_start is not None:
        warm_start = str(warm_start).strip().lower()
        if warm_start not in {"on", "off", "false", "none", "reference", "return_after_read"}:
            raise SchedulingError("warm_start must be on, off, reference, or return_after_read")
        requested["warm_start"] = warm_start
    # Existing timing environment values are mapped by run_e2e.py. Allow the
    # same explicit handoff controls without introducing a generic args escape.
    for name in ("final_reserve_s", "agent_timeout_ms", "time_tail_cap_s"):
        if handoff.get(name) is not None:
            requested[name] = _integer(handoff[name], name)
    args.update(requested)

    budget = args.get("time_budget_s")
    reserve = args.get("final_reserve_s")
    if budget is not None and reserve is not None and reserve >= budget:
        raise SchedulingError("final_reserve_s must be smaller than the actual time_budget_s")
    effective_reserve = (reserve if reserve is not None else min(3600, budget * 0.2)) if budget else None
    deadline = args.get("search_deadline_s")
    if deadline is not None:
        if budget is None or deadline > budget - effective_reserve:
            raise SchedulingError("search_deadline_s requires a finite budget and must precede its final reserve")

    # Preserve callers that have not opted into explicit scheduling. Timing-only
    # AgentX schedules are also checked: a reserve smaller than its measurements
    # must fail before Setup launches a server.
    configured = bool(requested) or reserve is not None or "agent_timeout_ms" in args
    if not configured:
        return None
    effective_deadline = deadline
    if budget is not None and effective_deadline is None:
        available = budget - effective_reserve
        effective_deadline = max(int(available * 1000 * 0.6) / 1000, available - args.get("time_tail_cap_s", 10800))
    report = {
        "requested_controls": requested,
        "time_budget_s": budget,
        "final_reserve_s": effective_reserve,
        "search_deadline_s": effective_deadline,
        "measurement_durations_changed": False,
    }
    spec = handoff.get("workload_spec")
    spec = spec if isinstance(spec, dict) else {}
    phases = {phase.strip() for phase in str(args.get("phases", "all")).split(",")}
    if spec.get("kind") != "agentx_trace_replay" or not ({"all", "final"} & phases):
        return report
    if budget is None:
        raise SchedulingError("bounded AgentX final validation requires an actual finite time_budget_s")
    full = _requested(spec, env, "duration_s", "GEAK_AGENTX_DURATION_S")
    loop = _requested(spec, env, "geak_loop_duration_s", "GEAK_AGENTX_LOOP_DURATION_S")
    full = _integer(3600 if full is None else full, "AgentX duration_s")
    loop = _integer(900 if loop is None else loop, "AgentX geak_loop_duration_s")
    # apply_workload_spec sets REPEATS=1 whenever it is absent, even if an
    # inherited REPLICAS exists. bench_e2e then gives REPEATS precedence. Keep
    # this normal-interface preflight aligned with that actual export order.
    samples = _integer(env.get("REPEATS", "1"), "REPEATS")
    validation = 2 * (1 + samples) * full
    final_min = validation + (1 + samples) * loop
    setup_min = (1 + samples) * full if {"all", "setup"} & phases else 0
    kernel_search = bool({"all", "kernel"} & phases) and args.get("budget", 6) > 0
    head_search = bool({"all", "head"} & phases) and args.get("head_budget", 3) > 0
    integration_min = 2 * (1 + samples) * loop if kernel_search or head_search else 0
    report["minimum_client_seconds"] = {
        "setup": setup_min, "one_integration": integration_min,
        "validate": validation, "final_phase": final_min,
    }
    report["overhead_included"] = False
    if effective_reserve <= final_min:
        raise SchedulingError(f"AgentX final_reserve_s must exceed {final_min}s of full warm-server measurements, plus launch/warmup/report overhead")
    if args.get("agent_timeout_ms", 7200000) <= validation * 1000:
        raise SchedulingError(f"AgentX agent_timeout_ms must exceed {validation * 1000}ms for both full validation legs, plus overhead")
    if budget - effective_reserve <= setup_min + integration_min:
        raise SchedulingError(f"AgentX budget before the final reserve must exceed {setup_min + integration_min}s of Setup measurements and one candidate A/B, plus optimization and overhead")
    if deadline is not None and integration_min and deadline <= setup_min:
        raise SchedulingError("AgentX search_deadline_s must leave time after the full Setup baseline to dispatch a candidate")
    return report
