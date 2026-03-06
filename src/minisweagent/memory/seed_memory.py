"""Seed the cross-session memory DB with data from GEAK eval runs.

This pre-populates the memory with known outcomes so the agent can
leverage past experience on subsequent runs.
"""

from minisweagent.memory.cross_session_memory import CrossSessionMemory


def seed_from_eval_runs(db_path=None):
    """Seed memory with actual outcomes from GEAK eval Run 1 + Run 2."""
    m = CrossSessionMemory(db_path)

    outcomes = [
        # Run 1 results
        {"kernel_category": "rope", "bottleneck_type": "balanced", "speedup_achieved": 2.94,
         "success": True, "cost_dollars": 0.27, "steps_taken": 93, "commandment_worked": True,
         "strategy_name": "direct_half_loading"},
        {"kernel_category": "rope", "bottleneck_type": "balanced", "speedup_achieved": 2.87,
         "success": True, "cost_dollars": 0.05, "steps_taken": 21, "commandment_worked": False,
         "strategy_name": "baseline", "failure_reason": "COMMANDMENT.md creation failed"},
        {"kernel_category": "topk", "bottleneck_type": "latency", "speedup_achieved": 1.53,
         "success": True, "cost_dollars": 0.12, "steps_taken": 43, "commandment_worked": True,
         "strategy_name": "manual_optimization"},
        {"kernel_category": "topk", "bottleneck_type": "latency", "speedup_achieved": 1.46,
         "success": True, "cost_dollars": 0.25, "steps_taken": 116, "commandment_worked": True,
         "strategy_name": "openevolve", "failure_reason": "OE BrokenPipeError at iteration 3"},
        {"kernel_category": "gemm", "bottleneck_type": "lds", "speedup_achieved": 0.25,
         "success": False, "cost_dollars": 0.10, "steps_taken": 39, "commandment_worked": True,
         "strategy_name": "openevolve",
         "failure_reason": "Triton GEMM fundamentally slower than rocBLAS (F.linear)"},
        {"kernel_category": "gemm", "bottleneck_type": "lds", "speedup_achieved": 0.15,
         "success": False, "cost_dollars": 0.09, "steps_taken": 36, "commandment_worked": True,
         "strategy_name": "openevolve",
         "failure_reason": "rocBLAS/cuBLAS is fundamentally faster for standard GEMM"},
        # fused_rms_fp8
        {"kernel_category": "fused", "bottleneck_type": "latency", "speedup_achieved": 2.02,
         "success": True, "cost_dollars": 0.11, "steps_taken": 35, "commandment_worked": True,
         "strategy_name": "tl_minimum_maximum"},
        # ff_backward
        {"kernel_category": "feedforward", "bottleneck_type": "compute", "speedup_achieved": 1.03,
         "success": True, "cost_dollars": 0.34, "steps_taken": 100, "commandment_worked": True,
         "strategy_name": "manual_tuning"},
        # nsa_backward
        {"kernel_category": "attention", "bottleneck_type": "memory", "speedup_achieved": 1.52,
         "success": True, "cost_dollars": 0.23, "steps_taken": 67, "commandment_worked": True,
         "strategy_name": "openevolve"},
        {"kernel_category": "attention", "bottleneck_type": "memory", "speedup_achieved": 0.88,
         "success": False, "cost_dollars": 0.17, "steps_taken": 65, "commandment_worked": True,
         "strategy_name": "openevolve", "failure_reason": "OE candidates worse than baseline"},
        # nsa_forward
        {"kernel_category": "attention", "bottleneck_type": "latency", "speedup_achieved": 4.01,
         "success": True, "cost_dollars": 0.15, "steps_taken": 68, "commandment_worked": False,
         "strategy_name": "baseline", "failure_reason": "COMMANDMENT SETUP parsing error"},
    ]

    for o in outcomes:
        m.record_outcome(
            kernel_type="triton",
            gpu_architecture="gfx942",
            profiling_metrics={},
            kernel_signature_hash="",
            **o,
        )

    # Record known pitfalls
    pitfalls = [
        ("gemm", "architecture_mismatch",
         "Triton GEMM is fundamentally slower than rocBLAS for standard shapes. "
         "Don't waste time optimizing -- the Triton kernel cannot beat F.linear."),
        ("gemm", "commandment_validation",
         "COMMANDMENT.md SETUP section must not use shell built-ins like 'export' or 'cd'. "
         "Wrap in bash -c or use absolute paths."),
        ("attention", "oe_instability",
         "OpenEvolve may produce candidates worse than baseline for attention backward kernels. "
         "Manual optimization may be more effective."),
        ("rope", "commandment_creation",
         "Agent often fails to create COMMANDMENT.md for rope kernels. "
         "Use auto-generated COMMANDMENT from commandment_library."),
        ("topk", "oe_crash",
         "OpenEvolve BrokenPipeError at iteration 3 for topk kernels. "
         "Ensure proper process isolation and error handling."),
    ]
    for cat, ptype, desc in pitfalls:
        m.record_pitfall(cat, ptype, desc)

    stats = m.get_stats()
    m.close()

    # Seed strategic principles
    try:
        from minisweagent.memory.procedural_memory import seed_principles
        seed_principles()
    except Exception:
        pass

    # Seed ReMe insights
    try:
        from minisweagent.memory.reme_memory import seed_reme_memory
        seed_reme_memory()
    except Exception:
        pass

    return stats


if __name__ == "__main__":
    stats = seed_from_eval_runs()
    print(f"Seeded memory DB: {stats}")
