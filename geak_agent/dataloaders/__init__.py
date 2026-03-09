# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# Lazy imports to avoid loading tb_eval dependency at import time
__all__ = [
    "ProblemState",
    "ProblemStateMLU",
    "tempCode",
    "TritonBench",
    "MLU",
]


def __getattr__(name):
    """Lazy import to avoid loading tb_eval at package import time."""
    if name in ("ProblemState", "ProblemStateMLU", "tempCode"):
        from geak_agent.dataloaders.ProblemState import ProblemState, ProblemStateMLU, tempCode
        if name == "ProblemState":
            return ProblemState
        elif name == "ProblemStateMLU":
            return ProblemStateMLU
        else:
            return tempCode
    elif name == "TritonBench":
        from geak_agent.dataloaders.TritonBench import TritonBench
        return TritonBench
    elif name == "MLU":
        from geak_agent.dataloaders.mlu import MLU
        return MLU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
