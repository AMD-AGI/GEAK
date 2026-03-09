# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

"""
GEAK-Agent: An LLM-based multi-agent framework for generating functional and efficient GPU kernels.
"""

__version__ = "0.1.0"

# Lazy imports - only import when accessed to avoid loading tb_eval dependency at import time
__all__ = [
    # Version
    "__version__",
    # Agents
    "BaseAgent",
    "SequentialBaseAgent",
    "GaAgent",
    "Reflexion",
    "Reflexion_Oneshot",
    "OptimAgent",
    "DirectPrompt",
    # Models
    "BaseModel",
    "OpenAIModel",
    "StandardOpenAIModel",
    "ClaudeModel",
    "StandardClaudeModel",
    "GeminiModel",
    # Dataloaders
    "TritonBench",
    "MLU",
    "ProblemState",
    "ProblemStateMLU",
    # Utils
    "load_config",
]


def __getattr__(name):
    """Lazy import to avoid loading tb_eval at package import time."""
    if name in ("BaseAgent", "SequentialBaseAgent"):
        from geak_agent.agents.Base import BaseAgent, SequentialBaseAgent
        return BaseAgent if name == "BaseAgent" else SequentialBaseAgent
    elif name == "GaAgent":
        from geak_agent.agents.GaAgent import GaAgent
        return GaAgent
    elif name == "Reflexion":
        from geak_agent.agents.Reflexion import Reflexion
        return Reflexion
    elif name == "Reflexion_Oneshot":
        from geak_agent.agents.reflexion_oneshot import Reflexion_Oneshot
        return Reflexion_Oneshot
    elif name == "OptimAgent":
        from geak_agent.agents.OptimAgent import OptimAgent
        return OptimAgent
    elif name == "DirectPrompt":
        from geak_agent.agents.DirectPrompt import DirectPrompt
        return DirectPrompt
    elif name == "BaseModel":
        from geak_agent.models.Base import BaseModel
        return BaseModel
    elif name in ("OpenAIModel", "StandardOpenAIModel"):
        from geak_agent.models.OpenAI import OpenAIModel, StandardOpenAIModel
        return OpenAIModel if name == "OpenAIModel" else StandardOpenAIModel
    elif name in ("ClaudeModel", "StandardClaudeModel"):
        from geak_agent.models.Claude import ClaudeModel, StandardClaudeModel
        return ClaudeModel if name == "ClaudeModel" else StandardClaudeModel
    elif name == "GeminiModel":
        from geak_agent.models.Gemini import GeminiModel
        return GeminiModel
    elif name == "TritonBench":
        from geak_agent.dataloaders.TritonBench import TritonBench
        return TritonBench
    elif name == "MLU":
        from geak_agent.dataloaders.mlu import MLU
        return MLU
    elif name in ("ProblemState", "ProblemStateMLU"):
        from geak_agent.dataloaders.ProblemState import ProblemState, ProblemStateMLU
        return ProblemState if name == "ProblemState" else ProblemStateMLU
    elif name == "load_config":
        from geak_agent.args_config import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
