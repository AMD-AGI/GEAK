"""Heterogeneous execution: LLM-generated diverse tasks dispatched across GPUs.

In heterogeneous mode the orchestrator asks an LLM to generate multiple
distinct optimization tasks and dispatches them across available GPU slots.
"""
