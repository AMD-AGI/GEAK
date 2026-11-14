"""Prepend this directory to PYTHONPATH and set GEAK_ITERATION_LOG_DIR to observe vLLM."""
import importlib.util
import os
import sys

from geak_vllm_observer import install

install()


def _chain():
    if getattr(sys, "_geak_iteration_env_chained", False):
        return
    sys._geak_iteration_env_chained = True
    here = os.path.dirname(os.path.abspath(__file__))
    seen = False
    for entry in sys.path:
        directory = os.path.abspath(entry or ".")
        if directory == here:
            seen = True
            continue
        candidate = os.path.join(directory, "sitecustomize.py")
        if not seen or not os.path.isfile(candidate):
            continue
        spec = importlib.util.spec_from_file_location("_geak_iteration_next_sitecustomize", candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return


_chain()
