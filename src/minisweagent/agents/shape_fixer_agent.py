"""Compatibility shim -- redirects to canonical location."""

import importlib
import sys

_impl = importlib.import_module("minisweagent.run.preprocess.shape_fixer_agent")
sys.modules[__name__] = _impl
