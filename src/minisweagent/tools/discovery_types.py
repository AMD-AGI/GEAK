"""Compatibility shim -- redirects to canonical location."""

import importlib
import sys

_impl = importlib.import_module("minisweagent.run.preprocess.discovery_types")
sys.modules[__name__] = _impl
