"""Compatibility shim -- redirects to canonical location."""

import importlib
import sys

_impl = importlib.import_module("minisweagent.run.preprocess.validate_commandment")
sys.modules[__name__] = _impl
