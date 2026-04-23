"""Centralized pipeline timeout constants.

Two knobs control all correctness/benchmark/profile timeouts across
preprocessing, agent execution, and evaluation:

  * ``GEAK_TEST_TIMEOUT``    – correctness, benchmark, full-benchmark, agent test
  * ``GEAK_PROFILE_TIMEOUT`` – profile harness mode, eval profile, profiler warmup

Each can be set via ``geak.yaml`` or an environment variable.  When both are
set and disagree, ``geak.yaml`` wins and a warning is logged.

YAML keys:
  * ``env.timeout``           → GEAK_TEST_TIMEOUT    (default 3600 s)
  * ``tools.profile_timeout`` → GEAK_PROFILE_TIMEOUT (default 3600 s)

Environment variables:
  * ``GEAK_TEST_TIMEOUT``
  * ``GEAK_PROFILE_TIMEOUT``
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_TEST_TIMEOUT = 3600
_DEFAULT_PROFILE_TIMEOUT = 3600


def _load_yaml_config() -> dict:
    try:
        from minisweagent.config import load_config

        return load_config("geak")
    except Exception:
        return {}


def _resolve(yaml_value: int | None, env_var: str, default: int) -> int:
    env_raw = os.environ.get(env_var)
    env_value = int(env_raw) if env_raw is not None else None

    if yaml_value is not None and env_value is not None and yaml_value != env_value:
        logger.warning(
            "%s=%d from env differs from geak.yaml value %d; using geak.yaml",
            env_var,
            env_value,
            yaml_value,
        )

    if yaml_value is not None:
        return yaml_value
    if env_value is not None:
        return env_value
    return default


def _init() -> tuple[int, int]:
    cfg = _load_yaml_config()
    env_cfg = cfg.get("env", {}) or {}
    tools_cfg = cfg.get("tools", {}) or {}

    yaml_test = env_cfg.get("timeout")
    yaml_profile = tools_cfg.get("profile_timeout")

    if yaml_test is not None:
        yaml_test = int(yaml_test)
    if yaml_profile is not None:
        yaml_profile = int(yaml_profile)

    test = _resolve(yaml_test, "GEAK_TEST_TIMEOUT", _DEFAULT_TEST_TIMEOUT)
    profile = _resolve(yaml_profile, "GEAK_PROFILE_TIMEOUT", _DEFAULT_PROFILE_TIMEOUT)
    return test, profile


GEAK_TEST_TIMEOUT, GEAK_PROFILE_TIMEOUT = _init()
