"""Config sub-package: configuration loading, editing, and parsing.

Key entry points:

- :func:`~.editor.load_and_merge_configs` -- merge CLI / prompt / YAML configs
- :func:`~.global_config.configure_if_first_time` -- first-run setup
- :func:`~.task_parser.parse_task_info` -- extract config from task content
- :func:`deep_merge` -- recursive dict merge
"""

from __future__ import annotations


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
