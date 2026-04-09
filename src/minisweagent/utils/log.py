"""Central logging for the minisweagent package.

Use ``logging.getLogger(__name__)`` in package modules so records propagate to
the ``minisweagent`` logger configured here. For entrypoints that are not
package modules, import ``logger`` from this module.
"""

import logging
import os
from pathlib import Path

from rich.logging import RichHandler


def _get_log_level_from_env() -> int:
    # Default to INFO unless explicitly overridden by env.
    level_name = os.getenv("MINISWEAGENT_LOG_LEVEL", "INFO").strip().upper()
    return logging._nameToLevel.get(level_name, logging.INFO)


def _silence_noisy_loggers() -> None:
    # Prevent third-party HTTP client INFO logs (e.g. "HTTP Request: POST ...")
    # from cluttering our file logs when root handlers are configured elsewhere.
    for name in ("httpx", "httpcore", "metrix"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(_get_log_level_from_env())
    if logger.handlers:
        _silence_noisy_loggers()
        return
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.propagate = False
    _silence_noisy_loggers()


def add_file_handler(path: Path | str, level: int | None = None, *, print_path: bool = True) -> None:
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level if level is not None else _get_log_level_from_env())
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


DEFAULT_LOG_FILENAME = "minisweagent.log"

_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["add_file_handler", "logger", "DEFAULT_LOG_FILENAME"]
