"""Central logging for the minisweagent package.

Use ``logging.getLogger(__name__)`` in package modules so records propagate to
the ``minisweagent`` logger configured here. For entrypoints that are not
package modules, import ``logger`` from this module.
"""

import logging
import os
import re
from pathlib import Path

from rich.logging import RichHandler


def _get_log_level_from_env() -> int:
    # Default to INFO unless explicitly overridden by env.
    level_name = os.getenv("MINISWEAGENT_LOG_LEVEL", "INFO").strip().upper()
    return logging._nameToLevel.get(level_name, logging.INFO)


def _silence_noisy_loggers() -> None:
    # Prevent third-party HTTP client INFO logs (e.g. "HTTP Request: POST ...")
    # from cluttering our file logs when root handlers are configured elsewhere.
    # Set ``MINISWEAGENT_VERBOSE_HTTP=1`` to keep them at INFO so retry / RateLimit
    # / 5xx responses from upstream LLM endpoints land in geak_agent.log.
    if os.getenv("MINISWEAGENT_VERBOSE_HTTP", "").strip() == "1":
        return
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


class _ProgressTickFilter(logging.Filter):
    """Suppress transient progress-tick records from the file handler.

    Progress threads tag their records with ``extra={"progress_tick": True}``.
    The terminal (RichHandler) still shows every tick; only file output is filtered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress_tick", False)


class _MarkupStrippingFormatter(logging.Formatter):
    """Strip Rich markup tags so log files contain plain text.

    The terminal handler (RichHandler) renders ``[bold cyan]...[/bold cyan]``
    as ANSI colours.  File handlers should not see those tags.  This formatter
    removes only known Rich style keywords to avoid false positives on
    legitimate bracket content like ``[0]`` or ``[ARCHITECTURE ALERT]``.
    """

    _RICH_TAG_RE = re.compile(
        r"\[/\]"
        r"|\[/?(?:bold|dim|italic|underline|red|green|blue|cyan|yellow|magenta|white|black|not)"
        r"(?:\s+(?:bold|dim|italic|underline|red|green|blue|cyan|yellow|magenta|white|black|not))*\]"
    )

    def format(self, record: logging.LogRecord) -> str:
        return self._RICH_TAG_RE.sub("", super().format(record))


def add_file_handler(path: Path | str, level: int | None = None, *, print_path: bool = True) -> None:
    """Attach a single FileHandler to BOTH the ``minisweagent`` subtree
    AND the root logger.

    Why both:
      * The ``minisweagent`` logger has ``propagate=False`` (set in
        :func:`_setup_root_logger`), so attaching only to it would miss
        records emitted by external dependencies that don't sit under
        ``minisweagent.*`` — e.g. ``tenacity`` (retry warnings),
        ``litellm`` / ``LiteLLM`` (RateLimitError, 5xx responses),
        ``anthropic``, ``openai``, ``urllib3``.
      * Root logger captures everything else, so retry / throttling
        events from the LLM API end up in ``geak_agent.log`` instead
        of being lost to the terminal scrollback.

    The same ``FileHandler`` instance is shared between the two
    loggers; records emitted from ``minisweagent.*`` are routed
    directly (since that subtree doesn't propagate) and records from
    everywhere else hit the root handler — no duplicates because
    ``minisweagent.propagate`` is ``False``.
    """
    handler = logging.FileHandler(path)
    handler.setLevel(level if level is not None else _get_log_level_from_env())
    handler.addFilter(_ProgressTickFilter())
    formatter = _MarkupStrippingFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logging.getLogger("minisweagent").addHandler(handler)

    root = logging.getLogger()
    # The root logger defaults to WARNING; without this its handlers
    # would silently drop INFO records even when our FileHandler is
    # set to INFO/DEBUG.  Match the file handler's level so anything
    # the handler is willing to emit actually reaches it.
    if root.level == logging.NOTSET or root.level > handler.level:
        root.setLevel(handler.level)
    root.addHandler(handler)

    if print_path:
        print(f"Logging to '{path}'")


DEFAULT_LOG_FILENAME = "geak_agent.log"

_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["add_file_handler", "logger", "DEFAULT_LOG_FILENAME"]
