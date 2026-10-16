"""Small helpers shared by the GEAK-EVO modules.

Saved record digests depend on the exact bytes of ``canonical_json``. Do not
change its separators, key order, or number handling.
"""

import hashlib
import json
import re
from decimal import Decimal

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EvoError(Exception):
    """Base for named engine errors; legacy built-in error types stay unchanged."""


def require(condition, message, error=EvoError):
    """Raise ``error(message)`` unless ``condition`` is true."""
    if not condition:
        raise error(message)


def _decimal_to_text(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError("not JSON serializable")


def canonical_json(value, *, allow_decimal=False, ensure_ascii=True):
    """Sorted, compact, finite JSON bytes (UTF-8).

    ``allow_decimal`` writes Decimal values as strings (the loop's ledger uses
    this); everywhere else a Decimal is an error, as before the port.
    ``ensure_ascii=False`` keeps non-ASCII text as UTF-8; the engineer, job and
    planner records were saved that way and must keep their bytes.
    """
    default = _decimal_to_text if allow_decimal else None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii,
                      allow_nan=False, default=default).encode("utf-8")


def sha256_hex(data):
    """Lowercase hex SHA-256 of ``data`` (bytes)."""
    return hashlib.sha256(data).hexdigest()


def is_sha256(value):
    """True for a lowercase 64-character hex digest string."""
    return type(value) is str and SHA256_PATTERN.fullmatch(value) is not None
