# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical bytes and hashes used by native source and evidence identities.

Keep the JSON separators, key ordering and number handling stable: saved
source, candidate and request identities depend on these exact bytes.
"""

import hashlib
import json


def require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def canonical_json(value, *, ensure_ascii=True):
    """Sorted, compact, finite JSON bytes (UTF-8)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ensure_ascii, allow_nan=False).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()
