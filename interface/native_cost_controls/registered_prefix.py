# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register an exact source catalog before native cache requests arrive."""

import re
from dataclasses import dataclass
from pathlib import Path

from .native_cache_layout import content_layout_eligible
from .native_prefix_marker import (
    DECODER,
    Decision,
    client_tool,
    mark_registered_prefix,
    sha,
    wire,
)

POLICY_NAME = "registered_native_prefix_v1"


@dataclass(frozen=True, init=False)
class RegisteredToolPrefix:
    """Keep only immutable fingerprints from a verified catalog artifact.

    The catalog is a JSON array of the exact leading tools to share. Its
    producer must record the native CLI, source entrypoint, and tool filters.
    This class never learns definitions from a provider request.
    """

    count: int
    prefix_sha256: str
    catalog_sha256: str

    def __init__(self, catalog: bytes, *, expected_sha256: str):
        if not isinstance(catalog, bytes):
            raise TypeError("The source catalog must contain bytes.")
        if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
            raise ValueError("The source catalog hash is invalid.")
        if sha(catalog) != expected_sha256:
            raise ValueError("The source catalog hash does not match its binding.")
        try:
            tools = DECODER.decode(catalog.decode("utf-8"))
            if (
                not isinstance(tools, list)
                or not tools
                or not all(client_tool(tool) and tool["name"] != "StructuredOutput"
                           and "cache_control" not in tool for tool in tools)
                or len({tool["name"] for tool in tools}) != len(tools)
            ):
                raise ValueError("The source catalog is not a complete shared tool prefix.")
            fingerprint = sha(wire(tools))
        except (ValueError, UnicodeError, TypeError, RecursionError) as error:
            raise ValueError("The source catalog is invalid.") from error
        object.__setattr__(self, "count", len(tools))
        object.__setattr__(self, "prefix_sha256", fingerprint)
        object.__setattr__(self, "catalog_sha256", expected_sha256)

    @classmethod
    def from_file(cls, path, *, expected_sha256):
        """Read and verify one explicitly selected source artifact."""
        return cls(Path(path).read_bytes(), expected_sha256=expected_sha256)


class RegisteredNativeToolCachePolicy:
    """Apply the measured marker to an immutable, explicitly supplied prefix.

    The caller owns transport and any experiment namespace. This policy adds
    no namespace, model setting, retry, or response transformation.
    """

    def __init__(self, prefix, *, enabled=False):
        if not isinstance(prefix, RegisteredToolPrefix):
            raise TypeError("Supply a verified source catalog registration.")
        if type(enabled) is not bool:
            raise ValueError("The enabled setting must be a boolean.")
        self.enabled = enabled
        self.prefix = prefix

    def apply(self, raw_body, *, method="POST", path="/v1/messages"):
        if not isinstance(raw_body, bytes):
            raise TypeError("The request body must contain bytes.")
        if not self.enabled:
            return Decision(raw_body, False, "disabled")
        if method != "POST" or not isinstance(path, str) or path.split("?", 1)[0] != "/v1/messages":
            return Decision(raw_body, False, "unsupported_endpoint")
        result = mark_registered_prefix(raw_body, count=self.prefix.count,
            prefix_sha256=self.prefix.prefix_sha256, enabled=True, method=method, path=path)
        if result.applied and not content_layout_eligible(raw_body):
            return Decision(raw_body, False, "unqualified_cache_layout")
        return result
