# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Submit tool for completing tasks."""

from typing import Any


class Submitted(Exception):
    """Raised when the agent submits final output and completes the task."""

    pass


class SubmitTool:
    """Tool to submit final result and complete the task."""

    def __call__(self, *, summary: str = "", **kwargs) -> dict[str, Any]:
        """Submit final result. Raises Submitted exception to terminate agent."""
        raise Submitted(summary)
