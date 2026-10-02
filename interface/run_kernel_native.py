#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a kernel Workflow through the existing persistent native SDK runner.

The E2E launcher maps a model handoff into E2E arguments. This entrypoint accepts
the public kernel Workflow arguments directly. It uses the same SDK lifecycle,
model, effort, settings, and permissions as the existing native E2E runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .run_e2e import (
        GEAK_ROOT,
        PROCESS_SAFETY,
        _invoke_via_sdk,
        _parse_last_json_line,
    )
except ImportError:
    from run_e2e import (
        GEAK_ROOT,
        PROCESS_SAFETY,
        _invoke_via_sdk,
        _parse_last_json_line,
    )


def invoke_kernel(arguments, timeout_s, *, settings_profile=None, working_directory=None):
    """Pass one explicit kernel request to the native runner."""
    if not isinstance(arguments, dict):
        raise TypeError("The kernel arguments must be a JSON object.")
    if not arguments.get("kernel_path") or not arguments.get("workflow_dir"):
        raise ValueError("The kernel arguments require kernel_path and workflow_dir.")
    if settings_profile not in (None, "isolated"):
        raise ValueError("The native settings profile is unsupported.")
    request = {"scriptPath": str(GEAK_ROOT / "kernel_workflow" / "kernel_workflow.js"), "args": arguments}
    prompt = (PROCESS_SAFETY + "Invoke the Workflow tool exactly once with this JSON object:\n"
        + json.dumps(request) + "\nPass args as a JSON object. Wait for the Workflow task to finish. "
        "Print its full return value as one final line of compact JSON.\n")
    native_options = {"workflow_request": request}
    if settings_profile is not None:
        native_options["settings_profile"] = settings_profile
    if working_directory is not None:
        directory = Path(working_directory).resolve()
        if not directory.is_dir():
            raise ValueError("The native working directory must exist.")
        native_options["native_cwd"] = str(directory)
    raw = _invoke_via_sdk(prompt, timeout_s, **native_options)
    return _parse_last_json_line(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arguments", type=Path, help="Read the kernel Workflow arguments from this JSON file.")
    parser.add_argument("output", type=Path, help="Write the complete Workflow result to this JSON file.")
    parser.add_argument("--timeout", type=int, default=7200, help="Stop the SDK after this many seconds.")
    parser.add_argument("--settings-profile", choices=["isolated"],
        help="Explicitly disable user, project, and local filesystem settings for this native run.")
    parser.add_argument("--cwd", type=Path,
        help="Use this native working directory. Local helpers require a directory outside Git repositories.")
    options = parser.parse_args()
    if options.timeout <= 0:
        parser.error("The timeout must be greater than zero.")
    result = invoke_kernel(json.loads(options.arguments.read_text()), options.timeout,
        settings_profile=options.settings_profile, working_directory=options.cwd)
    options.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
