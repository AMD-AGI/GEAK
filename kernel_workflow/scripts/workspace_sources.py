#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relocate workspace-owned links and reject measurements using ancestor sources.

External vendor/reference links remain shared. This is a source-link guard, not a
compiler tracer: generated builders must also validate their actual input with
``check-input`` immediately before compilation (or ``bind`` for owned overlays).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

STATE = Path(".geak/workspace.json")
INVALID = Path(".geak/invalid_measurements.jsonl")
# These are deliberately frozen by Setup, not candidate implementations.
FROZEN = {"reference_io.pt", "baseline_src", "baseline_ref", "baseline_overlay"}


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def lexical(path: Path) -> Path:
    """Normalize '..' without following links (including dangling source links)."""
    return Path(os.path.abspath(path))


def links(root: Path):
    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        dirs[:] = [d for d in dirs if d not in {".git", ".geak", "__pycache__"}]
        if base == root:
            dirs[:] = [d for d in dirs if d not in FROZEN]
            files = [f for f in files if f not in FROZEN]
        for name in dirs + files:
            path = base / name
            if path.is_symlink():
                yield path


def read_state(root: Path):
    path = root / STATE
    if not path.exists():
        return {"ancestors": []}
    state = json.loads(path.read_text())
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("ancestors"), list)
        or not all(
            isinstance(p, str) and Path(p).is_absolute() for p in state["ancestors"]
        )
    ):
        raise ValueError(f"invalid workspace source state: {path}")
    return state


def relocate(src: Path, dst: Path):
    ancestors = sorted(
        {str(src), str(src.resolve()), *read_state(src)["ancestors"]},
        key=len,
        reverse=True,
    )
    # Inspect only the physical destination tree, never mutate a shared vendor tree.
    for link in links(dst):
        raw = Path(os.readlink(link))
        # Relative links are interpreted where they lived BEFORE copying.
        target = lexical(
            raw if raw.is_absolute() else src / link.relative_to(dst).parent / raw
        )
        for ancestor in map(Path, ancestors):
            if within(target, ancestor):
                local = dst / target.relative_to(ancestor)
                link.unlink()
                link.symlink_to(os.path.relpath(local, link.parent))
                break
        else:
            if not raw.is_absolute():
                # A relative external dependency must still reach the same tree
                # when DST is at a different depth than SRC.
                link.unlink()
                link.symlink_to(target)
    (dst / STATE).parent.mkdir(exist_ok=True)
    (dst / STATE).write_text(json.dumps({"ancestors": ancestors}) + "\n")
    # Keep provenance and invalid-measurement history out of candidate patches.
    ignore = dst / ".gitignore"
    content = ignore.read_text() if ignore.exists() else ""
    if "/.geak/" not in content.splitlines():
        if ignore.is_symlink():
            ignore.unlink()
        ignore.write_text(content.rstrip("\n") + "\n/.geak/\n")
    check(dst)


def check(workspace: Path):
    ancestors = [Path(p) for p in read_state(workspace)["ancestors"]]
    if not ancestors:
        return
    for link in links(workspace):
        raw = Path(os.readlink(link))
        target = lexical(raw if raw.is_absolute() else link.parent / raw)
        resolved = link.resolve()
        for path in (target, resolved):
            # A child may itself live under SRC; local paths take precedence.
            if within(path, workspace):
                continue
            if any(within(path, ancestor) for ancestor in ancestors):
                raise ValueError(
                    f"{link} points outside candidate workspace to ancestor source {path}"
                )


def check_input(workspace: Path, source: Path, compiler_input: Path):
    expected = (workspace / source).resolve(strict=True)
    actual = (workspace / compiler_input).resolve(strict=True)
    if (
        not expected.is_file()
        or not within(expected, workspace)
        or set(expected.relative_to(workspace).parts[:1]) & FROZEN
    ):
        raise ValueError(
            f"candidate source must be editable and inside {workspace}: {expected}"
        )
    if actual != expected:
        raise ValueError(
            f"compiler input {compiler_input} resolves to {actual}; expected candidate {expected}"
        )
    return {
        "candidate": str(expected),
        "compiler_input": str(compiler_input),
        "resolved_input": str(actual),
        "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
    }


def bind(workspace: Path, source: Path, link: Path):
    expected = (workspace / source).resolve(strict=True)
    # Validate the candidate before replacing anything. The link's parent must be
    # local too: an overlay directory can itself be a shared vendor symlink.
    check_input(workspace, source, source)
    target = (workspace / link).parent.resolve() / link.name
    parent = target.parent
    if (
        not within(target, workspace)
        or set(target.relative_to(workspace).parts[:1]) & FROZEN
    ):
        raise ValueError(f"overlay link must be owned by {workspace}: {target}")
    if lexical(target) == expected:
        raise ValueError(f"overlay link would replace candidate source: {target}")
    parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()  # Refresh even existing/dangling links; never skip them.
    elif target.exists():
        raise ValueError(f"refusing to replace non-symlink overlay entry: {target}")
    target.symlink_to(os.path.relpath(expected, parent))
    return check_input(workspace, source, link)


def record_invalid(workspace: Path, reason: str):
    path = workspace / INVALID
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "status": "invalid_measurement",
                    "time": time.time(),
                    "workspace": str(workspace),
                    "reason": reason,
                }
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    copy = sub.add_parser("relocate")
    copy.add_argument("--src", type=Path, required=True)
    copy.add_argument("--dst", type=Path, required=True)
    for name in ("check", "check-input", "bind"):
        command = sub.add_parser(name)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        if name != "check":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument(
                "--input" if name == "check-input" else "--link",
                type=Path,
                required=True,
            )
    args = parser.parse_args()
    workspace = (args.dst if args.command == "relocate" else args.workspace).resolve()
    try:
        if args.command == "relocate":
            relocate(lexical(args.src), workspace)
        elif args.command == "check":
            check(workspace)
        elif args.command == "check-input":
            print(json.dumps(check_input(workspace, args.source, args.input)))
        else:
            print(json.dumps(bind(workspace, args.source, args.link)))
    except (OSError, ValueError, RuntimeError) as error:
        reason = str(error)
        print(
            f"GEAK_SOURCE_INVALID: {reason}. Discard this measurement, preserve the candidate, and repair the build.",
            file=sys.stderr,
        )
        record_invalid(workspace, reason)
        return 86
    return 0


if __name__ == "__main__":
    sys.exit(main())
