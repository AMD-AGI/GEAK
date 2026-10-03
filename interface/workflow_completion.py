# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion evidence for new workflow and phase invocations."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


def partial_return(result: dict) -> bool:
    """A completed phase subset is not a completed full workflow."""
    if not isinstance(result, dict):
        return False
    phases = result.get("phases_run")
    return result.get("validation_status") == "phase_partial" or (
        isinstance(phases, list)
        and bool(phases)
        and all(isinstance(phase, str) for phase in phases)
        and not {"all", "final"}.intersection(phases)
    )


class PhaseInvocation:
    """Keep prior returns as history, never as completion of newly requested work."""

    FILES = (
        "workflow_return.json",
        "director_e2e_validation.json",
        "runtime_result.json",
    )

    def __init__(self, args: dict):
        self.eval_dir = Path(args["eval_dir"]).absolute()
        self.phases = {
            part.strip()
            for part in str(args.get("phases") or "all").split(",")
            if part.strip()
        }
        self.prior = {name: self._read(name) for name in self.FILES}
        self.history = self.eval_dir / "workflow_invocations" / uuid.uuid4().hex
        self.state_sha256 = hashlib.sha256(
            json.dumps(
                args.get("state"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def _read(self, name: str):
        try:
            with (self.eval_dir / name).open("rb") as stream:
                info = os.fstat(stream.fileno())
                data = stream.read()
        except FileNotFoundError:
            return None
        identity = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        return identity, data

    def preserve(self) -> None:
        """Archive exact prior bytes before the workflow can replace canonical files."""
        self.history.mkdir(parents=True)
        hashes = {}
        for name, previous in self.prior.items():
            if previous is not None:
                data = previous[1]
                with (self.history / name).open("xb") as stream:
                    stream.write(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
        with (self.history / "invocation.json").open("x") as stream:
            json.dump(
                {
                    "eval_dir": str(self.eval_dir),
                    "phases": sorted(self.phases),
                    "carried_state_sha256": self.state_sha256,
                    "prior_sha256": hashes,
                },
                stream,
            )

    def current(self, name: str) -> dict:
        observed = self._read(name)
        if observed is None or observed == self.prior[name]:
            return {}
        try:
            value = json.loads(observed[1])
        except (ValueError, UnicodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def accepts(self, result: dict) -> bool:
        if not isinstance(result, dict):
            return False
        phases = result.get("phases_run")
        full = "all" in self.phases
        matching_phases = (
            isinstance(phases, list)
            and all(isinstance(p, str) for p in phases)
            and set(phases) == self.phases
        )
        # Older full-workflow returns predate phases_run; they remain valid,
        # but a partial phase can never satisfy a full-workflow request.
        if full and phases is None:
            matching_phases = True
        eval_dir = result.get("eval_dir")
        return (
            matching_phases
            and not (full and partial_return(result))
            and isinstance(eval_dir, str)
            and bool(eval_dir.strip())
            and Path(eval_dir).absolute() == self.eval_dir
        )

    def canonical_return(self) -> dict:
        value = self.current("workflow_return.json")
        return value if self.accepts(value) else {}

    def runtime_return(self) -> dict:
        value = self.current("runtime_result.json")
        return value if self.accepts(value) else {}

    def completed_return(self) -> dict:
        return self.runtime_return() or self.canonical_return()

    def validation(self) -> dict:
        return (
            self.current("director_e2e_validation.json")
            if self.phases.intersection({"all", "final"})
            else {}
        )

    def done(self) -> bool:
        return bool(self.completed_return() or self.validation())


def phase_invocation(args: dict) -> PhaseInvocation | None:
    phases = {
        p.strip() for p in str(args.get("phases") or "all").split(",") if p.strip()
    }
    invocation = PhaseInvocation(args)
    if phases and "all" not in phases:
        return invocation
    for name in ("workflow_return.json", "runtime_result.json"):
        previous = invocation.prior[name]
        if previous is not None:
            try:
                value = json.loads(previous[1])
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and partial_return(value):
                return invocation
            if (
                name == "workflow_return.json"
                and isinstance(value, dict)
                and value.get("eval_dir")
            ):
                # A completed canonical return supersedes older runtime files.
                return None
    return None
