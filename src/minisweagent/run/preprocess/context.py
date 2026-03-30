"""Preprocessing pipeline output contract.

Defines the structured boundary between the preprocessor and downstream
consumers (orchestrator, optimizer).  All fields are paths -- consumers
read files as needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class PreprocessContext:
    """Boundary contract: preprocess -> orchestration/optimization.

    First 4 fields are required (no defaults).  Everything else is
    optional and may be None if the corresponding preprocessing step
    was skipped (e.g. GEAK_HARNESS_ONLY=1 skips profiling).
    """

    # Required -- always set by a successful preprocessing run
    kernel_path: str
    repo_root: str
    harness_path: str
    preprocess_dir: str

    # Paths to artifacts (None if step was skipped)
    commandment_path: str | None = None
    codebase_context_path: str | None = None
    baseline_metrics_path: str | None = None
    profiling_result_path: str | None = None

    # Inline data (kept in memory, not just a path)
    test_command: str | None = None
    discovery: dict | None = None
    harness_results: list[dict] | None = None
    benchmark_baseline: str | None = None
    full_benchmark_baseline: str | None = None
    baseline_metrics: dict | None = None
    commandment: str | None = None
    testcase_selection: dict | None = None

    # Original resolution info
    resolved: dict | None = None

    def validate(self) -> list[str]:
        """Check that required fields are set and paths exist.

        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        for name in ("kernel_path", "repo_root", "harness_path", "preprocess_dir"):
            val = getattr(self, name)
            if not val:
                errors.append(f"Required field '{name}' is empty")
            elif name != "preprocess_dir" and not Path(val).is_file():
                if name == "repo_root":
                    if not Path(val).is_dir():
                        errors.append(f"'{name}' path does not exist: {val}")
                else:
                    errors.append(f"'{name}' file does not exist: {val}")

        if self.preprocess_dir and not Path(self.preprocess_dir).is_dir():
            errors.append(f"'preprocess_dir' does not exist: {self.preprocess_dir}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessContext:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> PreprocessContext:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_preprocessor_output(cls, ctx: dict[str, Any], output_dir: str | Path) -> PreprocessContext:
        """Convert the raw preprocessor output dict into a PreprocessContext."""
        out = Path(output_dir)
        return cls(
            kernel_path=ctx.get("kernel_path", ""),
            repo_root=ctx.get("repo_root", ""),
            harness_path=ctx.get("harness_path", ""),
            preprocess_dir=str(out),
            commandment_path=str(out / "COMMANDMENT.md") if (out / "COMMANDMENT.md").exists() else None,
            codebase_context_path=ctx.get("codebase_context_path"),
            baseline_metrics_path=str(out / "baseline_metrics.json")
            if (out / "baseline_metrics.json").exists()
            else None,
            profiling_result_path=str(out / "profile.json") if (out / "profile.json").exists() else None,
            test_command=ctx.get("test_command"),
            discovery=ctx.get("discovery"),
            harness_results=ctx.get("harness_results"),
            benchmark_baseline=ctx.get("benchmark_baseline"),
            full_benchmark_baseline=ctx.get("full_benchmark_baseline"),
            baseline_metrics=ctx.get("baseline_metrics"),
            commandment=ctx.get("commandment"),
            testcase_selection=ctx.get("testcase_selection"),
            resolved=ctx.get("resolved"),
        )
