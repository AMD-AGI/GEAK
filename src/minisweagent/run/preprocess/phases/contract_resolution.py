"""Contract resolution phase — freeze evaluation metadata after Discovery.

Runs **after** ``DiscoveryPhase`` (kernel path, codebase context, ATD
``discovery.json``, ``ctx.language``) and **before** ``HarnessPhase``.

Writes ``{output_dir}/evaluation_contract.json`` and sets
``ctx.evaluation_contract`` so ``ExplorePhase`` can pass
``compile_command`` into per-language ``commandment.j2`` renders.

This additive slice deliberately uses only deterministic inference. The
optional LLM normalizer from `refactor-test` depends on internal phase-worker
classes that are being renamed before they are ported.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.contract_normalize import build_evaluation_contract
from minisweagent.run.preprocess.phases.base import (
    Phase,
    PhaseContext,
)

logger = logging.getLogger(__name__)


class ContractResolutionPhase(Phase):
    name = "contract_resolution"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if not ctx.kernel_path:
            logger.warning(
                "ContractResolutionPhase: no kernel_path; skipping contract freeze."
            )
            ctx.phases_skipped.append((self.name, "no kernel_path"))
            return

        contract: dict[str, Any] = build_evaluation_contract(ctx)

        out_path = Path(ctx.output_dir) / "evaluation_contract.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
        ctx.evaluation_contract = contract
        logger.info(
            "  evaluation_contract written (%s, compile_command=%s)",
            out_path.name,
            "set" if contract.get("compile_command") else "none",
        )
        ctx.phases_run.append(self.name)


__all__ = ["ContractResolutionPhase"]
