"""Contract resolution phase — freeze evaluation metadata after Discovery.

Runs **after** ``DiscoveryPhase`` (kernel path, codebase context, ATD
``discovery.json``, ``ctx.language``) and **before** ``HarnessPhase``.

Writes ``{output_dir}/evaluation_contract.json`` and sets
``ctx.evaluation_contract`` so ``ExplorePhase`` can pass
``compile_command`` into per-language ``commandment.j2`` renders.

When deterministic compile inference fails but ``eval_command`` is set,
``ContractNormalizerAgent`` runs **if** ``ctx.model`` / ``model_factory``
yields a model; otherwise the phase finishes with Tier-0 only (no env
flag required).
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
    preprocess_model_display_name,
    resolve_preprocess_phase_model,
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

        if (
            not contract.get("compile_command")
            and isinstance(ctx.eval_command, str)
            and ctx.eval_command.strip()
            and ctx.language is not None
        ):
            model = resolve_preprocess_phase_model(ctx)
            if model is not None:
                try:
                    from minisweagent.subagents.base import SubagentConfig
                    from minisweagent.subagents.preprocess.contract_normalizer import (
                        ContractNormalizerAgent,
                    )

                    cfg = SubagentConfig(
                        name="contract_normalizer",
                        model_name=preprocess_model_display_name(model),
                        system_template="",
                        instance_template="",
                        step_limit=1,
                        cost_limit=1.0,
                        temperature=0.1,
                        extra={"max_rounds": 3},
                    )
                    agent = ContractNormalizerAgent(language=ctx.language, config=cfg)
                    agent.model = model  # type: ignore[attr-defined]
                    agent_out = agent.run(
                        eval_command=ctx.eval_command,
                        discovery_digest=contract.get("discovery_digest"),
                        codebase_excerpt=contract.get("codebase_context_excerpt", ""),
                        kernel_language=contract.get("kernel_language") or "",
                    )
                    if isinstance(agent_out, dict) and agent_out.get("compile_command"):
                        contract["compile_command"] = agent_out["compile_command"]
                        contract["agent_tier_used"] = True
                        contract["agent_attempts"] = agent_out.get("attempts_used", 0)
                except Exception as exc:
                    logger.warning(
                        "[yellow]ContractNormalizerAgent failed (non-fatal): %s[/yellow]",
                        exc,
                        exc_info=True,
                    )
            else:
                logger.debug(
                    "ContractNormalizerAgent skipped: no model on ctx (Tier-0 only)."
                )

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
