"""ContractNormalizerAgent — optional LLM helper for ``compile_command`` inference.

When deterministic heuristics in ``contract_normalize.infer_compile_command_from_eval``
fail, this subagent proposes a single JSON object::

    {"compile_command": "<bash fragment>" | null}

The outer ``ContractResolutionPhase`` merges a successful proposal into
``evaluation_contract.json`` whenever a model is available and Tier-0
did not infer ``compile_command``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from minisweagent.pipeline_workers.base import SubagentBase

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class ContractNormalizerAgent(SubagentBase):
    """Narrow subagent: map messy ``eval_command`` to ``compile_command`` JSON."""

    def run(self, **inputs: Any) -> dict[str, Any]:
        eval_command = str(inputs.get("eval_command") or "").strip()
        discovery_digest = inputs.get("discovery_digest")
        codebase_excerpt = str(inputs.get("codebase_excerpt") or "")
        kernel_language = str(inputs.get("kernel_language") or "")

        if not eval_command:
            return {"ok": False, "compile_command": None, "attempts_used": 0, "errors": ["empty eval_command"]}

        cfg_extra = (getattr(self, "config", None) and self.config.extra) or {}
        max_rounds = int(cfg_extra.get("max_rounds", 3))
        max_rounds = max(1, min(max_rounds, 8))

        errors: list[str] = []
        attempts_used = 0
        for attempt in range(1, max_rounds + 1):
            attempts_used = attempt
            sys_p, inst_p = self._compose_prompt(
                eval_command=eval_command,
                discovery_digest=discovery_digest,
                codebase_excerpt=codebase_excerpt,
                kernel_language=kernel_language,
                prior_errors=errors,
                attempt=attempt,
            )
            raw = self._query_model(sys_p, inst_p)
            parsed = self._parse_json_response(raw)
            if parsed is None:
                errors.append(f"attempt {attempt}: could not parse JSON from model output")
                continue
            cc = parsed.get("compile_command")
            if cc is None:
                return {"ok": True, "compile_command": None, "attempts_used": attempts_used, "errors": []}
            if not isinstance(cc, str) or not cc.strip():
                errors.append(f"attempt {attempt}: compile_command empty after strip")
                continue
            cc = cc.strip()
            if len(cc) > 4000:
                errors.append(f"attempt {attempt}: compile_command too long")
                continue
            if not self._looks_sane(cc):
                errors.append(f"attempt {attempt}: compile_command failed sanity check")
                continue
            return {
                "ok": True,
                "compile_command": cc,
                "attempts_used": attempts_used,
                "errors": [],
            }

        return {
            "ok": False,
            "compile_command": None,
            "attempts_used": attempts_used,
            "errors": errors,
        }

    @staticmethod
    def _looks_sane(cc: str) -> bool:
        low = cc.lower()
        if "\x00" in cc:
            return False
        # Require some build-like token to avoid arbitrary eval halves.
        return any(
            tok in low
            for tok in (
                "compile",
                "cmake",
                "ninja",
                "make",
                "hipcc",
                "nvcc",
                "meson",
                "pip install",
            )
        )

    def _compose_prompt(
        self,
        *,
        eval_command: str,
        discovery_digest: object,
        codebase_excerpt: str,
        kernel_language: str,
        prior_errors: list[str],
        attempt: int,
    ) -> tuple[str, str]:
        lang = getattr(self.language, "name", "unknown")
        system = (
            "You are a build-system assistant for GEAK kernel preprocessing. "
            "Given a full eval shell command, respond with ONLY a JSON object "
            'of the form {"compile_command": "<bash>"} or {"compile_command": null} '
            "where the bash fragment is the minimal prefix that compiles/builds "
            "the project (often `make`, `cmake --build`, or `python3 ... compile`). "
            "Do not include correctness, pytest, or benchmark/performance steps. "
            "No markdown fences, no commentary outside JSON."
        )
        parts = [
            f"KernelLanguage (registry): {kernel_language or lang}",
            "",
            "Full eval_command:",
            eval_command,
            "",
            "Discovery digest (may be truncated):",
            json.dumps(discovery_digest, default=str)[:6000],
            "",
            "CODEBASE_CONTEXT excerpt (may be truncated):",
            (codebase_excerpt or "")[:6000],
        ]
        if prior_errors:
            parts += ["", f"Attempt {attempt} — fix these issues:", *(f"- {e}" for e in prior_errors)]
        return system, "\n".join(parts)

    def _parse_json_response(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        stripped = text.strip()
        m = _JSON_FENCE.search(stripped)
        if m:
            stripped = m.group(1).strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Try largest {...} slice
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                data = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    def _query_model(self, sys_prompt: str, inst_prompt: str) -> str:
        model = getattr(self, "model", None)
        if model is None:
            from minisweagent.models import get_model

            model = get_model(self.config.model_name, {})

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": inst_prompt},
        ]
        response = model.query(messages)
        if isinstance(response, dict):
            return str(response.get("content", ""))
        return str(response)


__all__ = ["ContractNormalizerAgent"]
