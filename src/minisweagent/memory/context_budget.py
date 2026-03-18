"""Context Budget Manager.

Enforces a hard token limit on total memory injected into the agent's prompt.
When the combined memory context exceeds the budget, it prioritizes the most
impactful items and truncates the rest.

This addresses the "context overload" problem observed in the production stack
(Abl-14) where combining all memory components produced worse results than
individual experiments.

Priority order (highest first):
1. GPU specs (~100 tokens) -- always included
2. COMMANDMENT status (~50 tokens) -- always included
3. Pitfalls/warnings (~100 tokens) -- critical for avoiding waste
4. Strategy effectiveness (~150 tokens) -- guides optimization
5. Cross-kernel insights (~200 tokens) -- transfer learning
6. ReMe insights (~200 tokens) -- distilled experience
7. Principles (~200 tokens) -- abstract rules
"""

from __future__ import annotations

import os


def get_context_budget() -> int:
    """Get the context budget from environment or default."""
    return int(os.environ.get("GEAK_MEMORY_BUDGET", "800"))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token for English)."""
    return len(text) // 4


def enforce_budget(
    gpu_specs: str = "",
    commandment_status: str = "",
    pitfalls: str = "",
    strategy_effectiveness: str = "",
    cross_kernel: str = "",
    reme_insights: str = "",
    principles: str = "",
    budget: int | None = None,
) -> str:
    """Combine memory blocks within the token budget.

    Prioritizes blocks in order, truncating lower-priority items if budget exceeded.
    """
    if budget is None:
        budget = get_context_budget()

    blocks = [
        ("gpu_specs", gpu_specs),
        ("commandment", commandment_status),
        ("pitfalls", pitfalls),
        ("strategies", strategy_effectiveness),
        ("cross_kernel", cross_kernel),
        ("reme", reme_insights),
        ("principles", principles),
    ]

    result_parts = []
    tokens_used = 0

    for name, block in blocks:
        if not block or not block.strip():
            continue
        block_tokens = estimate_tokens(block)
        if tokens_used + block_tokens <= budget:
            result_parts.append(block)
            tokens_used += block_tokens
        else:
            remaining = budget - tokens_used
            if remaining > 50:
                truncated = block[:remaining * 4]
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    truncated = truncated[:last_newline]
                result_parts.append(truncated + "\n[...truncated due to memory budget]")
                tokens_used = budget
            break

    return "\n".join(result_parts)
