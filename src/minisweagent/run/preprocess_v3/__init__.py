"""``preprocess_v3`` — redesigned preprocessing pipeline.

The v3 pipeline keeps the proven legacy deterministic discovery front half
(``DiscoveryPhase`` / ATD / ``CODEBASE_CONTEXT.md``) and swaps the old
heterogeneous agent layer for two general preprocess agents:
``harness-generator`` and ``harness-verifier``. Language-specific behavior is
injected through the per-language KBs.
"""
