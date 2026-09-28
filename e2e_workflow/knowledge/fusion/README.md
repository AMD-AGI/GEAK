# Fusion strategy priors — provider-agnostic known fusions

Two layers decide "what can be fused", in this order:

1. **Scan (ground truth).** `scripts/fusion_catalog.py` enumerates the fusion
   kernels ACTUALLY installed in this environment across every known provider
   (aiter / sglang / vllm / flashinfer / …, auto-detected). That is the
   authority for what exists **here**.
2. **Prior (fills the gap).** This directory is the provider-agnostic knowledge
   of **which op-combinations are known to be fusable at all** — independent of
   whether a kernel for them is installed in this build. When the scan finds no
   installed kernel for a fusible region, the prior answers "is this a known
   fusion, and which library realizes it?" so the region becomes a *referenced*
   candidate (port / author with a known target) instead of a blind
   "no kernel → author-track".

## Contract (mirrors `learned/README.md`)

- Priors only **ADD**: they never prune a candidate the scan found, and never
  substitute for the on-box bake-off + e2e gate. A strategy that turns out not
  to apply is dispositioned with a reason, never silently dropped.
- A strategy names an **op-set** (tags from the catalog's `op_tag_vocab`), a
  dtype family, and the kernels known to realize it **per provider**. Matching is
  op-tag-set containment + dtype-compat — the same test the catalog uses, so a
  region matches a strategy exactly when it would match that provider's kernel.
- `confidence`: `verified` (measured e2e win on record — cite the `learned/`
  card) · `known` (kernel exists / documented, benefit unmeasured) ·
  `hypothesis` (plausible fusion, no known kernel — pure author-track lead).
- The whole point is **honesty about the boundary**: `fusion_catalog.json`
  declares which providers were scanned; a strategy whose `known_kernels` are all
  in a provider NOT in `providers_scanned` is a real "installed-here blind spot"
  the prior is covering for. Say so.

## Files

- `fusion_strategies.json` — the machine-readable priors (read by
  `fusion_candidate_harness.py --fusion-priors`).
- Add a strategy when a new fusible op-combination is learned; cite the kernel
  and, if measured, the `learned/` card.
