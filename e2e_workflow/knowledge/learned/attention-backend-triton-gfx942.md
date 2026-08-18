---
key: full attention (prefill) · gfx942 · sglang hybrid models
type: lever
confidence: ★★★
effect: +~5% e2e (1546.7→~1623 tok/s same-session); also exposes an editable attention surface
confirms: 2
last_seen: 2026-06-09
---
# `--attention-backend triton` — a cheap server-flag win on hybrid models
- lever: switching sglang to the Triton attention backend is a cheap, real e2e win on hybrid-dense
  gfx942 models. Try it FIRST (Config Tuner's job — it's a server flag, no source edit).
- apply: `--attention-backend triton`. Secondary benefit: it moves the full-attention prefill layers
  off the non-editable CK paged kernel onto the **editable** sglang Triton `_fwd_kernel`
  (extend/prefill_attention.py) — a source-editable surface the kernel track can then optimize.
- verify: greedy temp=0 parity holds (benign bf16 tie-break only). Confirm total GPU time stays flat
  (it's a scheduling win, not a GPU-time redistribution) so the order-of-bets is unchanged.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-05 & 06-09
