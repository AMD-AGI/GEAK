---
key: method (apply-back overlays) · any model · source-rebuild overlays that STACK
type: method
confidence: ★★★
effect: turned two dead legs (server would not start, misleading error) into two clean non-overlapping wins
confirms: 1
last_seen: 2026-08-30
---
# Two overlays owning ONE method: inspect.getsource() cannot read a rebuilt function
- symptom: the second overlay to rebuild the same method dies with
  `OSError('could not get source code')`. `inspect.getsource()` reads the file named by the code
  object, so anything previously compiled from a string has no source to find.
- THE TRAP: sglang's model registry SWALLOWS that import failure and falls back to the generic HF
  path, so what you actually see in the log is a completely unrelated error from a different code
  path — here `DeepseekV3ForCausalLM has no SGLang implementation, falling back to Transformers
  implementation` followed by `ImportError: cannot import name 'is_torch_fx_available'`. Chasing the
  transformers ImportError is chasing a decoy. Grep the server log for your own `[overlay-ERROR]`
  line first; it is printed BEFORE the fallback and names the real cause.
- fix: have the rebuild helper stash the generated text on the new function (`fn.__ovl_src__`) and
  prefer it over `inspect.getsource()`. Rebuilds then CHAIN — each overlay transforms the previous
  one's output. Keep the FIRST `__ovl_orig` (do not let the second rebuild overwrite it) so the
  overlay stays reversible. Inject helpers into the LIVE module globals so a later recompile can still
  resolve an earlier overlay's `_OVL_*` name.
- second-order trap: do NOT `textwrap.dedent` the stashed source. `inspect.getsource()` returns a
  method body indented inside its class, and anchors in other overlays are written against that
  indentation; dedenting shifts everything by one level and the next overlay reports a missing anchor.
  Dedent only for `compile()`.
- verify: smoke-test the REAL STACK, not one overlay at a time. A per-overlay smoke test that reverts
  between overlays passes happily and proves nothing about composition — that is exactly how this
  shipped to a GPU leg. Arm the overlays in leg order against the installed library with no server and
  no weights, then assert every `_OVL_*` injection is present in the final source AND resolvable in
  module globals.
- source: 2026-08-29 (applyback/smoke_stack.py,
  fusion/fusion_overlays/dsr1/_ovl_util.py; 05_FUSION_APPLYBACK.md "失败与修复")
