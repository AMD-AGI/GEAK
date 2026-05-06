# Fallback translation hints (generic)

Appended to the TranslationAgent prompt when no pair-specific hint
pack (``<src>_to_<tgt>.md``) exists. Generic guidance that applies
regardless of the specific source / target pair.

## Preserve the contract

- The translated kernel MUST expose the same entry-point name.
- The function signature (argument names, argument order, tensor
  shapes, dtypes, strides) MUST be identical.
- Output tensors MUST match shape-for-shape and dtype-for-dtype.
- Correctness is verified by tensor ``allclose`` against golden
  outputs captured from the source kernel's harness.

## Do not introduce new runtime dependencies

- Target language's standard idioms only.
- No new Python packages, no new system libraries, no new compilers
  beyond what the standard harness already uses.

## Preserve semantics, not lines

- A literal line-by-line translation is almost never correct; each
  language has different primitives. Express the same algorithm in
  the target language's native idioms.
- Reductions, broadcasts, and indexing schemes often rewrite
  differently but compute the same result.

## Use the previous-attempt feedback

If the previous attempt failed verification, the feedback block in
the prompt tells you WHICH tensor element(s) mismatched and by how
much. Focus your next attempt on the specific operation that
produced the wrong values, not on sweeping changes.
