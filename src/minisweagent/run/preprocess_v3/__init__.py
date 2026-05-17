"""``preprocess_v3`` — foundation for the redesigned preprocessing pipeline.

This package is intentionally minimal in PR 1; it ships only the deterministic
pre-step-0 building blocks (URL resolve + clone + split, language detection)
and the subagent registry skeleton that PR 3 will populate with YAML bodies.

The existing ``minisweagent.run.preprocess`` package is **not** touched by
this PR; the two packages are decoupled by design so the v3 redesign can land
incrementally without disrupting the in-tree preprocess flow.
"""
