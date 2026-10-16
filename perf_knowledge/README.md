# AMD kernel optimization references

Use this knowledge base to find implementation examples, APIs and tuning tools.
Its dated results and status labels are leads, not measured winners for the current
task. Match source, shape, dtype and device; compare correct implementations under
the task's timing boundary. Preserve the best accepted baseline.

| Need | Reference |
|---|---|
| Find candidate implementations | [Capability index](index/capability_index.yaml), [operator cards](operators/), [SOTA matrix](index/sota_matrix.md) |
| Write a kernel or call a library | [Languages](languages/), [backends](backends/) |
| Understand an operator or numerical contract | [Operators](operators/), [quantization](quantization/) |
| Investigate performance | [Profiling](profiling/), [optimization](optimization/), [hardware](hardware/) |
| Follow integration and tuning procedures | [Workflows](workflows/), [expert recipes](expert_skills/) |
| Find ecosystem tools | [Landscape](landscape/), [repository index](reference/repo_index.md), [environment variables](reference/env_vars.md) |

Use the workflow's configured recipe discovery; a reference does not enable an
optional tool or establish that a backend is installed. The [hardware-evidence
skill source](skills/kernel-hardware-evidence/SKILL.md) is prospective guidance.

Operator-card frontmatter feeds the generated registry and matrix. After editing
cards, regenerate from the repository root:

```bash
python3 perf_knowledge/index/_gen_registry.py
```

Do not edit generated catalogs by hand. Preserve canonical backend identifiers,
source revisions and numerical scope. Keep unsupported combinations explicit;
an absent measurement is not evidence that an implementation is unavailable.
For authoring conventions, see [conventions](index/conventions.md),
[taxonomy](index/taxonomy.md) and [templates](index/_templates/).

## Sources

- [Sourcing rules](index/sourcing_rules.md)
- [Build history](index/changelog.md)
