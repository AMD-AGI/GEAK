# Log

## 2026-06-22 (remove playbooks)
- **Update** — Removed the vendored `playbooks/` directory (the KernelForge knowledge-base
  snapshot) and all references to it: the root-index Playbooks section, the README structure
  tree / flow / scope notes, the "Related playbooks" sections in the methodology and three
  patterns, the CODEOWNERS rule, and the registry pointer. The KernelForge originals remain in
  place. This bundle now contains only original distilled content (patterns, anti-patterns,
  methodology, cases, catalog).

## 2026-06-22 (standalone)
- **Update** — Made the bundle self-contained for standalone-repo use. Vendored the
  aggregated-speedup data into `catalog/` (`kernel_speedups.md` + both CSVs) and repointed the
  former `../kernel_speedups*` parent references to `/catalog/...`. Stripped absolute machine
  paths (`/wekafs/chushi/agent-kernel-arena/`) and leading-slash external roots from citations
  in 16 authored files → clean repo-relative form. Citations to the source tree are now
  intentionally-dangling (OKF-tolerated). Verified: 0 outside-bundle refs, 0 broken
  bundle-relative links, conformance ALL PASS (182 concept docs). README gained a "Standalone
  use & citations" note.

## 2026-06-22 (catalog)
- **Update** — Added `catalog/kernel-registry.md` (type: Reference): all 91 referenced
  LLM-inference kernels grouped by operator domain, best measured speedup, cross-linked to
  the 18 deep case studies (22 links). Copied `kernel_speedups_llm_inference.csv` into
  `catalog/` as the backing data asset so the bundle is self-contained. Linked from root index.

## 2026-06-22 (later)
- **Update** — Integrated the KernelForge knowledge base into `playbooks/`. Copied a
  snapshot of `KernelForge/knowledge_base/` (144 `.md` + ISA/ASM/JSON assets) and made it
  OKF-conformant: added `type` frontmatter to all markdown (Reference for ISA-instruction
  and API/catalog docs, Playbook for guides/playbooks, Anti-Pattern for `pitfalls.md`).
  The **KernelForge originals were left untouched** — its tooling (`knowledge/loader.py`,
  config, prompts) reads them by path; this is a deliberate copy, not a move. Added
  `playbooks/index.md` and cross-links from the methodology and key patterns. Bundle now
  179 concept docs.

## 2026-06-22
- **Creation** — Initial OKF bundle. Patterns, anti-patterns, methodology, and the
  first wave of kernel case studies distilled from the documented optimization corpus
  (campaign20, KernelForge, spare_kernels). Scope limited to kernels with authored
  technique notes and verified speedup < 10×. High-speedup (>10×) automated geak runs
  deferred — their reports carry no "what changed", so their lessons would have to be
  reverse-engineered from source diffs.
