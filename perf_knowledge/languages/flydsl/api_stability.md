---
title: "FlyDSL — API stability policy and review (stable vs unstable `fx.*`)"
kind: language
gens: [gfx942, gfx950, gfx1250]
updated: 2026-09-23
source_commit: ROCm/FlyDSL@da731e68
sources:
  - ROCm/FlyDSL@da731e68:.claude/skills/api-stability/SKILL.md
  - ROCm/FlyDSL@main:docs/api_stability.md (re-fetched 2026-09-23)
---

> **Reference (compatibility), not a verdict and not a perf lever.** Ingested from the FlyDSL
> api-stability skill plus the policy it treats as source of truth. Nothing here makes a kernel
> faster; it tells you which `fx.*` paths survive a FlyDSL minor-version bump and which are a bet you
> are re-taking every release. Use it to decide what an authored kernel may depend on, and to audit
> one that already exists. Write kernels from [`authoring_tile_programming.md`](authoring_tile_programming.md);
> migrate legacy constructs with [`authoring_api_migration.md`](authoring_api_migration.md);
> optimize via [`authoring_optimization.md`](authoring_optimization.md).

# FlyDSL API stability

## Overview

FlyDSL publishes a compatibility contract over `python/flydsl/`: **stable** APIs keep their call
paths, signatures and semantics across minor releases (`0.3` → `0.4`), and a patch release may not
break one. Everything else is **unstable** and may change or vanish in any minor release without
notice. `deprecated` is not a third level — it is a lifecycle marker on a *stable* API that stays
protected until its removal window closes.

Two questions follow from that contract, and the upstream skill treats them as distinct review modes:

| Mode | Question | Who asks it in GEAK |
|---|---|---|
| **Producer review** | Does this PR / commit / branch break an API that was stable at the base revision? | Only when changing FlyDSL itself |
| **Consumer review** | Does this kernel or module use only stable FlyDSL APIs? | Every authored or migrated aiter/GEAK kernel |

Consumer review is the mode that matters for kernel work. Its output is not "fast/slow" but
"how much of this kernel is a bet on unpublished internals," which is what determines whether a
FlyDSL bump is a no-op or a rewrite.

**The rule of thumb:** an attribute existing and being callable proves nothing. `__all__` is not
access control. Classify the path the caller actually spells, against the policy at the revision
being reviewed.

---

## Version boundary (read before applying anything)

- The policy lives at `docs/api_stability.md` **in the FlyDSL tree**, and it is versioned with the
  source. Read it at the revision under review; do not classify a historic commit with today's
  manifests.
- aiter pins **`flydsl==0.3.2`** (the pin [`authoring_api_migration.md`](authoring_api_migration.md)
  is written against). Every entry in the §3 deprecation table below declares removal at **v0.4**, so
  for aiter kernels those rows are live compatibility debt, not future trivia.
- A sibling FlyDSL checkout can expose newer APIs — and a newer policy — than the pinned version.
  Resolve the imported version and module path before concluding an API is stable.
- The stable set is *declared*, not inferred. It changes when an export manifest changes, which is why
  the catalog is generated per revision rather than memorized.
- The guarantees hold **between released versions only**. Intermediate commits on FlyDSL's main
  branch may adjust semantics during development, so pin a release, not a commit, when a kernel has
  to survive an upgrade.
- The policy itself moves. Between the 2026-09-21 and 2026-09-23 fetches upstream added the
  `experimental` rule, the `flydsl.extension` branch and `_EXTENSION_MODULES`, and renumbered the
  explicit table to §2.4 and the catch-all to §2.5. Re-check section numbers before quoting one.

---

## 1. The classification procedure

Apply these branches by path prefix, in order. A name is stable only if a branch says so.

1. Any path whose module path contains a segment named **`experimental`**, at any depth, is
   **unstable** — regardless of export manifests or explicit listings. Any module-global name or
   namespace segment beginning with `_` is also unstable. On stable types and returned objects, only
   `__dunder__` Python special methods may begin with `_`; every other underscore-prefixed member is
   unstable.
2. A path under `flydsl.expr` — classify **only** under §2.1 below; unstable if it does not qualify.
3. A path under `flydsl.compiler` — direct members of `compiler.__all__` and members of
   `compiler.protocol.__all__` are stable. Other deep paths are stable only if listed in §2.4.
4. A path under `flydsl.extension` — classify **only** under §2.3; unstable if it does not qualify.
5. Every other path is stable only if listed exactly in §2.4.

For a member reached through a returned object, classify its producing API under these branches
first, then apply the returned-object rule below.

### Returned-object rule

An object returned by a stable API has a stable **public result interface**: its non-underscore
members and Python special methods are stable *when reached through that result*, recursively.

What this does **not** promise: the concrete implementation class, its constructor, or its import
path. FlyDSL may swap the class as long as the public members, signatures and semantics survive.

```python
flyc.from_torch_tensor(x).mark_shape_dynamic(0)   # stable — public member of a stable result
flyc.from_torch_tensor(x)._ensure_spec()          # unstable — underscore member
```

For a fluent chain `factory(...).member`, classify the factory first, then apply this rule. Never
classify by Python object identity.

## 2. Declared stable surface

### 2.1 `flydsl.expr` — three export chains

`expr/__init__.py` is the sole top-level aggregation manifest and has no `__all__` of its own. All
three chains below are subject to branch 1 (`experimental` / underscore).

**Direct-child module exports.** `<name>` in direct-child module `<module>` is stable if and only if
`expr/__init__.py` aggregates `<module>` via `from . import *`, **and** `<name>` is in that child's
`__all__`. All three access forms then have identical stability:

```python
fx.Int32              # stable
fx.numeric.Int32      # stable — same commitment
from flydsl.expr.numeric import Int32   # stable — same commitment
fx.arith._to_raw      # unstable under branch 1, even if a historical __all__ listed it
```

**Backend entry points and recursive child namespaces.** `fx.<backend>...<name>` is stable if and
only if: the first-level `<backend>` appears in `_BACKEND_MODULES` in `expr/__init__.py`; every
following child namespace appears in the `__all__` of its direct parent package; and the final
`<name>` appears in the `__all__` of its owning module or package. Intermediate namespaces that
satisfy the second condition are themselves stable namespaces. The rule recurses with no depth limit.

```python
fx.rocdl.cdna3.s_waitcnt   # stable — complete export chain (upstream's worked example)
fx.rocdl.cluster.anything  # unstable when `cluster` is absent from rocdl.__all__
```

An upstream-MLIR ODS builder that is re-exported but omitted from the final `__all__` is unstable.

**Extension aliases.** Entries in `_EXTENSION_MODULES` expose extension libraries as `fx.<alias>`.
The alias and its descendants follow §2.3 and have exactly the stability of the corresponding
canonical `flydsl.extension` path.

### 2.2 `flydsl.compiler`

Only **direct** members of `flydsl.compiler.__all__` are stable (e.g. `flydsl.compiler.jit` when
`jit` is in that manifest). The one exception is `flydsl.compiler.protocol`: every non-underscore
name in its `__all__` is stable, because it is the public extension namespace for user
implementations of the JIT / DSL-value protocols.

The rule is otherwise **not recursive** — `flydsl.compiler.<module>.<name>` does not become stable
merely because it imports; it needs an explicit §2.4 row.

### 2.3 `flydsl.extension`

Extension libraries use the same recursive export-chain rule as backend namespaces. An extension API
is stable if and only if its entry point is a target of `_EXTENSION_MODULES` in `expr/__init__.py`,
every following child namespace is in its direct parent's `__all__`, and the final name is in the
`__all__` of its owning module or package. `flydsl.extension`, registered entry points and qualifying
intermediate namespaces are stable namespaces; there is no depth limit.

### 2.4 Other explicitly stable APIs

This table is the only exception list; a new commitment requires an explicit row. Branch 1 still
applies to it.

| API | Description |
|---|---|
| `flydsl.runtime.device.get_rocm_arch` | Query the target ROCm architecture |
| `flydsl.runtime.device.is_rdna_arch` | Choose between CDNA and RDNA paths |

> `get_rocm_arch` is the arch-detection entry point [`overview.md`](overview.md) describes aiter
> wrappers gating on (`KERNEL_ASYNC_COPY`, LDS budget). It is stable by explicit row — one of exactly
> two listed paths outside the `expr` / `compiler` / `extension` chains.

### 2.5 Everything else

Every API that does not satisfy §2.1–§2.3 and is not listed in §2.4 is unstable: undeclared
`flydsl.*` submodules, all of `flydsl._mlir.*`, every underscore-prefixed name, and any path with an
`experimental` segment. **Direct invocation of an upstream MLIR dialect op is allowed but unstable**
— its name, arguments and semantics are controlled by upstream MLIR and FlyDSL makes no commitment
for it.

### Generating the catalog

```bash
python3 scripts/list_stable_apis.py
python3 scripts/list_stable_apis.py --format json
python3 scripts/list_stable_apis.py --repo-root <other-tree> --include-deprecated   # release review
```

The script reads export manifests **without importing FlyDSL** and lists canonical paths (extension
libraries under `flydsl.extension`). Know its deliberate blind spots before trusting a diff of it: it
omits equivalent `fx` aliases, excludes §3 deprecated APIs unless `--include-deprecated` is passed,
and does not enumerate public type or result-object members. It detects declared export-chain
changes; it cannot prove signature or semantic compatibility. Run it with the collector from the
revision being reviewed when the policy there differs from today's.

---

## 3. Stable but deprecated (removal declared for v0.4)

These satisfy §2 and remain stable through the §5 window, but new code must not use them. For §2.1
direct-child exports the deprecation applies equally to `fx.<name>`, `fx.<module>.<name>` and the
direct-import form; for extension libraries it applies to both the canonical path and its §2.3 alias.

| API | Replacement | Removal |
|---|---|---|
| `fx.get` | `fx.get_...().unpack()` or `IntTuple[...].unpack()` | v0.4 |
| `fx.index_cast` | `fx.Index(x)` | v0.4 |
| `fx.constant_vector` | `Numeric` / `Vector` member functions | v0.4 |
| `fx.tdm_ops` and everything reached through the alias | `fx.rocdl.tdm_ops` — **itself a target-specific unstable path** | v0.4 |
| `fx.Numeric.maximumf` / `minimumf` | `fx.max(x, y)` / `fx.min(x, y)` | v0.4 |
| `fx.Numeric.shrui` / `addf` | `fx.arith.shrui(x, amount)` / `x + y` inside a `fastmath` context | v0.4 |
| `fx.Numeric.exp2` | `fx.math.exp2(x)` | v0.4 |
| `fx.Numeric.shuffle_xor` | `fx.gpu.shuffle_xor(x, offset, width)` | v0.4 |
| `fx.rocdl.BufferCopyLDS64b` | `fx.rocdl.BufferCopyLDS32b`, or `BufferCopyLDS128b` on gfx950 | v0.4 |

`BufferCopyLDS64b` is a retirement worth reading twice: **no AMD target has an 8-byte LDS DMA
instruction**, so that entry point could never emit a working copy. It now raises instead of silently
failing instruction selection.

---

## 4. What counts as a breaking change

For an API that was stable at the base revision, each of these breaks it:

- removing it, or breaking its export chain — dropping an entry from a direct-child `__all__`, from
  `compiler.__all__` or `compiler.protocol.__all__`; removing a backend or extension entry point from
  `_BACKEND_MODULES` / `_EXTENSION_MODULES`, or any `__all__` entry in their recursive chains; or
  removing a §2.4 row without a new rule covering it;
- removing an argument, renaming one that may be passed by keyword, reordering positionals, removing
  a default, or changing a default;
- narrowing accepted types, architectures or value ranges;
- changing a returned scalar/value type, tuple arity, a returned object's stable public result
  interface (non-private members **and** `__dunder__` methods), or the numerical, layout or
  emitted-op semantics for previously valid input.

An implementation refactor that changes observable behavior is breaking **even with an unchanged
signature**. Conversely, these are *not* breaking, and mislabeling them is its own review error:

- adding an API, or an optional keyword whose default preserves existing behavior;
- widening accepted input;
- improving an error message, turning undefined behavior into a clear error, or changing the
  exception type on failure;
- swapping a returned object's concrete class while preserving its public result interface;
- changing only unstable APIs.

## 5. Retiring a stable API

1. Provide the replacement first, normally from an appropriate stable path.
2. Mark the original deprecated at its definition or export declaration, and add both to §3.
3. Retain it in the release where it is marked, `N`, and in `N+1`.
4. Remove no earlier than `N+2`, dropping the §3 row at the same time.

---

## 6. Consumer review: auditing a kernel or module

Audit the requested file or directory only; do not expand to unrelated callers.

### Inventory, then resolve

Find imports first, then trace aliases to actual attribute accesses and calls:

```bash
rg -n --glob '*.py' '^\s*(from|import)\s+(flydsl|mlir)(\.|$)|flydsl\._mlir' <scope>
```

Resolve the spelled path, not the object:

- `import flydsl.expr as fx` + `fx.foo` → `flydsl.expr.foo`
- `from flydsl.expr import arith as ea` + `ea.addi` → `flydsl.expr.arith.addi`
- `from flydsl.expr.typing import Vector as Vec` + `Vec.method` → `flydsl.expr.typing.Vector.method`
- `from flydsl.compiler import kernel` → `flydsl.compiler.kernel`

Report direct imports that are never used separately from actual calls.

### Statuses

| Status | Meaning |
|---|---|
| **STABLE** | The exact path passes the policy |
| **DEPRECATED** | Still stable for compatibility, disallowed in new code; prevents a stable-only result |
| **UNSTABLE** | The path does not pass, including raw `flydsl._mlir.*` and raw `fly` / `fly_rocdl` bindings |
| **UNRESOLVED** | Static inspection cannot establish the path — *not* evidence of stable usage |
| **UPSTREAM-MLIR** | Direct upstream MLIR dialect op; unstable, and additionally gets its own reminder |
| **PRIVATE-WRITE** | Assignment to an underscore-prefixed attribute of a FlyDSL object; strictly higher severity than UNSTABLE |

A module is **stable-only** only with zero DEPRECATED, UNSTABLE, PRIVATE-WRITE, UPSTREAM-MLIR *or*
UNRESOLVED uses. Dynamic `getattr`, `importlib`, wildcard imports, generated source, or an
unresolvable alias is UNRESOLVED — state the uncertainty rather than defaulting to stable.

Internal code may rely on unstable APIs deliberately. Report that as a consumer dependency; it is not
a producer-side compatibility break.

### Private-field writes rank first

Reading an unstable member is a bet on the next release. *Writing* one mutates FlyDSL internal state,
so it can break FlyDSL's invariants **at the reviewed revision**, not only after an upgrade. Rank
these above every other finding.

Emit one `[PRIVATE-WRITE]` per assignment to an underscore-prefixed attribute of a FlyDSL object —
overwriting a field FlyDSL sets, attaching one it does not define, and the `setattr` / `__dict__`
forms. Locate where FlyDSL assigns, validates and reads the field, then state the consequence plus any
aggravating factor: **bypassed validation** (the public path runs a check the write skips) or
**shared mutable object** (the write leaks across calls, configs or threads). Note the missing public
API, since that is why the workaround exists.

```text
[PRIVATE-WRITE] kernels/example.py:1003: kernel_impl._known_block_size = [...]
Writes a FlyDSL-internal field of a stable object; unstable under §1
(underscore rule). Bypasses _validate_known_block_size() in
compiler/kernel_function.py and mutates a module-level kernel shared
across configs.
```

### Upstream-MLIR reminders are mandatory and separate

Emit a distinct `[UPSTREAM-MLIR]` finding for every direct use of an upstream MLIR dialect op, with
import path, alias, call site and line number. This covers direct imports from `mlir.dialects.*` or
`flydsl._mlir.dialects` for:

```text
arith, builtin, func, gpu, llvm, math, memref, rocdl, scf, vector
```

and direct generated ODS-builder modules such as `flydsl._mlir.dialects._arith_ops_gen`. Typical
hits: `arith.addi`, `scf.ForOp`, `vector.LoadOp`, `llvm.*`.

```text
[UPSTREAM-MLIR] kernels/example.py:42: vector.LoadOp
Direct upstream MLIR builder; allowed, but unstable under
docs/api_stability.md §2.5. FlyDSL does not guarantee its name, signature,
or semantics across releases. Prefer a stable FlyDSL wrapper when one exists.
```

Two boundary cases: `flydsl._mlir.ir` alone is raw unstable MLIR infrastructure but is **not** by
itself an operation-use reminder; and raw `fly` / `fly_rocdl` dialect bindings are FlyDSL-specific
rather than upstream, so they are UNSTABLE raw FlyDSL bindings, **not** UPSTREAM-MLIR. An alias never
suppresses the reminder.

### Report shape

```text
## Stable API usage audit — STABLE-ONLY | NOT STABLE-ONLY | MANUAL FOLLOW-UP
Scope: <file or directory>

### Stable uses
- <path> — <locations>

### Private-field writes
- [PRIVATE-WRITE] <target expression> — <locations>; <what FlyDSL uses the field
  for, plus bypassed validation / shared-object impact>.

### Deprecated or unstable uses
- [DEPRECATED/UNSTABLE] <resolved path> — <locations>; <policy reason>.

### Upstream MLIR operations
- [UPSTREAM-MLIR] <operation> — <locations>; §2.5 reminder.

### Unresolved paths
- <source expression> — <why static resolution was insufficient>.
```

Private-field-write and upstream-MLIR findings get their own sections even when the verdict is
already `NOT STABLE-ONLY`; do not collapse them into a generic unstable list. Private writes come
first.

---

## 7. Producer review: a FlyDSL PR, commit or branch

Only relevant when changing FlyDSL itself. Summarized for completeness.

**Resolve an explicit base and head.** For a PR, take base and head from `gh pr view <PR-or-URL>` and
compare head against the merge base. For a single commit, compare `<commit>^` to `<commit>` — for a
merge commit, ask which parent is the intended baseline. For a branch, compare `HEAD` against its
merge base with the nominated base branch (normally `origin/main`).

```bash
git diff --find-renames --find-copies --unified=80 "$base" "$head" -- \
  python/flydsl docs/api_stability.md
```

Do not limit review to edited function bodies: inspect every touched `__init__.py`, `__all__`,
`_BACKEND_MODULES`, `_EXTENSION_MODULES`, `compiler.protocol`, and §2.4 / §3 table change.

**Diff the declared surface.** Run `scripts/list_stable_apis.py --repo-root <tree>` at both
revisions — prefer disposable detached worktrees so the primary worktree is untouched, check that each
tree's policy matches the collector's rules, and use that revision's collector for an older policy —
and compare the JSON. A path
present at base and absent at head is a candidate **BLOCKER**; a new path is not breaking but creates
a new commitment worth calling out; inspect §3 separately since the catalog excludes deprecated APIs.
The catalog diff is a starting point only: review every changed API that was stable at base,
including a stable class's public and dunder methods.

Then apply the §4 break conditions to each affected API, and for a retirement verify all four §5
requirements. For a release-to-release check rather than a single PR, upstream routes to its
separate `release-api-check` skill for baseline selection and the deprecation-window workflow.

**Validate and report.** Run the catalog in each reviewed worktree at minimum; when the PR changes
Python behavior, run focused tests over the affected public API where the environment permits, and
report commands that could not be run rather than assuming success.

```text
## API-stability producer review — PASS | NEEDS CHANGES | MANUAL FOLLOW-UP
Scope: <PR/commit/base...head>

### Breaking changes
- [BLOCKER] <stable API path> — <base behavior> → <head behavior>; §<policy section>.

### Public-contract additions and deprecations
- [INFO/WARN] <path> — <new commitment, migration state, or retirement issue>.

### Validation
- <catalog/test command> — <result or why it was not run>.
```

Return **PASS** only after reviewing the complete public-surface diff and all affected base-stable
APIs. A clean catalog diff alone is insufficient.

A review reports evidence and an outcome — do not edit, commit, push or change a reviewed PR unless
a fix is separately requested.

---

## 8. Cross-check against the rest of this base

The policy is the authority the migration doc implicitly appeals to, and reading them together
resolves several things neither states alone.

| Guidance elsewhere in this base | What the policy adds |
|---|---|
| [`authoring_api_migration.md`](authoring_api_migration.md)'s entire premise — prefer `fx.*` over raw `arith`/`scf`/`vector`/`llvm`/`memref`/`math`/`rocdl` | Policy-backed: §2.5 makes every one of those an **allowed but unstable** UPSTREAM-MLIR use. The migration doc's "keep the raw boundary local" is how you bound the unstable surface |
| Migration §1: `arith.index_cast(T.index, v)` → `fx.Index(v)` | Exact agreement with §3 — `fx.index_cast` is deprecated with replacement `fx.Index(x)`, **removal v0.4**. Not merely stylistic |
| Migration §3: `arith.maximumf/minimumf` → `fx.max`/`fx.min` | Same row in §3 (`fx.Numeric.maximumf`/`minimumf`), same v0.4 removal |
| Migration §1: "an explicit `arith.*FOp` is still warranted for non-default fastmath" | §3 gives a stable alternative — `x + y` **inside a `fastmath` context** replaces `fx.Numeric.addf`. Check that context before keeping a raw `*FOp` for fastmath alone |
| Migration §1: `arith.unwrap(v)` / `arith._to_raw(v)` → `v.ir_value()` | §2.1 names `fx.arith._to_raw` as unstable under branch 1 **even though a historical `__all__` listed it**. `.ir_value()` is non-underscore, so it inherits the returned-object rule — but what it returns is raw MLIR, i.e. unstable from there on |
| Migration §2: gfx1250 TDM via `fx.rocdl.make_tdm_atom` | §3 deprecates `fx.tdm_ops` toward `fx.rocdl.tdm_ops` and states the replacement is **itself a target-specific unstable path**. This migration moves off deprecated-stable *onto* unstable — record it as a known boundary, not a fix |
| Migration §3c: `fx.rocdl.s_waitcnt(vmcnt=/lgkmcnt=/expcnt=)` from `expr/rocdl/universal.py` | The policy's worked stable example is `fx.rocdl.cdna3.s_waitcnt`. The arch-dispatched top-level form is stable only if `s_waitcnt` is in `rocdl.__all__` at the pinned revision — **verify, do not assume**, since the chain rule is per-namespace |
| Migration §2 / §7: copy atoms such as `fx.rocdl.BufferCopy128b` | §3 retired `fx.rocdl.BufferCopyLDS64b` outright (it now raises — no AMD target has an 8-byte LDS DMA). When picking an LDS copy atom, check §3 first |
| Migration §4 and [`overview.md`](overview.md): `SmemAllocator` / `SmemPtr` from `flydsl/utils/smem_allocator.py` | An undeclared `flydsl.*` submodule → **unstable under §2.5**. The legacy dominance workaround the migration doc records, `ptr._view_cache = None`, is a textbook **[PRIVATE-WRITE]**: it mutates FlyDSL-internal state on a shared object. `SharedAllocator` removes both problems, which is a compatibility argument for §4 independent of the ergonomic one |
| [`deep.md`](deep.md) / [`overview.md`](overview.md) on `flydsl/_mlir/` and the ROCDL surface | `flydsl._mlir.*` is unstable wholesale. Kernels reaching in are taking a per-release bet — legitimate, but it should appear in an audit rather than pass silently |
| [`overview.md`](overview.md) on `get_rocm_arch()` arch gating | Stable by explicit §2.4 row, alongside `is_rdna_arch`. These two are the whole stable surface outside the `expr` / `compiler` / `extension` chains |

A practical consequence for GEAK: the aiter FlyDSL kernels in `aiter/ops/flydsl/` are **not**
stable-only, and are not expected to be. The useful deliverable from a consumer review is a ranked
list — private writes, then deprecated-with-a-declared-removal, then unstable and upstream-MLIR — so
a FlyDSL bump has a known blast radius before it is attempted.

---

## Quick reference

| Path form | Verdict |
|---|---|
| `fx.<name>` / `fx.<module>.<name>` where `<module>` is a `from . import *` child and `<name>` is in its `__all__` | stable (all access forms equally) |
| `fx.<backend>...<name>` with a complete `_BACKEND_MODULES` + `__all__` chain | stable |
| `flydsl.extension...` / `fx.<ext-alias>...` with a complete `_EXTENSION_MODULES` + `__all__` chain | stable (alias = canonical path) |
| any path with an `experimental` module segment | unstable, whatever the manifests say |
| `flydsl.compiler.<name>` in `compiler.__all__`; `flydsl.compiler.protocol.<name>` in its `__all__` | stable |
| `flydsl.compiler.<module>.<name>` not listed in §2.4 | unstable (rule is not recursive) |
| `flydsl.runtime.device.get_rocm_arch` / `is_rdna_arch` | stable (explicit §2.4 rows) |
| Public member of an object returned by a stable API | stable via returned-object rule (class/ctor/import path are not) |
| Anything with a leading `_` (except `__dunder__` on stable types) | unstable |
| `flydsl._mlir.*`, undeclared `flydsl.*` submodules | unstable |
| `mlir.dialects.*` / `flydsl._mlir.dialects.*` op calls | allowed, unstable, **UPSTREAM-MLIR** reminder required |
| raw `fly` / `fly_rocdl` dialect bindings | unstable raw FlyDSL bindings — **not** UPSTREAM-MLIR |
| write to `obj._field` on a FlyDSL object | **PRIVATE-WRITE** — highest severity, ranked first |
| importable and callable, but absent from `__all__` | unstable (`__all__` is not access control) |

## Sources
- ROCm/FlyDSL@da731e68:.claude/skills/api-stability/SKILL.md — origin of the two review modes,
  statuses, PRIVATE-WRITE / UPSTREAM-MLIR reminders, and both report shapes (ingested as reference).
- ROCm/FlyDSL@main:docs/api_stability.md (first fetched 2026-09-21, re-synced 2026-09-23) — the
  policy the skill treats as sole source of truth: §1 levels and returned-object rule, §2
  classification branches and export chains, §2.3 `flydsl.extension`, §2.4 explicit stable table,
  §2.5 catch-all and upstream-MLIR clause, §3 deprecation table, §4 break conditions, §5 retirement
  window. Ingested inline so the §-references above resolve without the FlyDSL tree. Placeholder
  arguments in the §3 `fx.get` row are elided (`...`); consult upstream for the exact spelling.
- ROCm/FlyDSL scripts/list_stable_apis.py — the per-revision catalog generator (`--repo-root`,
  `--include-deprecated`) and its documented blind spots (`fx` aliases, §3 deprecated APIs by
  default, result-object members).
- Version pin context: ROCm/aiter@b0ced008:requirements.txt — `flydsl==0.3.2`, which is why every
  v0.4 removal in §3 is live debt for aiter kernels.
- Cross-refs: [`authoring_api_migration.md`](authoring_api_migration.md) (the legacy→`fx.*` mapping
  this policy authorizes) · [`authoring_tile_programming.md`](authoring_tile_programming.md) (writing
  new kernels) · [`deep.md`](deep.md) (FLIR / ROCDL surface) · [`overview.md`](overview.md)
  (`get_rocm_arch` gating, `flydsl/` package layout) · [`debugging.md`](debugging.md)
