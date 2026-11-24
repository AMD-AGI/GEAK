# Classifying an inline-asm site: mechanism × intent

Part of `../inline-asm-reference.md`. Class names alone are not reproducible (they mix mechanism and
intent); bin on two decidable axes.

## Axis M — mechanism (read only the `asm` and `constraints` strings; first match wins)

**M0** empty asm (constraints are the whole instruction) · **M1** one mnemonic, *ignoring a trailing `v_mov`/`s_mov` that only fills the mandatory output* · **M2** ≥2 real mnemonics, no branch · **M3** protocol block: `s_cbranch*`, numeric label (`1:`/`1b`/`2f`), `${:uid}`, or `s_*saveexec_b64` · **Mx** computed string that did not fold — resolve at the shape you build, then re-bin.
Modifiers: `tie` (digit constraint), `mem` (`~{memory}`), `imm` (`n`), `phys` (`=&{v0}`, `~{s0}`), `gen` (computed constraints), `pure` (`is_pure=True`). **`pure` + `mem` together is a lint error**; `is_pure=True` without `~{memory}` makes no ordering claim at all. A body that writes `scc`/`vcc` without `~{scc}`/`~{vcc}` corrupts a live predicate.

## Axis I — intent

Delete the site, rebind its output to its input: **I-A** values change · **I-B** a race appears · **I-C** same answer, different instructions · **I-D** only schedule/allocation/live ranges move. Apply in order, first hit wins:

1. `s_setreg*`, a memory op with `sc0`/`sc1`/`glc`, any `*_atomic_*`, `buffer_inv`/`buffer_wbl2` → **I-A**
2. issues a memory op **and** waits for that same op (classify by the issue) → **I-C**
3. `s_waitcnt`, a backward branch, or `s_*saveexec_b64` → **I-B**
4. any instruction other than a fill-`mov`, `s_setprio`, `s_nop` → **I-C**
5. **M0 whose tied result replaces an LDS-read value under `~{memory}`** (dependency-carried drain) → **I-B**
6. **M0 register-class pin that changes downstream instruction form** (`"=s,0"` feeding a scalar base, 64-bit tie forming a pair, physical `{vN}`) → **I-C**
7. otherwise (other M0, `s_setprio`, fill-`mov` only) → **I-D**

| | I-A | I-B | I-C | I-D |
| --- | --- | --- | --- | --- |
| **M0** | — | dependency-carried drain (`classes-sync.md`) | register-file / pair pin (`classes-lane-register.md`) | dependency shaping (most M0) |
| **M1** | MODE write (`classes.md`) | drain marker (`classes-sync.md`) | the missing instruction | `s_setprio` |
| **M2** | cache maintenance, atomics | tied drain | fused scalar load, codec chains | — |
| **M3** | protocol block (`classes-sync.md`) | — | — | — |

**Delete-safety prefilter:** a site in an `if` whose `else:` computes the same in plain Gluon is I-C and safe to delete *for correctness* — but the AST form (non-empty `orelse`) also catches MODE writes under unrelated guards and asm/asm arms; confirm with `costs.md ### The one mechanical removal test — prefilter, then confirm`.

## PAIRED-WITH

Issue/wait (linked only by a tie), save/restore, prefetch/finish, release/acquire: **classify the pair, give both halves its intent**; auditing one half alone is wrong. `s_setprio 1…0` is a bracket that may be deliberately unbalanced across a shape branch — do not "fix" it.

## BUDGET-GROUP

Independent sites drawing on one resource (register budget, occupancy step). **Effects are non-additive and non-monotone**: removing the "most important" tie alone can cross the step while removing all lands back under it. A reading a few units over a threshold suggests a group; cross-lane moves appearing where one wide tie is absent are the tell. **A single-site A/B can mislead here**: the all-off arm and ≥1 grouped arm show what one site alone cannot.
