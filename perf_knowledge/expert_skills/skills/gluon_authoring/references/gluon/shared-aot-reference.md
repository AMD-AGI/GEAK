# Gluon AOT, Prebuilt, And Wrapper Contracts (gfx950 / gfx942)

**What this page owns: the packaging boundary** — target triple, launch attributes, and the
signature/artifact contract that a JIT compile does not exercise. Everything this file used to
sketch about shared memory and async copy now has a mechanism owner, and those owners carry the
source-proven signatures and the four-version gates this page never had.

| If you came here for | Read |
| --- | --- |
| `allocate_shared_memory`, descriptor views (`.index` / `.slice` / `.permute` / `._reinterpret`), `.gather` / `.scatter`, LDS capacity and its occupancy divisor | `smem-lds-reference.md` |
| `SwizzledSharedLayout` / `PaddedSharedLayout` / `SharedLinearLayout` — which constructor, and the compile-time `gl.bank_conflicts` check | `layout-reference.md ## Shared Layouts: The Three Constructors` |
| async copy entry points, `commit_group` / `wait_group`, `load_shared_relaxed`, and the per-dtype granularity floor | `memory-reference.md`, then `pipeline/async-ordering.md` for the gfx950 staging structure (`pipeline-reference.md` routes the rest) |
| barriers, waits, and what orders an async fill against the read of it in an authored ring | `pipeline/async-ordering.md ## Ordering the fill against the read — class S`. There is no `fence_async_shared` on this path — it is NVIDIA-only (`appendix-api.md ## Spellings that do not exist on the CDNA Gluon path`) |
| the padding-vs-swizzle bank-conflict recipe | `../tile-programming/layout-recipes.md` |

gfx1250 descriptor / TDM / tensor-memory paths are out of scope here
(`smem-lds-reference.md ## gfx1250`).

## AOT Signature And Prebuilt Notes

When AOT, prebuilt, or wrapper-sensitive paths are involved, check:

- target triple and warp-size assumptions (`hip:gfx950:64` / `hip:gfx942:64`);
- launch attributes such as `num_warps`, `num_ctas`, `waves_per_eu`, and `num_stages` — noting
  that `num_stages` is **inert on the Gluon path on all four versions** (dead in 3.8.0) and survives
  here only as a signature field (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`);
- pointer-type hints or divisibility assumptions encoded in the signature;
- disappearing constexpr values between source and AOT signature;
- scratch behavior and whether wrappers reject nonzero scratch sizes;
- prebuilt and fallback selection logic.

Compile success alone does not prove package/runtime contract validity. Do not delete fallback
gates or prebuilt selection just because a local JIT path compiles. Triton minor-version
sensitivity for AOT metadata lives in `../hardware/capability-matrix.md` /
`../hardware/planning-constants.md`.

**The version-gate trap this page exists to catch.** A kernel that compiles under JIT on your box
and ships as a prebuilt artifact has two version surfaces, not one, and the pack's four-version
tables answer only the first. A symbol that is 3.8.0-only —
`cdna4.compute_efficient_padded_shared_layout`, the defaulted `._reinterpret` signature, the
shared-descriptor `atomic_scatter_*` methods — makes the *source* unbuildable on an older
toolchain rather than the artifact invalid; a changed target triple or launch attribute makes the
*artifact* wrong while the source still builds everywhere. Decide which of the two you are
protecting before reading either table.

**There is a third class, and it is the dangerous one because it is neither of those.**
`loop_unroll_factor` is a **keyword on the loop construct**, not a symbol and not a launch
attribute, and the keyword is accepted by the frontend on every version. What is 3.8.0-only is the
pass that *reads* the attribute it sets. So the older toolchain does not refuse the source and does
not produce a wrong artifact — it produces a **correct artifact that quietly lost a transform**,
with no import error, no compile error and no diagnostic anywhere in the build log. Neither of the
two surfaces above catches it, which is why the toolchain minor belongs in the acceptance checklist
as a recorded fact rather than as something inferred from a successful build
(`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)` for the two directions
the knob is used in and for what to read back off the IR).

## Acceptance Checklist

```text
target family (gfx950 / gfx942) + triple:
launch attributes recorded (and num_stages understood as inert):
signature: pointer hints / divisibility / surviving constexprs:
scratch size and wrapper policy:
prebuilt vs fallback selection path exercised:
toolchain minor the artifact was built against:
loop annotations whose reading pass is version-gated (loop_unroll_factor) recorded, not inferred:
```
