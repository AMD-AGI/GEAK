---
key: native (.cu/.hip/.cpp/CK) apply-back · any gfx · sglang/vllm
type: method
confidence: ★★☆
effect: lets an e2e win on a COMPILED kernel actually deploy (Python overlay can't shadow a .so)
confirms: 0
last_seen: 2026-06-29
---
# Deploy a COMPILED-source kernel win into the live server (in-place recompile, reversible)

- problem: the PYTHONPATH overlay (`overlay_setup.py add-module`/`add-rebind`) can shadow a `.py` but NOT
  a compiled kernel — the change only takes effect after the package's `.so`/code-object is RECOMPILED in
  place. So a `.cu/.hip/.cpp/CK` winner that passes the isolated oracle is otherwise rejected
  `no_rebind_seam`/`editable=false` and never reaches e2e. HL's `apply_kernel_patch` handles this but with
  hardcoded framework paths + whole-package `pip install -e .`; we do it general + incremental instead.
- detection: a winner is NATIVE if `KERNEL_RESULT.apply_kind=="native"` OR the `code_patch`/`final_patch`
  touches a compiled suffix `{.cu .cuh .hip .cpp .cc .cxx .c .h .hpp}`. kernel_extractor reports
  `editable=true, apply_kind:"native"` for an op that resolves to a rebuildable native source (source +
  a discoverable build seam shipped in the install); `editable=false` only for opaque prebuilt libs / read-only.
- apply: framework-agnostic plumbing lives in `scripts/overlay_setup.py` (`add-native` / `verify-native` /
  `revert` / `gc-stale`); it NEVER invents a build command — the integrator DISCOVERS the install's own
  incremental build (the same way benchmark_engineer does) and passes it via `--build-cmd`/`--invalidate-cache`.
  Discover by walking UP from the source file (marker-driven, no hardcoded abs paths), first match wins:
  · `config.yaml/json compile_command` / `task_runner.py compile` / `Makefile` / `build.sh` → use verbatim.
  · torch `cpp_extension.load(name=…)`/`load_inline` → `--invalidate-cache $TORCH_EXTENSIONS_DIR/<name>`
    (set TORCH_EXTENSIONS_DIR per-eval + local-arch env), rebuilds on next import.
  · aiter cpp_itfs `MD_NAME` driver beside source → `--invalidate-cache <cpp_itfs_root>/<md_name>_<hash>`.
  · ninja `build.ninja` / CMake `build/` up-tree → build only the affected target (incremental).
  · fallback: package editable rebuild — LOG it as COARSE (never silent). No seam at all → `no_build_seam`.
- verify: `verify-native` confirms the built artifact actually changed (catches a silent no-op build →
  `native_build_no_op`). Only measure after FRESH_BUILD_OK.
- A/B order (native mutates the install GLOBALLY, can't toggle per-leg by env): measure REF leg FIRST on the
  clean install (`OVERLAY_PYTHONPATH=$CURRENT_OVERLAY`), THEN `add-native`, THEN CAND leg (native is live in
  the install; prior accepted .py overlays stay active). Mixed .py+native: `add-module` the .py into $CAND too,
  CAND leg uses `OVERLAY_PYTHONPATH=$CAND`.
- reversibility (MANDATORY): wrap the CAND measurement so EVERY exit path (accept/reject/crash/timeout) runs
  `overlay_setup.py revert --overlay $CAND` → install restored byte-exact; then assert `git -C <pkg> status
  --porcelain` empty. Runner calls `gc-stale --root $EVAL_DIR` at session start to clean a crashed run's
  still-applied native overlay. Serialize: never two native A/Bs against one install concurrently (the e2e
  gate leases all GPUs, so this holds by construction).
- the graph-capture-safety + memory-footprint gates apply to native kernels exactly as to authored ones.
- source: Phase 2 of the native apply-back design (DESIGN_native_apply_back.md); plumbing committed as Phase 1.
