# LLVM co-design stage reference

Use this reference only after the active stage context identifies a compiler-addressable residual.
It describes a current-run decision procedure, not a reusable pass recipe or a result from another
kernel.

## Entry conditions

- The current source, benchmark boundary, profile, and emitted IR/assembly are all identified.
- The active compiler-control path is explicitly authorized by the task and is available in the
  current runtime.
- The allocation/live-set and scheduling/dependency gates in
  [the backend decision reference](../hardware/llvm-backend-invariants.md)
  identify a live opportunity.

If any condition is missing, return to the lower-cost DSL/source stage or record the capability as
unavailable. Do not rebuild a compiler to discover whether a generic lever exists.

**Check the stock 3.8.0 controls before entering this stage.** The per-compile `llvm_fn_attrs`
option (AGPR form, scheduler strategy, VGPR cap) and the stock `coexec` scheduler strategy
(`TRITON_HIP_USE_COEXEC_SCHEDULER`; default-on on gfx1250 only, opt-in on gfx950 / gfx942) need no
build and no sanction; a residual they close never reaches this stage. The capability table and
the gates are `compiler-contract.md ## What upstream 3.8.0 actually gives you`. Fork-only
scheduler / RA / post-assembly env vars are **absent in upstream 3.8.0 and inert if set**
(`non-upstream-reserve.md`) — they are not a cheaper rung of this stage.

How this stage is reached: GEAK's kernel_workflow injects the skill and owns the round loop, so
the sanction comes from the task prompt / tech_lead's `deep_explore` direction, the deep_engineer
does the work in its own loop, and acceptance is GEAK's (`verify_engineer`, Director).

## Discovery and change contract

1. Obtain the compiler stage context from the round's brief (the deep_engineer's DIRECTION and
   inputs); the optional `toolctl` bookkeeping can serve the same stage card.
2. Use the returned tools' `--describe` and `--help` output to discover supported controls, output
   locations, and artifact schema.
3. Keep any build, cache, and source copy isolated from the shared environment.
4. Make one scoped change with a reversible enablement condition and an explicit cache identity.
5. Capture before/after IR or assembly, resource facts, correctness, and same-boundary timing.

Script source is not an API discovery mechanism. A suspected script or documentation defect follows
`source_read_request` and consumes only its resulting `source_excerpt`.

## Decision outcomes

| Evidence | Outcome |
| --- | --- |
| Control is unavailable or does not reach generated code | record a scoped unavailable/inert result; do not infer a hardware wall |
| Generated code changes but current-boundary result does not | record the changed mechanism and retain/revert by the declared objective |
| Live-set or dependency gate fails | return to source/layout/structure work; compiler control is not the next lever |
| Control, generated code, and boundary result agree | retain the change with its artifact references |

Never transplant a compiler configuration, register limit, or scheduler conclusion across a target,
toolchain, source identity, or measurement boundary without rerunning this procedure.

## Out-of-tree pass plugin skeleton

The cheapest executable form of a compiler-side hypothesis: a pass loaded into the **existing**
compiler at an end-of-pipeline extension point. No compiler source is modified and no LLVM is
rebuilt. What it does still require, and the two states that fail quietly, are in
`llir-codesign.md ## The plugin tier` — read that gate first.

The skeleton below is deliberately **policy-free**. It gives you the load path, the extension
point, and a transaction that guarantees a bad transform cannot reach code generation; the analysis
and the transform are yours. Do not copy a cost model, cadence rule, or window size into a new
target — those are per shape and per target (`../hardware/isa-mechanisms.md`), and a transplanted
one is the failure this handbook exists to prevent.

### Probe the three build states before writing anything

What each state means and how it fails is stated once in `llir-codesign.md ## The plugin tier`
(gates 1-3 of its table); these are the commands.

```bash
# 0. Gate 2 -- does the host export symbols for a plugin to bind against?
#    (Host-side build flag; on Triton it is TRITON_EXT_ENABLED -- discover it per project.)
nm -D --defined-only <host_lib> | grep -c ' T .*llvm::' || echo 'no exported LLVM symbols'

# 1. Gate 3 -- which compiler revision was the host built against? (ABI lock; record it
#    next to the built plugin.)
<llvm_dir>/bin/llvm-config --version && cat <host_src>/cmake/llvm-*.{txt,json} 2>/dev/null

# 2. Gate 1 -- does the host keep a target machine when a plugin is loaded? Upstream 3.8.0
#    does NOT, by design. Verify by generated code, not by the absence of an error.
```

### The pass, as a transaction

```cpp
// Self-contained: LLVM headers only, no host-project headers.
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
namespace {

// YOUR analysis + transform. Return true if it changed anything.
// Bail out unchanged on any shape you do not model: a skipped region is a correct
// outcome, a mispriced one is not.
static bool applyTransform(Function &F) { return false; }

struct MyCoDesignPass : PassInfoMixin<MyCoDesignPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (F.isDeclaration())
      return PreservedAnalyses::all();

    // Transaction: snapshot, transform, verify, roll back on failure. This is what makes
    // the tier safe to leave enabled -- the worst case is "no change", never bad code.
    ValueToValueMapTy VMap;
    Function *Backup = CloneFunction(&F, VMap);
    Backup->setName(F.getName() + ".codesign.bak");

    const bool Changed = applyTransform(F);

    if (verifyFunction(F, /*OS=*/nullptr)) {
      auto Linkage = F.getLinkage();
      F.deleteBody();
      F.splice(F.end(), Backup);
      F.setLinkage(Linkage);
      for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
        Backup->getArg(i)->replaceAllUsesWith(F.getArg(i));
      Backup->eraseFromParent();
      return PreservedAnalyses::all();          // rolled back: nothing changed
    }
    Backup->eraseFromParent();
    if (!Changed)
      return PreservedAnalyses::all();
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();              // in-block work only; CFG intact
    return PA;
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyCoDesign", "v0.1", [](PassBuilder &PB) {
            // End of the optimization pipeline: the pass sees near-final IR, i.e. the
            // individual matrix / memory / vector instructions rather than high-level ops.
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) {
                  MPM.addPass(createModuleToFunctionPassAdaptor(MyCoDesignPass()));
                });
            // Also reachable by name from opt-style drivers, for isolated A/B.
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "my-codesign")
                    return false;
                  FPM.addPass(MyCoDesignPass());
                  return true;
                });
          }};
}
```

### Build and load

```bash
# Build against the SAME LLVM the host was built with (step 1 above). Default visibility
# is required on the plugin as well, and the plugin does NOT link LLVM: it resolves LLVM
# symbols from the already-loaded host library at load time.
LLVM=<llvm_dir>
g++ -shared -fPIC -fvisibility=default \
    $("$LLVM/bin/llvm-config" --cxxflags) \
    -o libmycodesign.so MyCoDesignPass.cpp
```

Loading is via the host's plugin-path environment variable — on upstream Triton 3.8.0 the stock
`LLVM_PASS_PLUGIN_PATH=<abs>/libmycodesign.so`, read in `make_llir`'s module optimization so it
reaches the Gluon path (on another host, discover its spelling; do not assume) — plus whatever the
host needs to keep its target machine. Run the A/B cold (`TRITON_ALWAYS_COMPILE=1`): the
Python local-symbol-scope trap fails **only on a cold compile**
(`llir-codesign.md ## The plugin tier`, "Symbol scope, and why it hides").

### Before attributing anything to it

Prove-fires, then literal diff, then positive control — never timing first
(`../hardware/llvm-backend-invariants.md ## Co-design verification`). A pass that loaded and did
nothing, and a pass that never loaded, are indistinguishable from the boundary.

Keep the transform **gated off by default**, its policy environment-selectable and any global knob
scoped per kernel — the practices (and why each matters) are listed once in
`compiler-contract.md ## Scenario B: sanctioned compiler co-design`, together with the full
build / verify / A-B / restore loop and the sanction it needs. Run each patched variant in its own
process.
