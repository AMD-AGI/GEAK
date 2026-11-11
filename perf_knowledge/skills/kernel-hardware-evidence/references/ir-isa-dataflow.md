# IR, ISA and dataflow

Read the complete callable, reference and relevant producers/consumers. Identify
observable outputs/state, precision, masks, aliases and lifetimes. Sketch values,
owners (lane, wave, workgroup or kernel), reductions and materialized intermediates.
Existing launch boundaries and temporaries may be implementation choices. A wider
fusion needs the real surrounding computation and timing of all replaced work.

Derive the rewrite's equivalence and changed costs, then inspect whether lowering
realized it. Follow source/specialization → available compiler stages → executable
object/symbol → observed launch. Keep exact source/configuration, case and tool
identity. A similar cache file or kernel name does not identify the executed object.
Missing attribution limits the claim without making all available evidence unusable.

Use the backend's actual artifacts: Triton TTIR/TTGIR/LLVM when returned; HIP or
other backend IR/assembly when available; selected library dispatches for wrappers.
Disassembly of a vendor ELF does not recover compiler IR. Check layout conversions,
shuffles, LDS exchange, reloads, synchronization, matrix instructions and spills;
source reshapes or static instruction counts alone do not determine their cost.

For AMD retained code objects, a `.kd` symbol may be a descriptor. Resolve its entry
offset and relocation against the executable instead of stripping the suffix.
Check decoded instructions/addresses, not just LLVM exit status. A zero-size
function leaves its true extent unknown; a section/next-symbol inspection bound
is useful but is not a proven function extent.

Keep compiler resource requests, rounded allocations and runtime values distinct.
Use the target/wave-mode decoder; do not double-count AGPR/VGPR or assume a universal
register granularity. Include dynamic LDS allocation as well as fixed fields.
A zero fixed segment does not establish zero total LDS. Registers, launch count
and static occupancy estimates do not establish achieved residency or its cause.
For occupancy metrics check the installed definition and all-CU versus active-CU
denominator. In the reviewed gfx950/ROCm 7.2 SDK, `HIGH_RES` wave-level counters
accumulate every clock cycle. Check the installed definition and short-kernel
accuracy limits before using them to estimate time-averaged resident waves.

Check all launch dimensions and units, helper stages, scratch accesses and edge
masks. Compare the proposed mechanism with competing changes such as extra work,
communication or spills. A fallback or compiled-away intervention is untested.
Use full-operation correctness and unprofiled timing for acceptance. If the path
reuses state, exercise warmed same-storage input/metadata changes under the real
contract; fresh inputs may bypass the path. Keep reference-repeatability failures
separate from candidate defects and retain the declared tolerance.
