# Kernel hardware evidence

This shared method guide complements the role's existing experiment loop. Use the
supplied commands, resource locks, source boundaries, correctness contract and
budgets. The planner delegates execution; engineers stay within their assigned
edit authority. Apply the relevant reasoning when a decision depends on it, inside
the existing notes and output fields. There is no extra report schema or required
profiler sequence.

## Start from the complete measured computation

Read the callable, numerical reference and supplied producer/consumer context.
Account for observable outputs and state, masks, strides, precision, aliases and
lifetimes. Draw a small stage map: required values, implementation temporaries,
reductions, communication and materialized boundaries. Track ownership by thread,
wave, workgroup or kernel, and where a value moves when its consumer changes owner.
Separate mathematical requirements from choices in the current implementation.

Use that account to consider an equivalent organization as well as local tuning.
An intermediate may be avoidable, a reduction reorganizable, or a value reusable.
Derive numerical and synchronization obligations before changing ownership or
fusion boundaries. Check whether the proposed organization is already present.
Keep all replaced work inside the admitted measurement boundary; absent neighboring
computation is unknown, and this guide does not expand a specialist's focus files.

## Choose evidence that answers the current question

| Question | Inspect when available | What the observation can distinguish |
| --- | --- | --- |
| Which stage and implementation cost time? | Existing per-case results, matched launch trace, selected symbols and dispatch counts, including helper stages. | Device work versus launches or wrapper cost, with the same timing boundary. |
| Where do values move or get reused? | Source indexing and returned compiler layouts, conversions, loads/stores, reductions and synchronization. | Logical views versus shuffles, LDS exchange, reloads or materialized intermediates. |
| What work does the machine execute? | Bound final ISA and launch geometry; loops, masks, instruction types, scratch accesses and repeated dispatches. | Required arithmetic versus padded/repeated work, or an intended change that lowering removed. |
| Is a resource explanation plausible? | Compiler requirements, code-object metadata, actual launch allocation, and supported counters with their definitions. | Resource ceilings versus measured behavior; register counts alone do not establish residency or latency hiding. |

Follow source and specialization to the available compiler artifacts, executable
object/symbol, and matching launch. Record missing joins rather than treating a
matching name or cache filename as proof of execution. Triton IR stages may be
available from its build; other backends have different artifacts. Vendor binary
disassembly does not recover original IR. Virtual ISA is not final machine code,
and static instruction sites do not include loop trips, masks or repeated calls.
Check the relevant target's instruction definitions before deriving work or bytes.

Before using a number to choose or reject a direction, open the raw cited field
and its definition. Verify value, unit, source/case/configuration, physical device,
measurement interval and aggregation; use code for arithmetic and retain operands.
Read relevant fields together. Missing data is unknown, not zero. Compiler register
requests, rounded allocations and runtime measurements are different quantities;
fixed LDS metadata may omit dynamic launch allocation. Counter names and successful
tool exits do not establish measurement meaning, coverage or causal attribution.

## Keep hardware comparisons conditional

Separate useful mathematical work from executed work, and logical tensor bytes
from issued accesses, cache/interface requests and physical memory traffic. Each
has its own units, reuse assumptions and invocation count. An ISA load width or
semantic-byte rate is not measured HBM traffic. A high resource fraction can still
include work an equivalent organization removes.

For work W, matching bytes Q and elapsed time T, intensity is W/Q and rate is W/T.
A conditional time bound is max(W/P, Q/B), using compatible compute and bandwidth
comparators P and B. Match arithmetic/accumulation mode, memory level, device and
conditions. A microbenchmark or nameplate peak is not guaranteed attainable by
this callable; independent capacities need not be attainable together. Dependencies,
communication and launches may dominate. A model gap is not a promised speedup or
an additional stopping rule. Averaging dispatch rates differs from summing work
and dividing by summed duration; overlapping durations cannot be summed as elapsed
time. Keep separately collected or replayed counters scoped to their actual passes.

## Develop toward a fixed engineering reference

Use a supported positive `T_ref` as a fixed engineering target for the admitted complete
region: bind its useful-work and numerical/state contract, cases and weights,
physical hardware/runtime, and timing boundary. Record the reference version,
expression, operands/units and source or calibration evidence in the existing
report or supplied Design Genome; label a theoretical bound versus an empirical
comparator. An unsupported reference stays unknown; continue useful engineering.

For matching unprofiled time `T_measured`, track `efficiency = T_ref / T_measured`
and conditional gap `1 - efficiency`. A declared positive target fraction `q` gives
`T_goal = T_ref / q`; do not invent a universal target percentage. Preserve the
evaluator's case aggregation; a mean of percentages must not replace its weighted
score. At a fixed reference this is monotonic with latency, not another ranking
signal. Correctness and independent unprofiled timing still determine fitness;
the gap is conditional headroom, not promised removable time. It does not override
declared budgets or stop conditions.

Report applicable case-level attainment and missing reference coverage in the existing
notes/report. A weighted score does not establish that every case met its target, and
local attainment does not establish whole-workload attainment. Preserve residual cost
and broader-boundary requests for the outer owner to assess against its independent
serving-regime measurements and supported workload reference, when available.

Keep this comparison anchor fixed across comparable source revisions. An unchanged
mathematical/state contract does not prove a universal minimum operation count.
Do not improve the numerator by counting padding, redundant work or a candidate's
new formula; report formulation-dependent work/traffic roofs separately. If an
equivalent algorithm invalidates assumed work/bytes or beats the reference, retain
the faster correct source and measured result. Check the model, visibly version
any justified reference correction and recompute comparable history; never clamp
efficiency or reject a candidate for exceeding 100%.

Use source, profile and compiler evidence to explain the residual resource or
dataflow gap and choose the next discriminating experiment. High attainment alone
does not establish a host/launch floor or rule out an equivalent organization.
Reuse applicable evidence; refresh only what the changed source or question needs.

## Use the existing tool path honestly

Start with the supplied `profiling_summary`, `codebase_context`, task handoff and
their raw artifact references. If TraceLens reports or inference traces are
available, use [tracelens_evidence.md](tracelens_evidence.md) to inspect or produce
the deterministic tables through the existing report API/CLI. Reuse applicable
tables; a prose summary or matching operation name alone does not establish their
applicability. Record the precise missing observation when additional evidence
would change the next experiment, and acquire it through the admitted tools or
request it from the collection owner. Continue independent engineering while that
request is pending. For kernel profiler collection,
use the role's existing `scripts/profile_kernel.sh` invocation and resource locks;
read the raw-report and failure sections of [profiling_guide.md](profiling_guide.md).
The report identifies which profiler ran and lists artifacts. Inspect its errors,
coverage and missing fields before interpreting a summary. Tool presence or help
output alone does not qualify collection on this task. Source/IR/ISA observations
must come from supplied artifacts or task-authorized builds/tools that actually
produce them; the profile wrapper does not supply a dedicated disassembly,
thread-trace or PC-sampling mode. This document installs or registers no tools.

Use a competing explanation to choose the smallest diagnostic or implementation
change that could distinguish it. Predict how work, movement, resources or schedule
would change; inspect whether the measured path realized that change. Reuse
applicable evidence and refresh what a changed source or decomposition invalidates.
When the needed artifact or metric is unavailable, state that precise limit and
continue with a conditional hypothesis or other available evidence. Existing
unprofiled correctness/timing, realization records and retention rules remain the
judge of the experiment; a faster result alone does not establish its mechanism.
