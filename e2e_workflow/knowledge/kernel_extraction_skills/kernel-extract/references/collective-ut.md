# Multi-rank collective UTs

**Read this only when communication is explicitly in scope.** By default a collective is
marked `entity_kind: "collective"`, selected for mass accounting and given no UT at all
(see [the core skill](../SKILL.md) §2 and [selection coverage](selection-coverage.md)).
Nothing below is owed for an out-of-scope collective: not a rank harness, not a blocker,
not a deferral entry. Record the exclusion with its measured share and move on.

When a run IS scoped to optimize communication, a selected collective remains an
extraction target. Establish the actual rank
contract and native API before declaring it unsupported. A single-GPU runner
cannot exercise a TP8 collective, but an allocated and exclusively leased
eight-device set may support a small standalone rank harness without launching
the model. Preserve the source workload's world size, dtype and dispatch cases.

## Live implementation and editable source

Resolve the communicator used by the serving stack, including backend aliases.
For example, a SGLang deployment may use AITER's communicator rather than the
similarly named SGLang class. Inspect the actual call and captured device symbol.
Establish one native eager/graph replay before expanding the task; reuse this
proof when it already exists.

Copy the native implementation and required local headers, with original paths
and hashes. An installed Python wrapper or a compiled binary alone does not
prove native editability. Reuse compatible build settings from the deployed
stack, resolving relocated include paths against the current container. Record
dependency paths, compiler flags and the built library's hash. Keep generated
libraries and build files under the task; use distinct module identities for
different source contents so a cached library cannot hide an edit.

Bind the candidate through the same deployed callable/overlay contract as other
tasks. A prototype that directly installs a local native module is useful source
proof, but the final task still needs baseline isolation and candidate binding.
If an unchanged communicator constructs native state for the candidate call,
verify that its ABI and state layout remain identical. Preserve IPC allocation,
registration and destruction lifecycles; numerical inputs are not the complete
collective state.

## Rank execution and acceptance

Launch all ranks under one full-set lease using `gpu_lock_set.py`. Use bounded
process-group initialization and a launcher that terminates failed rank groups.
Collect per-rank results; a successful launcher or rank 0 alone is insufficient.
Keep numerical inputs distinct across ranks and compare every rank's output
with an independent sum/reference. Validate eager execution and the deployed
graph mode, including graph replay after in-place changes to rank inputs.

Follow the native communicator's graph-registration lifecycle. Some APIs
register captured buffers when their capture context exits; replaying before
that exit is invalid. Build the graph through that API, then time its replay
using the shared timing primitive. Record the actual graph mode. All ranks
must execute the same case order and fixed iteration counts; independently
chosen benchmark loop counts can leave peers waiting in a collective.

Prove native source execution with a reversible numerical change in the selected
device implementation, such as changing its reduction result. Every expected
rank must reject the returned values; an import, IPC or compilation error is not
numerical rejection. Restore the original source for final validation. Record
actual symbols and launch configurations per case and rank, including dispatch
switches that require different reference kernels.

One-case API, editable-source or negative-control success is an intermediate
result. Completion still requires the standard `unittest.py`, all declared
positive-mass cases on both legs, baseline/candidate identity, observed-call
weights, per-rank evidence and a final task report. Preserve the original
whole-workload denominator when a collective case remains uncovered.
