# Gates that also judge candidates

Optimization runs the task's frozen `unittest.py` on every candidate edit of
`kernel_src/`, so the UT is the optimizer's oracle as well as the extraction's
proof. Leaf invocation and per-leaf mutations ([task authoring](task-authoring.md)
steps 2 and 6) prove facts about the pristine snapshot: which leaves the deployed
op runs, and that each mutation breaks the outputs. Re-run verbatim on a candidate,
they fail every candidate that fuses leaves, adds a kernel or re-dispatches a
branch for no longer being the extracted code, and optimization can then only tune
leaves in place. On a candidate, bind and control the kernels it actually launches.

Correctness at served shapes, graph/compile behavior and input refresh,
restored-source integrity and both timing legs over every positive-mass signature
are unchanged. So are the frozen cases, weights and baseline.

## Launch census

In each process, record every device kernel the target op launches on each
positive-mass signature, as
`{"sigs": {sig: {name: {"launches": n, "source_file": path}}}}`. Attribute each
launch at its launcher (a JIT's compiled-kernel entry, a native binding, a library
call) with the file that defined the kernel. A launch the recorder cannot attribute has
`"source_file": null`; never attribute by name. Collect the baseline census in the
frozen-baseline process the UT already runs, with its outputs unchanged, and the
candidate census in an untimed candidate diagnostic process.

## Binding

`h.candidate_binding(baseline, candidate, kernel_src, leaf_prefixes, aliases=...)`
passes when, on every signature, the candidate launches at least one kernel from
its own `kernel_src/`, and every other kernel it launches there is one the
baseline launched from the same non-editable place on that signature.
`leaf_prefixes` are the `kernel_prefix` values of `meta.leaves`; a leaf kernel
launched from anywhere else is the installed copy, and without them binding fails
closed. Any other new kernel, attributed or not, is foreign. The
candidate overlay copies the bound module into `_cand_overlay/_patched/`: map that
copy to its `kernel_src/` name with `aliases`. Leaves the candidate no longer
launches (`retired_leaves`) and the kernels it adds (`added`) are reported, not
failed. `h.assert_legs_differ` still runs.

## Negative controls

The extraction's controls stay in `meta.leaves[]`, each with `module`, `file`,
`kernel_prefix` and `mutation: {find, replace, what}`. A candidate declares a
reversible mutation for each kernel it adds or rewrites in
`kernel_src/negative_controls.json`. The catalog is candidate source, not part of
the frozen UT:

```json
{"controls": [{"id": "fused_epilogue", "file": "fused.py",
  "find": "acc * scale", "replace": "acc * (scale * 0.5)",
  "kernels": ["fused_kernel_prefix"], "what": "halve the output scale"}]}
```

`file` is relative to `kernel_src/` and `kernels` are launched-name prefixes.
`h.load_declared_controls(kernel_src)` validates the catalog, and
`h.controls_to_run(own, h.extraction_controls(meta), declared)` selects every
declared control plus the extraction control of each leaf the candidate still
launches. Apply each with `h.apply_control` to a temporary copy, run the candidate
from that copy in a separate untimed process with the census recorder on, and
compare its outputs to the frozen baseline outputs.

`h.negative_control_verdict(own, results, tol)` passes when every declared control
passes and every kernel the candidate launches from `kernel_src/` has a passing
control in that kernel's own file. A control passes when its anchor applied, a
kernel it names launched from the mutated copy, and the outputs moved past ten
times the tolerance; where the launcher exposes a compile identity, it must also
have changed. Build or import failure never passes, and neither does a declared
control whose kernels no longer launch: a candidate removes a control with its
kernel. On a candidate, a failing extraction control is reported as stale: the
candidate may have rewritten what it mutated.

## Pristine snapshot, results and declaration

When `kernel_src/` matches the original hashes in `baseline_ref/`, every claimed
leaf must launch and every extraction control must pass: these are the
extraction proofs behind the [source records](source-delivery-records.md). Give
every kernel the pristine snapshot launches from `kernel_src/` its `meta.leaves`
entry, companions included, so the extraction controls cover the unmodified
task. On any other snapshot, report `retired_leaves` and stale extraction
controls in `run/ut_result.json` with the binding and control verdicts, so a lane
sees why a candidate failed.

The candidate overlay adds one module. Load new files the way the bound module
loads its siblings, from the task's own `kernel_src/`; a file the bound module
never loads never launches.

Declare the gates in `meta.json` and copy the same object into the extraction
return (see [workflow handoff](workflow-handoff.md)):

```json
"candidate_gates": {"version": 1, "catalog": "kernel_src/negative_controls.json",
                    "launchers": ["triton"]}
```

`launchers` lists the author languages (`flydsl`, `triton`, `hip`, `ck`) whose
kernels the recorder attributes at their launch path. Leave out a language whose
kernels would launch unattributed: binding fails them as foreign, and downstream
planning does not author that language for this task.
