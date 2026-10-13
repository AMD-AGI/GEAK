# Validation scope

This distribution is a standalone adaptation. It removes the external workflow
loader and repository layout, vendors the pure delivery contract, and resolves
all runtime helpers within this folder. Attribution to the tools' origin does
not imply a dependency on that repository.

Packaging verification and CPU regression results are recorded in
`validation/results.json` and its logs. Tests execute from an isolated relocated
copy with only the bundle on its Python search path. No external workflow tree
is mounted/copied into that test, and no workflow runner is shipped.

The checks cover selection and trace attribution, capture serialization,
baseline/candidate overlay handling, workload weighting, task timing interfaces,
synthetic-ID device locking, source evidence consistency and local delivery
preflight. Synthetic tests and evidence fixtures are marked as such; they are
not real GPU results. No model or serving process is started by bundle tests.

The full standalone extraction procedure has not yet completed a new real-GPU
campaign in this packaging task. Its donor version had historical GPU UT passes
with incomplete full source/delivery acceptance; those results are not inherited
as a PASS for this distribution. Users must validate their actual source,
workload, graph/compile modes, kernels, cases and multi-rank setup.
