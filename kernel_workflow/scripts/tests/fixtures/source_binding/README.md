# Wrong-candidate source-binding regression

This is a synthetic task, not the original incident archive. Its builder uses the
skip-existing-link logic reported in #480. Keep the harness and oracle unchanged
when comparing workspace copy paths.

The reference expects `y = 2*x`. Change only `return x * 2.0f;` in the candidate
source to `return x * 3.0f;`. Both implementations compile and execute successfully,
but the candidate must fail the independent Python oracle:

| Copy / source | Output for `[1, 2, -3, 0.5]` | Harness verdict |
|---|---|---|
| Parent, correct source | `[2, 4, -6, 1]` | Pass |
| Old copy, bad candidate, retained parent link | `[2, 4, -6, 1]` | False pass |
| Fixed copy, identical bad candidate | `[3, 6, -9, 1.5]` | Fail |
| Another Verify copy of the bad candidate | `[3, 6, -9, 1.5]` | Fail |
| Fixed copy, candidate repaired to `2*x` | `[2, 4, -6, 1]` | Pass |

Run the CPU regression from the repository root:

```bash
python3 -m pytest -q kernel_workflow/scripts/tests/test_workspace_sources.py \
  -k wrong_candidate_false_pass
```

The test compiles and executes the fixture with a host C++ compiler, requires
identical bad-candidate hashes across both copy paths, and checks the resolved
compiler input and unchanged harness. It always rebuilds, excluding stale-binary
reuse as the explanation for the false pass. `build/result.json` records the
source hashes, output, correctness verdict and benchmark eligibility.

The same source also has a HIP implementation. On an allocated gfx950 GPU, its
harness accepts `--compiler /opt/rocm/bin/hipcc`; then the output is computed by an
actual HIP kernel rather than the host loop. GPU execution is separate from the
CPU-only CI test.
