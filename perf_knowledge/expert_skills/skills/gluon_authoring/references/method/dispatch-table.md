# Shipping a per-shape dispatch table

When different tiles / depths / kernels win on different parts of the served range, ship a shape
predicate → implementation table with a default. Checks before it counts:

- **Suggested: mark each row `MEASURED` (timed interleaved vs the incumbent at the deployment boundary),
  `INFERRED` (write the measured rows it extrapolates from and what would falsify it), or `DEFAULT`.**
- **Every row, timed or not, is compiled and passes the correctness oracle** on a representative shape;
  a row that cannot is removed. Put alignment / divisibility constraints in the predicate.
- Half-open intervals; boundaries at measured crossovers (found by the served-range sweep,
  `benchmark-hygiene.md ## Served-range sweep`), with points just below / at / above each cutoff.
- Confirm each bucket launches the kernel you think (check the dispatched kernel name); a
  `tensor.item()` in the selector adds a GPU→CPU sync that can erase a sub-ms win (`front-end.md ## Host side`).
- Run an identical-arm null on the harness first; weigh rows by the served mix, not per-shape µs.
- The table is pinned to the box and Triton build; re-validate after a toolchain bump.
