# Report contents

Deliver model notes (Step 0), a report with the sections below, and the machine-readable list of all
candidates in the format the caller defines.

## Workload characterization

- concurrency
- batch / token distribution
- prefill / decode split
- parallelism
- dominant bottleneck
- average kernel duration
- launch overhead
- memory traffic
- communication
- resource utilization

## Boundary ledger

The per-phase kernel sequences and, for every adjacent boundary, the covering candidate or the reason
it is not fused (Requirement 5).

## Candidate fusion patterns

| Rank | Level | Fusion region | Kernels (trace names) | Launches before → after | Expected gain (weighted e2e %, upper / expected) | Main risk | Conflicts | Confidence |
|---|---|---|---|---|---|---|---|---|

## Recommended fusion patterns (implementable scope)

For each recommended pattern state the fusion boundary, the level, the reason for the choice, the
bottleneck it addresses and its seam. The recommended patterns are mutually compatible.

## Why not other levels

For each recommended pattern, why a smaller scope was not chosen and why a larger scope was not chosen.

## Required validation

What must be confirmed before implementation: dependencies, tensor shapes, VGPR/LDS, occupancy, tile
configuration, communication, correctness and e2e performance.
