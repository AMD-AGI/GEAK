# Fusion heuristics

Priors by workload regime. They order what to inspect first; they never decide a level on their own.

| Regime | Inspect first | Conditions and focus |
|---|---|---|
| Low concurrency | L4 MonoKernel | Only when there are many kernels, they are short, launch/sync overhead is significant, the dependency chain is strong and resource pressure is manageable. |
| High concurrency | L3 Mega / MegaMoE | Token count, per-expert batch size, GEMM utilization, dispatch/combine overhead, communication/computation overlap. |
| Medium concurrency | No preset level | Compare L2, L3 and partial L4, and decide by benchmark. This regime is the least explored. |
| Many short kernels | L1 / L2 | Remove launch overhead and HBM round trips. |
| GEMM or attention dominates | Protect the dominant operator | Keep its optimal tile, occupancy and MFMA efficiency; do not trade the dominant operator's performance for fusion. |
| Communication-heavy MoE | L3 | Overlap between dispatch, expert GEMM, combine and all-to-all. |
