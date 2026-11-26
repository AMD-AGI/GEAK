---
name: multinode-traces
description: Check and combine the traces of a finished multi-node (tp_span or pd_disagg) serving leg -- gate the per-rank trace set, plan PD sampling points, weigh the two PD engines by GPU-time mass, merge their Top-Ns and workloads. Reads files and writes reports only; no cluster, no ssh, no pin. Use from the multinode role fragments; not for single-node runs.
---

# Multi-node trace checks

One script, `scripts/mn_traces.py <command>` (stdlib only). The multinode launcher
(`scripts/adapters/launchers/multinode.sh`) runs `trace-gate` itself after every profile
window; the profiler and AgentX extraction agents run the others as their multinode
fragments (`roles/_fragments/profiler_mn.md`, `agentx_multinode.md`) say. Exit 2 means
bad input (the reason is on stderr); nothing is written then.

## trace-gate: the per-rank trace set

    mn_traces.py trace-gate --profile-dir DIR --expect-ranks N --nodes K \
        [--launch-json mn_launch.json [--role prefill|decode] [--window-s S]]

Every rank writes its trace into one directory on the shared mount, where `ls -t | head -1`
is an NFS flush race. The gate checks that every expected TP rank on every node left a
complete trace with collectives in it, and names the one to parse:

- `DIR/ranks/mn_trace_report.json`: `verdict` (`pass` -> exit 0, else 1), per-rank work volume,
  `parsed_trace` (TP-0).
- pd_disagg (`--launch-json`): each engine is gated in its own subdirectory, and
  `DIR/ranks/mn_trace_roles.json` names each engine's TP-0 trace and ranks. `--role` gates one
  engine (a sampling-point probe opens its window on one).

## pd-points: sampling points of a fixed-length PD run (sglang, ATOM)

    mn_traces.py pd-points --engine-log prefill=LOG --engine-log decode=LOG \
        --skill-root <kernel-profile> --out-dir DIR [--points-per-phase 3]

Plans the points from each engine's BASELINE server log with kernel-profile's
`observed_workload.py --mode multinode` (`--log-format atom` for an ATOM engine log) and
`plan_sampling_points.py`, then writes
`DIR/sampling_points.json` and prints `PROFILE_PROBE_IDS=<ids>` for `bench_e2e.sh`.

## pd-mass: AgentX ranking on both engines' GPU-time mass

    mn_traces.py pd-mass --coverage coverage.json [...] --roles mn_trace_roles.json [...] \
        --apply kernel_ranking.json

Weighs each engine by its phase's mass from `coverage_ledger.py` x its ranks, never by the graded
metric, and rewrites the ranking in place: each row's `selection_pct_gpu_time` becomes its
`share_of_phase_pct` (the ledger's `phase_mass_share_pct`) x those weights (the agent's number is
kept as `selection_pct_gpu_time_agent`), and `denominators` records the weights
(`rule = gpu_time_mass`). `agentx_kernel_preparation.mjs` refuses a PD ranking without it.
`--roles` once per probed engine.

## topn-merge: one Top-N for a PD run

    mn_traces.py topn-merge --engine prefill=topN.json --engine decode=topN.json --roles mn_trace_roles.json \
        [--coverage coverage.json ...] [--workload prefill=w.json --workload decode=w.json --workload-out W] \
        [--out PREFIX] [--top 25]

Weighs each engine's Top-N by its GPU time and writes `PREFIX.json` / `PREFIX.md`, with
`engine_weights` and `bottleneck_engine`:

- with `--coverage` (probed points): each engine's GPU-time mass over the baseline leg;
- without: AgentX's no-log fallback, each window's mean GPU time per step x `parse_profile.py`'s
  `analytic_calls` x ranks.

Never the GPU time of the profile windows themselves, and never TTFT / TPOT, which only explain.
`--workload-out` merges the two engines' workloads by the same weights.
