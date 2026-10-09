#!/usr/bin/env bash
# NUMA pinning for serving benchmarks. Sourced by bench_e2e.sh.
#
# Why: on a 2-socket host a fresh server lands on either NUMA node. Measured on MI300X
# (Qwen3.5-35B-A3B-FP8, vLLM, TP=1 on GPU 0, which sits on node 0): 3 fresh servers pinned
# to node 0 gave 1064.7 / 1065.2 / 1069.8 tok/s, 3 pinned to node 1 gave 1016.3 / 1014.3 /
# 1019.1. Unpinned, each fresh server picked a mode at random -- an ~5% bimodal spread that
# no ~1% A/B win can clear, and ~2.5% of throughput lost on average.

# geak_numa_cpus "<gpu ids, comma-separated>" -> the CPU list of the one NUMA node all those
# GPUs sit on; prints nothing (and returns 1) when topology is unknown or the GPUs span nodes.
# GEAK_TOPO_NUMA_CMD / GEAK_NODE_SYSFS override the probes (tests).
geak_numa_cpus() {
  local gpus="$1" topo node="" gpu n
  topo="$(${GEAK_TOPO_NUMA_CMD:-rocm-smi --showtoponuma} 2>/dev/null)" || return 1
  for gpu in ${gpus//,/ }; do
    n="$(printf '%s\n' "$topo" | sed -n "s/^GPU\[$gpu\][^:]*: (Topology) Numa Node: *\([0-9][0-9]*\).*/\1/p" | head -1)"
    [ -n "$n" ] || return 1
    if [ -z "$node" ]; then node="$n"; elif [ "$node" != "$n" ]; then return 1; fi
  done
  [ -n "$node" ] || return 1
  cat "${GEAK_NODE_SYSFS:-/sys/devices/system/node}/node$node/cpulist" 2>/dev/null || return 1
}
