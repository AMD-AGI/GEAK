#!/usr/bin/env bash
# =============================================================================
# config.sh — SINGLE place to change all GEAK CI timeouts / caps / knobs.
#
# This is the one file to edit. It is sourced by ci/lib.sh (which every other
# ci/*.sh sources or inherits from), and it only `export`s values, so every
# script and every child process (incl. the SPUR job + the container, via env
# propagation) sees them. Each line uses ${VAR:-default}, so an env override
# (CI secret, `--budget`, a one-off `FOO=... ci/...`) still wins over the file.
#
# Times are SECONDS unless noted. Toggles are 1=on / 0=off.
# =============================================================================

# ---- GEAK e2e budget --------------------------------------------------------
# Per-model GEAK wall-clock budget. The workflow passes this via --budget; this
# is only the fallback when nothing is passed.
export PERFSKILLS_E2E_TIMEOUT_S="${PERFSKILLS_E2E_TIMEOUT_S:-57600}"

# ---- Matrix orchestrator (run_matrix.sh) ------------------------------------
# NB: there is intentionally NO pending timeout — run_matrix.sh waits on PENDING
# jobs indefinitely (only the GitHub timeout-minutes bounds it). Cancel a
# long-pending job by hand on the cluster if needed.
export GEAK_MATRIX_POLL_S="${GEAK_MATRIX_POLL_S:-60}"       # squeue poll cadence while waiting (job-completion detection latency)
export GEAK_MATRIX_LOG_S="${GEAK_MATRIX_LOG_S:-1200}"      # 'queue:' status-line log cadence (20 min); also always logged on state change
export SPUR_DRYRUN="${SPUR_DRYRUN:-0}"                      # 1 = print sbatch cmds, don't submit (also --print)

# ---- SPUR / SLURM submission (slurm_submit.sh, lib.sh) ----------------------
export SPUR_PARTITION="${SPUR_PARTITION:-amd-spur}"         # the only partition on this cluster
export SPUR_CPUS_PER_GPU="${SPUR_CPUS_PER_GPU:-8}"          # cpus-per-task = gpus * this
export SPUR_TIME_HEADROOM_S="${SPUR_TIME_HEADROOM_S:-7200}" # added to the GEAK budget for pull/install/bench
export SPUR_PROBE_TIME="${SPUR_PROBE_TIME:-1:00:00}"        # wall time for --probe jobs (H:MM:SS; image pull + claude, no e2e)

# ---- Account/QoS auto-selection (lib.sh pick_account) -----------------------
# The partition has plenty of idle nodes; the real limit is the per-QoS group
# node cap. pick_account() probes candidates (per model, using that model's GPU
# footprint) and submits to the first that can place the job now; if none can,
# it submits to SPUR_ACCOUNT_FALLBACK and lets it pend.
export SPUR_AUTOSELECT="${SPUR_AUTOSELECT:-1}"             # 0 = disable; use SPUR_ACCOUNT/SPUR_QOS as-is
export SPUR_ACCOUNT_CANDIDATES="${SPUR_ACCOUNT_CANDIDATES:-amd-hyperloom:amd-hyperloom-qos amd-general:amd-general-qos amd-primus:amd-primus-qos}"
export SPUR_ACCOUNT_FALLBACK="${SPUR_ACCOUNT_FALLBACK:-amd-hyperloom:amd-hyperloom-qos}"
export SPUR_PROBE_WAIT_S="${SPUR_PROBE_WAIT_S:-24}"        # watch a probe this long before deeming a QoS full
export SPUR_PROBE_POLL_S="${SPUR_PROBE_POLL_S:-3}"         # probe poll interval
# Effective account/QoS used ONLY when auto-select is off, or for --print
# display; with auto-select on these are overwritten per job by pick_account().
# Default to the fallback pool so there is a SINGLE hardcoded account here.
export SPUR_ACCOUNT="${SPUR_ACCOUNT:-${SPUR_ACCOUNT_FALLBACK%%:*}}"
export SPUR_QOS="${SPUR_QOS:-${SPUR_ACCOUNT_FALLBACK##*:}}"

# ---- GPU arch / image selection (lib.sh) ------------------------------------
export GEAK_GPU_ARCH_DEFAULT="${GEAK_GPU_ARCH_DEFAULT:-MI355}"  # used when rocminfo can't be read (this cluster = gfx950)

# ---- Node runner (run_local.sh) ---------------------------------------------
export IMAGE_PULL_CAP="${IMAGE_PULL_CAP:-1800}"                    # `docker pull` cap on a cold node
export GPU_HEALTHCHECK_TIMEOUT_S="${GPU_HEALTHCHECK_TIMEOUT_S:-120}" # GPU preflight probe cap (0 = skip)
export GEAK_KILL_BUFFER_S="${GEAK_KILL_BUFFER_S:-300}"            # kill the container this long BEFORE the SLURM wall clock
export GEAK_SKIP_PULL="${GEAK_SKIP_PULL:-0}"                      # 1 = skip docker pull
export GEAK_SKIP_DSTATE_CHECK="${GEAK_SKIP_DSTATE_CHECK:-0}"      # 1 = skip GPU-wedge D-state pre-check
# Host-side liveness monitor (run_monitor.sh) watches a live run and kills it
# early if it WEDGES (vs limping to the wall clock). Two modes (GEAK_MONITOR_MODE):
#   * stall  — deterministic, NO deps: kills only on POSITIVE evidence of a wedge
#              (log flat AND GPUs idle AND container CPU idle, sustained). A long
#              silent bench/build/profile keeps GPU or CPU busy, so it is NEVER
#              killed; if GPU util can't be measured it degrades to warn-only.
#   * claude — LLM arbiter (needs the claude CLI): reads the log tail and votes.
# Default OFF regardless of mode. Enable with GEAK_MONITOR=1 (stall mode needs no
# claude; claude mode needs the CLI on the dispatched GPU host).
export GEAK_MONITOR="${GEAK_MONITOR:-1}"                          # 1 = start host-side liveness monitor
export GEAK_MONITOR_MODE="${GEAK_MONITOR_MODE:-stall}"           # stall (deterministic) | claude (LLM arbiter)
# GEAK_HARD_TIMEOUT_S: leave UNSET to auto-derive (budget + headroom - kill buffer);
# set it to force an explicit hard-timeout instead.

# ---- Preflight (gpu_dstate_check.sh) ----------------------------------------
export GEAK_DSTATE_SAMPLE_GAP_S="${GEAK_DSTATE_SAMPLE_GAP_S:-3}"  # gap between the two D-state samples

# ---- Host-side liveness monitor (run_monitor.sh) ----------------------------
export GEAK_MONITOR_INTERVAL_S="${GEAK_MONITOR_INTERVAL_S:-300}"       # normal poll cadence
export GEAK_MONITOR_RECHECK_S="${GEAK_MONITOR_RECHECK_S:-60}"          # faster re-poll while confirming a KILL
export GEAK_MONITOR_CONFIRM="${GEAK_MONITOR_CONFIRM:-2}"               # consecutive KILL votes required to act
export GEAK_MONITOR_TAIL_LINES="${GEAK_MONITOR_TAIL_LINES:-300}"       # log tail lines fed to the arbiter
export GEAK_MONITOR_CALL_TIMEOUT_S="${GEAK_MONITOR_CALL_TIMEOUT_S:-180}" # cap a single claude call (claude mode)
export GEAK_MONITOR_STARTUP_GRACE_S="${GEAK_MONITOR_STARTUP_GRACE_S:-300}" # grace before the first judgement
export GEAK_MONITOR_MODEL="${GEAK_MONITOR_MODEL:-claude-opus-4-8}"     # arbiter model (claude mode)
# ---- Deterministic stall watchdog (run_monitor.sh MODE=stall) ---------------
# A wedge is declared ONLY when the log has been flat AND both GPU and CPU idle
# for GEAK_STALL_KILL_S, confirmed GEAK_MONITOR_CONFIRM times. Generous by design
# so a long silent-but-working leg (bench/build/profile) is never killed.
export GEAK_STALL_KILL_S="${GEAK_STALL_KILL_S:-3600}"                 # flat+idle duration before a kill is considered (60 min)
export GEAK_STALL_GPU_UTIL_PCT="${GEAK_STALL_GPU_UTIL_PCT:-5}"        # max GPU util% counted as "idle"
export GEAK_STALL_CPU_PCT="${GEAK_STALL_CPU_PCT:-5}"                  # container CPU% counted as "idle"
