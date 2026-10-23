# Multi-node serving (one server, the machines the handoff names)

`BENCH_LAUNCHER=multinode` serves **one** server instance on the GPU machines the
caller names (one machine, a TP group spanning several, or a prefill/decode pair).
GEAK and its agents run on one machine, the driver, which sees no GPU and holds the
only GEAK code; it reaches every node over ssh and does not manage the cluster. The
bench client runs on rank0. A node needs sshd, the inference framework and `python3`,
nothing of GEAK. Only
*who starts the server* and *where the client runs* change:
the backend adapter, the bench client, the metric and the acceptance contract are
the single-host ones, which is what keeps a two-node tok/s number comparable to the
rest of GEAK's results instead of making it a separate dialect.

## Where the framework knowledge lives

Nothing under `scripts/multi_node/` knows how to start a server. The split is:

| layer | owns | lives in |
| --- | --- | --- |
| **where** | the handoff's machines reached over ssh, their NICs, teardown, trace gate | `launchers/multinode.sh`, `multi_node/*` |
| **how** | the launch line, and how the framework says *"I am rank i of n"* | `adapters/sglang.sh` |
| **what** | model flags and the flags GEAK sweeps | `EXTRA_SERVER_ARGS` / `EXTRA_ENV`, or the caller's `launch_command` |

The seam between the launcher and `mn.py` is the only one the runtime forces;
`mn_traces.py` is split off because it never touches the cluster:

| file | why it is separate |
| --- | --- |
| `adapters/launchers/multinode.sh` | `bench_e2e.sh` sources `adapters/launchers/$BENCH_LAUNCHER.sh` and calls the `adapter_*` functions it defines **in its own shell**. A launcher has to be a shell file at that path. |
| `multi_node/mn.py` | the entry point: `check`, `launch`, `supervise`, `stop`, `leg-env`, `active-pin`, `exec`, `digest`, `trace-gate`, `topn-merge`. They are separate *processes* (`supervise` is the pid `bench_e2e.sh` kills) but not separate files. Everything that reaches a node over ssh is here. |
| `multi_node/mn_traces.py` | what `trace-gate` and `topn-merge` run: reads a finished leg's traces, Top-Ns and logs and writes reports. No ssh, no pin. |

`mn.py launch` builds each rank's script itself: the same
exports `bench_e2e.sh` does on one box, then `. adapters/$BACKEND.sh` and
`adapter_launch`, with the structural flags appended last. The NIC is read from
the pin (`cluster_if`) and checked on the node: if it no longer carries the
handoff's IP the rank exits before joining any collective. The script is saved as
`OUT_DIR/mn_rank<N>.sh` (and `mn_lb.sh` for PD).

That is the **generated** mode (the prompt names roles and ssh only). In the
**provided** mode (Hyperloom passes each pod's `launch_command`, and the PD router's),
check parses every command (`engine_facts`: model, `--tp-size`, `--port`, PD role,
bootstrap port) and refuses one that disagrees with the layout, the handoff's model, or
the router's worker URLs; the commands are then pinned, so no leg can run a different one.
Each rank renders its command into `OUT_DIR/mn_cmd_rank<N>.sh`, a `launchers/magpie.sh`
server-phase wrapper, and runs it through that contract on its node with GEAK's layer
mapped onto it. Env precedence is recipe < the command's `K=V` prefix < `EXTRA_ENV` <
the names a run owns (a prefix that sets one of those is refused). GEAK's flags
(`EXTRA_SERVER_ARGS` → `EXTRA_<BACKEND>_ARGS`) go after the command's, minus the
structural ones (`--port`, `--tp-size`, the PD role and dist flags), which stay the
command's own; what was stripped is recorded as `extra_args_dropped` in `mn_launch.json`.
xPyD is one command per role: every replica of a role must run the same command apart
from `--host`/`--port`. In both modes a rank refuses to start if the overlay's content
digest (`mn.py digest`) differs from the one GEAK computed when it rendered the leg.

Each rank sources `adapters/$BACKEND.sh` **on its own node** and calls
`adapter_dist_args RANK NNODES MASTER DIST_PORT`. Adding a backend is therefore
two lines in its adapter, not a fork of the launcher:

```bash
adapter_dist_mode() { echo per_rank; }   # one server process per node
adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }
```

`adapter_dist_mode` exists because the process model is not universal: vllm's ray
executor fans out to the other nodes from a *single* process (`head_only`), so
there is no per-rank launch to do. The launcher implements `per_rank` and refuses
anything else by name rather than launching sglang's shape and calling it vllm's.
`adapters/vllm.sh` therefore uses vllm's `mp` executor instead, which is per-rank:
every node runs `vllm serve --nnodes N --node-rank i --master-addr --master-port`,
and ranks other than 0 add `--headless` (no HTTP front end; rank0 serves the client).

Flags are layered in this order, the same one `launchers/magpie.sh` uses:
GEAK's shared flags (`EXTRA_SERVER_ARGS`) → that engine's own accepted flags
(`GEAK_MN_PD_SWEEP_ARGS_<ROLE>`, pd_disagg only) → the rank/world or
prefill/decode triple **last**. argparse is last-wins, so a per-engine sweep
beats the shared sweep on that engine only, and nothing a sweep sets can change
which rank a process is or which half of a disaggregated pair it is. A model's
own flags (dp-attention, attention backend) are either the caller's
`launch_command` or the baseline's `EXTRA_SERVER_ARGS`.

Overlays follow the same rule on `PYTHONPATH`: that engine's own overlay
(`GEAK_MN_PD_OVERLAY_<ROLE>`, a kernel accepted on that engine's shapes) comes
ahead of the shared `OVERLAY_PYTHONPATH`, so import resolution picks the
engine's copy of a module first and the other engine never loads it.

`mn_launch.json` records what each engine was actually launched with
(`pd.effective_args_by_role`).

Measured on two MI300X nodes (DeepSeek-V4-Flash-FP8, TP16 DP2, ISL/OSL 1024,
CONC 8): **165.42 tok/s, spread 1.96%** over three timed rounds. That run used two
untimed warmup rounds instead of the default one, through a bench option that is not
part of this change.

## Why the server still has a local pid

ssh only spawns one process per node; the distributed work is the framework's
own (each rank is told which rank it is, and the ranks form their own
`torch.distributed` world over the cluster NIC). `bench_e2e.sh`'s lifecycle
contract is that it owns a **local pid** and `kill`ing it stops the server; every
stall check, teardown and acceptance gate is built on that. `mn.py supervise` *is*
that pid on the driver; on SIGTERM it reaps the remote ranks over ssh. A
supervisor killed with -9 cannot; its ranks keep the cards until the next
`mn.py launch`, which always tears down whatever the pin's nodes still run.

Each rank records its server's pid and start time on its own node
(`/tmp/geak_mn/rank<N>.pid`; provided mode records magpie's server once its
server phase is up). Liveness reads that record; a pid that now belongs to
another process counts as dead. A probe that does not come back over ssh is a
failed probe (six of them in a row give up), never a dead rank.
Teardown kills, per node: the recorded pid's process group, then
our servers by name, then the group with -9, and removes the record. A provided
router is a node of its own (`ranklb.pid`, possibly a pod with no GPU) torn down the same way.

## Where commands run

Every agent command, serving leg, `mn.py launch` and `mn.py supervise` runs on the
driver, and so does `run_e2e.py`: there is no forwarding. The driver sees no GPU:
`run_e2e.py` empties `ROCR/HIP/CUDA_VISIBLE_DEVICES` for a multinode handoff, and check
refuses a driver whose `rocminfo` still lists a GPU (start Claude with the three masks
empty on the prompt entry). What a node executes is either shipped by content over ssh
(check's probe, the GPU identity probe, each rank's script) or a run artifact under
`shared_fs` (the eval dir's `bench_e2e.sh` and `adapters/`, the overlay, the task dirs,
and `<work_dir>/.mn_tools`).

`mn.py check` records the active pin in `/tmp/geak_mn/state_path` on the driver,
owned by `run_e2e.py`, or (`check --owner claude`, a prompt-started workflow) by the
Claude Code process the check agent ran under; the record counts only while that
process lives. While it is active:

- `bench_e2e.sh` with no launcher (or `auto`) serves on the pin; any other launcher is
  refused. An isolated one-card server goes through `mn.py exec --gpu` (below), where
  `bench_e2e.sh` accepts only `BENCH_LAUNCHER=native TP=1` on one card;
- `gpu_lock.sh` and `profile_kernel.sh` run on a node through `mn.py exec --env`: a bare
  id on the master, `<engine>:<id>` (e.g. `decode:3`) on that engine's machine. What
  runs there is the copy check staged in `<work_dir>/.mn_tools`, and it locks the pin's
  card leases (`<shared_fs>/.mn_lease`) instead of `/tmp/team_gpu_locks`.

Another machine is reached explicitly:

```bash
GEAK_MN_STATE=<pin> python3 mn.py exec --target <master|rank<N>|prefill|decode|host> [--gpu 3] -- <cmd>
GEAK_MN_STATE=<pin> python3 mn.py exec --gpu decode:3 -- <cmd>        # the card names its engine
GEAK_MN_STATE=<pin> python3 mn.py exec --target rank0 --script job.sh
```

It runs on the pinned node, in the same directory (the shared mount is at the same
path everywhere), and returns the remote exit code; a target that is this machine
runs locally, without ssh. The command runs in its own session and exports
`GEAK_MN_HOP=1`, so a tool started there does not forward again; when the ssh closes
(the caller died, or got TERM/INT, or `--timeout` passed) that session gets TERM, then
KILL 180 s later. `--gpu` exports `HIP_VISIBLE_DEVICES` and `GPU_ID` and
holds a per-card `flock` under `<shared_fs>/.mn_lease/` while the command runs, so
parallel tasks never share a card. A serving leg holds the same leases for every
card of every pinned node (`multinode.sh`, until the leg ends), so no GEAK task runs
on a card GEAK is serving from; it waits instead. `--env` also exports this shell's
environment there (minus host paths, card masks and credentials). Every e2e role
prompt is told this command; an isolated one-card server is
`mn.py exec --target T --gpu N -- env BENCH_LAUNCHER=native GPU=N TP=1 ... bash bench_e2e.sh`.

A serving leg runs on the driver except for the ranks, so the health probe, the
profiler requests and the bench client cross the network to the client (or
load-balancer) port, which check dials from the driver. The bench client runs where
its dependencies are. The InferenceX client (the checkout Hyperloom names in
`handoff.inferencex_path` or `INFERENCEX_PATH`) and the AgentX aiperf client import no
serving framework and run on the driver, the vantage point Hyperloom's rebench measures
from; check finds `benchmark_serving.py` there, imports the client's packages and reads
the model's tokenizer. The backend's own client (`native`: `sglang.bench_serving`,
`vllm bench serve`, atom's; also what `auto` falls back to without a checkout) imports
its GPU stack and runs on rank0 through `mn.py exec --target rank0 --env`, with the
leg's shell variables and functions and a warning; check imports it on rank0 and dials
the client port from there, and the result records `bench_client_host: rank0`.
A profile round never reaches a client's `--profile`: the launcher starts a fresh load
and opens its own window, which spans every rank. The result line lands in
`RESULT_JSONL` on the shared mount. The single-host memory-parity guard is skipped: the driver's `rocm-smi`
cannot see the nodes' cards. The clients run from `/`: `python -m` puts the working directory first on
`sys.path`, and a task directory's `unittest.py` would shadow the stdlib.

## What the caller must hand over

GEAK selects no machines and does not create, resize or tear down the cluster.

**Through `run_e2e.py` (Hyperloom)** the cluster is the JSON file `MULTI_NODE_STATE`
names, a `node_state.json`; the handoff carries no machine information (a
`handoff.multinode` is refused). `run_e2e.py` translates it into the `multinode` block
below (provided mode) and nothing else reads it:

| `node_state.json` | becomes |
| --- | --- |
| `backend` | must equal the handoff's `framework`; picks the command parser |
| `pd_mode` | `disaggregated` → `pd_disagg`, else `tp_span` |
| `pods[]` (`role`, `podIP`, `launch_command`, `sshPort`/`sshUser`) | one node per engine pod, one pod = one replica (xPyD: `replicas_by_role` counts them); the `router` pod is `router`, and may hold no GPU |
| `gpus_per_node` | `tp_by_role` (each engine's TP; its command's `--tp-size` must agree) |
| `pods[].gpuIds` (optional, e.g. `"4,5,6,7"`) | the node's `gpu_ids`: the cards its engine serves from, instead of `0..TP-1`. Two engine pods may share a `podIP` (1P1D on one machine) only if both name disjoint `gpuIds` and different `--port`s |
| `service_url` | `serve_port`; its host must be the router pod (else the first engine) |
| `ssh_user` (default `root`), `ssh_port`, `ssh_key_path`, `ssh_prefix` | the `ssh` block |
| `shared_fs` | `shared_fs`; the pin is `$shared_fs/pins/mn_state_<hash of the pod IPs>.json` |
| `profile_trace`, `steady_state_trace` | TraceLens's first-round traces: one `<podIP>_<role>/` dir per engine pod; every pod's rank0 trace is annotated as one union (`trace_files`, `steady_state_trace` first). No by-role Top-N is written for this first round; GEAK's own re-profiles split by role as usual |
| `external` | ignored |

**Directly to `e2e_workflow.js`** (the interactive entry) the caller passes the
`multinode` block itself as `args.multinode`, generated or provided:

```json
{
  "tp": 16,
  "multinode": {
    "topology": "tp_span",
    "nodes": ["10.10.0.166", "10.10.0.32"],
    "shared_fs": "/shared/geak"
  }
}
```

| field | |
| --- | --- |
| `nodes` | **required**; one or more machines in rank order (rank0 is the master). Each is its fabric IP, or `{host, role?, rank?, ssh?, launch_command?}` |
| `shared_fs` | **required**; a path every node and the driver mount at the same path. The eval dir, the overlay, the per-rank logs and the traces live there, which is why nothing ships files between nodes. GEAK itself need not be on it |
| `work_dir` | set by the workflow (`args.eval_dir`, else `exp_root`); must be under `shared_fs`. check stages the kernel tools in `<work_dir>/.mn_tools` (`tools_dir` in the pin) |
| `topology` | `tp_span` (default: one TP group over the nodes, a single node included; `tp` is the world size) or `pd_disagg` (prefill then decode — node roles order them — `tp` is per engine) |
| `tp_by_role` | optional, `pd_disagg` only; `{"prefill": N, "decode": M}` when the two engines differ. An omitted role uses `tp` |
| `nodes_by_role` | optional, `pd_disagg` only; `{"prefill": N, "decode": M}` machines per engine replica (default 1 each); implied when every node names its `role` |
| `replicas_by_role` | optional, provided `pd_disagg` only; `{"prefill": X, "decode": Y}` engines per role (xPyD, default 1). Profiling and the trace gate use replica 0; health checks every replica |
| `backend`, `model` | optional, provided only; the command parser (default `BACKEND`) and the model every command must name |
| `pd` | optional, generated `pd_disagg` only; `{backend, ib_device, lb}`: the KV transfer backend (default `mooncake_tcp`), its IB device, and `router` (default) or `minilb` in front. Engine ports follow `serve_port` |
| `router` | **required** with provided `pd_disagg`; `{host, launch_command, ssh?}`, the backend's own router. `host` may be a node or a GPU-less pod of its own; its `--port` must be `serve_port` and its `--prefill`/`--decode` URLs exactly the engines'. That host is the master: the bench client dials it |
| `serve_port`, `dist_init_port` | optional; 30890, 30531 |
| `state` | optional; where the pin is written, default `$shared_fs/mn_state.json` |
| `ssh` | optional; `{key, port, user, known_hosts, port_offsets, prefix}`, all optional; a node's own `ssh` overrides it. `port` defaults to 22; `port_offsets` `{"decode": 10}` adds to it per PD engine; no `known_hosts` = `accept-new`; `prefix` wraps the remote `bash -s`, e.g. `sudo -n docker exec -i <container>` |

Either every node names a `launch_command` (provided, the Hyperloom form) or none
does (generated, the prompt form). Each node must accept a non-interactive ssh
from the driver (with the `ssh` block's key and `known_hosts` at those paths on the
driver) and have `python3`; check ships its probe over that ssh, so no GEAK file has
to exist on a node. The driver needs no bench client (rank0 runs it); the driver and
rank0 must reach the client / load-balancer port. Without `expected_gfx`/`expected_target` in the handoff, `run_e2e.py` reads the GPU
identity from every node over the same ssh and refuses nodes that disagree; check
refuses them too. A pinned node is `hostname/boot_id`, so an IP that now
answers from a recreated pod is refused rather than substituted. The session
inherits pid 1's environment for every variable sshd left unset.

```json
"multinode": {"topology": "pd_disagg", "shared_fs": "/shared/geak", "backend": "sglang",
              "serve_port": 8000, "ssh": {"user": "geak", "prefix": "sudo -n docker exec -i sglang"},
              "nodes": [{"host": "10.0.0.1", "role": "prefill", "launch_command": "python3 -m sglang.launch_server --model-path /m --port 8000 --tp-size 8 --disaggregation-mode prefill ..."},
                        {"host": "10.0.0.2", "role": "decode", "launch_command": "python3 -m sglang.launch_server --model-path /m --port 8000 --tp-size 8 --disaggregation-mode decode ..."}],
              "router": {"host": "10.0.0.3", "launch_command": "python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://10.0.0.1:8000 8998 --decode http://10.0.0.2:8000 --port 8000"}}
```

The top-level `tp` is sglang's `--tp-size`: one engine's card count. dp-attention
and EP ride in the server args and do not change it, so dp2 tp8 ep16 on two
nodes is `tp: 16`. Plain DP replicas across nodes (a router in front of N
independent non-PD engines) is not supported; PD replicas (xPyD) are.

## Running it

### 1. The environment check (automatic, before Setup)

Every workflow invocation with a `multinode` block has `mn.py check` run on the
driver before anything else: `GEAK_MN_SPEC` is the block plus `tp` and `work_dir`, and
`mn.normalize_spec` is the one parser of it. `run_e2e.py` runs it before starting the
workflow and passes the verdict as `args.multinode_check` (a refusal never starts
Claude). A workflow started from a prompt gets no verdict, so before Setup it has one
agent run the same command verbatim (`check --owner claude`; a Workflow script cannot
spawn a process) and refuses a relayed verdict that names other machines than
`args.multinode`. The check:

- refuses a driver that sees a GPU, and a `work_dir` outside `shared_fs`;
- probes each node over ssh (the probe goes by content) and records its identity, its NIC for that IP (generated
  mode), its software stack and its GPU (gfx, target, product, CU count); nodes on
  different sglang/vllm/torch/triton/aiter versions, or with different GPUs, are refused;
- provided mode: fingerprints every launch script on its node and refuses a missing one;
- opens every port GEAK hands out that one node dials on another (generated: the
  dist-init port, PD's bootstrap port) on its node and dials it from the others, and
  dials from the driver the client port (`tp_span`), or the load balancer's and each PD
  engine's HTTP port; an unreachable one is refused;
- refuses a node whose fullest card has less than `MEM_FRACTION` (default 0.90) of
  its VRAM free, and a node with any of the topology's ports already bound;
- derives each rank's engine TP and card count (`mn.rank_layout`: `tp / nodes`
  per node on `tp_span`, the role's TP on `pd_disagg`) and refuses a `tp` that
  does not split evenly;
- refuses an IP no node has, two IPs on one machine, too few cards, or a
  `shared_fs` that fails a bidirectional marker exchange (`df` showing the same
  size proves nothing);
- prints, without refusing, GPU-resident processes (the VRAM rule above decides).

Its verdict and every reason go to `mn_check.json` beside the pin, and its last
stdout line is `MN_CHECK={verdict, problems, at, spec}`. On a refusal `run_e2e.py`
(or, from a prompt, the workflow) returns at once with `multinode_check`, copied into
`result.json` (`multinode_check`, and the `error` text), so the caller learns why the
machines were rejected. Each launch re-reads the software stack after its teardown
and refuses one that drifted from the pin, and `mn_launch.json` records
`pin_checked_at` and each rank's loaded `overlay_digest`.

On success it writes `$shared_fs/mn_state.json` (the pin: node ids, NICs, master,
ports, per-rank `tp`/`gpus`, and `role` under PD). Launches read each rank's TP and
cards from it; nothing else re-derives them. Every role prompt then carries
`BENCH_LAUNCHER=multinode GEAK_MN_STATE=<pin>` on its SERVING env line, so every leg
reads the same pin and the baseline and the candidate run on the same cards. The
workflow's returned `state` carries the parsed block, so a phase-by-phase resume
that omits `args.multinode` checks and serves on the same machines.

### 2. Run the benchmark

The command that produced the numbers above:

```bash
cd e2e_workflow/scripts
GEAK_MN_SPEC='{"topology": "tp_span", "nodes": ["10.10.0.166", "10.10.0.32"],
               "shared_fs": "/shared/geak", "tp": 16, "work_dir": "/shared/geak/exp"}' \
ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 multi_node/mn.py check            # on the driver
mkdir -p /shared/geak/exp/bench && cp -r bench_e2e.sh adapters /shared/geak/exp/bench/
cd /shared/geak/exp/bench                 # the ranks source adapters/ from here, on the nodes
BACKEND=sglang \
MODEL=/apps/DeepSeek-V4-Flash-FP8 \
TP=16 GPU=0,1,2,3,4,5,6,7 \
MEM_FRACTION=0.90 \
BENCH_LAUNCHER=multinode GEAK_MN_STATE=/shared/geak/mn_state.json \
EXTRA_SERVER_ARGS="--dp 2 --enable-dp-attention --attention-backend dsv4 ..." \
EXTRA_ENV="SYNC_TOKEN_IDS_ACROSS_TP=1 ..." \
OUT_DIR=/shared/geak/exp/band_conc8 \
ISL=1024 OSL=1024 CONC=8 NUM_PROMPTS=40 \
GEAK_REPEAT_MODE=warm_server REPEATS=3 PROFILE=0 \
SKILL_DIR=/path/to/GEAK/e2e_workflow \
  bash ./bench_e2e.sh
```

`TP=16` is the world size across both boxes; `GPU=0,...,7` is the **per-node**
device list. Those two disagreeing is not a mistake: once the TP group spans
machines, the only meaning a device list can have is per-node, and it is what the
GPU allow-list is checked against and what the serving lock is keyed on.

### 3. Or run it under the workflow

Put the `multinode` block in the handoff (above), or pass it as
`args.multinode` to the Workflow tool. Per-rank TP and cards come from the pin;
`run_e2e.py` imports the same `mn.rank_layout` to validate the block early and
default `gpu_ids` to rank0's cards. Data-parallel width is an ordinary server flag: it
rides the normal server-args channel and the launcher never learns its spelling.

## Environment reference

Everything about the machines is in the handoff's `multinode` block, read once by
`mn.py check` and pinned; the variables below are the only ones the path reads.

| variable | default | meaning |
| --- | --- | --- |
| `GEAK_MN_SPEC` | — | `mn.py check` only: the `multinode` block plus `tp`, as JSON |
| `GEAK_MN_STATE` | the active pin | the pin; on every SERVING leg's env line |
| `GEAK_MN_PD_SWEEP_ARGS_PREFILL` / `GEAK_MN_PD_SWEEP_ARGS_DECODE` | — | pd_disagg only. Flags the sweep accepted for that engine alone. Appended after `EXTRA_SERVER_ARGS` on that engine; the other engine does not receive them |
| `GEAK_MN_PD_SWEEP_ENV_PREFILL` / `GEAK_MN_PD_SWEEP_ENV_DECODE` | — | pd_disagg only. Same split for env. Appended after `EXTRA_ENV` on that engine |
| `GEAK_MN_PD_OVERLAY_PREFILL` / `GEAK_MN_PD_OVERLAY_DECODE` | — | pd_disagg only. Kernel overlay accepted for that engine alone. Placed ahead of `OVERLAY_PYTHONPATH` on that engine; recorded in `mn_launch.json` as `pd.effective_args_by_role.<role>.overlay` |
| `GEAK_MN_ROCM_RUNTIME` | — | see [the ROCm runtime](#the-rocm-runtime-geak_mn_rocm_runtime) |
| `GEAK_MN_DRY_RUN` | — | `1` = render the per-rank scripts and stop, cluster untouched |
| `MEM_FRACTION` | 0.90 | check: the free share of the fullest card's VRAM a node must have |
| `SERVING_LOCK_WAIT` | 7200 | seconds a serving leg waits for a card a GEAK kernel task has leased |

The trace wait is fixed: bytes quiet for 30 s count as exported, an incomplete set is
given up after 180 s quiet, 1800 s at most, and the gate expects `TP` traces (under PD
the sum of the windowed engines' TP).

## Artifacts

Under `$OUT_DIR`:

| file | what to check |
| --- | --- |
| `bench_summary.json` | `throughput_tok_s_median`, `..._spread_pct`, `usable_for_acceptance` |
| `mn_launch.json` / `.log` | per-rank spawn record; `rc != 0` means no server was started |
| `mn_supervisor.json` | `verdict` (`torn_down` = ranks reaped), `cleared` |
| `server.log` / `server.rank1.log` | rank0's log is the one `bench_e2e.sh` greps for OOM/fatal |
| `profile/ranks/mn_trace_report.json` | the trace gate's verdict (see below) |

Under `shared_fs`: `mn_state.json` (the pin `mn.py check` wrote) and `mn_check.json`.

## The trace gate

With `PROFILE=1`, all N ranks export into the shared profile directory and the
gate decides whether that set can be believed. `verdict: pass` requires every
expected rank present and readable, **collective kernels in every rank** (a TP
group spanning machines that shows no collectives did not do what it claimed),
both nodes represented, and a trace for the selected rank (`--select-rank`, default 0),
which the report names as `parsed_trace` (`parsed_trace_by_role` for PD). A failed gate is non-fatal — the bench that came with it is still valid —
but the verdict is on disk so a Top-N built on a partial set is never mistaken
for one built on the whole server.

Two things it exists to prevent, both of which look like a successful capture:

- **Waiting for the file count returns too early.** torch creates each rank's
  file when the export starts and then streams ~25 MB into it, so the count
  reaches N while all N are still writing. Measured: three traces truncated
  mid-write, six at zero bytes, and a reported 62% node imbalance that did not
  exist. The completion signal is total bytes going quiet instead.
- **`ls -t | head -1` picks a random rank.** With N sibling traces the newest
  file is whichever rank's NFS write landed last, and two legs of an A/B would
  compare different ranks; a stray `.json` (the gate's own report, once) would be
  parsed *as* the trace. The profiler therefore parses `parsed_trace` from
  `ranks/mn_trace_report.json` and only falls back to `ls -t` on a single box.
  No trace is moved.

### The ROCm runtime (`GEAK_MN_ROCM_RUNTIME`)

On the ROCm 7.2.0 image the torch profiler has two problems with CUDA graphs:

- **The trace is almost empty.** roctracer drops the kernels a CUDA graph
  replays. Measured on Qwen3-8B TP8 decode: 6 kernels per replay instead of 725.
- **A `pd_disagg` decode rank can deadlock.** If its profiler window is open
  while KV is arriving, `hipGraphLaunch` can wedge. It happened in 2 of 3
  windows under load.

`GEAK_MN_ROCM_RUNTIME=<dir>` (opt-in) prepends `<dir>` to the `LD_LIBRARY_PATH`
of every engine, which loads a newer HIP / ROCr / roctracer set, e.g. 7.2.4
unpacked from repo.radeon.com. The directory must exist at the same path on
every pinned node, and a directory without `libamdhip64.so.7` fails the launch.

With 7.2.4, CUDA-graph kernels reach the trace (725 per replay), and 6 of 6
decode windows opened under load did not hang. A baseline and a candidate must
run on the same runtime to be comparable.

## When comparing two legs

- Same machines. A different pair measured ~13% apart under an identical
  protocol — larger than any flag result worth reporting. Compare the
  pins' `nodes` (`hostname/boot_id`).
- Beat the noise band by enough to mean it. Treat a difference not clearly
  larger than the reported spread as "add rounds", not as a result.

## Failure modes and what they print

| symptom | cause | what to do |
| --- | --- | --- |
| `no pinned node set at …` | the leg ran outside the workflow, or without its `GEAK_MN_STATE` | run `mn.py check` with `GEAK_MN_SPEC` |
| `<host> has no interface with this address` | the node answers ssh but the IP is not on its fabric NIC | fix the pod IPs (`MULTI_NODE_STATE`) / `multinode.nodes` |
| `probe on <ip> returned nothing` | ssh failed, or no `python3` on that node | fix the ssh block / the node's image |
| `this machine's environment sees N GPU(s)` | the driver was started with its cards visible | start it with `ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=` |
| `this machine (…) cannot reach the … port` | the driver's client could not dial the client / LB / engine port | open the port to the driver, or run the driver on the fabric |
| `the run directory … is not under` | `work_dir` (`eval_dir` / `exp_root`) is off `shared_fs` | put the run directory on the shared mount |
| `handoff.multinode is no longer read` | an old handoff names its machines | pass them as `MULTI_NODE_STATE` |
| `TP X is not the engine TP Y`, `the router names … the engines serve …`, `every … replica must run one command` | a provided command disagrees with the layout, the router, or its role's other replicas | fix the `launch_command`s |
| `A multinode run drives its nodes from this machine …` | `bench_e2e.sh` with another launcher on the driver while a pin is active | drop `BENCH_LAUNCHER`, or start the isolated server through `mn.py exec --gpu` |
| `On a pinned node … may only start an isolated one-card server` | through `mn.py exec`, a leg that is not `BENCH_LAUNCHER=native TP=1` on one card | serve the whole server from the driver instead |
| `launch through the eval_dir copy of bench_e2e.sh` | a serving leg from a `bench_e2e.sh` whose `adapters/` is not under `shared_fs` | run the eval dir's copy |
| `the multinode pin … has no tools_dir` | the pin was written by a check without `work_dir` | re-run check through the workflow, or with `work_dir` in `GEAK_MN_SPEC` |
| `the pinned machines have different GPUs` | the nodes report different gfx/product | name matching machines, or pass `expected_gfx`/`expected_target` |
| `cannot share …` | the mount is local, or read-only from one side | fix the mount; do not proceed |
| `the handoff's nodes did not come clear` | a previous scheduler still holds the dist-init port | wait, or name other machines |
| `the supervisor exited immediately` | a bad state file | see `mn_supervisor.log`; the launcher reaps the ranks it spawned |
| `N/M rank trace(s) landed` with N<M | a rank died before exporting | check that rank's `server.rank*.log` |
| `mn.py check rejected args.multinode` | the check failed; `multinode_check.problems` says why | fix the nodes / the mount |
| a rank hangs on the first collective | `NCCL_SOCKET_IFNAME` unset, so RCCL picked calico/docker0/veth | the rank script exports it from the pin; a hand-rolled `torchrun` has no such protection |

## Tests

```bash
cd e2e_workflow/scripts/tests
python3 -m unittest test_mn
```

The gate and window tests run without a cluster: they fabricate rank traces,
including a streaming writer that drips gzip bytes over several seconds to prove
the wait does not return early.
