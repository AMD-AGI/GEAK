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

## What the caller must hand over

GEAK selects no machines and does not create, resize or tear down the cluster.

**Through `run_e2e.py` (Hyperloom)** the cluster is the JSON file `MULTI_NODE_STATE`
names, a `node_state.json`; the handoff carries no machine information.
`run_e2e.py` translates it into the `multinode` block
below (provided mode) and nothing else reads it:

| `node_state.json` | becomes |
| --- | --- |
| `backend` | must equal the handoff's `framework` (`sglangrouter` counts as `sglang`); picks the command parser |
| `pd_mode` | `disaggregated` → `pd_disagg`, else `tp_span` |
| `pods[]` (`role`, `podIP`, `launch_command`, `sshPort`/`sshUser`) | one node per engine pod, one pod = one replica (xPyD: `replicas_by_role` counts them); the `router` pod is `router`, and may hold no GPU. One already serving (`managed_by: "platform"`) is stopped and relaunched from its `launch_command` with every leg's engines, so its worker health starts fresh |
| router `launch_command`'s worker URLs | each names one engine pod of that role and port: its `podIP`, a name resolving to it, or a name resolving to no pod (a k8s Service) paired by role and port. A name that does not resolve on the driver is refused |
| `gpus_per_node` | `tp_by_role` (each engine's TP; its command's `--tp-size` must agree) |
| `pods[].gpuIds` (optional, e.g. `"4,5,6,7"`) | the node's `gpu_ids`: the cards its engine serves from, instead of `0..TP-1`. Two engine pods may share a `podIP` (1P1D on one machine) only if both name disjoint `gpuIds` and different `--port`s |
| `service_url` | `serve_port`; an IP must be the router pod's (else the first engine's); a name gives only the port, and the client dials the router pod |
| `ssh_user` (default `root`), `ssh_port`, `ssh_key_path`, `ssh_known_hosts`, `ssh_prefix` | the `ssh` block |
| `user_data_path` | `shared_fs`: one directory every pod and the driver mount at the same path. The pin is `$user_data_path/pins/mn_state_<hash of the pod IPs>.json` |
| `profile_trace`, `steady_state_trace` | TraceLens's first-round traces: one `<podIP>_<role>/` dir per engine pod; every pod's rank0 trace is annotated as one union (`trace_files`, `steady_state_trace` first). No by-role Top-N is written for this first round; GEAK's own re-profiles split by role as usual |
| `external` | ignored |

**Directly to `e2e_workflow.js`** (the interactive entry) the caller passes the
`multinode` block itself as `args.multinode`, generated or provided. Two things `run_e2e.py`
does are then the caller's: Claude starts on the driver with
`ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=` (check refuses a driver
that sees a GPU), and, as on one host, the caller reads `expected_gfx` / `expected_target`
before the call, here on a node since the driver sees no GPU (`run_e2e.py` runs the
repo root's `scripts/gpu_identity.py` on every node).

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
| `ssh` | optional; `{key, port, user, known_hosts, prefix}`, all optional; a node's own `ssh` overrides it. `port` defaults to 22; no `known_hosts` = `accept-new`; `prefix` wraps the remote `bash -s`, e.g. `sudo -n docker exec -i <container>` |

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
Claude). Both that verdict's `spec` and `args.multinode` then go into the prompt without
their `launch_command`s, marked `commands_in_pin`: the main session's relay of the prompt
has dropped the commands' shell quoting (and whole fields), and nothing in the workflow
reads them, as every leg serves through the pin. If the verdict still arrives unusable,
the workflow refuses instead of re-checking. A workflow started from a prompt gets no verdict, so before Setup it has one
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
machines were rejected. As on a single node, the caller keeps the machines, the
software stack and the overlay unchanged between legs; launch does not re-verify them.

On success it writes `$shared_fs/mn_state.json` (the pin: node ids, NICs, master,
ports, per-rank `tp`/`gpus`, and `role` under PD). Launches read each rank's TP and
cards from it; nothing else re-derives them. Every role prompt then carries
`BENCH_LAUNCHER=multinode GEAK_MN_STATE=<pin>` on its SERVING env line, so every leg
reads the same pin and the baseline and the candidate run on the same cards. The
workflow's returned `state` carries the parsed block, so a phase-by-phase resume
that omits `args.multinode` checks and serves on the same machines.

