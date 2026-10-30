"""Serving on the GPU machines the caller names. GEAK runs on the driver, which sees no
GPU and reaches every node over ssh; the nodes carry no GEAK code. What a node runs is
either shipped by content here or a run artifact on the shared mount.

    mn.py check        probe the handoff's nodes (GEAK_MN_SPEC), prove the shared mount, write the pin
    mn.py launch       tear down the last launch, build or wrap one script per rank, ship it
    mn.py supervise    the local pid bench_e2e.sh watches and kills
    mn.py stop         reap a launch whose supervisor never came up
    mn.py leg-env      the pin's facts as shell assignments for multinode.sh
    mn.py active-pin   the pin of the multinode run on this machine ('' = none)
    mn.py exec         run a command on a pinned node
    mn.py digest       fingerprint an overlay tree (a rank refuses to start on another one)
    mn.py trace-gate   validate the per-rank trace set before parsing    (mn_traces.py)
    mn.py topn-merge   fold a pd_disagg run's per-engine Top-Ns into one (mn_traces.py)
    mn.py pd-mass      rewrite an AgentX pd_disagg ranking on both engines' GPU-time mass (mn_traces.py)
    mn.py pd-points    plan a fixed-length pd_disagg run's sampling points from its engine logs (mn_traces.py)
    mn.py probe        (internal) this node's probe

GEAK does not manage the cluster. A rank's script is either GENERATED here (each
rank sources adapters/$BACKEND.sh on its node) or PROVIDED by the caller (nodes[i].
launch_command, wrapped into a Magpie server-phase script with GEAK's layer on top).
adapters/launchers/multinode.sh is the bench_e2e.sh-facing side.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

_LOG_TAG = "mn"
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{_LOG_TAG} {stamp}] {msg}", flush=True)


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short(hostname: str) -> str:
    return hostname.split(".")[0]


# --------------------------------------------------------------------------
# CLUSTER
# --------------------------------------------------------------------------
# ssh only places processes on the named machines; sglang forms its own
# torch.distributed world. Every A/B leg runs on the same pin.

# Bracket classes keep pgrep from matching the `bash -c` that carries the pattern.
SERVE_PATTERN = "(sglang[ ]serve|sglang[.]launch_server|vllm[ ]serve)"
SCHED_PATTERN = "sglang::[s]cheduler"
# Engine children rename themselves (sglang::scheduler, VLLM::Worker_TP3, VLLM::EngineCore)
# and can outlive the serve process, holding the cards.
ENGINE_PATTERN = "(sglang|VLL[M])::"
# Without setproctitle the same children keep their multiprocessing argv; orphaned
# (ppid 1) they hold the cards and the dist-init port after the serve process is gone.
WORKER_PATTERN = "multiprocessing[.](spawn import spawn_main|resource_tracker import main)"
# A Ray head's dashboard children carry the same argv; they are the cluster's, not ours.
OUR_WORKER_PIDS = (
    f"for p in $(pgrep -f '{WORKER_PATTERN}'); do pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); "
    "cat /proc/$p/cmdline /proc/${pp:-0}/cmdline 2>/dev/null | tr '\\0' ' ' | grep -qE 'ray[/-]dashboard' || echo $p; done")
# Separate from SERVE_PATTERN: the LB holds no GPU. The router rewrites its argv
# to "sglang::router"; the mini-LB keeps its module path. No literal "sglang::" here:
# ENGINE_PATTERN's pkill would match the `bash -c` that carries this pattern and kill it.
LB_PATTERN = "(sglang:[:]router|sglang_router[.]mini_lb)"

# A rank's server pid, recorded on its own node (a pid means nothing elsewhere) with the
# process's start time, so a pid the kernel has since handed to another process is not ours.
PID_DIR = "/tmp/geak_mn"
_START_TIME = 'sed "s/.*) //" /proc/{pid}/stat 2>/dev/null | cut -d" " -f20'


def record_pid(rank, pid: str = "$SERVER_PID") -> str:
    return f'mkdir -p {PID_DIR} && echo "{pid} $({_START_TIME.format(pid=pid)})" > {PID_DIR}/rank{rank}.pid'


def live_pid(rank) -> str:
    """Shell: the recorded pid if that very process runs, `gone` if not, nothing if unrecorded."""
    return (f"if read p st < {PID_DIR}/rank{rank}.pid 2>/dev/null; then "
            f'if [ -n "$st" ] && [ "$({_START_TIME.format(pid="$p")})" = "$st" ]; then echo $p; else echo gone; fi; fi')


def serve_alive(rank) -> str:
    """Shell: `serve=<n>` by the pid file; the process name only when no pid was recorded."""
    return (f'p=$({live_pid(rank)}); case "$p" in gone) s=0 ;; "") s=$(pgrep -c -f \'{SERVE_PATTERN}\' || true) ;; *) s=1 ;; esac; '
            "echo serve=$s pid=${p:-none}")

TOPOLOGIES = ("tp_span", "pd_disagg")
PD_ROLES = ("prefill", "decode")


def state_path() -> str:
    path = active_pin()
    if not path:
        raise SystemExit("!!! multinode: no pin: GEAK_MN_STATE is unset and no multinode run on this machine owns one.")
    return path


def load_state() -> dict:
    path = state_path()
    if not os.path.isfile(path):
        raise SystemExit(f"!!! multinode: no pinned node set at {path}.\n    The workflow's Setup writes it from args.multinode (mn.py check).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, doc: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)
    return path


def topology_of(state: dict) -> str:
    """Read from the PIN: changing the topology means checking again."""
    topo = state.get("topology") or "tp_span"
    if topo not in TOPOLOGIES:
        raise SystemExit(
            f"!!! multinode: unknown topology {topo!r}; known: {', '.join(TOPOLOGIES)}")
    return topo


COTENANT_REFUSAL = (
    "!!! multinode: the handoff's nodes did not come clear. Launching now would\n    hit the held dist-init port and produce an orphaned rank1. Free them or\n    name other machines in handoff.multinode.nodes.")


PD_SPEC_KEYS = ("backend", "ib_device", "lb")
PD_LBS = ("router", "minilb")

# What a caller's engine launch_command says about the deployment, per backend: check
# reconciles it with the handoff, launch strips these flags from GEAK's swept ones (the
# caller's command owns them). --port is read from every command.
ENGINE_CMD_FACTS = {
    "sglang": {"model": ("--model-path", "--model"), "tp": ("--tp-size", "--tp"),
               "role": ("--disaggregation-mode",), "bootstrap": ("--disaggregation-bootstrap-port",),
               "other": ("--host", "--disaggregation-transfer-backend", "--disaggregation-ib-device",
                         "--dist-init-addr", "--nnodes", "--node-rank")},
    "vllm": {"model": ("--model",), "tp": ("--tensor-parallel-size", "-tp"),
             "role": ("--kv-transfer-config",), "bootstrap": (),
             "other": ("--host", "--pipeline-parallel-size", "-pp")},
}
SGLANG_BOOTSTRAP_PORT = 8998
VLLM_KV_ROLES = {"kv_producer": "prefill", "kv_consumer": "decode"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def split_command(command: str) -> tuple[list[str], list[str]]:
    """(leading NAME=VALUE env, argv) of a shell command line."""
    tokens = shlex.split(command)
    i = 0
    while i < len(tokens) and _ASSIGN.match(tokens[i]):
        i += 1
    return tokens[:i], tokens[i:]


def flag_value(argv: list, names) -> str | None:
    """The last value given for any of `names` (`--x v` or `--x=v`): argparse is last-wins."""
    val = None
    for i, tok in enumerate(argv):
        key, eq, inline = tok.partition("=")
        if key in names:
            val = inline if eq else (argv[i + 1] if i + 1 < len(argv) else "")
    return val


def structural_flags(backend: str) -> set:
    facts = ENGINE_CMD_FACTS[backend]
    return {"--port", *facts["model"], *facts["tp"], *facts["role"], *facts["bootstrap"], *facts["other"]}


def strip_flags(args: str, flags: set) -> tuple[str, list[str]]:
    """(args without `flags` and their values, what was dropped). Unchanged text when nothing is."""
    tokens, kept, dropped, i = shlex.split(args or ""), [], [], 0
    while i < len(tokens):
        key, eq, _ = tokens[i].partition("=")
        if key not in flags:
            kept.append(tokens[i])
            i += 1
            continue
        take = 1 if eq or i + 1 >= len(tokens) or tokens[i + 1].startswith("-") else 2
        dropped.append(" ".join(tokens[i:i + take]))
        i += take
    return (shlex.join(kept) if dropped else (args or "").strip()), dropped


def engine_facts(backend: str, command: str) -> dict:
    """Model, TP, port, PD role and bootstrap port an engine command serves with. Raises ValueError."""
    if backend not in ENGINE_CMD_FACTS:
        raise ValueError(f"a launch_command needs backend sglang or vllm, got {backend!r}")
    facts = ENGINE_CMD_FACTS[backend]
    env, argv = split_command(command)
    owned = sorted({kv.split("=", 1)[0] for kv in env} & _OWNED)
    if owned:
        raise ValueError(f"launch_command sets {owned}, which the launcher owns per run")
    model = flag_value(argv, facts["model"])
    if backend == "vllm" and not model and "serve" in argv:
        nxt = argv.index("serve") + 1
        model = argv[nxt] if nxt < len(argv) and not argv[nxt].startswith("-") else None
    role = flag_value(argv, facts["role"])
    if backend == "vllm" and role:
        try:
            role = VLLM_KV_ROLES.get(json.loads(role).get("kv_role"))
        except (ValueError, AttributeError):
            raise ValueError(f"--kv-transfer-config is not JSON: {role!r}") from None
    port = flag_value(argv, ("--port",))
    if not (port or "").isdigit():
        raise ValueError(f"launch_command names no --port: {command[:120]!r}")
    tp = flag_value(argv, facts["tp"]) or "1"
    boot = flag_value(argv, facts["bootstrap"]) if facts["bootstrap"] else None
    if backend == "sglang" and role == "prefill" and not boot:
        boot = str(SGLANG_BOOTSTRAP_PORT)
    return {"env": env, "argv": argv, "model": model, "tp": int(tp), "port": int(port),
            "role": role or "", "bootstrap_port": int(boot) if boot else None}


def _same_engine(backend: str, command: str) -> tuple:
    """A command with its --host/--port taken out: replicas of one role must agree on the rest."""
    env, argv = split_command(command)
    return tuple(env), tuple(shlex.split(strip_flags(shlex.join(argv), {"--host", "--port"})[0]))


def url_tokens(command: str) -> set:
    """The http(s)://host:port worker addresses a router command names."""
    return {t.rstrip("/") for t in shlex.split(command) if re.match(r"^https?://", t)}


def normalize_spec(block: dict, tp) -> dict:
    """handoff.multinode plus the engine TP, validated; the one place the block is parsed.
    check pins the result and run_e2e.py reads the master from it. Raises ValueError.

    A node is a fabric IP/hostname or {host, rank?, role?, ssh?, launch_command?}. An explicit
    rank orders them; pd_disagg with a role on every node puts the prefill machines first.
    Provided commands (every node's launch_command, plus a router {host, launch_command, ssh?}
    that may be a machine of its own) fix the ports and are reconciled with the handoff."""
    if not isinstance(block, dict):
        raise ValueError("multinode must be an object")
    topology = str(block.get("topology") or "tp_span").strip()
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; known: {', '.join(TOPOLOGIES)}")
    shared = str(block.get("shared_fs") or "").strip().rstrip("/")
    if not shared:
        raise ValueError("names no shared_fs. Every node must mount it: without it the authored overlay "
                         "reaches only the master, the other ranks serve stock code, and the run still reports a speedup")
    if not isinstance(block.get("nodes"), list) or not block["nodes"]:
        raise ValueError("nodes must name at least one machine")
    nodes = []
    for i, n in enumerate(block["nodes"]):
        d = n if isinstance(n, dict) else {"host": n}
        host = str(d.get("host") or d.get("ip") or "").strip()
        if not host:
            raise ValueError(f"nodes[{i}] names no host")
        if d.get("launch_script"):
            raise ValueError(f"nodes[{i}]: launch_script is no longer read; give the node's launch_command")
        node = {"host": host, "rank": int(d["rank"]) if d.get("rank") is not None else i}
        node.update({k: str(d[k]).strip() for k in ("role", "launch_command") if d.get(k)})
        if d.get("ssh"):
            node["ssh"] = ssh_config(d["ssh"])
        if d.get("gpu_ids") not in (None, "", []):
            ids = d["gpu_ids"].split(",") if isinstance(d["gpu_ids"], str) else d["gpu_ids"]
            try:
                node["gpu_ids"] = [int(g) for g in ids]
            except (TypeError, ValueError):
                raise ValueError(f"nodes[{i}]: gpu_ids must be card ids such as 0,1,2,3, got {d['gpu_ids']!r}") from None
        nodes.append(node)
    nodes.sort(key=lambda d: d.pop("rank"))
    roles = [d.get("role") for d in nodes]
    nodes_by_role = block.get("nodes_by_role") or {}
    tp_by_role = block.get("tp_by_role") or {}
    replicas = block.get("replicas_by_role") or {}
    for key, val in (("tp_by_role", tp_by_role), ("nodes_by_role", nodes_by_role),
                     ("replicas_by_role", replicas)):
        if not isinstance(val, dict) or set(val) - set(PD_ROLES):
            raise ValueError(f'{key} must be {{"prefill": N, "decode": N}}, got {val!r}')
    replicas = {r: int(replicas.get(r) or 1) for r in PD_ROLES}
    if topology == "pd_disagg" and all(roles):
        bad = sorted(set(roles) - set(PD_ROLES))
        if bad:
            raise ValueError(f"nodes: unknown role(s) {bad}")
        nodes.sort(key=lambda d: PD_ROLES.index(d["role"]))
        uneven = [r for r in PD_ROLES if roles.count(r) % replicas[r]]
        if uneven:
            raise ValueError(f"replicas_by_role {replicas} does not split the {uneven} nodes evenly")
        nodes_by_role = {r: roles.count(r) // replicas[r] for r in PD_ROLES if r in roles}
    given = [bool(d.get("launch_command")) for d in nodes]
    if any(given) and not all(given):
        raise ValueError("either every node names its launch_command or none does")
    router = block.get("router") or {}
    if router:
        if not (isinstance(router, dict) and router.get("host") and router.get("launch_command")):
            raise ValueError('router must be {"host": ..., "launch_command": ...}')
        router_ssh = router.get("ssh")
        router = {k: str(router[k]).strip() for k in ("host", "launch_command")}
        if router_ssh:
            router["ssh"] = ssh_config(router_ssh)
        if not (all(given) and topology == "pd_disagg"):
            raise ValueError("a router launch_command goes with provided pd_disagg engine commands")
    pd = block.get("pd") or {}
    if not isinstance(pd, dict) or set(pd) - set(PD_SPEC_KEYS):
        raise ValueError(f"pd takes {', '.join(PD_SPEC_KEYS)}, got {pd!r}")
    pd = {k: str(v).strip() for k, v in pd.items() if v is not None}
    if pd.get("lb", "router") not in PD_LBS:
        raise ValueError(f"pd.lb must be one of {', '.join(PD_LBS)}, got {pd['lb']!r}")
    tp = int(tp or 1)
    layout = rank_layout(topology, tp, len(nodes), tp_by_role, nodes_by_role,
                         replicas if topology == "pd_disagg" else None)
    for node, lay in zip(nodes, layout):
        ids = node.get("gpu_ids")
        if ids is not None and (len(ids) != lay["gpus"] or len(set(ids)) != len(ids)):
            raise ValueError(f"{node['host']}: gpu_ids {ids} are not the {lay['gpus']} "
                             "distinct cards its engine serves from there")
    spec = {
        "topology": topology, "nodes": nodes, "shared_fs": shared,
        "work_dir": str(block.get("work_dir") or "").strip().rstrip("/"),
        "state": str(block.get("state") or "").strip() or f"{shared}/mn_state.json",
        "serve_port": int(block.get("serve_port") or 30890),
        "dist_init_port": int(block.get("dist_init_port") or 30531),
        "ssh": ssh_config(block.get("ssh") or {}),
        "router": router, "pd": pd, "tp": tp, "layout": layout,
    }
    if topology == "pd_disagg":
        spec["tp_by_role"] = {r["role"]: r["tp"] for r in layout}
        spec["nodes_by_role"] = {r: sum(1 for x in layout if x["role"] == r and not x.get("replica"))
                                 for r in PD_ROLES}
        if any(v > 1 for v in replicas.values()):
            spec["replicas_by_role"] = replicas
    # The KB --topology: TP16 over two nodes or a PD split is not one host at the same tp.
    tpr = spec.get("tp_by_role") or {}
    spec["kb_topology"] = (f"pd_p{tpr['prefill']}x{replicas['prefill']}_d{tpr['decode']}x{replicas['decode']}"
                           if topology == "pd_disagg" else f"tpspan_n{len(nodes)}" if len(nodes) > 1 else "")
    for key in ("backend", "model"):
        if block.get(key):
            spec[key] = str(block[key]).strip().rstrip("/")
    if all(given):
        reconcile_commands(spec)
    return spec


def reconcile_commands(spec: dict) -> None:
    """Provided commands against the handoff: model, engine TP, PD role, one command per role
    (apart from --host/--port), and a router that names exactly the engines. Records each
    node's port (and prefill bootstrap port) and the router's. Raises ValueError."""
    backend, model = spec.get("backend") or "", spec.get("model") or ""
    pd = spec["topology"] == "pd_disagg"
    first = {}
    for i, (node, lay) in enumerate(zip(spec["nodes"], spec["layout"])):
        facts = engine_facts(backend, node["launch_command"])
        name = f"nodes[{i}] {node['host']}"
        if model and (facts["model"] or "").rstrip("/") != model:
            raise ValueError(f"{name} serves {facts['model']!r}; the handoff's model is {model!r}")
        if facts["tp"] != lay["tp"]:
            raise ValueError(f"{name}: its command's TP {facts['tp']} is not the engine TP {lay['tp']}")
        if pd and facts["role"] != lay["role"]:
            raise ValueError(f"{name} is a {lay['role']} node but its command serves {facts['role'] or 'no PD role'!r}")
        node["port"] = facts["port"]
        if facts["bootstrap_port"]:
            node["bootstrap_port"] = facts["bootstrap_port"]
        if pd and not lay.get("engine_rank"):
            same = first.setdefault(lay["role"], _same_engine(backend, node["launch_command"]))
            if same != _same_engine(backend, node["launch_command"]):
                raise ValueError(f"{name}: every {lay['role']} replica must run one command (apart from --host/--port)")
    router = spec["router"]
    if pd and not router:
        raise ValueError("provided pd_disagg engine commands need the router's launch_command")
    if not pd and spec["nodes"][0]["port"] != spec["serve_port"]:
        raise ValueError(f"rank0's command serves --port {spec['nodes'][0]['port']}, the service is on {spec['serve_port']}")
    if router:
        port = flag_value(split_command(router["launch_command"])[1], ("--port",))
        if port != str(spec["serve_port"]):
            raise ValueError(f"the router listens on --port {port}, the service is on {spec['serve_port']}")
        want = {f"http://{n['host']}:{n['port']}" for n, lay in zip(spec["nodes"], spec["layout"])
                if not lay.get("engine_rank")}
        got = url_tokens(router["launch_command"])
        if got != want:
            raise ValueError(f"the router names {sorted(got)}; the engines serve {sorted(want)}")
        router["port"] = spec["serve_port"]


def spec_targets(spec: dict) -> list[dict]:
    """The handoff's hosts as transport nodes, before any is pinned (no node_id yet)."""
    return [{"node_id": "", "rank": r, "ip": n["host"],
             **({"role": spec["layout"][r]["role"]} if spec["layout"][r].get("role") else {}),
             **({"ssh": n["ssh"]} if n.get("ssh") else {})}
            for r, n in enumerate(spec["nodes"])]


def spec_master(spec: dict) -> dict:
    """The router's host, else rank0: the node that serves the client port."""
    host = (spec.get("router") or {}).get("host") or spec["nodes"][0]["host"]
    return next((t for t in spec_targets(spec) if t["ip"] == host),
                {"node_id": "", "rank": "lb", "ip": host, "role": "router"})


def rank_layout(topology: str, tp: int, nnodes: int, tp_by_role: dict | None = None,
                nodes_by_role: dict | None = None, replicas_by_role: dict | None = None) -> list:
    """Per rank: the engine's TP and the cards that node serves. The one place this is
    derived: check writes it into the pin, launch reads it back, run_e2e imports it.

    pd_disagg: nodes are listed engine by engine (prefill first), replica by replica
    (xPyD: identical engines of one role, `replica` 1.. on all but the first). An engine
    on more than one node is one TP world across them; its ranks carry
    engine_rank / engine_nnodes for the dist-init triple."""
    if topology == "pd_disagg":
        tps = [int((tp_by_role or {}).get(role) or tp) for role in PD_ROLES]
        counts = [int((nodes_by_role or {}).get(role) or 1) for role in PD_ROLES]
        reps = [int((replicas_by_role or {}).get(role) or 1) for role in PD_ROLES]
        need = sum(c * r for c, r in zip(counts, reps))
        if nnodes != need or min(tps) < 1 or min(counts) < 1 or min(reps) < 1:
            raise ValueError(f"pd_disagg needs {need} nodes (per engine {dict(zip(PD_ROLES, counts))}, replicas {dict(zip(PD_ROLES, reps))}) and a TP per engine, got {nnodes} node(s), TP {dict(zip(PD_ROLES, tps))}")
        out = []
        for role, t, c, r in zip(PD_ROLES, tps, counts, reps):
            if t % c:
                raise ValueError(f"the {role} engine's TP={t} does not split evenly over {c} nodes")
            for rep in range(r):
                for i in range(c):
                    entry = {"role": role, "tp": t, "gpus": t // c}
                    if c > 1:
                        entry.update(engine_rank=i, engine_nnodes=c)
                    if rep:
                        entry["replica"] = rep
                    out.append(entry)
        return out
    if tp < nnodes or tp % nnodes:
        raise ValueError(f"TP={tp} does not split evenly over {nnodes} nodes")
    return [{"tp": tp, "gpus": tp // nnodes} for _ in range(nnodes)]


def pd_config(state: dict) -> dict:
    """The PD ports, all from serve_port, and the transfer settings check pinned from
    handoff.multinode.pd. The LB owns serve_port, so the client side needs no notion
    of PD; engines sit above it."""
    base = int(state.get("serve_port", 30890))
    pd = state.get("pd") or {}
    return {
        "roles": list(PD_ROLES),
        "lb_port": base,
        "role_ports": [base + 1 + i for i in range(len(PD_ROLES))],
        "bootstrap_port": base + 8,
        # No verbs provider on this fleet, so the RDMA backends have nothing to bind.
        "transfer_backend": pd.get("backend") or "mooncake_tcp",
        "ib_device": pd.get("ib_device") or "",
        "lb": pd.get("lb") or "router",
    }


def node_role(state: dict, node: dict) -> str:
    return node.get("role") or ""


def engine_nodes(state: dict, role: str) -> list:
    """The pinned nodes of one PD engine, its node-rank 0 (HTTP front end) first."""
    return [n for n in state["nodes"] if node_role(state, n) == role]


def engine_urls(state: dict) -> list[dict]:
    """One entry per PD engine replica, at its node-rank 0 (the other nodes of a spanning
    engine serve no HTTP): the only place an engine's address is derived. A provided
    command's own --port wins over the port derived from serve_port."""
    pd = pd_config(state)
    out = []
    for i, role in enumerate(pd["roles"]):
        nodes = engine_nodes(state, role)
        for rep in sorted({n.get("replica", 0) for n in nodes}):
            en = [n for n in nodes if n.get("replica", 0) == rep]
            port = en[0].get("port") or pd["role_ports"][i]
            out.append({"role": role, "replica": rep, "node": en[0], "port": port,
                        "url": f"http://{en[0]['ip']}:{port}", "nodes": en})
    return out


def lb_node(state: dict) -> dict:
    """Where the PD front end runs: the caller's router machine, else the master node."""
    return state.get("router") if (state.get("router") or {}).get("ip") else \
        next((n for n in state["nodes"] if n["ip"] == state.get("master_ip")), None)


def state_targets(state: dict) -> list[dict]:
    """Every machine a launch touches: the nodes, then a provided router's own machine."""
    router = state.get("router") or {}
    return [*state["nodes"], *([router] if router.get("node_id") else [])]


def _resident_gpu_pids(showpids: str) -> tuple[list[str], int]:
    """(pids holding VRAM, count of KFD pids with 0 VRAM) from `rocm-smi --showpids`.

    The host's gpuagent daemon holds a 0-VRAM KFD context; it is not occupancy."""
    holders, parked = [], 0
    for pid, vram in re.findall(r"^\s*(\d+)\s+\S+\s+\d+\s+(\d+)", showpids, re.M):
        if int(vram) > 0:
            holders.append(pid)
        else:
            parked += 1
    return holders, parked


SW_PACKAGES = ("sglang", "vllm", "torch", "triton", "amd-aiter")


def _versions() -> dict:
    """The stack a rank would load; ranks that disagree serve different code."""
    from importlib import metadata
    out = {}
    for pkg in SW_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def probe_node(ports: list | None = None) -> dict:
    """Runs ON the node (`mn.py probe`)."""
    def sh(cmd, timeout=90):
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout).stdout

    use = sh("rocm-smi --showuse 2>/dev/null || true")
    uses = [float(m.group(1)) for m in re.finditer(r"GPU use \(%\):\s*([0-9.]+)", use)]
    # Tightest card: every rank gets the same mem-fraction.
    mem = sh("rocm-smi --showmeminfo vram --csv 2>/dev/null || true")
    free_mib, total_mib = [], []
    for line in mem.splitlines()[1:]:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 3 and cols[1].isdigit() and cols[2].isdigit():
            total_mib.append(int(cols[1]) // (1024 * 1024))
            free_mib.append((int(cols[1]) - int(cols[2])) // (1024 * 1024))
    busy_ports = [int(p) for p in (ports or [])
                  if sh(f"ss -ltn 'sport = :{int(p)}' 2>/dev/null | tail -n +2 || true").strip()]
    resident_pids, parked = _resident_gpu_pids(sh("rocm-smi --showpids 2>/dev/null || true"))
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as f:
            boot_id = f.read().strip()
    except OSError:
        boot_id = ""
    serve = sh(f"pgrep -c -f '{SERVE_PATTERN}' || true").strip()
    workers = sh(f"({OUR_WORKER_PIDS}) | wc -l").strip()
    return {
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "sw": _versions(),
        "ngpu": len(uses),
        "max_use": max(uses) if uses else None,
        "gpu_pids": len(resident_pids),
        # rocm-smi reports HOST pids: one with no /proc entry here is another container's.
        "foreign_gpu_pids": sum(1 for p in resident_pids if not os.path.exists(f"/proc/{p}")),
        "parked_gpu_pids": parked,
        "serve_procs": int(serve or 0),
        "worker_procs": int(workers or 0),
        "free_vram_mib": min(free_mib) if free_mib else None,
        "total_vram_mib": min(total_mib) if total_mib else None,
        "card_free_mib": free_mib,
        "card_total_mib": total_mib,
        "busy_ports": busy_ports,
        "addrs": [list(a) for a in local_addrs()],
        "gpu": gpu_identity(),
    }


@functools.lru_cache(maxsize=None)
def local_addrs() -> tuple:
    """(interface, IPv4, prefix length) of every address on this machine."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    addrs = []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 4:
            ip, _, plen = cols[3].partition("/")
            addrs.append((cols[1], ip, int(plen) if plen.isdigit() else 24))
    return tuple(addrs)


GPU_IDENTITY_SCRIPT = os.path.join(HERE, os.pardir, os.pardir, os.pardir, "scripts", "gpu_identity.py")
GPU_KEYS = ("gfx", "target", "marketing_name", "physical_cu_count")


def python_by_content(path: str, args: str = "") -> str:
    """Shell that runs this machine's python file `path` on a node, from a heredoc."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return f"{' '.join(['python3', '-', args]).rstrip()} <<'GEAK_MN_PY'\n{src}\nGEAK_MN_PY\n"


def gpu_identity() -> dict:
    """run_e2e.py's GPU identity probe (rocminfo), which probe() ran on this node before
    mn.py and handed over in MN_GPU_IDENTITY; {"error": ...} if it failed."""
    out = os.environ.get("MN_GPU_IDENTITY", "")
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": out.strip()[-300:] or "the GPU identity probe printed nothing"}


# --------------------------------------------------------------------------
# TRANSPORT
# --------------------------------------------------------------------------
# GEAK owns no cluster: the driver reaches every node by ssh, the script on stdin (a node
# that is this machine gets a fresh local shell). A node is hostname/boot_id: an IP that
# now answers from another machine (a recreated pod) is not the pin.

SUBSTITUTION_REFUSAL = "Refusing to substitute another machine -- the A/B legs would\n    no longer be comparable."


SSH_KEYS = ("key", "port", "user", "known_hosts", "port_offsets", "prefix")
PROBE_TAG = "MN_PROBE="
IDENTITY_SH = 'echo "$(cat /proc/sys/kernel/hostname)/$(cat /proc/sys/kernel/random/boot_id)"\n'
# An sshd session lacks the container's env (k8s-injected NCCL/ROCm vars); take
# pid 1's value for every name the session left unset.
POD_ENV = ("while IFS= read -r -d '' kv; do k=${kv%%=*}; "
           '[[ $k =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!k+x}" ] && export "$kv"; '
           "done 2>/dev/null < /proc/1/environ\n")


def ssh_config(cfg: dict) -> dict:
    """handoff.multinode.ssh, validated (the pin's ssh block). A node's own `ssh` block
    has the same keys and overrides it."""
    if not isinstance(cfg, dict):
        raise ValueError("ssh settings must be a JSON object")
    unknown = sorted(set(cfg) - set(SSH_KEYS))
    if unknown:
        raise ValueError(f"unknown ssh key(s) {unknown}; known: {', '.join(SSH_KEYS)}")
    offsets = cfg.get("port_offsets") or {}
    if not isinstance(offsets, dict) or any(r not in PD_ROLES for r in offsets):
        raise ValueError(f"ssh.port_offsets must map {'/'.join(PD_ROLES)} to an int, got {offsets!r}")
    return cfg


SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR")


def node_identity(probe: dict) -> str:
    return f"{probe['hostname']}/{probe.get('boot_id', '')}"


@functools.lru_cache(maxsize=None)
def local_identity() -> str:
    """This machine as a pinned node_id (probe_node's hostname/boot_id); '' without a boot_id."""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as f:
            return f"{socket.gethostname()}/{f.read().strip()}"
    except OSError:
        return ""


def is_self(node: dict) -> bool:
    """This machine is `node`: a pinned node by hostname/boot_id, a handoff host (check,
    before any pin) by its address, a hostname resolved first."""
    if node.get("node_id"):
        return node["node_id"] == local_identity()
    try:
        ip = socket.gethostbyname(node["ip"])
    except (OSError, UnicodeError):
        return False
    return ip in {a[1] for a in local_addrs()}


def ssh_argv(state: dict, node: dict) -> list[str]:
    """The command that runs a script (on its stdin) on `node`: a fresh local shell, as
    ssh would give, when that node is this machine. Before the pin an address cannot tell
    the host from the container a `prefix` enters, so such a node is always reached by ssh."""
    cfg = {**(state.get("ssh") or {}), **(node.get("ssh") or {})}
    if is_self(node) and (node.get("node_id") or not cfg.get("prefix")):
        return ["env", "-i", f"PATH={os.environ.get('PATH', '')}", f"HOME={os.environ.get('HOME', '')}",
                "bash", "-s"]
    role = node_role(state, node) if topology_of(state) == "pd_disagg" else ""
    port = int(cfg.get("port") or 22) + int((cfg.get("port_offsets") or {}).get(role, 0))
    argv = ["ssh", *SSH_OPTS, "-p", str(port)]
    if cfg.get("key"):
        argv += ["-i", cfg["key"], "-o", "IdentitiesOnly=yes"]
    argv += (["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={cfg['known_hosts']}"]
             if cfg.get("known_hosts") else ["-o", "StrictHostKeyChecking=accept-new"])
    return argv + [f"{cfg['user']}@{node['ip']}" if cfg.get("user") else node["ip"],
                   f"{cfg.get('prefix') or ''} bash -s".strip()]


def ssh_run(state: dict, node: dict, script: str, timeout: int = 300, merged: bool = False) -> dict:
    """Run `script` on the node and wait: {hostname, ip, rc, out, err}. merged: one stream
    (a script that backgrounds the server; rc=0 then says nothing about readiness)."""
    # Bytes: a text-mode pipe on a Windows driver writes \r\n and remote bash chokes.
    proc = subprocess.run(ssh_argv(state, node), input=(POD_ENV + script).encode(), timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merged else subprocess.PIPE)
    out = {"hostname": node.get("hostname") or node["ip"], "ip": node["ip"], "rc": proc.returncode,
           "out": (proc.stdout or b"").decode("utf-8", "replace").strip()}
    if not merged:
        out["err"] = (proc.stderr or b"").decode("utf-8", "replace").strip()
    return out


def _parallel(calls: list) -> list:
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(lambda call: call(), calls))


def ssh_each(state: dict, jobs: list, merged: bool = False) -> list:
    """ssh_run for every (node, script, timeout) at once, results in job order."""
    return _parallel([functools.partial(ssh_run, state, n, s, t, merged) for n, s, t in jobs])


def ssh_stream(state: dict, node: dict, script: str, timeout=None) -> int:
    """Run `script` on the node with its output on ours; the remote exit code. stdin stays
    open: when this process dies, or closes it on TERM/INT or at `timeout`, the watchdog of
    an exec_script reads EOF. Returns only once ssh has, so the remote side has wound down."""
    proc = subprocess.Popen(ssh_argv(state, node), stdin=subprocess.PIPE)

    def hang_up(_signum=None, _frame=None):
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()

    previous = {s: signal.signal(s, hang_up) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with contextlib.suppress(OSError):
            proc.stdin.write((POD_ENV + script).encode())
            proc.stdin.flush()
        try:
            return proc.wait(timeout)
        except subprocess.TimeoutExpired:
            hang_up()
            return proc.wait()
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def probe(state: dict, node: dict, ports: list | None = None) -> dict:
    """probe_node on `node`: this mn.py and the GPU identity probe, shipped by content."""
    script = (f'MN_GPU_IDENTITY="$({python_by_content(GPU_IDENTITY_SCRIPT, "2>&1")})" '
              + python_by_content(os.path.join(HERE, "mn.py"),
                                  "probe " + " ".join(str(int(p)) for p in ports or [])))
    res = ssh_run(state, node, script, 600)
    line = next((ln for ln in reversed(res["out"].splitlines()) if ln.startswith(PROBE_TAG)), "")
    if not line:
        raise RuntimeError(f"probe on {node['ip']} returned nothing (rc={res['rc']}): {res['err'][-500:]}")
    return json.loads(line[len(PROBE_TAG):])


def probe_all(state: dict, nodes: list, ports: list | None = None) -> list:
    return _parallel([functools.partial(probe, state, n, ports) for n in nodes])


def resolve(state: dict) -> list[dict]:
    """The pinned machines (state_targets), each confirmed to still be the machine that was
    pinned. An ssh that does not get through is a RuntimeError (the node may be fine);
    another machine answering at the pinned IP is a hard stop."""
    nodes = state_targets(state)
    for node, res in zip(nodes, ssh_each(state, [(n, IDENTITY_SH, 60) for n in nodes])):
        if res["rc"] != 0 or not res["out"]:
            raise RuntimeError(f"pinned rank{node['rank']} {node['hostname']} ({node['ip']}) did not answer "
                               f"over ssh (rc={res['rc']}): {res.get('err') or 'no output'}")
        if res["out"] != node["node_id"]:
            raise SystemExit(
                f"!!! multinode: pinned rank{node['rank']} {node['hostname']} ({node['ip']}) now answers as {res['out'] or res.get('err') or 'nothing'!r}.\n    {SUBSTITUTION_REFUSAL}")
    return nodes


# The LB is killed even on tp_span: a survivor on the client port answers /health
# for engines that are gone.
KILL_OURS = "sleep 3; ".join(
    "".join(f"pkill {sig}-f '{p}' 2>/dev/null || true; "
            for p in (SERVE_PATTERN, ENGINE_PATTERN, LB_PATTERN))
    + f"kill {sig}$({OUR_WORKER_PIDS}) 2>/dev/null || true; "
    for sig in ("", "-9 ")) + "echo killed"


def kill_script(node: dict) -> str:
    """Kill order: the recorded server's process group (a provided router is found only
    this way: no name pattern knows every backend's router), then everything of ours by name."""
    steps = [f'p=$({live_pid(node["rank"])}); pg=""',
             'case "$p" in ""|gone) ;; *) pg=$(ps -o pgid= -p "$p" | tr -d " ") ;; esac',
             'case "$pg" in ""|0|1) pg="" ;; *) kill -TERM -- "-$pg" 2>/dev/null || true ;; esac',
             KILL_OURS,
             '[ -z "$pg" ] || kill -9 -- "-$pg" 2>/dev/null || true',
             f"rm -f {PID_DIR}/rank{node['rank']}.pid"]
    return "\n".join(steps) + "\n"


# role / ssh: the transport picks the node's sshd endpoint from them.
LAUNCH_NODE_KEYS = ("rank", "node_id", "hostname", "ip", "role", "ssh")


def launched_nodes(out_dir: str):
    """(nodes, router) the launch in `out_dir` spawned on, or None if unrecorded.

    The pin answers where the NEXT launch goes; another run rewriting it must not
    move this launch's supervisor and teardown onto other boxes."""
    try:
        with open(os.path.join(out_dir, "mn_launch.json"), encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    nodes = doc.get("nodes")
    if not nodes or not all(n.get("node_id") for n in nodes):
        return None
    return nodes, doc.get("router") or {}


def supervised_state(state: dict, out_dir: str) -> dict:
    launched = launched_nodes(out_dir)
    return {**state, "nodes": launched[0], "router": launched[1]} if launched else state


def teardown(state: dict, wait_idle_s: int = 180, verbose=print) -> list | None:
    """Kill every rank of a previous launch and wait for the cards to clear.
    Returns the last probe of every node once clear (launch reads the stack from it), else None.

    Runs before every launch: a surviving scheduler holds the dist-init port
    while /health still answers 200. It is also what clears the ranks of a
    supervisor that was killed with -9 (nothing else is left to reap them)."""
    nodes = resolve(state)
    ssh_each(state, [(n, kill_script(n), 120) for n in nodes])

    def still_ours(p) -> bool:
        # Another container's pids are the caller's choice of machines, not ours to wait on.
        return (bool(p["serve_procs"]) or bool(p.get("worker_procs"))
                or (p["gpu_pids"] or 0) > (p.get("foreign_gpu_pids") or 0))

    deadline = time.time() + wait_idle_s
    while True:
        probes = probe_all(state, nodes)
        busy = [
            f"{_short(p['hostname'])}(use={p['max_use']} pids={p['gpu_pids']}"
            + (f" of which {p['foreign_gpu_pids']} from another container"
               if p.get("foreign_gpu_pids") else "")
            + f" serve={p['serve_procs']} workers={p.get('worker_procs', 0)})"
            for p in probes if still_ours(p)
        ]
        if not busy:
            verbose(f">>> multinode teardown: all {len(nodes)} nodes clear")
            return probes
        if time.time() > deadline:
            foreign = [_short(p["hostname"]) for p in probes if p.get("foreign_gpu_pids")]
            verbose(f"!!! multinode teardown: still busy after {wait_idle_s}s: "
                    + ", ".join(busy)
                    + (f"; {', '.join(foreign)} has a co-tenant we cannot reap"
                       if foreign else ""))
            return None
        verbose(">>> multinode teardown: waiting on " + ", ".join(busy))
        time.sleep(10)


def serve_counts(state: dict) -> list[int]:
    nodes = resolve(state)
    results = ssh_each(state, [(n, serve_alive(n["rank"]), 90) for n in nodes])
    return [int(m.group(1)) if (m := re.search(r"serve=(\d+)", r["out"] or "")) else 0 for r in results]


def engine_health(state: dict) -> list[dict]:
    """Per rank: serve process (by its pid file), scheduler workers, zombies; plus the LB on a PD master.

    Zombie schedulers behind a live HTTP front end are a dead engine. The LB is
    asked on its client port, not found by name (the router renames itself).
    A node whose probe does not come back over ssh raises: that is not a dead rank."""
    nodes = resolve(state)
    workers = (f"echo sched=$(pgrep -c -f '{SCHED_PATTERN}' || true) "
               "zombie=$(ps -eo stat=,args= | awk '/(sglang|VLLM)::/ && $1 ~ /Z/' | wc -l)")
    watch_lb = topology_of(state) == "pd_disagg"
    # A pin without a "pd" block takes pd_config's defaults; reading the raw key would probe port None.
    lb_port = pd_config(state)["lb_port"] if watch_lb else None
    lb_probe = (
        f"echo lb=$(curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 http://127.0.0.1:{lb_port}/health || echo 000)"
    )

    lb_rank = (lb_node(state) or {}).get("rank") if watch_lb else None

    def is_lb(n):
        return watch_lb and n["rank"] == lb_rank

    def health_script(n):
        return "\n".join([serve_alive(n["rank"]), workers, *([lb_probe] if is_lb(n) else []), "echo ok=1"]) + "\n"

    results = ssh_each(state, [(n, health_script(n), 90) for n in nodes])
    out = []
    for node, res in zip(nodes, results):
        if "ok=1" not in (res["out"] or "").split():
            raise RuntimeError(f"rank{node['rank']} {node['ip']} did not answer over ssh "
                               f"(rc={res.get('rc')}): {(res.get('err') or '')[-300:]}")
        vals = dict(part.split("=", 1) for part in (res["out"] or "").split() if "=" in part)
        num = lambda key: int(vals[key]) if vals.get(key, "").isdigit() else 0  # noqa: E731
        entry = {
            "rank": node["rank"],
            "host": _short(node["hostname"]),
            "serve": num("serve"),
            "pid": vals.get("pid", "none"),
            "schedulers": num("sched"),
            "zombies": num("zombie"),
        }
        if is_lb(node):
            entry["lb_http"] = num("lb")
            entry["lb"] = 1 if entry["lb_http"] == 200 else 0
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# CHECK  (mn.py check)
# --------------------------------------------------------------------------
# The caller names the machines (handoff.multinode.nodes, rank order); this only
# proves they are usable and records their identities and NICs in mn_state.json.

def prove_shared(state: dict, nodes: list[dict]) -> dict:
    """Exchange marker files in both directions; `df` showing the same size proves nothing."""
    shared = state["shared_fs"]
    token = uuid.uuid4().hex[:12]
    probe_dir = os.path.join(shared, ".mn_probe")
    os.makedirs(probe_dir, exist_ok=True)
    local_marker = os.path.join(probe_dir, f"from_bench_{token}")
    with open(local_marker, "w", encoding="utf-8") as f:
        f.write(token)

    results = {"token": token, "bench_host": _short(socket.gethostname()),
               "per_node": [], "shared": True}
    for node in nodes:
        script = (
            f"set -e; test -r {local_marker} && echo READ_OK=$(cat {local_marker}) || echo READ_FAIL; echo {token} > {probe_dir}/from_rank{node['rank']}_{token} && echo WRITE_OK || echo WRITE_FAIL"
        )
        out = ssh_run(state, node, script, 120)
        readable = f"READ_OK={token}" in out["out"]
        wrote = "WRITE_OK" in out["out"]
        seen_here = os.path.isfile(os.path.join(probe_dir, f"from_rank{node['rank']}_{token}"))
        ok = readable and wrote and seen_here
        results["per_node"].append({
            "rank": node["rank"], "hostname": node["hostname"],
            "reads_bench_marker": readable, "can_write": wrote,
            "its_marker_visible_to_bench": seen_here, "ok": ok,
        })
        results["shared"] = results["shared"] and ok

    for name in os.listdir(probe_dir):
        if token in name:
            try:
                os.remove(os.path.join(probe_dir, name))
            except OSError:
                pass
    return results


def visible_gpus() -> int:
    """GPU agents rocminfo finds in this process's environment, which the agents' commands
    inherit; 0 without ROCm."""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return len(re.findall(r"^\s*Name:\s*gfx[0-9a-f]+\s*$", out, re.M))


# The kernel tools a node runs (gpu_lock.sh and profile_kernel.sh forward to these copies).
KERNEL_SCRIPTS = os.path.join(HERE, os.pardir, os.pardir, os.pardir, "kernel_workflow", "scripts")
TOOLS = ("gpu_lock.sh", "workspace_sources.py", "profile_kernel.sh", "profile_policy.sh")


def stage_tools(work_dir: str) -> str:
    """Copy TOOLS into <work_dir>/.mn_tools, a run artifact on the shared mount."""
    tools_dir = os.path.join(work_dir, ".mn_tools")
    os.makedirs(tools_dir, exist_ok=True)
    for name in TOOLS:
        shutil.copy2(os.path.join(KERNEL_SCRIPTS, name), os.path.join(tools_dir, name))
    return tools_dir


def locate(ips: list, probes: list, need_nic: bool = True, gpu_ids: list | None = None) -> tuple[list, list]:
    """(nodes, problems): rank i is the machine that answered at ips[i]. Its NIC is the
    one carrying that address; a provided launch script owns the NIC, so it may have none.
    Ranks share a machine only when each names its cards (gpu_ids[i]) and no card twice."""
    nodes, problems = [], []
    for rank, (ip, probe) in enumerate(zip(ips, probes)):
        nics = [a[0] for a in probe["addrs"] if a[1] == ip]
        if need_nic and len(nics) != 1:
            problems.append(f"rank{rank} {ip}: " + (
                f"carried by {len(nics)} interfaces" if nics
                else f"{_short(probe['hostname'])} has no interface with this address"))
            continue
        nodes.append({"rank": rank, "node_id": probe["node_id"], "ip": ip,
                      "hostname": probe["hostname"], "cluster_if": nics[0] if nics else ""})
    cards = {}
    for n in nodes:
        cards.setdefault(n["node_id"], []).append((gpu_ids or [None] * len(ips))[n["rank"]])
    for node_id, sets in cards.items():
        if len(sets) < 2:
            continue
        if None in sets:
            problems.append("two of the IPs are on the same machine")
        elif len({g for s in sets for g in s}) < sum(map(len, sets)):
            problems.append(f"the ranks on {_short(node_id.split('/')[0])} share a card: gpu_ids {sets}")
    return nodes, problems


NODE_OPT_KEYS = ("ssh", "launch_command", "port", "bootstrap_port", "gpu_ids")


def port_pairs(state: dict, provided: bool) -> list:
    """(listening node, port, dialers, what) for every port the engines bind that another pinned
    machine, or this one (dialer None: the bench client, health and profiler), dials.
    A provided command binds its own dist-init ports; its ports are the command's."""
    nodes = state["nodes"]
    client_at = [None, nodes[0]] if state.get("bench_client") == "native" else [None]
    if topology_of(state) != "pd_disagg":
        master = next(n for n in nodes if n["ip"] == state["master_ip"])
        return [(master, state["serve_port"], client_at, "client"),
                *([] if provided or len(nodes) < 2 else [(nodes[0], state["dist_init_port"], nodes[1:], "dist-init")])]
    pd = pd_config(state)
    lb = lb_node(state)
    pairs = [(lb, pd["lb_port"], client_at, "load balancer")]
    for e in engine_urls(state):
        pairs.append((e["node"], e["port"], [lb, None], f"{e['role']} engine HTTP"))
        if not provided and len(e["nodes"]) > 1:
            pairs.append((e["node"], state["dist_init_port"], e["nodes"][1:], f"{e['role']} dist-init"))
    decode = engine_nodes(state, "decode")
    for e in engine_urls(state):
        boot = e["node"].get("bootstrap_port") or (None if provided else pd["bootstrap_port"])
        if e["role"] == "prefill" and boot and decode:
            pairs.append((e["node"], boot, decode, "prefill bootstrap"))
    return [(node, port, [c for c in dialers if c is not node], what)
            for node, port, dialers, what in pairs if any(c is not node for c in dialers)]


# What each bench client imports where it runs (adapters/launchers/multinode.sh): the
# InferenceX client on this machine; a backend's own client on rank0, with its framework.
INFERENCEX_IMPORTS = "numpy, aiohttp, tqdm, transformers, huggingface_hub"
INFERENCEX_SCRIPTS = ("utils/bench_serving/benchmark_serving.py", "benchmarks/benchmark_serving.py",
                      "benchmark_serving.py")
NATIVE_BENCH_MODULES = {"sglang": "sglang.bench_serving", "vllm": "vllm.benchmarks.serve",
                        "atom": "atom.benchmarks.benchmark_serving"}


def inferencex_client_error(model: str) -> str:
    """'' when the InferenceX client can run here: inferencex.sh finds benchmark_serving.py,
    its python (PYTHON_BIN) imports its packages, and a model given as a path has its
    tokenizer on this machine."""
    script = os.environ.get("INFERENCEX_BENCH_SERVING", "")
    root = os.environ.get("INFERENCEX_PATH", "")
    if not os.path.isfile(script) and not (
            root and any(os.path.isfile(os.path.join(root, s)) for s in INFERENCEX_SCRIPTS)):
        return (f"INFERENCEX_PATH={root or '<unset>'} has no benchmark_serving.py on this machine "
                f"(looked under {', '.join(INFERENCEX_SCRIPTS)})")
    py = os.environ.get("PYTHON_BIN") or "python3"
    try:
        proc = subprocess.run([py, "-c", f"import {INFERENCEX_IMPORTS}"],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{py} does not start here: {exc}"
    if proc.returncode != 0:
        return f"{py} cannot import {INFERENCEX_IMPORTS}: {(proc.stderr or proc.stdout).strip()[-300:]}"
    if model.startswith("/") and not os.path.isfile(os.path.join(model, "tokenizer_config.json")):
        return f"it loads the tokenizer from {model}, which this machine cannot read (no tokenizer_config.json)"
    return ""


def client_problems(state: dict, spec: dict) -> list[str]:
    """The bench client can run where the launcher runs it. agentx (aiperf) is run_e2e.py's
    agentx_preflight; a client adapter of another name is checked by nobody here."""
    client = state["bench_client"]
    if client == "inferencex":
        err = inferencex_client_error(spec.get("model") or "")
        return [f"the InferenceX bench client runs on this machine and {err}"] if err else []
    if client == "native":
        backend = spec.get("backend") or os.environ.get("BACKEND", "sglang")
        module = NATIVE_BENCH_MODULES.get(backend)
        if not module:
            return [f"BENCH_CLIENT=native: no native client known for backend {backend!r}"]
        r = ssh_run(state, state["nodes"][0], f"cd / && python3 -c 'import {module}'\n", 120)
        if r["rc"] != 0:
            return [f"BENCH_CLIENT=native runs on rank0 {_short(r['hostname'])}, which cannot import "
                    f"{module}: {(r['err'] or r['out'])[-300:]}"]
    return []


def listen_script(port: int) -> str:
    """Hold `port` open for a minute on this node; prints `LISTENING <pid>` once it is."""
    py = ("import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
          f"s.bind(('',{port}));s.listen(64);time.sleep(60)")
    return (f"setsid timeout 60 python3 -c {shlex.quote(py)} </dev/null >/dev/null 2>&1 &\n"
            f"pid=$!; for i in $(seq 50); do ss -ltn 'sport = :{port}' 2>/dev/null | tail -n +2 | grep -q . "
            "&& { echo LISTENING $pid; exit 0; }; sleep 0.2; done; echo nothing listening\n")


def dials_here(ip: str, port: int) -> bool:
    try:
        socket.create_connection((ip, port), timeout=5).close()
        return True
    except OSError:
        return False


def reach_problems(state: dict, pairs: list) -> list[str]:
    """Open each port on its node and dial it from the others: a firewall or a wrong NIC
    shows up here, not as a collective that hangs at startup."""
    problems = []
    for node, port, dialers, what in pairs:
        up = ssh_run(state, node, listen_script(port), 30)
        listener = re.search(r"LISTENING (\d+)", up["out"] or "")
        if not listener:
            problems.append(f"rank{node['rank']} {_short(node['hostname'])} could not listen on the {what} "
                            f"port {port}: {up['out'] or up.get('err')}")
            continue
        remote = [c for c in dialers if c is not None]
        dials = ssh_each(state, [(c, f"timeout 5 bash -c 'exec 3<>/dev/tcp/{node['ip']}/{port}' && echo REACHED", 30)
                                 for c in remote])
        problems += [f"rank{c['rank']} {_short(c['hostname'])} cannot reach the {what} port {node['ip']}:{port} "
                     f"on rank{node['rank']}" for c, d in zip(remote, dials) if "REACHED" not in (d["out"] or "")]
        if None in dialers and not dials_here(node["ip"], port):
            problems.append(f"this machine ({socket.gethostname()}) cannot reach the {what} port {node['ip']}:{port} "
                            f"on rank{node['rank']}; the bench client and health checks run here")
        ssh_run(state, node, f"kill {listener.group(1)} 2>/dev/null || true", 30)
    return problems


# The check's verdict on its last stdout line, for the workflow that ran it.
CHECK_TAG = "MN_CHECK="


def check_verdict(pin: str, problems: list, spec: dict | None = None) -> int:
    """Record the verdict in mn_check.json beside the pin and hand it to the caller.
    On success the caller also gets the normalized spec it serves from."""
    report = {"verdict": "refused" if problems else "ok", "problems": problems, "at": _utc()}
    if pin and os.path.isdir(os.path.dirname(pin)):
        _write_json(os.path.join(os.path.dirname(pin), "mn_check.json"), report)
    if problems:
        sys.stderr.write("!!! multinode check: the handoff's nodes are not usable:\n"
                         + "".join(f"      - {p}\n" for p in problems))
    print(CHECK_TAG + json.dumps({**report, **({"spec": spec} if spec and not problems else {})}), flush=True)
    return 2 if problems else 0


# The pin of the multinode run on this machine, with its owner: the process that ran the
# check (run_e2e.py, which outlives the workflow), or for a prompt-started workflow the Claude Code
# process its check agent ran under. The pin counts only while that very process lives, so a single-host
# run after it, or after it was killed, is not taken for a multinode one. It also names this
# mn.py, for bench_e2e.sh copies that have no multi_node/ beside them.
ACTIVE_PIN = f"{PID_DIR}/state_path"


def _proc_start(pid) -> str:
    """The process's start time (as _START_TIME reads it); '' if there is no such process."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError, TypeError):
        return ""


def claude_ancestor() -> int:
    """The nearest ancestor that is the Claude Code process (0 if none). A Workflow's agents run in
    that one process and their Bash commands are its children, so it outlives the workflow, while
    the shell an agent ran check from is gone the moment the command returns."""
    pid = os.getppid()
    while pid > 1:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                stat = f.read()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")[:2]
        except OSError:
            return 0
        if stat[stat.index("(") + 1:stat.rindex(")")] == "claude" or any(
                os.path.basename(a) == "claude" or "@anthropic-ai/claude-code" in a for a in argv):
            return pid
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    return 0


def active_pin() -> str:
    """GEAK_MN_STATE (the handoff entry sets it), else the pin a live run on this machine owns."""
    if os.environ.get("GEAK_MN_STATE"):
        return os.environ["GEAK_MN_STATE"]
    try:
        with open(ACTIVE_PIN, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return ""
    alive = doc.get("start") and _proc_start(doc.get("pid")) == doc["start"]
    return str(doc.get("state") or "") if alive else ""


def cmd_active_pin(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="mn.py active-pin")
    ap.add_argument("--lease-dir", action="store_true",
                    help="also print the pin's card-lease directory and kernel-tools copy (tab-separated)")
    ap.add_argument("--release", action="store_true", help="drop the pointer the calling process owns")
    a = ap.parse_args(argv)
    if a.release:
        try:
            with open(ACTIVE_PIN, encoding="utf-8") as f:
                owner = json.load(f).get("pid")
            if owner == os.getppid():
                os.remove(ACTIVE_PIN)
        except (OSError, ValueError, AttributeError):
            pass
        return 0
    pin = active_pin()
    if pin and a.lease_dir:
        try:
            with open(pin, encoding="utf-8") as f:
                doc = json.load(f)
            print(f"{pin}\t{lease_dir(doc)}\t{doc.get('tools_dir') or ''}")
        except (OSError, ValueError, KeyError) as exc:
            sys.stderr.write(f"!!! multinode: the active pin {pin} is unreadable: {exc}\n")
            return 1
    elif pin:
        print(pin)
    return 0


def cmd_check(argv: list) -> int:
    """GEAK_MN_SPEC is handoff.multinode plus "tp" (the engine TP), as JSON."""
    ap = argparse.ArgumentParser(prog="mn.py check")
    ap.add_argument("--owner", choices=("parent", "claude"), default="parent",
                    help="whose lifetime the pin counts for: the calling process (run_e2e.py), or the "
                         "Claude Code process above an agent's shell (a workflow started from a prompt)")
    owner = os.getppid() if ap.parse_args(argv).owner == "parent" else claude_ancestor()
    raw = {}
    try:
        raw = json.loads(os.environ.get("GEAK_MN_SPEC") or "{}")
        spec = normalize_spec(raw, raw.get("tp") if isinstance(raw, dict) else None)
    except (TypeError, ValueError) as exc:
        given = raw if isinstance(raw, dict) else {}
        shared = str(given.get("shared_fs") or "").strip().rstrip("/")
        pin = str(given.get("state") or "").strip() or (f"{shared}/mn_state.json" if shared else "")
        return check_verdict(pin, [f"GEAK_MN_SPEC: {exc}"])
    pin, shared, topology, layout = spec["state"], spec["shared_fs"], spec["topology"], spec["layout"]
    ips, router = [n["host"] for n in spec["nodes"]], spec["router"]
    provided = bool(spec["nodes"][0].get("launch_command"))
    state = {
        "nodes": [], "topology": topology,
        "serve_port": spec["serve_port"], "dist_init_port": spec["dist_init_port"],
        "shared_fs": shared, "mn_dir": HERE, "ssh": spec["ssh"],
        "mode": "provided" if provided else "generated",
        **({"pd": spec["pd"]} if spec["pd"] else {}),
        **({"backend": spec["backend"]} if provided else {}),
    }
    problems = []
    if not owner:
        problems.append("--owner claude: no Claude Code process above this shell to own the pin; run "
                        "the check from the workflow's own agent, or pass its verdict as args.multinode_check")
    if not os.path.isdir(shared):
        problems.append(f"shared_fs {shared!r} is not a directory here")
    work_dir = spec["work_dir"]
    if work_dir and not under(work_dir, shared):
        problems.append(f"the run directory {work_dir} is not under {shared}: the nodes run the bench "
                        "scripts, overlay and kernel tools from it")
    ngpu = visible_gpus()
    if ngpu:
        problems.append(f"this machine's environment sees {ngpu} GPU(s); a multinode run's driver must see "
                        "none, or a command that bypasses the tools runs here. Start Claude with "
                        "ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=")
    if problems:
        return check_verdict(pin, problems)

    # Every port the topology binds; a busy one is refused.
    if provided:
        ports = sorted({state["serve_port"], *(n[k] for n in spec["nodes"]
                                               for k in ("port", "bootstrap_port") if n.get(k))})
    elif topology == "pd_disagg":
        pd = pd_config(state)
        ports = [state["serve_port"], *pd["role_ports"], pd["bootstrap_port"]]
        if any(r.get("engine_nnodes") for r in layout):
            ports.append(state["dist_init_port"])
    else:
        ports = [state["serve_port"], state["dist_init_port"]]

    # A provided router is a machine of its own: no GPU, no stack; reachable, ports free, mount shared.
    targets = spec_targets(spec)
    if router:
        targets.append({"node_id": "", "rank": "lb", "ip": router["host"], "role": "router",
                        **({"ssh": router["ssh"]} if router.get("ssh") else {})})
    try:
        probes = probe_all(state, targets, ports)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return check_verdict(pin, [f"a node could not be probed: {exc}"])
    for p in probes:
        p["node_id"] = node_identity(p)
    if router:
        rp = probes.pop()
        state["router"] = {"rank": "lb", "node_id": rp["node_id"], "ip": router["host"],
                           "hostname": rp["hostname"], "role": "router", "port": router["port"],
                           "launch_command": router["launch_command"],
                           **({"ssh": router["ssh"]} if router.get("ssh") else {})}
    state["nodes"], problems = locate(ips, probes, need_nic=not provided,
                                      gpu_ids=[n.get("gpu_ids") for n in spec["nodes"]])
    if router and rp["busy_ports"] and router["port"] in rp["busy_ports"]:
        problems.append(f"router {_short(rp['hostname'])}: port {router['port']} already bound")
    by_id = {p["node_id"]: p for p in probes}
    mem_fraction = float(os.environ.get("MEM_FRACTION") or 0.90)
    for n in state["nodes"]:
        p = by_id[n["node_id"]]
        n.update(layout[n["rank"]])
        n.update({k: spec["nodes"][n["rank"]][k] for k in NODE_OPT_KEYS if spec["nodes"][n["rank"]].get(k)})
        n["ngpu"] = p["ngpu"]
        name = f"rank{n['rank']} {_short(n['hostname'])}"
        if p["ngpu"] < n["gpus"]:
            problems.append(f"{name} has {p['ngpu']} GPUs, needs {n['gpus']}")
        elif n.get("gpu_ids") and max(n["gpu_ids"]) >= p["ngpu"]:
            problems.append(f"{name} has {p['ngpu']} GPUs, no card {max(n['gpu_ids'])}")
        free, total = p.get("free_vram_mib"), p.get("total_vram_mib")
        # A rank that names its cards serves only from those: another tenant elsewhere on the machine is not ours to fit around.
        ids, cards = n.get("gpu_ids"), p.get("card_free_mib") or []
        if ids and len(cards) > max(ids) and len(p.get("card_total_mib") or []) == len(cards):
            free, total = min(cards[i] for i in ids), min(p["card_total_mib"][i] for i in ids)
        if free is None or total is None or free < mem_fraction * total:
            problems.append(f"{name}: its fullest card has {free} of {total} MiB free; the server takes "
                            f"{mem_fraction:.0%} of a card (MEM_FRACTION)")
        if p["busy_ports"]:
            problems.append(f"{name}: port(s) {p['busy_ports']} already bound")
        busy = [f"{k}={p[k]}" for k in ("serve_procs", "gpu_pids", "foreign_gpu_pids") if p.get(k)]
        print(f"    rank{n['rank']} {_short(n['hostname'])} {n['ip']} if={n['cluster_if'] or '-'} gpus={p['ngpu']} serves {n['gpus']} (TP {n['tp']})"
              + (f"  !!! busy: {', '.join(busy)}" if busy else ""))
    bound = {}
    for n in [*state["nodes"], *([state["router"]] if router else [])]:
        for port in {n.get("port"), n.get("bootstrap_port")} - {None}:
            first = bound.setdefault((n["node_id"], port), n["rank"])
            if first != n["rank"]:
                problems.append(f"rank{first} and rank{n['rank']} both bind port {port} on {_short(n['hostname'])}")
    stacks = {json.dumps(p.get("sw") or {}, sort_keys=True) for p in probes}
    if len(stacks) > 1:
        problems.append("the nodes load different stacks: " + "; ".join(
            f"{_short(p['hostname'])} {p.get('sw')}" for p in probes))
    state["sw"] = probes[0].get("sw") or {}
    # The KB is addressed by product and CU count: every node must be the same GPU.
    gpus = [[(p.get("gpu") or {}).get(k) for k in GPU_KEYS] for p in probes]
    if len({json.dumps(g) for g in gpus}) > 1:
        problems.append(f"the nodes have different GPUs ({', '.join(GPU_KEYS)}): " + "; ".join(
            f"{_short(p['hostname'])} {g}" for p, g in zip(probes, gpus)))
    state["gpu"] = dict(zip(GPU_KEYS, gpus[0]))
    state["master_ip"] = router["host"] if router else state["nodes"][0]["ip"]
    # run_e2e.py's apply_bench_client exports it before the check, bench_e2e.sh's default after.
    state["bench_client"] = (os.environ.get("BENCH_CLIENT") or "native").strip().lower()
    if not problems:
        problems += client_problems(state, spec)
    if not problems:
        problems += reach_problems(state, port_pairs(state, provided))
    if not problems:
        proof = prove_shared(state, state_targets(state))
        problems += [f"rank{n['rank']} cannot share {shared}: {n}"
                     for n in proof["per_node"] if not n["ok"]]
    if problems:
        return check_verdict(pin, problems)
    if work_dir:
        state["tools_dir"] = stage_tools(work_dir)

    state["checked_at"] = _utc()
    print(f">>> multinode: {topology} on {len(ips)} node(s), master {state['master_ip']}, shared fs {shared} proven -> {_write_json(pin, state)}")
    _write_json(ACTIVE_PIN, {"state": pin, "pid": owner, "start": _proc_start(owner),
                             "mn_py": os.path.join(HERE, "mn.py")})
    return check_verdict(pin, [], spec)


# --------------------------------------------------------------------------
# LAUNCH  (mn.py launch)
# --------------------------------------------------------------------------
# Builds one script per rank (plus the PD front end) and spawns it on the pinned
# node. Returning 0 means SPAWNED; readiness is bench_e2e.sh's health loop.

# Run-scoped names the launcher sets; a swept EXTRA_ENV must not shadow them.
_OWNED = {
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "PYTHONPATH", "SGLANG_TORCH_PROFILER_DIR", "VLLM_TORCH_PROFILER_DIR",
}


def accepted_env(raw: str) -> tuple[list[str], list[str]]:
    """Keep strict IDENTIFIER=VALUE tokens (they become `env` operands, so a bare
    word would be run as the command) and drop launcher-owned names."""
    kept, dropped = [], []
    for tok in (raw or "").split():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            dropped.append(f"{tok} (not IDENTIFIER=VALUE; would be an env option/command)")
        elif tok.split("=", 1)[0] in _OWNED:
            dropped.append(f"{tok} (run-scoped name owned by the multinode launcher)")
        else:
            kept.append(tok)
    return kept, dropped


def span_sync_env(kept_env: list[str], spanning: bool) -> list[str]:
    """TP that spans nodes aborts in graph capture unless this is 1.

    Explicit off is refused: the caller asked for the setting that faults.
    Absence is filled in, because a config sweep that replaces EXTRA_ENV
    drops the variable and fails the same way, one launch later.
    """
    if not spanning:
        return list(kept_env)
    sync = [t.split("=", 1)[1] for t in kept_env
            if t.startswith("SYNC_TOKEN_IDS_ACROSS_TP=")]
    if sync and sync[-1] != "1":
        raise SystemExit(
            "!!! multinode: EXTRA_ENV turns SYNC_TOKEN_IDS_ACROSS_TP off. With TP\n    spanning nodes every rank then aborts with hipErrorCapturedEvent\n    during graph capture.")
    if not sync:
        return [*kept_env, "SYNC_TOKEN_IDS_ACROSS_TP=1"]
    return list(kept_env)


def _role_env(prefix: str, role: str) -> str:
    return (os.environ.get(f"{prefix}{(role or '').upper()}", "") or "").strip()


def pd_rank_swept(role: str, shared_args: str, shared_env: list) -> tuple:
    """(args, env, dropped) for one engine: GEAK_MN_PD_SWEEP_ARGS_/ENV_<ROLE> is
    appended after the shared sweep, so last-wins keeps it on this engine only."""
    extra = " ".join(p for p in ((shared_args or "").strip(),
                                 _role_env("GEAK_MN_PD_SWEEP_ARGS_", role)) if p)
    kept, dropped = accepted_env(_role_env("GEAK_MN_PD_SWEEP_ENV_", role))
    return extra, list(shared_env) + kept, dropped


def pd_rank_overlay(role: str, shared_overlay: str) -> str:
    """GEAK_MN_PD_OVERLAY_<ROLE> goes AHEAD of the shared overlay (first match wins)."""
    return ":".join(p for p in (_role_env("GEAK_MN_PD_OVERLAY_", role),
                                (shared_overlay or "").strip()) if p)


# topology -> (functions the adapter must define, mode function, its default, the mode we implement)
_ADAPTER_CONTRACT = {
    "tp_span": (("adapter_dist_args",), "adapter_dist_mode", "per_rank", "per_rank"),
    "pd_disagg": (("adapter_pd_args", "adapter_pd_lb_launch"), "adapter_pd_mode", "none", "pd_roles"),
}


def adapters_dir(shared: str) -> str:
    """The adapters/ every rank sources on its node: multinode.sh's own (the copy in the run
    directory) when it set GEAK_MN_ADAPTERS, else the one beside this file."""
    path = os.path.realpath(os.environ.get("GEAK_MN_ADAPTERS") or os.path.join(HERE, os.pardir, "adapters"))
    if not under(path, shared):
        raise SystemExit(f"!!! multinode: the adapters {path} are not under the shared mount {shared}, "
                         "but every rank sources them on its own node; launch through the eval_dir copy of bench_e2e.sh.")
    return path


def resolve_adapter(backend: str, shared: str, topology: str = "tp_span") -> str:
    """Locate the backend adapter and check it can speak this topology."""
    path = os.path.join(adapters_dir(shared), f"{backend}.sh")
    if not os.path.isfile(path):
        raise SystemExit(f"!!! multinode: no backend adapter at {path} (BACKEND={backend})")
    need, mode_fn, default, want = _ADAPTER_CONTRACT[topology]
    probe = subprocess.run(["bash", "-c", f". {shlex.quote(path)} >/dev/null 2>&1 || exit 3\n"
                            f"for fn in {' '.join(need)}; do declare -F $fn >/dev/null || "
                            '{ echo "missing:$fn"; exit 0; }; done\n'
                            f"declare -F {mode_fn} >/dev/null && {mode_fn} || echo {default}"],
                           capture_output=True, text=True)
    mode = probe.stdout.strip()
    if probe.returncode != 0 or not mode:
        raise SystemExit(f"!!! multinode: could not read {path}:\n{probe.stderr.strip()}")
    if mode.startswith("missing:"):
        raise SystemExit(f"!!! multinode: {path} does not define {mode[8:]}, so it cannot say how {backend} launches {topology}.")
    if mode != want:
        raise SystemExit(f"!!! multinode: {backend} declares {mode_fn}={mode}; this launcher implements {want} only.")
    return path


def under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == \
            os.path.realpath(root)
    except ValueError:
        return False


def launch_script(head: list, env: dict, adapter: str, call: str, structural: str = "", rank=None) -> str:
    """The same steps bench_e2e.sh runs locally: export, source the adapter, call it.
    A rank's server pid is recorded on its node for liveness and teardown.

    Structural flags go LAST (argparse last-wins), so no sweep can change the
    rank or the PD role."""
    q = shlex.quote
    lines = ["set -e", *head, *(f"export {k}={q(str(v))}" for k, v in env.items()),
             f". {q(adapter)}"]
    if structural:
        lines.append(f'EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS $({structural})"')
    lines += [call, *([record_pid(rank)] if rank is not None else []), "echo STARTED_PID=$SERVER_PID"]
    return "\n".join(lines) + "\n"


def rank_env_vars(*, model, tp, gpu, mem_fraction, port, base_url, log, profile_dir,
                  overlay, pythonpath, extra_env, extra_args) -> dict:
    """What bench_e2e.sh exports before adapter_launch. GPU is the PER-NODE list."""
    return {
        "MODEL": model, "TP": tp, "GPU": gpu, "MEM_FRACTION": mem_fraction,
        "HOST": "0.0.0.0", "PORT": port, "BASE_URL": base_url,
        "LOG": log, "PROFILE_DIR": profile_dir,
        "OVERLAY_PYTHONPATH": overlay, "PYTHONPATH": pythonpath,
        "EXTRA_ENV": extra_env,
        "EXTRA_SERVER_ARGS": extra_args,
        "SERVER_LAUNCH_PREFIX": "setsid",
    }


def rank_head(node: dict, master: str, log: str, profile_dir: str, launch_id: str) -> list:
    """Node facts from the pin; a NIC that lost its pinned IP fails here, not as a hung collective."""
    nic, ip, rank = node.get("cluster_if") or "", node["ip"], node["rank"]
    if not nic:
        raise SystemExit(
            f"!!! multinode: the pin has no cluster_if for rank{rank} ({node.get('hostname')}). Re-run mn.py check.")
    q = shlex.quote
    lost = f"!!! {nic} no longer has {ip}; fix handoff.multinode.nodes"
    banner = f"mn-launch id={launch_id} rank={rank} if={nic} ip={ip} master={master} log={log}"
    return [
        f"ip -4 -o addr show dev {q(nic)} | grep -qF {q(f'inet {ip}/')} || {{ echo {q(lost)} >&2; exit 1; }}",
        f"export NCCL_SOCKET_IFNAME={q(nic)} GLOO_SOCKET_IFNAME={q(nic)} MASTER_ADDR={q(master)}",
        f"mkdir -p {q(os.path.dirname(log))} {q(profile_dir)}",
        f"echo {q(banner)}",
    ]


# Read by magpie.sh / the caller's server-phase script when the bench leg sets them.
MAGPIE_PASSTHROUGH = ("MAX_MODEL_LEN", "RECIPE_ENV_FILE", "EFFECTIVE_SERVER_ARGS_COMPLETE", "PROFILE")


def tree_digest(paths: str) -> str:
    """Content fingerprint of a ':'-joined overlay path list (bytecode caches excluded)."""
    h = hashlib.sha256()
    for root in (p for p in (paths or "").split(":") if p):
        h.update(f"\0root:{root}\0".encode())
        if os.path.isfile(root):
            with open(root, "rb") as f:
                h.update(f.read())
            continue
        if not os.path.isdir(root):
            h.update(b"missing")
            continue
        for dirpath, dirs, files in os.walk(root, followlinks=True):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(f for f in files if not f.endswith((".pyc", ".lock"))):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).replace(os.sep, "/").encode() + b"\0")
                try:
                    with open(path, "rb") as f:
                        h.update(f.read())
                except OSError:
                    h.update(b"unreadable")
    return h.hexdigest()[:16]


def overlay_guard(overlay: str, digest: str, rank) -> list:
    """The rank refuses to start on an overlay other than the one GEAK measured,
    and reports the digest it loaded (launch records it in mn_launch.json).
    tree_digest goes inline: the node has no mn.py."""
    if not overlay:
        return []
    q = shlex.quote
    digest_py = f"import hashlib, os, sys\n{inspect.getsource(tree_digest)}print(tree_digest(sys.argv[1]))\n"
    bad = f"!!! rank{rank}: overlay {overlay} differs from the version GEAK pushed ({digest}); refusing to start"
    return [f"d=$(python3 -c {q(digest_py)} {q(overlay)} || echo unreadable)",
            f'[ "$d" = {q(digest)} ] || {{ echo {q(bad)} >&2; exit 1; }}',
            'echo "OVERLAY_DIGEST=$d"']


def command_script(backend: str, command: str, geak_env: list, adapters: str) -> str:
    """A caller's launch_command as a Magpie server-phase script. magpie.sh runs it under GEAK's
    layer (recipe, EXTRA_ENV, GPU mask, overlay PYTHONPATH, profiler dir) and passes GEAK's flags
    in EXTRA_<BACKEND>_ARGS, appended after the command's own (last-wins). The command's leading
    env is applied inside, so a name GEAK's EXTRA_ENV also sets is left to GEAK."""
    q = shlex.quote
    env, argv = split_command(command)
    mine = {kv.split("=", 1)[0] for kv in geak_env}
    base = [kv for kv in env if kv.split("=", 1)[0] not in mine]
    lines = ["#!/usr/bin/env bash", "# Rendered by mn.py launch from the caller's launch_command.", "prof=()"]
    if backend == "vllm":
        # vllm >=0.19 reads the profiler from the command line only; magpie's step bound goes after it.
        lines.append(f'if [ "${{PROFILE:-0}}" = 1 ]; then pj="$(. {q(os.path.join(adapters, "vllm.sh"))} '
                     '&& adapter_profiler_args)"; [ -z "$pj" ] || prof=(--profiler-config "$pj"); fi')
    launch = ["setsid env --", *(q(kv) for kv in base), shlex.join(argv), '${prof[@]+"${prof[@]}"}',
              f'$EXTRA_{backend.upper()}_ARGS > "$SERVER_LOG" 2>&1 < /dev/null &']
    lines += ["set -f", " ".join(launch), 'echo $! > "$MAGPIE_SERVER_PID_FILE"']
    return "\n".join(lines) + "\n"


def provided_script(head: list, env: dict, launch_script: str, rank_dir: str, rank, magpie: str) -> str:
    """A caller's server-phase script under GEAK's layer: magpie.sh maps EXTRA_SERVER_ARGS /
    EXTRA_ENV / OVERLAY_PYTHONPATH onto it. Structural flags are the caller's, never added here.
    Backgrounded: the server phase waits for readiness, far past the spawn's timeout. The
    wrapper's pid is recorded at once, magpie's server pid once its server phase is up."""
    q = shlex.quote
    inner = f'. "$GEAK_MN_MAGPIE"; adapter_launch && {record_pid(rank)}'
    lines = ["set -e", *head, *(f"export {k}={q(str(v))}" for k, v in env.items()),
             f"export MAGPIE_LAUNCH_SCRIPT={q(launch_script)} OUT_DIR={q(rank_dir)} GEAK_MN_MAGPIE={q(magpie)}",
             f"mkdir -p {q(rank_dir)}",
             f"setsid bash -c {q(inner)} >> {q(rank_dir + '/magpie_phase.log')} 2>&1 < /dev/null &",
             "pid=$!", record_pid(rank, "$pid"), "echo STARTED_PID=$pid"]
    return "\n".join(lines) + "\n"


def pd_lb_script(state, pd, out_dir, launch_id, *, router: str = "", adapter: str = "",
                 model: str = "", pythonpath: str = "", extra_env: str = "") -> str:
    """The PD front end, on lb_node. It holds no GPU and joins no collective, so no node
    facts. `router`: the caller's router launch_command, run as given on its own machine (check
    proved it names exactly the engines) with its pid recorded, since no name pattern knows
    every backend's router; else the backend adapter's adapter_pd_lb_launch."""
    q = shlex.quote
    urls = {r: " ".join(e["url"] for e in engine_urls(state) if e["role"] == r) for r in pd["roles"]}
    log, port = os.path.join(out_dir, "mn_lb.log"), pd["lb_port"]
    head = [f"mkdir -p {q(out_dir)}", f"echo {q(f'mn-launch id={launch_id} role=lb port={port} log={log}')}"]
    if router:
        return "\n".join(["set -e", *head,
                          f"setsid bash -c {q(router)} >> {q(log)} 2>&1 < /dev/null &",
                          "pid=$!", record_pid("lb", "$pid"), "echo STARTED_PID=$pid"]) + "\n"
    call = " ".join(q(str(a)) for a in ("0.0.0.0", port, urls["prefill"], pd["bootstrap_port"], urls["decode"]))
    return launch_script(
        head,
        {"MODEL": model, "LOG": log, "PYTHONPATH": pythonpath, "EXTRA_ENV": extra_env,
         "LB_EXTRA_ARGS": "--mini-lb" if pd["lb"] == "minilb" else "",
         "SERVER_LAUNCH_PREFIX": "setsid"},
        adapter, f"adapter_pd_lb_launch {call}")


def check_script(rank, script: str) -> None:
    """Catch a broken script before the spawn, not as a dead rank."""
    proc = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"!!! multinode: rank{rank}'s rendered script is not valid bash:\n{proc.stderr.strip()}")


def _last_line(text: str) -> str:
    return text.splitlines()[-1] if text else ""


def cmd_launch(argv: list) -> int:
    env = os.environ.get
    state = load_state()
    shared = state["shared_fs"]
    topology = topology_of(state)
    pd = pd_config(state) if topology == "pd_disagg" else None

    model = env("MODEL", "")
    if not model:
        raise SystemExit("!!! multinode: MODEL is required")
    nnodes = len(state["nodes"])
    tp = int(env("TP", 0) or 0)
    mem_fraction = env("MEM_FRACTION", "0.90")
    extra_args = (env("EXTRA_SERVER_ARGS", "") or "").strip()
    provided = state.get("mode") == "provided"
    # A provided script owns the launch line, so no backend adapter speaks for the ranks.
    adapter = "" if provided else resolve_adapter(env("BACKEND", "sglang"), shared, topology)
    overlay = (env("OVERLAY_PYTHONPATH", "") or "").strip()
    profile_dir = env("PROFILE_DIR", "")
    log_path = env("LOG", "")
    out_dir = env("OUT_DIR", os.path.dirname(log_path) or shared)

    if any("tp" not in n or "gpus" not in n for n in state["nodes"]):
        raise SystemExit("!!! multinode: the pin records no per-rank TP/cards; re-run mn.py check.")
    # The pin's TP is the deployment; a tp_span leg asking for another TP measured another one.
    if pd is None and tp and tp != state["nodes"][0]["tp"]:
        raise SystemExit(f"!!! multinode: TP={tp} but the pin was checked for TP={state['nodes'][0]['tp']}; re-run mn.py check for the new TP.")
    role_of = {n["rank"]: node_role(state, n) for n in state["nodes"]} if pd is not None else {}
    if pd is not None:
        print(">>> multinode: pd_disagg, TP per engine " + ", ".join(
            f"{e['role']}={e['node']['tp']}" + (f" over {len(e['nodes'])} nodes" if len(e["nodes"]) > 1 else "")
            + (f" x{sum(1 for x in engine_urls(state) if x['role'] == e['role'])} replicas"
               if any(x["replica"] for x in engine_urls(state) if x["role"] == e["role"]) else "")
            for e in engine_urls(state) if not e["replica"]))
    backend = state.get("backend") or env("BACKEND", "sglang")

    for name, path in (("LOG", log_path), ("PROFILE_DIR", profile_dir)):
        if not path:
            raise SystemExit(f"!!! multinode: {name} is required")
        if not under(path, shared):
            raise SystemExit(
                f"!!! multinode: {name}={path} is not under the shared mount {shared}.\n    Off the shared mount each node writes its own copy and the\n    benchmark host reads nothing. Put OUT_DIR under {shared}.")

    kept_env, dropped_env = accepted_env(env("EXTRA_ENV", ""))
    for tok in dropped_env:
        print(f">>> multinode: dropping EXTRA_ENV token: {tok}", file=sys.stderr)
    # TP across nodes only. Absence is injected: a sweep that replaces EXTRA_ENV drops
    # the variable and every rank then dies in graph capture.
    _before_sync = list(kept_env)
    kept_env = span_sync_env(kept_env, spanning=pd is None and nnodes > 1)
    if kept_env != _before_sync:
        print(">>> multinode: injected SYNC_TOKEN_IDS_ACROSS_TP=1 (tp spanning nodes, caller did not set it)", file=sys.stderr)

    pythonpath = env("PYTHONPATH", "")

    launch_id = uuid.uuid4().hex[:10]
    magpie = os.path.join(adapters_dir(shared), "launchers", "magpie.sh") if provided else ""
    scripts, effective_by_role, digests, rank_logs, args_dropped = {}, {}, {}, {}, {}
    for node in state["nodes"]:
        rank = node["rank"]
        # rank0's log IS $LOG (bench_e2e.sh's stall/OOM detector reads it).
        rank_log = log_path if rank == 0 else os.path.join(out_dir, f"server.rank{rank}.log")
        rank_logs[rank] = rank_log
        rank_extra, rank_env, rank_overlay = extra_args, list(kept_env), overlay
        master = state["master_ip"]
        if pd is None:
            port = state["serve_port"]
            base_url = f"http://{state['master_ip']}:{port}"
            structural = (f'adapter_dist_args {rank} {nnodes} "$MASTER_ADDR" {state["dist_init_port"]}')
        else:
            role = role_of[rank]
            engine = next(e for e in engine_urls(state)
                          if e["role"] == role and e["replica"] == node.get("replica", 0))
            port, head_node, base_url = engine["port"], engine["node"], engine["url"]
            structural = (f'adapter_pd_args {role} {pd["bootstrap_port"]} {pd["transfer_backend"]} {pd["ib_device"]}').rstrip()
            spans = int(node.get("engine_nnodes") or 1) > 1
            if spans:
                # One engine over several nodes: its own dist-init world, mastered by its node-rank 0.
                master = head_node["ip"]
                structural = (f'echo $({structural}) $(adapter_dist_args {node["engine_rank"]} '
                              f'{node["engine_nnodes"]} "$MASTER_ADDR" {state["dist_init_port"]})')
            rank_extra, rank_env, sweep_dropped = pd_rank_swept(role, extra_args, kept_env)
            for tok in sweep_dropped:
                print(f">>> multinode: dropping {role} sweep env token: {tok}", file=sys.stderr)
            if spans:
                rank_env = span_sync_env(rank_env, spanning=True)
            rank_overlay = pd_rank_overlay(role, overlay)
        if provided:
            # The caller's command owns its port, model, TP and PD wiring; a swept copy is dropped.
            rank_extra, args_dropped[rank] = strip_flags(rank_extra, structural_flags(backend))
            for tok in args_dropped[rank]:
                print(f">>> multinode: rank{rank}: dropping {tok!r}; the launch_command owns it", file=sys.stderr)
        if pd is not None:
            effective_by_role[role] = {
                "extra_args": rank_extra,
                "env": " ".join(rank_env), "overlay": rank_overlay,
            }
        if rank_overlay not in digests:
            digests[rank_overlay] = tree_digest(rank_overlay)
        guard = overlay_guard(rank_overlay, digests[rank_overlay], rank)
        rank_vars = rank_env_vars(model=model, tp=node["tp"],
                                  gpu=",".join(str(i) for i in node.get("gpu_ids") or range(node["gpus"])),
                                  mem_fraction=mem_fraction,
                                  port=port, base_url=base_url, log=rank_log,
                                  profile_dir=profile_dir, overlay=rank_overlay,
                                  pythonpath=pythonpath, extra_env=" ".join(rank_env),
                                  extra_args=rank_extra)
        if provided:
            wrapper = os.path.join(out_dir, f"mn_cmd_rank{rank}.sh")
            with open(wrapper, "w", encoding="utf-8") as f:
                f.write(command_script(backend, node["launch_command"], rank_env, adapters_dir(shared)))
            banner = f"mn-launch id={launch_id} rank={rank} command={wrapper} log={rank_log}"
            head = [f"mkdir -p {shlex.quote(os.path.dirname(rank_log))} {shlex.quote(profile_dir)}",
                    f"echo {shlex.quote(banner)}", *guard]
            rank_vars.update(BACKEND=backend, **{k: env(k) for k in MAGPIE_PASSTHROUGH if env(k)})
            # The command is the whole recorded launch; magpie must not prepend the recipe's flags again.
            rank_vars["EFFECTIVE_SERVER_ARGS_COMPLETE"] = "1"
            scripts[rank] = provided_script(head, rank_vars, wrapper,
                                            os.path.join(out_dir, f"rank{rank}"), rank, magpie)
        else:
            scripts[rank] = launch_script(
                rank_head(node, master, rank_log, profile_dir, launch_id) + guard,
                rank_vars, adapter, "adapter_launch", structural, rank)

    # The LB is not one of `scripts`: those are the GPU-holding engines.
    lb_script = None
    if pd is not None and state.get("router"):
        lb_script = pd_lb_script(state, pd, out_dir, launch_id, router=state["router"]["launch_command"])
    elif pd is not None:
        lb_script = pd_lb_script(state, pd, out_dir, launch_id,
                                 adapter=adapter or resolve_adapter(env("BACKEND", "sglang"), shared, topology),
                                 model=model, pythonpath=pythonpath, extra_env=" ".join(kept_env))

    rendered = {f"rank{r}": s for r, s in sorted(scripts.items())}
    if lb_script is not None:
        rendered["lb"] = lb_script
    for name, script in rendered.items():
        check_script(name[4:] if name.startswith("rank") else name, script)
    for name, script in rendered.items():
        with open(os.path.join(out_dir, f"mn_{name}.sh"), "w", encoding="utf-8") as f:
            f.write(script)

    # Before touching the cluster: a dry run must not tear down a live server.
    if env("GEAK_MN_DRY_RUN") == "1":
        print(">>> multinode: dry run -- rendered scripts written, nothing launched")
        for name, script in rendered.items():
            print(f"--- {name} ---\n{script}")
        return 0

    probes = teardown(state)
    if not probes:
        raise SystemExit(COTENANT_REFUSAL)
    drifted = [f"{_short(p['hostname'])} {p.get('sw')}" for p in probes if (p.get("sw") or {}) != state.get("sw", {})]
    if drifted:
        raise SystemExit(f"!!! multinode: the software stack changed since mn.py check (pinned {state.get('sw')}): "
                         f"{'; '.join(drifted)}.\n    Legs before and after would measure different stacks. Re-run mn.py check.")

    nodes = [n for n in resolve(state) if n["rank"] in scripts]
    results = ssh_each(state, [(n, scripts[n["rank"]], 300) for n in nodes], merged=True)

    # The LB goes up AFTER the engines; the router polls its workers before serving.
    lb_result = None
    if lb_script is not None:
        lb_result = ssh_run(state, lb_node(state), lb_script, 300, merged=True)

    summary = {
        "launch_id": launch_id,
        "launched_at": _utc(),
        "master": state["master_ip"],
        "serve_port": state["serve_port"],
        "dist_init_port": state["dist_init_port"],
        "base_url": f"http://{state['master_ip']}:{state['serve_port']}",
        "nnodes": nnodes,
        "log": log_path,
        "profile_dir": profile_dir,
        "overlay_pythonpath": overlay,
        "backend_adapter": adapter,
        "extra_server_args": extra_args,
        "extra_env_accepted": kept_env,
        "extra_env_dropped": dropped_env,
        "ranks": [
            {"rank": n["rank"], "hostname": r["hostname"], "ip": r["ip"],
             "tp": n["tp"], "gpus": n["gpus"], "rc": r["rc"], "out": r["out"],
             **({"extra_args_dropped": args_dropped[n["rank"]]} if args_dropped.get(n["rank"]) else {}),
             **({"overlay_digest": m.group(1)} if (m := re.search(r"^OVERLAY_DIGEST=(\S+)", r["out"] or "", re.M)) else {})}
            for n, r in zip(nodes, results)
        ],
        "pin_checked_at": state.get("checked_at"),
        # What the supervisor tears down, independent of any later pin.
        "nodes": [{k: n.get(k) for k in LAUNCH_NODE_KEYS} for n in state["nodes"]],
        **({"router": {k: state["router"].get(k) for k in LAUNCH_NODE_KEYS}}
           if (state.get("router") or {}).get("node_id") else {}),
        "topology": topology,
        # Two legs are comparable only if these agree (same machines, stack and code).
        "mode": state.get("mode", "generated"),
        "node_ids": [n["node_id"] for n in state["nodes"]],
        "sw": state.get("sw", {}),
        "overlay_digests": {k: v for k, v in digests.items() if k},
    }
    if pd is not None:
        # The commands own their ports and transfer backend; pd_config's would be fiction here.
        generated_only = ("role_ports", "bootstrap_port", "transfer_backend", "ib_device") if provided else ()
        summary["pd"] = {
            **{k: v for k, v in pd.items() if k not in generated_only},
            "engine_tp_by_role": {role_of[n["rank"]]: n["tp"] for n in state["nodes"]},
            "effective_args_by_role": effective_by_role,
            # The profile window and the agents address engines from here (the LB proxies
            # no profiler). log: the engine's front-end rank log; tp: its rank traces.
            "engines": [
                {"role": e["role"], "replica": e["replica"], "rank": e["node"]["rank"],
                 "host": _short(e["node"]["hostname"]), "ip": e["node"]["ip"],
                 "port": e["port"], "url": e["url"], "tp": e["node"]["tp"],
                 "log": rank_logs[e["node"]["rank"]],
                 **({"bootstrap_port": e["node"]["bootstrap_port"]} if e["node"].get("bootstrap_port") else {}),
                 **({"node_ranks": [n["rank"] for n in e["nodes"]]} if len(e["nodes"]) > 1 else {})}
                for e in engine_urls(state)
            ],
            "lb_url": f"http://{lb_node(state)['ip']}:{pd['lb_port']}",
            "lb_host": lb_node(state)["ip"],
            "lb_log": os.path.join(out_dir, "mn_lb.log"),
            "lb_rc": (lb_result or {}).get("rc"),
            "lb_out": (lb_result or {}).get("out", ""),
        }
    _write_json(os.path.join(out_dir, "mn_launch.json"), summary)
    for r in summary["ranks"]:
        role = f" ({role_of[r['rank']]})" if pd else ""
        print(f">>> multinode rank{r['rank']}{role} on {_short(r['hostname'])} rc={r['rc']} :: {_last_line(r['out'])}")

    if any(r["rc"] != 0 for r in summary["ranks"]):
        return abandon_launch(state)

    if lb_result is not None:
        print(f">>> multinode lb on {_short(lb_result['hostname'])} rc={lb_result['rc']} :: {_last_line(lb_result['out'])}")
        if lb_result["rc"] != 0:
            print(f"!!! multinode: the load balancer spawn failed; see {summary['pd']['lb_log']}.", file=sys.stderr)
            return abandon_launch(state)

    counts = serve_counts(state)
    print(f">>> multinode: serve processes per rank = {counts}")
    if any(c < 1 for c in counts):
        print("!!! multinode: at least one rank has no serve process after the spawn; see the rank logs.", file=sys.stderr)
        return abandon_launch(state)
    return 0


def abandon_launch(state: dict) -> int:
    """No supervisor exists yet, so reap whatever did spawn."""
    try:
        teardown(state, wait_idle_s=TEARDOWN_WAIT_S)
    except Exception as exc:
        print(f"!!! multinode: reaping the failed launch raised: {exc}", file=sys.stderr)
    return 1


def cmd_stop(argv: list) -> int:
    """`mn.py stop <out_dir>`: reap the launch in out_dir when no supervisor came up to do it."""
    out_dir = argv[0] if argv else os.environ.get("OUT_DIR", ".")
    return 0 if teardown(supervised_state(load_state(), out_dir)) else 1


# --------------------------------------------------------------------------
# SUPERVISE  (mn.py supervise)
# --------------------------------------------------------------------------
# The local pid bench_e2e.sh watches and kills. It exits when a rank stops
# serving (no process, zombie workers, or a dead PD front end) so the bench sees
# `died_early` within one poll, and its SIGTERM handler reaps the remote ranks
# (after a kill -9 they hold the cards until the next launch's teardown).
# It never writes to $LOG (bench_e2e.sh greps it).

SUP_POLL_S = 20
MISSES = 2          # empty observations before "dead"
ERR_MISSES = 6      # probe errors before giving up
# Must finish inside server_teardown.sh's TERM->KILL grace (SERVER_STOP_GRACE_S=90).
TEARDOWN_WAIT_S = 60

_state = None
_out_dir = os.environ.get("OUT_DIR", ".")
_stopping = False


def record(verdict: str, detail: dict) -> None:
    try:
        _write_json(os.path.join(_out_dir, "mn_supervisor.json"),
                    {"verdict": verdict, "at": _utc(), **detail})
    except OSError as exc:
        log(f"could not write mn_supervisor.json: {exc}")


def on_term(signum, _frame):
    """Reap the remote ranks and exit 0 (a requested stop is not a crash)."""
    global _stopping
    if _stopping:
        return
    _stopping = True
    log(f"signal {signum}: tearing down the remote ranks")
    try:
        cleared = teardown(_state, wait_idle_s=TEARDOWN_WAIT_S)
        log("remote ranks reaped" if cleared else
            "remote teardown did not fully clear; the next launch will wait for the cards")
        record("torn_down", {"signal": int(signum), "cleared": bool(cleared)})
    except Exception as exc:
        log(f"remote teardown raised: {exc}")
        record("torn_down", {"signal": int(signum), "error": str(exc)})
    sys.exit(0)


def _death_reason(rank: dict) -> str:
    if rank["serve"] < 1:
        return "no serve process"
    if rank["zombies"] > 0:
        return (f"{rank['zombies']} zombie worker(s) -- the engine aborted while the HTTP front end kept answering")
    return ("the disaggregation load balancer is gone -- both engines are up but nothing serves the client port")


def cmd_supervise(argv: list) -> int:
    global _state
    _state = supervised_state(load_state(), _out_dir)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    if signal.getsignal(signal.SIGTERM) is not on_term:
        log("!!! SIGTERM is not ours; the ranks would outlive this process")
        record("handler_lost", {})
        return 1
    ranks = [f"rank{n['rank']}@{_short(n['hostname'])}" for n in _state["nodes"]]
    log(f"watching {', '.join(ranks)} every {SUP_POLL_S}s (dead after {MISSES} empty polls, {ERR_MISSES} failed probes)")
    record("watching", {"ranks": ranks, "poll_s": SUP_POLL_S})

    empty = [0] * len(state_targets(_state))
    errors = 0
    while True:
        time.sleep(SUP_POLL_S)
        try:
            health = engine_health(_state)
            errors = 0
        except Exception as exc:
            errors += 1
            log(f"probe failed ({errors}/{ERR_MISSES}): {exc}")
            if errors >= ERR_MISSES:
                record("unobservable", {"error": str(exc), "consecutive_failures": errors})
                log("giving up: the ranks have been unobservable for too long")
                return 1
            continue

        for i, rank in enumerate(health):
            gone = rank["serve"] < 1 or rank["zombies"] > 0 or rank.get("lb", 1) < 1
            empty[i] = empty[i] + 1 if gone else 0

        dead = [i for i, misses in enumerate(empty) if misses >= MISSES]
        if dead:
            names = [f"rank{health[i]['rank']}@{health[i]['host']}" for i in dead]
            why = "; ".join(f"rank{health[i]['rank']}: {_death_reason(health[i])}" for i in dead)
            log(f"{why} -- exiting so the benchmark sees a dead server now instead of hanging until the stall window expires")
            record("rank_died", {"dead_ranks": names, "why": why, "health": health})
            # Survivors are blocked in a collective that will never complete.
            try:
                teardown(_state, wait_idle_s=60)
            except Exception as exc:
                log(f"post-mortem teardown raised: {exc}")
            return 1


# --------------------------------------------------------------------------
# EXEC  (mn.py exec)
# --------------------------------------------------------------------------
# The driver GEAK runs on sees no GPU; this reaches a pinned node for op benches, shape
# captures, kernel tests and one-card servers. --gpu leases the cards so parallel tasks
# never share one.

def exec_target(state: dict, name: str) -> dict:
    """master | rank<N> | prefill | decode | a pinned IP or hostname."""
    nodes, pd = state["nodes"], topology_of(state) == "pd_disagg"
    if name == "master":
        return next((n for n in nodes if n["ip"] == state.get("master_ip")), nodes[0])
    if re.fullmatch(r"rank\d+", name) and int(name[4:]) < len(nodes):
        return nodes[int(name[4:])]
    if pd and name in PD_ROLES and engine_nodes(state, name):
        return engine_nodes(state, name)[0]
    for n in nodes:
        if name in (n["ip"], n["hostname"], _short(n["hostname"])):
            return n
    raise SystemExit(f"!!! multinode exec: no pinned node {name!r}; use master, rank0..rank{len(nodes) - 1}"
                     + (", prefill, decode" if pd else "") + " or a pinned host")


def lease_dir(state: dict) -> str:
    return os.path.join(state["shared_fs"], ".mn_lease")


def lease_file(state: dict, node: dict, gpu) -> str:
    """One lock file per card on the shared mount: exec tasks, gpu_lock.sh and serving legs
    (multinode.sh) take the same files, so no two GEAK tasks hold one card. gpu_lock.sh
    writes the same name itself (its LOCK_PREFIX)."""
    os.makedirs(lease_dir(state), exist_ok=True)
    return os.path.join(lease_dir(state), f"{_short(node['hostname'])}.gpu{gpu}.lock")


@contextlib.contextmanager
def gpu_lease(state: dict, node: dict, gpus: str):
    """Exclusive per-card locks held while the command runs; yields the leased id(s)."""
    ids = sorted({int(g) for g in gpus.split(",") if g})   # one order: no deadlock
    if not ids:
        yield ""
        return
    try:
        import fcntl
    except ImportError:                                 # pragma: no cover - not Linux
        yield gpus
        return
    host = _short(node["hostname"])
    held = []
    try:
        for gpu in ids:
            f = open(lease_file(state, node, gpu), "w")
            held.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f">>> mn exec: waiting for {host} gpu{gpu}, leased by another task",
                      file=sys.stderr, flush=True)
                fcntl.flock(f, fcntl.LOCK_EX)
        yield ",".join(map(str, ids))
    finally:
        for f in held:
            f.close()


# Never shipped: this host's own paths and identity (the claude binary included), and card masks.
LOCAL_ENV = re.compile(r"\w*_VISIBLE_DEVICES|BASH\w*|SHLVL|PATH|HOME|PWD"
                       r"|OLDPWD|SHELL|USER|LOGNAME|HOSTNAME|TERM|TMPDIR|LANG|LC_\w+|LD_LIBRARY_PATH|LD_PRELOAD"
                       r"|PYTHONHOME|PYTHONPATH|VIRTUAL_ENV|CONDA_\w+|SSH_\w+|DISPLAY|MAIL\w*|XDG_\w+"
                       r"|(GEAK_)?CLAUDE_BIN")
# Credentials stay on the machine GEAK runs on.
CREDENTIAL_ENV = re.compile(r"\w*(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)\w*|ANTHROPIC_\w+|CLAUDE\w*")


def exported_env() -> str:
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(os.environ.items())
                   if re.fullmatch(r"[A-Z_][A-Z0-9_]*", k) and not LOCAL_ENV.fullmatch(k)
                   and not CREDENTIAL_ENV.fullmatch(k))


# Once the caller's ssh is gone, the command's session gets TERM and, this much later, KILL:
# room for bench_e2e.sh to tear down a one-card server (TERM->KILL 90s) and a bench to clean up.
EXEC_GRACE_S = 180


def exec_script(body: str, grace: int = EXEC_GRACE_S) -> str:
    """`body` in its own session on the node, watched for the caller hanging up. GEAK_MN_HOP
    tells bench_e2e.sh, gpu_lock.sh and profile_kernel.sh they already run on the node.
    One brace group: bash parses all of it before running any, so the watchdog's `cat`
    reads only what comes after the script -- nothing, until EOF."""
    q = shlex.quote
    watchdog = f'cat >/dev/null; pkill -TERM -s "$1"; sleep {grace}; pkill -KILL -s "$1"'
    eof = f"GEAK_MN_BODY_{uuid.uuid4().hex}"
    return ("{\n"
            "export GEAK_MN_HOP=1\n"
            'body=$(mktemp) || exit 97\n'
            f"cat > \"$body\" <<'{eof}'\n{body}\n{eof}\n"
            'setsid bash "$body" </dev/null &\n'
            "pid=$!\n"
            f"setsid bash -c {q(watchdog)} _ \"$pid\" <&0 >/dev/null 2>&1 &\n"
            'wait "$pid"\n'
            'rc=$?; rm -f "$body"; exit $rc\n'
            "}\n")


def cmd_exec(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="mn.py exec",
                                 description="Run a command on a pinned node, in this directory.")
    ap.add_argument("--target", default="", help="master (default) | rank<N> | prefill | decode | a pinned host")
    ap.add_argument("--gpu", default="",
                    help="[<target>:]card id(s) on that node, e.g. 3, 0,1 or decode:3: leased, "
                         "exported as HIP_VISIBLE_DEVICES and GPU_ID")
    ap.add_argument("--script", default="", help="bash script file to run instead ('-' = stdin)")
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--env", action="store_true", help="also export this shell's environment there")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if bool(cmd) == bool(a.script):
        ap.error("give either a command after -- or --script")
    spec = re.fullmatch(r"(?:([\w.-]+):)?(\d+(?:,\d+)*)", a.gpu) if a.gpu else None
    if a.gpu and not spec:
        ap.error(f"--gpu takes card ids such as 3, 0,1 or decode:3; got {a.gpu!r}")
    prefix, gpus = spec.groups() if spec else (None, "")
    if prefix and a.target and prefix != a.target:
        ap.error(f"--gpu {a.gpu} names {prefix!r} but --target is {a.target!r}")
    if a.script:
        with (contextlib.nullcontext(sys.stdin) if a.script == "-" else open(a.script, encoding="utf-8")) as f:
            body = f.read()
    else:
        body = shlex.join(cmd)
    state = load_state()
    node = exec_target(state, a.target or prefix or "master")
    with gpu_lease(state, node, gpus) as leased:
        script = (f"cd {shlex.quote(os.getcwd())} 2>/dev/null || true\n"
                  + (exported_env() if a.env else "")
                  + (f"export HIP_VISIBLE_DEVICES={leased} GPU_ID={leased}\n" if leased else "")
                  + body + "\n")
        return ssh_stream(state, node, exec_script(script), a.timeout or None)


# --------------------------------------------------------------------------
# LEG ENV  (mn.py leg-env)
# --------------------------------------------------------------------------

def cmd_leg_env(argv: list) -> int:
    """The pin's facts multinode.sh needs, as `_MN_<NAME>=<value>` lines for eval. Exits 3
    on a pin it cannot serve from, before multinode.sh takes any lease or lock."""
    state = load_state()
    if not state.get("master_ip") or not state.get("serve_port"):
        sys.stderr.write(f"!!! multinode launcher: {state_path()} has no master_ip/serve_port; re-run mn.py check.\n")
        return 3
    topology = topology_of(state)
    facts = {
        "HOST": state["master_ip"], "PORT": state["serve_port"], "TOPOLOGY": topology,
        "NNODES": len(state["nodes"]), "SHARED_FS": state["shared_fs"],
        # The serving lock is keyed on the pinned node set, so runs on disjoint sets do not serialise.
        "LOCK_KEY": "+".join(sorted(_short(n["hostname"]).lower() for n in state["nodes"])),
        # One engine per role (replica 0) opens the profile window; health asks every replica.
        "PD_ENGINES": "\n".join(f"{e['role']}={e['url']}" for e in engine_urls(state) if not e["replica"])
                      if topology == "pd_disagg" else "",
        "PD_HEALTH": "\n".join(e["url"] for e in engine_urls(state)) if topology == "pd_disagg" else "",
        # Once per card: ranks sharing a machine name the same files, and multinode.sh would
        # block on its own lock taking one twice.
        "LEASES": "\n".join(dict.fromkeys(lease_file(state, n, g) for n in state["nodes"] for g in range(n["ngpu"]))),
    }
    print("".join(f"_MN_{k}={shlex.quote(str(v))}\n" for k, v in facts.items()), end="")
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def cmd_probe(argv: list) -> int:
    """probe_node on this host; check runs it on every node over ssh."""
    print(PROBE_TAG + json.dumps(probe_node([int(p) for p in argv])), flush=True)
    return 0


def cmd_digest(argv: list) -> int:
    print(tree_digest(argv[0] if argv else ""))
    return 0


def _offline(name: str):
    """trace-gate / topn-merge / pd-mass live in mn_traces.py beside this file. Imported on use:
    run_e2e loads mn.py by path, where this directory is not on sys.path."""
    def run(argv: list) -> int:
        import mn_traces
        return getattr(mn_traces, name)(argv)
    return run


COMMANDS = {
    "check": cmd_check,
    "launch": cmd_launch,
    "supervise": cmd_supervise,
    "stop": cmd_stop,
    "leg-env": cmd_leg_env,
    "active-pin": cmd_active_pin,
    "exec": cmd_exec,
    "digest": cmd_digest,
    "trace-gate": _offline("cmd_trace_gate"),
    "topn-merge": _offline("cmd_topn_merge"),
    "pd-mass": _offline("cmd_pd_mass"),
    "pd-points": _offline("cmd_pd_points"),
    "probe": cmd_probe,
}
TAGS = {"supervise": "mn-sup"}


def main(argv: list) -> int:
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write("usage: mn.py {%s} [options]\n" % "|".join(COMMANDS))
        return 2
    global _LOG_TAG
    _LOG_TAG = TAGS.get(argv[0], _LOG_TAG)
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
