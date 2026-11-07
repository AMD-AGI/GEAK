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
import errno
import functools
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
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
    return state.get("topology") or "tp_span"


COTENANT_REFUSAL = (
    "!!! multinode: the handoff's nodes did not come clear. Launching now would\n    hit the held dist-init port and produce an orphaned rank1. Free them or\n    name other machines in node_state.json's pods or args.multinode.nodes.")


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


def plain_argv0(command: str) -> str:
    """The program a launch_command runs when it is one plain command; '' for a compound
    line (`source venv/bin/activate && ...`) or a path, which a PATH lookup cannot judge."""
    try:
        _env, argv = split_command(command)
    except ValueError:
        return ""
    if not argv or "/" in argv[0] or any(t in ("&&", "||", "|", "&") or ";" in t for t in argv):
        return ""
    return argv[0]


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
    """args.multinode plus the engine TP, validated; the one place the block is parsed.
    check pins the result. Raises ValueError.

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
        host = str(d.get("host") or "").strip()
        if not host:
            raise ValueError(f"nodes[{i}] names no host")
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
    args.multinode.pd. The LB owns serve_port, so the client side needs no notion
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


def engine_nodes(state: dict, role: str) -> list:
    """The pinned nodes of one PD engine, its node-rank 0 (HTTP front end) first."""
    return [n for n in state["nodes"] if n.get("role") == role]


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


def _resident_gpu_pids(showpids: str) -> list[str]:
    """The pids holding VRAM, from `rocm-smi --showpids`.

    The host's gpuagent daemon holds a 0-VRAM KFD context; it is not occupancy."""
    return [pid for pid, vram in re.findall(r"^\s*(\d+)\s+\S+\s+\d+\s+(\d+)", showpids, re.M) if int(vram) > 0]


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


def port_listened(port: int) -> bool:
    """Something LISTENs on `port` here. Not `ss`: some images lack iproute2. SO_REUSEADDR
    lets the bind past a just-stopped server's TIME_WAIT, as `ss -ltn` would."""
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            s.bind((addr, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
        finally:
            s.close()
    return False


def probe_node(ports: list | None = None, cmds: list | None = None) -> dict:
    """Runs ON the node (`mn.py probe`), in the shell a launch gets: `cmds` resolve here as there."""
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
    busy_ports = [int(p) for p in (ports or []) if port_listened(int(p))]
    missing_cmds = [c for c in cmds or [] if not shutil.which(c)]
    resident_pids = _resident_gpu_pids(sh("rocm-smi --showpids 2>/dev/null || true"))
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
        "missing_cmds": missing_cmds,
        "sw": _versions(),
        "ngpu": len(uses),
        "max_use": max(uses) if uses else None,
        "gpu_pids": len(resident_pids),
        # rocm-smi reports HOST pids: one with no /proc entry here is another container's.
        "foreign_gpu_pids": sum(1 for p in resident_pids if not os.path.exists(f"/proc/{p}")),
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


SSH_KEYS = ("key", "port", "user", "known_hosts", "prefix")
PROBE_TAG = "MN_PROBE="
IDENTITY_SH = 'echo "$(cat /proc/sys/kernel/hostname)/$(cat /proc/sys/kernel/random/boot_id)"\n'
# An sshd session lacks the container's env (k8s-injected NCCL/ROCm vars); take
# pid 1's value for every name the session left unset.
POD_ENV = ("while IFS= read -r -d '' kv; do k=${kv%%=*}; "
           '[[ $k =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!k+x}" ] && export "$kv"; '
           "done 2>/dev/null < /proc/1/environ\n")


def ssh_config(cfg: dict) -> dict:
    """args.multinode.ssh, validated (the pin's ssh block). A node's own `ssh` block
    has the same keys and overrides it."""
    if not isinstance(cfg, dict):
        raise ValueError("ssh settings must be a JSON object")
    unknown = sorted(set(cfg) - set(SSH_KEYS))
    if unknown:
        raise ValueError(f"unknown ssh key(s) {unknown}; known: {', '.join(SSH_KEYS)}")
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


DRIVER_BIN_PREFIX = "geak_mn_driver_bin_"


def real_path() -> str:
    """PATH without the GPU-tool stubs run_e2e.py puts in front of the driver's agents: a
    node's shell, and the driver's own GPU count, need the real rocm-smi and rocminfo."""
    return os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if not os.path.basename(p).startswith(DRIVER_BIN_PREFIX))


def ssh_argv(state: dict, node: dict) -> list[str]:
    """The command that runs a script (on its stdin) on `node`: a fresh local shell, as
    ssh would give, when that node is this machine. Before the pin an address cannot tell
    the host from the container a `prefix` enters, so such a node is always reached by ssh."""
    cfg = {**(state.get("ssh") or {}), **(node.get("ssh") or {})}
    if is_self(node) and (node.get("node_id") or not cfg.get("prefix")):
        return ["env", "-i", f"PATH={real_path()}", f"HOME={os.environ.get('HOME', '')}",
                "bash", "-s"]
    argv = ["ssh", *SSH_OPTS, "-p", str(int(cfg.get("port") or 22))]
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


def probe(state: dict, node: dict, ports: list | None = None, cmds: list | None = None) -> dict:
    """probe_node on `node`: this mn.py and the GPU identity probe, shipped by content."""
    args = [str(int(p)) for p in ports or []] + [a for c in cmds or [] for a in ("--cmd", shlex.quote(c))]
    script = (f'MN_GPU_IDENTITY="$({python_by_content(GPU_IDENTITY_SCRIPT, "2>&1")})" '
              + python_by_content(os.path.join(HERE, "mn.py"), "probe " + " ".join(args)))
    res = ssh_run(state, node, script, 600)
    line = next((ln for ln in reversed(res["out"].splitlines()) if ln.startswith(PROBE_TAG)), "")
    if not line:
        raise RuntimeError(f"probe on {node['ip']} returned nothing (rc={res['rc']}): {res['err'][-500:]}")
    return json.loads(line[len(PROBE_TAG):])


def probe_all(state: dict, nodes: list, ports: list | None = None, cmds: list | None = None) -> list:
    """`cmds`: per node, the program names its shell must resolve."""
    return _parallel([functools.partial(probe, state, n, ports, (cmds or [None] * len(nodes))[i])
                      for i, n in enumerate(nodes)])




# --------------------------------------------------------------------------
# CHECK  (mn.py check)
# --------------------------------------------------------------------------
# The caller names the machines (args.multinode.nodes, rank order); this only
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

    results = {"per_node": []}
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
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60,
                             env={**os.environ, "PATH": real_path()}).stdout
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
NATIVE_BENCH_MODULES = {"sglang": "sglang.bench_serving", "vllm": "vllm.benchmarks.serve"}


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
    dial = f"import socket;socket.create_connection(('127.0.0.1',{port}),1).close()"
    return (f"setsid timeout 60 python3 -c {shlex.quote(py)} </dev/null >/dev/null 2>&1 &\n"
            f"pid=$!; for i in $(seq 50); do python3 -c {shlex.quote(dial)} 2>/dev/null "
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


def cmd_check(argv: list) -> int:
    """GEAK_MN_SPEC is args.multinode plus "tp" (the engine TP), as JSON."""
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

    # A provided router is a machine of its own: no GPU; reachable, ports free, mount shared,
    # its command resolvable (a missing one is only `command not found` in mn_lb.log).
    targets = spec_targets(spec)
    launches = [n.get("launch_command") or "" for n in spec["nodes"]]
    if router:
        targets.append({"node_id": "", "rank": "lb", "ip": router["host"], "role": "router",
                        **({"ssh": router["ssh"]} if router.get("ssh") else {})})
        launches.append(router["launch_command"])
    cmds = [[c] if (c := plain_argv0(cmd)) else [] for cmd in launches]
    for t, cmd, c in zip(targets, launches, cmds):
        if cmd and not c:
            print(f"    {t['ip']}: not checking that its launch_command resolves (a path or a compound line)")
    try:
        probes = probe_all(state, targets, ports, cmds)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return check_verdict(pin, [f"a node could not be probed: {exc}"])
    for t, p in zip(targets, probes):
        p["node_id"] = node_identity(p)
        for c in p.get("missing_cmds") or []:
            who = "router" if t["rank"] == "lb" else f"{t.get('role') or 'node'} {t['ip']}"
            problems.append(f"{who} {_short(p['hostname'])}: {c!r} is not on PATH in the shell mn.py launches it from")
    if router:
        rp = probes.pop()
        state["router"] = {"rank": "lb", "node_id": rp["node_id"], "ip": router["host"],
                           "hostname": rp["hostname"], "role": "router", "port": router["port"],
                           "launch_command": router["launch_command"],
                           **({"ssh": router["ssh"]} if router.get("ssh") else {})}
    state["nodes"], located = locate(ips, probes, need_nic=not provided,
                                     gpu_ids=[n.get("gpu_ids") for n in spec["nodes"]])
    problems += located
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
    # The KB is addressed by product and CU count: every node must be the same GPU.
    gpus = [[(p.get("gpu") or {}).get(k) for k in GPU_KEYS] for p in probes]
    if len({json.dumps(g) for g in gpus}) > 1:
        problems.append(f"the nodes have different GPUs ({', '.join(GPU_KEYS)}): " + "; ".join(
            f"{_short(p['hostname'])} {g}" for p, g in zip(probes, gpus)))
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
# Run-scoped names the launcher sets; a swept EXTRA_ENV must not shadow them.
_OWNED = {
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "PYTHONPATH", "SGLANG_TORCH_PROFILER_DIR", "VLLM_TORCH_PROFILER_DIR",
}


def under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == \
            os.path.realpath(root)
    except ValueError:
        return False

# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def cmd_probe(argv: list) -> int:
    """probe_node on this host; check runs it on every node over ssh. `--cmd <name>`: a program
    this shell must resolve."""
    ap = argparse.ArgumentParser(prog="mn.py probe")
    ap.add_argument("ports", nargs="*", type=int)
    ap.add_argument("--cmd", action="append", default=[])
    args = ap.parse_args(argv)
    print(PROBE_TAG + json.dumps(probe_node(args.ports, args.cmd)), flush=True)
    return 0


COMMANDS = {
    "check": cmd_check,
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
