#!/usr/bin/env python3
"""Multi-node tests for run_e2e: Hyperloom hands the cluster over in MULTI_NODE_STATE."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent

_MN_ENV = ("BENCH_LAUNCHER", "GEAK_MN_STATE", "MAGPIE_LAUNCH_SCRIPT", "GEAK_GPU_IDENTITY_JSON", "MULTI_NODE_STATE")
_MI300X = {"gfx": "gfx942", "target": "unknown", "marketing_name": "AMD Instinct MI300X", "physical_cu_count": 304}
_MODEL = "/apps/DeepSeek-V4-Flash-FP8"
_PREFILL, _DECODE, _ROUTER = ["10.0.1.11", "10.0.1.12"], ["10.0.2.21", "10.0.2.22"], "10.0.0.10"
_ENGINES = _PREFILL + _DECODE


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {name: os.environ.pop(name, None) for name in _MN_ENV}
    # The pinned machines' identity, without ssh-ing to them.
    os.environ["GEAK_GPU_IDENTITY_JSON"] = json.dumps(_MI300X)
    yield
    for name, val in saved.items():
        os.environ.pop(name, None) if val is None else os.environ.__setitem__(name, val)


def _engine_cmd(role: str) -> str:
    boot = " --disaggregation-bootstrap-port 8998" if role == "prefill" else ""
    return ("SGLANG_USE_ROCM700A=0 TORCH_BLAS_PREFER_HIPBLASLT=1 python3 -m sglang.launch_server "
            f"--model-path {_MODEL} --host 0.0.0.0 --port 8000 --tp-size 8 --mem-fraction-static 0.90 "
            f"--disaggregation-mode {role} --disaggregation-transfer-backend mooncake_tcp{boot} --page-size 256")


def _router_cmd(prefill=_PREFILL, decode=_DECODE) -> str:
    return ("python3 -m sglang_router.launch_router --pd-disaggregation "
            + " ".join(f"--prefill http://{ip}:8000 8998" for ip in prefill) + " "
            + " ".join(f"--decode http://{ip}:8000" for ip in decode)
            + f" --host 0.0.0.0 --port 8000 --model-path {_MODEL}")


_DOC = {
    "backend": "sglang", "external": True, "pd_mode": "disaggregated", "nodes": 5, "gpus_per_node": 8,
    "ssh_port": 2222, "ssh_user": "root", "ssh_key_path": "/k/id_ed25519",
    "user_data_path": "/shared/geak", "service_url": f"http://{_ROUTER}:8000",
    "pods": [{"role": "router", "podIP": _ROUTER, "sshPort": 2222, "launch_command": _router_cmd()}]
    + [{"role": "prefill", "podIP": ip, "sshPort": 2222, "launch_command": _engine_cmd("prefill")} for ip in _PREFILL]
    + [{"role": "decode", "podIP": ip, "sshPort": 2222, "launch_command": _engine_cmd("decode")} for ip in _DECODE],
}


@pytest.fixture
def node_state(tmp_path, monkeypatch):
    """Write a node_state.json (edited by the test first) and point MULTI_NODE_STATE at it."""
    def write(doc=None, name="node_state.json"):
        path = tmp_path / name
        path.write_text(json.dumps(_DOC if doc is None else doc), encoding="utf-8")
        monkeypatch.setenv("MULTI_NODE_STATE", str(path))
        return path
    return write


def _doc(**over) -> dict:
    return {**copy.deepcopy(_DOC), **over}


def _handoff(**over) -> dict:
    h = {
        "model_path": _MODEL,
        "exp_root": "/shared/geak/exp",
        "framework": "sglang",
        "tp": 8,
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
    }
    h.update(over)
    return h


def _block(h: dict) -> dict:
    return rx.map_args(h, timeout_s=3600).get("multinode", {})


def _probe_nodes(monkeypatch, replies: dict) -> list:
    """Every node answers gpu_identity.py with replies[ip]; returns the (ip, script) sent."""
    mod = rx._mn_module()
    sent = []

    def run(state, node, script, timeout=300, merged=False):
        sent.append((node["ip"], script))
        out = replies[node["ip"]]
        return {"hostname": node["ip"], "ip": node["ip"], "rc": 0 if out else 1,
                "out": json.dumps(out) if out else "", "err": "" if out else "no rocminfo"}

    monkeypatch.setattr(mod, "ssh_run", run)
    monkeypatch.setattr(rx, "_mn_module", lambda: mod)
    monkeypatch.delenv("GEAK_GPU_IDENTITY_JSON", raising=False)
    return sent


def test_node_state_becomes_the_serving_block(node_state) -> None:
    node_state()
    block = rx._multinode(_handoff())
    assert block["topology"] == "pd_disagg" and block["serve_port"] == 8000
    assert block["ssh"] == {"user": "root", "port": 2222, "key": "/k/id_ed25519"}
    assert block["router"] == {"host": _ROUTER, "launch_command": _router_cmd()}
    assert [(n["host"], n["role"]) for n in block["nodes"]] == [(ip, "prefill") for ip in _PREFILL] + [
        (ip, "decode") for ip in _DECODE]
    assert all("ssh" not in n for n in block["nodes"]), "a pod's sshPort equal to ssh_port is not repeated"
    assert (block["tp_by_role"], block["replicas_by_role"]) == ({"prefill": 8, "decode": 8}, {"prefill": 2, "decode": 2})
    assert (block["backend"], block["model"]) == ("sglang", _MODEL)


def test_node_state_validates_as_two_prefill_and_two_decode_replicas(node_state) -> None:
    node_state()
    spec = rx._mn_spec(_handoff())
    assert [(x["role"], x.get("replica", 0), x["tp"]) for x in spec["layout"]] == [
        ("prefill", 0, 8), ("prefill", 1, 8), ("decode", 0, 8), ("decode", 1, 8)]
    assert spec["router"]["host"] == _ROUTER, "the client port is the router's"
    assert rx.map_args(_handoff(), timeout_s=3600)["gpu_ids"] == "0,1,2,3,4,5,6,7"


def test_the_repo_example_validates(monkeypatch) -> None:
    example = _HERE.parent.parent / "node_state.json"
    if not example.is_file():
        pytest.skip("no node_state.json beside GEAK/")
    monkeypatch.setenv("MULTI_NODE_STATE", str(example))
    assert len(rx._mn_spec(_handoff())["layout"]) == 4


def test_ssh_user_defaults_to_root_and_a_pods_own_port_is_kept(node_state) -> None:
    doc = _doc()
    del doc["ssh_user"]
    doc["pods"][2]["sshPort"] = 2233
    doc["pods"][3]["sshUser"] = "geak"
    node_state(doc)
    block = rx._multinode(_handoff())
    assert block["ssh"]["user"] == "root"
    assert [n.get("ssh") for n in block["nodes"]] == [None, {"port": 2233}, {"user": "geak"}, None]


def test_a_pods_gpu_ids_name_its_engines_cards(node_state) -> None:
    doc = _doc()
    doc["pods"][1]["gpuIds"] = "7,6,5,4,3,2,1,0"
    node_state(doc)
    assert [n.get("gpu_ids") for n in rx._multinode(_handoff())["nodes"]] == ["7,6,5,4,3,2,1,0", None, None, None]


def test_ssh_prefix_reaches_the_ssh_settings(node_state) -> None:
    node_state(_doc(ssh_prefix="sudo -n docker exec -i c"))
    assert rx._multinode(_handoff())["ssh"]["prefix"] == "sudo -n docker exec -i c"


@pytest.mark.parametrize("over, why", [
    ({"backend": "vllm"}, "framework is 'sglang'"),
    ({"service_url": "http://10.0.1.11:8000"}, "is not the router pod"),
    ({"service_url": "http://10.0.0.10"}, "names no host:port"),
    ({"user_data_path": ""}, "names no user_data_path"),
])
def test_a_node_state_that_disagrees_is_refused(node_state, over, why) -> None:
    node_state(_doc(**over))
    with pytest.raises(SystemExit, match=why):
        _block(_handoff())


def test_a_command_that_disagrees_with_the_handoff_is_refused(node_state) -> None:
    doc = _doc()
    doc["pods"][3]["launch_command"] = _engine_cmd("decode").replace("--tp-size 8", "--tp-size 4")
    node_state(doc)
    with pytest.raises(SystemExit, match="TP 4 is not the engine TP 8"):
        _block(_handoff())


def test_an_unreadable_node_state_names_its_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MULTI_NODE_STATE", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="missing.json"):
        _block(_handoff())


def test_multinode_identity_comes_from_the_engine_pods(monkeypatch, node_state) -> None:
    node_state()
    sent = _probe_nodes(monkeypatch, {ip: _MI300X for ip in _ENGINES})
    assert rx._expected_gpu_identity(_handoff()) == _MI300X
    assert [ip for ip, _ in sent] == _ENGINES, "the router holds no GPU and is not asked"
    source = rx.GPU_IDENTITY_SCRIPT.read_text(encoding="utf-8")
    assert all(s.startswith("python3 - <<'GEAK_MN_PY'\n") and source in s for _, s in sent), \
        "the nodes have no GEAK: the probe goes by content"


def test_multinode_identity_refuses_mixed_or_silent_machines(monkeypatch, node_state) -> None:
    node_state()
    _probe_nodes(monkeypatch, {**{ip: _MI300X for ip in _ENGINES}, _DECODE[1]: dict(_MI300X, gfx="gfx950")})
    with pytest.raises(RuntimeError, match="different GPUs"):
        rx._expected_gpu_identity(_handoff())
    _probe_nodes(monkeypatch, {**{ip: _MI300X for ip in _ENGINES}, _DECODE[1]: None})
    with pytest.raises(RuntimeError, match=f"{_DECODE[1]}: no rocminfo"):
        rx._expected_gpu_identity(_handoff())


def test_single_node_handoff_is_untouched() -> None:
    args = rx.map_args(_handoff(), timeout_s=3600)
    assert "multinode" not in args and "nodes" not in args and "gpus_per_node" not in args
    assert args["gpu_ids"] == "0,1,2,3,4,5,6,7"


def test_tp_alone_never_implies_multiple_nodes() -> None:
    assert "multinode" not in rx.map_args(_handoff(tp=16), timeout_s=3600)
    assert rx.map_args(_handoff(tp=16), timeout_s=3600)["gpu_ids"].count(",") == 15


def test_the_block_is_forwarded_to_the_workflow_without_side_effects(monkeypatch, node_state) -> None:
    node_state()
    monkeypatch.setattr(rx.subprocess, "run", lambda *a, **k: pytest.fail("run_e2e ran a command"))
    before = dict(os.environ)
    assert _block(_handoff()) == rx._spec_from_node_state(_DOC, _handoff()), "mn.py check parses it, from GEAK_MN_SPEC"
    assert dict(os.environ) == before


def test_node_state_selects_the_multinode_launcher_and_names_the_pin(node_state) -> None:
    node_state()
    assert rx.apply_bench_launcher(_handoff()) == "multinode"
    pin = os.environ["GEAK_MN_STATE"]
    assert pin.startswith("/shared/geak/pins/mn_state_") and pin.endswith(".json")
    other = _doc()
    other["pods"][-1]["podIP"] = "10.0.9.9"
    assert rx._spec_from_node_state(other, _handoff())["state"] != pin, "another cluster pins elsewhere"
    assert rx._spec_from_node_state(_doc(), _handoff())["state"] == pin, "the same machines share one pin"


def test_a_requested_launcher_or_magpie_script_does_not_win(monkeypatch, node_state) -> None:
    node_state()
    monkeypatch.setenv("MAGPIE_LAUNCH_SCRIPT", "/opt/magpie/run.sh")
    assert rx.apply_bench_launcher(_handoff(bench_launcher="native")) == "multinode"
    os.environ["BENCH_LAUNCHER"] = "magpie"
    assert rx.apply_bench_launcher(_handoff()) == "multinode"


def test_a_single_host_handoff_names_no_pin() -> None:
    rx.apply_bench_launcher(_handoff())
    assert "GEAK_MN_STATE" not in os.environ


def test_single_node_still_prefers_magpie_when_a_script_exists(monkeypatch) -> None:
    monkeypatch.setenv("MAGPIE_LAUNCH_SCRIPT", "/opt/magpie/run.sh")
    assert rx.apply_bench_launcher(_handoff()) == "magpie"


def test_a_multinode_driver_masks_its_own_gpus(monkeypatch, node_state) -> None:
    masks = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("PATH", os.environ["PATH"])
    for mask in masks:
        monkeypatch.setenv(mask, "0,1")
    rx.apply_bench_launcher(_handoff())
    assert [os.environ[m] for m in masks] == ["0,1"] * 3, "one host keeps its cards"
    node_state()
    assert rx.apply_bench_launcher(_handoff()) == "multinode"
    assert [os.environ[m] for m in masks] == [""] * 3


@pytest.mark.skipif(os.name != "posix", reason="the stubs are sh scripts")
def test_a_multinode_driver_shadows_the_gpu_inventory_tools(monkeypatch, node_state, request) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep)
                                               if "geak_mn_driver_bin_" not in p))
    rx.apply_bench_launcher(_handoff())
    assert "geak_mn_driver_bin_" not in os.environ["PATH"], "one host keeps them"
    node_state()
    rx.apply_bench_launcher(_handoff())
    rx.apply_bench_launcher(_handoff())
    first, *rest = os.environ["PATH"].split(os.pathsep)
    assert first not in rest, "re-applying does not stack the stub dir"
    r = subprocess.run(["rocm-smi", "--showuse"], capture_output=True, text=True, env=dict(os.environ))
    state = os.environ["GEAK_MN_STATE"]
    assert r.returncode == 1 and r.stdout == ""
    assert f"GEAK_MN_STATE={state} python3 " in r.stderr
    assert "mn.py exec --target <master|rank<N>|prefill|decode> -- rocm-smi --showuse" in r.stderr

    # Not tmp_path: under TMPDIR=/dev/shm (noexec) PATH lookup would skip these and find the real tools.
    real = rx._exec_dir(f"geak_mn_test_real_{os.getpid()}")
    assert real is not None
    request.addfinalizer(lambda: shutil.rmtree(real, True))
    for tool, out in (("rocm-smi", "real rocm-smi"), ("rocminfo", "  Name:                    gfx942")):
        (real / tool).write_text(f"#!/bin/sh\necho '{out}'\n", encoding="utf-8")
        (real / tool).chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([first, str(real), *rest]))
    mn = rx._mn_module()
    monkeypatch.setattr(mn, "is_self", lambda node: True)
    r = subprocess.run(mn.ssh_argv({}, {"node_id": "driver/b0"}), input="rocm-smi --showuse",
                       capture_output=True, text=True)
    assert (r.returncode, r.stdout.strip()) == (0, "real rocm-smi"), "a rank on the driver's machine"
    assert mn.visible_gpus() == 1, "the driver's GPU gate reads the real rocminfo"


def test_the_stub_dir_skips_a_noexec_tmpdir(monkeypatch, tmp_path) -> None:
    noexec, ok = tmp_path / "shm", tmp_path / "tmp"
    monkeypatch.setattr(rx.tempfile, "gettempdir", lambda: str(noexec))
    flag = getattr(os, "ST_NOEXEC", 8)
    monkeypatch.setattr(os, "statvfs", lambda p: type("S", (), {"f_flag": flag if str(p).startswith(str(noexec)) else 0})(),
                        raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(ok))
    made = rx._exec_dir("geak_mn_driver_bin_x")
    assert made is not None and not str(made).startswith(str(noexec))
    monkeypatch.setattr(os, "statvfs", lambda p: type("S", (), {"f_flag": flag})(), raising=False)
    assert rx._exec_dir("geak_mn_driver_bin_y") is None, "nowhere may execute: the caller warns"


def test_no_hook_moves_agent_commands() -> None:
    src = (_HERE / "run_e2e.py").read_text(encoding="utf-8")
    assert "PreToolUse" not in src and "GEAK_REMOTE_GPU" not in src


def test_a_multinode_run_executes_here_without_ssh_to_the_master(monkeypatch, tmp_path, node_state) -> None:
    node_state()
    handoff, result = tmp_path / "handoff.json", tmp_path / "result.json"
    handoff.write_text(json.dumps(_handoff()), encoding="utf-8")
    monkeypatch.setattr(rx.subprocess, "Popen", lambda *a, **k: pytest.fail("run_e2e spawned a process"))
    monkeypatch.setattr(rx, "map_args", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached map_args")))
    with pytest.raises(RuntimeError, match="reached map_args"):
        rx.main([str(handoff), str(result)])
    assert not hasattr(rx, "_run_on_master")


# ---- the first-round profile Hyperloom already ran (profile_trace) ----

def _profile_tree(root: Path, dirs=None) -> dict:
    """<root>/<podIP>_<role>/ with a rank0 and a rank1 trace plus capture_traces/; returns role -> rank0s."""
    out = {}
    for ip, role in dirs or [(ip, "prefill") for ip in _PREFILL] + [(ip, "decode") for ip in _DECODE]:
        d = root / f"{ip}_{role}"
        (d / "capture_traces").mkdir(parents=True)
        (d / "capture_traces" / "rank0.trace.json.gz").write_bytes(b"x")
        for rank in (0, 1):
            (d / f"dp0_tp{rank}_rank{rank}.pt.trace.json.gz").write_bytes(b"x")
        out.setdefault(role, []).append(str(d / "dp0_tp0_rank0.pt.trace.json.gz"))
    return out


_GLOBBED = {"analysis_md": "/s/kernel-agent/a/tracelens/analysis.md", "trace_file": "/s/runs/roofline/x/torch_trace"}


def test_profile_trace_becomes_one_rank0_trace_per_engine_pod_steady_first(tmp_path) -> None:
    root = tmp_path / "torch_trace"
    rank0 = _profile_tree(root)
    steady = rank0["decode"][0]
    tl = rx._node_state_tracelens(_doc(profile_trace=str(root), steady_state_trace=steady), dict(_GLOBBED))
    assert tl["trace_file"] == str(root), "the caller's profile, not the glob's"
    assert tl["analysis_md"] == _GLOBBED["analysis_md"]
    assert tl["trace_files"][0] == steady
    assert sorted(tl["trace_files"]) == sorted(rank0["prefill"] + rank0["decode"])
    assert not any("capture_traces" in t for t in tl["trace_files"])


def test_a_directory_that_names_no_pod_is_ignored(tmp_path) -> None:
    root = tmp_path / "torch_trace"
    rank0 = _profile_tree(root, [(_PREFILL[0], "prefill"), (_DECODE[0], "decode"), ("10.9.9.9", "decode"),
                                 (_PREFILL[1], "decode")])
    tl = rx._node_state_tracelens(_doc(profile_trace=str(root)), dict(_GLOBBED))
    assert sorted(tl["trace_files"]) == sorted([rank0["prefill"][0], rank0["decode"][0]])


def test_a_role_without_traces_drops_the_prior(tmp_path) -> None:
    root = tmp_path / "torch_trace"
    _profile_tree(root, [(ip, "prefill") for ip in _PREFILL])
    tl = rx._node_state_tracelens(_doc(profile_trace=str(root)), dict(_GLOBBED))
    assert "trace_files" not in tl and "analysis_md" not in tl, "GEAK profiles on its own"


def test_a_missing_profile_trace_never_falls_back_to_the_glob(tmp_path) -> None:
    tl = rx._node_state_tracelens(_doc(profile_trace=str(tmp_path / "gone")), dict(_GLOBBED))
    assert tl == {"analysis_md": _GLOBBED["analysis_md"]}


def test_map_args_forwards_the_trace_list(tmp_path, node_state) -> None:
    exp = tmp_path / "session"
    (exp / "kernel-agent" / "r" / "tracelens").mkdir(parents=True)
    (exp / "kernel-agent" / "r" / "tracelens" / "analysis.md").write_text("# a", encoding="utf-8")
    root = exp / "runs" / "profile" / "torch_trace"
    _profile_tree(root)
    node_state(_doc(profile_trace=str(root)))
    tl = rx.map_args(_handoff(exp_root=str(exp / "geak")), timeout_s=3600)["tracelens"]
    assert tl["trace_file"] == str(root) and len(tl["trace_files"]) == 4
    assert tl["analysis_md"].endswith("analysis.md")


def test_a_pd_profile_keeps_only_dirs_with_a_rank0_and_a_steady_trace_from_elsewhere(tmp_path, capsys) -> None:
    root = tmp_path / "torch_trace"
    rank0 = _profile_tree(root, [(_PREFILL[0], "prefill"), (_DECODE[0], "decode")])
    (root / "README").write_text("not a pod", encoding="utf-8")
    (root / f"{_PREFILL[1]}_prefill").mkdir()
    steady = tmp_path / "steady.pt.trace.json.gz"
    steady.write_bytes(b"x")
    tl = rx._node_state_tracelens(_doc(profile_trace=str(root), steady_state_trace=str(steady)), dict(_GLOBBED))
    assert tl["trace_files"][0] == str(steady)
    assert sorted(tl["trace_files"][1:]) == sorted(rank0["prefill"] + rank0["decode"])
    assert "has no rank0 trace at its top level" in capsys.readouterr().err


def test_a_tp_span_profile_is_the_trace_dir_alone(tmp_path) -> None:
    root = tmp_path / "torch_trace"
    root.mkdir()
    tl = rx._node_state_tracelens(_doc(profile_trace=str(root), pd_mode="none"), dict(_GLOBBED))
    assert tl == {"analysis_md": _GLOBBED["analysis_md"], "trace_file": str(root)}


# ---- the current node_state.json: a platform-launched router and k8s Service names ----

_SVC = "claw-1.ns.svc.cluster.local"
_NEW_IPS = {"prefill": "10.0.1.11", "decode": "10.0.2.21", "router": "192.168.1.5"}


def _new_engine(role: str) -> str:
    port, boot = (8010, 8998) if role == "prefill" else (8020, 9098)
    return (f"python3 -m sglang.launch_server --model-path {_MODEL} --host 0.0.0.0 --port {port} --tp-size 8 "
            f"--disaggregation-mode {role} --disaggregation-transfer-backend mori --disaggregation-bootstrap-port {boot}")


_NEW = {
    "backend": "sglangrouter", "external": True, "pd_mode": "disaggregated", "nodes": 2, "gpus_per_node": 8,
    "ssh_port": 29960, "ssh_key_path": "/tmp/key", "ssh_known_hosts": "/tmp/claw/known_hosts",
    "user_data_path": "/primus-shared/hyperloom", "service_url": f"http://{_SVC}:8000",
    "pods": [
        {"role": "router", "podIP": _NEW_IPS["router"], "sshPort": 29960, "managed_by": "platform",
         "launch_command": "python3 -m sglang_router.launch_router --pd-disaggregation "
                           "--prefill http://claw-1-prefill.ns.svc.cluster.local:8010 8998 "
                           "--decode http://claw-1-decode.ns.svc.cluster.local:8020 --host 0.0.0.0 --port 8000"},
        {"role": "prefill", "podIP": _NEW_IPS["prefill"], "sshPort": 29960, "launch_command": _new_engine("prefill")},
        {"role": "decode", "podIP": _NEW_IPS["decode"], "sshPort": 29960, "launch_command": _new_engine("decode")},
    ],
}


@pytest.fixture
def dns(monkeypatch):
    """The Service names resolve to ClusterIPs (no pod address); any other name does not resolve."""
    import socket
    names = {"claw-1-prefill.ns.svc.cluster.local": "10.96.0.11", "claw-1-decode.ns.svc.cluster.local": "10.96.0.12"}

    def lookup(host, *a, **k):
        if host in names:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (names[host], 0))]
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)


def test_a_platform_router_is_one_geak_restarts(node_state, dns) -> None:
    node_state(copy.deepcopy(_NEW))
    block = _block(_handoff())
    assert block["router"] == {"host": _NEW_IPS["router"], "launch_command": _NEW["pods"][0]["launch_command"]}
    assert (block["backend"], block["shared_fs"], block["serve_port"]) == ("sglang", "/primus-shared/hyperloom", 8000)
    assert block["ssh"]["known_hosts"] == "/tmp/claw/known_hosts"
    assert rx._mn_spec(_handoff())["router"]["host"] == _NEW_IPS["router"], "Service-named workers pair by role and port"


def test_a_worker_name_that_does_not_resolve_is_refused(node_state, dns) -> None:
    doc = copy.deepcopy(_NEW)
    doc["pods"][0]["launch_command"] = doc["pods"][0]["launch_command"].replace("claw-1-decode", "claw-9-decode")
    node_state(doc)
    with pytest.raises(SystemExit, match="claw-9-decode.* does not resolve"):
        rx._mn_spec(_handoff())


def test_a_trace_directory_per_role(tmp_path) -> None:
    root = tmp_path / "torch_trace"
    for role in ("prefill", "decode"):
        (root / role).mkdir(parents=True)
        (root / role / "123-TP-0.trace.json.gz").write_bytes(b"x")
    steady = str(root / "decode" / "123-TP-0.trace.json.gz")
    tl = rx._node_state_tracelens({**_NEW, "profile_trace": str(root), "steady_state_trace": steady}, dict(_GLOBBED))
    assert tl["trace_files"] == [steady, str(root / "prefill" / "123-TP-0.trace.json.gz")]
    assert tl["analysis_md"] == _GLOBBED["analysis_md"]


def test_a_node_state_that_is_not_an_object_is_refused(tmp_path, monkeypatch) -> None:
    path = tmp_path / "node_state.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE", str(path))
    rx._read_node_state.cache_clear()
    with pytest.raises(SystemExit, match="not a JSON object"):
        _block(_handoff())


def _pods(edit):
    doc = _doc()
    edit(doc["pods"])
    return doc


@pytest.mark.parametrize("doc, why", [
    (_doc(pods={"router": _ROUTER}), "pods must be a list of objects"),
    (_pods(lambda pods: pods.insert(0, dict(pods[0]))), "at most one router pod"),
    (_pods(lambda pods: pods[1].pop("launch_command")), "names no launch_command"),
])
def test_a_malformed_pod_list_is_refused(node_state, doc, why) -> None:
    if isinstance(doc["pods"], list) and len(doc["pods"]) == 6:
        doc["nodes"] = 6
    node_state(doc)
    with pytest.raises(SystemExit, match=why):
        _block(_handoff())


def test_the_carried_state_block_serves_when_args_have_none() -> None:
    carried = {"nodes": [{"host": "10.0.0.1"}]}
    assert rx._multinode_block({"state": {"multinode": carried}}) == carried
    assert rx._multinode_block({"multinode": {"nodes": []}, "state": {}}) == {}


def _check(monkeypatch, tmp_path, run):
    monkeypatch.setattr(rx.subprocess, "run", run)
    return rx._multinode_check({"multinode": {"nodes": [{"host": "10.0.0.1"}]}, "tp": 8,
                                "eval_dir": str(tmp_path)})


def test_the_check_hands_back_mn_pys_verdict(monkeypatch, tmp_path) -> None:
    tag = rx._mn_module().CHECK_TAG
    seen = {}

    def run(argv, env=None, **_kw):
        seen["spec"] = json.loads(env["GEAK_MN_SPEC"])
        return rx.subprocess.CompletedProcess(argv, 0, f"probing\n{tag}" + json.dumps({"verdict": "ok", "spec": {}}), "")

    assert _check(monkeypatch, tmp_path, run) == {"verdict": "ok", "spec": {}}
    assert (seen["spec"]["tp"], seen["spec"]["work_dir"]) == (8, str(tmp_path))


def test_the_check_takes_over_the_callers_service_unless_turned_off(monkeypatch, tmp_path) -> None:
    tag = rx._mn_module().CHECK_TAG
    seen = []

    def run(argv, env=None, **_kw):
        seen.append(env.get("GEAK_MN_TAKEOVER"))
        return rx.subprocess.CompletedProcess(argv, 0, tag + json.dumps({"verdict": "ok", "spec": {}}), "")

    monkeypatch.delenv("GEAK_MN_TAKEOVER", raising=False)
    _check(monkeypatch, tmp_path, run)
    monkeypatch.setenv("GEAK_MN_TAKEOVER", "0")
    _check(monkeypatch, tmp_path, run)
    assert seen == ["1", "0"]


def test_a_refused_or_silent_or_hung_check_is_refused(monkeypatch, tmp_path) -> None:
    tag = rx._mn_module().CHECK_TAG
    refused = json.dumps({"verdict": "refused", "problems": ["no free VRAM"], "at": "t"})
    v = _check(monkeypatch, tmp_path, lambda argv, **k: rx.subprocess.CompletedProcess(argv, 2, tag + refused, ""))
    assert v == {"verdict": "refused", "problems": ["no free VRAM"]}
    v = _check(monkeypatch, tmp_path, lambda argv, **k: rx.subprocess.CompletedProcess(argv, 1, f"{tag}{{", "Traceback"))
    assert v["verdict"] == "refused" and "exited 1 without a verdict: Traceback" in v["problems"][0]

    def hang(argv, **k):
        raise rx.subprocess.TimeoutExpired(argv, 1800)

    assert "timed out" in _check(monkeypatch, tmp_path, hang)["problems"][0]


def test_the_prompt_carries_no_launch_command() -> None:
    cmd = "vllm serve m --kv-transfer-config '{\"kv_role\":\"kv_producer\"}'"
    block = {"topology": "pd_disagg", "state": "/s/pins/p.json", "shared_fs": "/s", "tp_by_role": {"prefill": 8},
             "nodes": [{"host": "10.0.0.1", "role": "prefill", "launch_command": cmd, "gpu_ids": "0"}],
             "router": {"host": "10.0.0.9", "launch_command": "vllm-router --port 1"}}
    out = rx._mn_relayed(block)
    assert "launch_command" not in json.dumps(out) and out["commands_in_pin"] is True
    assert out["nodes"] == [{"host": "10.0.0.1", "role": "prefill", "gpu_ids": "0"}]
    assert (out["router"], out["state"], out["tp_by_role"]) == ({"host": "10.0.0.9"}, "/s/pins/p.json", {"prefill": 8})
    assert block["nodes"][0]["launch_command"] == cmd


def test_a_carried_block_without_commands_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(rx.subprocess, "run", lambda *a, **k: pytest.fail("mn.py check ran without commands"))
    v = rx._multinode_check({"state": {"multinode": {"nodes": [{"host": "10.0.0.1"}], "commands_in_pin": True}},
                             "tp": 8, "eval_dir": str(tmp_path)})
    assert v["verdict"] == "refused" and "pass MULTI_NODE_STATE" in v["problems"][0]


@pytest.mark.skipif(os.name != "posix", reason="main() publishes process groups")
def test_a_passed_check_reaches_the_workflow_without_launch_commands(monkeypatch, tmp_path, node_state) -> None:
    node_state()
    exp = tmp_path / "geak"
    handoff, result = tmp_path / "handoff.json", tmp_path / "result.json"
    handoff.write_text(json.dumps(_handoff(exp_root=str(exp))), encoding="utf-8")
    spec = {"state": "/s/pins/p.json", "shared_fs": "/s", "nodes": [{"host": "10.0.1.11", "launch_command": "x 'y'"}]}
    monkeypatch.setattr(rx, "_multinode_check", lambda ps_args: {"verdict": "ok", "problems": [], "spec": spec})
    seen = {}

    def invoke(prompt, timeout_s, eval_dir=None, ps_args=None):
        seen.update(prompt=prompt, ps_args=ps_args)
        raise SystemExit(0)

    monkeypatch.setattr(rx, "invoke_workflow", invoke)
    with pytest.raises(SystemExit):
        rx.main([str(handoff), str(result)])
    assert "launch_command" not in seen["prompt"]
    assert seen["ps_args"]["multinode_check"]["spec"]["commands_in_pin"] is True
    assert seen["ps_args"]["multinode"]["commands_in_pin"] is True


@pytest.mark.skipif(os.name != "posix", reason="main() publishes process groups")
def test_a_refused_check_ends_the_run_with_its_reasons(monkeypatch, tmp_path, node_state) -> None:
    node_state()
    exp = tmp_path / "geak"
    handoff, result = tmp_path / "handoff.json", tmp_path / "result.json"
    handoff.write_text(json.dumps(_handoff(exp_root=str(exp))), encoding="utf-8")
    monkeypatch.setattr(rx, "_multinode_check",
                        lambda ps_args: {"verdict": "refused", "problems": ["rank1 has no free VRAM"]})
    monkeypatch.setattr(rx, "invoke_workflow", lambda *a, **k: pytest.fail("the workflow was started"))
    assert rx.main([str(handoff), str(result)]) == 1
    out = json.loads(result.read_text(encoding="utf-8"))
    assert (out["status"], out["error_class"]) == ("error", "multinode_check_refused")
    assert out["error"] == "mn.py check refused the machines: rank1 has no free VRAM"
    assert out["multinode_check"]["problems"] == ["rank1 has no free VRAM"]
