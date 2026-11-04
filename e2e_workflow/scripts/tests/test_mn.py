"""The multi-node suite: node check, per-rank render, profile window, gate."""
import contextlib
import glob
import gzip
import io
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
MN = os.path.join(SCRIPTS_DIR, "multi_node")
sys.path.insert(0, MN)

COLLECTIVE = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage)"

import mn  # noqa: E402
import mn_traces  # noqa: E402


def probe(hostname, *addrs, ngpu=8, **kw):
    return {"hostname": hostname, "ngpu": ngpu, "serve_procs": 0, "gpu_pids": 0,
            "foreign_gpu_pids": 0, "busy_ports": [], "addrs": [list(a) for a in addrs],
            "free_vram_mib": 190000, "total_vram_mib": 192000,
            "node_id": hostname, "boot_id": "b", **kw}


def node_reply(node, script):
    """A healthy node: the reach test's listener comes up and is dialed; anything else prints a digest."""
    if "sport = :" in script:
        return {"rc": 0, "out": "LISTENING 99", "err": ""}
    if "/dev/tcp/" in script:
        return {"rc": 0, "out": "REACHED", "err": ""}
    if "echo REACHED " in script:
        return {"rc": 0, "out": "\n".join(f"REACHED {r}" for r in re.findall(r"echo REACHED (\d+)", script)), "err": ""}
    return {"rc": 0, "out": "f" * 64, "err": ""}


def fake_ssh(handler):
    """Every ssh answers handler(task, node, args) instead of running: task is mn.probe_node
    for a probe (args = (ports,)), else "run" with args = (script,)."""
    def run(state, node, script, timeout=300, merged=False):
        return {"hostname": node.get("hostname") or node["ip"], "ip": node["ip"], "out": "", "err": "",
                **handler("run", node, (script,))}

    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(mn, "ssh_run", run))
    stack.enter_context(mock.patch.object(
        mn, "probe", lambda state, node, ports=None: handler(mn.probe_node, node, (list(ports or []),))))
    return stack


def on_machine(node_id, *addrs):
    """This process runs on the machine pinned as node_id, carrying addrs."""
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(mn, "local_identity", lambda: node_id))
    stack.enter_context(mock.patch.object(mn, "local_addrs", lambda: tuple(tuple(a) for a in addrs)))
    return stack


C489 = probe("node-c489", ("lo", "127.0.0.1", 8), ("ens50f0", "10.10.0.32", 24))
FC54 = probe("node-fc54", ("ens51np0", "10.10.0.166", 24))
F293 = probe("node-f293", ("ens50f0", "10.10.0.183", 24))


def sg_cmd(role, env="SGLANG_X=1 "):
    """A Hyperloom-style engine launch_command (node_state.json pods[].launch_command)."""
    boot = " --disaggregation-bootstrap-port 8998" if role == "prefill" else ""
    return (f"{env}python3 -m sglang.launch_server --model-path /m --host 0.0.0.0 --port 8000 --tp-size 8 "
            f"--disaggregation-mode {role} --disaggregation-transfer-backend mooncake_tcp{boot} --page-size 256")


def router_cmd(prefill, decode, port=8000):
    return ("python3 -m sglang_router.launch_router --pd-disaggregation "
            + " ".join(f"--prefill http://{h}:8000 8998" for h in prefill) + " "
            + " ".join(f"--decode http://{h}:8000" for h in decode)
            + f" --host 0.0.0.0 --port {port} --model-path /m")


def provided_block(prefill=("p",), decode=("d",), router="r", **over):
    """What run_e2e.py makes of a MULTI_NODE_STATE document."""
    return {"topology": "pd_disagg", "shared_fs": "/w", "backend": "sglang", "model": "/m",
            "serve_port": 8000, "tp_by_role": {"prefill": 8, "decode": 8},
            "replicas_by_role": {"prefill": len(prefill), "decode": len(decode)},
            "nodes": [{"host": h, "role": "prefill", "launch_command": sg_cmd("prefill")} for h in prefill]
            + [{"host": h, "role": "decode", "launch_command": sg_cmd("decode")} for h in decode],
            "router": {"host": router, "launch_command": router_cmd(prefill, decode)}, **over}


class LocateTest(unittest.TestCase):
    def test_rank_i_is_the_machine_that_answered_at_ip_i_and_its_nic_carries_it(self):
        nodes, problems = mn.locate(["10.10.0.166", "10.10.0.32"], [FC54, C489])
        self.assertEqual(problems, [])
        self.assertEqual([(n["rank"], n["hostname"], n["cluster_if"]) for n in nodes],
                         [(0, "node-fc54", "ens51np0"), (1, "node-c489", "ens50f0")])

    def test_a_machine_without_that_address_is_named(self):
        _nodes, problems = mn.locate(["10.10.0.166", "10.9.9.9"], [FC54, C489])
        self.assertEqual(problems, ["rank1 10.9.9.9: node-c489 has no interface with this address"])

    def test_a_provided_script_owns_the_nic(self):
        nodes, problems = mn.locate(["node-c489.example"], [C489], need_nic=False)
        self.assertEqual(problems, [])
        self.assertEqual(nodes[0]["cluster_if"], "")

    def test_two_ips_on_one_machine_are_refused(self):
        a = probe("a", ("ens1", "10.0.0.1", 24), ("ens2", "10.0.0.2", 24))
        _nodes, problems = mn.locate(["10.0.0.1", "10.0.0.2"], [a, a])
        self.assertIn("two of the IPs are on the same machine", problems)

    def test_one_machine_holds_two_ranks_on_disjoint_cards(self):
        nodes, problems = mn.locate(["10.10.0.166"] * 2, [FC54, FC54], gpu_ids=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(problems, [])
        self.assertEqual([n["node_id"] for n in nodes], ["node-fc54", "node-fc54"])

    def test_ranks_on_one_machine_need_cards_of_their_own(self):
        _nodes, problems = mn.locate(["10.10.0.166"] * 2, [FC54, FC54], gpu_ids=[[0, 1, 2, 3], [3, 4, 5, 6]])
        self.assertEqual(problems, ["the ranks on node-fc54 share a card: gpu_ids [[0, 1, 2, 3], [3, 4, 5, 6]]"])
        _nodes, problems = mn.locate(["10.10.0.166"] * 2, [FC54, FC54], gpu_ids=[[0, 1, 2, 3], None])
        self.assertEqual(problems, ["two of the IPs are on the same machine"])

    def test_an_ip_on_two_interfaces_is_ambiguous(self):
        a = probe("a", ("docker0", "172.17.0.1", 16), ("br0", "172.17.0.1", 16))
        _nodes, problems = mn.locate(["172.17.0.1"], [a])
        self.assertIn("carried by 2 interfaces", problems[0])


class NormalizeSpecTest(unittest.TestCase):
    BLOCK = {"nodes": ["10.0.0.1", "10.0.0.2"], "shared_fs": "/w/"}

    def test_defaults_fill_in_and_tp_is_split(self):
        spec = mn.normalize_spec(self.BLOCK, 16)
        self.assertEqual((spec["topology"], spec["shared_fs"], spec["state"]), ("tp_span", "/w", "/w/mn_state.json"))
        self.assertEqual((spec["serve_port"], spec["dist_init_port"], spec["tp"]), (30890, 30531, 16))
        self.assertEqual(spec["nodes"], [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}])
        self.assertEqual(spec["layout"], [{"tp": 16, "gpus": 8}, {"tp": 16, "gpus": 8}])
        self.assertEqual(spec["kb_topology"], "tpspan_n2")
        self.assertEqual(mn.normalize_spec({"nodes": ["10.0.0.1"], "shared_fs": "/w"}, 8)["kb_topology"], "",
                         "one node keeps the one-host KB address")

    def test_roles_order_pd_nodes_and_count_them(self):
        spec = mn.normalize_spec({"topology": "pd_disagg", "shared_fs": "/w", "tp_by_role": {"decode": 4},
                                  "nodes": [{"host": "d", "role": "decode"}, {"host": "p", "role": "prefill"}]}, 8)
        self.assertEqual([n["host"] for n in spec["nodes"]], ["p", "d"])
        self.assertEqual((spec["tp_by_role"], spec["nodes_by_role"]),
                         ({"prefill": 8, "decode": 4}, {"prefill": 1, "decode": 1}))
        self.assertEqual(spec["kb_topology"], "pd_p8x1_d4x1")

    def test_an_explicit_rank_orders_the_nodes(self):
        spec = mn.normalize_spec({"shared_fs": "/w", "nodes": [{"host": "b", "rank": 1}, {"host": "a", "rank": 0}]}, 16)
        self.assertEqual([n["host"] for n in spec["nodes"]], ["a", "b"])

    def test_the_spec_is_its_own_fixed_point(self):
        spec = mn.normalize_spec({"topology": "pd_disagg", "shared_fs": "/w", "ssh": {"port": 2},
                                  "pd": {"backend": "mooncake", "lb": "minilb"},
                                  "nodes": [{"host": "p", "role": "prefill", "ssh": {"user": "u"}},
                                            {"host": "d", "role": "decode"}]}, 8)
        self.assertEqual(mn.normalize_spec(spec, spec["tp"]), spec)

    def test_bad_blocks_are_refused_by_name(self):
        for block, why in (
                ({"nodes": ["a"]}, "shared_fs"),
                ({"nodes": [], "shared_fs": "/w"}, "at least one"),
                ({"nodes": ["a"], "shared_fs": "/w", "topology": "ring"}, "unknown topology"),
                ({"nodes": [{"host": "a", "launch_command": sg_cmd("prefill")}, "b"], "shared_fs": "/w"},
                 "every node"),
                ({"nodes": [{"host": "a", "launch_script": "/s"}], "shared_fs": "/w"}, "launch_script is no longer read"),
                ({"nodes": ["a"], "shared_fs": "/w", "pd": {"lb": "nginx"}}, "pd.lb"),
                ({"nodes": ["a"], "shared_fs": "/w", "pd": {"roles": "x"}}, "pd takes"),
                ({"nodes": [{"host": "a", "ssh": {"pasword": "x"}}], "shared_fs": "/w"}, "unknown ssh key"),
                ({"nodes": ["a", "b"], "shared_fs": "/w", "router": {"host": "b", "launch_command": "x"}},
                 "a router launch_command goes with provided pd_disagg")):
            with self.assertRaisesRegex(ValueError, why):
                mn.normalize_spec(block, 16)

    def test_gpu_ids_name_the_cards_of_the_engines_share(self):
        block = {"topology": "pd_disagg", "shared_fs": "/w",
                 "nodes": [{"host": "h", "role": "prefill", "gpu_ids": "0,1,2,3"},
                           {"host": "h", "role": "decode", "gpu_ids": [4, 5, 6, 7]}]}
        spec = mn.normalize_spec(block, 4)
        self.assertEqual([n["gpu_ids"] for n in spec["nodes"]], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(mn.normalize_spec(spec, spec["tp"]), spec)
        for ids, why in (("0,1,2", "are not the 4 distinct cards"), ("0,1,1,2", "are not the 4 distinct cards"),
                         ("0,a,2,3", "must be card ids")):
            block["nodes"][0]["gpu_ids"] = ids
            with self.assertRaisesRegex(ValueError, why):
                mn.normalize_spec(block, 4)

    def test_the_master_is_the_routers_host_else_rank0(self):
        plain = mn.normalize_spec(self.BLOCK, 16)
        self.assertEqual(mn.spec_master(plain)["ip"], "10.0.0.1")
        routed = mn.normalize_spec(provided_block(), 8)
        self.assertEqual((mn.spec_master(routed)["ip"], mn.spec_master(routed)["rank"]), ("r", "lb"))


class ProvidedSpecTest(unittest.TestCase):
    """Hyperloom's per-pod launch_command: ports and wiring come from the commands."""

    def test_xpyd_replicas_are_one_engine_each_with_the_commands_ports(self):
        spec = mn.normalize_spec(provided_block(prefill=("p0", "p1"), decode=("d0", "d1")), 8)
        self.assertEqual([n["host"] for n in spec["nodes"]], ["p0", "p1", "d0", "d1"])
        self.assertEqual([(x["role"], x.get("replica", 0)) for x in spec["layout"]],
                         [("prefill", 0), ("prefill", 1), ("decode", 0), ("decode", 1)])
        self.assertEqual((spec["nodes_by_role"], spec["replicas_by_role"]),
                         ({"prefill": 1, "decode": 1}, {"prefill": 2, "decode": 2}))
        self.assertEqual(spec["kb_topology"], "pd_p8x2_d8x2")
        self.assertEqual([(n["port"], n.get("bootstrap_port")) for n in spec["nodes"]],
                         [(8000, 8998), (8000, 8998), (8000, None), (8000, None)])
        self.assertEqual((spec["router"]["host"], spec["router"]["port"]), ("r", 8000))
        self.assertEqual(mn.normalize_spec(spec, spec["tp"]), spec)

    def test_engine_urls_are_one_per_replica(self):
        state = {"topology": "pd_disagg", "serve_port": 8000, "nodes": [
            {"rank": 0, "ip": "p0", "role": "prefill", "port": 8000},
            {"rank": 1, "ip": "p1", "role": "prefill", "port": 8000, "replica": 1},
            {"rank": 2, "ip": "d0", "role": "decode", "port": 8000}]}
        self.assertEqual([(e["role"], e["replica"], e["url"]) for e in mn.engine_urls(state)],
                         [("prefill", 0, "http://p0:8000"), ("prefill", 1, "http://p1:8000"),
                          ("decode", 0, "http://d0:8000")])

    def test_commands_that_disagree_with_the_handoff_are_refused_by_name(self):
        def node(i, cmd):
            block = provided_block()
            block["nodes"][i]["launch_command"] = cmd
            return block
        other_router = provided_block()
        other_router["router"]["launch_command"] = router_cmd(["p"], ["d", "x"])
        cases = (
            (node(0, sg_cmd("prefill").replace("/m", "/other")), "serves '/other'"),
            (node(1, sg_cmd("decode").replace("--tp-size 8", "--tp-size 4")), "TP 4 is not the engine TP 8"),
            (node(1, sg_cmd("prefill")), "decode node but its command serves 'prefill'"),
            (node(0, sg_cmd("prefill").replace("--port 8000", "")), "names no --port"),
            (node(0, sg_cmd("prefill", env="PYTHONPATH=/x ")), "owns per run"),
            (other_router, r"the router names .*http://x:8000"),
            (provided_block(router=None) | {"router": {}}, "need the router's launch_command"),
            (provided_block(serve_port=9000), "the router listens on --port 8000"),
        )
        for block, why in cases:
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                mn.normalize_spec(block, 8)

    def test_replicas_of_one_role_must_run_one_command(self):
        block = provided_block(prefill=("p0", "p1"))
        block["nodes"][1]["launch_command"] += " --chunked-prefill-size 4096"
        with self.assertRaisesRegex(ValueError, "every prefill replica must run one command"):
            mn.normalize_spec(block, 8)
        block["nodes"][1]["launch_command"] = sg_cmd("prefill").replace("--port 8000", "--port 8001")
        block["router"]["launch_command"] = block["router"]["launch_command"].replace("p1:8000", "p1:8001")
        self.assertEqual(mn.normalize_spec(block, 8)["nodes"][1]["port"], 8001)

    def test_a_vllm_command_says_its_role_through_kv_transfer_config(self):
        facts = mn.engine_facts("vllm", "VLLM_X=1 vllm serve /m --port 8100 -tp 8 "
                                        "--kv-transfer-config '{\"kv_role\": \"kv_producer\"}'")
        self.assertEqual((facts["env"], facts["model"], facts["tp"], facts["port"], facts["role"],
                          facts["bootstrap_port"]), (["VLLM_X=1"], "/m", 8, 8100, "prefill", None))

    def test_flags_the_command_owns_are_stripped_from_geaks(self):
        flags = mn.structural_flags("sglang")
        self.assertEqual(mn.strip_flags("--chunked-prefill-size 4096 --port 1 --tp-size=4 --host x", flags),
                         ("--chunked-prefill-size 4096", ["--port 1", "--tp-size=4", "--host x"]))
        self.assertEqual(mn.strip_flags(" --a '1 2' ", flags), ("--a '1 2'", []))


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec = {"nodes": ["10.10.0.166", "10.10.0.32"], "shared_fs": self.tmp,
                     "state": os.path.join(self.tmp, "mn_state.json")}
        self.tp = 16
        self.probes = [FC54, C489]
        self.sent = []
        self.run_reply = node_reply
        # The driver: not one of the nodes, sees no GPU, reaches every port it dials.
        self.here = ("sandbox/b", ("eth0", "10.0.9.9", 24))
        self.dialed, self.reaches, self.gpus = [], True, 0
        # The InferenceX client's check on this machine: the model each call asked about, and its answer.
        self.ix_models, self.ix_error, self.client = [], "", "inferencex"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def handler(self, task, node, args):
        if task is mn.probe_node:
            return dict(F293 if node["rank"] == "lb" else self.probes[node["rank"]])
        self.sent.append((node, args[0]))
        return dict(self.run_reply(node, args[0]))

    def check(self, here=None, argv=()):
        with contextlib.ExitStack() as stack:
            stack.enter_context(fake_ssh(self.handler))
            stack.enter_context(on_machine(*self.here))
            stack.enter_context(mock.patch.object(mn, "HERE", here or self.tmp))
            stack.enter_context(mock.patch.object(mn, "ACTIVE_PIN", os.path.join(self.tmp, "state_path")))
            stack.enter_context(mock.patch.dict(os.environ, {"GEAK_MN_SPEC": json.dumps({**self.spec, "tp": self.tp}),
                                                             "BENCH_CLIENT": self.client}))
            stack.enter_context(mock.patch.object(
                mn, "prove_shared", lambda s, nodes: {"per_node": [{"ok": True} for _ in nodes]}))
            stack.enter_context(mock.patch.object(
                mn, "dials_here", lambda ip, port: self.dialed.append((ip, port)) or self.reaches))
            stack.enter_context(mock.patch.object(mn, "visible_gpus", lambda: self.gpus))
            stack.enter_context(mock.patch.object(
                mn, "inferencex_client_error", lambda model: self.ix_models.append(model) or self.ix_error))
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                rc = mn.cmd_check(list(argv))
        self.verdict = json.loads(out.getvalue().splitlines()[-1][len(mn.CHECK_TAG):])
        return rc, err.getvalue()

    def pin(self):
        with open(self.spec["state"], encoding="utf-8") as f:
            return json.load(f)

    def provide(self):
        """1P1D from launch_commands; the router is a third machine (node-f293) with no GPU work."""
        self.spec = provided_block(prefill=("10.10.0.166",), decode=("10.10.0.32",),
                                   router="10.10.0.183", shared_fs=self.tmp, state=self.spec["state"])
        self.tp = 8

    def test_check_runs_on_a_driver_that_is_no_node_and_on_a_node(self):
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.here = ("node-c489/b", ("ens50f0", "10.10.0.32", 24))
        rc, err = self.check()
        self.assertEqual(rc, 0, err)

    def test_a_driver_that_sees_a_gpu_is_refused(self):
        self.gpus = 8
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("sees 8 GPU(s)", err)
        self.assertIn("ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=", err)
        self.assertFalse(os.path.exists(self.spec["state"]))

    def test_the_driver_must_reach_the_client_port(self):
        self.reaches = False
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("cannot reach the client port 10.10.0.166:30890 on rank0", err)
        self.assertEqual(self.dialed, [("10.10.0.166", 30890)])

    def test_an_unset_client_is_bench_e2es_native(self):
        self.client = ""
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.pin()["bench_client"], "native")
        self.assertEqual(self.ix_models, [])

    def test_the_inferencex_client_needs_its_checkout_here(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"INFERENCEX_PATH": root}):
            os.environ.pop("INFERENCEX_BENCH_SERVING", None)
            self.assertIn(f"INFERENCEX_PATH={root} has no benchmark_serving.py on this machine",
                          mn.inferencex_client_error(""))
            os.makedirs(os.path.join(root, "utils", "bench_serving"))
            open(os.path.join(root, "utils", "bench_serving", "benchmark_serving.py"), "w").close()
            self.assertNotIn("benchmark_serving.py", mn.inferencex_client_error(""))

    def test_the_inferencex_client_must_run_here(self):
        self.spec["model"] = "/models/R1"
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.ix_models, ["/models/R1"])
        self.assertEqual(self.pin()["bench_client"], "inferencex")
        self.ix_error = "python3 cannot import numpy, aiohttp, tqdm, transformers, huggingface_hub: No module named 'aiohttp'"
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("the InferenceX bench client runs on this machine and python3 cannot import", err)

    def test_the_native_client_must_import_on_rank0_and_reach_the_port_from_there(self):
        self.client, self.spec["backend"] = "native", "sglang"
        reply = self.run_reply
        self.run_reply = lambda node, script: (
            {"rc": 1, "out": "", "err": "ModuleNotFoundError: No module named 'sglang'"}
            if "import sglang.bench_serving" in script else reply(node, script))
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("BENCH_CLIENT=native runs on rank0 node-fc54, which cannot import sglang.bench_serving", err)
        self.assertEqual(self.ix_models, [])
        self.run_reply = reply
        self.provide()
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.pin()["bench_client"], "native")
        rank0 = "10.10.0.166"
        dialed_lb = [s for n, s in self.sent if n["ip"] == rank0 and "/dev/tcp/10.10.0.183/" in s]
        self.assertTrue(dialed_lb, "rank0 dials the load balancer the client talks to")

    def test_aiperf_is_checked_by_the_agentx_preflight(self):
        self.client = "agentx"
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.ix_models, [])
        self.assertFalse([s for _, s in self.sent if "bench_serving" in s or "benchmarks" in s])

    def test_a_run_directory_off_the_shared_fs_is_refused(self):
        self.spec["work_dir"] = "/somewhere/else"
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("the run directory /somewhere/else is not under", err)

    def test_the_kernel_tools_are_staged_in_the_run_directory(self):
        self.spec["work_dir"] = os.path.join(self.tmp, "eval")
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        tools = self.pin()["tools_dir"]
        self.assertEqual(tools, os.path.join(self.tmp, "eval", ".mn_tools"))
        for name in mn.TOOLS:
            with open(os.path.join(mn.KERNEL_SCRIPTS, name), "rb") as src, open(os.path.join(tools, name), "rb") as dst:
                self.assertEqual(src.read(), dst.read(), name)

    def test_a_bad_spec_is_a_verdict_not_a_traceback(self):
        self.spec["topology"] = "ring"
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("GEAK_MN_SPEC: unknown topology", err)
        self.assertEqual(json.load(open(os.path.join(self.tmp, "mn_check.json")))["verdict"], "refused")

    def test_a_node_ssh_cannot_reach_is_a_verdict_not_a_traceback(self):
        reply = self.handler

        def handler(task, node, args):
            if task is mn.probe_node and node["rank"] == 1:
                raise RuntimeError("probe on 10.10.0.32 returned nothing (rc=255): Permission denied (publickey)")
            return reply(task, node, args)
        self.handler = handler
        rc, err = self.check()
        self.assertEqual((rc, self.verdict["verdict"]), (2, "refused"))
        self.assertIn("10.10.0.32 returned nothing (rc=255): Permission denied", self.verdict["problems"][0])
        self.assertFalse(os.path.exists(self.spec["state"]))

    def test_the_handoffs_nodes_become_the_pin(self):
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        state = self.pin()
        self.assertEqual(state["master_ip"], "10.10.0.166")
        self.assertEqual((state["topology"], state["mode"], state["mn_dir"]), ("tp_span", "generated", self.tmp))
        self.assertNotIn("transport", state)
        self.assertEqual([n["node_id"] for n in state["nodes"]], ["node-fc54/b", "node-c489/b"])
        self.assertEqual([(n["tp"], n["gpus"]) for n in state["nodes"]], [(16, 8), (16, 8)])

    def test_a_passed_check_hands_back_the_spec_and_claims_the_machine(self):
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.verdict["verdict"], "ok")
        self.assertEqual(self.verdict["spec"]["state"], self.spec["state"])
        self.assertEqual(self.verdict["spec"]["layout"], [{"tp": 16, "gpus": 8}, {"tp": 16, "gpus": 8}])
        pointer = json.load(open(os.path.join(self.tmp, "state_path")))
        self.assertEqual((pointer["state"], pointer["pid"]), (self.spec["state"], os.getppid()))
        self.assertEqual(pointer["mn_py"], os.path.join(self.tmp, "mn.py"))
        self.assertEqual(json.load(open(os.path.join(self.tmp, "mn_check.json")))["verdict"], "ok")
        self.assertEqual([n["ngpu"] for n in self.pin()["nodes"]], [8, 8])

    def test_owner_claude_hands_the_pin_to_the_claude_code_process(self):
        with mock.patch.object(mn, "claude_ancestor", lambda: 4242):
            rc, err = self.check(argv=["--owner", "claude"])
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.load(open(os.path.join(self.tmp, "state_path")))["pid"], 4242)

    def test_owner_claude_without_one_is_refused(self):
        with mock.patch.object(mn, "claude_ancestor", lambda: 0):
            rc, err = self.check(argv=["--owner", "claude"])
        self.assertEqual(rc, 2)
        self.assertIn("no Claude Code process above this shell", err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "state_path")))

    def test_one_node_is_a_single_machine_server(self):
        self.spec["nodes"], self.tp = ["10.10.0.166"], 8
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual([(n["tp"], n["gpus"]) for n in self.pin()["nodes"]], [(8, 8)])
        self.assertEqual([s for n, s in self.sent if "/dev/tcp/" in s], [], "a single node has no route between nodes")
        self.assertEqual(self.dialed, [("10.10.0.166", 30890)])

    def test_the_other_nodes_must_reach_the_dist_init_port(self):
        self.run_reply = lambda node, script: (
            {"rc": 1, "out": "", "err": "Connection timed out"} if "/dev/tcp/" in script else node_reply(node, script))
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("rank1 node-c489 cannot reach the dist-init port 10.10.0.166:30531 on rank0", err)
        listen = [s for n, s in self.sent if "sport = :30531" in s]
        self.assertEqual(len(listen), 1)
        self.assertTrue(any(s.startswith("kill 99") for n, s in self.sent), "the listener is stopped")
        self.assertEqual(json.load(open(os.path.join(self.tmp, "mn_check.json")))["verdict"], "refused")

    def test_pd_tests_the_ports_each_side_dials(self):
        self.spec["topology"], self.tp = "pd_disagg", 8
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        dials = sorted((n["ip"], re.search(r"/dev/tcp/(\S+)'", s).group(1)) for n, s in self.sent if "/dev/tcp/" in s)
        # master (prefill head) dials the decode HTTP port; decode dials prefill's bootstrap port.
        self.assertEqual(dials, [("10.10.0.166", "10.10.0.32/30892"), ("10.10.0.32", "10.10.0.166/30898")])
        # The driver dials the LB and both engines' HTTP ports (client, health, profiler).
        self.assertEqual(self.dialed, [("10.10.0.166", 30890), ("10.10.0.166", 30891), ("10.10.0.32", 30892)])

    def test_pd_transport_settings_are_pinned(self):
        self.spec.update(topology="pd_disagg", pd={"backend": "mooncake", "ib_device": "mlx5_0"})
        self.tp = 8
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        pd = mn.pd_config(self.pin())
        self.assertEqual((pd["transfer_backend"], pd["ib_device"], pd["lb"]), ("mooncake", "mlx5_0", "router"))

    def test_too_little_free_vram_is_refused_with_the_reason(self):
        self.probes = [FC54, dict(C489, free_vram_mib=50000)]
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("rank1 node-c489: its fullest card has 50000 of 192000 MiB free", err)
        report = json.load(open(os.path.join(self.tmp, "mn_check.json")))
        self.assertEqual(report["verdict"], "refused")
        self.assertIn("50000 of 192000 MiB", report["problems"][0])
        self.assertFalse(os.path.exists(self.spec["state"]))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "state_path")), "a refusal claims nothing")

    def test_a_bound_port_is_refused(self):
        self.probes = [dict(FC54, busy_ports=[30890]), C489]
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("rank0 node-fc54: port(s) [30890] already bound", err)

    def test_nodes_on_different_stacks_are_refused(self):
        self.probes = [dict(FC54, sw={"sglang": "0.5.1"}), dict(C489, sw={"sglang": "0.5.2"})]
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("the nodes load different stacks", err)

    def test_nodes_with_different_gpus_are_refused(self):
        mi300 = {"gfx": "gfx942", "target": "unknown", "marketing_name": "AMD Instinct MI300X", "physical_cu_count": 304}
        self.probes = [dict(FC54, gpu=mi300), dict(C489, gpu=dict(mi300, marketing_name="AMD Instinct MI308X",
                                                                  physical_cu_count=80))]
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("the nodes have different GPUs", err)

    def test_the_gpu_is_pinned(self):
        gpu = {"gfx": "gfx942", "target": "unknown", "marketing_name": "AMD Instinct MI308X", "physical_cu_count": 80}
        self.probes = [dict(p, sw={"sglang": "0.5.1"}, gpu=gpu) for p in self.probes]
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.pin()["gpu"], gpu)

    def test_pd_engines_with_different_tp_are_pinned_per_rank(self):
        self.spec.update(topology="pd_disagg", tp_by_role={"prefill": 8, "decode": 4})
        self.tp = 8
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual([(n["role"], n["tp"], n["gpus"]) for n in self.pin()["nodes"]],
                         [("prefill", 8, 8), ("decode", 4, 4)])

    def test_a_tp_that_does_not_split_over_the_nodes_is_refused(self):
        self.tp = 15
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("TP=15 does not split evenly over 2 nodes", err)

    def test_too_few_cards_is_refused_and_nothing_is_pinned(self):
        self.probes = [dict(p, ngpu=4) for p in self.probes]
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("has 4 GPUs, needs 8", err)
        self.assertFalse(os.path.exists(self.spec["state"]))

    def test_pd_disagg_takes_exactly_two_nodes(self):
        self.spec["topology"] = "pd_disagg"
        self.spec["nodes"].append("10.10.0.183")
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("pd_disagg needs 2 nodes", err)

    def test_a_shared_fs_that_is_not_a_directory_is_refused(self):
        missing = os.path.join(self.tmp, "missing")
        self.spec.update(shared_fs=missing, state=os.path.join(missing, "mn_state.json"))
        rc, err = self.check(here=missing)
        self.assertEqual(rc, 2)
        self.assertIn("is not a directory here", err)

    def test_geak_need_not_be_on_the_shared_fs(self):
        rc, err = self.check(here="/somewhere/else")
        self.assertEqual(rc, 0, err)

    def test_ssh_settings_and_a_nodes_own_ssh_are_pinned(self):
        self.spec.update(ssh={"key": "/k/id", "port": 2233},
                         nodes=[{"host": "10.10.0.166", "ssh": {"port": 2244}}, "10.10.0.32"])
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        state = self.pin()
        self.assertEqual(state["ssh"], {"key": "/k/id", "port": 2233})
        self.assertEqual([n.get("ssh") for n in state["nodes"]], [{"port": 2244}, None])

    def test_provided_commands_pin_their_ports_and_the_router_machine(self):
        self.provide()
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        state = self.pin()
        self.assertEqual((state["mode"], state["backend"], state["master_ip"]), ("provided", "sglang", "10.10.0.183"))
        self.assertEqual([(n["role"], n["port"], n.get("bootstrap_port")) for n in state["nodes"]],
                         [("prefill", 8000, 8998), ("decode", 8000, None)])
        self.assertEqual(state["nodes"][1]["launch_command"], sg_cmd("decode"))
        router = state["router"]
        self.assertEqual((router["rank"], router["node_id"], router["port"]), ("lb", "node-f293/b", 8000))
        self.assertEqual(router["launch_command"], self.spec["router"]["launch_command"])
        self.assertNotIn("ngpu", router, "the router holds no GPU")

    def test_provided_ports_are_dialed_where_the_commands_bind_them(self):
        self.provide()
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        dials = sorted((n["ip"], re.search(r"/dev/tcp/(\S+)'", s).group(1)) for n, s in self.sent if "/dev/tcp/" in s)
        # The router dials both engines; decode dials prefill's bootstrap port.
        self.assertEqual(dials, [("10.10.0.183", "10.10.0.166/8000"), ("10.10.0.183", "10.10.0.32/8000"),
                                 ("10.10.0.32", "10.10.0.166/8998")])
        self.assertEqual(self.dialed, [("10.10.0.183", 8000), ("10.10.0.166", 8000), ("10.10.0.32", 8000)])

    def test_a_busy_router_port_is_refused(self):
        self.provide()
        with mock.patch.dict(F293, busy_ports=[8000]):
            rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("router node-f293: port 8000 already bound", err)

    def one_machine(self, decode_port=8002):
        """1P1D on node-fc54: prefill on cards 0-3 at :8001, decode on 4-7, the router at :8000."""
        host = "10.10.0.166"
        prefill = sg_cmd("prefill").replace("--tp-size 8", "--tp-size 4").replace("--port 8000", "--port 8001")
        decode = sg_cmd("decode").replace("--tp-size 8", "--tp-size 4").replace("--port 8000", f"--port {decode_port}")
        self.spec = provided_block(prefill=(host,), decode=(host,), router=host, tp_by_role={"prefill": 4, "decode": 4},
                                   shared_fs=self.tmp, state=self.spec["state"])
        self.spec["nodes"] = [{"host": host, "role": "prefill", "launch_command": prefill, "gpu_ids": "0,1,2,3"},
                              {"host": host, "role": "decode", "launch_command": decode, "gpu_ids": "4,5,6,7"}]
        self.spec["router"]["launch_command"] = (
            f"python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://{host}:8001 8998 "
            f"--decode http://{host}:{decode_port} --host 0.0.0.0 --port 8000 --model-path /m")
        self.tp = 4
        reply = self.handler
        self.handler = lambda task, node, args: dict(FC54) if task is mn.probe_node else reply(task, node, args)

    def test_one_machine_serves_1p1d_on_disjoint_cards(self):
        self.one_machine()
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        state = self.pin()
        self.assertEqual([(n["role"], n["node_id"], n["port"], n["gpu_ids"]) for n in state["nodes"]],
                         [("prefill", "node-fc54/b", 8001, [0, 1, 2, 3]), ("decode", "node-fc54/b", 8002, [4, 5, 6, 7])])
        self.assertEqual((state["router"]["node_id"], state["router"]["port"]), ("node-fc54/b", 8000))

    def test_engines_on_one_machine_must_bind_their_own_ports(self):
        self.one_machine(decode_port=8001)
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("rank0 and rank1 both bind port 8001 on node-fc54", err)

    def test_free_vram_is_judged_on_the_cards_a_rank_names(self):
        self.one_machine()
        crowded = dict(FC54, free_vram_mib=90000, card_total_mib=[192000] * 8,
                       card_free_mib=[90000, 90000, 90000, 90000, 190000, 190000, 190000, 190000])
        self.handler = lambda task, node, args, reply=self.handler: (
            dict(crowded) if task is mn.probe_node else reply(task, node, args))
        rc, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("rank0 node-fc54: its fullest card has 90000 of 192000 MiB free", err)
        self.assertNotIn("rank1 node-fc54: its fullest card", err, "decode's cards 4-7 are free")


class TransportTest(unittest.TestCase):
    NODE = {"rank": 1, "node_id": "node-c489/b1", "hostname": "node-c489", "ip": "10.0.0.2"}

    def state(self, **ssh):
        return {"ssh": ssh, "mn_dir": "/shared/mn", "nodes": [dict(self.NODE)]}

    def argv(self, node=None, **ssh):
        with on_machine("driver/b0"):
            return mn.ssh_argv(self.state(**ssh), node or self.NODE)

    def test_config_rejects_unknown_keys_and_roles(self):
        self.assertEqual(mn.ssh_config({}), {})
        for bad in ({"pasword": "x"}, {"port_offsets": {"router": 1}}, [1]):
            with self.assertRaises(ValueError):
                mn.ssh_config(bad)

    def test_argv_carries_port_offset_key_user_and_prefix(self):
        with on_machine("driver/b0"):
            argv = mn.ssh_argv(dict(self.state(key="/k/id", port=2233, user="root", port_offsets={"decode": 10},
                                               prefix="sudo -n docker exec -i c1"), topology="pd_disagg"),
                               dict(self.NODE, role="decode"))
        self.assertEqual(argv[argv.index("-p") + 1], "2243")
        self.assertEqual(argv[argv.index("-i") + 1], "/k/id")
        self.assertIn("StrictHostKeyChecking=accept-new", argv)
        self.assertEqual(argv[-2:], ["root@10.0.0.2", "sudo -n docker exec -i c1 bash -s"])

    def test_known_hosts_makes_host_keys_strict(self):
        argv = self.argv(known_hosts="/k/known")
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("UserKnownHostsFile=/k/known", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "22")
        self.assertEqual(argv[-2:], ["10.0.0.2", "bash -s"])

    def test_a_nodes_own_ssh_overrides_the_pins(self):
        argv = self.argv(dict(self.NODE, ssh={"port": 2244, "user": "geak"}), port=2233, user="root")
        self.assertEqual(argv[argv.index("-p") + 1], "2244")
        self.assertEqual(argv[-2], "geak@10.0.0.2")

    def test_this_machine_runs_without_ssh(self):
        with on_machine("node-c489/b1"):
            self.assertEqual(mn.ssh_argv(self.state(prefix="docker exec -i c"), self.NODE)[-2:], ["bash", "-s"])
            self.assertNotIn("ssh", mn.ssh_argv(self.state(), self.NODE))
            self.assertEqual(mn.ssh_argv(self.state(), dict(self.NODE, node_id="peer/b"))[0], "ssh")

    def test_before_the_pin_an_address_is_this_machine_unless_a_prefix_enters_a_container(self):
        host = {"node_id": "", "rank": 0, "ip": "10.0.0.2"}
        with on_machine("node-c489/b1", ("ens1", "10.0.0.2", 24)):
            self.assertEqual(mn.ssh_argv(self.state(), host)[0], "env")
            self.assertEqual(mn.ssh_argv(self.state(prefix="docker exec -i c"), host)[0], "ssh")

    @staticmethod
    def ssh_returns(rc, out, err=b""):
        return mock.patch.object(mn.subprocess, "run", side_effect=lambda *a, **k:
                                 subprocess.CompletedProcess([], rc, stdout=out, stderr=err))

    def test_probe_ships_mn_py_and_the_identity_probe_by_content(self):
        with on_machine("driver/b0"), self.ssh_returns(0, b'noise\nMN_PROBE={"hostname": "h"}\n') as run:
            out = mn.probe_all(self.state(), [self.NODE], [30000, 30001])
        self.assertEqual(out, [{"hostname": "h"}])
        sent = run.call_args.kwargs["input"]
        self.assertIsInstance(sent, bytes)
        self.assertNotIn(b"\r", sent)
        self.assertIn(b"python3 - probe 30000 30001 <<'GEAK_MN_PY'\n", sent)
        self.assertIn(b'MN_GPU_IDENTITY="$(python3 - 2>&1 <<', sent)
        for path in (os.path.join(MN, "mn.py"), mn.GPU_IDENTITY_SCRIPT):
            with open(path, "rb") as f:
                self.assertIn(f.read().replace(b"\r\n", b"\n"), sent.replace(b"\r\n", b"\n"))
        self.assertNotIn(b"/shared/mn", sent)
        self.assertIn(b"/proc/1/environ", sent)

    @unittest.skipUnless(shutil.which("bash") and sys.platform.startswith("linux"), "bash on Linux")
    def test_the_shipped_probe_runs_on_a_node_without_geak(self):
        node = {"rank": 0, "node_id": mn.local_identity(), "hostname": socket.gethostname(), "ip": "127.0.0.1"}
        with tempfile.TemporaryDirectory() as cwd:
            old = os.getcwd()
            os.chdir(cwd)
            try:
                out = mn.probe({"ssh": {}, "nodes": [node]}, node, [1])
            finally:
                os.chdir(old)
        self.assertEqual(out["hostname"], socket.gethostname())
        self.assertTrue(out["gpu"].get("gfx") or out["gpu"].get("error"))

    def test_a_probe_without_its_tag_is_an_error(self):
        with on_machine("driver/b0"), self.ssh_returns(127, b"", b"python3: not found"):
            with self.assertRaisesRegex(RuntimeError, "python3: not found"):
                mn.probe(self.state(), self.NODE)

    def test_a_node_that_answers_as_another_machine_is_refused(self):
        with on_machine("driver/b0"), self.ssh_returns(0, b"node-c489/b2\n"):
            with self.assertRaises(SystemExit) as exc:
                mn.resolve(self.state())
        self.assertIn("Refusing to substitute", str(exc.exception))

    def test_the_same_machine_resolves(self):
        with on_machine("driver/b0"), self.ssh_returns(0, b"node-c489/b1\n"):
            self.assertEqual(mn.resolve(self.state()), [self.NODE])

    @staticmethod
    def remote(code):
        """ssh replaced by a local python that runs `code` on what it is sent."""
        return mock.patch.object(mn, "ssh_argv", lambda state, node: [sys.executable, "-c", code])

    def test_stream_sends_bytes_and_returns_the_remote_exit_code(self):
        n = len((mn.POD_ENV + "echo hi\n").encode())
        code = f"import sys; sent = sys.stdin.buffer.read({n}); sys.exit(3 if sent.endswith(b'echo hi\\n') and b'\\r' not in sent else 1)"
        with self.remote(code):
            self.assertEqual(mn.ssh_stream(self.state(), self.NODE, "echo hi\n", 30), 3)

    def test_stream_keeps_stdin_open_and_hangs_up_at_its_timeout(self):
        # The remote waits for EOF, as an exec_script watchdog does: only the hang-up ends it.
        code = "import sys; sys.stdin.buffer.read(); sys.exit(5)"
        start = time.time()
        with self.remote(code):
            self.assertEqual(mn.ssh_stream(self.state(), self.NODE, "echo hi\n", 1), 5)
        self.assertLess(time.time() - start, 20)

    def test_exec_script_marks_the_hop_and_watches_stdin(self):
        script = mn.exec_script("echo hi\n", grace=7)
        self.assertTrue(script.startswith("{\nexport GEAK_MN_HOP=1\n"))
        self.assertIn('setsid bash "$body" </dev/null &', script)
        self.assertIn('pkill -TERM -s "$1"; sleep 7; pkill -KILL -s "$1"', script)
        self.assertIn("\necho hi\n\n", script)

    @unittest.skipUnless(shutil.which("bash") and shutil.which("pkill") and sys.platform.startswith("linux"),
                         "bash + pkill on Linux")
    def test_a_hang_up_terms_the_command_then_kills_it(self):
        def run(body, grace, hang_up_after):
            # On a node this runs under a fresh sshd shell. Here it inherits the pytest process,
            # where an in-process run_e2e.main() earlier in the suite leaves SIGTERM ignored.
            proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL))
            proc.stdin.write(mn.exec_script(body, grace=grace).encode())
            proc.stdin.flush()
            time.sleep(hang_up_after)
            start = time.time()
            proc.stdin.close()
            out = proc.stdout.read().decode()
            return proc.wait(), out, time.time() - start

        rc, out, took = run("echo hop=$GEAK_MN_HOP; sleep 60\n", 30, 2)
        self.assertIn("hop=1", out)
        self.assertNotEqual(rc, 0)
        self.assertLess(took, 10, "TERM ends the command at once")
        rc, _out, took = run("trap '' TERM; sleep 60\n", 2, 2)
        self.assertNotEqual(rc, 0)
        self.assertLess(took, 15, "KILL follows the grace when TERM is ignored")


class RankLayoutTest(unittest.TestCase):
    def test_tp_span_splits_one_world_evenly(self):
        self.assertEqual(mn.rank_layout("tp_span", 16, 2),
                         [{"tp": 16, "gpus": 8}, {"tp": 16, "gpus": 8}])

    def test_dp_attention_with_ep_is_still_one_tp16_world(self):
        # dp2 tp8 ep16 on two nodes is sglang --tp-size 16; DP/EP ride in the server args.
        self.assertEqual([r["gpus"] for r in mn.rank_layout("tp_span", 16, 2)], [8, 8])

    def test_tp_span_refuses_an_uneven_or_short_world(self):
        for tp in (12, 1):
            with self.assertRaises(ValueError):
                mn.rank_layout("tp_span", tp, 4 if tp == 1 else 8)

    def test_pd_defaults_both_engines_to_tp_and_honours_tp_by_role(self):
        self.assertEqual([(r["role"], r["tp"]) for r in mn.rank_layout("pd_disagg", 8, 2)],
                         [("prefill", 8), ("decode", 8)])
        self.assertEqual(mn.rank_layout("pd_disagg", 8, 2, {"decode": 4})[1],
                         {"role": "decode", "tp": 4, "gpus": 4})

    def test_pd_refuses_a_third_node(self):
        with self.assertRaises(ValueError):
            mn.rank_layout("pd_disagg", 8, 3)

    def test_pd_decode_engine_over_two_nodes(self):
        # 1P (TP8) + 1D (dp2 tp8 ep16 = sglang --tp 16 over two nodes).
        layout = mn.rank_layout("pd_disagg", 8, 3, {"decode": 16}, {"decode": 2})
        self.assertEqual(layout, [
            {"role": "prefill", "tp": 8, "gpus": 8},
            {"role": "decode", "tp": 16, "gpus": 8, "engine_rank": 0, "engine_nnodes": 2},
            {"role": "decode", "tp": 16, "gpus": 8, "engine_rank": 1, "engine_nnodes": 2},
        ])
        with self.assertRaises(ValueError):
            mn.rank_layout("pd_disagg", 8, 2, {"decode": 16}, {"decode": 2})
        with self.assertRaises(ValueError):
            mn.rank_layout("pd_disagg", 8, 3, {"decode": 15}, {"decode": 2})


HAVE_SH = bool(shutil.which("bash"))

FAKE_ADAPTER = """\
adapter_dist_mode() { echo per_rank; }
adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }
adapter_launch() {
  echo "FINAL_ARGS=[$EXTRA_SERVER_ARGS]"
  echo "FINAL_ENV=[$EXTRA_ENV]"
  echo "GPU=$GPU HOST=$HOST PORT=$PORT TP=$TP MODEL=$MODEL"
  SERVER_PID=4242
}
"""


def adapter_root(tmp, name, body):
    """tmp as a scripts/ tree with adapters/<name>.sh; mn.py (HERE) looks only there."""
    os.makedirs(os.path.join(tmp, "multi_node"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "adapters"), exist_ok=True)
    with open(os.path.join(tmp, "adapters", f"{name}.sh"), "w", encoding="utf-8") as f:
        f.write(body)
    return mock.patch.object(mn, "HERE", os.path.join(tmp, "multi_node"))


def rank_vars(tmp, **over):
    kw = dict(model="/models/m", tp=16, gpu="0,1,2,3,4,5,6,7", mem_fraction="0.90",
              port=30890, base_url="http://10.10.0.166:30890",
              log=os.path.join(tmp, "server.rank1.log"),
              profile_dir=os.path.join(tmp, "prof"), overlay="/ovl", pythonpath="",
              extra_env="SHARED_V=1 SWEPT_V=2",
              extra_args="--attention-backend dsv4 --page-size 64")
    kw.update(over)
    return mn.rank_env_vars(**kw)


def render(tmp, structural='adapter_dist_args 1 2 "$MASTER_ADDR" 30531', **over):
    # No node-facts head: it probes a real NIC.
    return mn.launch_script(["export MASTER_ADDR=10.10.0.166"], rank_vars(tmp, **over),
                            os.path.join(tmp, "fake.sh"), "adapter_launch", structural)


class Layering(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_ADAPTER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_script_delegates_to_the_adapter_for_the_launch_line(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("STARTED_PID=4242", out.stdout)
        self.assertIn("GPU=0,1,2,3,4,5,6,7 HOST=0.0.0.0 PORT=30890", out.stdout)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_swept_flags_go_first_and_the_rank_triple_last(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertLess(args.index("--page-size 64"), args.index("--node-rank 1"))
        self.assertIn("--nnodes 2 --node-rank 1 --dist-init-addr 10.10.0.166:30531",
                      args)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_env_reaches_the_adapter_as_given(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        env = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ENV=")][0]
        self.assertEqual(env, "FINAL_ENV=[SHARED_V=1 SWEPT_V=2]")


class AdapterSeam(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, body):
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        return p

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_a_backend_that_cannot_say_how_it_spells_rank_is_refused(self):
        with adapter_root(self.tmp, "quiet", "adapter_launch() { :; }\n"):
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("quiet", self.tmp)
        self.assertIn("adapter_dist_args", str(e.exception))

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_a_head_only_backend_is_refused_rather_than_launched_as_per_rank(self):
        with adapter_root(self.tmp, "vllm", "adapter_dist_mode() { echo head_only; }\n"
                                            "adapter_dist_args() { echo x; }\n"):
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("vllm", self.tmp)
        self.assertIn("head_only", str(e.exception))

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_an_adapter_off_the_shared_mount_is_refused(self):
        with adapter_root(self.tmp, "sglang", "adapter_dist_args() { echo x; }\n"):
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("sglang", os.path.join(self.tmp, "elsewhere"))
        self.assertIn("shared", str(e.exception))

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_real_sglang_adapter_declares_per_rank(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        self.assertTrue(
            mn.resolve_adapter("sglang", root).endswith("adapters/sglang.sh"))

    @unittest.skipUnless(shutil.which("bash"), "needs bash")
    def test_the_real_vllm_adapter_runs_rank0_serving_and_the_rest_headless(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        path = mn.resolve_adapter("vllm", root)
        self.assertTrue(path.endswith("adapters/vllm.sh"))

        def dist(rank):
            return subprocess.run(
                ["bash", "-c", f". {shlex.quote(path)}; adapter_dist_args {rank} 2 10.0.0.1 30531"],
                capture_output=True, text=True, check=True).stdout.split()
        r0, r1 = dist(0), dist(1)
        for args in (r0, r1):
            self.assertEqual(args[args.index("--distributed-executor-backend") + 1], "mp")
            self.assertEqual(args[args.index("--nnodes") + 1], "2")
            self.assertEqual(args[args.index("--master-addr") + 1], "10.0.0.1")
            self.assertEqual(args[args.index("--master-port") + 1], "30531")
        self.assertEqual(r1[r1.index("--node-rank") + 1], "1")
        self.assertNotIn("--headless", r0)
        self.assertIn("--headless", r1)

    def test_teardown_also_kills_engine_children_that_rename_themselves(self):
        for title in ("sglang::scheduler_TP0", "VLLM::Worker_TP3", "VLLM::EngineCore"):
            self.assertRegex(title, mn.ENGINE_PATTERN)
        self.assertNotRegex("pkill -f '(sglang|VLL[M])::'", mn.ENGINE_PATTERN)

    def test_teardown_kills_workers_that_kept_their_multiprocessing_argv(self):
        for argv in ("/usr/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=42, pipe_handle=31) --multiprocessing-fork",
                     "/usr/bin/python -c from multiprocessing.resource_tracker import main;main(41)"):
            self.assertRegex(argv, mn.WORKER_PATTERN)
        self.assertNotRegex(f"pkill -f '{mn.WORKER_PATTERN}'", mn.WORKER_PATTERN)
        self.assertIn(f"kill -9 $({mn.OUR_WORKER_PIDS})", mn.KILL_OURS)

    def test_no_kill_pattern_matches_the_kill_script_itself(self):
        # pkill spares itself but not the `bash -c` running KILL_OURS; a match
        # there ends the script before the later patterns run.
        for p in (mn.SERVE_PATTERN, mn.ENGINE_PATTERN, mn.WORKER_PATTERN, mn.LB_PATTERN):
            self.assertNotRegex(mn.KILL_OURS, p)

    @unittest.skipUnless(shutil.which("bash") and sys.platform.startswith("linux"), "needs linux bash")
    def test_kill_script_reaches_its_last_step(self):
        # Neutralise kill/pkill so the script runs against nothing on this box.
        out = subprocess.run(["bash", "-c", "pkill() { :; }; kill() { :; }; sleep() { :; }; " + mn.KILL_OURS],
                             capture_output=True, text=True, timeout=60).stdout
        self.assertEqual(out.strip(), "killed")

    @unittest.skipUnless(shutil.which("bash") and sys.platform.startswith("linux"), "needs linux bash")
    def test_ray_dashboard_children_are_not_our_workers(self):
        worker = "from multiprocessing.spawn import spawn_main; import time; time.sleep(60)"
        dash = subprocess.Popen([sys.executable, "-c",
                                 f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {worker!r}]); time.sleep(60)",
                                 "ray/dashboard/dashboard.py"])
        ours = subprocess.Popen([sys.executable, "-c", worker])
        try:
            time.sleep(1)
            dash_child = subprocess.run(["bash", "-c", f"ps -o pid= --ppid {dash.pid}"],
                                        capture_output=True, text=True).stdout.split()
            self.assertEqual(len(dash_child), 1)
            pids = subprocess.run(["bash", "-c", mn.OUR_WORKER_PIDS],
                                  capture_output=True, text=True, timeout=30).stdout.split()
            self.assertIn(str(ours.pid), pids)
            self.assertNotIn(dash_child[0], pids)
        finally:
            subprocess.run(["bash", "-c", f"pkill -P {dash.pid}"])
            dash.kill()
            ours.kill()


LAUNCHER = os.path.join(SCRIPTS_DIR, "adapters", "launchers", "multinode.sh")
BASH = shutil.which("bash")


DRIVER = """set -uo pipefail
HERE="{scripts}"
source "{launcher}"
{stub}
adapter_profile_window
echo "WINDOW_RC=$?"
"""

STUB_NATIVE = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" {count} <<'PY'
import gzip, json, sys
d, n = sys.argv[1], int(sys.argv[2])
for tp in range(n):
    events = [{{"cat": "kernel", "name": "{collective}", "ts": 1, "dur": 9}}]
    with gzip.open(f"{{d}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz", "wt") as fh:
        fh.write(json.dumps({{"traceEvents": events}}))
PY
  return 0
}}
"""

STUB_STREAMING = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" {count} <<'PY' &
import gzip, io, json, sys, time
d, n = sys.argv[1], int(sys.argv[2])
paths = [f"{{d}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz" for tp in range(n)]
for path in paths:
    open(path, "wb").close()          # created, still zero bytes
blob = io.BytesIO()
with gzip.GzipFile(fileobj=blob, mode="wb") as gz:
    events = [{{"cat": "kernel", "name": "{collective}", "ts": 1, "dur": 9}}] * 200
    gz.write(json.dumps({{"traceEvents": events}}).encode())
raw = blob.getvalue()
for piece in range(4):
    time.sleep(2)
    lo, hi = len(raw) * piece // 4, len(raw) * (piece + 1) // 4
    for path in paths:
        with open(path, "ab") as fh:
            fh.write(raw[lo:hi])
PY
  sleep 1
  return 0
}}
"""

SHOWPIDS_HOST_DAEMON_ONLY = """
============================ ROCm System Management Interface ============================
===================================== KFD Processes ======================================
KFD process information:
PID    \tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY
2727680\tgpuagent    \t0     \t0        \t0        \t0
==========================================================================================
"""

SHOWPIDS_REAL_CO_TENANT = """
============================ ROCm System Management Interface ============================
===================================== KFD Processes ======================================
KFD process information:
PID    \tPROCESS NAME\tGPU(s)\tVRAM USED \tSDMA USED\tCU OCCUPANCY
3979771\tpython3     \t0     \t0         \t0        \t0
1792981\tpython3     \t1     \t1323409408\t0        \t0
==========================================================================================
"""


class ResidentGpuPidsTest(unittest.TestCase):
    def test_a_zero_vram_host_daemon_is_not_occupancy(self):
        holders, parked = mn._resident_gpu_pids(SHOWPIDS_HOST_DAEMON_ONLY)
        self.assertEqual(holders, [])
        self.assertEqual(parked, 1)

    def test_a_neighbour_holding_memory_still_counts(self):
        holders, parked = mn._resident_gpu_pids(SHOWPIDS_REAL_CO_TENANT)
        self.assertEqual(holders, ["1792981"])
        self.assertEqual(parked, 1)

    def test_no_kfd_processes_at_all(self):
        holders, parked = mn._resident_gpu_pids("no KFD processes\n")
        self.assertEqual((holders, parked), ([], 0))


MASTER_DRIVER = """set -uo pipefail
HERE="{scripts}"
ISL=512 GH_TOKEN=t0k
_bench_line() {{ echo "BENCH args=[$*] port=[$PORT] isl=[$ISL] token=[${{GH_TOKEN:-}}] hop=[${{GEAK_MN_HOP:-}}]"; }}
adapter_bench() {{ _bench_line "$@"; return 7; }}
adapter_health() {{ echo HEALTH; }}
source "{launcher}"
echo "SOURCED_OK"
adapter_bench one "two words"
echo "BENCH_RC=$?"
adapter_health
"""

PROF_DRIVER = """set -uo pipefail
HERE="{scripts}"
adapter_bench() {{ echo "$@" >> load.txt; sleep 5; }}
source "{launcher}"
adapter_profile_window() {{ echo WINDOW; return 5; }}
PROFILE_WARMUP_SEC=1
adapter_bench 4 2 1; echo "PROF_RC=$?"
adapter_bench() {{ echo "$@" >> load.txt; }}
source "{launcher}"
PROFILE_WARMUP_SEC=1
adapter_bench 4 2 1; echo "PROF_RC=$?"
"""


@unittest.skipIf(BASH is None, "bash is required to exercise the launcher adapter")
def run_mn(argv):
    """`mn.py <argv>` in this process, so coverage sees it; shaped like subprocess.run's result."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mn.main(list(argv))
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
    return subprocess.CompletedProcess(["mn.py", *argv], rc, out.getvalue(), err.getvalue())


COMPUTE = "Cijk_Alik_Bljk_HHS_BH_MT128x128x16"


class TopologyOfTest(unittest.TestCase):
    def test_the_pin_decides_not_the_environment(self):
        self.assertEqual(mn.topology_of(PD_STATE), "pd_disagg")

    def test_an_older_pin_without_the_field_is_tp_span(self):
        self.assertEqual(mn.topology_of({"serve_port": 30890}), "tp_span")

    def test_an_unknown_topology_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as e:
            mn.topology_of(dict(PD_STATE, topology="half_and_half"))
        self.assertIn("half_and_half", str(e.exception))


class RankHeadTest(unittest.TestCase):

    def test_the_nic_comes_from_the_pin_and_is_checked_on_the_node(self):
        head = "\n".join(mn.rank_head(PD_STATE["nodes"][1], "10.10.0.166",
                                      "/w/server.rank1.log", "/w/prof", "id1"))
        self.assertIn("export NCCL_SOCKET_IFNAME=ens50f0 GLOO_SOCKET_IFNAME=ens50f0 "
                      "MASTER_ADDR=10.10.0.166", head)
        self.assertIn("ip -4 -o addr show dev ens50f0 | grep -qF 'inet 10.10.0.183/'", head)

    def test_a_pin_without_a_nic_is_refused(self):
        with self.assertRaises(SystemExit) as e:
            mn.rank_head({"rank": 0, "ip": "10.0.0.1", "hostname": "h"}, "10.0.0.1", "/l", "/p", "x")
        self.assertIn("Re-run mn.py check", str(e.exception))


OTHER_PIN = dict(PD_STATE, nodes=[
    PD_STATE["nodes"][0],
    {"rank": 1, "node_id": "n2", "ip": "10.10.0.247",
     "hostname": "node-f2b9", "cluster_if": "ens50f0", "ngpu": 8},
])


class LaunchRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pins = os.path.join(self.tmp, "pins")
        os.makedirs(self.pins)
        self.env = {k: os.environ.get(k) for k in
                    ("GEAK_MN_STATE", "MODEL", "LOG",
                     "PROFILE_DIR", "OUT_DIR", "GEAK_MN_DRY_RUN")}
        os.environ["GEAK_MN_STATE"] = os.path.join(self.pins, "run.json")
        os.environ.pop("GEAK_MN_DRY_RUN", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_launch(self, nodes):
        with open(os.path.join(self.tmp, "mn_launch.json"), "w", encoding="utf-8") as f:
            json.dump({"launch_id": "x", "nodes": nodes}, f)

    def test_supervise_follows_the_launch_not_a_rewritten_pin(self):
        self.write_launch([{k: n.get(k) for k in mn.LAUNCH_NODE_KEYS} for n in PD_STATE["nodes"]])
        state = mn.supervised_state(OTHER_PIN, self.tmp)
        self.assertEqual([n["node_id"] for n in state["nodes"]], ["n0", "n1"])
        self.assertEqual(state["serve_port"], OTHER_PIN["serve_port"])

    def test_an_older_launch_record_falls_back_to_the_pin(self):
        with open(os.path.join(self.tmp, "mn_launch.json"), "w", encoding="utf-8") as f:
            json.dump({"launch_id": "x", "ranks": []}, f)
        self.assertIs(mn.supervised_state(OTHER_PIN, self.tmp), OTHER_PIN)
        self.assertIs(mn.supervised_state(OTHER_PIN, os.path.join(self.tmp, "nope")), OTHER_PIN)

    def test_a_launch_record_without_node_ids_is_not_trusted(self):
        self.write_launch([{"rank": 0, "hostname": "a"}, {"rank": 1, "hostname": "b"}])
        self.assertIsNone(mn.launched_nodes(self.tmp))

    def test_a_failed_launch_reaps_what_it_spawned(self):
        torn = []
        with mock.patch.object(mn, "teardown", lambda s, **k: torn.append(s) or True):
            self.assertEqual(mn.abandon_launch(PD_STATE), 1)
        self.assertEqual([n["node_id"] for n in torn[0]["nodes"]], ["n0", "n1"])



class SpanSyncEnvTest(unittest.TestCase):
    def test_tp_span_injects_the_var_when_a_sweep_dropped_it(self):
        self.assertEqual(
            mn.span_sync_env(["FOO=1"], spanning=True),
            ["FOO=1", "SYNC_TOKEN_IDS_ACROSS_TP=1"])

    def test_an_explicit_one_is_left_alone(self):
        env = ["SYNC_TOKEN_IDS_ACROSS_TP=1", "FOO=1"]
        self.assertEqual(mn.span_sync_env(env, spanning=True), env)

    def test_explicit_off_is_refused_on_a_spanning_group(self):
        with self.assertRaises(SystemExit) as caught:
            mn.span_sync_env(["SYNC_TOKEN_IDS_ACROSS_TP=0"], spanning=True)
        self.assertIn("SYNC_TOKEN_IDS_ACROSS_TP", str(caught.exception))

    def test_pd_disagg_does_not_inject_or_refuse(self):
        self.assertEqual(mn.span_sync_env(["SYNC_TOKEN_IDS_ACROSS_TP=0"], spanning=False),
                         ["SYNC_TOKEN_IDS_ACROSS_TP=0"])
        self.assertEqual(mn.span_sync_env([], spanning=False), [])


class _PinnedEnv(unittest.TestCase):
    """A temp shared mount with a pin in it; os.environ restored afterwards."""
    ENV = ("GEAK_MN_STATE", "MODEL", "LOG",
           "PROFILE_DIR", "OUT_DIR", "GEAK_MN_DRY_RUN", "TP", "BACKEND", "MAX_MODEL_LEN",
           "OVERLAY_PYTHONPATH", "EXTRA_SERVER_ARGS", "EXTRA_ENV", "GEAK_MN_ADAPTERS")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in self.ENV}
        for k in self.ENV:
            os.environ.pop(k, None)
        os.environ.update(GEAK_MN_STATE=os.path.join(self.tmp, "pin.json"),
                          MODEL="/m", LOG=os.path.join(self.tmp, "server.log"),
                          PROFILE_DIR=os.path.join(self.tmp, "prof"), OUT_DIR=self.tmp)

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pin(self, state):
        with open(os.environ["GEAK_MN_STATE"], "w", encoding="utf-8") as f:
            json.dump(dict(state, shared_fs=self.tmp, mn_dir=MN), f)

    def rendered(self, name):
        with open(os.path.join(self.tmp, f"mn_{name}.sh"), encoding="utf-8") as f:
            return f.read()

    def launch(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return mn.cmd_launch([])


class ProvidedLaunchTest(_PinnedEnv):
    def setUp(self):
        super().setUp()
        self.pin(PROVIDED_PD)
        ovl = os.path.join(self.tmp, "ovl")
        os.makedirs(ovl)
        with open(os.path.join(ovl, "k.py"), "w", encoding="utf-8") as f:
            f.write("X = 1\n")
        os.environ.update(GEAK_MN_DRY_RUN="1", BACKEND="sglang", MAX_MODEL_LEN="8192",
                          OVERLAY_PYTHONPATH=ovl, EXTRA_ENV="SGLANG_X=2",
                          EXTRA_SERVER_ARGS="--chunked-prefill-size 4096 --port 9 --tp-size 4",
                          GEAK_MN_ADAPTERS=os.path.join(self.tmp, "adapters"))

    def test_each_rank_runs_its_command_under_geaks_layer(self):
        self.assertEqual(self.launch(), 0)
        magpie = os.path.join(os.path.realpath(self.tmp), "adapters", "launchers", "magpie.sh")
        for rank in (0, 1):
            script = self.rendered(f"rank{rank}")
            wrapper = os.path.join(self.tmp, f"mn_cmd_rank{rank}.sh")
            self.assertIn(f"MAGPIE_LAUNCH_SCRIPT={shlex.quote(wrapper)}", script)
            self.assertIn(f"OUT_DIR={os.path.join(self.tmp, f'rank{rank}')}", script)
            self.assertIn(f"GEAK_MN_MAGPIE={shlex.quote(magpie)}", script)
            self.assertIn("export EXTRA_SERVER_ARGS='--chunked-prefill-size 4096'", script,
                          "the command owns --port and --tp-size; GEAK's copies are dropped")
            self.assertIn("export EFFECTIVE_SERVER_ARGS_COMPLETE=1", script)
            self.assertIn("export PORT=8000", script)
            self.assertIn("export MAX_MODEL_LEN=8192", script)
            self.assertNotIn(MN, script, "the node runs no GEAK file")
            self.assertIn("adapter_launch && mkdir -p /tmp/geak_mn && echo \"$SERVER_PID", script)
            self.assertIn(f"> /tmp/geak_mn/rank{rank}.pid' >>", script)
            self.assertNotIn("adapter_pd_args", script, "structural flags are the caller's")

    def test_a_rank_with_gpu_ids_serves_from_those_cards(self):
        nodes = [dict(PROVIDED_PD["nodes"][0], tp=4, gpus=4, gpu_ids=[0, 1, 2, 3]),
                 dict(PROVIDED_PD["nodes"][1], tp=4, gpus=4, gpu_ids=[4, 5, 6, 7])]
        self.pin(dict(PROVIDED_PD, nodes=nodes))
        self.assertEqual(self.launch(), 0)
        self.assertIn("export GPU=0,1,2,3\n", self.rendered("rank0"))
        self.assertIn("export GPU=4,5,6,7\n", self.rendered("rank1"))

    def test_without_gpu_ids_a_rank_serves_from_its_first_cards(self):
        self.assertEqual(self.launch(), 0)
        self.assertIn("export GPU=0,1,2,3,4,5,6,7\n", self.rendered("rank1"))

    def test_the_wrapper_is_the_command_with_geaks_flags_last(self):
        self.assertEqual(self.launch(), 0)
        with open(os.path.join(self.tmp, "mn_cmd_rank1.sh"), encoding="utf-8") as f:
            wrapper = f.read()
        argv = mn.split_command(sg_cmd("decode"))[1]
        self.assertIn(f'setsid env -- {shlex.join(argv)} ${{prof[@]+"${{prof[@]}}"}} $EXTRA_SGLANG_ARGS '
                      '> "$SERVER_LOG"', wrapper)
        self.assertNotIn("SGLANG_X=1", wrapper, "GEAK's EXTRA_ENV sets SGLANG_X; the command's value yields")
        self.assertIn('echo $! > "$MAGPIE_SERVER_PID_FILE"', wrapper)

    def test_the_wrapper_layers_env_and_flags_when_run(self):
        wrapper = os.path.join(self.tmp, "w.sh")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(mn.command_script("sglang", "A=1 B=2 sh -c 'echo \"$A $B $*\"' sh --port 8000",
                                      ["B=9"], self.tmp))
        log, pidf = os.path.join(self.tmp, "srv.log"), os.path.join(self.tmp, "pid")
        env = dict(os.environ, B="9", EXTRA_SGLANG_ARGS="--x 1", SERVER_LOG=log, MAGPIE_SERVER_PID_FILE=pidf)
        self.assertEqual(subprocess.run(["bash", wrapper], env=env).returncode, 0)
        for _ in range(50):
            if os.path.exists(log) and os.path.getsize(log):
                break
            time.sleep(0.1)
        with open(log, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "1 9 --port 8000 --x 1")
        self.assertTrue(open(pidf, encoding="utf-8").read().strip().isdigit())

    def test_the_router_command_runs_as_given_and_records_its_pid(self):
        self.assertEqual(self.launch(), 0)
        lb = self.rendered("lb")
        self.assertIn(f"setsid bash -c {shlex.quote(PROVIDED_ROUTER['launch_command'])} >>", lb)
        self.assertIn('pid=$!\nmkdir -p /tmp/geak_mn && echo "$pid ', lb)
        self.assertIn("> /tmp/geak_mn/ranklb.pid", lb)

    def test_the_launch_record_names_the_router_machine(self):
        os.environ.pop("GEAK_MN_DRY_RUN")
        ran = []

        def ssh_run(state, node, script, timeout=300, merged=False):
            ran.append(node["rank"])
            return {"hostname": node["hostname"], "ip": node["ip"], "rc": 0, "out": "STARTED_PID=1", "err": ""}

        with mock.patch.object(mn, "teardown", lambda s: [{}]), \
                mock.patch.object(mn, "resolve", mn.state_targets), \
                mock.patch.object(mn, "ssh_run", ssh_run), \
                mock.patch.object(mn, "ssh_each", lambda s, tasks, merged=False: [ssh_run(s, *t) for t in tasks]), \
                mock.patch.object(mn, "serve_counts", lambda s: [1, 1, 1]):
            self.assertEqual(self.launch(), 0)
        self.assertEqual(ran, [0, 1, "lb"], "engines first, then the router on its own machine")
        with open(os.path.join(self.tmp, "mn_launch.json"), encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["router"]["rank"], "lb")
        self.assertEqual((doc["pd"]["lb_host"], doc["pd"]["lb_url"]), ("10.10.0.32", "http://10.10.0.32:8000"))
        self.assertEqual([(e["role"], e["replica"], e["url"]) for e in doc["pd"]["engines"]],
                         [("prefill", 0, "http://10.10.0.166:8000"), ("decode", 0, "http://10.10.0.183:8000")])
        self.assertEqual(doc["ranks"][0]["extra_args_dropped"], ["--port 9", "--tp-size 4"])
        self.assertEqual([e.get("bootstrap_port") for e in doc["pd"]["engines"]], [8998, None])
        self.assertFalse({"role_ports", "bootstrap_port", "transfer_backend"} & set(doc["pd"]), "the commands own them")
        self.assertEqual(doc["pd"]["lb_url"], "http://10.10.0.32:8000")


@unittest.skipUnless(HAVE_SH, "needs bash")
class SingleNodeGeneratedLaunchTest(_PinnedEnv):
    def setUp(self):
        super().setUp()
        here = adapter_root(self.tmp, "fake", FAKE_ADAPTER)
        here.start()
        self.addCleanup(here.stop)
        self.pin({"topology": "tp_span", "mode": "generated", "master_ip": "10.10.0.166",
                  "serve_port": 30890, "dist_init_port": 30531,
                  "nodes": [dict(PD_STATE["nodes"][0], tp=8, gpus=8)]})
        os.environ.update(GEAK_MN_DRY_RUN="1", TP="8", BACKEND="fake")

    def test_one_machine_is_not_a_spanning_group(self):
        self.assertEqual(self.launch(), 0)
        script = self.rendered("rank0")
        self.assertNotIn("SYNC_TOKEN_IDS_ACROSS_TP", script)
        self.assertIn("adapter_dist_args 0 1", script)

    def test_the_rank_records_its_server_pid_on_its_node(self):
        self.assertEqual(self.launch(), 0)
        self.assertIn('adapter_launch\nmkdir -p /tmp/geak_mn && echo "$SERVER_PID ', self.rendered("rank0"))
        self.assertIn("> /tmp/geak_mn/rank0.pid\necho STARTED_PID=$SERVER_PID", self.rendered("rank0"))


class PidFileTest(unittest.TestCase):
    def test_an_unanswered_ssh_is_not_a_substituted_machine(self):
        state = {"nodes": [{"rank": 0, "node_id": "h/b", "hostname": "h", "ip": "10.0.0.1"}], "ssh": {}}
        with fake_ssh(lambda task, node, args: {"rc": 255, "out": "", "err": "Connection refused"}):
            with self.assertRaisesRegex(RuntimeError, "did not answer over ssh .*Connection refused"):
                mn.resolve(state)
        with fake_ssh(lambda task, node, args: {"rc": 0, "out": "other/b", "err": ""}):
            with self.assertRaisesRegex(SystemExit, "Refusing to substitute"):
                mn.resolve(state)

    @unittest.skipUnless(HAVE_SH and sys.platform.startswith("linux"), "bash + /proc")
    def test_a_recorded_pid_is_alive_only_while_that_very_process_runs(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with mock.patch.object(mn, "PID_DIR", tmp):
            run = lambda script: subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout  # noqa: E731
            self.assertIn("serve=0 pid=none", run(mn.serve_alive(0).replace(mn.SERVE_PATTERN, "no-such-proc-[x]yz")))
            sleeper = subprocess.Popen(["sleep", "30"])
            try:
                run(mn.record_pid(0, str(sleeper.pid)))
                self.assertIn(f"serve=1 pid={sleeper.pid}", run(mn.serve_alive(0)))
            finally:
                sleeper.kill()
                sleeper.wait()
            self.assertIn("serve=0 pid=gone", run(mn.serve_alive(0)))
            with open(os.path.join(tmp, "rank0.pid"), "w") as f:
                f.write(f"{os.getpid()} 1\n")   # a live pid, but not the process that was recorded
            self.assertIn("serve=0 pid=gone", run(mn.serve_alive(0)))


def completed(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


SHOWUSE = "GPU[0]\t\t: GPU use (%): 3\nGPU[1]\t\t: GPU use (%): 40\n"
MEMINFO = ("device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
           "card0,206158430208,1073741824\ncard1,206158430208,2147483648\n")
IP_ADDR = ("1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
           "2: ens51np0    inet 10.10.0.166/24 brd 10.10.0.255 scope global ens51np0\n")


class NodeProbeTest(unittest.TestCase):
    """mn.py probe, as it runs on a node: every number comes from rocm-smi, ss, pgrep and ip."""

    def setUp(self):
        mn.local_addrs.cache_clear()
        self.addCleanup(mn.local_addrs.cache_clear)

    def fake_run(self, argv, **_kw):
        if argv[0] == "ip":
            return completed(IP_ADDR)
        cmd = argv[-1]
        for needle, out in (("--showuse", SHOWUSE), ("--showmeminfo", MEMINFO),
                            ("--showpids", SHOWPIDS_REAL_CO_TENANT), ("sport = :30000", "LISTEN 0 4096 *:30000\n"),
                            ("sport = :", ""), ("wc -l", "2\n"), ("pgrep -c", "1\n")):
            if needle in cmd:
                return completed(out)
        raise AssertionError(f"unexpected command {cmd!r}")

    def test_the_probe_reports_the_tightest_card_busy_ports_and_our_processes(self):
        with mock.patch.object(mn.subprocess, "run", self.fake_run), \
                mock.patch.dict(os.environ, {"MN_GPU_IDENTITY": 'noise\n{"gfx": "gfx942"}'}):
            p = mn.probe_node([30000, 30001])
        self.assertEqual((p["ngpu"], p["max_use"]), (2, 40.0))
        self.assertEqual(p["card_total_mib"], [196608, 196608])
        self.assertEqual(p["free_vram_mib"], 196608 - 2048)
        self.assertEqual(p["busy_ports"], [30000])
        self.assertEqual((p["gpu_pids"], p["parked_gpu_pids"]), (1, 1))
        self.assertEqual((p["serve_procs"], p["worker_procs"]), (1, 2))
        self.assertEqual(p["addrs"], [["lo", "127.0.0.1", 8], ["ens51np0", "10.10.0.166", 24]])
        self.assertEqual(p["gpu"], {"gfx": "gfx942"})
        self.assertIsInstance(p["sw"], dict)

    def test_a_machine_without_ip_has_no_addresses(self):
        with mock.patch.object(mn.subprocess, "run", side_effect=OSError("no ip")):
            self.assertEqual(mn.local_addrs(), ())

    def test_a_silent_identity_probe_is_an_error_not_an_identity(self):
        with mock.patch.dict(os.environ, {"MN_GPU_IDENTITY": ""}):
            self.assertIn("printed nothing", mn.gpu_identity()["error"])

    def test_visible_gpus_counts_gfx_agents_and_is_zero_without_rocm(self):
        agents = "  Name:                    gfx942\n  Name:                    gfx942\n  Name:   AMD EPYC\n"
        with mock.patch.object(mn.subprocess, "run", return_value=completed(agents)):
            self.assertEqual(mn.visible_gpus(), 2)
        with mock.patch.object(mn.subprocess, "run", side_effect=OSError("no rocminfo")):
            self.assertEqual(mn.visible_gpus(), 0)


class SharedFsProofTest(unittest.TestCase):
    def test_a_node_that_cannot_read_or_write_the_mount_breaks_the_proof(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        nodes = [{"rank": 0, "hostname": "node-a"}, {"rank": 1, "hostname": "node-b"}]

        def ssh_run(_state, node, script, _timeout):
            token = re.search(r"from_bench_(\w+)", script).group(1)
            if node["rank"] == 0:
                with open(os.path.join(tmp, ".mn_probe", f"from_rank0_{token}"), "w") as f:
                    f.write(token)
                return {"out": f"READ_OK={token}\nWRITE_OK"}
            return {"out": "READ_FAIL\nWRITE_FAIL"}

        with mock.patch.object(mn, "ssh_run", ssh_run):
            res = mn.prove_shared({"shared_fs": tmp}, nodes)
        self.assertFalse(res["shared"])
        self.assertEqual([n["ok"] for n in res["per_node"]], [True, False])
        self.assertEqual(os.listdir(os.path.join(tmp, ".mn_probe")), [], "markers are cleaned up")


class ClientPreflightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.makedirs(os.path.join(self.tmp, "utils", "bench_serving"))
        open(os.path.join(self.tmp, "utils", "bench_serving", "benchmark_serving.py"), "w").close()
        env = mock.patch.dict(os.environ, {"INFERENCEX_PATH": self.tmp, "PYTHON_BIN": "py-x",
                                           "INFERENCEX_BENCH_SERVING": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_a_python_that_does_not_start_is_named(self):
        with mock.patch.object(mn.subprocess, "run", side_effect=OSError("not found")):
            self.assertIn("py-x does not start here", mn.inferencex_client_error("m"))

    def test_missing_packages_are_named(self):
        with mock.patch.object(mn.subprocess, "run", return_value=completed(rc=1, stderr="No module named aiohttp")):
            self.assertIn("No module named aiohttp", mn.inferencex_client_error("m"))

    def test_a_model_path_this_machine_cannot_read_is_named(self):
        with mock.patch.object(mn.subprocess, "run", return_value=completed()):
            self.assertIn("no tokenizer_config.json", mn.inferencex_client_error("/no/such/model"))
            self.assertEqual(mn.inferencex_client_error("org/model"), "")

    def test_a_native_client_for_an_unknown_backend_is_refused(self):
        problems = mn.client_problems({"bench_client": "native", "nodes": [{}]}, {"backend": "trt"})
        self.assertIn("no native client known for backend 'trt'", problems[0])


class ReachTest(unittest.TestCase):
    def test_a_port_that_cannot_be_opened_is_a_problem(self):
        node = {"rank": 0, "hostname": "node0", "ip": "10.0.0.1"}
        with mock.patch.object(mn, "ssh_run", lambda *a: {"out": "nothing listening"}):
            problems = mn.reach_problems({}, [(node, 30531, [None], "dist-init")])
        self.assertIn("could not listen on the dist-init port 30531", problems[0])

    def test_dials_here_tells_an_open_port_from_a_closed_one(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        self.assertTrue(mn.dials_here("127.0.0.1", port))
        srv.close()
        self.assertFalse(mn.dials_here("127.0.0.1", port))

