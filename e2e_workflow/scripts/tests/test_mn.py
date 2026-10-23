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

    def test_the_stack_and_the_gpu_are_pinned(self):
        gpu = {"gfx": "gfx942", "target": "unknown", "marketing_name": "AMD Instinct MI308X", "physical_cu_count": 80}
        self.probes = [dict(p, sw={"sglang": "0.5.1"}, gpu=gpu) for p in self.probes]
        rc, err = self.check()
        self.assertEqual(rc, 0, err)
        self.assertEqual((self.pin()["sw"], self.pin()["gpu"]), ({"sglang": "0.5.1"}, gpu))

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


class ActivePinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pointer = os.path.join(self.tmp, "state_path")
        self.patch = mock.patch.object(mn, "ACTIVE_PIN", self.pointer)
        self.patch.start()
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        os.environ.pop("GEAK_MN_STATE", None)

    def tearDown(self):
        self.env.stop()
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def claim(self, pid, start):
        with open(self.pointer, "w", encoding="utf-8") as f:
            json.dump({"state": "/w/mn_state.json", "pid": pid, "start": start, "mn_py": "/w/mn.py"}, f)

    def test_the_pin_counts_while_its_owner_lives(self):
        self.claim(4242, "777")
        with mock.patch.object(mn, "_proc_start", lambda pid: "777" if int(pid) == 4242 else ""):
            self.assertEqual(mn.active_pin(), "/w/mn_state.json")

    def test_a_dead_or_reused_owner_pid_releases_the_machine(self):
        self.claim(4242, "777")
        with mock.patch.object(mn, "_proc_start", lambda pid: ""):
            self.assertEqual(mn.active_pin(), "")
        with mock.patch.object(mn, "_proc_start", lambda pid: "999"):
            self.assertEqual(mn.active_pin(), "", "the pid now belongs to another process")

    def test_geak_mn_state_wins_and_no_pointer_is_none(self):
        self.assertEqual(mn.active_pin(), "")
        os.environ["GEAK_MN_STATE"] = "/x/pin.json"
        self.assertEqual(mn.active_pin(), "/x/pin.json")

    def test_only_the_owner_releases_it(self):
        self.claim(os.getppid() + 1, "1")
        mn.cmd_active_pin(["--release"])
        self.assertTrue(os.path.exists(self.pointer))
        self.claim(os.getppid(), "1")
        mn.cmd_active_pin(["--release"])
        self.assertFalse(os.path.exists(self.pointer))

    def ancestors(self, procs):
        """claude_ancestor over a fake /proc: {pid: (comm, ppid, argv)}, starting at the parent 300."""
        def fake_open(path, mode="r", **kw):
            pid, name = int(path.split("/")[2]), path.split("/")[3]
            if pid not in procs:
                raise FileNotFoundError(path)
            comm, ppid, argv = procs[pid]
            if name == "stat":
                return io.StringIO(f"{pid} ({comm}) S {ppid} 1 1")
            return io.BytesIO("\0".join(argv).encode() + b"\0")
        with mock.patch.object(mn.os, "getppid", lambda: 300), mock.patch("builtins.open", fake_open):
            return mn.claude_ancestor()

    def test_claude_ancestor_is_the_nearest_claude_code_process(self):
        shell = (("bash", 200, ["/bin/bash", "-c", "source snap; mn.py check"]))
        self.assertEqual(self.ancestors({300: shell, 200: ("claude", 100, ["claude", "-p"]),
                                         100: ("python3", 1, ["python3", "run_e2e.py"])}), 200)
        self.assertEqual(self.ancestors({300: shell, 200: ("node", 1, [
            "node", "/usr/lib/node_modules/@anthropic-ai/claude-code/cli.js"])}), 200, "an npm install runs as node")
        self.assertEqual(self.ancestors({300: shell, 200: ("python3", 1, ["python3", "run_e2e.py"])}), 0)
        self.assertEqual(self.ancestors({300: shell}), 0, "an unreadable ancestor ends the walk")

    def test_the_lease_dir_is_the_one_gpu_lock_and_the_leases_share(self):
        pin, leases = os.path.join(self.tmp, "mn_state.json"), os.path.join(self.tmp, ".mn_lease")
        with open(pin, "w", encoding="utf-8") as f:
            json.dump({"shared_fs": self.tmp, "nodes": [], "tools_dir": "/w/run/.mn_tools"}, f)
        os.environ["GEAK_MN_STATE"] = pin
        with contextlib.redirect_stdout(io.StringIO()) as out:
            mn.cmd_active_pin(["--lease-dir"])
        self.assertEqual(out.getvalue().strip(), f"{pin}\t{leases}\t/w/run/.mn_tools")
        node = {"hostname": "node-fc54.example"}
        self.assertEqual(mn.lease_file({"shared_fs": self.tmp}, node, 3), os.path.join(leases, "node-fc54.gpu3.lock"))
        gpu_lock = open(os.path.join(SCRIPTS_DIR, os.pardir, os.pardir, "kernel_workflow", "scripts",
                                     "gpu_lock.sh"), encoding="utf-8").read()
        self.assertIn('LOCK_DIR="$_MN_LEASE_DIR"', gpu_lock)
        self.assertIn('LOCK_PREFIX="${_host%%.*}.gpu"', gpu_lock)
        self.assertIn('"${LOCK_DIR}/${LOCK_PREFIX}${GPU_ID}.lock"', gpu_lock)


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
    def test_the_sglang_client_runs_here_from_the_root_directory(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        path = mn.resolve_adapter("sglang", root)
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(
                ["bash", "-c", f". {shlex.quote(path)}; python() {{ echo \"LOCAL $PWD\" \"$@\"; }}; adapter_bench 4 2"],
                capture_output=True, text=True, check=True, cwd=d,
                env={"PATH": os.environ["PATH"], "MODEL": "/m"}).stdout
        self.assertRegex(out, r"^LOCAL / -m sglang\.bench_serving ")
        self.assertIn("--num-prompts 4 --max-concurrency 2", out)

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
class MultinodeDriverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_master_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "mn_state.json")
        self.pin(mn.local_identity() or "n0")

    def pin(self, master_id):
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump({"nodes": [{"rank": 0, "node_id": master_id, "ip": "10.0.0.1", "hostname": "h0", "ngpu": 1},
                                 {"rank": 1, "node_id": "n1", "ip": "10.0.0.2", "hostname": "h1", "ngpu": 1}],
                       "master_ip": "10.0.0.1", "serve_port": 30890, "shared_fs": self.tmp, "mn_dir": MN}, f)

    def _source(self, **env_over):
        driver = os.path.join(self.tmp, "driver.sh")
        with open(driver, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(MASTER_DRIVER.format(scripts=SCRIPTS_DIR.replace("\\", "/"),
                                              launcher=LAUNCHER.replace("\\", "/")))
        env = {k: v for k, v in os.environ.items() if k not in ("TP", "BENCH_CLIENT")}
        env.update({"GEAK_MN_STATE": self.state, "TP": "1", "BACKEND": "fake", **env_over})
        return subprocess.run([BASH, driver], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=120)

    def test_a_client_adapter_runs_here_with_the_legs_variables(self):
        proc = self._source(BENCH_CLIENT="inferencex")
        self.assertIn("SOURCED_OK", proc.stdout, proc.stderr)
        self.assertIn("BENCH args=[one two words] port=[30890] isl=[512] token=[t0k] hop=[]", proc.stdout)
        self.assertIn("BENCH_RC=7", proc.stdout)
        self.assertNotIn("runs on rank0", proc.stderr)

    def test_a_profile_round_is_a_fresh_load_and_the_launchers_window(self):
        driver = os.path.join(self.tmp, "prof.sh")
        with open(driver, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(PROF_DRIVER.format(scripts=SCRIPTS_DIR.replace("\\", "/"),
                                            launcher=LAUNCHER.replace("\\", "/")))
        env = {k: v for k, v in os.environ.items() if k not in ("TP", "BENCH_CLIENT")}
        env.update({"GEAK_MN_STATE": self.state, "TP": "1", "BACKEND": "fake", "BENCH_CLIENT": "inferencex"})
        proc = subprocess.run([BASH, driver], env=env, cwd=self.tmp, capture_output=True, text=True, timeout=120)
        self.assertIn("WINDOW\nPROF_RC=5", proc.stdout, proc.stderr)
        self.assertIn("PROF_RC=1", proc.stdout)
        self.assertIn("the profiling load ended before the window opened", proc.stderr)
        with open(os.path.join(self.tmp, "load.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read().split(), ["4", "2", "0", "4", "2", "0"])

    @unittest.skipUnless(mn.local_identity(), "rank0 must be this machine (a Linux boot_id)")
    def test_the_native_bench_runs_on_rank0_with_the_legs_variables_but_not_its_secrets(self):
        proc = self._source(BENCH_CLIENT="native")
        self.assertIn("SOURCED_OK", proc.stdout, proc.stderr)
        self.assertIn("BENCH_CLIENT=native runs on rank0", proc.stderr)
        self.assertIn("BENCH args=[one two words] port=[30890] isl=[512] token=[] hop=[1]", proc.stdout)
        self.assertIn("BENCH_RC=7", proc.stdout)
        self.assertIn("HEALTH", proc.stdout)

    def test_the_ranks_source_the_adapters_beside_the_launcher(self):
        driver = os.path.join(self.tmp, "adapters_dir.sh")
        with open(driver, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(MASTER_DRIVER.format(scripts=SCRIPTS_DIR.replace("\\", "/"),
                                              launcher=LAUNCHER.replace("\\", "/"))
                         + 'echo "ADAPTERS=$GEAK_MN_ADAPTERS"\n')
        env = dict(os.environ, GEAK_MN_STATE=self.state, TP="1", BACKEND="fake")
        proc = subprocess.run([BASH, driver], env=env, cwd=self.tmp, capture_output=True, text=True, timeout=120)
        got = re.search(r"^ADAPTERS=(.*)$", proc.stdout, re.M)
        self.assertTrue(got, proc.stderr)
        self.assertTrue(os.path.samefile(got.group(1), os.path.join(SCRIPTS_DIR, "adapters")), got.group(1))

    def test_a_pin_without_mn_dir_is_refused_before_anything_runs(self):
        with open(self.state, encoding="utf-8") as f:
            doc = json.load(f)
        doc.pop("mn_dir")
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        proc = self._source()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("names no mn_dir", proc.stderr)

    def test_a_one_card_leg_is_served_remotely_too(self):
        proc = self._source(GEAK_MN_STATE=os.path.join(self.tmp, "no_such_pin.json"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("SOURCED_OK", proc.stdout)
        self.assertIn("no pinned node set", proc.stderr)


class DriverSubprocessTest(unittest.TestCase):
    """The workflow's own subprocesses (best-effort file/JSON helpers) and the caller's mn.py check."""

    def test_the_workflow_starts_only_host_side_helpers(self):
        with open(os.path.join(SCRIPTS_DIR, "..", "e2e_workflow.js"), encoding="utf-8") as f:
            src = f.read()
        self.assertEqual(src.count("require('child_process')"), 5)
        firsts = re.findall(r"(?:execFileSync|spawn)\('python3', \[\s*(?:'-B',\s*)?(`[^`]*?`|'[^']*'|MN_PY)", src)
        self.assertEqual(len(firsts), 5, firsts)
        for first in firsts:
            self.assertTrue(first == "'-c'"
                            or first.endswith(("/geak_trace_collector.py`", "/capture_shapes.py`")), first)
        self.assertIn("const MN_PY = `${WORKFLOW_DIR}/scripts/multi_node/mn.py`;", src)
        self.assertNotIn("runMn", src)
        self.assertIn("let verdict = A.multinode_check || await agentMnCheck();", src)
        self.assertIn("python3 ${MN_PY} check --owner claude", src)

    def test_run_e2e_checks_the_machines_before_the_workflow(self):
        # A Workflow script cannot spawn a process, so the check runs in run_e2e.py.
        with open(os.path.join(SCRIPTS_DIR, "..", "..", "interface", "run_e2e.py"), encoding="utf-8") as f:
            src = f.read()
        call = src.index("check = _multinode_check(ps_args)")
        # A resume that carries the block only in state is checked too, as the workflow serves on it.
        self.assertIn("if _multinode_block(ps_args):\n        check = _multinode_check(ps_args)", src)
        self.assertIn('(ps_args.get("state") or {}).get("multinode")', src)
        self.assertLess(call, src.index("wf = invoke_workflow(prompt"))
        self.assertIn('ps_args["multinode_check"] = check', src[call:])
        self.assertIn("prompt = build_prompt(ps_args)", src[call:src.index("wf = invoke_workflow(prompt")])


@unittest.skipIf(BASH is None, "bash is required to run bench_e2e.sh")
class BenchPinGuardTest(unittest.TestCase):
    def bench(self, **over):
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODEL", "REUSE_SERVER", "BENCH_LAUNCHER", "TP", "GPU", "GEAK_MN_STATE", "GEAK_MN_HOP")}
        env.update(GEAK_MN_STATE="/nonexistent/pin.json", **over)
        return subprocess.run([BASH, os.path.join(SCRIPTS_DIR, "bench_e2e.sh")], env=env,
                              capture_output=True, text=True, timeout=120)

    def test_the_driver_serves_nothing_itself_while_a_pin_is_active(self):
        for tp, gpu in (("8", "0,1,2,3,4,5,6,7"), ("1", "3")):
            proc = self.bench(BENCH_LAUNCHER="native", TP=tp, GPU=gpu)
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("drives its nodes from this machine", proc.stderr)
            self.assertIn("exec --env --gpu", proc.stderr)

    def test_on_a_node_a_one_card_isolated_server_is_allowed(self):
        proc = self.bench(BENCH_LAUNCHER="native", TP="1", GPU="3", GEAK_MN_HOP="1")
        self.assertNotIn("On a pinned node", proc.stderr)
        self.assertNotIn("drives its nodes", proc.stderr)
        proc = self.bench(TP="1", GPU="3", GEAK_MN_HOP="1")
        self.assertNotIn("On a pinned node", proc.stderr, "no launcher on a node is the native one")

    def test_on_a_node_nothing_bigger_starts(self):
        for over in ({"TP": "8", "GPU": "0,1,2,3,4,5,6,7"}, {"BENCH_LAUNCHER": "multinode", "TP": "1", "GPU": "3"}):
            proc = self.bench(GEAK_MN_HOP="1", **over)
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("On a pinned node", proc.stderr)

    def test_no_launcher_goes_to_the_pin(self):
        proc = self.bench()
        self.assertIn("no pinned node set at '/nonexistent/pin.json'", proc.stderr)

    def test_reusing_a_server_is_allowed(self):
        proc = self.bench(BENCH_LAUNCHER="native", TP="8", GPU="0,1", REUSE_SERVER="1")
        self.assertNotIn("drives its nodes", proc.stderr)


@unittest.skipIf(BASH is None or not mn.local_identity(), "bash and a Linux boot_id")
class MultinodeProfileWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_window_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.profile = os.path.join(self.tmp, "profile")
        os.makedirs(self.profile)
        self.state = os.path.join(self.tmp, "mn_state.json")
        with open(self.state, "w", encoding="utf-8") as handle:
            json.dump({
                "nodes": [
                    {"rank": 0, "node_id": mn.local_identity() or "n0", "ip": "10.10.0.32",
                     "hostname": "h0", "cluster_if": "ens50f0", "ngpu": 8},
                    {"rank": 1, "node_id": "n1", "ip": "10.10.0.166",
                     "hostname": "h1", "cluster_if": "ens50f0", "ngpu": 8},
                ],
                "master_ip": "10.10.0.32", "serve_port": 30890,
                "dist_init_port": 30531,
                "shared_fs": self.tmp, "mn_dir": MN,
            }, handle)

    def _run(self, *, traces=16, expect=16, stub=STUB_NATIVE, timeout=120,
             quiet_s=5, settle_s=5, base_url="http://127.0.0.1:1"):
        # The trace wait is constants in the launcher; a copy with test-sized ones.
        with open(LAUNCHER, encoding="utf-8") as handle:
            text = handle.read()
        for name, value in (("_MN_TRACE_SETTLE_S", settle_s), ("_MN_TRACE_QUIET_S", quiet_s),
                            ("_MN_TRACE_TIMEOUT_S", timeout)):
            text, n = re.subn(rf"^{name}=\d+$", f"{name}={value}", text, flags=re.M)
            self.assertEqual(n, 1, name)
        launcher = os.path.join(self.tmp, "multinode.sh")
        with open(launcher, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        driver = os.path.join(self.tmp, "driver.sh")
        with open(driver, "w", encoding="utf-8") as handle:
            handle.write(DRIVER.format(
                scripts=SCRIPTS_DIR.replace("\\", "/"),
                launcher=launcher.replace("\\", "/"),
                stub=stub.format(count=traces, collective=COLLECTIVE)
                if stub else ""))
        env = dict(os.environ)
        env.update(
            GEAK_MN_STATE=self.state,
            PROFILE_DIR=self.profile,
            PROFILE_NUM_STEPS="40",
            TP=str(expect),
            BASE_URL=base_url,
            # The launcher copy has no adapters/ beside it, so no client check to pass.
            BACKEND="fake",
        )
        proc = subprocess.run([BASH, driver], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=300)
        report_path = os.path.join(self.profile, "ranks", "mn_trace_report.json")
        report = {}
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        return proc, report

    def _rc(self, proc):
        for line in proc.stdout.splitlines():
            if line.startswith("WINDOW_RC="):
                return int(line.split("=", 1)[1])
        self.fail(f"the driver never reported a return code:\n{proc.stdout}\n{proc.stderr}")

    def test_complete_set_passes_and_is_gated(self):
        proc, report = self._run(traces=16, expect=16)
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertEqual(report.get("verdict"), "pass", report)
        self.assertIn("16/16 rank trace(s) landed", proc.stdout)
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))

    def test_incomplete_set_is_reported_but_does_not_fail_the_round(self):
        proc, report = self._run(traces=8, expect=16, quiet_s=5, timeout=40)
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertEqual(report.get("verdict"), "fail", report)
        self.assertIn("did NOT pass the gate", proc.stderr)
        self.assertIn("8/16 rank trace(s) landed", proc.stdout)

    def test_no_traces_at_all_fails_the_window(self):
        proc, _ = self._run(traces=0, expect=16, quiet_s=5, timeout=30)
        self.assertEqual(self._rc(proc), 1, proc.stdout + proc.stderr)

    def test_waits_for_streaming_writers_to_finish(self):
        import time
        start = time.time()
        proc, report = self._run(traces=16, expect=16, stub=STUB_STREAMING,
                                 settle_s=5, quiet_s=60, timeout=300)
        elapsed = time.time() - start
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertGreater(elapsed, 8, "returned before the writers finished")
        self.assertEqual(report.get("verdict"), "pass", report.get("problems"))
        self.assertTrue(all(r["readable"] for r in report["per_rank"]),
                        [r for r in report["per_rank"] if not r["readable"]])
        self.assertTrue(all(r["kernel_events"] > 0 for r in report["per_rank"]))

    def test_empty_files_do_not_count_as_landed_traces(self):
        stub = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" <<'PY'
import sys
for tp in range(16):
    open(f"{{sys.argv[1]}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz", "wb").close()
PY
  return 0
}}
"""
        proc, _ = self._run(traces=0, expect=16, stub=stub, quiet_s=5,
                            settle_s=5, timeout=40)
        self.assertEqual(self._rc(proc), 1, proc.stdout + proc.stderr)
        self.assertIn("0/16 rank trace(s) landed", proc.stdout)

    def test_stuck_export_stops_waiting_instead_of_burning_the_timeout(self):
        import time
        start = time.time()
        proc, _ = self._run(traces=8, expect=16, quiet_s=5, timeout=600)
        elapsed = time.time() - start
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertLess(elapsed, 120, f"waited {elapsed:.0f}s; the quiet-window "
                                      "check did not end the wait")
        self.assertIn("no new bytes", proc.stderr)



GATE = os.path.join(SCRIPTS_DIR, "multi_node", "mn.py")

COMPUTE = "Cijk_Alik_Bljk_HHS_BH_MT128x128x16"


def write_trace(directory, tp_rank, dp_rank, *, collectives=2, kernels=5,
                truncate=False):
    events = []
    for i in range(kernels):
        events.append({"cat": "kernel", "name": COMPUTE, "ts": 1000 + i,
                       "dur": 42, "pid": tp_rank, "tid": 1})
    for i in range(collectives):
        events.append({"cat": "kernel", "name": COLLECTIVE, "ts": 2000 + i,
                       "dur": 137, "pid": tp_rank, "tid": 2})
    payload = json.dumps({"schemaVersion": 1, "traceEvents": events})
    path = os.path.join(
        directory, f"prof-2026-09-15-TP-{tp_rank}-DP-{dp_rank}.trace.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    if truncate:
        with open(path, "rb") as handle:
            raw = handle.read()
        with open(path, "wb") as handle:
            handle.write(raw[: len(raw) // 2])
    return path


def full_set(directory, ranks=range(16), gpus_per_node=8, **kwargs):
    for rank in ranks:
        write_trace(directory, rank, rank // gpus_per_node, **kwargs)


def run_gate(directory, *args):
    if "--nodes" not in args:
        args = ("--nodes", "2", *args)
    proc = subprocess.run(
        [sys.executable, GATE, "trace-gate", "--profile-dir", directory, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    report = {}
    for candidate in (os.path.join(directory, "ranks", "mn_trace_report.json"),
                      os.path.join(directory, "mn_trace_report.json")):
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                report = json.load(handle)
            break
    return proc, report


class TraceGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_gate_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def test_complete_set_passes_and_reports_both_nodes(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["verdict"], "pass", report["problems"])
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertEqual(report["nodes_represented"], [0, 1])
        self.assertEqual(len(report["ranks_with_collectives"]), 16)

    def test_the_report_names_the_selected_rank_and_moves_nothing(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))
        self.assertEqual(len(glob.glob(os.path.join(self.tmp, "*.json.gz"))), 16)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "ranks")), ["mn_trace_report.json"])

    def test_the_report_does_not_shadow_the_trace_it_approves(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(
            os.path.isfile(os.path.join(self.tmp, "mn_trace_report.json")),
            "the report is in the directory the parser globs")
        self.assertTrue(os.path.isfile(
            os.path.join(self.tmp, "ranks", "mn_trace_report.json")))

    def test_a_newer_stray_json_does_not_change_the_parsed_trace(self):
        full_set(self.tmp)
        with open(os.path.join(self.tmp, "zz_someone_elses_notes.json"), "w", encoding="utf-8") as handle:
            json.dump({"not": "a trace"}, handle)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))

    def test_selected_rank_is_configurable(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16",
                                "--select-rank", "8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("TP-8-DP-1", os.path.basename(report["parsed_trace"]))

    def test_missing_ranks_fail(self):
        full_set(self.tmp, ranks=[r for r in range(16) if r != 11])
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(any("11" in p for p in report["problems"]),
                        report["problems"])

    def test_whole_remote_node_missing_fails(self):
        full_set(self.tmp, ranks=range(8))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["nodes_represented"], [0])
        self.assertTrue(any("node" in p for p in report["problems"]),
                        report["problems"])

    def test_truncated_export_fails(self):
        full_set(self.tmp, ranks=range(15))
        write_trace(self.tmp, 15, 1, truncate=True)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("unreadable" in p or "truncated" in p
                            for p in report["problems"]), report["problems"])
        bad = [r for r in report["per_rank"] if r["tp_rank"] == 15][0]
        self.assertFalse(bad["readable"])
        self.assertTrue(bad["error"])

    def test_set_without_collectives_fails(self):
        full_set(self.tmp, collectives=0)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["ranks_with_collectives"], [])
        self.assertTrue(any("RCCL" in p for p in report["problems"]),
                        report["problems"])

    def test_empty_dir_is_a_hard_failure(self):
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(report, {})
        self.assertIn("no traces", proc.stderr)

    def test_missing_dir_is_a_hard_failure(self):
        proc, _ = run_gate(os.path.join(self.tmp, "nope"), "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 2)

    def test_selected_rank_absent_is_a_failure(self):
        full_set(self.tmp, ranks=range(1, 16))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16",
                                "--select-rank", "0")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["parsed_trace"], "")

    def test_node_imbalance_is_quantified(self):
        full_set(self.tmp, ranks=range(8), kernels=10)          # node0 busy
        for rank in range(8, 16):                               # node1 idle-ish
            write_trace(self.tmp, rank, 1, kernels=2)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        by_node = {int(k): v for k, v in report["kernel_events_by_node"].items()}
        self.assertGreater(by_node[0], by_node[1])
        self.assertGreater(report["node_kernel_imbalance_pct"], 0)

    def test_a_merged_trace_is_not_a_rank(self):
        full_set(self.tmp)
        with gzip.open(os.path.join(self.tmp, "merged-prof.trace.json.gz"),
                       "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"traceEvents": []}))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertEqual(report["unranked_files"], [])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))

    def test_the_gate_is_idempotent(self):
        full_set(self.tmp)
        run_gate(self.tmp, "--expect-ranks", "16")
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))

    def test_the_node_count_drives_the_node_mapping(self):
        full_set(self.tmp, gpus_per_node=4)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16", "--nodes", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["nodes_represented"], [0, 1, 2, 3])


def pd_launch_json(directory, roles=("prefill", "decode"), tp_by_role=None):
    path = os.path.join(directory, "mn_launch.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"topology": "pd_disagg", "pd": {
            "engine_tp_by_role": tp_by_role or {r: 8 for r in roles},
            "engines": [{"role": r, "rank": i, "url": f"http://10.0.0.{i}:3089{i}"}
                        for i, r in enumerate(roles)],
        }}, handle)
    return path


class PdTraceGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_pd_gate_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.profile = os.path.join(self.tmp, "profile")
        os.makedirs(self.profile)

    def engine(self, role, ranks=range(8), **kwargs):
        directory = os.path.join(self.profile, role)
        os.makedirs(directory, exist_ok=True)
        for rank in ranks:
            write_trace(directory, rank, 0, **kwargs)
        return directory

    def gate(self, *args, tp_by_role=None):
        return run_gate(self.profile, "--launch-json",
                        pd_launch_json(self.tmp, tp_by_role=tp_by_role), *args)

    def test_engines_with_different_tp_are_each_held_to_their_own(self):
        self.engine("prefill", ranks=range(8))
        self.engine("decode", ranks=range(4))
        proc, report = self.gate(tp_by_role={"prefill": 8, "decode": 4})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        by_role = {r["role"]: r for r in report["per_role"]}
        self.assertEqual([by_role[r]["expected_ranks"] for r in ("prefill", "decode")], [8, 4])

    def test_a_short_decode_engine_fails_against_its_own_tp(self):
        self.engine("prefill", ranks=range(4))
        self.engine("decode", ranks=range(4))
        proc, report = self.gate(tp_by_role={"prefill": 4, "decode": 8})
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("decode:7" in p for p in report["problems"]), report["problems"])

    def test_both_engines_present_passes_with_per_role_accounting(self):
        self.engine("prefill")
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["verdict"], "pass", report["problems"])
        self.assertEqual(report["topology"], "pd_disagg")
        by_role = {r["role"]: r for r in report["per_role"]}
        self.assertEqual(by_role["prefill"]["found_ranks"], list(range(8)))
        self.assertEqual(by_role["decode"]["found_ranks"], list(range(8)))
        self.assertEqual(by_role["prefill"]["node_index"], 0)
        self.assertEqual(by_role["decode"]["node_index"], 1)
        self.assertEqual(report["nodes_represented"], [0, 1])

    def test_each_engine_keeps_its_own_rank0_for_the_parser(self):
        self.engine("prefill")
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for role in ("prefill", "decode"):
            trace = report["parsed_trace_by_role"][role]
            self.assertEqual(os.path.dirname(trace), os.path.join(self.profile, role))
            self.assertIn("TP-0-DP-0", os.path.basename(trace))

    def test_the_roles_index_names_one_trace_per_engine(self):
        self.engine("prefill")
        self.engine("decode")
        self.gate()
        with open(os.path.join(self.profile, "ranks", "mn_trace_roles.json"),
                  encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual([r["role"] for r in index["roles"]],
                         ["prefill", "decode"])
        for entry in index["roles"]:
            self.assertTrue(os.path.isfile(entry["trace"]), entry)
            self.assertIn(f"{os.sep}{entry['role']}{os.sep}", entry["trace"])
            self.assertEqual(entry["ranks"], 8)

    def test_a_silent_engine_is_named_not_counted_as_a_missing_node(self):
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("prefill" in p and "NO traces" in p
                            for p in report["problems"]), report["problems"])
        by_role = {r["role"]: r for r in report["per_role"]}
        self.assertEqual(by_role["prefill"]["found_ranks"], [])
        self.assertEqual(by_role["decode"]["found_ranks"], list(range(8)))
        with open(os.path.join(self.profile, "ranks", "mn_trace_roles.json"),
                  encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual({r["role"]: bool(r["trace"]) for r in index["roles"]},
                         {"prefill": False, "decode": True})

    def test_a_rank_missing_inside_one_engine_names_the_engine(self):
        self.engine("prefill", ranks=[r for r in range(8) if r != 5])
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("prefill:5" in p for p in report["problems"]),
                        report["problems"])

    def test_the_shared_directory_collision_is_what_this_avoids(self):
        for rank in range(8):
            write_trace(self.profile, rank, 0)
        proc, report = run_gate(self.profile, "--expect-ranks", "8", "--nodes", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["nodes_represented"], [0])
        self.assertNotIn("per_role", report)


class PdTopnMergeTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_topn_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def topn(self, role, total_ms, kernels):
        path = os.path.join(self.tmp, f"profile_topN_{role}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "source": "torch-trace", "total_gpu_time_ms": total_ms,
                "num_kernel_launches": 100, "num_distinct_kernels": len(kernels),
                "top_kernels": [
                    {"rank": i + 1, "short_name": name, "calls": 10,
                     "total_ms": total_ms * pct / 100.0, "avg_us": 1.0,
                     "pct_gpu_time": pct, "shapes": [], "dtypes": [],
                     "classification": "library_gemm", "backend_guess": "hipblaslt",
                     "editable": False, "opt_hint": "hint"}
                    for i, (name, pct) in enumerate(kernels)],
            }, handle)
        return f"{role}={path}"

    def merge(self, *roles):
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             *[a for r in roles for a in ("--engine", r)]],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        doc = {}
        if os.path.isfile(out + ".json"):
            with open(out + ".json", encoding="utf-8") as handle:
                doc = json.load(handle)
        return proc, doc

    def test_rows_are_weighted_by_each_engine_share_of_gpu_time(self):
        proc, doc = self.merge(self.topn("prefill", 100.0, [("gemm_p", 60.0)]),
                               self.topn("decode", 300.0, [("attn_d", 40.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(doc["total_gpu_time_ms"], 400.0)
        rows = {r["short_name"]: r for r in doc["top_kernels"]}
        self.assertAlmostEqual(rows["gemm_p"]["pct_gpu_time"], 15.0, places=3)
        self.assertAlmostEqual(rows["attn_d"]["pct_gpu_time"], 30.0, places=3)
        self.assertEqual(doc["top_kernels"][0]["short_name"], "attn_d")
        self.assertAlmostEqual(rows["gemm_p"]["pct_gpu_time_in_engine"], 60.0)

    def test_every_row_carries_its_engine(self):
        _proc, doc = self.merge(self.topn("prefill", 100.0, [("shared", 50.0)]),
                                self.topn("decode", 100.0, [("shared", 50.0)]))
        self.assertEqual(sorted(r["engine"] for r in doc["top_kernels"]),
                         ["decode", "prefill"])

    def test_one_unreadable_engine_still_yields_the_other(self):
        proc, doc = self.merge(self.topn("decode", 100.0, [("attn_d", 40.0)]),
                               f"prefill={os.path.join(self.tmp, 'nope.json')}")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([r["engine"] for r in doc["per_engine"]], ["decode"])
        self.assertAlmostEqual(doc["top_kernels"][0]["pct_gpu_time"], 40.0)
        self.assertIn("skipping prefill", proc.stderr)

    def bench(self, ttft_ms, tpot_ms):
        path = os.path.join(self.tmp, "bench_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"ttft_ms_median": ttft_ms, "tpot_ms_median": tpot_ms}, handle)
        return path

    def merge_with_bench(self, bench, osl, *roles):
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             "--bench-summary", bench, "--osl", str(osl),
             *[a for r in roles for a in ("--engine", r)]],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        with open(out + ".json", encoding="utf-8") as handle:
            return proc, json.load(handle)

    def test_the_bench_latency_split_names_the_engine_throughput_waits_on(self):
        proc, doc = self.merge_with_bench(
            self.bench(200.0, 25.0), 33,
            self.topn("prefill", 300.0, [("gemm_p", 50.0)]),
            self.topn("decode", 100.0, [("attn_d", 50.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(doc["bottleneck_engine"], "decode")
        self.assertAlmostEqual(doc["latency_split"]["share"]["decode"], 0.8, places=4)
        roles = {r["engine"]: r for r in doc["per_engine"]}
        self.assertAlmostEqual(roles["prefill"]["e2e_share_pct"], 20.0)
        rows = {r["short_name"]: r for r in doc["top_kernels"]}
        self.assertEqual(doc["top_kernels"][0]["short_name"], "gemm_p")
        self.assertAlmostEqual(rows["gemm_p"]["pct_e2e_est"], 10.0, places=3)
        self.assertAlmostEqual(rows["attn_d"]["pct_e2e_est"], 40.0, places=3)

    def test_a_bench_without_latencies_leaves_the_gpu_time_merge_alone(self):
        path = os.path.join(self.tmp, "bench_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"throughput_tok_s_median": 100.0}, handle)
        proc, doc = self.merge_with_bench(
            path, 33, self.topn("prefill", 100.0, [("gemm_p", 60.0)]),
            self.topn("decode", 300.0, [("attn_d", 40.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no TTFT/TPOT", proc.stderr)
        self.assertNotIn("latency_split", doc)
        self.assertNotIn("bottleneck_engine", doc)
        self.assertIsNone(doc["top_kernels"][0]["pct_e2e_est"])
        self.assertAlmostEqual(doc["top_kernels"][0]["pct_gpu_time"], 30.0, places=3)

    def engine_log(self, name, lines):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("[2026-09-23 12:47:59] server_args={'page_size': 1}\n")
            handle.write("".join(line + "\n" for line in lines))
        return path

    def test_engine_logs_expose_ttft_that_is_kv_handoff_not_prefill_compute(self):
        pre = self.engine_log("server.log", [
            "[2026-09-23 12:48:04 TP0] Prefill batch, #new-seq: 1, #new-token: 1, "
            "#cached-token: 511, token usage: 0.00, #running-req: 0, #queue-req: 0, "
            "#pending-token: 0, #bootstrap-req: 0, #inflight-req: 1, #optimistic-req: 0, "
            "cuda graph: False, input throughput (token/s): 12.3",
            "[2026-09-23 12:48:04 TP0] Prefill batch, #new-seq: 3, #new-token: 3, "
            "#cached-token: 1533, token usage: 0.00, #running-req: 0, #queue-req: 0, "
            "#pending-token: 0, #bootstrap-req: 0, #inflight-req: 5, #optimistic-req: 0, "
            "cuda graph: False, input throughput (token/s): 40.1"])
        dec = self.engine_log("server.rank1.log", [
            "[2026-09-23 12:48:05 TP0] Decode batch, #running-req: 1, #token: 7529, "
            "token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, "
            "#transfer-req: 15, #retracted-req: 0, cuda graph: True, "
            "gen throughput (token/s): 78.57, #queue-req: 0",
            "[2026-09-23 12:48:08 TP0] Decode batch, #running-req: 3, #token: 8198, "
            "token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, "
            "#transfer-req: 13, #retracted-req: 0, cuda graph: True, "
            "gen throughput (token/s): 86.13, #queue-req: 0"])
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             "--engine", self.topn("prefill", 100.0, [("ar_p", 60.0)]),
             "--engine", self.topn("decode", 100.0, [("gemm_d", 40.0)]),
             "--engine-log", f"prefill={pre}", "--engine-log", f"decode={dec}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out + ".json", encoding="utf-8") as handle:
            kv = json.load(handle)["kv_handoff"]
        self.assertEqual(kv["decode"]["transfer_req_mean"], 14.0)
        self.assertEqual(kv["decode"]["running_req_mean"], 2.0)
        self.assertAlmostEqual(kv["decode"]["wait_share"], 28 / 32, places=4)
        self.assertEqual(kv["prefill"]["new_seq_mean"], 2.0)
        self.assertEqual(kv["prefill"]["single_seq_batch_share"], 0.5)
        self.assertEqual(kv["prefill"]["computed_tokens_per_seq"], 1.0)
        self.assertAlmostEqual(kv["prefill"]["prefix_cache_share"], 2044 / 2048, places=4)
        with open(out + ".md", encoding="utf-8") as handle:
            md = handle.read()
        self.assertIn("## KV handoff", md)
        self.assertIn("87.5% of decode-side in-flight requests are parked", md)
        self.assertTrue(kv["verdict"]["handoff_bound"])
        self.assertIn("handoff_bound = true", md)

    def test_real_prefill_compute_does_not_hide_a_handoff_bound_ttft(self):
        pre = self.engine_log("server.log", [
            "[2026-09-24 18:48:19 TP0] Prefill batch, #new-seq: 1, #new-token: 256, "
            "#cached-token: 256, full token usage: 0.00, #running-req: 0, #queue-req: 0, "
            "#pending-token: 0, #bootstrap-req: 0, #inflight-req: 10, #optimistic-req: 0",
            "[2026-09-24 18:48:20 TP0] Prefill batch, #new-seq: 2, #new-token: 1024, "
            "#cached-token: 0, full token usage: 0.00, #running-req: 0, #queue-req: 0, "
            "#pending-token: 0, #bootstrap-req: 0, #inflight-req: 14, #optimistic-req: 0"])
        dec = self.engine_log("server.rank1.log", [
            "[2026-09-24 18:48:20 TP0] Decode batch, #running-req: 2, #full token: 7168, "
            "#prealloc-req: 0, #transfer-req: 14, #retracted-req: 0, cuda graph: True",
            "[2026-09-24 18:48:21 TP0] Decode batch, #running-req: 1, #full token: 4864, "
            "#prealloc-req: 0, #transfer-req: 15, #retracted-req: 0, cuda graph: True"])
        kv = mn_traces.kv_handoff(pre, dec)
        self.assertGreater(kv["prefill"]["computed_tokens_per_seq"], 300)
        self.assertTrue(kv["verdict"]["handoff_bound"])
        self.assertIn("computed_tokens_per_seq", kv["verdict"]["reason"])

    def test_decode_that_is_busy_decoding_is_not_handoff_bound(self):
        v = mn_traces.handoff_verdict({"decode": {"wait_share": 0.2, "running_req_mean": 12.0,
                                           "transfer_req_mean": 3.0}})
        self.assertFalse(v["handoff_bound"])

    def test_a_queue_upstream_of_the_transfer_is_not_handoff_bound(self):
        v = mn_traces.handoff_verdict({
            "decode": {"wait_share": 0.7, "running_req_mean": 4.0, "transfer_req_mean": 9.0},
            "prefill": {"inflight_req_mean": 1.5}})
        self.assertFalse(v["handoff_bound"])
        self.assertIn("upstream", v["reason"])

    def test_no_decode_side_gives_no_verdict(self):
        self.assertEqual(mn_traces.handoff_verdict({"prefill": {"inflight_req_mean": 9.0}}), {})

    def test_logs_without_batch_lines_leave_no_handoff_block(self):
        empty = self.engine_log("server.log", ["[x] loading weights"])
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             "--engine", self.topn("prefill", 100.0, [("ar_p", 60.0)]),
             "--engine-log", f"prefill={empty}",
             "--engine-log", f"decode={os.path.join(self.tmp, 'missing.log')}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no kv_handoff block", proc.stderr)
        with open(out + ".json", encoding="utf-8") as handle:
            self.assertNotIn("kv_handoff", json.load(handle))


class PdRankOverlayTest(unittest.TestCase):

    NAMES = ("GEAK_MN_PD_OVERLAY_PREFILL", "GEAK_MN_PD_OVERLAY_DECODE")

    def setUp(self):
        self._saved = {n: os.environ.pop(n, None) for n in self.NAMES}

    def tearDown(self):
        for name, value in self._saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def test_the_engine_overlay_goes_ahead_of_the_shared_one_on_that_engine_only(self):
        os.environ["GEAK_MN_PD_OVERLAY_DECODE"] = "/geak/overlay/cand_attn"
        self.assertEqual(mn.pd_rank_overlay("decode", "/geak/overlay/shared"),
                         "/geak/overlay/cand_attn:/geak/overlay/shared")
        self.assertEqual(mn.pd_rank_overlay("prefill", "/geak/overlay/shared"),
                         "/geak/overlay/shared")

    def test_unset_leaves_the_shared_overlay_byte_identical(self):
        self.assertEqual(mn.pd_rank_overlay("prefill", "/ovl"), "/ovl")
        self.assertEqual(mn.pd_rank_overlay("decode", ""), "")

    def test_an_engine_overlay_with_no_shared_layer_stands_alone(self):
        os.environ["GEAK_MN_PD_OVERLAY_PREFILL"] = "/geak/overlay/cand_gemm"
        self.assertEqual(mn.pd_rank_overlay("prefill", ""), "/geak/overlay/cand_gemm")

    def test_the_rank_script_exports_the_composed_overlay(self):
        os.environ["GEAK_MN_PD_OVERLAY_DECODE"] = "/dec"

        def script(role):
            return mn.launch_script([], rank_vars("/t", overlay=mn.pd_rank_overlay(role, "/ovl")),
                                    "/a.sh", "adapter_launch")
        self.assertIn("export OVERLAY_PYTHONPATH=/dec:/ovl\n", script("decode"))
        self.assertIn("export OVERLAY_PYTHONPATH=/ovl\n", script("prefill"))
        self.assertNotIn("/dec:", script("prefill"))


PD_STATE = {
    "topology": "pd_disagg",
    "master_ip": "10.10.0.166",
    "serve_port": 30890, "dist_init_port": 30531,
    "shared_fs": "/shared/geak",
    "nodes": [
        {"rank": 0, "node_id": "n0", "ip": "10.10.0.166",
         "hostname": "node-fc54", "cluster_if": "ens50f0", "ngpu": 8, "role": "prefill"},
        {"rank": 1, "node_id": "n1", "ip": "10.10.0.183",
         "hostname": "node-f2a0", "cluster_if": "ens50f0", "ngpu": 8, "role": "decode"},
    ],
}

FAKE_PD_ADAPTER = """\
adapter_pd_mode() { echo pd_roles; }
adapter_pd_args() {
  local role="$1" boot="$2" backend="${3:-mooncake_tcp}" ib="${4:-}"
  local out="--disaggregation-mode $role --disaggregation-transfer-backend $backend"
  [ "$role" = "prefill" ] && out="$out --disaggregation-bootstrap-port $boot"
  [ -n "$ib" ] && out="$out --disaggregation-ib-device $ib"
  echo "$out"
}
adapter_pd_lb_launch() {
  echo "LB_CALL=[$1 $2 $3 $4 $5]"
  echo "LB_MODEL=$MODEL LB_EXTRA=[$LB_EXTRA_ARGS]"
  SERVER_PID=777
}
adapter_launch() {
  echo "FINAL_ARGS=[$EXTRA_SERVER_ARGS]"
  echo "GPU=$GPU HOST=$HOST PORT=$PORT TP=$TP BASE_URL=$BASE_URL"
  SERVER_PID=4242
}
"""


class PdConfigTest(unittest.TestCase):

    def test_the_client_port_is_the_load_balancers(self):
        pd = mn.pd_config(PD_STATE)
        self.assertEqual(pd["lb_port"], PD_STATE["serve_port"])
        self.assertNotIn(PD_STATE["serve_port"], pd["role_ports"])

    def test_defaults_are_derived_from_the_pin(self):
        pd = mn.pd_config(PD_STATE)
        self.assertEqual(pd["roles"], ["prefill", "decode"])
        self.assertEqual(pd["role_ports"], [30891, 30892])
        self.assertEqual(pd["bootstrap_port"], 30898)

    def test_the_transfer_backend_defaults_to_tcp(self):
        self.assertEqual(mn.pd_config(PD_STATE)["transfer_backend"], "mooncake_tcp")

    def test_the_handoffs_pd_block_sets_the_transport_and_ports_follow_serve_port(self):
        state = dict(PD_STATE, serve_port=31000, pd={"backend": "mooncake", "ib_device": "mlx5_0", "lb": "minilb"})
        pd = mn.pd_config(state)
        self.assertEqual((pd["lb_port"], pd["role_ports"], pd["bootstrap_port"]), (31000, [31001, 31002], 31008))
        self.assertEqual((pd["transfer_backend"], pd["ib_device"], pd["lb"]), ("mooncake", "mlx5_0", "minilb"))


class TopologyOfTest(unittest.TestCase):
    def test_the_pin_decides_not_the_environment(self):
        self.assertEqual(mn.topology_of(PD_STATE), "pd_disagg")

    def test_an_older_pin_without_the_field_is_tp_span(self):
        self.assertEqual(mn.topology_of({"serve_port": 30890}), "tp_span")

    def test_an_unknown_topology_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as e:
            mn.topology_of(dict(PD_STATE, topology="half_and_half"))
        self.assertIn("half_and_half", str(e.exception))


class PdRenderTest(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_PD_ADAPTER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self, rank):
        pd = mn.pd_config(PD_STATE)
        node = PD_STATE["nodes"][rank]
        port = pd["role_ports"][rank]
        structural = (f'adapter_pd_args {pd["roles"][rank]} {pd["bootstrap_port"]} '
                      f'{pd["transfer_backend"]}')
        env = rank_vars(self.tmp, tp=8, port=port, base_url=f"http://{node['ip']}:{port}",
                        log=os.path.join(self.tmp, f"server.rank{rank}.log"),
                        extra_env="SHARED_V=1",
                        extra_args="--attention-backend aiter --chunked-prefill-size 8192")
        return mn.launch_script([], env, os.path.join(self.tmp, "fake.sh"),
                                "adapter_launch", structural)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_prefill_engine_gets_the_role_and_the_bootstrap_registry(self):
        out = subprocess.run(["bash"], input=self.engine(0),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertIn("--disaggregation-mode prefill", args)
        self.assertIn("--disaggregation-bootstrap-port 30898", args)
        self.assertIn("--disaggregation-transfer-backend mooncake_tcp", args)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_decode_engine_is_not_given_a_bootstrap_port(self):
        out = subprocess.run(["bash"], input=self.engine(1),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertIn("--disaggregation-mode decode", args)
        self.assertNotIn("--disaggregation-bootstrap-port", args)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_no_rank_or_world_reaches_a_disaggregated_engine(self):
        for rank in (0, 1):
            out = subprocess.run(["bash"], input=self.engine(rank),
                                 capture_output=True, text=True)
            args = [l for l in out.stdout.splitlines()
                    if l.startswith("FINAL_ARGS=")][0]
            for leak in ("--nnodes", "--node-rank", "--dist-init-addr"):
                self.assertNotIn(leak, args, f"rank{rank} was given {leak}")

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_each_engine_answers_on_its_own_endpoint_not_the_clients(self):
        for rank, port in ((0, "30891"), (1, "30892")):
            out = subprocess.run(["bash"], input=self.engine(rank),
                                 capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("GPU=")][0]
            self.assertIn(f"PORT={port}", line)
            self.assertIn("TP=8", line)
            self.assertNotIn(":30890", line)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_swept_flags_go_ahead_of_the_pd_args(self):
        out = subprocess.run(["bash"], input=self.engine(0),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertLess(args.index("--attention-backend aiter --chunked-prefill-size 8192"),
                        args.index("--disaggregation-mode"))
        self.assertLess(args.index("--chunked-prefill-size 8192"),
                        args.index("--disaggregation-mode"))


PD3_STATE = {
    "topology": "pd_disagg", "master_ip": "10.10.0.183",
    "serve_port": 30890, "dist_init_port": 30531,
    "nodes": [
        {"rank": 0, "node_id": "n0", "ip": "10.10.0.183", "hostname": "node-f2a0",
         "cluster_if": "ens50f0", "role": "prefill", "tp": 8, "gpus": 8},
        {"rank": 1, "node_id": "n1", "ip": "10.10.0.247", "hostname": "node-f2b9",
         "cluster_if": "ens50f0", "role": "decode", "tp": 16, "gpus": 8,
         "engine_rank": 0, "engine_nnodes": 2},
        {"rank": 2, "node_id": "n2", "ip": "10.10.0.166", "hostname": "node-fc54",
         "cluster_if": "ens50f0", "role": "decode", "tp": 16, "gpus": 8,
         "engine_rank": 1, "engine_nnodes": 2},
    ],
}


class PdSpanningEngineLaunchTest(unittest.TestCase):
    """A dry-run launch of 1P + a decode engine over two nodes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in (
            "GEAK_MN_STATE", "MODEL", "BACKEND",
            "LOG", "PROFILE_DIR", "OUT_DIR", "GEAK_MN_DRY_RUN", "TP", "GEAK_MN_PD_SWEEP_ARGS_DECODE")}
        here = adapter_root(self.tmp, "fake", FAKE_PD_ADAPTER +
                            'adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }\n')
        here.start()
        self.addCleanup(here.stop)
        state = dict(PD3_STATE, shared_fs=self.tmp)
        with open(os.path.join(self.tmp, "pin.json"), "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.environ.update(
            GEAK_MN_STATE=os.path.join(self.tmp, "pin.json"), BACKEND="fake", MODEL="/m",
            LOG=os.path.join(self.tmp, "server.log"), PROFILE_DIR=os.path.join(self.tmp, "prof"),
            OUT_DIR=self.tmp, GEAK_MN_DRY_RUN="1", TP="8",
            GEAK_MN_PD_SWEEP_ARGS_DECODE="--dp 2 --enable-dp-attention --ep-size 16")

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rendered(self, name):
        with open(os.path.join(self.tmp, f"mn_{name}.sh"), encoding="utf-8") as f:
            return f.read()

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_decode_ranks_form_their_own_world_and_prefill_stays_closed(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(mn.cmd_launch([]), 0)
        pre, d0, d1, lb = (self.rendered(n) for n in ("rank0", "rank1", "rank2", "lb"))
        self.assertNotIn("adapter_dist_args", pre)
        for rank, script in ((0, d0), (1, d1)):
            self.assertIn(f"adapter_dist_args {rank} 2", script)
            self.assertIn("MASTER_ADDR=10.10.0.247", script)
            self.assertIn("export PORT=30892", script)
            self.assertIn("export BASE_URL=http://10.10.0.247:30892", script)
            self.assertIn("SYNC_TOKEN_IDS_ACROSS_TP=1", script)
            self.assertIn("--enable-dp-attention", script)
        self.assertIn("http://10.10.0.183:30891", lb)
        self.assertIn("http://10.10.0.247:30892", lb)
        self.assertNotIn("10.10.0.166:30892", lb)
        # Without the node-facts NIC probe, which needs the real interface.
        script = "\n".join(l for l in d1.splitlines() if not l.startswith("ip -4"))
        out = subprocess.run(["bash"], input=script, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertIn("--disaggregation-mode decode", args)
        self.assertIn("--nnodes 2 --node-rank 1 --dist-init-addr 10.10.0.247:30531", args)
        self.assertLess(args.index("--enable-dp-attention"), args.index("--disaggregation-mode"))


class PdSpanningTraceGateTest(unittest.TestCase):
    def test_a_spanning_engines_ranks_map_onto_its_nodes(self):
        self.assertEqual([mn_traces.lane_node([1, 2], r, 16) for r in (0, 7, 8, 15)], [1, 1, 2, 2])
        self.assertEqual(mn_traces.lane_node([0], 5, 8), 0)


class PdRankSweepTest(unittest.TestCase):

    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "GEAK_MN_PD_SWEEP_ARGS_PREFILL", "GEAK_MN_PD_SWEEP_ARGS_DECODE",
                "GEAK_MN_PD_SWEEP_ENV_PREFILL", "GEAK_MN_PD_SWEEP_ENV_DECODE",
            )
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_each_engine_keeps_only_its_own_sweep(self):
        os.environ["GEAK_MN_PD_SWEEP_ARGS_PREFILL"] = "--chunked-prefill-size 8192"
        os.environ["GEAK_MN_PD_SWEEP_ARGS_DECODE"] = "--max-running-requests 64"
        os.environ["GEAK_MN_PD_SWEEP_ENV_PREFILL"] = "PREFILL_ONLY=1"
        os.environ["GEAK_MN_PD_SWEEP_ENV_DECODE"] = "DECODE_ONLY=1"
        shared = ["SHARED=1"]
        prefill_args, prefill_env, prefill_dropped = mn.pd_rank_swept(
            "prefill", "--attention-backend aiter", shared)
        decode_args, decode_env, decode_dropped = mn.pd_rank_swept(
            "decode", "--attention-backend aiter", shared)
        self.assertEqual(
            prefill_args, "--attention-backend aiter --chunked-prefill-size 8192")
        self.assertNotIn("max-running-requests", prefill_args)
        self.assertEqual(
            decode_args, "--attention-backend aiter --max-running-requests 64")
        self.assertNotIn("chunked-prefill-size", decode_args)
        self.assertLess(prefill_args.index("--attention-backend"),
                        prefill_args.index("--chunked-prefill-size"))
        self.assertEqual(prefill_env, ["SHARED=1", "PREFILL_ONLY=1"])
        self.assertEqual(decode_env, ["SHARED=1", "DECODE_ONLY=1"])
        self.assertEqual(prefill_dropped, [])
        self.assertEqual(decode_dropped, [])
        self.assertEqual(shared, ["SHARED=1"])

    def test_unset_role_sweep_leaves_the_shared_layer(self):
        args, env, dropped = mn.pd_rank_swept("prefill", "--page-size 64", ["A=1"])
        self.assertEqual(args, "--page-size 64")
        self.assertEqual(env, ["A=1"])
        self.assertEqual(dropped, [])


class PdLoadBalancerRenderTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_PD_ADAPTER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def script(self, lb="router"):
        pd = dict(mn.pd_config(PD_STATE), lb=lb)
        return mn.pd_lb_script(PD_STATE, pd, self.tmp, "pd01", adapter=os.path.join(self.tmp, "fake.sh"),
                               model="/models/m", pythonpath="", extra_env="")

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_lb_script_delegates_and_is_handed_both_engines(self):
        out = subprocess.run(["bash"], input=self.script(),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        call = [l for l in out.stdout.splitlines() if l.startswith("LB_CALL=")][0]
        self.assertEqual(
            call,
            "LB_CALL=[0.0.0.0 30890 http://10.10.0.166:30891 30898 "
            "http://10.10.0.183:30892]")
        self.assertIn("STARTED_PID=777", out.stdout)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_xpyd_hands_the_lb_every_replica_and_the_router_takes_them_all(self):
        extra = dict(PD_STATE["nodes"][0], rank=2, ip="10.10.0.32", replica=1)
        state = dict(PD_STATE, nodes=[PD_STATE["nodes"][0], extra, PD_STATE["nodes"][1]])
        script = mn.pd_lb_script(state, mn.pd_config(state), self.tmp, "pd01", adapter=os.path.join(self.tmp, "fake.sh"),
                                 model="/models/m", pythonpath="", extra_env="")
        self.assertIn("adapter_pd_lb_launch 0.0.0.0 30890 'http://10.10.0.166:30891 http://10.10.0.32:30891' "
                      "30898 http://10.10.0.183:30892", script)
        log = os.path.join(self.tmp, "lb.log")
        real = os.path.realpath(os.path.join(HERE, "..", "adapters", "sglang.sh"))
        subprocess.run(["bash", "-c", f'. "$1"; SERVER_LAUNCH_PREFIX=echo MODEL=/m LOG={shlex.quote(log)}; '
                        'adapter_pd_lb_launch 0.0.0.0 30890 "http://a:1 http://b:1" 30898 "http://c:2 http://d:2"; wait',
                        "x", real], check=True)
        with open(log, encoding="utf-8") as f:
            line = f.read()
        self.assertIn("--pd-disaggregation --prefill http://a:1 30898 --prefill http://b:1 30898 "
                      "--decode http://c:2 --decode http://d:2 --host 0.0.0.0 --port 30890", line)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_lb_choice_travels_as_an_argument_not_a_second_template(self):
        out = subprocess.run(["bash"], input=self.script(lb="minilb"),
                             capture_output=True, text=True)
        self.assertIn("LB_EXTRA=[--mini-lb]", out.stdout)


class PdAdapterSeamTest(unittest.TestCase):

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
    def test_an_adapter_with_no_pd_functions_is_refused(self):
        with adapter_root(self.tmp, "tponly", "adapter_dist_mode() { echo per_rank; }\n"
                                              "adapter_dist_args() { echo x; }\n"):
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("tponly", self.tmp, "pd_disagg")
        self.assertIn("adapter_pd_args", str(e.exception))

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_an_adapter_that_can_split_but_not_balance_is_refused(self):
        with adapter_root(self.tmp, "half", "adapter_pd_mode() { echo pd_roles; }\n"
                                            "adapter_pd_args() { echo x; }\n"):
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("half", self.tmp, "pd_disagg")
        self.assertIn("adapter_pd_lb_launch", str(e.exception))

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_real_sglang_adapter_declares_pd_roles(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        self.assertTrue(mn.resolve_adapter("sglang", root, "pd_disagg")
                        .endswith("adapters/sglang.sh"))


class PdTeardownTest(unittest.TestCase):
    def test_the_load_balancer_is_in_the_kill_set(self):
        self.assertIn("mini_lb", mn.LB_PATTERN)
        self.assertNotIn("router", mn.SERVE_PATTERN)

    def test_the_pattern_matches_the_name_the_router_gives_itself(self):
        seen = "sglang::router"          # what ps/pgrep actually report
        self.assertRegex(seen, mn.LB_PATTERN)
        launched_as = "python -m sglang_router.launch_router --pd-disaggregation"
        self.assertNotRegex(launched_as, mn.LB_PATTERN)

    def test_killing_the_load_balancer_does_not_match_the_engines(self):
        self.assertNotRegex("sglang::scheduler", mn.LB_PATTERN)
        self.assertNotRegex("sglang::scheduler_TP0", mn.LB_PATTERN)


class PdLoadBalancerLivenessTest(unittest.TestCase):

    def setUp(self):
        self.state = dict(PD_STATE, pd=mn.pd_config(PD_STATE))
        self.sent = {}
        replies = self.replies = {}

        def reply(_task, node, args):
            self.sent[node["rank"]] = args[0]
            return {"out": replies.get(node["rank"], "serve=1 sched=8 zombie=0") + " ok=1"}

        self.stack = contextlib.ExitStack()
        self.stack.enter_context(fake_ssh(reply))
        self.stack.enter_context(mock.patch.object(mn, "resolve", lambda s: s["nodes"]))

    def tearDown(self):
        self.stack.close()

    def test_the_probe_queries_the_client_port(self):
        mn.engine_health(self.state)
        master = self.sent[0]
        self.assertIn(str(self.state["pd"]["lb_port"]), master)
        self.assertIn("/health", master)

    def test_a_pin_without_a_pd_block_probes_the_default_client_port(self):
        bare = {k: v for k, v in self.state.items() if k != "pd"}
        mn.engine_health(bare)
        self.assertIn(f"127.0.0.1:{mn.pd_config(bare)['lb_port']}/health", self.sent[0])
        self.assertNotIn(":None/", self.sent[0])

    def test_a_router_that_answers_is_not_reported_dead(self):
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=200"
        master = mn.engine_health(self.state)[0]
        self.assertEqual(master["lb"], 1)
        self.assertEqual(master["lb_http"], 200)

    def test_nothing_listening_is_reported_dead(self):
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=000"
        self.assertEqual(mn.engine_health(self.state)[0]["lb"], 0)

    def test_listening_but_refusing_is_reported_dead_and_distinguishable(self):
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=503"
        master = mn.engine_health(self.state)[0]
        self.assertEqual(master["lb"], 0)
        self.assertEqual(master["lb_http"], 503)

    def test_only_the_master_is_asked_about_the_load_balancer(self):
        mn.engine_health(self.state)
        self.assertNotIn("/health", self.sent[1])

    def test_a_tp_span_pin_is_not_asked_at_all(self):
        span = dict(PD_STATE, topology="tp_span")
        span.pop("pd", None)
        health = mn.engine_health(span)
        self.assertNotIn("/health", self.sent[0])
        self.assertNotIn("lb", health[0])

    def test_a_probe_that_does_not_come_back_is_an_error_not_a_dead_rank(self):
        self.replies[1] = ""
        with fake_ssh(lambda task, node, args: {"rc": 255, "out": "", "err": "ssh: connect timed out"}
                      if node["rank"] == 1 else {"rc": 0, "out": "serve=1 sched=8 zombie=0 lb=200 ok=1"}):
            with self.assertRaisesRegex(RuntimeError, "rank1 .* did not answer over ssh .*connect timed out"):
                mn.engine_health(self.state)


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



class TeardownTest(unittest.TestCase):
    def test_orphaned_workers_keep_the_node_busy(self):
        probes = iter([{"hostname": "h", "serve_procs": 0, "worker_procs": 8, "gpu_pids": 0,
                        "foreign_gpu_pids": 0, "max_use": 0.0},
                       {"hostname": "h", "serve_procs": 0, "worker_procs": 0, "gpu_pids": 0,
                        "foreign_gpu_pids": 0, "max_use": 0.0}])
        seen = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(fake_ssh(
                lambda task, node, args: next(probes) if task is mn.probe_node else {"rc": 0}))
            stack.enter_context(mock.patch.object(mn, "resolve", lambda s: s["nodes"][:1]))
            stack.enter_context(mock.patch.object(mn.time, "sleep", lambda s: None))
            self.assertTrue(mn.teardown(PD_STATE, wait_idle_s=60, verbose=seen.append))
        self.assertTrue(any("workers=8" in m for m in seen))

    def test_kills_every_rank_then_reports_clear(self):
        scripts = []
        probe = {"hostname": "node-f2c5", "serve_procs": 0, "gpu_pids": 0,
                 "foreign_gpu_pids": 0, "max_use": 0.0}

        def reply(task, _node, args):
            if task is mn.probe_node:
                return dict(probe)
            scripts.append(args[0])
            return {"hostname": "node-f2c5", "out": "", "rc": 0}

        with contextlib.ExitStack() as stack:
            stack.enter_context(fake_ssh(reply))
            stack.enter_context(mock.patch.object(mn, "resolve", lambda s: s["nodes"]))
            cleared = mn.teardown(PD_STATE, wait_idle_s=0, verbose=lambda *_: None)
        self.assertEqual(cleared, [probe, probe], "the clear probes come back for launch's stack check")
        self.assertEqual(scripts, [mn.kill_script(n) for n in PD_STATE["nodes"]])
        for rank, script in enumerate(scripts):
            pgroup, by_name, pid_file = (script.index('kill -TERM -- "-$pg"'), script.index(mn.KILL_OURS),
                                         script.index(f"rm -f /tmp/geak_mn/rank{rank}.pid"))
            self.assertLess(pgroup, by_name)
            self.assertLess(by_name, pid_file)

    def test_a_provided_router_is_reaped_by_its_recorded_pid(self):
        # No name pattern knows a vllm router; its recorded process group is what kills it.
        script = mn.kill_script({"rank": "lb"})
        self.assertLess(script.index("/tmp/geak_mn/ranklb.pid"), script.index('kill -TERM -- "-$pg"'))
        self.assertIn("rm -f /tmp/geak_mn/ranklb.pid", script)
        router = {"rank": "lb", "node_id": "r", "ip": "10.10.0.32", "hostname": "node-c489"}
        self.assertEqual(mn.state_targets(dict(PD_STATE, router=router))[-1], router)
        self.assertEqual(mn.state_targets(PD_STATE), PD_STATE["nodes"])

    def test_the_supervisor_reaps_the_router_its_launch_recorded(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        router = {k: "r" for k in mn.LAUNCH_NODE_KEYS} | {"rank": "lb"}
        with open(os.path.join(tmp, "mn_launch.json"), "w", encoding="utf-8") as f:
            json.dump({"nodes": PD_STATE["nodes"], "router": router}, f)
        self.assertEqual(mn.supervised_state(dict(PD_STATE, router={}), tmp)["router"], router)


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


class LegEnvTest(unittest.TestCase):
    def test_leg_env_is_the_pins_facts_on_the_driver(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        pin = os.path.join(tmp, "pin.json")
        with open(pin, "w", encoding="utf-8") as f:
            json.dump(dict(PD_STATE, shared_fs=tmp), f)
        with mock.patch.dict(os.environ, {"GEAK_MN_STATE": pin}):
            with on_machine("sandbox/b"), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(mn.cmd_leg_env([]), 0)
        facts = dict(line.split("=", 1) for line in out.getvalue().splitlines() if line.startswith("_MN_") and "=" in line)
        self.assertEqual((facts["_MN_HOST"], facts["_MN_PORT"], facts["_MN_TOPOLOGY"], facts["_MN_NNODES"]),
                         ("10.10.0.166", "30890", "pd_disagg", "2"))
        self.assertEqual(facts["_MN_LOCK_KEY"], "node-f2a0+node-fc54")
        leases = shlex.split(out.getvalue().split("_MN_LEASES=", 1)[1])[0].splitlines()
        self.assertEqual(len(leases), 16)
        self.assertEqual(leases[3], mn.lease_file(dict(PD_STATE, shared_fs=tmp), PD_STATE["nodes"][0], 3))

    def test_ranks_on_one_machine_lease_each_card_once(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        p, d = PD_STATE["nodes"]
        same = dict(d, ip=p["ip"], hostname=p["hostname"], node_id=p["node_id"])
        pin = os.path.join(tmp, "pin.json")
        with open(pin, "w", encoding="utf-8") as f:
            json.dump(dict(PD_STATE, shared_fs=tmp, nodes=[p, same]), f)
        with mock.patch.dict(os.environ, {"GEAK_MN_STATE": pin}), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(mn.cmd_leg_env([]), 0)
        leases = shlex.split(out.getvalue().split("_MN_LEASES=", 1)[1])[0].splitlines()
        self.assertEqual(leases, [mn.lease_file(dict(PD_STATE, shared_fs=tmp), p, g) for g in range(8)])

    def test_xpyd_profiles_one_engine_per_role_and_health_asks_every_replica(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        extra = dict(PD_STATE["nodes"][0], rank=2, node_id="n2", ip="10.10.0.32", hostname="node-c489", replica=1)
        pin = os.path.join(tmp, "pin.json")
        with open(pin, "w", encoding="utf-8") as f:
            json.dump(dict(PD_STATE, shared_fs=tmp, nodes=[PD_STATE["nodes"][0], extra, PD_STATE["nodes"][1]]), f)
        with mock.patch.dict(os.environ, {"GEAK_MN_STATE": pin}), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(mn.cmd_leg_env([]), 0)
        def fact(name):
            return shlex.split(out.getvalue().split(f"_MN_{name}=", 1)[1])[0].splitlines()
        self.assertEqual(fact("PD_ENGINES"),
                         ["prefill=http://10.10.0.166:30891", "decode=http://10.10.0.183:30892"])
        self.assertEqual(fact("PD_HEALTH"),
                         ["http://10.10.0.166:30891", "http://10.10.0.32:30891", "http://10.10.0.183:30892"])


PROVIDED_ROUTER = {"rank": "lb", "node_id": "r", "ip": "10.10.0.32", "hostname": "node-c489", "role": "router",
                   "port": 8000, "launch_command": router_cmd(["10.10.0.166"], ["10.10.0.183"])}
PROVIDED_PD = dict(PD_STATE, mode="provided", backend="sglang", master_ip="10.10.0.32", serve_port=8000, nodes=[
    dict(PD_STATE["nodes"][0], role="prefill", tp=8, gpus=8, port=8000, bootstrap_port=8998,
         launch_command=sg_cmd("prefill")),
    dict(PD_STATE["nodes"][1], role="decode", tp=8, gpus=8, port=8000, launch_command=sg_cmd("decode")),
], router=PROVIDED_ROUTER)


@unittest.skipUnless(HAVE_SH, "needs bash")
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
            self.assertIn(f"print(tree_digest(sys.argv[1]))\n' {os.path.join(self.tmp, 'ovl')}", script)
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

        with mock.patch.object(mn, "teardown", lambda s: [{"sw": {}}]), \
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


class OverlayDigestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        os.makedirs(os.path.join(self.tmp, "pkg", "__pycache__"))
        self.write("pkg/k.py", "X = 1\n")

    def write(self, rel, text):
        with open(os.path.join(self.tmp, rel), "w", encoding="utf-8") as f:
            f.write(text)

    def test_the_digest_follows_source_not_bytecode(self):
        before = mn.tree_digest(self.tmp)
        self.write("pkg/__pycache__/k.cpython-312.pyc", "junk")
        self.write("pkg/x.lock", "")
        self.assertEqual(mn.tree_digest(self.tmp), before)
        self.write("pkg/k.py", "X = 2\n")
        self.assertNotEqual(mn.tree_digest(self.tmp), before)

    def test_no_overlay_no_guard(self):
        self.assertEqual(mn.overlay_guard("", "", 0), [])

    @unittest.skipUnless(HAVE_SH and shutil.which("python3"), "needs bash and python3")
    def test_a_rank_refuses_an_overlay_that_changed_after_the_push(self):
        guard = "\n".join(mn.overlay_guard(self.tmp.replace("\\", "/"), mn.tree_digest(self.tmp), 1))
        self.assertNotIn("mn.py", guard, "the digest runs inline: the node has no mn.py")
        run = lambda: subprocess.run(["bash", "-c", guard + "\necho STARTED"],  # noqa: E731
                                     capture_output=True, text=True)
        self.assertIn(f"OVERLAY_DIGEST={mn.tree_digest(self.tmp)}\nSTARTED", run().stdout)
        self.write("pkg/k.py", "X = 3\n")
        refused = run()
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("refusing to start", refused.stderr)


class ExecTest(_PinnedEnv):
    def setUp(self):
        super().setUp()
        self.pin(PD_STATE)
        self.sent = []

    def exec(self, *argv):
        def stream(state, node, script, timeout=None):
            self.sent.append((node, script, timeout))
            return 5
        with mock.patch.object(mn, "ssh_stream", stream):
            return mn.cmd_exec(list(argv))

    def test_targets_resolve_to_pinned_nodes(self):
        state = dict(PD_STATE, pd=mn.pd_config(PD_STATE))
        pick = lambda name: mn.exec_target(state, name)["rank"]  # noqa: E731
        self.assertEqual([pick(t) for t in ("master", "rank1", "prefill", "decode",
                                            "10.10.0.183", "node-fc54")],
                         [0, 1, 0, 1, 1, 0])
        with self.assertRaisesRegex(SystemExit, "no pinned node 'rank2'"):
            pick("rank2")

    def test_a_command_runs_on_the_target_in_this_directory_on_the_leased_card(self):
        self.assertEqual(self.exec("--target", "decode", "--gpu", "3", "--timeout", "9",
                                   "--", "python3", "-c", "print('hi there')"), 5)
        node, script, timeout = self.sent[0]
        self.assertEqual((node["ip"], timeout), ("10.10.0.183", 9))
        self.assertIn(f"cd {shlex.quote(os.getcwd())}", script)
        self.assertIn("export HIP_VISIBLE_DEVICES=3 GPU_ID=3\n", script)
        self.assertIn("\npython3 -c 'print('\"'\"'hi there'\"'\"')'\n\nGEAK_MN_BODY_", script)
        self.assertTrue(script.startswith("{\nexport GEAK_MN_HOP=1\n"), "run under the watchdog")

    def test_a_script_file_runs_verbatim(self):
        path = os.path.join(self.tmp, "job.sh")
        with open(path, "w", encoding="utf-8") as f:
            f.write("set -e\nrocm-smi\n")
        self.exec("--script", path)
        node, script, _ = self.sent[0]
        self.assertEqual(node["rank"], 0)
        self.assertNotIn("HIP_VISIBLE_DEVICES", script)
        self.assertIn("set -e\nrocm-smi\n", script)

    def test_env_ships_exported_vars_but_not_host_identity_masks_or_secrets(self):
        env = {"OVERLAY_PYTHONPATH": "/w/ov", "GEAK_GPU_ALLOWED": "0,1", "TORCH_EXTENSIONS_DIR": "/w/a b",
               "GEAK_MN_STATE": os.environ["GEAK_MN_STATE"], "PATH": "/x", "CLAUDE_BIN": "/opt/claude",
               "HOME": "/h", "HIP_VISIBLE_DEVICES": "", "ANTHROPIC_API_KEY": "k", "GH_TOKEN": "t"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.exec("--env", "--", "true")
            plain = mn.exported_env()
        script = self.sent[0][1]
        self.assertIn("export TORCH_EXTENSIONS_DIR='/w/a b'\n", script)
        self.assertIn("export GEAK_GPU_ALLOWED=0,1\n", script)
        for name in ("CLAUDE_BIN", "PATH", "HOME", "HIP_VISIBLE_DEVICES", "ANTHROPIC", "GH_TOKEN"):
            self.assertNotIn(f"export {name}", script)
        self.assertEqual(sorted(re.findall(r"export (\w+)=", plain)),
                         ["GEAK_GPU_ALLOWED", "GEAK_MN_STATE", "OVERLAY_PYTHONPATH", "TORCH_EXTENSIONS_DIR"])
        self.exec("--", "true")
        self.assertNotIn("TORCH_EXTENSIONS_DIR", self.sent[1][1])

    @unittest.skipUnless(HAVE_SH and mn.local_identity(), "bash + a Linux boot_id")
    def test_a_target_that_is_this_machine_runs_here_in_this_directory(self):
        state = dict(PD_STATE, nodes=[dict(PD_STATE["nodes"][0], node_id=mn.local_identity()),
                                      PD_STATE["nodes"][1]])
        self.pin(state)
        work = os.path.join(self.tmp, "ws")
        os.makedirs(work)
        proc = subprocess.run([sys.executable, os.path.join(MN, "mn.py"), "exec", "--target", "master",
                               "--", "bash", "-c", 'echo "$PWD"; exit 7'],
                              cwd=work, capture_output=True, text=True, timeout=60)
        self.assertEqual((proc.returncode, proc.stdout.strip()), (7, work), proc.stderr)

    @unittest.skipUnless(HAVE_SH and sys.platform.startswith("linux") and mn.local_identity(), "bash + flock")
    def test_gpu_lock_holds_the_same_card_lease_as_mn_exec(self):
        import socket
        gpu_lock = os.path.realpath(os.path.join(MN, "..", "..", "..", "kernel_workflow", "scripts", "gpu_lock.sh"))
        lock = mn.lease_file(dict(PD_STATE, shared_fs=self.tmp), {"hostname": socket.gethostname()}, 2)
        probe = ("import fcntl,sys; f=open(sys.argv[1],'w')\n"
                 "try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('free')\n"
                 "except BlockingIOError: print('held')")
        # This machine as both nodes: `decode:2` and the bare pool both forward back here through
        # mn.py exec, to the run directory's copy, which finds the leases without mn.py.
        me = {"node_id": mn.local_identity(), "hostname": socket.gethostname()}
        self.pin(dict(PD_STATE, nodes=[n | me for n in PD_STATE["nodes"]],
                      tools_dir=mn.stage_tools(os.path.join(self.tmp, "run"))))
        env = dict(os.environ, GEAK_GPU_REQUIRE_IDLE="0",
                   KERNEL_ENV_SKIP_ENUM_REAP="1", KERNEL_ENV_KEEP_ARCH="1")
        for spec in ("decode:2", "2,9"):
            proc = subprocess.run(["bash", gpu_lock, spec, "bash", "-c",
                                   f'echo "$HIP_VISIBLE_DEVICES"; {shlex.quote(sys.executable)} -c "$0" "$1"',
                                   probe, lock], cwd=self.tmp, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.stdout.split(), ["2", "held"], (spec, proc.stderr))

    def test_bad_usage_is_refused(self):
        for argv in (("--gpu", "all", "--", "true"), ("--script", "x", "--", "true"), ()):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.exec(*argv)
        self.assertEqual(self.sent, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "flock")
    def test_a_leased_card_is_held_until_the_command_ends(self):
        probe = ("import fcntl,sys; f=open(sys.argv[1],'w')\n"
                 "try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('free')\n"
                 "except BlockingIOError: print('held')")
        lock = os.path.join(self.tmp, ".mn_lease", "node-f2a0.gpu3.lock")
        with mn.gpu_lease(PD_STATE | {"shared_fs": self.tmp}, PD_STATE["nodes"][1], "3"):
            inside = subprocess.run([sys.executable, "-c", probe, lock], capture_output=True, text=True).stdout
        after = subprocess.run([sys.executable, "-c", probe, lock], capture_output=True, text=True).stdout
        self.assertEqual((inside.strip(), after.strip()), ("held", "free"))

    def test_an_engine_prefix_on_the_card_picks_the_node(self):
        self.exec("--gpu", "decode:3", "--", "true")
        node, script, _ = self.sent[0]
        self.assertEqual(node["ip"], "10.10.0.183")
        self.assertIn("export HIP_VISIBLE_DEVICES=3 GPU_ID=3\n", script)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.exec("--target", "prefill", "--gpu", "decode:3", "--", "true")
        self.assertEqual(len(self.sent), 1)



@unittest.skipUnless(HAVE_SH, "needs bash")
class LaunchStackDriftTest(_PinnedEnv):
    def test_a_stack_changed_since_check_is_refused_before_anything_spawns(self):
        self.pin(dict(PROVIDED_PD, sw={"sglang": "0.5.1"}))
        os.environ["GEAK_MN_ADAPTERS"] = os.path.join(self.tmp, "adapters")
        moved = [{"hostname": "a", "sw": {"sglang": "0.5.1"}}, {"hostname": "b", "sw": {"sglang": "0.5.2"}}]
        with mock.patch.object(mn, "teardown", lambda s, **k: moved), \
                mock.patch.object(mn, "ssh_each", side_effect=AssertionError("spawned")):
            with self.assertRaisesRegex(SystemExit, "software stack changed since mn.py check.*b {'sglang': '0.5.2'}"):
                self.launch()


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
