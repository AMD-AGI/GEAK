"""The capture -> census -> ledger -> ranking -> selection path against a fake server.

No GPU, no sglang: a stdlib HTTP server answers /generate, /metrics, /start_profile,
/stop_profile and /abort_request, and writes kineto-formatted traces for two TP ranks.
"""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import trace_scan  # noqa: E402
import point_census  # noqa: E402

RANKS = 2
COLL = "aiter::cross_device_reduce_1stage<bf16>"


def kineto_event(cat, name, ts, dur, pid=2, tid=6, args=None):
    inner = json.dumps(args if args is not None else {"device": 0})[1:-1]
    return ('  {\n    "ph": "X", "cat": "%s", "name": %s, "pid": %s, "tid": %s,\n'
            '    "ts": %.3f, "dur": %.3f,\n    "args": {\n      %s\n    }\n  }'
            % (cat, json.dumps(name), pid, tid, ts, dur, inner))


def write_trace(path, events):
    body = ",\n".join(kineto_event(*e) for e in events)
    text = '{\n  "schemaVersion": 1,\n  "traceEvents": [\n%s\n]}\n' % body
    with gzip.open(path, "wt") as fh:
        fh.write(text)


MOE_FRAME = "sglang/srt/models/fake_model.py(40): moe_forward"
MLA_FRAME = "sglang/srt/layers/attention/fake_mla.py(70): mla_forward"
LAYER_FRAME = "sglang/srt/models/fake_model.py(10): layer_forward"


def host_step(i, h, rows):
    """Host-side record_shapes / with_stack events of one step, whose kernels carry the
    correlation ids 3i.. : a layer frame over everything, a MoE frame with its cpu_op
    (External id) launching moe_sort + gemm, and an attention frame launching a Triton
    kernel with no cpu_op (resolved by correlation and containment only)."""
    c0, c1, c2, ext = 3 * i + 1, 3 * i + 2, 3 * i + 3, 100 + i
    dims = {"External id": ext, "Input Dims": [[rows, 6144], [257, 512, 3072]],
            "Input type": ["c10::BFloat16", "c10::Float4_e2m1fn_x2"],
            "Input Strides": [[6144, 1], [1572864, 3072, 1]], "Concrete Inputs": ["", ""]}
    return [
        ("python_function", LAYER_FRAME, h, 400.0, 99, 99, {"Python id": 1}),
        ("python_function", MOE_FRAME, h + 5, 200.0, 99, 99, {"Python id": 2}),
        ("cpu_op", "aiter::fused_moe_", h + 10, 100.0, 99, 99, dims),
        ("cuda_runtime", "hipModuleLaunchKernel", h + 20, 5.0, 99, 99, {"correlation": c0}),
        ("cuda_runtime", "hipModuleLaunchKernel", h + 30, 5.0, 99, 99, {"correlation": c1}),
        ("python_function", MLA_FRAME, h + 250, 100.0, 99, 99, {"Python id": 3}),
        ("cuda_runtime", "hipModuleLaunchKernel", h + 260, 5.0, 99, 99, {"correlation": c2}),
    ], (c0, c1, c2)


REPLAY_FRAMES = ["torch/cuda/graphs.py(139): replay",
                 "sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py(203): replay",
                 "sglang/srt/model_executor/model_runner.py(1713): forward"]


def replay_host_step(i, h):
    """What --enable-profile-cuda-graph records on the host for a replayed step: the
    graph runner's frames around every launch, no cpu_op."""
    c0, c1, c2 = 3 * i + 1, 3 * i + 2, 3 * i + 3
    ev = [("python_function", f, h - k, 400.0 + 2 * k, 99, 99, {"Python id": k})
          for k, f in enumerate(REPLAY_FRAMES)]
    ev += [("cuda_runtime", "hipGraphLaunch", h + 20 + 10 * j, 5.0, 99, 99, {"correlation": c})
           for j, c in enumerate((c0, c1, c2))]
    return ev, (c0, c1, c2)


def decode_events(rank, bs, steps=5, graph=False, replay=False):
    """`steps` VERIFY anchors; each step: moe_sort 5us + gemm 100us (consecutive, one MoE
    frame), attn 50us (its own frame), and a collective that takes 10us on the laggard and
    10+500us on the rank that waited."""
    ev, t = [], 1000.0
    for i in range(steps):
        span = "step[VERIFY bs=%d]" % bs
        ev.append(("user_annotation", span, t, 400.0, 99, 99))
        if not graph:
            ev.append(("gpu_user_annotation", span, t, 400.0))
            ev.append(("gpu_user_annotation", "step[DRAFT bs=%d]" % bs, t + 300, 50.0))
        host, (c0, c1, c2) = (replay_host_step(i, 5e6 + t) if replay
                              else host_step(i, 5e6 + t, bs * 6))
        ev += host
        ev.append(("kernel", "moe_sort_kernel", t + 4, 20.0, 2, 6, {"correlation": c0}))
        ev.append(("kernel", "gemm_a4w4_moe", t + 25, 100.0, 2, 6,
                   {"correlation": c1, "grid": [8, 1, 1], "block": [256, 1, 1]}))
        ev.append(("kernel", "sparse_mla_attn", t + 120, 50.0, 2, 6, {"correlation": c2}))
        spin = 500.0 if (i + rank) % 2 else 0.0
        ev.append(("kernel", COLL, t + 200, 10.0 + spin))
        t += 1000.0
    return ev


def prefill_events(toks, steps=3):
    ev, t = [], 1000.0
    for i in range(steps):
        ev.append(("gpu_user_annotation", "step[EXTEND bs=1 toks=%d]" % toks, t, 600.0))
        host, (c0, c1, c2) = host_step(i, 5e6 + t, toks)
        ev += host
        ev.append(("kernel", "moe_sort_kernel", t + 4, 20.0, 2, 6, {"correlation": c0}))
        ev.append(("kernel", "gemm_a4w4_moe", t + 25, 290.0, 2, 6,
                   {"correlation": c1, "grid": [64, 1, 1], "block": [256, 1, 1]}))
        ev.append(("kernel", "flash_prefill", t + 320, 200.0, 2, 6, {"correlation": c2}))
        ev.append(("gpu_user_annotation", "step[DRAFT bs=1]", t + 600, 50.0))
        ev.append(("kernel", "draft_gemm", t + 610, 20.0))
        t += 1000.0
    return ev


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class FakeServer:
    """sglang's probe surface. Decode requests stay resident until aborted; the realtime
    decode counter advances while they are; prefill traces carry the tokens actually
    extended: a radix cache by content, so each request extends past its longest prefix
    shared with any earlier request."""

    def __init__(self, graph=False, reject_shapes=False, replay=False):
        self.graph, self.reject_shapes, self.replay = graph, reject_shapes, replay
        self.active, self.out_dir, self.armed = {}, None, False
        self.requests, self.profile_bodies, self.seen = [], [], []
        self.decode_total, self.prefill_total, self.t_last = 0.0, 0.0, time.time()
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype="application/json", code=200):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with fake.lock:
                    now = time.time()
                    fake.decode_total += 50.0 * len(fake.active) * (now - fake.t_last)
                    fake.t_last = now
                    n = len(fake.active)
                self._send("sglang:num_running_reqs %d\nsglang:gen_throughput %.1f\n"
                           'sglang:realtime_tokens_total{mode="decode"} %.1f\n'
                           'sglang:realtime_tokens_total{mode="prefill_compute"} %.1f\n'
                           % (n, 10.0 if n else 0.0, fake.decode_total, fake.prefill_total),
                           "text/plain")

            def do_POST(self):
                ln = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(ln) or b"{}")
                if self.path == "/generate":
                    ids = list(body["input_ids"])
                    n = len(ids)
                    with fake.lock:
                        hit = max([0] + [_lcp(ids, seen) for seen in fake.seen])
                        fake.seen.append(ids)
                    fake.requests.append({"tokens": n, "armed": fake.armed, "extend": n - hit,
                                          "max_new": body["sampling_params"]["max_new_tokens"]})
                    fake.prefill_total += n
                    if body["sampling_params"]["max_new_tokens"] > 1:
                        ev = threading.Event()
                        rid = body.get("rid") or str(id(ev))
                        with fake.lock:
                            fake.active[rid] = ev
                        ev.wait(30)
                        with fake.lock:
                            fake.active.pop(rid, None)
                    self._send("{}")
                elif self.path == "/abort_request":
                    ev = fake.active.get(body.get("rid"))
                    if ev:
                        ev.set()
                    self._send("{}")
                elif self.path == "/start_profile":
                    if fake.reject_shapes and "record_shapes" in body:
                        self._send('{"detail": "extra fields not permitted"}', code=422)
                        return
                    fake.profile_bodies.append(body)
                    fake.out_dir, fake.armed, fake.req0 = body["output_dir"], True, len(fake.requests)
                    if fake.active:
                        bs = len(fake.active)
                        threading.Timer(0.2, fake.dump, ("decode", bs)).start()
                    self._send("Start profiling.\n", "text/plain")
                elif self.path == "/stop_profile":
                    if not fake.active and fake.out_dir:
                        after = fake.requests[fake.req0:]
                        fake.dump("prefill", sum(r["extend"] for r in after))
                    fake.out_dir, fake.armed = None, False
                    self._send("Stop profiling.\n", "text/plain")
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def dump(self, phase, shape):
        out = self.out_dir
        if not out:
            return
        stamp = "%.4f" % time.time()
        for r in range(RANKS):
            ev = (decode_events(r, shape, graph=self.graph, replay=self.replay)
                  if phase == "decode"
                  else prefill_events(shape))
            write_trace(str(Path(out) / ("%s-TP-%d.trace.json.gz" % (stamp, r))), ev)

    def close(self):
        for ev in list(self.active.values()):
            ev.set()
        self.httpd.shutdown()


def points_plan(path):
    pts = [
        {"point_id": "pr_64", "phase": "PREFILL", "new_tokens": 64, "running": 1,
         "represented": {"instances": 10, "max_shape_ratio": 1.0}},
        {"point_id": "pr_256", "phase": "PREFILL", "new_tokens": 256, "running": 2,
         "cached_tokens": 512, "represented": {"instances": 10, "max_shape_ratio": 1.0}},
        {"point_id": "de_2", "phase": "DECODE", "batch_size": 2, "context_tokens": 64,
         "represented": {"instances": 100, "max_shape_ratio": 1.0}},
        {"point_id": "de_1", "phase": "DECODE", "batch_size": 1, "context_tokens": 64,
         "represented": {"instances": 100, "max_shape_ratio": 1.0}},
    ]
    path.write_text(json.dumps({"points": pts}))


def observed(path):
    """observed_workload.py's raw rows: prefill batches around 64 and 256 tokens plus one
    band (~1000) no point covers; decode at batch 1 and 2."""
    pre = [64] * 6 + [70] * 4 + [256] * 8 + [240] * 2 + [1000] * 5
    path.write_text(json.dumps({
        "decode_log_interval": 10,
        "raw_prefill_rows": [{"new_token": n} for n in pre],
        "raw_decode_rows": [{"running": 1, "iterations": 10}] * 10
                           + [{"running": 2, "iterations": 10}] * 10}))


FAST = ["--probe-arg=--settle-poll=0.2", "--probe-arg=--arm-timeout=30",
        "--probe-arg=--window-timeout=20", "--probe-arg=--drain-timeout=20",
        "--probe-arg=--steady-timeout=20"]


class TraceScanTest(unittest.TestCase):
    def test_regex_matches_json_and_falls_back_on_unknown_layout(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "t.trace.json.gz")
            ev = decode_events(0, 3) + [("kernel", 'name "with" quotes\\', 9e3, 1.0)]
            write_trace(p, ev)
            s = trace_scan.scan(p)
            self.assertEqual(s.method, "regex")
            ref, _ = trace_scan._scan_json(p, trace_scan.DEFAULT_CATS)
            self.assertEqual(sorted(e[:4] for e in s.events), sorted(e[:4] for e in ref))
            # With args: host events and kernel args parsed exactly as json.load sees them.
            cats = trace_scan.DEFAULT_CATS + ("cpu_op", "python_function", "cuda_runtime")
            args = ("kernel", "cpu_op", "cuda_runtime")
            s = trace_scan.scan(p, cats, args)
            ref, _ = trace_scan._scan_json(p, cats, args)
            key = lambda e: (e[0], e[1], e[2], json.dumps(e[6], sort_keys=True))  # noqa: E731
            self.assertEqual(s.method, "regex")
            self.assertEqual(sorted(map(key, s.events)), sorted(map(key, ref)))
            # Reordered fields: the regex finds nothing, the counts disagree, json takes over.
            q = str(Path(d) / "r.trace.json")
            Path(q).write_text(json.dumps({"traceEvents": [
                {"name": "k", "ph": "X", "cat": "kernel", "ts": 1, "dur": 2, "pid": 0,
                 "tid": 0}]}))
            s = trace_scan.scan(q)
            self.assertEqual((s.method, len(s.events)), ("json", 1))


class PointCensusTest(unittest.TestCase):
    def test_decode_instances_and_per_launch_collective_split(self):
        with tempfile.TemporaryDirectory() as d:
            for r in range(RANKS):
                write_trace(str(Path(d) / ("1-TP-%d.trace.json.gz" % r)), decode_events(r, 4))
            out = point_census.census(d, "DECODE", "de_4", workers=1)
            self.assertEqual((out["instances"], out["achieved_shape"]), (4, 4))
            for kv in out["per_rank"].values():
                self.assertAlmostEqual(kv[COLL], 4 * 10.0)
                self.assertAlmostEqual(kv["gemm_a4w4_moe"], 4 * 100.0)
            waits = sum(kv[point_census.WAIT_KERNEL] for kv in out["per_rank"].values())
            # 4 instances, one rank spins 500 us in each.
            self.assertAlmostEqual(waits, 4 * 500.0)

    def test_graph_hidden_decode_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            write_trace(str(Path(d) / "1-TP-0.trace.json.gz"), decode_events(0, 2, graph=True))
            with self.assertRaises(SystemExit) as cm:
                point_census.census(d, "DECODE", workers=1)
            self.assertIn("--disable-cuda-graph", str(cm.exception))

    def test_prefill_attaches_named_draft(self):
        with tempfile.TemporaryDirectory() as d:
            write_trace(str(Path(d) / "1-TP-0.trace.json.gz"), prefill_events(128))
            out = point_census.census(d, "PREFILL", workers=1)
            self.assertEqual((out["instances"], out["prefill_steps_without_draft"]), (3, 0))
            self.assertAlmostEqual(out["per_rank"]["0"]["draft_gemm"], 60.0)


class RunPointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fast path ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        points_plan(self.root / "points.json")
        observed(self.root / "observed.json")

    def run_points(self, server, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "run_points.py"), "--points",
             str(self.root / "points.json"), "--base-url", server.url,
             "--trace-root", str(self.root / "traces"), "--expect-files", str(RANKS),
             "--eval-dir", str(self.root / "eval"), *FAST, *extra],
            capture_output=True, text=True, timeout=300)

    def script(self, name, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)],
                              capture_output=True, text=True, timeout=300)

    def test_template_path_from_capture_to_ut_scaffold(self):
        server = FakeServer()
        self.addCleanup(server.close)
        p = self.run_points(server, "--decode-eager", "--observed", self.root / "observed.json")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        ev = self.root / "eval"
        run = json.loads((ev / "profile_run.json").read_text())
        self.assertEqual(run["status"], "ok")
        self.assertEqual(list(run["points"]), ["pr_64", "pr_256", "de_1", "de_2"])
        self.assertTrue(all(len(v["attempts"]) == 1 for v in run["points"].values()))
        # Probe: armed with record_shapes; the two-sequence prefill split its batch totals
        # and warmed both prefixes BEFORE the window (the trace then holds exactly 256).
        self.assertTrue(all(b.get("record_shapes") for b in server.profile_bodies))
        warm = [r["tokens"] for r in server.requests if not r["armed"] and r["max_new"] == 1]
        self.assertIn(256, warm)
        probe = json.loads((ev / "evidence" / "probe" / "pr_256.json").read_text())
        self.assertEqual(probe["status"], "ok", probe["verify_reason"])
        steady = [e for e in json.loads((ev / "evidence" / "probe" / "de_1.json").read_text())
                  ["journal"] if e["event"] == "steady_state_confirmed"]
        self.assertEqual(steady[0]["counters"], "realtime_tokens_total")
        # Ranking and selection.
        rank = json.loads((ev / "kernel_ranking.json").read_text())
        self.assertNotIn(point_census.WAIT_KERNEL, [r["name"] for r in rank["ranking"]])
        excl = rank["denominators"]["excluded_from_denominator"]
        self.assertGreater(excl["DECODE"]["mass_us"], 0)
        sel = json.loads((ev / "selection_targets.json").read_text())
        ids = [t["id"] for t in sel["targets"]]
        self.assertIn("gemm_a4w4_moe", ids)
        self.assertEqual(server.active, {})
        # Relabel: the ~1000-token band is outside every point and becomes a bucket.
        final = json.loads((ev / "sampling_points_final.json").read_text())
        self.assertEqual(final["phase_summary"]["PREFILL"]["total_instances"], 25)
        # Evidence: shapes with host ops and frames, source map, adjacency, draft.
        sh = json.loads((ev / "evidence" / "shapes" / "de_1.json").read_text())
        g = sh["kernels"]["gemm_a4w4_moe"]["variants"][0]
        self.assertEqual(g["launches_per_instance"], 1.0)
        self.assertEqual(g["samples"][0]["host_op"]["op"], "aiter::fused_moe_")
        self.assertEqual(g["samples"][0]["host_op"]["input_dims"][0], [6, 6144])
        mla = sh["kernels"]["sparse_mla_attn"]["variants"][0]["samples"][0]
        self.assertEqual(mla["resolved_by"], "launch_correlation")
        self.assertEqual(mla["py_frames"][0], MLA_FRAME)
        decls = json.loads((ev / "task_decls.json").read_text())
        groups = {tuple(sorted(t["kernels"])): t for t in decls["tasks"]}
        moe = groups[("gemm_a4w4_moe", "moe_sort_kernel")]
        self.assertEqual(moe["ut_family"], "fused_moe")
        self.assertIn("moe_forward", moe["target_callable"])
        # prefill's flash_prefill and decode's sparse_mla_attn never launch together but
        # enter through one callable: one task, two dispatch branches.
        attn = groups[("flash_prefill", "sparse_mla_attn")]
        self.assertEqual(attn["ut_family"], "sparse_mla")
        self.assertTrue(all(len(j["required_kernels"]) == 1 for j in attn["joint_phases"]))
        # A collective stays a target: a task on the deployment's world size, not an
        # exclusion (its lane builds the multi-rank harness).
        coll = next(t for t in decls["tasks"] if COLL in t["kernels"])
        self.assertEqual(coll["required_gpu_count"], RANKS)
        self.assertEqual(coll["ut_family"], "all_reduce")
        self.assertNotIn(COLL, [k for x in decls["excluded"] for k in x["kernels"]])
        # Selection ran per operator over resolved stacks: the MoE op carries its sort
        # kernel as a member, whatever that kernel's own share.
        self.assertEqual(sel["policy"]["unit"], "operator")
        moe_op = next(o for o in sel["ops"] if "gemm_a4w4_moe" in o["members"])
        self.assertIn("moe_sort_kernel", moe_op["members"])
        self.assertTrue((ev / "attribution_candidates.json").exists())
        # Unreviewed draft is refused; the reviewed one builds the plan and the scaffold.
        p = self.script("build_task_plan.py", "--eval-dir", ev, "--check")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("needs_review", p.stdout)
        for t in decls["tasks"] + decls["excluded"]:
            t["needs_review"] = []
        moe["axes"] = {"model_dim": 6144, "inter_dim": 256, "n_experts": 257, "topk": 9}
        moe["attrs"] = {"quant": "mxfp4"}
        moe["case_axes"] = {"PREFILL": {"n_tokens": "shape"}, "DECODE": {"n_tokens": "shape * 6"}}
        (ev / "task_decls.json").write_text(json.dumps(decls))
        p = self.script("build_task_plan.py", "--eval-dir", ev)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        plan = json.loads((ev / "task_plan.json").read_text())
        owned = {k for t in plan["tasks"] for k in t["selected_identities"]}
        blocked = {b["kernel"] for b in plan["blocked_identities"]}
        self.assertEqual(owned | blocked, set(ids))
        mt = next(t for t in plan["tasks"] if t["label"] == moe["label"])
        self.assertEqual(mt["scaffold"]["status"], "ok", mt["scaffold"])
        tdir = Path(mt["task_dir"])
        cases = json.loads((tdir / "run" / "ut_cases.json").read_text())
        kinds = {c["kind"] for c in cases["cases"]}
        self.assertEqual(kinds, {"point", "bucket"})
        dec = [c for c in cases["cases"] if c["regime"] == "DECODE"]
        self.assertEqual(sorted(c["axes"]["n_tokens"] for c in dec), [6, 12])
        self.assertEqual(len([s for s in cases["replay_sigs"] if s.startswith("DECODE")]), 2)
        self.assertTrue((tdir / "unittest.py").exists())
        req = json.loads((tdir / "run" / "requested_kernel.json").read_text())
        self.assertEqual(len(req["shared"]["census"]), 4)
        at = next(t for t in plan["tasks"] if t["label"] == attn["label"])
        areq = json.loads(Path(at["requested_kernel"]).read_text())
        self.assertIsNone(areq["joint_ut"])
        self.assertEqual(sorted(r["device_kernel"] for r in areq["dispatch_ut"]["phases"]),
                         ["flash_prefill", "sparse_mla_attn"])
        # build_task_plan stamps what each phase must launch, for the UT's launches gate.
        meta = json.loads((tdir / "meta.json").read_text())
        self.assertEqual(meta["expected_launches"]["DECODE"], ["gemm_a4w4_moe", "moe_sort_kernel"])
        ct = next(t for t in plan["tasks"] if t["label"] == coll["label"])
        self.assertEqual(json.loads(Path(ct["requested_kernel"]).read_text())
                         ["required_gpu_count"], RANKS)
        # requested_kernel.json is the delivery gate's KERNEL object (delivery-contract.md):
        # entity_kind gpu_kernel, and seam candidates the gate keeps -- a numeric depth and
        # device_kernels naming the kernel each one must launch.
        for r in (req, areq):
            self.assertEqual(r["entity_kind"], "gpu_kernel")
            head = r["seam_candidates"]
            self.assertEqual(len(head), 1)
            self.assertIsInstance(head[0]["depth"], int)
            self.assertIn(r["device_kernel"], head[0]["device_kernels"])
        for p in areq["dispatch_ut"]["phases"]:
            self.assertEqual(p["seam_candidates"][0]["device_kernels"], [p["device_kernel"]])
            self.assertIsInstance(p["seam_candidates"][0]["depth"], int)
        self.assertGateAcceptsDispatch(areq)

    def assertGateAcceptsDispatch(self, req):
        """The gate's own kernelSelectionVerified must accept build_task_plan's dispatch
        branches given a matching raw verdict per branch (the shape kernel_selection.py
        writes). Before the KERNEL object carried entity_kind and structured seams, every
        branch failed with 'no relevant structured seam_candidates were returned'."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed; the gate check needs Node.js 18+")
        target = req["target_callable"]
        rows = [{"regime": p["regime"], "target_callable": target, "selection_validation": {
            "contract": "kernel_selection", "ok": True, "target_callable": target,
            "device_kernel": p["device_kernel"], "total_calls_observed": 1,
            "target_marker_calls": 1, "matched_kernel_calls": 1,
            "candidate_targets_tested": [target], "live_candidate_targets": [target],
            "deeper_live_candidates": [], "deepest_verified": True, "failed": []}}
            for p in req["dispatch_ut"]["phases"]]
        ext = {"target_callable": target, "phase_selection_validations": rows}
        contract = SCRIPTS.parent / "shared" / "delivery_contract.mjs"
        js = ("const m = await import(process.argv[1]);"
              "const [h, e] = JSON.parse(process.argv[2]);"
              "console.log(JSON.stringify(m.kernelSelectionVerified(h, e)));")
        p = subprocess.run([node, "--input-type=module", "-e", js, contract.as_uri(),
                            json.dumps([req, ext])], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        verdict = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(verdict["ok"], verdict["why"])

    def test_graph_mode_decode_fails_fast_without_recapture(self):
        server = FakeServer(graph=True)
        self.addCleanup(server.close)
        p = self.run_points(server, "--point-id", "de_1")
        self.assertNotEqual(p.returncode, 0)
        run = json.loads((self.root / "eval" / "profile_run.json").read_text())
        att = run["points"]["de_1"]["attempts"]
        self.assertEqual(len(att), 1)
        self.assertTrue(att[0]["graph_hidden"])
        self.assertIn("--disable-cuda-graph", att[0]["reason"])

    def test_server_rejecting_record_shapes_still_captures(self):
        server = FakeServer(reject_shapes=True)
        self.addCleanup(server.close)
        p = self.run_points(server, "--point-id", "de_1", "--no-evidence")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        j = json.loads((self.root / "eval" / "evidence" / "probe" / "de_1.json").read_text())
        self.assertIn("record_shapes_rejected", [e["event"] for e in j["journal"]])

    def test_resume_skips_censused_points(self):
        server = FakeServer()
        self.addCleanup(server.close)
        self.assertEqual(self.run_points(server, "--no-evidence").returncode, 0)
        n = len(server.profile_bodies)
        p = self.run_points(server, "--no-evidence")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(server.profile_bodies), n)
        p = self.run_points(server, "--no-evidence", "--redo", "de_1")
        self.assertEqual(len(server.profile_bodies), n + 1)


class EagerEvidenceTest(unittest.TestCase):
    setUp = RunPointsTest.setUp
    run_points = RunPointsTest.run_points

    def test_graph_replayed_decode_gets_one_eager_evidence_window(self):
        graph = FakeServer(replay=True)
        self.addCleanup(graph.close)
        obs = ("--observed", self.root / "observed.json")
        p = self.run_points(graph, *obs)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        ev = self.root / "eval"
        req = json.loads((ev / "evidence" / "eager_evidence_request.json").read_text())
        self.assertEqual(req["point"], "de_1")
        self.assertIn("gemm_a4w4_moe", req["kernels"])
        sm = json.loads((ev / "evidence" / "source_map.json").read_text())
        self.assertTrue(sm["kernels"]["sparse_mla_attn"]["graph_replayed_only"])
        # What probe_server.py does next: the same command on an eager server.
        eager = FakeServer()
        self.addCleanup(eager.close)
        p = self.run_points(eager, *obs, "--evidence-capture")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(eager.profile_bodies), 1)          # one window, nothing redone
        self.assertTrue((ev / "evidence" / "shapes" / "de_1@eager.json").exists())
        sm = json.loads((ev / "evidence" / "source_map.json").read_text())
        self.assertFalse(sm["kernels"]["sparse_mla_attn"]["graph_replayed_only"])
        self.assertEqual(sm["kernels"]["sparse_mla_attn"]["frames"][0]["frame"], MLA_FRAME)
        decls = json.loads((ev / "task_decls.json").read_text())
        groups = {tuple(sorted(t["kernels"])) for t in decls["tasks"]}
        self.assertIn(("gemm_a4w4_moe", "moe_sort_kernel"), groups)
        # Costs still come from the graph windows only.
        cov = json.loads((ev / "coverage.json").read_text())
        self.assertEqual({p["point_id"] for p in cov["phases"]["DECODE"]["points"]}, {"de_1", "de_2"})


class BudgetPlannerTest(unittest.TestCase):
    """Prefill coverage is two-dimensional: the same extend on a short and on a long
    cached prefix are different workloads."""

    def test_same_extend_on_two_contexts_needs_two_points_and_relabels_in_2d(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pre = ([{"new_token": 1000, "cached_token": 3000, "new_seq": 1}] * 10
                   + [{"new_token": 1000, "cached_token": 120000, "new_seq": 1}] * 10)
            (d / "obs.json").write_text(json.dumps({
                "decode_log_interval": 10, "raw_prefill_rows": pre,
                "raw_decode_rows": [{"running": 2, "token": 200000, "iterations": 10}] * 5}))
            p = subprocess.run([sys.executable, str(SCRIPTS / "plan_points_budget.py"),
                                "--observed", str(d / "obs.json"), "--time-budget-s", "300",
                                "--out", str(d / "pts.json")], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            plan = json.loads((d / "pts.json").read_text())
            pre_pts = [x for x in plan["points"] if x["phase"] == "PREFILL"]
            self.assertEqual(sorted(x["context_tokens"] for x in pre_pts), [4000, 121000])
            self.assertTrue(all(x["new_tokens"] == 1000 and x["running"] == 1 for x in pre_pts))
            self.assertEqual(plan["phase_summary"]["PREFILL"]["mass_outside_tolerance_pct"], 0.0)
            # Relabel with only the short-context point measured: the long-context half of
            # the workload is outside tolerance in 2-D, though its token count matches.
            sys.path.insert(0, str(SCRIPTS))
            import relabel_points
            cdir = d / "census"
            cdir.mkdir()
            short = next(x for x in pre_pts if x["context_tokens"] == 4000)
            (cdir / (short["point_id"] + ".json")).write_text(json.dumps(
                {"achieved_shape": 1000, "instances": 1}))
            out = relabel_points.relabel(json.loads((d / "obs.json").read_text()),
                                         {"points": pre_pts}, str(cdir), str(d), 1.25, 1.5)
            ph = out["phase_summary"]["PREFILL"]
            self.assertAlmostEqual(ph["instances_outside_tolerance_pct"], 50.0)
            self.assertEqual(ph["unrepresented_bands"][0]["context"], 121000)
            out1 = relabel_points.relabel(json.loads((d / "obs.json").read_text()),
                                          {"points": pre_pts}, str(cdir), str(d), 1.25, None)
            self.assertEqual(out1["phase_summary"]["PREFILL"]["instances_outside_tolerance_pct"], 0.0)


FAKE_LAUNCHER = r"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
if os.environ.get("FAKE_REJECT_PROFILE_GRAPH") and "--enable-profile-cuda-graph" in argv:
    sys.stderr.write("launch_server: error: unrecognized arguments: --enable-profile-cuda-graph\n")
    sys.exit(2)
port = int(argv[argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"argv": argv, "profiler_dir": os.environ.get("SGLANG_TORCH_PROFILER_DIR"),
                           "gpu_archs": os.environ.get("GPU_ARCHS")}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


class ProbeServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="probe server ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        launcher = self.root / "fake_launch.py"
        launcher.write_text(FAKE_LAUNCHER)
        import socket
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        self.serving = self.root / "serving.json"
        self.serving.write_text(json.dumps({
            "model": "/m", "tp": 2, "port": port, "gpus": "0,1",
            "launcher": [sys.executable, str(launcher)], "args": ["--trust-remote-code"],
            "env": {"GPU_ARCHS": "gfx950"}}))

    def run_probe(self, env_extra=None, cmd=("true",)):
        import os
        env = dict(os.environ, **(env_extra or {}))
        return subprocess.run([sys.executable, str(SCRIPTS / "probe_server.py"), "run",
                               "--serving", str(self.serving), "--run-dir", str(self.root / "run"),
                               "--ready-timeout", "60", "--", *cmd],
                              capture_output=True, text=True, timeout=180, env=env)

    def test_runs_command_against_the_declared_server_then_stops_it(self):
        p = self.run_probe(cmd=(sys.executable, "-c", "import sys; sys.exit(3)"))
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        st = json.loads((self.root / "run" / "probe_server.json").read_text())
        self.assertEqual((st["decode_capture"], st["decode_eager"], st["expect_files"]),
                         ("profile_cuda_graph", False, 2))
        info = json.loads((self.root / "run" / "server_info.json").read_text())
        self.assertIn("--enable-profile-cuda-graph", info["argv"])
        self.assertIn("--trust-remote-code", info["argv"])
        self.assertEqual(info["profiler_dir"], st["trace_root"])
        self.assertEqual(info["gpu_archs"], "gfx950")
        import os
        with self.assertRaises(ProcessLookupError):
            os.killpg(st["pgid"], 0)

    def test_rejected_profile_graph_flag_falls_back_to_eager(self):
        p = self.run_probe({"FAKE_REJECT_PROFILE_GRAPH": "1"})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        st = json.loads((self.root / "run" / "probe_server.json").read_text())
        self.assertEqual((st["decode_capture"], st["decode_eager"]), ("eager", True))
        info = json.loads((self.root / "run" / "server_info.json").read_text())
        self.assertIn("--disable-cuda-graph", info["argv"])
        self.assertNotIn("--enable-profile-cuda-graph", info["argv"])


if __name__ == "__main__":
    unittest.main()
