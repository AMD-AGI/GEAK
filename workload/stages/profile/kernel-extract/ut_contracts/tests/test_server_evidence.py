"""GPU tests for the server-evidence gates (server_calls, compile_variants, server_preimports).

A toy "serving stack" reproduces the three ways Muse Spark's attention_2d_prefill and
attention_2d_decode passed their UTs and failed in the server:

  * the server's first (warmup) call builds a Triton variant no UT case builds: an int
    argument equal to 1 is specialized to a constexpr, so M=1 compiles a different kernel
    from M=4 or M=200 although every M is "covered";
  * a candidate that is wrong only on that path (its M=1 branch does not compile);
  * the framework rebinds `triton.next_power_of_2` at import, so a candidate calling it in a
    jit body compiles in a bare UT process and fails in the server.

The server is a subprocess with the probe server's compile_watch overlay on PYTHONPATH; the
evidence goes through build_task_plan.server_evidence exactly as in a profile run.

Run (needs torch, triton and a GPU):
  python3 workload/stages/profile/kernel-extract/ut_contracts/tests/test_server_evidence.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
SKILL = os.path.dirname(UTC)
sys.path.insert(0, os.path.join(SKILL, "scripts"))
sys.path.insert(0, os.path.join(SKILL, "shared"))

try:
    import torch
    import triton  # noqa: F401
    GPU = torch.cuda.is_available()
except ImportError:
    torch, GPU = None, False

needs_gpu = unittest.skipUnless(GPU, "needs torch, triton and a GPU")

TOY_GEMM = textwrap.dedent('''\
    import torch
    import triton
    import triton.language as tl


    @triton.jit
    def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :],
                        mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
            b = tl.load(b_ptr + rn[:, None] * K + (k0 + rk)[None, :],
                        mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < K), other=0.0)
            acc += tl.dot(a, tl.trans(b))
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))
    #EXTRA_KERNEL


    def gemm(a, b):
        """toy installed GEMM: Triton, fp32 accumulation, bf16 out"""
        M, K = a.shape
        N = b.shape[0]
        c = torch.empty(M, N, dtype=a.dtype, device=a.device)
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
        #DISPATCH
        _gemm_kernel[grid](a, b, c, M, N, K, BM=16, BN=16, BK=32)
        return c
    ''')

# A decode-only branch that does not compile: wrong only where the server's warmup call goes.
BROKEN_M1 = TOY_GEMM.replace("#EXTRA_KERNEL\n", textwrap.dedent('''

    @triton.jit
    def _gemm_m1_kernel(a_ptr, b_ptr, c_ptr, N, K, BN: tl.constexpr):
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        tl.store(c_ptr + rn, tl.load(a_ptr) + undefined_in_this_branch, mask=rn < N)
    ''')).replace("    #DISPATCH\n", """    if M == 1:
        _gemm_m1_kernel[grid](a, b, c, N, K, BN=16)
        return c
""")

# Uses triton.next_power_of_2 inside the jit body: fine in a bare process, a compile error
# once the framework has rebound it to a plain function.
NEXT_POW2 = TOY_GEMM.replace("acc += tl.dot(a, tl.trans(b))",
                             "acc += tl.dot(a, tl.trans(b)) * (triton.next_power_of_2(BK) // BK)")

FRAMEWORK = textwrap.dedent('''\
    """toy serving framework: rebinds a Triton helper at import, as sglang.srt.utils.common does"""
    import triton


    def next_power_of_2(n):
        return 1 if n == 0 else 2 ** (n - 1).bit_length()


    triton.next_power_of_2 = next_power_of_2
    ''')

SERVER = textwrap.dedent('''\
    import sys
    import toyfw.common  # the framework is imported before any kernel compiles
    import torch
    from toyk.ops import gemm
    b = torch.randn(48, 256, device="cuda", dtype=torch.bfloat16)
    for m in (1, 4, 4, 200):        # warmup (one token), then served traffic
        gemm(torch.randn(m, 256, device="cuda", dtype=torch.bfloat16), b)
    torch.cuda.synchronize()
    ''')


@needs_gpu
class ServerEvidenceTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="utc_srv_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        site = os.path.join(self.tmp, "site")
        for pkg, files in (("toyk", {"ops.py": TOY_GEMM}), ("toyfw", {"common.py": FRAMEWORK})):
            os.makedirs(os.path.join(site, pkg))
            open(os.path.join(site, pkg, "__init__.py"), "w").close()
            for name, text in files.items():
                with open(os.path.join(site, pkg, name), "w") as f:
                    f.write(text)
        self.env = dict(os.environ, PYTHONPATH=site, HARNESS_CACHE_FLUSH_MB="1")
        self.eval = os.path.join(self.tmp, "eval")
        self.serve()
        self.task = os.path.join(self.eval, "tasks", "toy_gemm")
        self.scaffold()

    # ------------------------------------------------------------------ fixtures
    def serve(self):
        """The 'server': compile_watch overlay exactly as probe_server.py writes it."""
        import compile_watch
        import probe_server
        run = os.path.join(self.tmp, "probe")
        watch = os.path.join(run, "compile_watch", "0.profile_cuda_graph")
        os.makedirs(watch)
        overlay = probe_server.watch_overlay(run)
        env = dict(self.env, PYTHONPATH=overlay + ":" + self.env["PYTHONPATH"],
                   COMPILE_WATCH_DIR=watch, COMPILE_WATCH_FRAMES="6")
        script = os.path.join(self.tmp, "server.py")
        with open(script, "w") as f:
            f.write(SERVER)
        p = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True,
                           timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        doc = compile_watch.collect(watch, None, "profile_cuda_graph")
        os.makedirs(os.path.join(self.eval, "evidence"))
        with open(os.path.join(self.eval, "evidence", "compile_variants.json"), "w") as f:
            json.dump(doc, f)
        self.server_doc = doc

    def scaffold(self):
        manifest = {"cases": [
            {"sig": "DECODE_m4", "regime": "DECODE", "axes": {"m": 4}, "effective_calls": 900,
             "timed": True, "random": True},
            {"sig": "PREFILL_m200", "regime": "PREFILL", "axes": {"m": 200}, "effective_calls": 3,
             "timed": True, "random": True}], "replay_sigs": []}
        regimes = {"DECODE": {"cuda_graph": False}, "PREFILL": {"cuda_graph": False}}
        for name, obj in (("m.json", manifest), ("r.json", regimes)):
            with open(os.path.join(self.tmp, name), "w") as f:
                json.dump(obj, f)
        p = self.run_py([os.path.join(UTC, "scaffold_ut.py"), "new", "--family", "gemm",
                         "--task", self.task, "--target", "toyk.ops:gemm", "--bind-module", "toyk.ops",
                         "--bind-file", "kernel_src/ops.py", "--attr", "act_dtype=bfloat16",
                         "--axis", "n=48", "--axis", "k=256",
                         "--manifest", os.path.join(self.tmp, "m.json"),
                         "--phase-regimes", os.path.join(self.tmp, "r.json")])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.write_src(TOY_GEMM)
        cases = os.path.join(self.task, "cases.py")
        with open(cases) as f:
            text = f.read()
        text = text.replace('''    raise NotImplementedError("FILL call(): bind toyk.ops:gemm with the deployed caller's "
                              "arguments and return (out)")''', '''    return _entry()(args["a"], args["b"])''')
        with open(cases, "w") as f:
            f.write(text)
        meta = self.meta()
        meta["workload"]["serving_weight_model"]["evidence_path"] = "run/observed_workload.json"
        meta["ut_contract"]["negative_controls"] = [{
            "leaf": "kernel_src/ops.py", "find": "acc += tl.dot(a, tl.trans(b))",
            "replace": "acc += tl.dot(a, tl.trans(b)) * 1.5",
            "expect": "every output scaled by 1.5, far outside dot_product"}]
        self.write_meta(meta)

    def evidence(self, **decl):
        """build_task_plan's per-task step, on this task."""
        import build_task_plan as B
        task = dict({"label": "toy_gemm", "kernels": ["_gemm_kernel"], "companions": [],
                     "target_callable": "toyk.ops:gemm"}, **decl)
        se = B.server_evidence(self.eval, task, self.task)
        B.stamp_meta(self.task, se, task.get("compile_variant_waivers"))
        return se

    def meta(self):
        with open(os.path.join(self.task, "meta.json")) as f:
            return json.load(f)

    def write_meta(self, meta):
        with open(os.path.join(self.task, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def write_src(self, text):
        with open(os.path.join(self.task, "kernel_src", "ops.py"), "w") as f:
            f.write(text)

    def run_py(self, argv):
        return subprocess.run([sys.executable] + argv, env=self.env, capture_output=True,
                              text=True, timeout=1800, cwd=self.tmp)

    def ut(self, *extra):
        p = self.run_py([os.path.join(self.task, "unittest.py"), "--skip", "timing", "--keep-going",
                         *extra])
        with open(os.path.join(self.task, "run", "ut_report.json")) as f:
            return p, json.load(f)

    # ------------------------------------------------------------------ tests
    def test_server_records_the_warmup_variant_and_its_call(self):
        recs = [r for r in self.server_doc["variants"] if r["name"] == "_gemm_kernel"]
        m = sorted(r["signature"].get("M") for r in recs)
        # M=4 and M=200 are one variant (neither is divisible by 16); M=1 is a constexpr
        self.assertEqual(len(recs), 2, json.dumps(recs, indent=1)[:2000])
        self.assertIn("constexpr", m)
        one = next(r for r in recs if r["signature"].get("M") == "constexpr")
        self.assertEqual(one["constexprs"].get("M"), 1)
        fr = next(f for f in one["frames"] if f["callable"] == "toyk.ops:gemm")
        self.assertEqual(fr["args"]["a"]["shape"], [1, 256])
        self.assertEqual(self.server_doc["env"]["triton.next_power_of_2"], "function:toyfw.common")

    def test_without_the_server_call_the_m1_variant_is_reported_missing(self):
        self.evidence(server_preimports=["toyfw.common"])
        meta = self.meta()
        os.remove(os.path.join(self.task, meta["server_evidence"]["server_calls"]))
        del meta["server_evidence"]["server_calls"]
        self.write_meta(meta)
        p, rep = self.ut()
        cv = rep["gates"]["compile_variants"]
        self.assertEqual(cv["status"], "fail", json.dumps(cv, indent=1)[:3000])
        self.assertEqual(len(cv["missing"]), 1, cv)
        diff = {d["field"]: d for d in cv["missing"][0]["differs"]}
        self.assertEqual(diff["signature[M]"]["server"], "constexpr")
        self.assertEqual(diff["constexprs[M]"]["server"], 1)
        self.assertEqual(rep["gates"]["correctness"]["status"], "pass")

    def test_a_waiver_with_a_reason_accepts_the_missing_variant(self):
        self.evidence(server_preimports=["toyfw.common"])
        meta = self.meta()
        del meta["server_evidence"]["server_calls"]
        one = next(r for r in self.server_doc["variants"]
                   if r["name"] == "_gemm_kernel" and r["signature"]["M"] == "constexpr")
        meta["ut_contract"]["compile_variant_waivers"] = [{"id": one["id"], "why": "test"}]
        self.write_meta(meta)
        p, rep = self.ut()
        self.assertEqual(rep["gates"]["compile_variants"]["status"], "pass",
                         json.dumps(rep["gates"]["compile_variants"], indent=1)[:2000])

    def test_replayed_server_calls_cover_every_variant_and_the_ut_passes(self):
        se = self.evidence(server_preimports=["toyfw.common"])
        self.assertEqual(se["server_evidence"]["calls"], 2)
        p, rep = self.ut()
        g = rep["gates"]
        self.assertEqual(g["server_calls"]["status"], "pass", json.dumps(g["server_calls"])[:3000])
        self.assertEqual(g["server_calls"]["replayed_by_baseline"], g["server_calls"]["calls"])
        self.assertEqual(g["compile_variants"]["status"], "pass",
                         json.dumps(g["compile_variants"], indent=1)[:3000])
        bad = {k: v["status"] for k, v in g.items() if v["status"] not in ("pass", "not_required",
                                                                          "skipped")}
        self.assertFalse(bad, json.dumps(g, indent=1, default=str)[:4000])

    def test_a_candidate_broken_only_on_the_warmup_path_fails_server_calls(self):
        self.evidence(server_preimports=["toyfw.common"])
        self.write_src(BROKEN_M1)
        p, rep = self.ut()
        g = rep["gates"]
        self.assertEqual(g["correctness"]["status"], "pass")           # every M the UT covers
        self.assertEqual(g["server_calls"]["status"], "fail", json.dumps(g["server_calls"])[:3000])
        bad = [r for r in g["server_calls"]["rows"] if not r["ok"]]
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("candidate", bad[0]["why"])

    def test_without_preimports_the_env_differs_and_with_them_next_pow2_fails(self):
        self.evidence(server_preimports=[])
        self.write_src(NEXT_POW2)
        p, rep = self.ut()
        g = rep["gates"]
        self.assertEqual(g["correctness"]["status"], "pass")           # the bare UT is green
        env = {d["attr"]: d for d in g["compile_variants"]["env_differs"]}
        self.assertIn("triton.next_power_of_2", env, g["compile_variants"])
        self.assertEqual(g["compile_variants"]["status"], "fail")
        self.evidence(server_preimports=["toyfw.common"])
        p, rep = self.ut()
        self.assertEqual(rep["gates"]["correctness"]["status"], "fail")   # as in the server
        self.assertFalse(rep["gates"]["compile_variants"]["env_differs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
