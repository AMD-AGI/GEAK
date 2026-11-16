"""Real CPU tensor and subprocess regressions for bounded UT validation."""
import copy
import importlib.util
import json
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import test_ut_contracts as fixtures

U, torch = fixtures.U, fixtures.torch
UTC = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    previous = sys.path[:]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path[:] = previous
    return m


D = module('tensor_lifecycle_driver', UTC / 'ut_driver.py')
H = module('tensor_lifecycle_harness', UTC.parent / 'shared/harness_lib.py')


@unittest.skipIf(torch is None, 'requires CPU torch')
class TensorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.doc = U.freeze_task_contract(U.load_family('gemm'),
            attrs={'act_dtype': 'bfloat16'}, const_axes={'n': 16, 'k': 32})

    def build(self, cache, m=4, seed=7, doc=None):
        return U.build_inputs(doc or self.doc, {'m': m}, torch.Generator().manual_seed(seed),
                              'cpu', with_logical=False, weight_cache=cache)[0]

    def test_readonly_weights_reuse_across_shapes_and_keep_private_storage(self):
        cache = U.WeightCache(1 << 20)
        a = self.build(cache)
        original = a['b'].clone()
        a['b'].zero_()  # represents any mutation, including writes bypassing _version
        b = self.build(cache, m=9)
        self.assertTrue(torch.equal(b['b'], original))
        self.assertNotEqual(a['b'].data_ptr(), b['b'].data_ptr())
        self.assertEqual((cache.misses, cache.hits), (1, 1))
        self.assertNotEqual(a['a'].shape, b['a'].shape)
        self.assertFalse(torch.equal(b['b'], self.build(cache, seed=8)['b']))

    def test_cache_budget_never_changes_random_samples(self):
        cache, uncached = U.WeightCache(1024), U.WeightCache(0)
        for m, seed in [(4, 1), (8, 1), (4, 2), (4, 1)]:
            a, b = self.build(cache, m, seed), self.build(uncached, m, seed)
            self.assertTrue(U.inputs_equal(a, b))
            self.assertLessEqual(cache.bytes, cache.max_bytes)
        self.assertEqual(uncached.bytes, 0)
        self.assertGreater(cache.misses, 1)

    def test_writable_state_and_hooks_are_not_cached(self):
        for field, value in [('writable', True), ('lifetime', 'state'),
                             ('layout', {'hook': 'shuffle'}), ('dist', {'kind': 'hook', 'hook': 'quant'})]:
            spec = copy.deepcopy(self.doc['inputs']['b'])
            spec[field] = value
            self.assertFalse(U.cacheable_weight(spec))
        doc = copy.deepcopy(self.doc)
        doc['inputs']['b']['writable'] = True
        cache = U.WeightCache()
        self.build(cache, doc=doc)
        self.assertEqual((cache.hits, cache.misses), (0, 0))

    def test_distribution_changes_invalidate_reuse(self):
        cache = U.WeightCache()
        a = self.build(cache)
        doc = copy.deepcopy(self.doc)
        doc['inputs']['b']['dist'] = {'kind': 'constant', 'value': 5}
        b = self.build(cache, doc=doc)
        self.assertFalse(torch.equal(a['b'], b['b']))
        self.assertTrue(bool((b['b'] == 5).all()))
        self.assertEqual(cache.misses, 2)

    def test_input_gate_constructs_twice_and_keeps_all_checks(self):
        count = []
        def make(spec, gen):
            count.append(1)
            args, _, sym = U.build_inputs(self.doc, spec['axes'], gen, 'cpu', with_logical=False)
            return dict(args, _sym=sym)
        task = SimpleNamespace(U=U, doc=self.doc,
            random_specs=lambda: [{'sig': 'a', 'axes': {'m': 4}}],
            cases=SimpleNamespace(case_seed=lambda _: 3, make_args=make))
        self.assertEqual(D.gate_inputs(task, None)['status'], 'pass')
        self.assertEqual(len(count), 2)

    def test_exact_comparison_preserves_nan_and_signed_zero_bits(self):
        a = torch.tensor([float('nan'), 0.0])
        b = a.clone()
        self.assertTrue(U.inputs_equal(a, b))
        b[1] = -0.0
        self.assertFalse(U.inputs_equal(a, b))
        b = a.clone()
        b.view(torch.int32)[0] ^= 1
        self.assertFalse(U.inputs_equal(a, b))

    def test_reproducibility_detects_builder_reusing_mutable_buffers(self):
        shared = torch.zeros(4)
        def build(_):
            shared.add_(1)
            return {'x': shared}
        self.assertFalse(U.check_reproducible(build, 0, 'cpu')[0])

    def test_isolation_reuses_first_execution_and_detects_mutations(self):
        calls = []
        def call(args):
            calls.append(1)
            return args['x'].clone() * 2
        args, other = {'x': torch.ones(3)}, {'x': torch.zeros(3)}
        ok, _, first = U.inputs_unmodified(call, args)
        self.assertTrue(ok)
        self.assertTrue(H.assert_independent_outputs(call, args, other, first_output=first)[0])
        self.assertEqual(len(calls), 2)
        self.assertFalse(U.inputs_unmodified(lambda a: a['x'].data.zero_(), args)[0])
        persistent = torch.zeros(3)
        def shared(a):
            persistent.copy_(a['x'])
            return persistent
        first = shared(args)
        self.assertFalse(H.assert_independent_outputs(shared, args, other, first_output=first)[0])

    def test_missing_shard_outputs_fail_and_temporary_files_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_leg(*args, out, **kwargs):
                torch.save({}, out)
            task = SimpleNamespace(U=U, dir=tmp, meta={}, gpus=None, sig_filter=None,
                random_specs=lambda: [{"sig": "case"}],
                h=SimpleNamespace(gpu_env=lambda _: {}, _run_leg=fake_leg))
            job = {"overlay": tmp, "path": str(Path(tmp)/'out.pt'), "seed": 0, "draws": 1}
            with self.assertRaisesRegex(RuntimeError, 'case/draw'):
                D._oracles(task, {'scratch': tmp}, [job])
            self.assertFalse(list(Path(tmp).glob('*.pt*')))

    def test_worker_crash_is_reported_and_process_is_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'leg_runner.py').write_text(
                "import sys\nsys.stdin.readline()\nraise RuntimeError('fixture worker crash')\n")
            task = SimpleNamespace(dir=tmp, meta={}, mc={'oracle_timeout': 5},
                                   h=SimpleNamespace(gpu_env=lambda _: {}))
            worker = D.OracleWorker(task, {'scratch': tmp}, {'overlay': tmp}, None)
            try:
                with self.assertRaisesRegex(RuntimeError, 'fixture worker crash'):
                    worker.run({'seed': 0, 'draws': 1}, ['a'], str(Path(tmp)/'out.pt'))
            finally:
                worker.close()
            self.assertIsNotNone(worker.proc.poll())



# A worker-protocol leg_runner: each blob value encodes (process, oracle path, STAGE env) so a
# test can tell which process and which path produced it.
_PATH_WORKER = """import json, os, sys, torch
for line in sys.stdin:
    r = json.loads(line)
    if r.get("path", "call") not in ("call", "split"):
        print("UT_HARNESS_INCOMPLETE: oracle path %r is not defined" % r["path"], flush=True)
        sys.exit(3)
    tag = "%d|%s|%s" % (os.getpid(), r.get("path", "call"), os.environ.get("STAGE", ""))
    blob = {"%s|%d" % (s, n): tag for s in r["sigs"] for n in range(r["draws"])}
    torch.save(blob, r["out"])
    print("GEAK_ORACLE_BATCH " + json.dumps({"out": r["out"], "n": len(blob)}), flush=True)
"""


@unittest.skipIf(torch is None, 'requires CPU torch')
class OraclePathWorkerTests(unittest.TestCase):
    """DESIGN 1.4: jobs carry oracle_path and env; the worker key separates both."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.tmp, True)
        Path(self.tmp, 'leg_runner.py').write_text(_PATH_WORKER)
        self.task = SimpleNamespace(dir=self.tmp, meta={}, mc={'oracle_timeout': 60}, gpus=None,
            sig_filter=None, U=U, random_specs=lambda: [{'sig': 'r8'}, {'sig': 'r16'}],
            h=SimpleNamespace(gpu_env=lambda _: {}))
        self.state = {'scratch': self.tmp}

    def job(self, name, **kw):
        return dict(overlay=self.tmp, path=str(Path(self.tmp, name)), seed=0, draws=1, **kw)

    def batch(self, jobs):
        return D._oracles(self.task, self.state, jobs)

    def test_call_and_split_under_one_overlay_produce_different_blobs(self):
        with D.oracle_workers(self.task, self.state):
            call, split = self.batch([self.job('c.pt'), self.job('s.pt', oracle_path='split')])
        self.assertEqual(sorted(call), sorted(split))
        self.assertTrue(all(v.split('|')[1] == 'call' for v in call.values()))
        self.assertTrue(all(v.split('|')[1] == 'split' for v in split.values()))
        self.assertEqual(self.state['oracle_worker_starts'], 2)

    def test_same_env_reuses_the_worker_and_a_different_env_starts_a_new_one(self):
        with D.oracle_workers(self.task, self.state):
            a, = self.batch([self.job('d.pt', env={'STAGE': 'dpre'})])
            again, = self.batch([self.job('d.pt', env={'STAGE': 'dpre'})])
            self.assertEqual(self.state['oracle_worker_starts'], 1)
            other, = self.batch([self.job('d.pt', env={'STAGE': 'dmlp'})])
            self.assertEqual(self.state['oracle_worker_starts'], 2)
        pid = lambda blob: {v.split('|')[0] for v in blob.values()}
        self.assertEqual(pid(a), pid(again))
        self.assertNotEqual(pid(a), pid(other))
        self.assertEqual({v.split('|')[2] for v in other.values()}, {'dmlp'})

    def test_env_key_ignores_insertion_order_and_rejects_non_strings(self):
        k1 = D.oracle_worker_key(self.job('x.pt', env={'A': '1', 'B': '2'}), 0)
        k2 = D.oracle_worker_key(self.job('x.pt', env={'B': '2', 'A': '1'}), 0)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, D.oracle_worker_key(self.job('x.pt', env={'A': '1'}), 0))
        self.assertEqual(D.oracle_worker_key(self.job('x.pt'), 0),
                         D.oracle_worker_key(self.job('x.pt', oracle_path='call', env={}), 0))
        with self.assertRaises(ValueError):
            D.oracle_worker_key(self.job('x.pt', env={'A': 1}), 0)

    def test_a_missing_oracle_path_surfaces_the_sentinel(self):
        with self.assertRaisesRegex(RuntimeError, D.SENTINEL):
            with D.oracle_workers(self.task, self.state):
                self.batch([self.job('n.pt', oracle_path='native')])

    def test_one_shot_legs_get_the_path_and_env(self):
        seen = []
        def leg(task_dir, overlay, mode, *, out, env_extra, oracle_path=None, **kw):
            seen.append((oracle_path, env_extra.get('STAGE')))
            torch.save({'%s|0' % s: 0 for s in ('r8', 'r16')}, out)
        self.task.h = SimpleNamespace(gpu_env=lambda _: {}, _run_leg=leg)
        D._oracle(self.task, self.tmp, str(Path(self.tmp, 'o.pt')), 0, 1, self.state,
                  oracle_path='split', env={'STAGE': 'dattn'})
        self.assertEqual(seen, [('split', 'dattn')])


@unittest.skipIf(torch is None, 'requires CPU torch')
class BoundedOracleTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.EndToEndTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def run_correctness(self, batch_size):
        meta = json.loads(Path(self.f.meta_path).read_text())
        meta['ut_contract']['oracle_batch_cases'] = batch_size
        self.f.write_meta(meta)
        return self.f.ut('--validation-mode', 'correctness', '--gates', 'correctness')

    def test_batches_preserve_every_case_draw_and_reuse_isolated_processes(self):
        p, small = self.run_correctness(1)
        self.assertEqual(small['gates']['correctness']['status'], 'pass', p.stdout + p.stderr)
        p, large = self.run_correctness(32)
        a, b = small['gates']['correctness'], large['gates']['correctness']
        self.assertEqual(a['cases'], b['cases'])
        self.assertEqual((a['batches'], b['batches']), (2, 1))
        self.assertEqual(len(a['cases']), 6)
        self.assertEqual(small['data_handling']['oracle_worker_starts'],
                         large['data_handling']['oracle_worker_starts'])
        self.assertLess(small['data_handling']['peak_oracle_files_bytes'],
                        large['data_handling']['peak_oracle_files_bytes'])
        self.assertTrue(all(w['hits'] > 0 for w in small['data_handling']['weight_cache_workers']))
        self.assertFalse(list(Path(self.f.task).rglob('*.pt')))

    def test_failure_in_a_later_batch_is_not_dropped(self):
        source = Path(self.f.task) / 'kernel_src/ops.py'
        text = source.read_text()
        source.write_text(text.replace('a.float() @ b.float().T',
            'a.float() @ b.float().T + (1 if a.shape[0] > 10 else 0)'))
        p, report = self.run_correctness(1)
        gate = report['gates']['correctness']
        self.assertEqual(gate['status'], 'fail', p.stdout + p.stderr)
        self.assertEqual(len(gate['cases']), 6)
        self.assertTrue(any(not r['ok'] and 'PREFILL' in r['case'] for r in gate['cases']))

    def test_oversized_case_is_reported_and_never_skipped(self):
        task = D.Task(self.f.task)
        task.mc['oracle_batch_bytes'] = 1
        batches = list(D.oracle_batches(task, 3, 3))
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(sigs) == 1 and size > 1 for sigs, size in batches))
        original = task.sig_filter
        with self.assertRaises(RuntimeError):
            with D.case_batch(task, [batches[0][0][0]]):
                raise RuntimeError('interrupted batch')
        self.assertIs(task.sig_filter, original)


if __name__ == '__main__':
    unittest.main()
