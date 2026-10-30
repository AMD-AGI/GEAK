"""serving_probe.py against fake HTTP servers: a router that admits requests and an engine
whose scheduler log and profiler are separate, as on a PD deployment without /metrics."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import observed_workload  # noqa: E402
import serving_probe  # noqa: E402

SWA_DECODE = ('[2026-10-27 03:15:48 TP0] Decode batch, #running-req: 3, #full token: 210944, '
              'full token usage: 0.02, #swa token: 512, swa token usage: 0.00, accept len: 2.52, '
              'accept rate: 0.51, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0, '
              '#retracted-req: 0, cuda graph: True, gen throughput (token/s): 11.54, #queue-req: 0')


def decode_line(running, rank=0):
    return ('[2026-10-01 00:00:00 TP%d] Decode batch, #running-req: %d, #token: 64, '
            'cuda graph: True, gen throughput (token/s): 50.0, #queue-req: 0\n' % (rank, running))


class FakeDeployment:
    """`running` requests are resident while /generate sleeps; a thread logs them like the
    scheduler. The engine writes one trace per rank when /stop_profile closes the window."""

    def __init__(self, log, ranks, request_s=8.0, bs_cap=None):
        self.log, self.ranks, self.request_s, self.bs_cap = log, ranks, request_s, bs_cap
        self.running, self.lock, self.calls, self.window = 0, threading.Lock(), [], None
        self.stop = threading.Event()
        self.router = self._serve(self._router_handler())
        self.engine = self._serve(self._engine_handler())
        threading.Thread(target=self._logger, daemon=True).start()

    def url(self, server):
        return 'http://127.0.0.1:%d' % server.server_address[1]

    def close(self):
        self.stop.set()
        for server in (self.router, self.engine):
            server.shutdown()
            server.server_close()

    def _serve(self, handler):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _logger(self):
        while not self.stop.wait(0.2):
            with self.lock:
                running = self.running if self.bs_cap is None else min(self.running, self.bs_cap)
            if running:
                with open(self.log, 'a') as fh:
                    fh.write(decode_line(running, rank=1))
                    fh.write(decode_line(running))

    def _reply(self, handler, text='{}'):
        handler.send_response(200)
        handler.end_headers()
        handler.wfile.write(text.encode())

    def _router_handler(self):
        dep = self

        class Router(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length'] or 0))
                dep.calls.append(('router', self.path, json.loads(body or b'{}')))
                if self.path == '/generate':
                    with dep.lock:
                        dep.running += 1
                    time.sleep(dep.request_s)
                    with dep.lock:
                        dep.running -= 1
                dep._reply(self)
        return Router

    def _engine_handler(self):
        dep = self

        class Engine(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'] or 0)) or b'{}')
                dep.calls.append(('engine', self.path, body))
                if self.path == '/start_profile':
                    with dep.lock:
                        dep.window = (body['output_dir'], dep.running)
                elif self.path == '/stop_profile' and dep.window:
                    threading.Thread(target=dep._export, args=dep.window, daemon=True).start()
                dep._reply(self)
        return Engine

    def _export(self, out_dir, bs):
        os.makedirs(out_dir, exist_ok=True)
        events = [{'cat': 'gpu_user_annotation', 'name': 'step[DECODE bs=%d]' % bs, 'ts': 0, 'dur': 10},
                  {'cat': 'kernel', 'name': 'k', 'ts': 1, 'dur': 5}]
        for rank in range(self.ranks):
            time.sleep(0.5)
            with open(os.path.join(out_dir, 'prof-TP-%d.trace.json' % rank), 'w') as fh:
                json.dump({'traceEvents': events}, fh)


class ProbeOverLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='probe ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / 'decode.log'
        self.log.write_text(decode_line(2))          # before the probe: must not arm it
        self.points = self.root / 'points.json'
        self.points.write_text(json.dumps({'points': [
            {'point_id': 'de_2', 'phase': 'DECODE', 'batch_size': 2, 'context_tokens': 8}]}))

    def probe(self, dep, *extra):
        out = self.root / 'de_2.journal.json'
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / 'serving_probe.py'), '--points', self.points,
             '--point-id', 'de_2', '--base-url', dep.url(dep.router),
             '--profile-url', dep.url(dep.engine), '--occupancy-log', self.log,
             '--trace-dir', self.root / 'traces', '--expect-traces', '2', '--trace-settle', '1',
             '--trace-quiet', '6', '--steady-timeout', '8', '--decode-tokens', '4',
             '--record-shapes', '--with-stack', 'false', '--out', out, *map(str, extra)],
            capture_output=True, text=True, timeout=120)
        return proc, json.loads(out.read_text())

    def events(self, result, name):
        return [e for e in result['journal'] if e['event'] == name]

    def test_arms_on_the_logged_batch_and_profiles_the_engine_not_the_router(self):
        dep = FakeDeployment(self.log, ranks=2)
        self.addCleanup(dep.close)
        proc, result = self.probe(dep)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(result['status'], 'ok', result['verify_reason'])
        self.assertEqual(self.events(result, 'steady_state_confirmed')[0]['observed_bs'], 2)
        self.assertEqual(len(self.events(result, 'traces_settled')), 1)
        paths = {(who, path) for who, path, _ in dep.calls}
        self.assertIn(('engine', '/start_profile'), paths)
        self.assertIn(('engine', '/stop_profile'), paths)
        self.assertNotIn(('router', '/start_profile'), paths)
        start = next(body for who, path, body in dep.calls if path == '/start_profile')
        self.assertEqual((start['record_shapes'], start['with_stack']), (True, False))

    def test_a_batch_the_scheduler_never_reaches_is_not_armed(self):
        dep = FakeDeployment(self.log, ranks=2, request_s=12, bs_cap=1)
        self.addCleanup(dep.close)
        proc, result = self.probe(dep)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(self.events(result, 'steady_state_NOT_confirmed'))
        self.assertNotIn('/start_profile', [path for _w, path, _b in dep.calls])

    def test_a_rank_that_never_exports_fails_the_point(self):
        dep = FakeDeployment(self.log, ranks=1)
        self.addCleanup(dep.close)
        proc, result = self.probe(dep)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.events(result, 'traces_INCOMPLETE')[0]['non_empty'], 1)


class DrivePrefillTest(unittest.TestCase):
    def test_sends_the_batch_as_its_sequences_not_running_copies_of_it(self):
        sent = []
        real = serving_probe.generate
        serving_probe.generate = lambda _base, ids, max_new, *_a: sent.append((len(ids), max_new))
        self.addCleanup(setattr, serving_probe, 'generate', real)
        args = type('Args', (), {'seed': 1, 'request_timeout': 5})()
        serving_probe.drive_prefill('http://x', {'new_tokens': 1024, 'new_seq': 2, 'running': 7}, args, [])
        self.assertEqual(sorted(sent), [(512, 1), (512, 1)])
        sent.clear()
        serving_probe.drive_prefill('http://x', {'new_tokens': 300, 'running': 3}, args, [])
        self.assertEqual(sent, [(300, 1)])


class PausedPrefillTest(unittest.TestCase):
    """The point's engine is paused while its sequences are sent, then resumed once."""

    def engine(self, missing):
        calls = self.calls = []

        class Engine(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length'] or 0))
                calls.append(self.path)
                self.send_response(404 if self.path in missing else 200)
                self.end_headers()
                self.wfile.write(b'{}')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Engine)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:%d' % server.server_address[1]

    def drive(self, missing, point):
        url = self.engine(missing)
        real = serving_probe.generate
        serving_probe.generate = lambda _base, ids, *_a: self.calls.append('generate %d' % len(ids))
        self.addCleanup(setattr, serving_probe, 'generate', real)
        args = type('Args', (), {'seed': 1, 'request_timeout': 5, 'gather_s': 0.1})()
        journal = []
        serving_probe.drive_prefill('http://router', point, args, journal, url)
        return [e['event'] for e in journal if e['event'].startswith(('scheduler', 'pause'))]

    def test_sglang_is_paused_in_place_and_continued_after_every_sequence_is_sent(self):
        events = self.drive(set(), {'new_tokens': 1536, 'new_seq': 3})
        self.assertEqual(self.calls, ['/pause_generation'] + ['generate 512'] * 3 + ['/continue_generation'])
        self.assertEqual(events, ['scheduler_paused', 'scheduler_resumed'])

    def test_vllm_is_paused_with_keep_when_the_sglang_path_is_absent(self):
        self.drive({'/pause_generation'}, {'new_tokens': 1024, 'new_seq': 2})
        self.assertEqual(self.calls, ['/pause_generation', '/pause?mode=keep',
                                      'generate 512', 'generate 512', '/resume'])

    def test_no_pause_api_sends_unpaused_and_says_so(self):
        events = self.drive({'/pause_generation', '/pause?mode=keep'}, {'new_tokens': 1024, 'new_seq': 2})
        self.assertEqual(self.calls[2:], ['generate 512', 'generate 512'])
        self.assertEqual(events, ['pause_unavailable'])

    def test_sequences_add_up_to_new_tokens_when_it_does_not_divide(self):
        self.drive(set(), {'new_tokens': 6229, 'new_seq': 3})
        self.assertEqual(sorted(c for c in self.calls if c.startswith('generate')),
                         ['generate 2076', 'generate 2076', 'generate 2077'])

    def test_one_sequence_needs_no_pause(self):
        self.drive(set(), {'new_tokens': 512, 'new_seq': 1})
        self.assertEqual(self.calls, ['generate 512'])


class VerifyTraceTest(unittest.TestCase):
    def verify(self, *contexts):
        events = [{'cat': 'kernel', 'name': 'gemm', 'ts': 1, 'dur': 5}]
        events += [{'cat': 'gpu_user_annotation', 'ts': 10 * i, 'dur': 5,
                    'name': 'execute_model context_1(%d) generation_0(0)' % n}
                   for i, n in enumerate(contexts)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'TP-0.trace.json')
            Path(path).write_text(json.dumps({'traceEvents': events}))
            return serving_probe.verify_trace([{'path': path}], 'PREFILL', 'new_tokens', 1024, [])

    def test_the_planned_step_alone_is_ok(self):
        self.assertIs(self.verify(1024)['verified'], True)

    def test_a_stray_step_in_the_window_makes_the_point_partial(self):
        res = self.verify(1024, 1)
        self.assertEqual(res['verified'], 'partial')
        self.assertIn('[1]', res['reason'])


class LogOccupancyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='occupancy ')
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / 'decode.log'

    def test_reads_only_rows_written_after_it_started(self):
        self.log.write_text(decode_line(7) + decode_line(5)[:30])
        occupancy = serving_probe.LogOccupancy(str(self.log))
        self.assertEqual(occupancy(), (None, None, None))
        with open(self.log, 'a') as fh:
            fh.write(decode_line(5)[30:] + decode_line(3, rank=1) + decode_line(2))
        self.assertEqual(occupancy(), (2, 50.0, None))
        self.assertEqual(occupancy(), (2, 50.0, None))

    def test_pd_engine_lines_parse(self):
        decode = ('[2026-10-21 07:43:45 TP0] Decode batch, #running-req: 4, #token: 512, '
                  'token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0, '
                  '#retracted-req: 0, cuda graph: True, gen throughput (token/s): 1.48, #queue-req: 0')
        prefill = ('[2026-10-21 07:43:08 TP0] Prefill batch, #new-seq: 1, #new-token: 4, '
                   '#cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, '
                   '#pending-token: 0, #bootstrap-req: 0, #inflight-req: 1, #optimistic-req: 0, '
                   'cuda graph: False, input throughput (token/s): 0.49')
        rec = observed_workload.parse_line(decode)
        self.assertEqual((rec['kind'], rec['running'], rec['cuda_graph']), ('decode', 4, True))
        rec = observed_workload.parse_line(prefill)
        self.assertEqual((rec['kind'], rec['new_token'], rec['cuda_graph']), ('prefill', 4, False))

    def test_a_tp1_engine_logs_without_a_rank_tag_and_is_rank_0(self):
        rec = observed_workload.parse_line(
            '[2026-10-30 04:22:19] Decode batch, #running-req: 4, #token: 851, token usage: 0.00, '
            'cuda graph: True, gen throughput (token/s): 30.40, #queue-req: 0')
        self.assertEqual((rec['rank'], rec['running']), (0, 4))

    def test_the_pd_swa_dialect_parses_for_arming_and_for_the_weights(self):
        rec = observed_workload.parse_line(SWA_DECODE)
        self.assertEqual((rec['kind'], rec['running'], rec['gen_tps'], rec['token']),
                         ('decode', 3, 11.54, 210944))
        self.log.write_text('')
        occupancy = serving_probe.LogOccupancy(str(self.log))
        with open(self.log, 'a') as fh:
            fh.write(SWA_DECODE + '\n')
        self.assertEqual(occupancy()[:2], (3, 11.54))


if __name__ == '__main__':
    unittest.main()
