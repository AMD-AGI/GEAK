"""adapter_profile_window must not hand back a trace that is still being written.

A 16-step stack trace of Qwen3.5 NEXTN was cut off mid-write (no gzip end
marker) because the window returned 2 s after the DECODE file first appeared
and the server was torn down while sglang was still flushing it.
"""
import gzip
import os
import subprocess
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.path.join(SCRIPTS, "adapters", "sglang.sh")


def _run(profile_dir, timeout):
    cmd = ('source "%s"; PROFILE_FLUSH_POLL=0.2 _sgl_wait_traces_complete "%s" %s'
           % (ADAPTER, profile_dir, timeout))
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)


class ProfileFlushWaitTest(unittest.TestCase):
    def test_complete_traces_pass_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stage in ("EXTEND", "DECODE"):
                with gzip.open(os.path.join(tmp, "1-TP-0-%s.trace.json.gz" % stage), "wt") as fh:
                    fh.write('{"traceEvents": []}')
            # an empty merged file must not block the wait
            open(os.path.join(tmp, "merged-1.trace.json.gz"), "w").close()
            result = _run(tmp, 5)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_truncated_trace_waits_then_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "1-TP-0-DECODE.trace.json.gz")
            with gzip.open(path, "wt") as fh:
                fh.write('{"traceEvents": [' + '{"a": 1},' * 5000 + "{}]}")
            data = open(path, "rb").read()
            open(path, "wb").write(data[: len(data) // 2])
            result = _run(tmp, 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
