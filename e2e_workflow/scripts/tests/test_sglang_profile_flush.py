"""sglang adapter_profile_window must not return while a trace is still being written.

It used to return 2 s after the first new trace file appeared; with one ~10 MB .gz per rank the
caller then tore the server down mid-write and every trace was truncated.
"""
import gzip
import os
import subprocess
import textwrap
import unittest
import tempfile

ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "sglang.sh")


def _payload():
    return os.urandom(200_000)  # incompressible, so the .gz grows chunk by chunk


class ProfileFlushTest(unittest.TestCase):
    def run_window(self, writer):
        d = tempfile.mkdtemp(prefix="sgl_prof_")
        script = textwrap.dedent(f"""
            source {ADAPTER}
            PROFILE_DIR={d}; BASE_URL=http://127.0.0.1:1; PROFILE_NUM_STEPS=4
            PROFILE_FLUSH_POLL_S=0.5; PROFILE_FLUSH_POLLS=40; PROFILE_WINDOW_TIMEOUT=30
            curl() {{ return 0; }}                  # stands in for the server's /start_profile
            ( {writer} ) &
            adapter_profile_window; echo "rc=$?"
            for f in {d}/*.gz; do gzip -t "$f" && echo "whole $(basename $f)"; done
        """)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60).stdout

    def test_waits_for_slow_multi_rank_writes(self):
        # Two ranks, each written in 8 chunks 0.7 s apart; rank 1 starts 3 s late, so the last write
        # lands ~8.6 s in -- well after the old 'first file + 2 s' return.
        py = ("import gzip,os,sys,time;"
              "p=sys.argv[1];time.sleep(float(sys.argv[2]));"
              "f=gzip.open(p,'wb')\n"
              "for _ in range(8):\n f.write(os.urandom(200000)); f.flush(); time.sleep(0.7)\n"
              "f.close()")
        writer = (f"python3 -c \"{py}\" $PROFILE_DIR/x-TP-0.trace.json.gz 0 & "
                  f"python3 -c \"{py}\" $PROFILE_DIR/x-TP-1.trace.json.gz 3 & wait")
        out = self.run_window(writer)
        self.assertIn("rc=0", out)
        self.assertIn("whole x-TP-0.trace.json.gz", out)
        self.assertIn("whole x-TP-1.trace.json.gz", out)

    def test_gives_up_on_a_trace_that_never_completes(self):
        writer = ("python3 -c \"import gzip,os,sys; f=open(sys.argv[1],'wb'); "
                  "f.write(gzip.compress(os.urandom(100000))[:5000])\" $PROFILE_DIR/y-TP-0.trace.json.gz")
        out = self.run_window(writer)
        self.assertIn("rc=1", out)


if __name__ == "__main__":
    unittest.main()
