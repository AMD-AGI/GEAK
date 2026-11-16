"""Stop waits for live TP workers, not unreaped zombies or only the group leader."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_server


class ProbeStopTests(unittest.TestCase):
    def test_proc_groups_ignore_zombies_and_handle_parentheses_in_names(self):
        with tempfile.TemporaryDirectory() as d:
            for pid, state, group in ((12, 'Z', 12), (13, 'S', 12), (14, 'S', 14)):
                p = Path(d) / str(pid)
                p.mkdir()
                (p / 'stat').write_text('%d (worker (odd)) %s 1 %d 0' % (pid, state, group))
            self.assertTrue(probe_server.process_group_active(12, d))
            (Path(d) / '13/stat').write_text('13 (worker) Z 1 12 0')
            self.assertFalse(probe_server.process_group_active(12, d))
            self.assertTrue(probe_server.process_group_active(14, d))

    def test_dead_leader_does_not_skip_live_workers(self):
        with tempfile.TemporaryDirectory() as d:
            s = probe_server.Server({}, d)
            s.proc = Mock(pid=123)
            s.proc.poll.return_value = 0
            with patch.object(probe_server, 'process_group_active',
                              side_effect=[True, False, False, False]), \
                 patch.object(probe_server.os, 'killpg') as kill:
                s.stop(wait=0)
            kill.assert_called_once_with(123, probe_server.signal.SIGTERM)

    def test_only_zombies_requires_no_signal_or_wait(self):
        with tempfile.TemporaryDirectory() as d:
            s = probe_server.Server({}, d)
            s.proc = Mock(pid=123)
            with patch.object(probe_server, 'process_group_active', return_value=False), \
                 patch.object(probe_server.os, 'killpg') as kill, \
                 patch.object(probe_server.time, 'sleep') as sleep:
                s.stop()
            kill.assert_not_called()
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
