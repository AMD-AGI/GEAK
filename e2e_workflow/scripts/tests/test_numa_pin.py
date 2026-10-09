import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMA_PIN = os.path.join(SCRIPTS, "numa_pin.sh")
BASH = shutil.which("bash")

TOPO = """======================= Numa Nodes =======================
GPU[0]\t\t: (Topology) Numa Node: 0
GPU[0]\t\t: (Topology) Numa Affinity: 0
GPU[1]\t\t: (Topology) Numa Node: 0
GPU[4]\t\t: (Topology) Numa Node: 1
"""


@unittest.skipIf(BASH is None, "bash is required")
class NumaPinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        topo = os.path.join(self.tmp, "topo.txt")
        with open(topo, "w") as fh:
            fh.write(TOPO)
        for node, cpus in (("0", "0-55,112-167"), ("1", "56-111,168-223")):
            os.makedirs(os.path.join(self.tmp, "nodes", "node" + node))
            with open(os.path.join(self.tmp, "nodes", "node" + node, "cpulist"), "w") as fh:
                fh.write(cpus + "\n")
        self.env = dict(os.environ, GEAK_TOPO_NUMA_CMD="cat " + topo,
                        GEAK_NODE_SYSFS=os.path.join(self.tmp, "nodes"))

    def _cpus(self, gpus, **env):
        proc = subprocess.run(
            [BASH, "-c", 'source "%s"; geak_numa_cpus "%s"' % (NUMA_PIN, gpus)],
            env=dict(self.env, **env), capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()

    def test_single_gpu_maps_to_its_node(self):
        self.assertEqual(self._cpus("0"), (0, "0-55,112-167"))
        self.assertEqual(self._cpus("4"), (0, "56-111,168-223"))

    def test_gpus_on_one_node_share_it(self):
        self.assertEqual(self._cpus("0,1"), (0, "0-55,112-167"))

    def test_gpus_spanning_nodes_are_not_pinned(self):
        code, out = self._cpus("0,4")
        self.assertNotEqual(code, 0)
        self.assertEqual(out, "")

    def test_unknown_gpu_or_failed_probe_is_not_pinned(self):
        self.assertNotEqual(self._cpus("7")[0], 0)
        self.assertNotEqual(self._cpus("0", GEAK_TOPO_NUMA_CMD="false")[0], 0)


if __name__ == "__main__":
    unittest.main()
