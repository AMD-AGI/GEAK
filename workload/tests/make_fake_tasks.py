#!/usr/bin/env python3
"""Build a set of fake frozen-harness tasks to exercise every mechanism of the optimize
stage — the attempt loop, KEEP/revert, credit/heartbeat, the cold re-verification gate —
**without a GPU and without spending tokens**.

A fake task's unittest.py never touches the GPU: it reads a single line `LEVEL = <n>` from
kernel_src/mod.py, maps n to a speedup, and prints CORRECTNESS / GEAK_GEOMEAN_SPEEDUP in
the agreed format. Together with runners/stub.sh (which edits that line via STUB_SCRIPT)
the whole chain runs end to end.

    python3 make_fake_tasks.py --dir /tmp/faketasks --count 2
"""
import argparse, json, os, shutil

UNITTEST = '''\
import os, re, sys
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_src", "mod.py")
text = open(src).read()
m = re.search(r"^LEVEL\\s*=\\s*([0-9.]+)", text, re.M)
level = float(m.group(1)) if m else 0.0
# Fake correctness: a negative LEVEL means the code was broken. CHEAT=1 simulates the
# "PASS while hot / FAIL from cold" cheat.
cheat = "CHEAT = 1" in text
cold = os.environ.get("FAKE_COLD", "1") == "1"
ok = level >= 0 and not (cheat and cold)
print("CORRECTNESS: %s" % ("PASS" if ok else "FAIL"))
sp = 1.0 + 0.10 * level
print("GEAK_GEOMEAN_SPEEDUP %.4f" % sp)
print("GEAK_WEIGHTED_SPEEDUP %.4f" % sp)
sys.exit(0 if ok else 1)
'''

MOD = '''\
# Fake kernel. The only meaningful thing here is the LEVEL line below - it decides the
# speedup unittest.py reports.
LEVEL = 0
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=2)
    args = ap.parse_args()
    root = os.path.abspath(args.dir)
    if os.path.exists(root):
        shutil.rmtree(root)
    names = []
    for i in range(args.count):
        k = "fake_kernel_%d" % i
        names.append(k)
        d = os.path.join(root, k)
        os.makedirs(os.path.join(d, "kernel_src"))
        os.makedirs(os.path.join(d, "kernel_src_original"))
        open(os.path.join(d, "unittest.py"), "w").write(UNITTEST)
        open(os.path.join(d, "cases.py"), "w").write("CASES = []\n")
        # The pristine copy: verify.py's code-identity check uses it to catch "claims a
        # speedup without having changed anything".
        for sub in ("kernel_src", "kernel_src_original"):
            open(os.path.join(d, sub, "mod.py"), "w").write(MOD)
        json.dump({"kernel": k, "module": "fake", "pct_gpu_time": 10.0 - i,
                   "bound_type": "memory", "attainable_speedup": 2.0, "num_cases": 3},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
    print("built %d fake tasks -> %s" % (len(names), root))
    print(" ".join(names))


if __name__ == "__main__":
    main()
