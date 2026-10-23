#!/usr/bin/env python3
# Copied into each generated author-HIP task as freeze.py. Computes the three
# integrity anchors and writes them into meta.json, so the oracle can detect a
# tampered baseline / harness / oracle. Idempotent.
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _tree_sha(root):
    items = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "build")]
        for f in fn:
            if f.endswith((".pyc", ".so", ".o")):
                continue
            p = os.path.join(dp, f)
            items.append((os.path.relpath(p, root), open(p, "rb").read()))
    hsh = hashlib.sha256()
    for rel, content in sorted(items, key=lambda x: x[0]):
        hsh.update(rel.encode())
        hsh.update(b"\0")
        hsh.update(content)
    return hsh.hexdigest()


def _file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


meta_path = os.path.join(HERE, "meta.json")
meta = json.load(open(meta_path))
meta["baseline_src_sha256"] = _tree_sha(os.path.join(HERE, "baseline_src"))
meta["harness_lib_sha256"] = _file_sha(os.path.join(HERE, "harness_lib.py"))
meta["unittest_sha256"] = _file_sha(os.path.join(HERE, "unittest.py"))
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
print("froze integrity anchors:")
print("  baseline_src_sha256 =", meta["baseline_src_sha256"][:16])
print("  harness_lib_sha256  =", meta["harness_lib_sha256"][:16])
print("  unittest_sha256     =", meta["unittest_sha256"][:16])
