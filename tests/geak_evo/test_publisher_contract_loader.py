"""Check the root reader closure without installing or launching a runtime.

The source hashes and private import namespaces are real. Only the immutable
root-owner ancestry check is replaced for this ordinary temporary checkout.
"""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest


class PublisherContractLoaderTests(unittest.TestCase):
    namespace = "_geak_evo_publisher_contract"

    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        original = json.loads((cls.repository / "docs/evo/owner-dispatch-base.json").read_bytes())
        row = next(row for row in original["files"] if row["name"] == "host_driver.py")
        source = Path(row["source_path"])
        if not source.is_file():
            raise unittest.SkipTest("retained wrapper patch base is not installed")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise AssertionError("retained wrapper differs from registered base")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        path = Path(temporary.name) / "host_driver.py"
        path.write_bytes(raw)
        subprocess.run(["/usr/bin/git", "apply", "--include=host_driver.py",
                        str(cls.repository / "docs/evo/owner-dispatch.patch")],
                       cwd=temporary.name, check=True, capture_output=True)
        contract = json.loads((cls.repository / "docs/evo/publisher-contract-base.json").read_bytes())
        row = next(row for row in contract["files"] if row["name"] == "host_driver.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise AssertionError("wrapper contract has a different composed base")
        subprocess.run(["/usr/bin/git", "apply", "--include=host_driver.py",
                        str(cls.repository / "docs/evo/publisher-contract.patch")],
                       cwd=temporary.name, check=True, capture_output=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["patched_sha256"]:
            raise AssertionError("wrapper contract differs from registered result")
        wanted = {"need", "reference", "checked", "load_cleaned_publisher_contract"}
        tree = ast.parse(path.read_bytes(), filename=str(path))
        members = [node for node in tree.body if getattr(node, "name", None) in wanted]
        cls.code = compile(ast.Module(body=members, type_ignores=[]), str(path), "exec")

    def setUp(self):
        self.addCleanup(self.clear_namespace)
        self.clear_namespace()
        self.canonical = {name: value for name, value in sys.modules.items()
                          if name == "geak_evo" or name.startswith("geak_evo.")}
        namespace = dict(hashlib=hashlib, Path=Path, sys=sys, ModuleType=ModuleType,
                         SimpleNamespace=SimpleNamespace,
                         immutable=lambda path: Path(path).resolve(strict=True))
        exec(self.code, namespace)
        self.load = namespace["load_cleaned_publisher_contract"]
        refs = {str(path.relative_to(self.repository)): {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (self.repository / "geak_evo").rglob("*.py")}
        self.manifest = {"cleaned_root": str(self.repository), "cleaned_source_refs": refs,
                         "source_refs": list(refs.values())}

    def clear_namespace(self):
        for name in list(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + "."):
                del sys.modules[name]

    def test_exact_source_closure_preserves_canonical_runtime(self):
        contract = self.load(self.manifest)
        expected = {"", ".archive", ".planner", ".engineer", "._util", ".archive.store",
                    ".archive.refs", ".archive.claims", ".archive.policy", ".archive.trials",
                    ".archive.succession", ".planner.context", ".engineer.instructions"}
        loaded = {name for name in sys.modules
                  if name == self.namespace or name.startswith(self.namespace + ".")}
        self.assertEqual(loaded, {self.namespace + suffix for suffix in expected})
        for suffix in ("", ".archive", ".planner", ".engineer"):
            self.assertEqual(sys.modules[self.namespace + suffix].__path__, [])
        self.assertEqual({name: sys.modules[name] for name in self.canonical}, self.canonical)
        for name in ("parent_snapshot", "archive_files", "selected_parent_material"):
            self.assertEqual(getattr(contract, name).__module__, self.namespace + ".archive.succession")
        self.assertEqual(contract.tool_addendum.__module__, self.namespace + ".engineer.instructions")
        self.assertEqual(contract.bound_evidence.__module__, self.namespace + ".archive.store")
        self.assertFalse(any(name.endswith((".host", ".model", ".investigation", ".operations"))
                             for name in loaded))
        with self.assertRaisesRegex(ValueError, "fresh publisher reader namespace required"):
            self.load(self.manifest)

    def test_wrong_hash_rejected_before_any_reader_executes(self):
        ref = self.manifest["cleaned_source_refs"]["geak_evo/archive/succession.py"]
        ref["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source reference differs"):
            self.load(self.manifest)
        self.assertNotIn(self.namespace, sys.modules)

    def test_reader_missing_from_inventory_rejected_before_execution(self):
        ref = self.manifest["cleaned_source_refs"]["geak_evo/engineer/instructions.py"]
        self.manifest["source_refs"].remove(ref)
        with self.assertRaisesRegex(ValueError, "publisher reader omitted"):
            self.load(self.manifest)
        self.assertNotIn(self.namespace, sys.modules)


if __name__ == "__main__":
    unittest.main()
