"""Real interpreter coverage of manifest module overlays and helper startup."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("overlay_module_startup_setup", SCRIPTS / "overlay_setup.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class TestModuleStartup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.overlay = self.root / "overlay"
        self.installed = self.root / "installed"
        self.installed.mkdir()
        self.write("installed/probe_pkg/__init__.py", "")
        self.write("installed/probe_pkg/target.py", "def entry():\n    return 'stock'\n")
        self.write("installed/probe_pkg/sibling.py", "VALUE = 'sibling'\n")
        self.write("installed/framework_dependency.py", "VALUE = 'framework'\n")
        self.patch("probe_pkg.target", "import framework_dependency\ndef entry():\n    return 'patched'\n")

    def write(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
        return path

    def patch(self, name, source):
        source_path = self.write("sources/" + name + ".py", source)
        SETUP.cmd_add_module(type("Args", (), {
            "overlay": str(self.overlay), "base": "", "module": name,
            "patched_file": str(source_path),
        })())

    def run_python(self, source, *, overlay=True, script=False):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in
                                           ([self.overlay] if overlay else []) + [self.installed])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-B"]
        if script:
            command.append(str(self.write("helper_probe.py", source)))
        else:
            command.extend(["-c", textwrap.dedent(source)])
        return subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, timeout=20,
                              check=False)

    def assert_success(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_plain_helper_does_not_import_target_or_framework(self):
        process = self.run_python("""
            import sys
            assert 'probe_pkg' not in sys.modules
            assert 'probe_pkg.target' not in sys.modules
            assert 'framework_dependency' not in sys.modules
        """)
        self.assert_success(process)
        self.assertNotIn("injected module", process.stderr)

    def test_real_spawned_helper_does_not_import_framework(self):
        process = self.run_python("""
            import multiprocessing
            import sys

            def helper():
                assert 'probe_pkg.target' not in sys.modules
                assert 'framework_dependency' not in sys.modules

            if __name__ == '__main__':
                context = multiprocessing.get_context('spawn')
                child = context.Process(target=helper)
                child.start()
                child.join(10)
                if child.is_alive():
                    child.terminate()
                    child.join(5)
                    raise AssertionError('helper timed out')
                assert child.exitcode == 0, child.exitcode
        """, script=True)
        self.assert_success(process)

    def test_first_import_uses_exact_module_and_preserves_parent_and_sibling(self):
        process = self.run_python("""
            import hashlib
            import importlib
            import pathlib
            import sys
            target = importlib.import_module('probe_pkg.target')
            from probe_pkg import target as parent_alias, sibling
            from probe_pkg.target import entry
            assert target is parent_alias
            assert entry is target.entry
            assert entry() == 'patched'
            assert sibling.VALUE == 'sibling'
            assert 'framework_dependency' in sys.modules
            source = pathlib.Path('sources/probe_pkg.target.py').read_bytes()
            assert pathlib.Path(target.__file__).read_bytes() == source
            assert target.__spec__.name == 'probe_pkg.target'
            assert target.__package__ == 'probe_pkg'
            assert pathlib.Path(target.entry.__code__.co_filename) == pathlib.Path(target.__file__)
            assert importlib.import_module('probe_pkg.target') is target
        """)
        self.assert_success(process)
        self.assertEqual(process.stderr.count("injected module probe_pkg.target"), 1)

    def test_all_module_mappings_are_ready_before_a_patched_module_imports_another(self):
        self.patch("probe_pkg.target", "from .other import VALUE\ndef entry():\n    return VALUE\n")
        self.patch("probe_pkg.other", "VALUE = 'patched-other'\n")
        self.write("installed/probe_pkg/other.py", "VALUE = 'stock-other'\n")
        process = self.run_python("from probe_pkg.target import entry\nassert entry() == 'patched-other'\n")
        self.assert_success(process)

    def test_already_imported_stock_module_is_reported(self):
        process = self.run_python("""
            import runpy, sys, types
            sys.modules['probe_pkg.target'] = types.ModuleType('probe_pkg.target')
            runpy.run_path('overlay/sitecustomize.py')
            assert not hasattr(sys.modules['probe_pkg.target'], 'entry')
        """)
        self.assert_success(process)
        self.assertIn('target already imported before overlay registration', process.stderr)

    def test_package_initializer_replacement_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'package __init__.py'):
            SETUP.cmd_add_module(type('Args', (), {
                'module': 'probe_pkg', 'overlay': str(self.overlay),
                'patched_file': str(self.installed / 'probe_pkg/__init__.py'),
            })())

    def test_mixed_module_and_rebind_keeps_existing_rebind_activation(self):
        self.write("overlay/rebound_impl.py", "def replacement():\n    return 'rebound'\n")
        manifest_path = self.overlay / "_overlay_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rebinds"] = [{"target": "probe_pkg.target:entry", "impl_module": "rebound_impl",
                                "impl_attr": "replacement"}]
        manifest_path.write_text(json.dumps(manifest))
        process = self.run_python("""
            import sys
            assert 'probe_pkg.target' in sys.modules
            import probe_pkg.target as target
            from probe_pkg import target as parent_alias
            from probe_pkg.target import entry
            assert parent_alias is target
            assert entry is target.entry
            assert entry() == 'rebound'
            assert 'framework_dependency' in sys.modules
        """)
        self.assert_success(process)

    def test_missing_dependency_raises_on_import_without_poisoning_module_cache(self):
        self.patch("probe_pkg.target", "import intentionally_missing_overlay_dependency\n")
        process = self.run_python("""
            import importlib
            import sys
            assert 'probe_pkg.target' not in sys.modules
            for attempt in range(2):
                try:
                    importlib.import_module('probe_pkg.target')
                except ModuleNotFoundError as error:
                    assert error.name == 'intentionally_missing_overlay_dependency'
                else:
                    raise AssertionError('overlay import failure was swallowed')
                assert 'probe_pkg.target' not in sys.modules
            from probe_pkg import sibling
            assert sibling.VALUE == 'sibling'
        """)
        self.assert_success(process)

    def test_removing_overlay_path_restores_stock(self):
        process = self.run_python("""
            from probe_pkg.target import entry
            import sys
            assert entry() == 'stock'
            assert 'framework_dependency' not in sys.modules
        """, overlay=False)
        self.assert_success(process)


if __name__ == "__main__":
    unittest.main()
