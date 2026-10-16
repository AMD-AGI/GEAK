"""Compare planner decisions and retained context with the frozen source.

GEAK_EVO_OCT15_SOURCE names the read-only reference package. These tests execute
only synthetic CPU fixtures; they never construct a model/process boundary.
"""

import ast
import base64
import copy
from contextlib import nullcontext
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = os.environ.get("GEAK_EVO_OCT15_SOURCE")


def snapshot(side, root):
    if side == "old":
        from geak_evo import cli_planner as context
        import native_planner as native
    else:
        # During the port, dependencies may live in the independently developed
        # worktree. Use its actual modules, not stand-ins. The final tree needs no
        # support path and this branch does nothing.
        support = os.environ.get("GEAK_EVO_PORT_SUPPORT")
        if support:
            import geak_evo
            geak_evo.__path__.insert(0, str(ROOT / "geak_evo"))
            geak_evo.__path__.append(str(Path(support) / "geak_evo"))
        from geak_evo.planner import context, native

    root.mkdir(mode=0o700)
    results = {}

    def record(name, function):
        try:
            results[name] = {"value": function()}
        except Exception as exc:
            results[name] = {"error": type(exc).__name__, "message": str(exc)}

    def ref(name):
        return {"evidence_id": name * 64, "artifact": "claim.json"}

    a, b, c, d = [ref(name) for name in "abcd"]
    binding = {"job_id": "job", "source_id": "source", "measurement_epoch_id": "epoch"}

    def claim(reference, state, source, dependencies=()):
        return {"ref": reference, "state": state,
                "record": {"binding": dict(binding, source_id=source), "depends_on": list(dependencies)}}

    view = {"head": d, "claims": {
        "d": claim(d, "disputed", "replacement"),
        "b": claim(b, "active", "source", [a]),
        "a": claim(a, "superseded", "parent"),
        "c": claim(c, "superseded", "replacement"),
    }, "corrections": [{"targets": [a], "replacement": c},
                        {"targets": [c], "replacement": d}]}
    record("claims_empty", lambda: context.current_claim_context({}))
    record("claims_unbound", lambda: context.current_claim_context(view))
    record("claims_bound", lambda: context.current_claim_context(view, binding=binding))
    record("claims_wrong_binding", lambda: context.current_claim_context(view, binding={"source_id": "source"}))
    missing = copy.deepcopy(view)
    del missing["claims"]["a"]
    record("claims_missing_dependency", lambda: context.current_claim_context(missing, binding=binding))
    projected = context.current_claim_context(view, binding=binding)
    projected["claim_states"][0]["state"] = "mutated"
    results["claims_input_unchanged"] = view

    for name, value in (("unicode", {"é": "λ"}), ("decimal", {"d": Decimal("1.20")}),
                        ("invalid", {"x": object()}), ("nonfinite", {"x": float("nan")})):
        record("json_" + name, lambda value=value: context._json(value))
    for index, raw in enumerate((b'{"a":1,"a":2}', b'{"x":NaN}', b'null', b'{"x":"\xce\xbb"}')):
        record("decode_%d" % index, lambda raw=raw: context._decode(raw))
    record("bounded_json_small", lambda: context._bounded_json_object({"small": True}, 256))
    record("bounded_json_omitted", lambda: context._bounded_json_object({"large": "x" * 2000}, 256))
    record("bounded_json_impossible", lambda: context._bounded_json_object({"large": "x" * 2000}, 8))

    public = root / "public"
    public.mkdir(mode=0o700)
    brief = {"large_b": ["b" * 4000] * 6, "large_a": ["a" * 4000] * 6,
             "inline": {"full_value_ref": "ordinary data"}, "small": [1, 2]}
    brief_text = context._json(brief).decode()
    (public / "brief.json").write_text(brief_text)
    index = []
    results["navigation"] = context._bounded_decision_navigation(brief_text, public, index)
    results["navigation_index"] = index
    results["navigation_files"] = {p.name: p.read_bytes() for p in public.iterdir()}

    for number, raw in enumerate((b'null', b'{"direction_id":"d"}',
            b'{"proposal":null,"recommendation":{"op":"keep"}}',
            b'{"proposal":{"direction_id":"d"},"recommendation":{"op":"clear","reason":"r"}}',
            b'```json\nnull\n```', b'[1]', b'{"a":1,"a":2}', b'not json', b'x' * (native.MAX_PROPOSAL_BYTES + 1))):
        record("response_%d" % number, lambda raw=raw: native._parse_recommendation_response(raw))
    for arm in (None, "G", "EVO"):
        policy = None if arm is None else {"arm": arm, "study_root": str(root), "shared_read_roots": []}
        for objective in ("ordinary", "history"):
            record("scope_%s_%s" % (arm, objective),
                   lambda policy=policy, objective=objective: native.scope_for_policy(policy, review_objective=objective))
    results["prompt"] = context._PROMPT
    results["native_prompt"] = native.NATIVE_PROMPT

    followup = {"reservation": {"wall_seconds": 180, "evaluations": 0}, "decisions_after_current": 2}
    for index, (available, end) in enumerate(((6000, 10_000), (400, 1000), (0, 1000))):
        record("reservation_%d" % index, lambda available=available, end=end:
               native._bounded_engineer_reservation({"wall_seconds": available, "evaluations": 25},
                   end * native.NS, 200 * native.NS, 60, followup))

    for error in (native.ProposalError("typed refusal"), ValueError("unknown value"), RuntimeError("unknown implementation")):
        with mock.patch.object(native, "_proposal_admission", side_effect=error):
            record("proposal_failure_" + type(error).__name__, lambda:
                   native._proposal_rejection(None, {}, {}, None, {"context_sha256": "x"}, {}, {}))

    store = SimpleNamespace(root=root)
    frozen = {"campaign_id": "campaign", "state": {"history": [], "opportunities": [], "binding": binding},
              "direction_notes": {}, "claims": view}
    frozen_path = root / "context.json"
    frozen_path.write_bytes(context._json(frozen))
    record("outcome_unknown", lambda: context.outcome_prompt_context(store, frozen, context.file_ref(frozen_path)))
    changed = copy.deepcopy(frozen)
    changed["direction_notes"]["unexpected"] = {}
    record("outcome_changed", lambda: context.outcome_prompt_context(store, changed, context.file_ref(frozen_path)))

    actions = root / "actions.md"
    actions.write_text("Synthetic action instructions.\n")
    supplied = root / "source.py"
    supplied.write_text("def kernel(x):\n    return x\n")
    freeze_context = dict(frozen, claims={}, archive={"directions": {}},
        instructions={"ref": context.file_ref(actions), "text": actions.read_text()})
    freeze_context["state"] = dict(frozen["state"], limits={"wall_seconds": 100, "evaluations": 10}, deadline=200)
    builder = context.CLIPlanner.__new__(context.CLIPlanner)
    builder.store, builder.read_only, builder.evidence_refs = store, [context.file_ref(supplied)], []
    builder.search_policy, builder.campaign_root = None, None
    freeze_call = root / "freeze"
    freeze_call.mkdir(mode=0o700)
    results["frozen_view"] = builder._freeze(freeze_context, freeze_call)
    results["frozen_files"] = {str(p.relative_to(freeze_call)): p.read_bytes()
                              for p in freeze_call.rglob("*") if p.is_file()}

    constructor_store = root / "store"
    constructor_store.mkdir(mode=0o700)
    clock_file = root / "clock.json"
    clock_file.write_text("{}")
    config = {"schema": "geak-standalone-model-config-v1", "launch_admitted": True,
              "clock_path": str(clock_file), "work_root": str(root / "work"),
              "state_root": str(root / "model"), "read_roots": [], "max_budget_usd": 1}
    constructor_loop = SimpleNamespace(store=SimpleNamespace(root=constructor_store))
    backend = SimpleNamespace(__file__=str(supplied), run=lambda *args, **kwargs: None)
    dependencies = (
        SimpleNamespace(QualifiedClaudeTransport=lambda *args, **kwargs: None),
        SimpleNamespace(validate_model_route=lambda value: value,
                        inspect_claude_events=lambda *args, **kwargs: None),
        SimpleNamespace(FileSettingsRuntime=lambda runtime: runtime))
    planner = native.NativePlanner(constructor_loop, root / "native-constructor",
        model_backend=backend, model_config=config, clock_ref=context.file_ref(clock_file),
        native_envelope_seconds=60, campaign_root=root,
        native_evidence=SimpleNamespace(episode=lambda *args: None), dependencies=dependencies)
    results["native_constructor_binding"] = planner._binding()
    def encode(value):
        if isinstance(value, bytes):
            return {"bytes_base64": base64.b64encode(value).decode()}
        raise TypeError(type(value).__name__)
    return json.dumps(results, sort_keys=True, default=encode)


class PlannerContextPortTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE, "set GEAK_EVO_OCT15_SOURCE for source parity")
    def test_exact_synthetic_context_and_recovery_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture"
            results = []
            for side in ("old", "new"):
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                env["PYTHONPATH"] = (str(Path(SOURCE) / "evolution/src") + os.pathsep +
                    str(Path(SOURCE) / "bridge")) if side == "old" else str(ROOT)
                if side == "new" and env.get("GEAK_EVO_PORT_SUPPORT"):
                    env["PYTHONPATH"] += os.pathsep + env["GEAK_EVO_PORT_SUPPORT"]
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--snapshot", side, str(path)], env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                results.append(json.loads(process.stdout))
                for directory in path.rglob("*"):
                    if directory.is_dir():
                        directory.chmod(0o700)
                shutil.rmtree(path)
            self.assertEqual(set(results[0]), set(results[1]))
            for key in results[0]:
                with self.subTest(case=key):
                    self.assertEqual(results[0][key], results[1][key])

    def test_current_context_freezes_exact_inputs_and_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            result = json.loads(snapshot("new", root))
            brief, references = result["frozen_view"]
            self.assertEqual(brief["binding"], {
                "job_id": "job", "source_id": "source", "measurement_epoch_id": "epoch"})
            self.assertEqual(brief["resource_limits"], {"wall_seconds": 100, "evaluations": 10})
            self.assertEqual(brief["legal_existing_directions"], [])
            expected_files = {
                "view/actions.md", "view/snapshot.json", "view/notes.json",
                "view/files.json", "view/brief.json", "view/replicas/input-0000-source.py",
            }
            self.assertEqual(set(result["frozen_files"]), expected_files)
            self.assertEqual({str(Path(ref["path"]).relative_to(root / "freeze"))
                              for ref in references}, expected_files)
            for ref in references:
                self.assertEqual(hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest(), ref["sha256"])
            replica = root / "freeze/view/replicas/input-0000-source.py"
            original = root / "source.py"
            self.assertEqual(replica.read_bytes(), original.read_bytes())
            original.write_text("changed after freeze")
            self.assertEqual(replica.read_bytes(), b"def kernel(x):\n    return x\n")
            self.assertNotIn("instructions", json.loads((root / "freeze/view/snapshot.json").read_text()))
            (root / "freeze/view/replicas").chmod(0o700)

    def test_native_runtime_and_planner_share_the_selected_clock(self):
        from geak_evo.planner.context import file_ref
        from geak_evo.planner.model import ModelRuntime
        from geak_evo.planner.native import NativePlanner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store = root / "store"
            store.mkdir(mode=0o700)
            clock_file = root / "clock.json"
            clock_file.write_text("{}")
            backend_file = root / "backend.py"
            backend_file.write_text("# synthetic model service\n")
            config = {"schema": "geak-standalone-model-config-v1", "launch_admitted": True,
                      "clock_path": str(clock_file), "work_root": str(root / "work"),
                      "state_root": str(root / "model"), "read_roots": [], "max_budget_usd": 1}
            original_clock, explicit_clock = object(), object()
            dependencies = (
                SimpleNamespace(QualifiedClaudeTransport=lambda *args, **kwargs: None),
                SimpleNamespace(validate_model_route=lambda value: value,
                                inspect_claude_events=lambda *args, **kwargs: None),
                SimpleNamespace(FileSettingsRuntime=lambda runtime: runtime))
            runtime = ModelRuntime(*dependencies, clock=original_clock)
            common = {"model_backend": SimpleNamespace(__file__=str(backend_file), run=lambda *args: None),
                      "model_config": config, "clock_ref": file_ref(clock_file),
                      "native_envelope_seconds": 60, "campaign_root": root,
                      "native_evidence": SimpleNamespace(episode=lambda *args: None)}
            loop = SimpleNamespace(store=SimpleNamespace(root=store))
            default = NativePlanner(loop, root / "default", runtime=runtime, **common)
            self.assertIs(default.clock, original_clock)
            self.assertIs(default.runtime.clock, default.clock)
            explicit = NativePlanner(loop, root / "explicit", runtime=runtime, clock=explicit_clock, **common)
            self.assertIs(explicit.clock, explicit_clock)
            self.assertIs(explicit.runtime.clock, explicit.clock)
            self.assertIs(runtime.clock, original_clock)
            tuple_bound = NativePlanner(loop, root / "tuple", dependencies=runtime.dependencies,
                                        clock=explicit_clock, **common)
            self.assertIs(tuple_bound.runtime.clock, explicit_clock)

    def test_missing_runtime_fails_before_any_planner_side_effect(self):
        from unittest.mock import Mock
        from geak_evo.planner.native import NativePlanner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backend = Mock()
            loop = SimpleNamespace(store=SimpleNamespace(root=root / "store"))
            common = {"model_backend": backend, "model_config": {}, "clock_ref": {},
                      "native_envelope_seconds": 60, "campaign_root": root}
            for name, arguments, message in (
                    ("missing", {}, "explicit model runtime"),
                    ("malformed", {"dependencies": (None,)}, "model dependencies"),
                    ("incomplete", {"dependencies": (None, None, None)}, "route validation")):
                with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                    NativePlanner(loop, root / name, **common, **arguments)
                self.assertFalse((root / name).exists())
                self.assertEqual(list(root.iterdir()), [])
            backend.run.assert_not_called()

    def test_new_call_requires_profile_before_reservation_but_replay_uses_saved_result(self):
        from geak_evo.planner import native
        with tempfile.TemporaryDirectory() as temporary:
            call = Path(temporary) / "call"
            planner = native.NativePlanner.__new__(native.NativePlanner)
            planner.clock = SimpleNamespace(monotonic_ns=lambda: 1)
            planner._lock = nullcontext
            planner._call = lambda operation_id: call
            planner.model_config = {"workflow_backend": "claude"}
            planner.runtime = None
            planner._reserved = mock.Mock()
            planner._result = mock.Mock(return_value={"saved": "original result"})
            with self.assertRaisesRegex(ValueError, "explicit pinned profile"):
                planner({}, "operation")
            planner._reserved.assert_not_called()
            self.assertFalse(call.exists())
            call.mkdir()
            (call / "context.json").write_text("{}")
            with mock.patch.object(native, "planner_model_profile", side_effect=AssertionError("must use saved admission")):
                self.assertEqual(planner({}, "operation"), {"saved": "original result"})
            planner._result.assert_called_once_with(call)
            planner._reserved.assert_not_called()

    @unittest.skipUnless(SOURCE, "set GEAK_EVO_OCT15_SOURCE for source parity")
    def test_prompts_and_unchanged_algorithm_bodies(self):
        trial_additions = {
            "retained_trials = set()",
            "retained_trials.update(retained_trial_candidates(self.store, artifacts))",
            "candidates.update(retained_trials)",
            "parent_candidates[-1]['retained_trial_source'] = identity in retained_trials",
            "parent_candidates[-1]['quality_scope'] = 'Source availability is not correctness or eligible fitness; read the exact trial outcomes.'",
        }
        profile_additions = {
            "NativePlanner.__call__": ["planner_model_profile(self.model_config, runtime=self.runtime)"],
            "NativePlanner._result": [
                "if 'model_profile' in receipt:\n"
                "    require(receipt['model_profile'] == planner_model_profile(intent['admission']['model_config'], runtime=self.runtime).as_dict(),\n"
                "            'model profile differs from original admission', ValueError)"],
            "NativePlanner._historical_settlement": [
                "if 'model_profile' in receipt:\n"
                "    require(receipt['model_profile'] == planner_model_profile(admission['model_config'], runtime=self.runtime).as_dict(),\n"
                "            'historical model profile differs from original admission', ValueError)"],
        }

        class Normalize(ast.NodeTransformer):
            def __init__(self, trial_inventory=False, profile_statements=()):
                self.trial_inventory = trial_inventory
                self.profile_statements = profile_statements

            def visit_Import(self, node):
                return None

            def visit_ImportFrom(self, node):
                return None

            def visit_Call(self, node):
                self.generic_visit(node)
                node.keywords = [item for item in node.keywords if item.arg != "runtime"]
                if isinstance(node.func, ast.Name) and node.func.id in ("require", "_need", "need"):
                    node.func.id = "check"
                    node.args = node.args[:2]
                if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "self"
                        and node.func.value.attr == "clock"):
                    if node.func.attr == "boot_id":
                        return ast.parse("Path('/proc/sys/kernel/random/boot_id').read_text().strip()", mode="eval").body
                    node.func.value = ast.Name(id="time", ctx=ast.Load())
                return node

            def visit_Assign(self, node):
                if self.trial_inventory and ast.unparse(node) in trial_additions:
                    return None
                if ast.unparse(node) == "cpu = self.runtime":
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if ast.dump(node) in self.profile_statements:
                    return None
                if self.trial_inventory and ast.unparse(node) in trial_additions:
                    return None
                if isinstance(node.value, ast.Call) and any(
                    isinstance(arg, ast.Constant) and arg.value == "planner runtime must be supplied explicitly"
                    for arg in node.value.args
                ):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                if ast.dump(node) in self.profile_statements:
                    return None
                if ast.unparse(node.test) == "self.runtime is None":
                    return [self.visit(item) for item in node.body]
                return self.generic_visit(node)

        # Intentional retirement of the unused CLI transport. Retain the original
        # parity comparison and its explicit exclusions for surviving helpers.
        retired_cli = {
            "CLIPlannerPending.__init__", "_object", "CLIPlanner.__init__",
            "CLIPlanner._lock", "CLIPlanner._call", "CLIPlanner._policy_binding",
            "CLIPlanner._audit", "CLIPlanner.__call__", "CLIPlanner._run",
            "CLIPlanner._result", "CLIPlanner.recover", "CLIPlanner.close_incomplete",
            "CLIPlanner.verify_recovery",
        }
        for old_path, new_name, exclusions in (
            ("evolution/src/geak_evo/cli_planner.py", "context", {"_need", "_private"} | retired_cli),
            ("bridge/native_planner.py", "native", {"planner_model_identity", "planner_model_observation", "_program_lease_prompt_ref", "NativePlanner.__init__"}),
        ):
            def functions(tree):
                result = {}
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        result[node.name] = node
                    elif isinstance(node, ast.ClassDef):
                        result.update({node.name + "." + fn.name: fn for fn in node.body if isinstance(fn, ast.FunctionDef)})
                return result
            old = functions(ast.parse((Path(SOURCE) / old_path).read_text()))
            new = functions(ast.parse((ROOT / "geak_evo/planner" / (new_name + ".py")).read_text()))
            for name in old.keys() - exclusions:
                with self.subTest(module=new_name, function=name):
                    self.assertIn(name, new)
                    trial_inventory = new_name == "native" and name == "NativePlanner._freeze"
                    if trial_inventory:
                        for statement in trial_additions:
                            self.assertEqual(sum(isinstance(node, (ast.Expr, ast.Assign))
                                                 and ast.unparse(node) == statement for node in ast.walk(new[name])), 1)
                    profile_statements = [ast.dump(ast.parse(statement).body[0])
                                          for statement in profile_additions.get(name, ())] if new_name == "native" else []
                    for statement in profile_statements:
                        self.assertEqual(sum(ast.dump(node) == statement for node in ast.walk(new[name])), 1)
                    self.assertEqual(ast.dump(Normalize().visit(old[name])),
                                     ast.dump(Normalize(trial_inventory, profile_statements).visit(new[name])))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--snapshot":
        print(snapshot(sys.argv[2], Path(sys.argv[3])))
    else:
        unittest.main()
