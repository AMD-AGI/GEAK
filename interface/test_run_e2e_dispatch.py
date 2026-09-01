#!/usr/bin/env python3
"""Unit tests for run_e2e.py's DISPATCH half — everything between reading
handoff.json and having a workflow return in hand (stdlib only; no GPU, no
claude SDK, no network, no real subprocess).

The two existing interface test modules (test_run_e2e_recovery.py,
test_run_e2e_alignment.py) cover the RESULT half: disk recovery,
normalize_result's numbers, and the guaranteed-emit contract. Nothing covered
the path that actually *launches* the optimizer, so a break there is invisible
until a real 12-hour GPU run dies. This module covers:

  - handoff -> workflow args (``map_args``): the optional knobs (launch_recipe,
    phases, carried state, time_budget_s), the minted-vs-pinned
    eval_dir, and the TraceLens artifact bridge. A dropped knob here silently
    re-runs a phase that was meant to be resumed, or mints a second abandoned
    eval_dir beside the authoritative one.
  - measurement-protocol exports (``apply_bench_client`` /
    ``apply_bench_launcher`` / ``apply_bench_protocol``): these are the ONLY
    channel that makes GEAK measure the way the calling orchestrator measured.
    A wrong export is not a crash, it is an unfalsifiable throughput number.
  - agent invocation (``_invoke_via_sdk`` / ``_invoke_via_cli`` /
    ``invoke_workflow``): the SDK completion gate (background task lifecycle +
    the on-disk terminal marker + the bounded grace poll) is what keeps a
    still-running detached A/B from being orphaned; the CLI fallback's exact
    argv is what makes the Workflow tool actually execute the JS pipeline.
  - transcript scraping (``_iter_message_text`` / ``_iter_json_objects`` /
    ``_parse_last_json_line``): recovering the workflow return from whatever
    shape the agent emitted it in.
  - ``main()``'s control flow: usage/handoff guards, --dry-run, the
    resume-from-cache short-circuit, the SIGTERM self-stop handler, and every
    degradation branch inside the guaranteed ``_emit`` (normalize raised,
    persist raised, report render raised, atomic write failed).

``anyio`` and ``claude_agent_sdk`` are injected into sys.modules as fakes for
the duration of each test — run_e2e imports both LAZILY inside the invocation
functions, so the module itself imports on a box with neither installed. The
fake anyio drives the real coroutine through stdlib asyncio and makes
``anyio.sleep`` an instant, observable hook, so the grace-poll loop is exercised
without spending its 1800s budget. Fakes are installed per-test and removed on
cleanup so the sibling interface test modules keep seeing an SDK-free image.

Run: python3 -m pytest GEAK/interface/test_run_e2e_dispatch.py -v
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_SENTINEL = object()


def _load(name: str = "run_e2e"):
    spec = importlib.util.spec_from_file_location(name, _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


# --------------------------------------------------------------------------- #
# fake SDK message objects: only ``type(msg).__name__`` and a few attributes
# matter to the completion gate, which is exactly what these carry.
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class AssistantMessage(_Msg):
    pass


class ResultMessage(_Msg):
    pass


class TaskStartedMessage(_Msg):
    pass


class TaskNotificationMessage(_Msg):
    pass


class SystemMessage(_Msg):
    pass


def _make_fake_anyio(sleep_hook=None):
    """anyio stand-in: real coroutine execution via asyncio, instant sleep.

    ``sleep_hook`` runs on each awaited sleep, which is how the grace-poll test
    makes the workflow's terminal marker appear mid-loop without wall time.
    """
    mod = types.ModuleType("anyio")
    calls: dict[str, list] = {"fail_after": [], "sleep": []}

    @contextlib.contextmanager
    def fail_after(seconds):
        calls["fail_after"].append(seconds)
        yield None

    async def sleep(seconds):
        calls["sleep"].append(seconds)
        if sleep_hook is not None:
            sleep_hook()

    def run(fn, *args):
        return asyncio.run(fn(*args))

    mod.fail_after = fail_after
    mod.sleep = sleep
    mod.run = run
    mod.calls = calls
    return mod


def _make_fake_sdk(script=None, *, with_client=True, query_script=None):
    mod = types.ModuleType("claude_agent_sdk")
    state: dict[str, list] = {
        "options": [], "clients": [], "prompts": [], "consumed": []
    }

    class ClaudeAgentOptions:
        def __init__(self, **kw):
            self.kwargs = kw
            state["options"].append(kw)

    mod.ClaudeAgentOptions = ClaudeAgentOptions

    if with_client:
        class ClaudeSDKClient:
            def __init__(self, options=None):
                self.options = options
                self.closed = False
                state["clients"].append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                self.closed = True
                return False

            async def query(self, prompt):
                state["prompts"].append(prompt)

            async def receive_messages(self):
                for msg in (script or []):
                    state["consumed"].append(msg)
                    yield msg

        mod.ClaudeSDKClient = ClaudeSDKClient

    if query_script is not None:
        async def query(prompt=None, options=None):
            state["prompts"].append(prompt)
            for msg in query_script:
                state["consumed"].append(msg)
                yield msg

        mod.query = query

    mod.state = state
    return mod


class _RunE2ECase(unittest.TestCase):
    """Restores os.environ, patched run_e2e globals, sys.modules and the SIGTERM
    disposition after every test — run_e2e mutates all four as a side effect."""

    def setUp(self):
        self._env = dict(os.environ)
        self._attrs: list[tuple[str, object]] = []
        self._mods: list[tuple[str, object]] = []
        self._sigterm = signal.getsignal(signal.SIGTERM)
        self.tmp = Path(tempfile.mkdtemp(prefix="run_e2e_dispatch_"))
        self.addCleanup(self._restore)

    def _restore(self):
        for name, prev in reversed(self._attrs):
            if prev is _SENTINEL:
                delattr(rx, name)
            else:
                setattr(rx, name, prev)
        for name, prev in reversed(self._mods):
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        os.environ.clear()
        os.environ.update(self._env)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, self._sigterm)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch_rx(self, name, value):
        self._attrs.append((name, getattr(rx, name, _SENTINEL)))
        setattr(rx, name, value)

    def install_module(self, name, mod):
        self._mods.append((name, sys.modules.get(name, _SENTINEL)))
        sys.modules[name] = mod

    def write_json(self, path: Path, obj) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path


# =========================================================================== #
# handoff -> workflow args
# =========================================================================== #
class TestMapArgs(_RunE2ECase):
    def _handoff(self, **extra):
        h = {
            "model_path": "/models/fake-8b",
            "exp_root": str(self.tmp / "exp" / "geak"),
            "workload": {"isl": 2048, "osl": 256, "conc": 8},
            "tp": 4,
        }
        h.update(extra)
        return h

    def test_optional_workflow_knobs_are_forwarded_verbatim(self):
        """launch_recipe / phases / carried state are the resume channel:
        dropping one silently re-runs a phase the caller pinned.

        `e2e_repeats` is deliberately NOT in that channel. The timed-round count
        is a property of the measurement lifecycle (GEAK_REPEAT_MODE +
        MEASUREMENT_PURPOSE), so a handoff carrying it must be ignored rather
        than allowed to pull one leg off the lifecycle the rest of the run used.
        """
        h = self._handoff(
            eval_dir=str(self.tmp / "e2e_pinned"),
            launch_recipe="/recipes/launch_vllm.sh",
            phases="final",
            e2e_repeats=1,
            state={"headQueue": [{"short_name": "h0"}]},
        )
        ps = rx.map_args(h, timeout_s=3600)
        self.assertEqual(ps["launch_script"], "/recipes/launch_vllm.sh")
        self.assertEqual(ps["phases"], "final")
        self.assertNotIn("e2e_repeats", ps)
        self.assertEqual(ps["state"], {"headQueue": [{"short_name": "h0"}]})
        self.assertEqual(ps["time_budget_s"], 3600)
        self.assertEqual(ps["eval_dir"], str(self.tmp / "e2e_pinned"))
        # tp=4 with no explicit gpu_ids => the serving device set matches TP.
        self.assertEqual(ps["gpu_ids"], "0,1,2,3")
        self.assertEqual(ps["config_tune"], "false")
        self.assertEqual(ps["apply_to_original"], "true")

    def test_budget_omitted_when_unknown(self):
        """No timeout => the workflow stays budget-unaware (byte-identical to a
        direct, non-interface invocation)."""
        ps = rx.map_args(self._handoff(eval_dir=str(self.tmp / "e2e_x")))
        self.assertNotIn("time_budget_s", ps)
        ps_zero = rx.map_args(
            self._handoff(eval_dir=str(self.tmp / "e2e_x")), timeout_s=0
        )
        self.assertNotIn("time_budget_s", ps_zero)

    def test_unparseable_fidelity_knobs_are_dropped_not_raised(self):
        """A junk max_model_len/mem_fraction must degrade to the adapter default,
        never abort the run before it starts."""
        h = self._handoff(
            eval_dir=str(self.tmp / "e2e_x"),
            framework="vllm",
            max_model_len="not-a-number",
            mem_fraction="nope",
        )
        ps = rx.map_args(h)
        self.assertNotIn("max_model_len", ps)
        self.assertNotIn("mem_fraction", ps)
        self.assertEqual(ps["initial_extra_server_args"], "")

    def test_eval_dir_is_minted_under_exp_root_when_unpinned(self):
        os.environ.pop("GEAK_EVAL_DIR", None)
        exp_root = self.tmp / "exp" / "geak"
        ps = rx.map_args(self._handoff())
        minted = Path(ps["eval_dir"])
        self.assertEqual(minted.parent, exp_root)
        self.assertTrue(minted.name.startswith("e2e_fake-8b_"))
        self.assertTrue(minted.name.endswith("Z"))

    def test_two_unpinned_runs_get_distinct_eval_dirs(self):
        """Concurrent/same-second runs must not share replay/artifact paths."""
        os.environ.pop("GEAK_EVAL_DIR", None)
        first = rx.map_args(self._handoff())["eval_dir"]
        second = rx.map_args(self._handoff())["eval_dir"]
        self.assertNotEqual(first, second)

    def test_env_pinned_eval_dir_wins_over_minting(self):
        os.environ["GEAK_EVAL_DIR"] = str(self.tmp / "e2e_from_env")
        ps = rx.map_args(self._handoff())
        self.assertEqual(ps["eval_dir"], str(self.tmp / "e2e_from_env"))

    def test_tracelens_artifacts_are_bridged_into_workflow_args(self):
        """The four upstream kernel-agent artifacts live BESIDE ``geak`` under the
        experiment root; they must reach args.tracelens (not just the prompt) or
        the JS Profile/Strategize phases lose their prior."""
        root = self.tmp / "exp"
        analysis = root / "kernel-agent" / "r1" / "tracelens" / "analysis.md"
        cands = root / "kernel-agent" / "r1" / "kernel_candidates.json"
        report = root / "kernel-agent" / "r1" / "tracelens" / "tracelens_report.json"
        trace = root / "runs" / "roofline" / "r9" / "torch_trace"
        for p in (analysis, cands, report, trace):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")
        ps = rx.map_args(self._handoff(eval_dir=str(self.tmp / "e2e_x")))
        self.assertEqual(ps["tracelens"], {
            "analysis_md": str(analysis),
            "kernel_candidates_json": str(cands),
            "tracelens_report_json": str(report),
            "trace_file": str(trace),
        })
        self.assertNotIn("search_root", ps["tracelens"])

    def test_tracelens_key_omitted_when_nothing_discoverable(self):
        ps = rx.map_args(self._handoff(eval_dir=str(self.tmp / "e2e_x")))
        self.assertNotIn("tracelens", ps)

    def test_experiment_root_strips_only_a_geak_leaf(self):
        self.assertEqual(
            rx._experiment_root_from_exp_root("/a/b/geak/"), "/a/b"
        )
        self.assertEqual(rx._experiment_root_from_exp_root("/a/b/other"), "/a/b/other")
        self.assertEqual(rx._experiment_root_from_exp_root(""), "")

    def test_resolve_tracelens_report_without_root_is_all_none(self):
        report = rx.resolve_tracelens_report("")
        self.assertEqual(report["search_root"], "")
        self.assertIsNone(report["analysis_md"])
        self.assertIsNone(report["trace_file"])

    def test_build_prompt_carries_script_args_and_tracelens_block(self):
        ps = rx.map_args(self._handoff(eval_dir=str(self.tmp / "e2e_prompt")))
        prompt = rx.build_prompt(ps)
        self.assertIn(f'scriptPath: "{rx.E2E_SCRIPT}"', prompt)
        self.assertIn(json.dumps(ps), prompt)
        self.assertIn(f'"{ps["eval_dir"]}/workflow_return.json"', prompt)
        self.assertIn("tracelens_report", prompt)
        # search_root is internal bookkeeping and must never reach the agent.
        self.assertNotIn("search_root", prompt)

    def test_build_prompt_leads_with_process_safety(self):
        """The driver agent holds Bash under bypassPermissions as a direct child of
        this process, so it needs the same pattern-kill ban the role agents get: one
        `pkill -f vllm` from it reaches the caller's orchestrator (issue #397)."""
        ps = rx.map_args(self._handoff(eval_dir=str(self.tmp / "e2e_safety")))
        prompt = rx.build_prompt(ps)
        self.assertTrue(prompt.startswith("## PROCESS SAFETY"), prompt[:80])
        for banned in ("pkill -f", "killall", "kill -- -PGID"):
            self.assertIn(banned, prompt)


class TestProtectedPgids(_RunE2ECase):
    """`_publish_protected_pgids` is the caller-side half of the #397 fix: the
    teardown in e2e_workflow/scripts/server_teardown.sh vetoes a group kill against
    any pgid listed in GEAK_PROTECTED_PGIDS, and this is the only publisher."""

    def setUp(self):
        super().setUp()
        self._saved = os.environ.get("GEAK_PROTECTED_PGIDS")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("GEAK_PROTECTED_PGIDS", None)
        else:
            os.environ["GEAK_PROTECTED_PGIDS"] = self._saved

    def test_publishes_own_group_parent_group_and_init(self):
        os.environ.pop("GEAK_PROTECTED_PGIDS", None)
        value = rx._publish_protected_pgids()
        published = value.split()
        self.assertEqual(value, os.environ["GEAK_PROTECTED_PGIDS"])
        self.assertIn("1", published)
        self.assertIn(str(os.getpgid(0)), published)
        self.assertIn(str(os.getpgid(os.getppid())), published)

    def test_merges_existing_value_and_drops_non_numeric(self):
        """A caller may pre-export its own pgids; keep them, drop garbage, no dupes."""
        os.environ["GEAK_PROTECTED_PGIDS"] = "77 bogus 77"
        published = rx._publish_protected_pgids().split()
        self.assertIn("77", published)
        self.assertNotIn("bogus", published)
        self.assertEqual(len(published), len(set(published)))

    def test_is_idempotent(self):
        os.environ.pop("GEAK_PROTECTED_PGIDS", None)
        first = rx._publish_protected_pgids()
        self.assertEqual(first, rx._publish_protected_pgids())


class TestFlagPresent(_RunE2ECase):
    def test_unbalanced_quote_falls_back_to_whitespace_split(self):
        """A malformed caller flag string must not raise out of the fidelity
        fold — it degrades to a naive split so the dedup still works."""
        self.assertTrue(rx._flag_present('--max-model-len 8192 --foo "bar', "--foo"))
        self.assertFalse(rx._flag_present('--other 1 "unclosed', "--foo"))

    def test_empty_inputs_are_absent(self):
        self.assertFalse(rx._flag_present("", "--foo"))
        self.assertFalse(rx._flag_present("--foo 1", ""))


class TestWarnThresholdEnv(_RunE2ECase):
    def test_invalid_threshold_env_falls_back_to_default(self):
        """A negative/garbage GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT must not make
        every run unclassifiable — the module clamps back to 3.0%."""
        os.environ["GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT"] = "-4"
        self.assertEqual(_load("run_e2e_negthr").SAME_CONFIG_DIVERGENCE_WARN_PCT, 3.0)
        os.environ["GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT"] = "banana"
        self.assertEqual(_load("run_e2e_badthr").SAME_CONFIG_DIVERGENCE_WARN_PCT, 3.0)
        os.environ["GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT"] = "7.5"
        self.assertEqual(_load("run_e2e_okthr").SAME_CONFIG_DIVERGENCE_WARN_PCT, 7.5)


# =========================================================================== #
# measurement-protocol exports
# =========================================================================== #
class TestBenchClient(_RunE2ECase):
    def test_auto_selects_inferencex_when_a_checkout_is_named(self):
        os.environ.pop("INFERENCEX_PATH", None)
        client = rx.apply_bench_client({"inferencex_path": "/opt/inferencex"})
        self.assertEqual(client, "inferencex")
        self.assertEqual(os.environ["INFERENCEX_PATH"], "/opt/inferencex")
        self.assertEqual(os.environ["BENCH_CLIENT"], "inferencex")

    def test_auto_falls_back_to_native_without_a_checkout(self):
        os.environ.pop("INFERENCEX_PATH", None)
        self.assertEqual(rx.apply_bench_client({}), "native")
        self.assertEqual(os.environ["BENCH_CLIENT"], "native")

    def test_auto_honours_the_ambient_env_checkout(self):
        os.environ["INFERENCEX_PATH"] = "/env/inferencex"
        self.assertEqual(rx.apply_bench_client({"bench_client": "auto"}), "inferencex")

    def test_explicit_inferencex_without_path_degrades_loudly(self):
        """Silently measuring with a different client than the orchestrator is
        the failure this warning exists to prevent."""
        os.environ.pop("INFERENCEX_PATH", None)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            client = rx.apply_bench_client({"bench_client": "InferenceX"})
        self.assertEqual(client, "native")
        self.assertEqual(os.environ["BENCH_CLIENT"], "native")
        self.assertIn("measurement protocol NOT aligned", err.getvalue())

    def test_explicit_native_is_respected_even_with_a_checkout(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            client = rx.apply_bench_client(
                {"bench_client": "native", "inferencex_path": "/opt/ix"}
            )
        self.assertEqual(client, "native")
        self.assertEqual(err.getvalue(), "")


class TestBenchLauncher(_RunE2ECase):
    def setUp(self):
        super().setUp()
        for key in ("BENCH_LAUNCHER", "MAGPIE_LAUNCH_SCRIPT",
                    "MAGPIE_LAUNCH_SCRIPT_SOURCE", "MAGPIE_VLLM_SCRIPT",
                    "MAGPIE_SGLANG_SCRIPT", "MAX_MODEL_LEN",
                    "RECIPE_ENV_FILE", "RECIPE_ENV_SOURCE", "RECIPE_ENV_REPLAYED",
                    "RECIPE_ENV_GEAK_OWNED", "GEAK_STRICT_RECIPE_ENV",
                    "EFFECTIVE_SERVER_ARGS_COMPLETE"):
            os.environ.pop(key, None)

    def _recipe(self, *, script="vllm_mi355x.sh", subdir="", root=None,
                with_lib=True, write_script=True):
        """An orchestrator launch recipe next to a checkout it can point at.

        ``write_script=False`` + ``with_lib=False`` leaves the checkout path
        untouched so callers can point the recipe at a path that does not exist
        on this box (the orchestrator's host-local checkout). Creating under
        ``/nonexistent/...`` would PermissionError on CI.
        """
        checkout = Path(root) if root else self.tmp / "InferenceX@abc123"
        bench_dir = checkout / "benchmarks" / subdir if subdir else checkout / "benchmarks"
        if write_script or with_lib:
            bench_dir.mkdir(parents=True, exist_ok=True)
        if write_script:
            (bench_dir / script).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        if with_lib:
            (bench_dir / "benchmark_lib.sh").write_text("# lib\n", encoding="utf-8")
        recipe = self.tmp / "baseline_config.with_envs.yaml"
        recipe.write_text(
            "benchmark:\n"
            "  framework: vllm\n"
            "  model: /models/Qwen3-8B\n"
            "  envs:\n"
            "    TP: 1\n"
            "    MAX_MODEL_LEN: 6144\n"
            "  runner_type: mi355x\n"
            f"  benchmark_script: {script}\n"
            f"  inferencex_path: {checkout}\n",
            encoding="utf-8",
        )
        return str(recipe), str(bench_dir / script)

    def test_handoff_script_enables_magpie_and_normalises_the_env_var(self):
        launcher = rx.apply_bench_launcher(
            {"launch_server_script": "/magpie/launch.sh", "framework": "vllm"}
        )
        self.assertEqual(launcher, "magpie")
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], "/magpie/launch.sh")
        self.assertEqual(os.environ["BENCH_LAUNCHER"], "magpie")

    def test_schema_v2_marks_resolved_args_complete_without_launch_flags(self):
        recipe, _ = self._recipe()
        rx.apply_bench_launcher({
            "schema_version": 2,
            "baseline_env_spec": {"config": {"server_launch_flags": ""}},
            "launch_recipe": recipe,
            "framework": "vllm",
            "eval_dir": str(self.tmp / "eval"),
        })
        self.assertEqual(os.environ["EFFECTIVE_SERVER_ARGS_COMPLETE"], "1")

    def test_legacy_launcher_clears_complete_args_marker(self):
        os.environ["EFFECTIVE_SERVER_ARGS_COMPLETE"] = "1"
        rx.apply_bench_launcher(
            {"launch_server_script": "/magpie/launch.sh", "framework": "vllm"}
        )
        self.assertNotIn("EFFECTIVE_SERVER_ARGS_COMPLETE", os.environ)

    def test_per_backend_env_script_is_discovered(self):
        os.environ["MAGPIE_SGLANG_SCRIPT"] = "/magpie/sglang.sh"
        self.assertEqual(rx.apply_bench_launcher({"framework": "sglang"}), "magpie")
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], "/magpie/sglang.sh")

    def test_unsupported_backend_keeps_native_even_with_a_script(self):
        launcher = rx.apply_bench_launcher(
            {"launch_server_script": "/magpie/launch.sh", "framework": "trtllm"}
        )
        self.assertEqual(launcher, "native")
        # The script is still normalised so a manual launcher can find it.
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], "/magpie/launch.sh")

    def test_explicit_request_overrides_discovery(self):
        launcher = rx.apply_bench_launcher({
            "bench_launcher": "Native",
            "launch_server_script": "/magpie/launch.sh",
            "framework": "vllm",
        })
        self.assertEqual(launcher, "native")

    def test_auto_request_still_discovers(self):
        os.environ["MAGPIE_LAUNCH_SCRIPT"] = "/magpie/generic.sh"
        self.assertEqual(
            rx.apply_bench_launcher({"bench_launcher": "auto", "framework": "vllm"}),
            "magpie",
        )

    def test_no_script_anywhere_is_native(self):
        self.assertEqual(rx.apply_bench_launcher({"framework": "vllm"}), "native")
        self.assertNotIn("MAGPIE_LAUNCH_SCRIPT", os.environ)

    # -- deriving the script from the recipe the orchestrator did send -------- #
    def test_the_launch_script_is_derived_from_the_recipe(self):
        """No handoff has ever named the launch script, but every one names the
        recipe — and the recipe names both the checkout and the script."""
        recipe, script = self._recipe()

        launcher = rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm", "max_model_len": 6144}
        )

        self.assertEqual(launcher, "magpie")
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], script)
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT_SOURCE"], "launch_recipe")

    def test_a_script_in_a_checkout_subdirectory_is_found(self):
        recipe, script = self._recipe(subdir="single_node")

        rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"})

        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], script)

    def test_an_explicit_script_outranks_the_recipe(self):
        recipe, _ = self._recipe()

        rx.apply_bench_launcher({
            "launch_recipe": recipe,
            "launch_server_script": "/magpie/explicit.sh",
            "framework": "vllm",
        })

        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], "/magpie/explicit.sh")
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT_SOURCE"], "handoff")

    def test_a_checkout_missing_its_script_library_degrades_to_native(self):
        """The script sources benchmark_lib.sh from its own directory and dies
        without it. Better to serve natively than to fail every bench."""
        recipe, _ = self._recipe(with_lib=False)

        launcher = rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"})

        self.assertEqual(launcher, "native")
        self.assertNotIn("MAGPIE_LAUNCH_SCRIPT", os.environ)

    def test_an_unreachable_checkout_degrades_to_native(self):
        """The recipe is written on the orchestrator's box; its checkout path
        need not exist on ours."""
        # Under self.tmp but never created — avoids mkdir('/nonexistent') which
        # PermissionErrors on GitHub Actions while still being a missing path.
        missing = self.tmp / "InferenceX@dead"
        self.assertFalse(missing.exists())
        recipe, _ = self._recipe(
            root=str(missing), write_script=False, with_lib=False
        )

        self.assertEqual(
            rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"}),
            "native",
        )

    # -- the recipe's checkout is host-local; the handoff's usually is not ---- #
    def _checkout(self, name, script="vllm_mi355x.sh"):
        """A usable InferenceX checkout that no recipe points at."""
        bench_dir = self.tmp / name / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        (bench_dir / script).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (bench_dir / "benchmark_lib.sh").write_text("# lib\n", encoding="utf-8")
        return str(self.tmp / name), str(bench_dir / script)

    def test_a_stale_recipe_checkout_falls_back_to_the_handoff_checkout(self):
        """The recipe names a container-local, content-hash addressed path. A
        rebuild invalidates it while the same checkout survives on durable
        storage — which the handoff names, and which the bench client already
        resolves against."""
        recipe, _ = self._recipe(root=str(self.tmp / "stale"),
                                 write_script=False, with_lib=False)
        survivor, script = self._checkout("durable")

        launcher = rx.apply_bench_launcher(
            {"launch_recipe": recipe, "inferencex_path": survivor,
             "framework": "vllm"}
        )

        self.assertEqual(launcher, "magpie")
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], script)
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT_SOURCE"], "launch_recipe")

    def test_the_recipe_checkout_wins_whenever_it_still_resolves(self):
        """The recipe's path is the only one the orchestrator provably launched
        from, so a fallback must never displace it."""
        recipe, recipe_script = self._recipe()
        other, other_script = self._checkout("durable")

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "inferencex_path": other, "framework": "vllm"}
        )

        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], recipe_script)
        self.assertNotEqual(recipe_script, other_script)

    def test_the_inferencex_path_env_is_the_last_resort(self):
        recipe, _ = self._recipe(root=str(self.tmp / "stale"),
                                 write_script=False, with_lib=False)
        survivor, script = self._checkout("from_env")
        os.environ["INFERENCEX_PATH"] = survivor
        self.addCleanup(os.environ.pop, "INFERENCEX_PATH", None)

        self.assertEqual(
            rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"}),
            "magpie",
        )
        self.assertEqual(os.environ["MAGPIE_LAUNCH_SCRIPT"], script)

    def test_no_usable_checkout_anywhere_still_degrades_to_native(self):
        recipe, _ = self._recipe(root=str(self.tmp / "stale"),
                                 write_script=False, with_lib=False)

        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_recipe": recipe,
                 "inferencex_path": str(self.tmp / "also_gone"),
                 "framework": "vllm"}
            ),
            "native",
        )
        self.assertNotIn("MAGPIE_LAUNCH_SCRIPT", os.environ)

    # -- replaying the orchestrator's recorded launch environment -------------
    # The launcher running the same SCRIPT only aligns the two servers where
    # their ${X:-default} expansions agree. These pin the stronger property: what
    # the orchestrator recorded is applied, what GEAK must own is not, and a
    # recipe that recorded nothing refuses to launch rather than quietly falling
    # back onto defaults that merely happened to match once.

    def _recipe_with_envs(self, envs: str):
        checkout = self.tmp / "InferenceX@abc123"
        bench = checkout / "benchmarks"
        bench.mkdir(parents=True, exist_ok=True)
        (bench / "vllm_mi355x.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (bench / "benchmark_lib.sh").write_text("# lib\n", encoding="utf-8")
        recipe = self.tmp / "baseline_config.with_envs.yaml"
        recipe.write_text(
            "benchmark:\n"
            "  framework: vllm\n"
            f"{envs}"
            "  runner_type: mi355x\n"
            "  benchmark_script: vllm_mi355x.sh\n"
            f"  inferencex_path: {checkout}\n",
            encoding="utf-8",
        )
        return str(recipe)

    def test_the_recorded_env_is_replayed_and_run_scoped_names_are_not(self):
        # Use real directories for PATH so existence filtering keeps them.
        venv_bin = self.tmp / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        recipe = self._recipe_with_envs(
            "  envs:\n"
            f"    PATH: {venv_bin}:/usr/bin\n"
            "    VLLM_ROCM_USE_AITER: '1'\n"
            "    NUM_WARMUPS: 8\n"
            "    PORT: 8844\n"                    # GEAK-owned: this run's port
            "    ROCR_VISIBLE_DEVICES: '0'\n"     # GEAK-owned: this run's GPU
        )
        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_recipe": recipe, "framework": "vllm",
                 "eval_dir": str(self.tmp / "eval")}
            ),
            "magpie",
        )
        replayed = os.environ["RECIPE_ENV_REPLAYED"].split()
        self.assertEqual(replayed, ["NUM_WARMUPS", "PATH", "VLLM_ROCM_USE_AITER"])
        self.assertEqual(
            os.environ["RECIPE_ENV_GEAK_OWNED"].split(),
            ["PORT", "ROCR_VISIBLE_DEVICES"],
        )
        # PATH carries no spaces here, but the file must still be NUL-delimited
        # so that a value which does carry one survives the launcher's env call.
        blob = Path(os.environ["RECIPE_ENV_FILE"]).read_bytes()
        expected_path = f"PATH={venv_bin}:/usr/bin".encode()
        self.assertEqual(
            blob.split(b"\0")[:-1],
            [b"NUM_WARMUPS=8", expected_path, b"VLLM_ROCM_USE_AITER=1"],
        )

    def test_missing_path_components_are_dropped_from_replay(self):
        """A host-local venv prefix that is gone on this box must not lead PATH."""
        alive = self.tmp / "alive_bin"
        alive.mkdir()
        dead = self.tmp / "dead_venv" / "bin"   # never created
        recipe = self._recipe_with_envs(
            "  envs:\n"
            f"    PATH: {dead}:{alive}:/usr/bin\n"
            "    VLLM_ROCM_USE_AITER: '1'\n"
        )

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm",
             "eval_dir": str(self.tmp / "eval")}
        )

        blob = Path(os.environ["RECIPE_ENV_FILE"]).read_bytes()
        self.assertIn(f"PATH={alive}:/usr/bin".encode(), blob)
        self.assertNotIn(str(dead).encode(), blob)

    def test_a_fully_missing_path_is_omitted_from_replay(self):
        """Nothing usable left -> keep the ambient PATH rather than export junk."""
        dead = self.tmp / "no_such_venv" / "bin"
        recipe = self._recipe_with_envs(
            "  envs:\n"
            f"    PATH: {dead}\n"
            "    VLLM_ROCM_USE_AITER: '1'\n"
        )

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm",
             "eval_dir": str(self.tmp / "eval")}
        )

        self.assertEqual(
            os.environ["RECIPE_ENV_REPLAYED"].split(), ["VLLM_ROCM_USE_AITER"]
        )
        self.assertNotIn(b"PATH=", Path(os.environ["RECIPE_ENV_FILE"]).read_bytes())

    def test_the_replay_says_nothing_on_stdout(self):
        """Callers eval this function's stdout as shell (run_ab.sh does), so a
        progress line on stdout is executed as a command and kills the run."""
        recipe = self._recipe_with_envs("  envs:\n    VLLM_ROCM_USE_AITER: '1'\n")

        with contextlib.redirect_stdout(io.StringIO()) as out:
            rx.apply_bench_launcher(
                {"launch_recipe": recipe, "framework": "vllm",
                 "eval_dir": str(self.tmp / "eval")}
            )
        self.assertEqual(out.getvalue(), "")

    def test_what_the_launcher_overrides_is_not_reported_as_replayed(self):
        """The launcher passes MODEL and TP positionally, after the replay, so
        the recipe's values lose. Reporting them as replayed would name this run
        as inheriting a model and a shard count it does not actually serve."""
        recipe = self._recipe_with_envs(
            "  envs:\n"
            "    MODEL: /models/Some-Other-Model\n"
            "    TP: 8\n"
            "    VLLM_ROCM_USE_AITER: '1'\n"
        )

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm",
             "eval_dir": str(self.tmp / "eval")}
        )

        self.assertEqual(
            os.environ["RECIPE_ENV_REPLAYED"].split(), ["VLLM_ROCM_USE_AITER"]
        )
        self.assertEqual(os.environ["RECIPE_ENV_GEAK_OWNED"].split(), ["MODEL", "TP"])
        self.assertNotIn(b"TP=8", Path(os.environ["RECIPE_ENV_FILE"]).read_bytes())

    def test_strict_mode_accepts_an_env_block_containing_only_owned_names(self):
        """A complete, known override set needs no replay file and is not absent."""
        recipe = self._recipe_with_envs(
            "  envs:\n"
            "    MODEL: /models/reference\n"
            "    TP: 8\n"
            "    PORT: 9000\n"
            "    ROCR_VISIBLE_DEVICES: 7\n"
        )
        os.environ["GEAK_STRICT_RECIPE_ENV"] = "1"

        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_recipe": recipe, "framework": "vllm",
                 "eval_dir": str(self.tmp / "eval")}
            ),
            "magpie",
        )
        self.assertNotIn("RECIPE_ENV_FILE", os.environ)
        self.assertEqual(os.environ["RECIPE_ENV_SOURCE"], recipe)
        self.assertEqual(os.environ["RECIPE_ENV_REPLAYED"], "")
        self.assertEqual(
            os.environ["RECIPE_ENV_GEAK_OWNED"].split(),
            ["MODEL", "PORT", "ROCR_VISIBLE_DEVICES", "TP"],
        )

    def test_a_nested_map_under_envs_contributes_no_variables(self):
        """Its children are keys of a structure, not names any shell exports;
        flattening them would invent variables the orchestrator never set."""
        recipe = self._recipe_with_envs(
            "  envs:\n"
            "    VLLM_ROCM_USE_AITER: '1'\n"
            "    sweep:\n"
            "      ISL: 128\n"
            "      nested_deeper:\n"
            "        OSL: 256\n"
            "    NUM_WARMUPS: 8\n"          # the block resumes after the subtree
        )

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm",
             "eval_dir": str(self.tmp / "eval")}
        )

        self.assertEqual(
            os.environ["RECIPE_ENV_REPLAYED"].split(),
            ["NUM_WARMUPS", "VLLM_ROCM_USE_AITER"],
        )

    def test_a_recipe_recording_no_env_warns_but_launches(self):
        recipe = self._recipe_with_envs("")

        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_recipe": recipe, "framework": "vllm",
                 "eval_dir": str(self.tmp / "eval")}
            ),
            "magpie",
        )
        self.assertNotIn("RECIPE_ENV_FILE", os.environ)

    def test_strict_mode_refuses_when_no_env_is_recorded(self):
        recipe = self._recipe_with_envs("")
        os.environ["GEAK_STRICT_RECIPE_ENV"] = "1"

        with self.assertRaises(SystemExit) as caught:
            rx.apply_bench_launcher(
                {"launch_recipe": recipe, "framework": "vllm",
                 "eval_dir": str(self.tmp / "eval")}
            )
        self.assertIn("records no envs", str(caught.exception))

    def test_strict_mode_also_applies_when_script_source_is_handoff(self):
        """An explicit script must not bypass strictness for its launch recipe."""
        recipe = self._recipe_with_envs("")
        os.environ["GEAK_STRICT_RECIPE_ENV"] = "1"
        with self.assertRaises(SystemExit) as caught:
            rx.apply_bench_launcher({
                "launch_recipe": recipe,
                "launch_server_script": "/magpie/explicit.sh",
                "framework": "vllm",
                "eval_dir": str(self.tmp / "eval"),
            })
        self.assertIn("records no envs", str(caught.exception))

    def test_a_script_named_by_the_handoff_is_not_held_to_the_recipe(self):
        # Nothing was derived from a recipe, so there is no recorded env to be
        # missing and fail-closed must not fire.
        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_server_script": "/magpie/launch.sh", "framework": "vllm"}
            ),
            "magpie",
        )
        self.assertNotIn("RECIPE_ENV_FILE", os.environ)

    def test_the_recorded_max_model_len_outranks_the_handoff_scalar(self):
        recipe = self._recipe_with_envs("  envs:\n    MAX_MODEL_LEN: 6144\n")

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm", "max_model_len": 4096,
             "eval_dir": str(self.tmp / "eval")}
        )
        # Cleared, so the launcher's pass-through cannot land on top of the
        # replayed value; the recipe's 6144 reaches the script via the replay.
        self.assertNotIn("MAX_MODEL_LEN", os.environ)
        self.assertIn(b"MAX_MODEL_LEN=6144",
                      Path(os.environ["RECIPE_ENV_FILE"]).read_bytes())

    def test_replay_file_is_atomically_replaced_without_temp_residue(self):
        recipe = self._recipe_with_envs("  envs:\n    FOO: new value\n")
        eval_dir = self.tmp / "eval"
        eval_dir.mkdir()
        replay = eval_dir / "recipe_env.nul"
        replay.write_bytes(b"FOO=old\0")

        rx.apply_bench_launcher({
            "launch_recipe": recipe,
            "framework": "vllm",
            "eval_dir": str(eval_dir),
        })

        self.assertEqual(replay.read_bytes(), b"FOO=new value\0")
        self.assertEqual(list(eval_dir.glob(".recipe_env.*.tmp")), [])

    def test_a_missing_recipe_file_is_not_an_error(self):
        self.assertEqual(rx._recipe_fields("/nonexistent/recipe.yaml"), {})
        self.assertEqual(
            rx.apply_bench_launcher(
                {"launch_recipe": "/nonexistent/recipe.yaml", "framework": "vllm"}
            ),
            "native",
        )

    def test_the_escape_hatch_outranks_a_derivable_recipe(self):
        recipe, _ = self._recipe()
        os.environ["BENCH_LAUNCHER"] = "native"

        self.assertEqual(
            rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"}),
            "native",
        )

    # -- max-model-len reaches the script as env, not as a duplicate flag ----- #
    def test_max_model_len_reaches_the_magpie_script(self):
        """Magpie's script defaults max-model-len to 4096. Left unset, the right
        value only arrives as a duplicate flag that wins on argparse ordering.

        Which CHANNEL carries it depends on whether the recipe recorded it: the
        replay when it did, the pass-through when it did not. Asserting the
        outcome rather than the channel keeps this pinned to the property that
        matters -- 6144 reaches the script exactly once.
        """
        recipe, _ = self._recipe()   # records MAX_MODEL_LEN: 6144

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm", "max_model_len": 6144,
             "eval_dir": str(self.tmp / "eval")}
        )

        via_replay = b"MAX_MODEL_LEN=6144" in Path(
            os.environ.get("RECIPE_ENV_FILE", os.devnull)).read_bytes()
        via_passthrough = os.environ.get("MAX_MODEL_LEN") == "6144"
        self.assertTrue(via_replay or via_passthrough)
        self.assertFalse(via_replay and via_passthrough, "must not arrive twice")

    def test_max_model_len_falls_back_to_the_handoff_when_unrecorded(self):
        """A recipe that records an env but not this one still gets the value."""
        recipe = self._recipe_with_envs("  envs:\n    TP: 1\n")

        rx.apply_bench_launcher(
            {"launch_recipe": recipe, "framework": "vllm", "max_model_len": 6144,
             "eval_dir": str(self.tmp / "eval")}
        )

        self.assertEqual(os.environ["MAX_MODEL_LEN"], "6144")

    def test_max_model_len_is_not_forwarded_on_the_native_path(self):
        """Nothing on the native path reads it, and exporting it would leak a
        server knob into an unrelated launch."""
        rx.apply_bench_launcher({"framework": "vllm", "max_model_len": 6144})

        self.assertEqual(rx.apply_bench_launcher({"framework": "vllm"}), "native")
        self.assertNotIn("MAX_MODEL_LEN", os.environ)

    def test_an_absent_max_model_len_leaves_the_script_default_alone(self):
        """The script's own default IS what the orchestrator served with."""
        recipe, _ = self._recipe()

        rx.apply_bench_launcher({"launch_recipe": recipe, "framework": "vllm"})

        self.assertNotIn("MAX_MODEL_LEN", os.environ)

    def test_the_recipe_parser_ignores_the_nested_env_map(self):
        recipe, _ = self._recipe()

        fields = rx._recipe_fields(recipe)

        self.assertEqual(fields["benchmark_script"], "vllm_mi355x.sh")
        self.assertEqual(fields["framework"], "vllm")
        self.assertEqual(fields["runner_type"], "mi355x")
        self.assertNotIn("MAX_MODEL_LEN", fields)
        self.assertNotIn("TP", fields)


class TestBenchProtocol(_RunE2ECase):
    def test_only_provided_keys_are_exported(self):
        """bench_e2e.sh keeps its own defaults for anything the orchestrator did
        not pin; exporting a blank would override a good default with nothing."""
        for var in rx._BENCH_PROTOCOL_ENV.values():
            os.environ.pop(var, None)
        exported = rx.apply_bench_protocol({"bench_protocol": {
            "random_range_ratio": 0.0,
            "num_prompts": 256,
            "num_warmups": None,
            "seed": "   ",
            "unknown_knob": "ignored",
        }})
        self.assertEqual(
            exported, {"RANDOM_RANGE_RATIO": "0.0", "NUM_PROMPTS": "256"}
        )
        self.assertEqual(os.environ["RANDOM_RANGE_RATIO"], "0.0")
        self.assertEqual(os.environ["NUM_PROMPTS"], "256")
        self.assertNotIn("NUM_WARMUPS", os.environ)
        self.assertNotIn("SEED", os.environ)

    def test_absent_protocol_exports_nothing(self):
        self.assertEqual(rx.apply_bench_protocol({}), {})

    def test_non_dict_protocol_is_ignored(self):
        self.assertEqual(rx.apply_bench_protocol({"bench_protocol": "0.5"}), {})

    def test_schema_v2_uses_actual_hyperloom_wrapper_protocol(self):
        """Historical metadata cannot override the wrapper's 2*conc policy."""
        exported = rx.apply_bench_protocol({
            "schema_version": 2,
            "workload": {"conc": 64},
            "baseline_env_spec": {"config": {}},
            "bench_protocol": {
                "random_range_ratio": 0,
                "num_prompts": 192,
                "num_warmups": 8,
            },
        })
        self.assertEqual(exported["NUM_PROMPTS"], "192")
        self.assertEqual(exported["NUM_WARMUPS"], "128")
        self.assertEqual(exported["SEED"], "0")
        self.assertEqual(exported["RANDOM_RANGE_RATIO"], "1")
        self.assertEqual(exported["GEAK_REPEAT_MODE"], "warm_server")
        self.assertEqual(exported["GEAK_VALIDATION_REPEAT_MODE"], "warm_server")
        self.assertEqual(exported["REPLICA_RETRIES"], "1")
        # A pinned count is the caller telling us what it measured.
        self.assertNotIn("NUM_PROMPTS_ADAPTIVE", exported)

    def test_schema_v2_without_a_pinned_count_aligns_the_prompt_count(self):
        """NUM_PROMPTS was the one protocol knob this block left unaligned, and
        the two defaults disagree: bench_e2e.sh falls back to Magpie's fixed
        CONC*10 while the caller scales DOWN with sequence cost. Switching on
        the adaptive table (which is that same rule) is the alignment."""
        exported = rx.apply_bench_protocol({
            "schema_version": 2,
            "workload": {"conc": 64},
            "baseline_env_spec": {"config": {}},
            "bench_protocol": {"random_range_ratio": 0},
        })
        self.assertEqual(exported["NUM_PROMPTS_ADAPTIVE"], "1")
        self.assertNotIn("NUM_PROMPTS", exported)


class TestAlignmentFlags(_RunE2ECase):
    def test_cold_final_defaults_off(self):
        os.environ.pop("BENCH_COLD_FINAL", None)
        self.assertEqual(rx.apply_alignment_flags({}), {"BENCH_COLD_FINAL": "0"})
        self.assertEqual(os.environ["BENCH_COLD_FINAL"], "0")

    def test_explicit_truthy_handoff_enables(self):
        self.assertEqual(
            rx.apply_alignment_flags({"bench_cold_final": "1"}),
            {"BENCH_COLD_FINAL": "1"},
        )

    def test_explicit_falsey_handoff_disables(self):
        self.assertEqual(
            rx.apply_alignment_flags({"bench_cold_final": "0"}),
            {"BENCH_COLD_FINAL": "0"},
        )

    def test_env_value_is_used_when_the_handoff_is_silent(self):
        os.environ["BENCH_COLD_FINAL"] = "yes"
        self.assertEqual(rx.apply_alignment_flags({}), {"BENCH_COLD_FINAL": "1"})
        # An empty env var carries no intent, so the default (off) applies.
        os.environ["BENCH_COLD_FINAL"] = ""
        self.assertEqual(rx.apply_alignment_flags({}), {"BENCH_COLD_FINAL": "0"})


# =========================================================================== #
# transcript scraping
# =========================================================================== #
class TestIterMessageText(_RunE2ECase):
    def test_every_fragment_shape_is_collected_in_order(self):
        block = _Msg(text="block-text")
        inner_attr = _Msg(text="inner-attr")
        msg = _Msg(
            text="flat-text",
            result="flat-result",
            content=[
                block,
                {"text": "dict-text", "content": "inner-str"},
                {"content": [inner_attr, {"text": "inner-dict"}]},
                {"text": "   "},
            ],
        )
        self.assertEqual(rx._iter_message_text(msg), [
            "flat-text", "flat-result", "block-text",
            "dict-text", "inner-str", "inner-attr", "inner-dict",
        ])

    def test_plain_string_content(self):
        self.assertEqual(
            rx._iter_message_text(_Msg(content="just a string")), ["just a string"]
        )

    def test_dict_shaped_message(self):
        self.assertEqual(
            rx._iter_message_text({"text": "d-text", "result": "d-result"}),
            ["d-text", "d-result"],
        )

    def test_message_with_nothing_extractable(self):
        self.assertEqual(rx._iter_message_text(_Msg(task_id="t1")), [])
        self.assertEqual(rx._iter_message_text(object()), [])


class TestJsonScrape(_RunE2ECase):
    RAW = (
        "Some prose {oops} more\n"
        "```json\n"
        "{\n"
        '  "eval_dir": "/run/a",\n'
        '  "note": "brace { inside \\"quoted\\" text"\n'
        "}\n"
        "```\n"
        'Trailing prose {"eval_dir": "/run/b", "throughput_speedup": 1.2}\n'
        "{nope}\n"
        '{"eval_dir": "/run/c", "throughput_speedup": 1.4}\n'
    )

    def test_multiline_fenced_and_inline_objects_are_all_recovered(self):
        objs = list(rx._iter_json_objects(self.RAW))
        self.assertEqual(
            [o["eval_dir"] for o in objs], ["/run/a", "/run/b", "/run/c", "/run/c"]
        )
        # The brace inside a JSON string never split the span.
        self.assertEqual(objs[0]["note"], 'brace { inside "quoted" text')

    def test_last_eval_dir_bearing_object_wins(self):
        parsed = rx._parse_last_json_line(self.RAW)
        self.assertEqual(parsed["eval_dir"], "/run/c")
        self.assertEqual(parsed["throughput_speedup"], 1.4)

    def test_objects_without_eval_dir_are_not_the_return(self):
        raw = '{"status": "ok"}\n{"eval_dir": "", "x": 1}\n'
        with self.assertRaises(rx.WorkflowParseError) as ctx:
            rx._parse_last_json_line(raw)
        self.assertIn("Could not parse a JSON workflow return", str(ctx.exception))
        self.assertIn('{"status": "ok"}', str(ctx.exception))

    def test_empty_transcript_raises_parse_error(self):
        self.assertEqual(list(rx._iter_json_objects("")), [])
        with self.assertRaises(rx.WorkflowParseError):
            rx._parse_last_json_line(None)

    def test_objects_nested_in_other_containers_are_still_recovered(self):
        """The scan is brace-matched, not line-oriented, so a return wrapped in a
        JSON array (or trailed by stray punctuation) is still found."""
        self.assertEqual(
            [o["eval_dir"] for o in rx._iter_json_objects('[{"eval_dir": "/x"}] }')],
            ["/x"],
        )
        self.assertEqual(list(rx._iter_json_objects("no braces here")), [])


class TestClassifyError(_RunE2ECase):
    def test_stable_error_classes(self):
        """Hyperloom's session breakdown attributes misses by these exact
        strings; renaming one silently reclassifies every historical miss."""
        self.assertEqual(rx._classify_error(TimeoutError("budget")), "timeout")
        self.assertEqual(
            rx._classify_error(rx.WorkflowParseError("no json")), "workflow_parse_error"
        )
        self.assertEqual(
            rx._classify_error(ImportError("no claude_agent_sdk")), "sdk_import_failed"
        )
        self.assertEqual(
            rx._classify_error(RuntimeError("claude CLI failed (rc=2): boom")),
            "cli_failed",
        )
        self.assertEqual(rx._classify_error(ValueError("other")), "runner_error")


# =========================================================================== #
# SDK invocation + completion gate
# =========================================================================== #
class TestInvokeViaSdk(_RunE2ECase):
    def _install(self, script, *, sleep_hook=None, **sdk_kw):
        anyio = _make_fake_anyio(sleep_hook=sleep_hook)
        sdk = _make_fake_sdk(script, **sdk_kw)
        self.install_module("anyio", anyio)
        self.install_module("claude_agent_sdk", sdk)
        return anyio, sdk

    def test_synchronous_in_turn_path_stops_on_the_result_message(self):
        """No background task was ever spawned => the turn's ResultMessage is
        itself terminal; nothing after it may be consumed."""
        never = AssistantMessage(text="must-not-be-consumed")
        script = [
            AssistantMessage(content=[{"text": '{"eval_dir": "/run/sync"}'}]),
            ResultMessage(result="done"),
            never,
        ]
        anyio, sdk = self._install(script)
        raw = rx._invoke_via_sdk("PROMPT", 900, str(self.tmp / "e2e_never"))
        self.assertEqual(raw, '{"eval_dir": "/run/sync"}\ndone')
        self.assertNotIn(never, sdk.state["consumed"])
        self.assertEqual(sdk.state["prompts"], ["PROMPT"])
        self.assertEqual(anyio.calls["fail_after"], [900])
        self.assertEqual(anyio.calls["sleep"], [])
        self.assertTrue(sdk.state["clients"][0].closed)

    def test_on_disk_terminal_marker_ends_the_stream_immediately(self):
        """The workflow's own marker is the authoritative done signal — it wins
        even mid-turn, before any ResultMessage."""
        eval_dir = self.tmp / "e2e_marked"
        eval_dir.mkdir()
        (eval_dir / rx.WORKFLOW_RETURN_FILE).write_text("{}", encoding="utf-8")
        later = ResultMessage(result="later")
        _anyio, sdk = self._install([AssistantMessage(text="first"), later])
        raw = rx._invoke_via_sdk("P", 60, str(eval_dir))
        self.assertEqual(raw, "first")
        self.assertNotIn(later, sdk.state["consumed"])

    def test_background_task_keeps_the_client_open_until_the_marker_lands(self):
        """The killer case: a task notified terminal and the turn ended, but the
        detached A/B is still running. The runner must poll for the marker
        instead of tearing the workflow down."""
        eval_dir = self.tmp / "e2e_bg"
        eval_dir.mkdir()
        out_file = self.tmp / "task_output.json"
        out_file.write_text('{"eval_dir": "/run/bg"}', encoding="utf-8")

        def land_marker():
            (eval_dir / "director_e2e_validation.json").write_text(
                "{}", encoding="utf-8")

        script = [
            TaskStartedMessage(task_id="t1"),
            AssistantMessage(text="still working"),
            TaskNotificationMessage(
                task_id="t1", output_file=str(out_file), summary="task complete"
            ),
            ResultMessage(result="turn done"),
        ]
        anyio, _sdk = self._install(script, sleep_hook=land_marker)
        raw = rx._invoke_via_sdk("P", 1200, str(eval_dir))
        self.assertEqual(raw.splitlines(), [
            "still working", '{"eval_dir": "/run/bg"}', "task complete", "turn done",
        ])
        # Exactly one grace poll: the marker landed on the first sleep.
        self.assertEqual(anyio.calls["sleep"], [rx.DONE_POLL_S])
        self.assertTrue((eval_dir / "director_e2e_validation.json").is_file())
        # The task's output_file is what carries the return on the background
        # path, so the scrape must still find it.
        self.assertEqual(rx._parse_last_json_line(raw)["eval_dir"], "/run/bg")

    def test_unreadable_task_output_file_is_tolerated(self):
        eval_dir = self.tmp / "e2e_badout"
        eval_dir.mkdir()
        script = [
            TaskStartedMessage(task_id="t1"),
            TaskNotificationMessage(
                task_id="t1", output_file=str(self.tmp / "missing.json"), summary=" "
            ),
            ResultMessage(result="turn done"),
        ]
        self.patch_rx("DONE_GRACE_S", 0.0)
        anyio, _sdk = self._install(script)
        raw = rx._invoke_via_sdk("P", 30, str(eval_dir))
        self.assertEqual(raw, "turn done")
        self.assertEqual(anyio.calls["sleep"], [])

    def test_task_without_an_id_is_not_tracked_as_pending(self):
        script = [
            TaskStartedMessage(task_id=None),
            ResultMessage(result="done"),
            AssistantMessage(text="unreachable"),
        ]
        _anyio, _sdk = self._install(script)
        self.assertEqual(rx._invoke_via_sdk("P", 30, None), "done")

    def test_options_carry_effort_sandbox_env_and_pinned_cli(self):
        """These four are what make the Workflow tool actually execute the JS
        pipeline (rather than the agent 'backgrounding' it) under root."""
        self.patch_rx("CLAUDE_EFFORT", "high")
        self.patch_rx("CLAUDE_BIN", "/opt/claude-2.1.181/claude")
        _anyio, sdk = self._install([ResultMessage(result="done")])
        real_geteuid = os.geteuid
        os.geteuid = lambda: 0
        try:
            rx._invoke_via_sdk("P", 30, None)
        finally:
            os.geteuid = real_geteuid
        kwargs = sdk.state["options"][0]
        self.assertEqual(kwargs["model"], rx.CLAUDE_MODEL)
        self.assertEqual(kwargs["allowed_tools"], rx.ALLOWED_TOOLS)
        self.assertEqual(kwargs["permission_mode"], "bypassPermissions")
        self.assertEqual(kwargs["settings"], rx.WORKFLOW_SETTINGS)
        self.assertEqual(kwargs["cwd"], str(rx.E2E_DIR))
        self.assertEqual(kwargs["extra_args"], {"effort": "high"})
        self.assertEqual(kwargs["env"], {"IS_SANDBOX": "1"})
        self.assertEqual(kwargs["cli_path"], "/opt/claude-2.1.181/claude")

    def test_public_build_effort_is_not_forwarded(self):
        """'ultracode' is rejected by public claude builds; it must be carried by
        the settings layer only, never as --effort."""
        self.patch_rx("CLAUDE_EFFORT", "ultracode")
        self.patch_rx("CLAUDE_BIN", "")
        _anyio, sdk = self._install([ResultMessage(result="done")])
        rx._invoke_via_sdk("P", 30, None)
        kwargs = sdk.state["options"][0]
        self.assertEqual(kwargs["extra_args"], {})
        self.assertNotIn("cli_path", kwargs)
        self.assertIn("enableWorkflows", kwargs["settings"])

    def test_legacy_sdk_without_streaming_client_uses_one_shot_query(self):
        """An old SDK has no ClaudeSDKClient; the runner must still drive the
        synchronous in-turn path rather than failing to import."""
        query_script = [
            AssistantMessage(text='{"eval_dir": "/run/legacy"}'),
            ResultMessage(result="ok"),
        ]
        anyio, sdk = self._install(
            None, with_client=False, query_script=query_script
        )
        raw = rx._invoke_via_sdk("LEGACY-PROMPT", 45, None)
        self.assertEqual(raw, '{"eval_dir": "/run/legacy"}\nok')
        self.assertEqual(sdk.state["prompts"], ["LEGACY-PROMPT"])
        self.assertEqual(anyio.calls["fail_after"], [45])


class TestWorkflowDoneOnDisk(_RunE2ECase):
    def test_missing_eval_dir_is_never_done(self):
        self.assertFalse(rx._workflow_done_on_disk(None))
        self.assertFalse(rx._workflow_done_on_disk(""))

    def test_either_terminal_marker_counts(self):
        d = self.tmp / "e2e_m"
        d.mkdir()
        self.assertFalse(rx._workflow_done_on_disk(str(d)))
        (d / "director_e2e_validation.json").write_text("{}", encoding="utf-8")
        self.assertTrue(rx._workflow_done_on_disk(str(d)))


# =========================================================================== #
# CLI fallback
# =========================================================================== #
class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestInvokeViaCli(_RunE2ECase):
    def _fake_subprocess(self, proc):
        calls = []

        def run(cmd, **kw):
            calls.append((cmd, kw))
            return proc

        self.patch_rx("subprocess", types.SimpleNamespace(run=run))
        return calls

    def _fake_which(self, resolved):
        self.patch_rx("shutil", types.SimpleNamespace(which=lambda name: resolved))

    def test_marshalled_argv_is_exactly_the_workflow_contract(self):
        """The Workflow/parallel/phase primitives are gated behind --settings; a
        dropped flag turns the run into a no-op agent chat."""
        self.patch_rx("CLAUDE_EFFORT", "max")
        self._fake_which("/usr/bin/claude")
        calls = self._fake_subprocess(_FakeProc(stdout='{"result": "SCRAPED"}'))
        out = rx._invoke_via_cli("THE PROMPT", 777)
        self.assertEqual(out, "SCRAPED")
        cmd, kw = calls[0]
        self.assertEqual(cmd, [
            "/usr/bin/claude", "-p", "THE PROMPT",
            "--output-format", "json",
            "--settings", rx.WORKFLOW_SETTINGS,
            "--model", rx.CLAUDE_MODEL,
            "--allowed-tools", ",".join(rx.ALLOWED_TOOLS),
            "--permission-mode", "auto",
            "--effort", "max",
        ])
        self.assertEqual(kw["cwd"], str(rx.E2E_DIR))
        self.assertEqual(kw["timeout"], 777)
        self.assertTrue(kw["capture_output"] and kw["text"])
        self.assertEqual(kw["env"]["IS_SANDBOX"], "1")

    def test_unsupported_effort_is_omitted_from_argv(self):
        self.patch_rx("CLAUDE_EFFORT", "ultracode")
        self._fake_which("/usr/bin/claude")
        calls = self._fake_subprocess(_FakeProc(stdout="plain text out"))
        self.assertEqual(rx._invoke_via_cli("P", 10), "plain text out")
        self.assertNotIn("--effort", calls[0][0])

    def test_binary_falls_back_to_the_env_override(self):
        self.patch_rx("CLAUDE_EFFORT", "ultracode")
        self._fake_which(None)
        os.environ["CLAUDE_BIN"] = "/custom/claude"
        calls = self._fake_subprocess(_FakeProc(stdout=""))
        rx._invoke_via_cli("P", 10)
        self.assertEqual(calls[0][0][0], "/custom/claude")

    def test_nonzero_exit_raises_a_classifiable_error(self):
        self._fake_which("/usr/bin/claude")
        self._fake_subprocess(_FakeProc(returncode=3, stderr="x" * 3000 + "TAIL"))
        with self.assertRaises(RuntimeError) as ctx:
            rx._invoke_via_cli("P", 10)
        msg = str(ctx.exception)
        self.assertIn("claude CLI failed (rc=3)", msg)
        self.assertIn("TAIL", msg)
        self.assertEqual(rx._classify_error(ctx.exception), "cli_failed")

    def test_json_wrapper_variants_are_unwrapped_or_passed_through(self):
        self._fake_which("/usr/bin/claude")
        self._fake_subprocess(_FakeProc(stdout='  {"text": "FROM-TEXT"}  '))
        self.assertEqual(rx._invoke_via_cli("P", 10), "FROM-TEXT")
        self._fake_subprocess(_FakeProc(stdout='{"other": 1}'))
        self.assertEqual(rx._invoke_via_cli("P", 10), '{"other": 1}')
        self._fake_subprocess(_FakeProc(stdout='[{"eval_dir": "/x"}]'))
        self.assertEqual(rx._invoke_via_cli("P", 10), '[{"eval_dir": "/x"}]')


class TestInvokeWorkflow(_RunE2ECase):
    def test_sdk_path_is_preferred_and_receives_the_pinned_eval_dir(self):
        self.install_module("claude_agent_sdk", _make_fake_sdk([]))
        seen = []
        self.patch_rx("_invoke_via_sdk", lambda p, t, ed: (
            seen.append((p, t, ed)) or '{"eval_dir": "/run/sdk", "throughput_speedup": 1.3}'
        ))
        self.patch_rx("_invoke_via_cli", lambda p, t: self.fail("CLI must not run"))
        wf = rx.invoke_workflow("P", 100, "/pinned/e2e")
        self.assertEqual(wf["eval_dir"], "/run/sdk")
        self.assertEqual(seen, [("P", 100, "/pinned/e2e")])

    def test_missing_sdk_falls_back_to_the_cli(self):
        # ``None`` in sys.modules is the documented way to make an import raise.
        self.install_module("claude_agent_sdk", None)
        self.patch_rx("_invoke_via_cli", lambda p, t: '{"eval_dir": "/run/cli"}')
        self.assertEqual(rx.invoke_workflow("P", 100, None)["eval_dir"], "/run/cli")

    def test_unparseable_agent_output_raises_workflow_parse_error(self):
        self.install_module("claude_agent_sdk", None)
        self.patch_rx("_invoke_via_cli", lambda p, t: "no json at all")
        with self.assertRaises(rx.WorkflowParseError):
            rx.invoke_workflow("P", 100, None)


# =========================================================================== #
# numeric + attribution helpers
# =========================================================================== #
class TestNumericHelpers(_RunE2ECase):
    def test_safe_ratio_rejects_junk_and_non_positive(self):
        self.assertEqual(rx._safe_ratio(535.352, 461.314), 1.1605)
        self.assertIsNone(rx._safe_ratio("abc", 1.0))
        self.assertIsNone(rx._safe_ratio(None, 1.0))
        self.assertIsNone(rx._safe_ratio(1.0, 0.0))
        self.assertIsNone(rx._safe_ratio(-1.0, 1.0))

    def test_divergence_pct_requires_two_finite_positives(self):
        self.assertEqual(rx._divergence_pct(110.0, 100.0), 10.0)
        self.assertIsNone(rx._divergence_pct("nan-ish", 100.0))
        self.assertIsNone(rx._divergence_pct(None, 100.0))
        self.assertIsNone(rx._divergence_pct(float("inf"), 100.0))
        self.assertIsNone(rx._divergence_pct(100.0, 0.0))

    def test_positive_finite_float_normalizes_for_strict_json(self):
        self.assertEqual(rx._positive_finite_float("12.5"), 12.5)
        self.assertEqual(rx._positive_finite_float(float("nan")), 0.0)
        self.assertEqual(rx._positive_finite_float(None), 0.0)
        self.assertEqual(rx._positive_finite_float(-3.0), 0.0)

    def test_best_accepted_delta_skips_junk_entries(self):
        wf = {
            "accepted_heads": ["not-a-dict", {"e2e_delta_pct": "bad"},
                               {"e2e_delta_pct": 4.0}],
            "accepted_kernels": [{"e2e_delta_pct": 12.5}, {"e2e_delta_pct": -3.0}],
        }
        self.assertEqual(rx._wf_best_accepted_delta_pct(wf), 12.5)
        self.assertEqual(rx._wf_best_accepted_delta_pct({}), 0.0)

    def test_best_ledger_win_never_fabricates(self):
        wf = {"state": {"history": {"ledger": [
            "junk",
            {"e2e_delta_pct": 99.0},                       # unnamed => ignored
            {"direction": "a", "e2e_delta_pct": "bad"},    # unparseable => 0
            {"direction": "b", "e2e_delta_pct": 3.0},
            {"short_name": "c", "e2e_delta_pct": 8.0},
        ]}}}
        self.assertEqual(rx._best_ledger_win(wf)["short_name"], "c")
        self.assertIsNone(rx._best_ledger_win({}))
        self.assertIsNone(rx._best_ledger_win(
            {"state": {"history": {"ledger": [{"direction": "z", "e2e_delta_pct": -1}]}}}
        ))

    def test_state_op_names(self):
        wf = {"state": {"headQueue": [{"short_name": "h0"}, {"no_name": 1}, "junk"]}}
        self.assertEqual(rx._state_op_names(wf, "headQueue"), {"h0"})
        self.assertEqual(rx._state_op_names(wf, "kernelQueue"), set())

    def test_ir_float_degrades_on_unparseable_values(self):
        self.assertEqual(rx._ir_float({"e2e": {"delta_pct": "16.0"}}, "delta_pct"), 16.0)
        self.assertEqual(rx._ir_float({"delta_pct": "n/a"}, "delta_pct"), 0.0)
        self.assertEqual(rx._ir_float({}, "delta_pct"), 0.0)

    def test_parity_normalisation(self):
        self.assertTrue(rx._parity_passed({"status": "PASS"}))
        self.assertTrue(rx._parity_passed("identical"))
        self.assertFalse(rx._parity_passed("mismatch"))
        self.assertIsNone(rx._parity_passed("not_measured"))
        self.assertIsNone(rx._parity_passed(None))

    def test_backend_enum_is_closed(self):
        self.assertEqual(rx._norm_backend("Claude"), "claude")
        self.assertEqual(rx._norm_backend("some-vendor"), "geak")
        self.assertEqual(rx._norm_backend(None), "geak")

    def test_kernel_id_canonicalisation_and_fuzzy_key(self):
        self.assertEqual(rx._canon_kid("_fwd_grouped_kernel_stage1"),
                         "fwd_grouped_kernel_stage1")
        self.assertEqual(rx._norm_kname("_Fwd_Kernel"), "fwd_kernel")
        self.assertEqual(rx._fuzzy_kid_key("_fwd_grouped_kernel_stage1"),
                         rx._fuzzy_kid_key("fwd_grouped_stage1"))


class TestOrchestratorHotBaseline(_RunE2ECase):
    """Hyperloom's anchor is ``baseline_tput``; whether it is HOT is told by
    ``baseline_warm_runtime_sec`` (the measure round's wall-clock, written only on
    the double-run path) and ``baseline_measure_round_dropped``."""

    def test_absent_exp_root_is_unknown(self):
        self.assertEqual(rx.read_orchestrator_baseline_lifecycle({}), (0.0, "unknown"))
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": "  "}),
            (0.0, "unknown"),
        )

    def test_hot_baseline_found_two_levels_up(self):
        session = self.tmp / "session"
        exp_root = session / "run" / "geak"
        exp_root.mkdir(parents=True)
        self.write_json(session / "state.json", {
            "baseline_tput": 612.5,
            "baseline_warm_runtime_sec": 176.4,
            "baseline_measure_round_dropped": False,
        })
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": str(exp_root)}),
            (612.5, "hot_measure_round"),
        )
        self.assertEqual(
            rx.read_orchestrator_hot_baseline({"exp_root": str(exp_root)}), 612.5
        )

    def test_nested_baseline_block_is_read(self):
        exp_root = self.tmp / "geak"
        exp_root.mkdir(parents=True)
        self.write_json(exp_root / "state.json", {"baseline": {
            "baseline_tput": "701.25",
            "baseline_warm_runtime_sec": "88.0",
        }})
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": str(exp_root)}),
            (701.25, "hot_measure_round"),
        )

    def test_dropped_measure_round_is_cold_and_not_offered_as_hot(self):
        """Budget could not fund the hot pass, so the anchor is the cold round —
        dividing GEAK's hot final by it would return the warm-up as speedup."""
        exp_root = self.tmp / "geak"
        exp_root.mkdir(parents=True)
        self.write_json(exp_root / "state.json", {
            "baseline_tput": 500.0,
            "baseline_warm_runtime_sec": 0.0,
            "baseline_measure_round_dropped": True,
        })
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": str(exp_root)}),
            (0.0, "cold_single_round"),
        )

    def test_single_round_baseline_is_unknown_not_hot(self):
        exp_root = self.tmp / "geak"
        exp_root.mkdir(parents=True)
        self.write_json(exp_root / "state.json", {"baseline_tput": 500.0})
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": str(exp_root)}),
            (0.0, "unknown"),
        )

    def test_unusable_values_degrade_to_zero(self):
        exp_root = self.tmp / "geak"
        exp_root.mkdir(parents=True)
        self.write_json(exp_root / "state.json", {"baseline_tput": "not-a-number"})
        self.assertEqual(
            rx.read_orchestrator_hot_baseline({"exp_root": str(exp_root)}), 0.0
        )

    def test_missing_state_json_degrades_to_zero(self):
        exp_root = self.tmp / "geak"
        exp_root.mkdir(parents=True)
        self.assertEqual(
            rx.read_orchestrator_baseline_lifecycle({"exp_root": str(exp_root)}),
            (0.0, "unknown"),
        )


# =========================================================================== #
# normalize_result: cold/hot final-basis selection
# =========================================================================== #
class TestColdFinalBasis(_RunE2ECase):
    def _eval_dir(self, *, cold_final=None, cold_baseline=None) -> Path:
        eval_dir = self.tmp / "e2e_cold"
        base = {"output_throughput_tok_s_median": 450.0}
        final = {"output_throughput_tok_s_median": 500.0}
        if cold_baseline is not None:
            base["cold_output_throughput_tok_s"] = cold_baseline
        if cold_final is not None:
            final["cold_output_throughput_tok_s"] = cold_final
        self.write_json(eval_dir / "baseline" / "bench_summary.json", base)
        self.write_json(eval_dir / "validation" / "final" / "bench_summary.json", final)
        return eval_dir

    def _wf(self, eval_dir: Path) -> dict:
        return {
            "eval_dir": str(eval_dir),
            "baseline_throughput_tok_s": 450.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.1111,
            "output_parity": "pass",
        }

    def test_a_cold_win_never_replaces_the_promoted_hot_median(self):
        """A cold round measured mid-session is a warm round wearing the label,
        so a flattering cold ratio must not become the headline: the promoted
        pair stays hot-over-hot and the cold numbers stay diagnostic."""
        eval_dir = self._eval_dir(cold_final=520.0, cold_baseline=460.0)
        out = rx.normalize_result(
            {"raw_baseline_tput": 440.0}, self._wf(eval_dir)
        )
        self.assertEqual(out["final_throughput_basis"], "hot")
        self.assertEqual(out["final_throughput_tok_s"], 500.0)
        self.assertEqual(out["throughput_speedup"], 1.1111)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["alignment_metrics"]["final_basis"], "hot")
        # Still reported, just no longer load-bearing.
        self.assertEqual(out["alignment_metrics"]["cold_speedup"],
                         rx._safe_ratio(520.0, 440.0))
        self.assertEqual(out["alignment_metrics"]["cold_geak_speedup"],
                         rx._safe_ratio(520.0, 460.0))

    def test_cold_loss_keeps_the_hot_median(self):
        """An authored overlay pays a one-off JIT/graph-capture cost on the cold
        round; a genuine steady-state win must not be thrown away as a cold loss."""
        eval_dir = self._eval_dir(cold_final=380.0, cold_baseline=460.0)
        out = rx.normalize_result({"raw_baseline_tput": 440.0}, self._wf(eval_dir))
        self.assertEqual(out["final_throughput_basis"], "hot")
        self.assertEqual(out["final_throughput_tok_s"], 500.0)
        self.assertEqual(out["throughput_speedup"], 1.1111)

    def test_cold_numbers_alone_never_move_the_promoted_pair(self):
        """No orchestrator baseline, so the only cold ratio available is GEAK's
        own — which used to be enough to switch the basis. It no longer is."""
        eval_dir = self._eval_dir(cold_final=520.0, cold_baseline=460.0)
        out = rx.normalize_result({}, self._wf(eval_dir))
        self.assertEqual(out["final_throughput_basis"], "hot")
        self.assertIsNone(out["alignment_metrics"]["cold_speedup"])
        self.assertEqual(out["throughput_speedup"], 1.1111)

    def test_a_cold_final_without_a_cold_baseline_changes_nothing(self):
        eval_dir = self._eval_dir(cold_final=520.0)
        out = rx.normalize_result({"raw_baseline_tput": 440.0}, self._wf(eval_dir))
        self.assertEqual(out["final_throughput_basis"], "hot")
        self.assertEqual(out["final_throughput_tok_s"], 500.0)
        self.assertEqual(out["throughput_speedup"], 1.1111)
        self.assertIsNone(out["alignment_metrics"]["cold_geak_speedup"])

    def test_no_cold_round_is_byte_identical_hot_behaviour(self):
        eval_dir = self._eval_dir()
        out = rx.normalize_result({"raw_baseline_tput": 440.0}, self._wf(eval_dir))
        self.assertEqual(out["final_throughput_basis"], "hot")
        self.assertIsNone(out["alignment_metrics"]["geak_cold_final_tok_s"])
        self.assertEqual(out["metric_basis"], "aggregate_output_tok_s")
        self.assertEqual(out["ttft_ms"], None)


# =========================================================================== #
# report rendering + section upsert
# =========================================================================== #
class TestAlignmentReportPlumbing(_RunE2ECase):
    def test_section_is_prepended_when_the_report_has_no_h1(self):
        text = "Some notes without a heading.\n"
        out = rx._upsert_marked_markdown_section(
            text, "SECTION", begin_marker="<!--B-->", end_marker="<!--E-->"
        )
        self.assertEqual(out, "SECTION\n\n" + text)

    def test_no_eval_dir_updates_nothing(self):
        self.assertEqual(rx._update_baseline_alignment_reports({}), [])

    def test_relative_report_path_resolves_under_the_eval_dir(self):
        eval_dir = self.tmp / "e2e_rel"
        eval_dir.mkdir()
        (eval_dir / "final_report.md").write_text("# R\n", encoding="utf-8")
        updated = rx._update_baseline_alignment_reports({
            "eval_dir": str(eval_dir),
            "report_path": "final_report.md",
            "baseline_basis": {"geak_measured_baseline_tok_s": 100.0},
            "baseline_alignment": {"status": "aligned", "warning_threshold_pct": 3.0},
        })
        # The relative candidate and the default both resolve to the same file,
        # so it is updated exactly once.
        self.assertEqual(updated, [str(eval_dir / "final_report.md")])
        rendered = (eval_dir / "final_report.md").read_text(encoding="utf-8")
        self.assertEqual(rendered.count(rx.BASELINE_ALIGNMENT_BEGIN), 1)
        self.assertIn("100.000 tok/s", rendered)

    def test_unavailable_numbers_render_as_unavailable(self):
        section = rx._render_baseline_alignment_section(
            {"baseline_basis": {}, "baseline_alignment": {}}
        )
        self.assertIn("- GEAK measured baseline: unavailable", section)
        self.assertIn("`unavailable`", section)
        self.assertEqual(rx._format_optional_number(None), "unavailable")


# =========================================================================== #
# on-disk recovery plumbing not exercised by the recovery suite
# =========================================================================== #
class TestRecoveryPlumbing(_RunE2ECase):
    def test_git_sha_degrades_on_failure(self):
        self.patch_rx("subprocess", types.SimpleNamespace(
            run=lambda *a, **k: _FakeProc(returncode=0, stdout="deadbee\n")))
        self.assertEqual(rx._git_short_sha(self.tmp), "deadbee")
        self.patch_rx("subprocess", types.SimpleNamespace(
            run=lambda *a, **k: _FakeProc(returncode=128, stdout="")))
        self.assertEqual(rx._git_short_sha(self.tmp), "")

        def boom(*a, **k):
            raise OSError("git not installed")

        self.patch_rx("subprocess", types.SimpleNamespace(run=boom))
        self.assertEqual(rx._git_short_sha(self.tmp), "")

    def test_discover_eval_dir_edge_cases(self):
        os.environ.pop("GEAK_EVAL_DIR", None)
        self.assertIsNone(rx._discover_eval_dir(self.tmp / "does_not_exist"))
        empty = self.tmp / "empty_root"
        empty.mkdir()
        self.assertIsNone(rx._discover_eval_dir(empty))
        # No completion marker anywhere => newest e2e_* dir is the fallback.
        (empty / "e2e_a").mkdir()
        self.assertEqual(rx._discover_eval_dir(empty), empty / "e2e_a")

    def test_pinned_eval_dir_short_circuits_the_glob(self):
        pinned = self.tmp / "e2e_pinned"
        pinned.mkdir()
        other = self.tmp / "root"
        (other / "e2e_other").mkdir(parents=True)
        os.environ["GEAK_EVAL_DIR"] = str(pinned)
        self.assertEqual(rx._discover_eval_dir(other), pinned)

    def test_recover_returns_none_without_a_discoverable_eval_dir(self):
        os.environ.pop("GEAK_EVAL_DIR", None)
        self.assertIsNone(rx._recover_workflow_return(self.tmp / "nothing"))

    def test_director_speedup_is_derived_when_only_endpoints_survived(self):
        """The 20260615-era director schema carries no throughput_speedup; the
        recovery must derive it instead of reporting no_gain over a real win."""
        os.environ.pop("GEAK_EVAL_DIR", None)
        exp_root = self.tmp / "exp"
        eval_dir = exp_root / "e2e_legacy"
        (eval_dir / "overlay" / "cand_mla_decode_fwd").mkdir(parents=True)
        self.write_json(eval_dir / "director_e2e_validation.json", {
            "provided_baseline_throughput": 400.0,
            "final": {"median": 480.0},
            "output_parity": "pass",
            "validation_status": "validated_win",
            "serving_config": {"baseline_flags": "--trust-remote-code",
                               "baseline_env": "X=1"},
        })
        wf = rx._recover_workflow_return(exp_root)
        self.assertEqual(wf["baseline_throughput_tok_s"], 400.0)
        self.assertEqual(wf["final_throughput_tok_s"], 480.0)
        self.assertAlmostEqual(wf["throughput_speedup"], 1.2)
        self.assertTrue(wf["recovered_from_disk"])
        self.assertEqual(wf["accepted_config"],
                         {"flags": "--trust-remote-code", "env": "X=1"})
        # Exactly one accepted kernel => it owns the whole measured delta.
        self.assertEqual(len(wf["accepted_kernels"]), 1)
        self.assertEqual(wf["accepted_kernels"][0]["short_name"], "mla_decode_fwd")
        self.assertAlmostEqual(wf["accepted_kernels"][0]["e2e_delta_pct"], 20.0, 4)

    def test_two_accepted_kernels_leave_per_kernel_gain_unattributed(self):
        os.environ.pop("GEAK_EVAL_DIR", None)
        exp_root = self.tmp / "exp2"
        eval_dir = exp_root / "e2e_two"
        (eval_dir / "overlay" / "cand_a").mkdir(parents=True)
        (eval_dir / "overlay" / "cand_b").mkdir(parents=True)
        self.write_json(eval_dir / "director_e2e_validation.json", {
            "baseline_throughput_tok_s": 400.0,
            "director_verified_throughput_tok_s": 480.0,
            "throughput_speedup": 1.2,
        })
        wf = rx._recover_workflow_return(exp_root)
        self.assertEqual([k["short_name"] for k in wf["accepted_kernels"]], ["a", "b"])
        self.assertNotIn("e2e_delta_pct", wf["accepted_kernels"][0])

    def test_intermediate_recovery_ignores_ungated_and_flat_candidates(self):
        """Only a gate==accepted candidate with a POSITIVE measured delta is a
        salvageable win; a rejected or zero-delta A/B must never be promoted."""
        eval_dir = self.tmp / "e2e_nowin"
        for name, ir in (
            ("cand_rejected", {"gate": "rejected", "e2e_delta_pct": 12.0,
                               "cand_med": 500.0}),
            ("cand_flat", {"gate": "accepted", "e2e_delta_pct": 0.0,
                           "cand_med": 500.0}),
            ("cand_empty", None),
        ):
            (eval_dir / "overlay" / name).mkdir(parents=True)
            if ir is not None:
                self.write_json(
                    eval_dir / "overlay" / name / "integrate_result.json", ir)
        self.assertIsNone(rx._recover_best_intermediate_win(eval_dir))

    def test_no_gain_recovery_rejects_an_unparseable_baseline(self):
        eval_dir = self.tmp / "e2e_badbase"
        self.write_json(eval_dir / "baseline" / "baseline_official.json",
                        {"baseline_throughput_tok_s": "six hundred"})
        self.assertIsNone(rx._recover_completed_no_gain(eval_dir))

    def test_persist_workflow_return_is_best_effort(self):
        """A read-only eval_dir must not turn a successful run into a crash."""
        missing = self.tmp / "nope" / "deeper"
        rx._persist_workflow_return(missing, {"eval_dir": "x"})
        self.assertFalse(missing.exists())

    def test_enumerate_overlay_kernels_dedups_across_both_overlays(self):
        eval_dir = self.tmp / "e2e_ov"
        (eval_dir / "overlay" / "cand_a").mkdir(parents=True)
        (eval_dir / "final" / "overlay" / "cand_a").mkdir(parents=True)
        (eval_dir / "final" / "overlay" / "cand_b").mkdir(parents=True)
        (eval_dir / "overlay" / "not_a_cand").mkdir()
        self.assertEqual(rx._enumerate_overlay_kernels(eval_dir), ["a", "b"])


# =========================================================================== #
# kernel_journey: the workflow-return (live) path
# =========================================================================== #
class TestJourneyReturnPath(_RunE2ECase):
    def setUp(self):
        super().setUp()
        self.patch_rx("_git_short_sha", lambda root: "abc1234")

    def test_live_accepted_kernel_becomes_an_entry_and_a_synthetic_discovery(self):
        """On the live path there is no overlay or profiler table on disk; the
        accepted kernels must still surface as journey entries AND as a
        discovery run, or the orchestrator sees an orphaned optimization."""
        eval_dir = self.tmp / "e2e_live"
        eval_dir.mkdir()
        wf = {
            "eval_dir": str(eval_dir),
            "output_parity": "pass",
            "accepted_config": {"flags": "--max-num-batched-tokens 16384"},
            "accepted_kernels": [{
                "short_name": "my_kernel_fwd",
                "backend": "claude",
                "isolated": 1.8,
                "final_patch": "/patches/k.diff",
                "pct_gpu_time": 42.0,
                "e2e_delta_pct": 9.5,
                "target_callable": "mod.fn",
                "kernel_eval_dir": "/runs/k0",
                "bound_type": "memory",
            }],
            "accepted_heads": [
                "junk-not-a-dict",
                {"short_name": "my_kernel_fwd"},   # duplicate id => deduped
                {"op_kind": "gemm"},               # unnamed => op_kind is the name
            ],
        }
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual(journey["schema_version"], rx.KERNEL_JOURNEY_SCHEMA_VERSION)
        self.assertEqual(journey["versions"]["geak"]["commit"], "abc1234")
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["my_kernel_fwd", "gemm"])

        entry = journey["kernels"][0]
        self.assertEqual(entry["name"], "my_kernel_fwd")
        self.assertEqual(entry["gpu_pct"], 42.0)
        self.assertEqual(entry["micro_speedup"], 1.8)
        self.assertEqual(entry["dispatch"]["backends"], ["claude"])
        attempt = entry["backend_result"]["attempts"][0]
        self.assertEqual(attempt["attempt_id"], "my_kernel_fwd-claude-0")
        self.assertEqual(attempt["decision"], "KEEP")
        self.assertTrue(attempt["correctness_passed"])
        self.assertEqual(attempt["optimized_files"], ["/patches/k.diff"])
        self.assertEqual(entry["backend_result"]["run_id"], "/runs/k0")
        self.assertEqual(entry["e2e"]["e2e_gain_pct"], 9.5)
        self.assertEqual(entry["e2e"]["target_file"], "mod.fn")
        self.assertEqual(entry["e2e"]["extra_server_args"],
                         "--max-num-batched-tokens 16384")

        disc = journey["discovery_runs"][0]
        self.assertEqual(disc["source"], "bypass")
        self.assertEqual(disc["hot_kernel_count"], 2)
        self.assertEqual(disc["scan"]["candidates_path"], f"geak:{eval_dir}")
        self.assertTrue(all(h["selected_for_optimization"] for h in disc["hot_kernels"]))
        self.assertEqual(disc["hot_kernels"][0]["recommended_backends"], ["claude"])

    def test_return_entry_adopts_the_profiler_symbol_on_a_fuzzy_match(self):
        """The overlay/return spelling and the profiler symbol can differ by the
        generic ``kernel`` infix; both substreams must still fold into one id."""
        eval_dir = self.tmp / "e2e_fuzzy"
        self.write_json(eval_dir / "profile" / "round_0" / "profile_topN.json", {
            "source": "rocprofv3",
            "top_kernels": [
                {"short_name": "_fwd_grouped_kernel_stage1",
                 "name": "_fwd_grouped_kernel_stage1",
                 "pct_gpu_time": 22.0, "editable": True, "classification": "triton"},
            ],
        })
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "fwd_grouped_stage1"}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        # The fold key is shared with the discovery substream, so the assembler
        # produces ONE journey entry for this kernel.
        disc = journey["discovery_runs"][0]
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["fwd_grouped_kernel_stage1"])
        self.assertEqual([h["kernel_id"] for h in disc["hot_kernels"]],
                         ["fwd_grouped_kernel_stage1"])
        # Unlike the overlay path, the return path keeps the return's own
        # spelling in ``name``; only the id is adopted from the profiler.
        self.assertEqual(journey["kernels"][0]["name"], "fwd_grouped_stage1")
        self.assertEqual(disc["hot_kernels"][0]["name"], "_fwd_grouped_kernel_stage1")
        # Real profiler discovery is present, so nothing is synthesized.
        self.assertEqual(len(journey["discovery_runs"]), 1)
        self.assertEqual(disc["hot_kernel_count"], 1)

    def test_symbol_less_profiler_row_yields_an_empty_discovery_kernel_id(self):
        """Documents a divergence between the two consumers of profile_topN.json:
        the cross-source match index SKIPS a row carrying neither short_name nor
        name, but the discovery substream still emits it — as a hot_kernel whose
        kernel_id is the empty string, which the journey schema requires to be
        unique and non-empty."""
        eval_dir = self.tmp / "e2e_blankrow"
        self.write_json(eval_dir / "profile" / "round_0" / "profile_topN.json", {
            "top_kernels": [{"pct_gpu_time": 1.0}, {"short_name": "gemm"}],
        })
        journey = rx.build_kernel_journey(
            {"eval_dir": str(eval_dir)}, {"eval_dir": str(eval_dir)}
        )
        hot = journey["discovery_runs"][0]["hot_kernels"]
        self.assertEqual([h["kernel_id"] for h in hot], ["", "gemm"])

    def test_repeated_profiler_symbols_keep_unique_ids(self):
        eval_dir = self.tmp / "e2e_dupsym"
        self.write_json(eval_dir / "profile" / "profile_topN.json", {
            "top_kernels": [
                {"short_name": "gemm", "rank": 1, "pct_gpu_time": 30.0},
                {"short_name": "gemm", "rank": 2, "pct_gpu_time": 10.0},
            ],
        })
        journey = rx.build_kernel_journey(
            {"eval_dir": str(eval_dir)}, {"eval_dir": str(eval_dir)}
        )
        ids = [h["kernel_id"] for h in journey["discovery_runs"][0]["hot_kernels"]]
        self.assertEqual(ids, ["gemm", "gemm#2"])

    def test_overlay_scan_skips_non_directories_and_duplicate_ids(self):
        eval_dir = self.tmp / "e2e_overlay_junk"
        (eval_dir / "overlay").mkdir(parents=True)
        (eval_dir / "overlay" / "cand_stray_file").write_text("x", encoding="utf-8")
        (eval_dir / "overlay" / "cand_").mkdir()
        (eval_dir / "overlay" / "cand_real").mkdir()
        (eval_dir / "final" / "overlay" / "cand_real").mkdir(parents=True)
        self.write_json(eval_dir / "profile" / "round_0" / "profile_topN.json", {
            "top_kernels": [{"short_name": "real", "name": "real",
                             "pct_gpu_time": 12.5, "editable": True}],
        })
        journey = rx.build_kernel_journey(
            {"eval_dir": str(eval_dir)}, {"eval_dir": str(eval_dir)}
        )
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]], ["real"])
        # An EXACT profiler match hands the entry the profiler's gpu%.
        self.assertEqual(journey["kernels"][0]["gpu_pct"], 12.5)
        self.assertTrue(
            journey["discovery_runs"][0]["hot_kernels"][0]["selected_for_optimization"]
        )

    # --- overlay-vs-return identity (one acceptance, two spellings) ---------- #

    def _overlay(self, eval_dir: Path, tag: str, ir: dict | None) -> Path:
        """One candidate overlay dir, with or without its integrate_result."""
        cand = eval_dir / "overlay" / f"cand_{tag}"
        cand.mkdir(parents=True, exist_ok=True)
        if ir is not None:
            self.write_json(cand / "integrate_result.json", ir)
        return cand

    def test_return_acceptance_already_on_disk_as_an_overlay_is_not_re_emitted(self):
        """The overlay dir is named for the CANDIDATE TAG and the workflow return
        for the KERNEL SYMBOL, so the id dedup never fires and one acceptance is
        emitted twice — once measured, once with a null gpu%. integrate_result
        records both spellings, so the symbol it claims must fold them."""
        eval_dir = self.tmp / "e2e_alias"
        self._overlay(eval_dir, "c0_triton", {
            "gate": "accepted", "short_name": "dsa_sparse_attn_prefill_main_kernel",
            "cand_tag": "c0_triton", "pct_gpu_time": 20.2,
            "isolated_speedup": 2.2, "e2e_delta_pct": 29.994,
        })
        wf = {"eval_dir": str(eval_dir), "accepted_heads": [{
            "short_name": "dsa_sparse_attn_prefill_main_kernel",
            "backend": "triton", "isolated": 1.13, "e2e_delta_pct": 29.994,
        }]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]], ["c0_triton"])
        # The surviving entry is the MEASURED one (gpu% from integrate_result),
        # not the return's null.
        self.assertEqual(journey["kernels"][0]["gpu_pct"], 20.2)
        self.assertEqual(journey["kernels"][0]["e2e"]["e2e_gain_pct"], 29.994)

    def test_sibling_candidates_for_one_symbol_each_stay_their_own_entry(self):
        """Folding is against the RETURN, not between overlays: two candidate
        implementations of the same kernel are two real attempts and must both
        survive, with only the return's duplicate dropped."""
        eval_dir = self.tmp / "e2e_siblings"
        self._overlay(eval_dir, "c0_triton", {
            "gate": "accepted", "short_name": "dsa_fwd", "e2e_delta_pct": 29.994})
        self._overlay(eval_dir, "c1_tilelang", {
            "gate": "stack", "short_name": "dsa_fwd", "e2e_delta_pct": 2.818})
        wf = {"eval_dir": str(eval_dir),
              "accepted_heads": [{"short_name": "dsa_fwd", "e2e_delta_pct": 29.994}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["c0_triton", "c1_tilelang"])

    def test_symbol_less_overlay_folds_the_return_on_its_integrated_delta(self):
        """Some runs record no usable symbol (integrate_result echoes the tag).
        The return copies the overlay's own A/B delta rather than recomputing it,
        so an exact, unambiguous hit is the same measurement."""
        eval_dir = self.tmp / "e2e_delta_fold"
        self._overlay(eval_dir, "decode_attention_grouped_mla", {
            "gate": "accepted", "short_name": "decode_attention_grouped_mla",
            "e2e_delta_pct": 11.71})
        wf = {"eval_dir": str(eval_dir), "accepted_heads": [
            {"short_name": "_fwd_grouped_kernel_stage1 (+_fwd_kernel_stage2)",
             "e2e_delta_pct": 11.71}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["decode_attention_grouped_mla"])

    def test_an_ambiguous_delta_never_folds_two_kernels(self):
        """Two overlays sharing a delta cannot identify which acceptance the
        return meant, so the return entry is kept rather than guessed away."""
        eval_dir = self.tmp / "e2e_delta_ambig"
        self._overlay(eval_dir, "c0_triton", {"gate": "accepted", "e2e_delta_pct": 5.0})
        self._overlay(eval_dir, "c1_ck", {"gate": "accepted", "e2e_delta_pct": 5.0})
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "some_other_gemm",
                                    "e2e_delta_pct": 5.0}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertIn("some_other_gemm", [k["kernel_id"] for k in journey["kernels"]])

    def test_one_overlay_is_consumed_by_at_most_one_return_acceptance(self):
        """A single overlay whose delta matches must not swallow BOTH accepted
        records; the second is a distinct kernel and keeps its entry."""
        eval_dir = self.tmp / "e2e_claim_once"
        self._overlay(eval_dir, "c0_aiter", {"gate": "accepted", "e2e_delta_pct": 6.779})
        wf = {"eval_dir": str(eval_dir), "accepted_kernels": [
            {"short_name": "gemm_down_proj", "e2e_delta_pct": 6.779},
            {"short_name": "gemm_gate_up", "e2e_delta_pct": 6.779},
        ]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["c0_aiter", "gemm_gate_up"])

    def test_a_rejected_overlay_claims_nothing(self):
        """Do-no-harm: an overlay the A/B REVERTED did not integrate that kernel,
        so an acceptance the return asserts is new information, not a duplicate."""
        eval_dir = self.tmp / "e2e_rejected"
        self._overlay(eval_dir, "c0_triton", {
            "gate": "rejected", "short_name": "my_gemm", "e2e_delta_pct": 1.5})
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "my_gemm", "e2e_delta_pct": 1.5}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["c0_triton", "my_gemm"])

    def test_an_incomplete_ab_overlay_claims_nothing(self):
        """No integrate_result at all (cut off mid-A/B) is not an acceptance."""
        eval_dir = self.tmp / "e2e_incomplete"
        self._overlay(eval_dir, "c0_triton", None)
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "my_gemm", "e2e_delta_pct": 1.5}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual([k["kernel_id"] for k in journey["kernels"]],
                         ["c0_triton", "my_gemm"])

    def test_folded_return_acceptance_is_absent_from_synthetic_discovery(self):
        """A folded acceptance must not come back as a synthesized hot_kernel,
        which would re-orphan the duplicate in the discovery substream."""
        eval_dir = self.tmp / "e2e_fold_disc"
        self._overlay(eval_dir, "c0_triton", {
            "gate": "accepted", "short_name": "dsa_fwd", "e2e_delta_pct": 3.0})
        wf = {"eval_dir": str(eval_dir),
              "accepted_heads": [{"short_name": "dsa_fwd", "e2e_delta_pct": 3.0}]}
        journey = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})
        self.assertEqual(journey["discovery_runs"], [])

    def test_journey_without_an_eval_dir_is_empty_but_valid(self):
        journey = rx.build_kernel_journey({}, {})
        self.assertEqual(journey["kernels"], [])
        self.assertEqual(journey["discovery_runs"], [])
        self.assertEqual(journey["eval_dir"], "")

    def test_write_kernel_journey_returns_the_atomic_path(self):
        eval_dir = self.tmp / "e2e_write"
        path = rx._write_kernel_journey(eval_dir, None, {"status": "error"})
        self.assertEqual(path, str(eval_dir / rx.KERNEL_JOURNEY_FILE))
        self.assertFalse((eval_dir / (rx.KERNEL_JOURNEY_FILE + ".tmp")).exists())
        self.assertEqual(json.loads(Path(path).read_text())["status"], "error")


# =========================================================================== #
# main(): control flow + guaranteed-emit degradation branches
# =========================================================================== #
class TestMain(_RunE2ECase):
    def setUp(self):
        super().setUp()
        self.patch_rx("_git_short_sha", lambda root: "abc1234")
        self.exp_root = self.tmp / "exp" / "geak"
        self.exp_root.mkdir(parents=True)
        self.eval_dir = self.exp_root / "e2e_main"
        self.result_path = self.tmp / "out" / "result.json"

    def _handoff(self, **extra) -> Path:
        h = {
            "schema_version": 2,
            "model_path": "/models/fake-8b",
            "framework": "vllm",
            "tp": 2,
            "workload": {"isl": 1024, "osl": 1024, "conc": 64},
            "exp_root": str(self.exp_root),
            "eval_dir": str(self.eval_dir),
        }
        h.update(extra)
        path = self.tmp / "handoff.json"
        return self.write_json(path, h)

    def _run(self, handoff: Path, *flags) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = rx.main([str(handoff), str(self.result_path), *flags])
        return rc, buf.getvalue()

    def test_usage_error_without_both_paths(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(rx.main(["--dry-run"]), 2)
        self.assertIn("usage: run_e2e.py", err.getvalue())

    def test_unreadable_handoff_is_a_usage_error(self):
        bad = self.tmp / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = rx.main([str(bad), str(self.result_path)])
        self.assertEqual(rc, 2)
        self.assertIn("empty/invalid handoff", err.getvalue())
        self.assertFalse(self.result_path.exists())

    def test_dry_run_reports_the_full_dispatch_plan_without_invoking(self):
        """--dry-run is the only way an operator can audit what GEAK is about to
        launch; it must show the mapped args AND every measurement export."""
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("dry-run must not invoke"))
        handoff = self._handoff(
            bench_client="native",
            bench_launcher="magpie",
            launch_server_script="/magpie/launch.sh",
            bench_protocol={"num_prompts": 512, "seed": 7},
        )
        rc, stdout = self._run(handoff, "--dry-run")
        self.assertEqual(rc, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["mapped_args"]["eval_dir"], str(self.eval_dir))
        self.assertEqual(plan["mapped_args"]["backend"], "vllm")
        self.assertEqual(plan["mapped_args"]["time_budget_s"], 43200)
        self.assertEqual(plan["bench_client"], "native")
        self.assertEqual(plan["bench_launcher"], "magpie")
        self.assertEqual(plan["magpie_launch_script"], "/magpie/launch.sh")
        self.assertEqual(plan["bench_protocol"],
                         {"NUM_PROMPTS": "512", "SEED": "7"})
        self.assertEqual(plan["alignment_flags"], {"BENCH_COLD_FINAL": "0"})
        self.assertIn("recipe_env_file", plan)
        self.assertIn("recipe_env_replayed", plan)
        self.assertIn("recipe_env_geak_owned", plan)
        self.assertEqual(plan["e2e_script"], str(rx.E2E_SCRIPT))
        self.assertIn("Invoke the Workflow tool exactly once", plan["prompt"])
        self.assertFalse(self.result_path.exists())

    def test_protected_pgids_are_published_before_any_launch(self):
        """Pins the CALL SITE, not just the helper: the veto must be in the
        environment before main() can reach a bench, so it is set even on the
        --dry-run path, which returns before invoke_workflow()."""
        saved = os.environ.pop("GEAK_PROTECTED_PGIDS", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "GEAK_PROTECTED_PGIDS", saved)
        else:
            self.addCleanup(os.environ.pop, "GEAK_PROTECTED_PGIDS", None)
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("dry-run must not invoke"))
        rc, _stdout = self._run(self._handoff(), "--dry-run")
        self.assertEqual(rc, 0)
        published = os.environ["GEAK_PROTECTED_PGIDS"].split()
        self.assertIn("1", published)
        self.assertIn(str(os.getpgid(0)), published)

    def _pin_env_timeout(self, value: str | None):
        """Set/clear GEAK_E2E_TIMEOUT_S for one test and always restore it, so
        budget tests cannot leak a budget into whatever runs next."""
        saved = os.environ.pop("GEAK_E2E_TIMEOUT_S", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "GEAK_E2E_TIMEOUT_S", saved)
        else:
            self.addCleanup(os.environ.pop, "GEAK_E2E_TIMEOUT_S", None)
        if value is not None:
            os.environ["GEAK_E2E_TIMEOUT_S"] = value

    def test_timeout_budget_is_read_from_the_environment(self):
        self._pin_env_timeout("600")
        self.patch_rx("invoke_workflow", lambda *a, **k: {})
        _rc, stdout = self._run(self._handoff(), "--dry-run")
        self.assertEqual(json.loads(stdout)["mapped_args"]["time_budget_s"], 600)

    def test_timeout_budget_is_read_from_the_cli(self):
        """Hyperloom's coordinator states its REAL kill time as
        `--timeout-s <runner_timeout>`. We used to drop that value into an
        ignored third positional and pace against the 12h env default instead,
        so a caller killing early tore the run down mid-optimization and no
        final report was written (Hyperloom #1202)."""
        self._pin_env_timeout(None)
        self.patch_rx("invoke_workflow", lambda *a, **k: {})
        for argv in (["--timeout-s", "28800"], ["--timeout-s=28800"]):
            with self.subTest(argv=argv):
                _rc, stdout = self._run(self._handoff(), *argv, "--dry-run")
                plan = json.loads(stdout)
                self.assertEqual(plan["mapped_args"]["time_budget_s"], 28800)
                # The value must NOT shift the positionals it sits behind.
                self.assertEqual(plan["mapped_args"]["eval_dir"],
                                 str(self.eval_dir))

    def test_smallest_stated_budget_wins(self):
        """Flag and env var both name a deadline someone actually enforces, so
        pace against the SMALLER. Junk states nothing, and with nothing stated
        the 12h default applies — a bad budget must not abort the run."""
        self.patch_rx("invoke_workflow", lambda *a, **k: {})
        default = rx.DEFAULT_E2E_TIMEOUT_S
        for env, cli, expected in (("14400", "28800", 14400),
                                   ("28800", "14400", 14400),
                                   (None, None, default),
                                   ("not-a-number", "later", default)):
            with self.subTest(env=env, cli=cli):
                self._pin_env_timeout(env)
                argv = ["--timeout-s", cli] if cli else []
                _rc, stdout = self._run(self._handoff(), *argv, "--dry-run")
                self.assertEqual(
                    json.loads(stdout)["mapped_args"]["time_budget_s"], expected)

    def test_final_reserve_is_overridable_from_the_environment(self):
        """Widening the finalize window must not need a code edit: an operator
        who measures the final phase taking longer than the 60min default sets
        GEAK_FINAL_RESERVE_S. Unset => the arg is absent and the JS default wins."""
        self.patch_rx("invoke_workflow", lambda *a, **k: {})
        for env, expected in (("3000", 3000), (None, None)):
            with self.subTest(env=env):
                saved = os.environ.pop("GEAK_FINAL_RESERVE_S", None)
                if saved is not None:
                    self.addCleanup(
                        os.environ.__setitem__, "GEAK_FINAL_RESERVE_S", saved)
                if env is not None:
                    os.environ["GEAK_FINAL_RESERVE_S"] = env
                    self.addCleanup(os.environ.pop, "GEAK_FINAL_RESERVE_S", None)
                _rc, stdout = self._run(self._handoff(), "--dry-run")
                self.assertEqual(
                    json.loads(stdout)["mapped_args"].get("final_reserve_s"),
                    expected)

    def test_successful_run_emits_result_and_journey(self):
        report = self.eval_dir / "final_report.md"
        report.parent.mkdir(parents=True)
        report.write_text("# GEAK final report\n", encoding="utf-8")
        seen = {}

        def ok_invoke(prompt, timeout_s, eval_dir):
            seen.update(prompt=prompt, timeout_s=timeout_s, eval_dir=eval_dir)
            return {"eval_dir": str(self.eval_dir),
                    "baseline_throughput_tok_s": 461.314,
                    "final_throughput_tok_s": 535.352,
                    "throughput_speedup": 1.1605,
                    "output_parity": "pass",
                    "report_path": str(report)}

        self.patch_rx("invoke_workflow", ok_invoke)
        rc, stdout = self._run(self._handoff(raw_baseline_tput=455.0))
        self.assertEqual(rc, 0)
        # The pinned eval_dir reaches the invocation AND the environment, so the
        # completion gate and the disk recovery target the same directory.
        self.assertEqual(seen["eval_dir"], str(self.eval_dir))
        self.assertEqual(seen["timeout_s"], 43200)
        self.assertEqual(os.environ["GEAK_EVAL_DIR"], str(self.eval_dir))
        self.assertEqual(json.loads(stdout), {
            "status": "ok",
            "result_json": str(self.result_path),
            "speedup": 1.1605,
        })
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result_source"], "workflow_return")
        self.assertEqual(out["kernel_journey_path"],
                         str(self.eval_dir / rx.KERNEL_JOURNEY_FILE))
        self.assertEqual(out["baseline_alignment_report_paths"], [str(report)])
        # The live return is persisted beside the artifacts so a later re-entry
        # never has to re-scrape the transcript.
        persisted = json.loads(
            (self.eval_dir / rx.WORKFLOW_RETURN_FILE).read_text())
        self.assertEqual(persisted["final_throughput_tok_s"], 535.352)

    def test_resume_short_circuits_a_terminal_eval_dir(self):
        """Re-entering a completed eval_dir must re-emit from disk, never burn a
        second full workflow run."""
        self.write_json(self.eval_dir / rx.WORKFLOW_RETURN_FILE, {
            "schema_version": 1,
            "eval_dir": str(self.eval_dir),
            "baseline_throughput_tok_s": 400.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.25,
            "output_parity": "pass",
        })
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-run the workflow"))
        rc, stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout)["speedup"], 1.25)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result_source"], "workflow_return")
        self.assertEqual(out["final_throughput_tok_s"], 500.0)

    def test_resume_with_a_failing_recovery_still_emits_an_error_file(self):
        """Both recovery attempts (the short-circuit and the one inside _emit)
        blow up — result.json must STILL exist and be parseable."""
        self.write_json(self.eval_dir / "director_e2e_validation.json", {})

        def boom(exp_root):
            raise OSError("artifact tree unreadable")

        self.patch_rx("_recover_workflow_return", boom)
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-run the workflow"))
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 1)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_class"], "runner_error")
        journey = json.loads(
            (self.eval_dir / rx.KERNEL_JOURNEY_FILE).read_text())
        self.assertEqual(journey["kernels"], [])

    def test_sigterm_handler_self_stops_as_a_timeout(self):
        """The outer runner's graceful stop must be converted into a TimeoutError
        so the finally-block flushes the interface files instead of being killed."""
        def invoke_then_term(prompt, timeout_s, eval_dir):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
            raise AssertionError("the SIGTERM handler must raise")

        self.patch_rx("invoke_workflow", invoke_then_term)
        rc, stdout = self._run(self._handoff())
        # A timeout is a miss, not a runner error: the exit code stays 0 so the
        # caller reads the emitted file rather than treating the run as broken.
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout)["status"], "timeout")
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "timeout")
        self.assertEqual(out["error_class"], "timeout")
        self.assertIn("self-stop to flush interface files", out["error"])

    def test_recovery_failure_after_a_crashed_workflow_is_contained(self):
        """Both the post-crash recovery and the one inside _emit raise; the run
        must degrade to a parseable error file, never propagate."""
        def boom_invoke(prompt, timeout_s, eval_dir):
            raise RuntimeError("agent died")

        def boom_recover(exp_root):
            raise OSError("artifact tree unreadable")

        self.patch_rx("invoke_workflow", boom_invoke)
        self.patch_rx("_recover_workflow_return", boom_recover)
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 1)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_class"], "runner_error")
        self.assertIn("agent died", out["error"])

    def test_workflow_failure_with_a_recoverable_disk_win(self):
        (self.eval_dir / "overlay" / "cand_moe_gemm").mkdir(parents=True)
        self.write_json(
            self.eval_dir / "overlay" / "cand_moe_gemm" / "integrate_result.json",
            {"short_name": "moe_gemm", "gate": "accepted", "winner_kind": "authored",
             "ref_med": 400.0, "cand_med": 500.0, "e2e_delta_pct": 25.0,
             "output_parity": "pass"},
        )

        def boom(prompt, timeout_s, eval_dir):
            raise rx.WorkflowParseError("agent printed prose")

        self.patch_rx("invoke_workflow", boom)
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["recovered_from_disk"])
        self.assertEqual(out["result_source"], "disk_intermediate_win")
        self.assertEqual(out["final_throughput_tok_s"], 500.0)
        self.assertEqual(out["accepted_kernels"][0]["short_name"], "moe_gemm")

    def test_normalize_failure_still_produces_a_parseable_error(self):
        def bad_normalize(h, wf):
            raise ValueError("schema drift")

        self.patch_rx("normalize_result", bad_normalize)
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: {"eval_dir": str(self.eval_dir)})
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 1)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["error_class"], "normalize_failed")
        self.assertIn("ValueError: schema drift", out["error"])
        # No eval_dir in the failed output, so the pinned hint gives the journey
        # a home anyway.
        self.assertEqual(out["kernel_journey_path"],
                         str(self.eval_dir / rx.KERNEL_JOURNEY_FILE))

    def test_persist_and_report_failures_are_contained(self):
        """Neither a failed return-persist nor a failed report render may lose
        result.json; the report failure is surfaced, not swallowed."""
        def boom_persist(eval_dir, wf):
            raise OSError("read-only eval_dir")

        def boom_report(result):
            raise RuntimeError("markdown render blew up")

        self.patch_rx("_persist_workflow_return", boom_persist)
        self.patch_rx("_update_baseline_alignment_reports", boom_report)
        self.patch_rx("invoke_workflow", lambda *a, **k: {
            "eval_dir": str(self.eval_dir),
            "baseline_throughput_tok_s": 400.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.25,
        })
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        out = json.loads(self.result_path.read_text())
        self.assertEqual(out["status"], "ok")
        self.assertIn("markdown render blew up", out["baseline_alignment_report_error"])
        self.assertNotIn("kernel_journey_error", out)

    def test_atomic_write_failure_falls_back_to_a_direct_write(self):
        """A blocked tmp path must never leave the caller with NOTHING."""
        self.result_path.parent.mkdir(parents=True)
        (self.result_path.parent / (self.result_path.name + ".tmp")).mkdir()
        self.patch_rx("invoke_workflow", lambda *a, **k: {
            "eval_dir": str(self.eval_dir),
            "baseline_throughput_tok_s": 400.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.25,
        })
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(self.result_path.read_text())["status"], "ok")

    def test_sigterm_ignore_failure_does_not_block_the_flush(self):
        """Hardening the flush against a second SIGTERM is best-effort; a platform
        that refuses SIG_IGN must not lose result.json."""
        def fake_signal(sig, handler):
            if handler is signal.SIG_IGN:
                raise OSError("cannot ignore SIGTERM here")
            return signal.SIG_DFL

        self.patch_rx("signal", types.SimpleNamespace(
            SIGTERM=signal.SIGTERM, SIG_IGN=signal.SIG_IGN, signal=fake_signal))
        self.patch_rx("invoke_workflow", lambda *a, **k: {
            "eval_dir": str(self.eval_dir),
            "baseline_throughput_tok_s": 400.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.25,
        })
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(self.result_path.read_text())["status"], "ok")

    def test_journey_write_failure_is_surfaced_into_result(self):
        def boom(eval_dir, wf, normalized):
            raise OSError("no space left on device")

        self.patch_rx("_write_kernel_journey", boom)
        self.patch_rx("invoke_workflow", lambda *a, **k: {
            "eval_dir": str(self.eval_dir),
            "baseline_throughput_tok_s": 400.0,
            "final_throughput_tok_s": 500.0,
            "throughput_speedup": 1.25,
        })
        rc, _stdout = self._run(self._handoff())
        self.assertEqual(rc, 0)
        out = json.loads(self.result_path.read_text())
        self.assertIn("no space left on device", out["kernel_journey_error"])
        self.assertNotIn("kernel_journey_path", out)


class TestE2EDenominatorIsPublished(_RunE2ECase):
    """A journey e2e block must carry the two throughputs its percentage is a
    ratio of.

    A percentage alone does not compose: two wins measured against different
    denominators cannot be added. The consumer needs the pair to restate each
    win in points of one fixed baseline. Both journey builders read the pair
    off the same A/B that produced the delta, so the three numbers always
    agree.
    """

    def _overlay(self, eval_dir, ir):
        d = eval_dir / "overlay" / "cand_my_kernel_fwd"
        self.write_json(d / "integrate_result.json", ir)
        return d

    def test_overlay_path_publishes_the_pair_and_it_reproduces_the_delta(self):
        eval_dir = self.tmp / "e2e_flat"
        self._overlay(eval_dir, {"gate": "accepted", "e2e_delta_pct": 25.0,
                                 "ref_med": 400.0, "cand_med": 500.0})
        journey = rx.build_kernel_journey({"eval_dir": str(eval_dir)},
                                          {"eval_dir": str(eval_dir)})
        e2e = journey["kernels"][0]["e2e"]
        self.assertEqual(e2e["base_tput"], 400.0)
        self.assertEqual(e2e["new_tput"], 500.0)
        self.assertAlmostEqual(
            (e2e["new_tput"] / e2e["base_tput"] - 1.0) * 100.0,
            e2e["e2e_gain_pct"], places=6)

    def test_overlay_path_reads_the_pair_from_the_nested_shape_too(self):
        """The integrator emits either shape; the delta already reads both, and
        the pair it is a ratio of must not be lost on the nested one."""
        eval_dir = self.tmp / "e2e_nested"
        self._overlay(eval_dir, {"gate": "accepted",
                                 "e2e": {"delta_pct": 25.0,
                                         "ref_median_tok_s": 400.0,
                                         "cand_median_tok_s": 500.0}})
        e2e = rx.build_kernel_journey({"eval_dir": str(eval_dir)},
                                      {"eval_dir": str(eval_dir)})["kernels"][0]["e2e"]
        self.assertEqual((e2e["base_tput"], e2e["new_tput"]), (400.0, 500.0))

    def test_return_path_carries_the_pair_from_the_workflow_return(self):
        eval_dir = self.tmp / "e2e_return"
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "my_kernel_fwd",
                                    "e2e_delta_pct": 25.0,
                                    "base_tput": 400.0, "new_tput": 500.0}]}
        e2e = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})["kernels"][0]["e2e"]
        self.assertEqual((e2e["base_tput"], e2e["new_tput"]), (400.0, 500.0))

    def test_a_missing_pair_is_null_not_fabricated(self):
        """An older workflow return has no throughputs. The block must say so
        rather than invent a denominator the A/B never measured."""
        eval_dir = self.tmp / "e2e_nopair"
        wf = {"eval_dir": str(eval_dir),
              "accepted_kernels": [{"short_name": "my_kernel_fwd",
                                    "e2e_delta_pct": 25.0}]}
        e2e = rx.build_kernel_journey(wf, {"eval_dir": str(eval_dir)})["kernels"][0]["e2e"]
        self.assertIsNone(e2e["base_tput"])
        self.assertIsNone(e2e["new_tput"])
        self.assertEqual(e2e["e2e_gain_pct"], 25.0)


if __name__ == "__main__":
    raise SystemExit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
