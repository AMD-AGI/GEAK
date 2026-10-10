#!/usr/bin/env python3
"""The per-run workload declaration reaches every bench, and only when it exists.

Under an orchestrator, ``BENCH_CLIENT`` / ``E2E_METRIC`` / ``AGENTX_*`` arrive in
the environment GEAK inherits (``interface/run_e2e.py`` exports them). Standalone
there is no such parent, and the bench commands are issued by ROLE AGENTS from
static prompt templates, so there is nowhere to inject an env prefix. Instead the
workflow drops one ``bench_env.sh`` beside the copied bench script and both
``bench_e2e.sh`` and ``bench_replica.sh`` source it from next to themselves.

Two properties matter more than the feature itself:

* **Absent file => nothing happens.** A fixed ISL/OSL run never gets a
  ``bench_env.sh``, so the channel must be completely inert without one.
* **The environment outranks the file.** The declaration only assigns a name that
  is unset or empty, so an orchestrated run -- which exports these same names for
  its own reasons -- keeps behaving exactly as it did before the channel existed,
  and an operator can override a single knob on the command line.

The sourcing block is EXTRACTED from the real scripts rather than restated here,
so if it is edited or dropped these tests fail instead of passing against a copy.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

BASH = shutil.which("bash")
SCRIPTS = Path(__file__).resolve().parents[1]
BENCH_E2E = SCRIPTS / "bench_e2e.sh"
BENCH_REPLICA = SCRIPTS / "bench_replica.sh"
WORKFLOW_JS = SCRIPTS.parent / "e2e_workflow.js"

# The names a declaration sets; used to prove the block is inert without a file.
DECLARED = ("BENCH_CLIENT", "E2E_METRIC", "GEAK_METRIC_BASIS", "GEAK_WORKLOAD_KIND")

# A representative declaration, in the form e2e_workflow.js emits: assign only
# when the name is unset or empty, with the value in real shell single quotes.
# (`: "${K:='v'}"` would be shorter and wrong -- those quotes are literal
# characters of the value, so every consumer would see `'agentx'`.)
# Generation is covered by scripts/test_agentx_declaration.js; this pins consumption.
DECLARATION = """\
#!/usr/bin/env bash
[ -n "${GEAK_WORKLOAD_KIND:-}" ] || GEAK_WORKLOAD_KIND='agentx_trace_replay'
[ -n "${BENCH_CLIENT:-}" ] || BENCH_CLIENT='agentx'
[ -n "${E2E_METRIC:-}" ] || E2E_METRIC='p90_intvty_inferencex'
[ -n "${GEAK_METRIC_BASIS:-}" ] || GEAK_METRIC_BASIS='p90_intvty_inferencex'
export GEAK_WORKLOAD_KIND BENCH_CLIENT E2E_METRIC GEAK_METRIC_BASIS
"""


def _extract_source_block(script: Path) -> str:
    """Pull the `BENCH_ENV_FILE=...` sourcing block out of a real bench script."""
    text = script.read_text()
    match = re.search(
        r'^BENCH_ENV_FILE="\$\{BENCH_ENV_FILE:-\$HERE/bench_env\.sh\}"\n'
        r'if \[ -f "\$BENCH_ENV_FILE" \]; then\n.*?^fi$',
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, f"the bench_env.sh sourcing block is missing from {script.name}"
    return match.group(0)


@unittest.skipIf(BASH is None, "bash is required to exercise the sourcing block")
class BenchEnvChannelTest(unittest.TestCase):
    """Run the extracted block with HERE pointed at a scratch dir."""

    scripts = ((BENCH_E2E, "bench_e2e.sh"), (BENCH_REPLICA, "bench_replica.sh"))

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="bench_env_channel_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, script: Path, *, declaration: str | None, env=None) -> dict:
        """Source the block, then report the resulting values of DECLARED."""
        if declaration is not None:
            (self.tmp / "bench_env.sh").write_text(declaration)
        block = _extract_source_block(script)
        probe = "\n".join(
            f'printf "%s=%s\\n" {name} "${{{name}-<unset>}}"' for name in DECLARED
        )
        run_env = {
            k: v
            for k, v in os.environ.items()
            # Start from a clean slate: a value leaking in from the developer's
            # own shell would make "inherited wins" pass for the wrong reason.
            if k not in DECLARED
        }
        run_env.update(env or {})
        proc = subprocess.run(
            [BASH, "-c", f'set -uo pipefail\nHERE="{self.tmp}"\n{block}\n{probe}'],
            env=run_env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        values = dict(
            line.split("=", 1) for line in proc.stdout.strip().splitlines() if "=" in line
        )
        values["_stdout"] = proc.stdout
        return values

    # ── absent declaration: the synthetic path must not notice this exists ──
    def test_no_declaration_leaves_every_name_unset(self):
        for script, label in self.scripts:
            with self.subTest(script=label):
                got = self._run(script, declaration=None)
                for name in DECLARED:
                    self.assertEqual(
                        got[name],
                        "<unset>",
                        msg=f"{label} set {name} with no bench_env.sh present",
                    )

    def test_no_declaration_is_silent(self):
        """No file => no log line, so a synthetic run's output is unchanged too."""
        got = self._run(BENCH_E2E, declaration=None)
        self.assertNotIn("sourcing per-run workload env", got["_stdout"])

    def test_an_empty_dir_is_not_mistaken_for_a_declaration(self):
        """`-f` not `-e`: a directory of that name must not be sourced."""
        (self.tmp / "bench_env.sh").mkdir()
        got = self._run(BENCH_E2E, declaration=None)
        self.assertEqual(got["BENCH_CLIENT"], "<unset>")

    # ── present declaration: it reaches the bench ──
    def test_a_declaration_selects_the_client_and_the_metric_axis(self):
        for script, label in self.scripts:
            with self.subTest(script=label):
                got = self._run(script, declaration=DECLARATION)
                self.assertEqual(got["BENCH_CLIENT"], "agentx", msg=label)
                self.assertEqual(got["E2E_METRIC"], "p90_intvty_inferencex", msg=label)
                self.assertEqual(
                    got["GEAK_METRIC_BASIS"], "p90_intvty_inferencex", msg=label
                )
                self.assertEqual(
                    got["GEAK_WORKLOAD_KIND"], "agentx_trace_replay", msg=label
                )

    def test_bench_e2e_announces_the_declaration_it_picked_up(self):
        got = self._run(BENCH_E2E, declaration=DECLARATION)
        self.assertIn("sourcing per-run workload env", got["_stdout"])

    # ── authority order: the environment always outranks the file ──
    def test_an_inherited_value_outranks_the_declaration(self):
        """This is what keeps an orchestrated run byte-identical."""
        for script, label in self.scripts:
            with self.subTest(script=label):
                got = self._run(
                    script,
                    declaration=DECLARATION,
                    env={"BENCH_CLIENT": "inferencex", "E2E_METRIC": "output"},
                )
                self.assertEqual(got["BENCH_CLIENT"], "inferencex", msg=label)
                self.assertEqual(got["E2E_METRIC"], "output", msg=label)
                # Names the caller did NOT pin still come from the declaration.
                self.assertEqual(
                    got["GEAK_WORKLOAD_KIND"], "agentx_trace_replay", msg=label
                )

    def test_an_explicit_bench_env_file_overrides_the_neighbour_lookup(self):
        """`BENCH_ENV_FILE` lets a caller point the channel somewhere else."""
        elsewhere = self.tmp / "other_env.sh"
        elsewhere.write_text(DECLARATION.replace("agentx", "inferencex"))
        got = self._run(
            BENCH_E2E, declaration=DECLARATION, env={"BENCH_ENV_FILE": str(elsewhere)}
        )
        self.assertEqual(got["BENCH_CLIENT"], "inferencex")

    # ── the file's own form is part of the contract ──
    def test_the_declaration_form_cannot_clobber_an_exported_value(self):
        """An unguarded `VAR=` line would break the authority order."""
        for line in DECLARATION.splitlines():
            if not line or line.startswith(("#", "export ")):
                continue
            self.assertRegex(
                line,
                r'^\[ -n "\$\{([A-Z_][A-Z0-9_]*):-\}" \] \|\| \1=\'',
                msg="a declaration may only assign an unset/empty name, "
                "so an inherited value always wins",
            )


class BothBenchScriptsCarryTheChannelTest(unittest.TestCase):
    """bench_replica.sh runs the client-specific setup BEFORE it re-execs
    bench_e2e.sh, so it cannot rely on the re-entered script to source this."""

    def test_both_scripts_source_the_declaration(self):
        for script in (BENCH_E2E, BENCH_REPLICA):
            with self.subTest(script=script.name):
                self.assertIn("bench_env.sh", script.read_text())
                _extract_source_block(script)  # asserts the exact form

    def test_the_repo_ships_no_default_declaration(self):
        """A bench_env.sh committed beside the scripts would silently apply to
        EVERY run, including synthetic ones. The file is per-run only."""
        self.assertFalse(
            (SCRIPTS / "bench_env.sh").exists(),
            "scripts/bench_env.sh must never be committed; it is written per run",
        )


def _js_table(name: str) -> dict[str, str]:
    """A flat ``const NAME = { key: 'value', ... };`` table from e2e_workflow.js."""
    match = re.search(
        rf"^const {name} = \{{\n(.*?)^\}};", WORKFLOW_JS.read_text(), re.DOTALL | re.MULTILINE
    )
    assert match, f"{name} is missing from e2e_workflow.js"
    return dict(re.findall(r"^\s*(\w+): '([^']*)',", match.group(1), re.MULTILINE))


class DeclaredAxisMatchesTheSummaryTest(unittest.TestCase):
    """The workflow writes E2E_METRIC into bench_env.sh and later checks the baseline's
    metric_basis against what it declared. Both only hold if its tables say what
    bench_summarize.py does: a token it does not know is fatal there, and a token it maps
    to another basis fails Setup after a full baseline replay."""

    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "bench_summarize_for_axis_parity", SCRIPTS / "bench_summarize.py"
        )
        cls.summarize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.summarize)
        cls.bases = {axis: entry[1] for axis, entry in cls.summarize._BASES.items()}

    def test_every_axis_the_summary_accepts_maps_to_the_same_basis(self):
        self.assertEqual(_js_table("METRIC_BASIS_OF_AXIS"), self.bases)

    def test_the_token_declared_for_each_basis_measures_that_basis(self):
        declared = _js_table("AXIS_OF_METRIC_BASIS")
        self.assertEqual(set(declared), set(self.bases.values()),
                         "every basis the summary can record needs a token to select it")
        for basis, axis in declared.items():
            with self.subTest(basis=basis):
                self.assertEqual(self.bases.get(axis), basis)

    def test_every_basis_has_a_prompt_label(self):
        self.assertEqual(set(_js_table("AXIS_LABEL")), set(self.bases.values()))

    def test_the_standalone_default_is_an_interactivity_basis_with_a_guard(self):
        default = re.search(
            r"^  metric_basis: '([^']+)',", WORKFLOW_JS.read_text(), re.MULTILINE
        ).group(1)
        self.assertEqual(default, self.summarize.P90_INTVTY_BASIS)
        self.assertIn(default, self.summarize._INTVTY_BASES)


if __name__ == "__main__":
    unittest.main()
