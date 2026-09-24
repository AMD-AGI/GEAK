"""Actual adapter -> env -> CPU child checks for the EXTRA_ENV string contract."""

import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
LAUNCHERS = ("sglang", "vllm", "magpie_sglang", "magpie_vllm")
LITERALS = {
    "CUSTOM_CONFIG_PATH": "/models/(a b)/config.json",
    "SGLANG_TEST_CONFIG": '{"pattern":"(a|b)","key":"x;y"}',
    "AUDIT_QUOTES": "a'b\"c\\d\nsecond line\n",
    "AUDIT_CONTROLS": "a;b&c|d<e>f(g)`literal` $(literal) * ? [abc]",
    "AUDIT_EMPTY": "",
    "AUDIT_MARKER": "__GEAK_JSON_0__",
    "RUN_EVAL": "true",
}


def test_decoder_emits_only_nul_delimited_literal_assignments(monkeypatch, capsysbinary):
    decoder = runpy.run_path(str(SCRIPTS / "adapters/extra_env.py"))
    raw = shlex.join(["SGLANG_TEST_CONFIG=two words\nsecond line", "AUDIT_EMPTY="])
    monkeypatch.setattr(sys, "argv", ["extra_env.py", raw])
    decoder["main"]()
    captured = capsysbinary.readouterr()
    assert captured.out == b"SGLANG_TEST_CONFIG=two words\nsecond line\0AUDIT_EMPTY=\0"
    assert captured.err == b""


def test_decoder_rejects_malformed_quotes_without_partial_output(monkeypatch, capsysbinary):
    decoder = runpy.run_path(str(SCRIPTS / "adapters/extra_env.py"))
    monkeypatch.setattr(sys, "argv", ["extra_env.py", "RUN_EVAL=true CONFIG='broken"])
    with pytest.raises(ValueError, match="No closing quotation"):
        decoder["main"]()
    assert capsysbinary.readouterr().out == b""


def launch(tmp_path, launcher, raw, expected, *, staged=False, overrides=None):
    capture = tmp_path / "child.json"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['ENV_CAPTURE']).write_text(json.dumps({"
        "'env': {k: os.environ.get(k) for k in json.loads(os.environ['ENV_KEYS'])}, "
        "'argv': sys.argv[1:]}))\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("python", "vllm"):
        executable = bindir / name
        executable.write_text(f"#!{sys.executable}\n" + child.read_text())
        executable.chmod(0o755)
    magpie = tmp_path / "magpie.sh"
    magpie.write_text(
        f"#!/usr/bin/env bash\n{shlex.quote(sys.executable)} {shlex.quote(str(child))}\n"
        "echo \"$ENV_STUB_PID\" > \"$MAGPIE_SERVER_PID_FILE\"\n"
    )
    backend = launcher.removeprefix("magpie_")
    adapters = SCRIPTS / "adapters"
    if staged:
        # Director stages only adapters/, not the repo's interface/ or packages.
        adapters = Path(shutil.copytree(adapters, tmp_path / "staged/adapters"))
        python3 = bindir / "python3"
        python3.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
        python3.chmod(0o755)
    adapter = adapters / ("launchers/magpie.sh" if launcher.startswith("magpie_") else backend + ".sh")
    env = dict(os.environ)
    for key in [*expected, "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "RECIPE_ENV_FILE", "PYTHONPATH", "OVERLAY_PYTHONPATH"]:
        env.pop(key, None)
    env.update(PATH=str(bindir) + os.pathsep + os.environ["PATH"], EXTRA_ENV=raw,
               EXTRA_SERVER_ARGS="", GPU_ARCHS="gfx950", GPU="0", PROFILE_DIR="", PROFILE="0",
               MODEL="/cpu-only/no-model", HOST="127.0.0.1", PORT="1", TP="1", MEM_FRACTION="0.9",
               LOG=str(tmp_path / "server.log"), OUT_DIR=str(tmp_path), BACKEND=backend,
               MAGPIE_LAUNCH_SCRIPT=str(magpie), SERVER_LAUNCH_PREFIX="", WATCHDOG_TIMEOUT="",
               SGLANG_SRC_PYTHONPATH="", ENV_CAPTURE=str(capture), ENV_KEYS=json.dumps(list(expected)))
    env.update(overrides or {})
    # A raw glob must remain literal even when a matching filename exists.
    (tmp_path / "AUDIT_GLOB=expanded").touch()
    driver = 'set -eu -o pipefail\nsource "$1"\nadapter_launch\n'
    stub = None
    if not launcher.startswith("magpie_"):
        driver += 'wait "$SERVER_PID"\n'
    else:
        # Magpie returns a live server PID for the adapter's ownership checks.
        # Keep that CPU stand-in owned by this test so it can be joined exactly;
        # the actual env child above still records the adapter's launch values.
        stub = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", backend],
            start_new_session=True,
        )
        env["ENV_STUB_PID"] = str(stub.pid)
    try:
        result = subprocess.run(["bash", "-c", driver, "env-transport-test", str(adapter)],
                                env=env, cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False)
    finally:
        if stub is not None:
            stub.kill()
            stub.wait(timeout=5)
    return result, capture


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(("raw", "expected"), [
    (shlex.join([f"{key}={value}" for key, value in LITERALS.items()]), LITERALS),
    ('SGLANG_TEST_CONFIG={"pattern": "(a|b)", "key": "x;y"} RUN_EVAL=true',
     {"SGLANG_TEST_CONFIG": '{"pattern": "(a|b)", "key": "x;y"}', "RUN_EVAL": "true"}),
    ("RUN_EVAL=true\nAUDIT_GLOB=* AUDIT_EMPTY=", {"RUN_EVAL": "true", "AUDIT_GLOB": "*", "AUDIT_EMPTY": ""}),
    ('AUDIT_MARKER=__GEAK_JSON_0__ SGLANG_TEST_CONFIG={"x":1}',
     {"AUDIT_MARKER": "__GEAK_JSON_0__", "SGLANG_TEST_CONFIG": '{"x":1}'}),
])
def test_actual_child_receives_literal_values(tmp_path, launcher, raw, expected, staged):
    result, capture = launch(tmp_path, launcher, raw, expected, staged=staged)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(capture.read_text())["env"] == expected


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_malformed_quotes_stop_before_child_launch(tmp_path, launcher):
    result, capture = launch(tmp_path, launcher, "SGLANG_TEST_CONFIG='unterminated", {"SGLANG_TEST_CONFIG": None})
    assert result.returncode != 0
    assert "invalid EXTRA_ENV" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_invalid_identifiers_stop_before_child_launch(tmp_path, launcher):
    raw = shlex.join(["A\n=bad", "-SCUDA_VISIBLE_DEVICES=7", "notanassignment", "RUN_EVAL=true"])
    result, capture = launch(tmp_path, launcher, raw, {"A\n": None, "RUN_EVAL": "true"})
    assert result.returncode != 0
    assert "invalid EXTRA_ENV" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("readd", [False, True])
def test_removed_recipe_and_ambient_env_reach_real_child(tmp_path, launcher, staged, readd):
    replay = tmp_path / "recipe.nul"
    replay.write_bytes(b"SGLANG_AITER_MLA_PERSIST=recipe\0KEEP_RECIPE=kept\0")
    expected = {"SGLANG_AITER_MLA_PERSIST": "3" if readd else None,
                "AMBIENT_ONLY": None, "KEEP_AMBIENT": "kept"}
    raw = "SGLANG_AITER_MLA_PERSIST=3" if readd else ""
    result, capture = launch(tmp_path, launcher, raw, expected, staged=staged, overrides={
        "GEAK_UNSET_ENVS": json.dumps(["SGLANG_AITER_MLA_PERSIST", "AMBIENT_ONLY"]),
        "SGLANG_AITER_MLA_PERSIST": "ambient", "AMBIENT_ONLY": "ambient",
        "KEEP_AMBIENT": "kept", "RECIPE_ENV_FILE": str(replay),
    })
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(capture.read_text())["env"] == expected


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("invalid", ['["-S"]', '["X=1"]', '[null]', '{"X":1}', 'broken'])
def test_invalid_unset_transport_stops_before_child(tmp_path, launcher, invalid):
    result, capture = launch(tmp_path, launcher, "", {"AUDIT": None},
                             overrides={"GEAK_UNSET_ENVS": invalid})
    assert result.returncode != 0
    assert not capture.exists()


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_actual_handoff_resolver_and_launcher_preserve_env_removal(tmp_path, launcher, monkeypatch):
    from interface import run_e2e as rx

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("benchmark:\n  envs:\n    SGLANG_AITER_MLA_PERSIST: '1'\n")
    handoff = {
        "schema_version": 2, "framework": launcher.removeprefix("magpie_"),
        "model_path": "/cpu-only/no-model", "exp_root": str(tmp_path / "geak"),
        "eval_dir": str(tmp_path / "eval"), "launch_recipe": str(recipe),
        "bench_launcher": "magpie" if launcher.startswith("magpie_") else "native",
        "baseline_env_spec": {"config": {"args_mode": "replace", "extra_server_args": "",
            "server_launch_flags": "", "unset_envs": ["SGLANG_AITER_MLA_PERSIST"]}},
    }
    mapped = rx.map_args(handoff)
    assert mapped["initial_extra_server_args"] == ""
    assert mapped["initial_extra_env"] == ""
    assert mapped["initial_env_complete"] is True
    assert mapped["initial_unset_envs"] == ["SGLANG_AITER_MLA_PERSIST"]
    with monkeypatch.context() as context:
        context.setattr(os, "environ", dict(os.environ))
        rx.apply_bench_launcher(handoff)
        overrides = {key: os.environ[key] for key in
                     ("GEAK_UNSET_ENVS", "RECIPE_ENV_FILE") if key in os.environ}
    overrides["SGLANG_AITER_MLA_PERSIST"] = "ambient"
    result, capture = launch(tmp_path, launcher, mapped["initial_extra_env"],
                             {"SGLANG_AITER_MLA_PERSIST": None}, overrides=overrides)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(capture.read_text())["env"] == {"SGLANG_AITER_MLA_PERSIST": None}


@pytest.mark.parametrize("launcher,key", [
    ("sglang", "GPU_ARCHS"), ("vllm", "GPU_ARCHS"),
    ("magpie_sglang", "MAX_MODEL_LEN"), ("magpie_vllm", "MAX_MODEL_LEN"),
])
@pytest.mark.parametrize("readd", [False, True])
def test_adapter_defaults_do_not_restore_removed_or_override_readded_env(tmp_path, launcher, key, readd):
    # A resolved re-add clears the final unset list, yet its explicit assignment
    # must still outrank the launcher's ambient/default value.
    raw = f"{key}=explicit" if readd else ""
    expected = {key: "explicit" if readd else None}
    result, capture = launch(tmp_path, launcher, raw, expected, overrides={
        key: "ambient", "GEAK_UNSET_ENVS": json.dumps([] if readd else [key]),
    })
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(capture.read_text())["env"] == expected
