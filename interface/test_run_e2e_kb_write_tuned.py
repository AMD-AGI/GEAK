#!/usr/bin/env python3
"""Tests for ``_kb_write_tuned_ops`` — the salvage path that files proven tuned tables.

CONTRACT under test. The orchestrator (``e2e_workflow.js``, step ``kernel-kb:write-tuned``) is the
normal writer. Under an outer runner that slices wall-clock, a run can finish its tuning A/B, prove
engagement, and be killed with the verdict living only in the role's context. This function reads
the role's persisted return off disk and files it instead. So:

  1. It NEVER double-files. The store has no delete, so a second write with the same content mints a
     second record that the page counts as an independent REPRODUCTION — which can promote a
     candidate on one measurement seen twice. Presence of the orchestrator's receipt is the lock.
  2. It gates per op exactly as the orchestrator does: ``engaged is True`` AND
     ``isolated_speedup > 1.0`` AND a non-empty artifact path, plus a phase gate of ``accepted``.
  3. It writes ``--carrier tuned_artifact``, which is what distinguishes a tuned data table from a
     source ``patch`` in the unified store.
  4. The serving context reaches ``value.upstream`` and never the key. ``dims`` is the raw argv of
     the e2e read, so its keys carry the FLAG spellings — ``framework-version``, hyphenated. Reading
     ``framework_version`` here finds nothing and the record lands unlabelled, where it is
     indistinguishable from a backlog entry and can outrank a correctly-labelled one on a
     mismatched-dtype read. That spelling is pinned by a test here because nothing else catches it.
  5. It never raises. A missed record is not a failed run — every subprocess failure mode becomes a
     row in the receipt.

Run: python3 -m pytest GEAK/interface/test_run_e2e_kb_write_tuned.py -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e_kb_write_tuned", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


# --------------------------------------------------------------------------------------- fixtures


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _op(**extra) -> dict:
    op = {
        "op": "gemm_a16w16",
        "backend": "aiter",
        "engaged": True,
        "isolated_speedup": 1.0108,
        "artifact": "/tmp/tuning/deploy/files/aiter/configs/kimik3_m8192_bf16_tuned_gemm.csv",
        "tuner": "aiter-gemm-tuner",
        "shapes": "m8192,n1024",
    }
    op.update(extra)
    return op


def _tuning(**extra) -> dict:
    t = {
        "gate": "accepted",
        "ops_tuned": [_op()],
        "apply_env": "AITER_TUNE_GEMM_CONFIG=/opt/configs",
        "cache_invalidation": ["rm -rf /tmp/aiter_configs", "rm -rf ~/.cache/aiter"],
    }
    t.update(extra)
    return t


def _identity(**extra) -> dict:
    d = {
        "plane": "local",
        "store": "",
        "dims": {
            "gfx": "gfx950",
            "precision": "mxfp4",
            "framework": "sglang",
            "framework-version": "0.5.15",
        },
    }
    d.update(extra)
    return d


@pytest.fixture()
def eval_dir(tmp_path: Path) -> Path:
    """An eval dir with a clean, accepted tuning result and a warm-start identity file."""
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    return tmp_path


class _Recorder:
    """Stands in for subprocess.run, capturing the argv the writer builds."""

    def __init__(self, stdout: str = '{"written": true, "id": "rec-1"}', exc: Exception | None = None):
        self.calls: list[list[str]] = []
        self._stdout = stdout
        self._exc = exc

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if self._exc is not None:
            raise self._exc
        return subprocess.CompletedProcess(argv, 0, stdout=self._stdout, stderr="")

    @property
    def cmd(self) -> list[str]:
        """The experience_store.py argv of the single recorded call (past the `bash -c` wrapper)."""
        assert len(self.calls) == 1, self.calls
        argv = self.calls[0]
        # ["bash", "-c", shell, "bash", sys.executable, store_script, verb, ...]
        return argv[4:]

    def flag(self, name: str) -> str:
        cmd = self.cmd
        return cmd[cmd.index(name) + 1]


@pytest.fixture()
def rec(monkeypatch) -> _Recorder:
    r = _Recorder()
    monkeypatch.setattr(rx.subprocess, "run", r)
    return r


# ------------------------------------------------------------------------------------- skip gates


def test_write_back_disabled_by_env(eval_dir, monkeypatch):
    monkeypatch.setenv("GEAK_E2E_KB_WRITE_BACK", "0")
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["skipped"] is True
    assert "GEAK_E2E_KB_WRITE_BACK" in out["why"]


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", " No "])
def test_env_off_spellings_all_disable(eval_dir, monkeypatch, value):
    monkeypatch.setenv("GEAK_E2E_KB_WRITE_BACK", value)
    assert rx._kb_write_tuned_ops(eval_dir).get("skipped") is True


def test_orchestrator_receipt_is_the_lock(eval_dir, rec):
    """Presence of the workflow's own receipt must stop this from minting a duplicate."""
    _write(eval_dir / rx.TUNING_KB_WRITE_FILE, {"ok": True, "written": 1})
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["skipped"] is True
    assert rx.TUNING_KB_WRITE_FILE in out["why"]
    assert rec.calls == []


def test_no_tuning_result_file(tmp_path):
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["skipped"] is True
    assert rx.TUNING_RESULT_FILE in out["why"]


@pytest.mark.parametrize("gate", ["no_win", "rejected", "", None])
def test_phase_gate_must_be_accepted(tmp_path, gate, rec):
    """The Kimi-K3 shape: a real op-level win whose e2e delta sat inside the noise band."""
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(gate=gate))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["skipped"] is True
    assert "not accepted" in out["why"]
    assert rec.calls == []


def test_missing_gfx_is_refused(tmp_path, rec):
    """A tuned table is valid for exactly one arch; without gfx there is nothing to key it to."""
    ident = _identity()
    ident["dims"] = {k: v for k, v in ident["dims"].items() if k != "gfx"}
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, ident)
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["skipped"] is True
    assert "gfx" in out["why"]
    assert rec.calls == []


def test_missing_identity_file_reads_as_no_gfx(tmp_path, rec):
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["skipped"] is True
    assert "gfx" in out["why"]
    assert rec.calls == []


@pytest.mark.parametrize("bad", [
    {"engaged": False},                 # tuned, but the runtime never read it
    {"engaged": None},                  # unproven is not proven
    {"isolated_speedup": 1.0},          # a tie is not a win
    {"isolated_speedup": 0.98},
    {"isolated_speedup": None},
    {"isolated_speedup": "not-a-number"},
    {"artifact": ""},                   # a win with nothing to file
    {"artifact": "   "},
])
def test_per_op_gate_rejects(tmp_path, bad, rec):
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=[_op(**bad)]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["skipped"] is True
    assert "isolated_speedup>1.0" in out["why"]
    assert rec.calls == []


def test_non_dict_ops_are_ignored(tmp_path, rec):
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=["gemm", None, 7]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    assert rx._kb_write_tuned_ops(tmp_path).get("skipped") is True
    assert rec.calls == []


# ------------------------------------------------------------------------------------- happy path


def test_files_a_proven_op_as_a_tuned_artifact(eval_dir, rec):
    out = rx._kb_write_tuned_ops(eval_dir)

    assert out["ok"] is True
    assert out["measured_by"] == "run_e2e:salvage"
    assert out["ops"] == 1 and out["written"] == 1
    assert out["results"] == [{"written": True, "id": "rec-1"}]

    cmd = rec.cmd
    assert cmd[1] == str(rx.KERNEL_STORE_SCRIPT)
    assert cmd[2] == "write"                       # local plane: no remote verb, no plane flags
    assert "--plane" not in cmd and "--store" not in cmd
    assert rec.flag("--carrier") == "tuned_artifact"
    assert rec.flag("--kernel-class") == "tuning"
    assert rec.flag("--metric-kind") == "tuning_isolated"
    assert rec.flag("--kernel-name") == "gemm_a16w16"
    assert rec.flag("--language") == "aiter"
    assert rec.flag("--gfx") == "gfx950"
    assert rec.flag("--speedup") == "1.0108"
    assert rec.flag("--tuner") == "aiter-gemm-tuner"
    assert rec.flag("--direction") == "tuning-aiter"
    assert rec.flag("--eval-dir") == str(eval_dir)


def test_receipt_lands_on_disk_so_a_rerun_is_a_no_op(eval_dir, rec, monkeypatch):
    first = rx._kb_write_tuned_ops(eval_dir)
    receipt_path = eval_dir / rx.TUNING_KB_WRITE_FILE
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == first

    second = rx._kb_write_tuned_ops(eval_dir)
    assert second["skipped"] is True
    assert len(rec.calls) == 1                     # the second call never reached the store


def test_multiple_ops_each_get_their_own_write(tmp_path, rec):
    _write(tmp_path / rx.TUNING_RESULT_FILE,
           _tuning(ops_tuned=[_op(), _op(op="fmoe_fp8", backend="triton"), _op(engaged=False)]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["ops"] == 2 and out["written"] == 2
    assert len(rec.calls) == 2
    names = [argv[argv.index("--kernel-name") + 1] for argv in rec.calls]
    assert names == ["gemm_a16w16", "fmoe_fp8"]


def test_short_name_is_accepted_when_op_is_absent(tmp_path, rec):
    op = _op()
    del op["op"]
    op["short_name"] = "gemm_a8w8"
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=[op]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.flag("--kernel-name") == "gemm_a8w8"


def test_unnamed_op_is_a_failed_row_not_a_crash(tmp_path, rec):
    op = _op()
    del op["op"]
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=[op]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    out = rx._kb_write_tuned_ops(tmp_path)
    assert out["ok"] is True and out["written"] == 0
    assert out["results"] == [{"written": False, "reason": "unnamed op"}]
    assert rec.calls == []


def test_backend_defaults_to_tuned_and_direction_is_slugged(tmp_path, rec):
    op = _op(backend="AITER GEMM v2")
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=[op]))
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.flag("--direction") == "tuning-aiter-gemm-v2"

    rec.calls.clear()
    op2 = _op()
    del op2["backend"]
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(ops_tuned=[op2]))
    (tmp_path / rx.TUNING_KB_WRITE_FILE).unlink()
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.flag("--language") == "tuned"
    assert rec.flag("--direction") == "tuning-op"


def test_apply_env_and_cache_invalidation_ride_along(eval_dir, rec):
    rx._kb_write_tuned_ops(eval_dir)
    assert rec.flag("--apply-env") == "AITER_TUNE_GEMM_CONFIG=/opt/configs"
    assert rec.flag("--cache-invalidation") == "rm -rf /tmp/aiter_configs && rm -rf ~/.cache/aiter"


def test_case_names_use_semicolons(eval_dir, rec):
    """The store splits on commas, so a comma-separated shape list would read as one bad name."""
    rx._kb_write_tuned_ops(eval_dir)
    assert rec.flag("--case-names") == "m8192;n1024"


def test_report_path_is_optional(tmp_path, rec):
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity())
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    rx._kb_write_tuned_ops(tmp_path)
    assert "--report" not in rec.cmd

    rec.calls.clear()
    (tmp_path / rx.TUNING_KB_WRITE_FILE).unlink()
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning(report_path=" /tmp/tuning/report.md "))
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.flag("--report") == "/tmp/tuning/report.md"


# ------------------------------------------------------------- the serving context (value.upstream)


def test_upstream_flags_use_the_hyphenated_dims_spelling(eval_dir, rec):
    """``dims`` is the raw argv of the e2e read: ``framework-version``, not ``framework_version``.

    Underscoring it here silently drops the version and the record lands unlabelled.
    """
    rx._kb_write_tuned_ops(eval_dir)
    assert rec.flag("--precision") == "mxfp4"
    assert rec.flag("--serving-framework") == "sglang"
    assert rec.flag("--serving-framework-version") == "0.5.15"


def test_underscored_framework_version_is_not_read(tmp_path, rec):
    ident = _identity()
    ident["dims"].pop("framework-version")
    ident["dims"]["framework_version"] = "0.5.15"
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, ident)
    rx._kb_write_tuned_ops(tmp_path)
    assert "--serving-framework-version" not in rec.cmd


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_upstream_values_are_omitted_not_sent_empty(tmp_path, rec, blank):
    ident = _identity()
    ident["dims"]["precision"] = blank
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, ident)
    rx._kb_write_tuned_ops(tmp_path)
    assert "--precision" not in rec.cmd


def test_upstream_never_enters_the_key(eval_dir, rec):
    """Precision is a FILTER. If it forked the address, every new dtype would start cold."""
    rx._kb_write_tuned_ops(eval_dir)
    cmd = rec.cmd
    key_flags = {"--kernel-name", "--gfx", "--kernel-class", "--carrier"}
    assert key_flags.issubset(set(cmd))
    # the key is name/gfx/class/carrier; precision rides beside it, not inside it
    assert cmd.index("--precision") > cmd.index("--carrier")


# ----------------------------------------------------------------------------------- remote plane


def test_remote_plane_writes_to_both(tmp_path, rec):
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE,
           _identity(plane="remote", store="/srv/geak/kb_store_local"))
    rx._kb_write_tuned_ops(tmp_path)
    cmd = rec.cmd
    assert cmd[2] == "write-remote"
    assert rec.flag("--plane") == "both"
    assert rec.flag("--store") == "/srv/geak/kb_store_local"


def test_remote_plane_without_a_store_falls_back_to_local_write(tmp_path, rec):
    """No store path means nothing to write remotely; the local verb still records it."""
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, _identity(plane="remote", store=""))
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.cmd[2] == "write"
    assert "--plane" not in rec.cmd


def test_plane_defaults_to_remote_when_unrecorded(tmp_path, rec):
    ident = _identity(store="/srv/kb")
    del ident["plane"]
    _write(tmp_path / rx.TUNING_RESULT_FILE, _tuning())
    _write(tmp_path / rx.KB_IDENTITY_FILE, ident)
    rx._kb_write_tuned_ops(tmp_path)
    assert rec.cmd[2] == "write-remote"


# ------------------------------------------------------------------------------- failure is a row


def test_timeout_becomes_a_row(eval_dir, monkeypatch):
    def boom(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, rx.KB_WRITE_TIMEOUT_S)

    monkeypatch.setattr(rx.subprocess, "run", boom)
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["ok"] is True and out["written"] == 0
    assert out["results"][0]["written"] is False
    assert "timed out" in out["results"][0]["reason"]


def test_arbitrary_subprocess_failure_becomes_a_row(eval_dir, monkeypatch):
    monkeypatch.setattr(rx.subprocess, "run", _Recorder(exc=OSError("no bash on this box")))
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["ok"] is True and out["written"] == 0
    assert out["results"][0]["reason"].startswith("OSError: no bash on this box")


def test_unparseable_store_output_becomes_a_row(eval_dir, monkeypatch):
    monkeypatch.setattr(rx.subprocess, "run", _Recorder(stdout="Traceback (most recent call last)"))
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["ok"] is True and out["written"] == 0
    assert "JSONDecodeError" in out["results"][0]["reason"]


def test_an_unwritable_eval_dir_does_not_sink_the_run(eval_dir, rec, monkeypatch):
    """The receipt is best-effort. Losing it costs the idempotence lock, never the run."""
    def no_write(self, *a, **kw):
        raise OSError("read-only file system")

    monkeypatch.setattr(rx.Path, "write_text", no_write)
    out = rx._kb_write_tuned_ops(eval_dir)
    assert out["ok"] is True and out["written"] == 1
