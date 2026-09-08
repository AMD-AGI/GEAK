#!/usr/bin/env python3
"""Tests for the Claude Code LLM-ledger mirror.

CONTRACT under test: a run's token/cost ledger must end up inside the run's
own output directory, in a layout the report tool can read unchanged, without
any failure in this path ever reaching the run.

The layout assertions here glob exactly the way the consumer globs rather than
describing the layout in prose — a mirror that "looks right" but is one
directory level off is unreadable, and only a literal glob catches that.

Run: python3 -m pytest GEAK/interface/test_run_e2e_trace_mirror.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "claude_trace_mirror", _HERE / "claude_trace_mirror.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ctm = _load()


# --------------------------------------------------------------------------- #
# Fixture: a miniature Claude Code home
# --------------------------------------------------------------------------- #
def _home(
    tmp_path: Path,
    *,
    run_id: str = "wf_abc123",
    eval_dir: str = "/runs/exp/eval_0",
    exp_root: str = "/runs/exp",
    session: str = "sess-1",
    slug: str = "-runs-exp",
    agents: int = 2,
    timestamp: str = "2026-01-01T00:00:00Z",
) -> Path:
    """Build a home with one workflow record and its transcripts."""
    home = tmp_path / "home"
    sess_dir = home / "projects" / slug / session
    (sess_dir / "workflows").mkdir(parents=True)
    (sess_dir / "workflows" / f"{run_id}.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "timestamp": timestamp,
                "args": {"eval_dir": eval_dir, "exp_root": exp_root},
            }
        ),
        encoding="utf-8",
    )
    sess_dir.with_suffix(".jsonl").write_text('{"type":"user"}\n', encoding="utf-8")
    tdir = sess_dir / "subagents" / "workflows" / run_id
    tdir.mkdir(parents=True)
    for i in range(agents):
        (tdir / f"agent-{i}.jsonl").write_text(
            '{"usage":{"input_tokens":1}}\n', encoding="utf-8"
        )
    return home


# --------------------------------------------------------------------------- #
# Layout — pinned against the consumer's own glob
# --------------------------------------------------------------------------- #
def test_mirror_layout_matches_consumer_glob(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    assert hit is not None
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    ctm.mirror(hit[0], hit[1], dest)

    # This is the consumer's discovery glob, verbatim.
    found = sorted(dest.glob(ctm.WORKFLOW_GLOB))
    assert len(found) == 1, f"consumer would find {found}"
    record_path = found[0]

    # And this is how the consumer derives the transcript dir from it.
    tdir = record_path.parent.parent / "subagents" / "workflows" / "wf_abc123"
    assert sorted(p.name for p in tdir.glob("agent-*.jsonl")) == [
        "agent-0.jsonl",
        "agent-1.jsonl",
    ]

    # The orchestrator conversation rides along beside the session dir.
    assert record_path.parent.parent.with_suffix(".jsonl").is_file()


def test_mirror_carries_only_this_runs_transcripts(tmp_path):
    """A session can drive several runs; the others are not this run's ledger."""
    home = _home(tmp_path)
    sess = home / "projects" / "-runs-exp" / "sess-1"
    other = sess / "subagents" / "workflows" / "wf_other"
    other.mkdir(parents=True)
    (other / "agent-9.jsonl").write_text("{}\n", encoding="utf-8")

    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    ctm.mirror(hit[0], hit[1], dest)
    assert not list(dest.rglob("agent-9.jsonl"))
    assert list(dest.rglob("agent-0.jsonl"))


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_find_record_matches_eval_dir(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    assert hit is not None and hit[1]["runId"] == "wf_abc123"


def test_find_record_falls_back_to_exp_root(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/somewhere/else", exp_root="/runs/exp")
    assert hit is not None and hit[1]["runId"] == "wf_abc123"


def test_find_record_ignores_another_run(tmp_path):
    home = _home(tmp_path)
    assert ctm.find_record([home], eval_dir="/runs/other/eval_0") is None


def test_find_record_returns_none_without_selector(tmp_path):
    home = _home(tmp_path)
    assert ctm.find_record([home]) is None


def test_find_record_prefers_newest_among_identity_matches(tmp_path):
    home = _home(tmp_path, timestamp="2026-01-01T00:00:00Z")
    sess = home / "projects" / "-runs-exp" / "sess-1" / "workflows"
    (sess / "wf_newer.json").write_text(
        json.dumps(
            {
                "runId": "wf_newer",
                "timestamp": "2026-02-01T00:00:00Z",
                "args": {"eval_dir": "/runs/exp/eval_0"},
            }
        ),
        encoding="utf-8",
    )
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    assert hit[1]["runId"] == "wf_newer"


def test_iter_records_skips_unreadable_json(tmp_path):
    home = _home(tmp_path)
    bad = home / "projects" / "-runs-exp" / "sess-1" / "workflows" / "wf_bad.json"
    bad.write_text("{not json", encoding="utf-8")
    ids = {r.get("runId") for _, r in ctm.iter_records([home])}
    assert ids == {"wf_abc123"}


# --------------------------------------------------------------------------- #
# Never raises
# --------------------------------------------------------------------------- #
def test_missing_home_is_a_clean_miss(tmp_path):
    assert ctm.find_record([tmp_path / "nope"], eval_dir="/runs/exp") is None


def test_mirror_run_trace_reports_no_record(tmp_path):
    out = ctm.mirror_run_trace(tmp_path / "eval", homes=[tmp_path / "nope"])
    assert out["status"] == "no_record"


def test_mirror_run_trace_succeeds_and_is_reported(tmp_path):
    home = _home(tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    out = ctm.mirror_run_trace(
        eval_dir, exp_root="/runs/exp", homes=[home]
    )
    assert out["status"] == "ok"
    assert out["run_id"] == "wf_abc123"
    assert out["files"] == 4  # record + convo + 2 agents
    assert Path(out["path"]) == eval_dir / ctm.MIRROR_DIRNAME


def test_mirror_run_trace_never_raises_on_a_bad_eval_dir():
    out = ctm.mirror_run_trace(None)  # type: ignore[arg-type]
    assert out["status"] in {"error", "no_record"}


# --------------------------------------------------------------------------- #
# Budget and idempotency
# --------------------------------------------------------------------------- #
def test_budget_skips_and_records_rather_than_truncating(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    manifest = ctm.mirror(hit[0], hit[1], dest, max_bytes=1)
    assert manifest["skipped"], "an over-budget file must be named, not silently dropped"
    assert all(s["reason"] == "max_bytes" for s in manifest["skipped"])
    # Whatever was skipped was not written half-way.
    for entry in manifest["skipped"]:
        assert not (dest / entry["path"]).exists()


def test_second_pass_copies_only_what_changed(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    first = ctm.mirror(hit[0], hit[1], dest)
    assert first["bytes_copied"] > 0

    second = ctm.mirror(hit[0], hit[1], dest)
    assert second["bytes_copied"] == 0
    assert len(second["files"]) == len(first["files"])

    # Append to one transcript; only that one is re-copied.
    grew = home / "projects" / "-runs-exp" / "sess-1" / "subagents" / "workflows"
    target = grew / "wf_abc123" / "agent-0.jsonl"
    target.write_text(target.read_text() + '{"usage":{"input_tokens":2}}\n', encoding="utf-8")
    os.utime(target, (target.stat().st_atime + 10, target.stat().st_mtime + 10))
    third = ctm.mirror(hit[0], hit[1], dest)
    assert third["bytes_copied"] == target.stat().st_size


def test_deadline_stops_the_pass_and_says_so(tmp_path, monkeypatch):
    """_emit may be running under a SIGTERM grace period: it must give up."""
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    ticks = iter([0.0] + [999.0] * 50)
    monkeypatch.setattr(ctm.time, "monotonic", lambda: next(ticks))
    manifest = ctm.mirror(hit[0], hit[1], dest, deadline_s=1.0)
    assert manifest["deadline_hit"] is True
    assert all(s["reason"] == "deadline" for s in manifest["skipped"])
    assert manifest["bytes_copied"] == 0


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_manifest_is_valid_json_with_the_documented_keys(tmp_path):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    ctm.mirror(hit[0], hit[1], dest)
    written = json.loads((dest / ctm.MANIFEST_NAME).read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "source_home",
        "source_record",
        "recorded_paths",
        "mirrored_at",
        "files",
        "skipped",
        "errors",
        "bytes_copied",
        "max_bytes",
        "deadline_hit",
    ):
        assert key in written, key
    assert written["source_home"] == str(home)
    assert written["recorded_paths"] == ["/runs/exp/eval_0", "/runs/exp"]


# --------------------------------------------------------------------------- #
# Volatility warning
# --------------------------------------------------------------------------- #
def test_no_warning_when_home_and_exp_root_share_a_filesystem(tmp_path):
    assert ctm.warn_if_volatile(tmp_path, tmp_path) is None


def test_warning_names_both_paths_and_the_mitigation(tmp_path, monkeypatch):
    class _St:
        def __init__(self, dev):
            self.st_dev = dev

    devs = {str(tmp_path / "home"): 1, str(tmp_path / "root"): 2}
    monkeypatch.setattr(ctm.os, "stat", lambda p: _St(devs[str(p)]))
    msg = ctm.warn_if_volatile(tmp_path / "home", tmp_path / "root")
    assert msg is not None
    assert str(tmp_path / "home") in msg and str(tmp_path / "root") in msg
    assert "CLAUDE_CONFIG_DIR" in msg
    assert ctm.MIRROR_DIRNAME in msg


def test_warning_is_silent_when_a_path_cannot_be_stated(tmp_path):
    assert ctm.warn_if_volatile(tmp_path / "gone", tmp_path) is None


# --------------------------------------------------------------------------- #
# Home resolution and budget knob
# --------------------------------------------------------------------------- #
def test_candidate_homes_follows_the_readers_precedence(tmp_path, monkeypatch):
    (tmp_path / "cfg").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    homes = ctm.candidate_homes()
    assert homes[0] == tmp_path / "cfg"
    assert Path.home() / ".claude" in homes


def test_candidate_homes_accepts_extras(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert tmp_path in ctm.candidate_homes(extra=[tmp_path])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ctm.DEFAULT_MAX_BYTES),
        ("junk", ctm.DEFAULT_MAX_BYTES),
        ("0", ctm.DEFAULT_MAX_BYTES),
        ("8", 8 * 1024 * 1024),
    ],
)
def test_max_bytes_env_knob(monkeypatch, raw, expected):
    monkeypatch.setenv("GEAK_TRACE_MIRROR_MAX_MB", raw)
    assert ctm._max_bytes() == expected


# --------------------------------------------------------------------------- #
# Rendered report is strictly best-effort
# --------------------------------------------------------------------------- #
def test_render_is_skipped_when_no_renderer_is_reachable(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_LLM_REPORT_CMD", raising=False)
    monkeypatch.setenv("HYPERLOOM_SRC", str(tmp_path / "nowhere"))
    out = ctm.render_report(tmp_path / "mirror", tmp_path / "reports")
    assert out["status"] == "skipped"


def test_render_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAK_LLM_REPORT_CMD", "false")
    out = ctm.render_report(tmp_path / "mirror", tmp_path / "reports")
    assert out["status"] in {"failed", "error", "skipped"}


# --------------------------------------------------------------------------- #
# Degraded environments — the mirror gives up quietly, never loudly
# --------------------------------------------------------------------------- #
def test_iter_records_survives_an_unglobbable_home(tmp_path, monkeypatch):
    home = _home(tmp_path)

    def _boom(self, pattern):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "glob", _boom)
    assert list(ctm.iter_records([home])) == []


def test_flat_subagent_transcripts_are_taken_too(tmp_path):
    """A session that ran no Workflow tool has no subagents/workflows/ at all."""
    home = _home(tmp_path)
    flat = home / "projects" / "-runs-exp" / "sess-1" / "subagents" / "agent-flat.jsonl"
    flat.write_text("{}\n", encoding="utf-8")
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    ctm.mirror(hit[0], hit[1], dest)
    assert list(dest.rglob("agent-flat.jsonl"))


def test_session_id_narrows_two_records_naming_the_same_run(tmp_path):
    home = _home(tmp_path)
    other = home / "projects" / "-runs-exp" / "sess-2" / "workflows"
    other.mkdir(parents=True)
    (other / "wf_zzz.json").write_text(
        json.dumps(
            {
                "runId": "wf_zzz",
                "timestamp": "2099-01-01T00:00:00Z",  # newest, so it would win
                "args": {"eval_dir": "/runs/exp/eval_0"},
            }
        ),
        encoding="utf-8",
    )
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0", session_id="sess-1")
    assert hit[1]["runId"] == "wf_abc123"

    # An unknown session id must not narrow to nothing.
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0", session_id="sess-9")
    assert hit[1]["runId"] == "wf_zzz"


def test_a_copy_failure_is_recorded_and_the_pass_continues(tmp_path, monkeypatch):
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME

    def _boom(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(ctm.shutil, "copy2", _boom)
    manifest = ctm.mirror(hit[0], hit[1], dest)
    assert len(manifest["errors"]) == 4
    assert manifest["bytes_copied"] == 0


def test_a_source_that_vanishes_mid_pass_is_recorded_not_raised(tmp_path):
    """Transcripts are live files; one can be rotated away between listing and copy."""
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")
    hit[0].unlink()  # the record itself disappears after being listed
    manifest = ctm.mirror(hit[0], hit[1], tmp_path / "eval" / ctm.MIRROR_DIRNAME)
    assert len(manifest["errors"]) == 1
    assert manifest["files"], "the surviving files are still mirrored"


def test_an_unwritable_manifest_does_not_raise(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    ctm._write_manifest(blocker / "mirror", {"run_id": "x"})  # must not raise


def test_no_renderer_without_hyperloom_src(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_LLM_REPORT_CMD", raising=False)
    monkeypatch.delenv("HYPERLOOM_SRC", raising=False)
    assert ctm._report_command(tmp_path, tmp_path) is None


def test_renderer_is_built_when_the_tool_is_present(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_LLM_REPORT_CMD", raising=False)
    tool = tmp_path / "src" / "hyperloom" / "inference_optimizer" / "tools"
    tool.mkdir(parents=True)
    (tool / "dump_geak_call_report.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_SRC", str(tmp_path / "src"))
    argv = ctm._report_command(tmp_path / "mirror", tmp_path / "out")
    assert argv is not None
    assert argv[0] == "python3"
    assert "--claude-home" in argv and str(tmp_path / "mirror") in argv


def test_a_renderer_that_cannot_be_executed_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAK_LLM_REPORT_CMD", str(tmp_path / "not-an-executable"))
    out = ctm.render_report(tmp_path / "mirror", tmp_path / "reports")
    assert out["status"] == "error"


def test_a_nonzero_renderer_is_reported(tmp_path, monkeypatch):
    script = tmp_path / "renderer.sh"
    script.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("GEAK_LLM_REPORT_CMD", str(script))
    out = ctm.render_report(tmp_path / "mirror", tmp_path / "reports")
    assert out["status"] == "error" and out.get("returncode") == 3


def test_render_result_is_folded_into_the_manifest(tmp_path, monkeypatch):
    home = _home(tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    monkeypatch.setenv("GEAK_LLM_REPORT_CMD", "true")
    out = ctm.mirror_run_trace(eval_dir, exp_root="/runs/exp", homes=[home], render=True)
    assert out["status"] == "ok"
    assert out["report"]["status"] == "ok"
    written = json.loads(
        (eval_dir / ctm.MIRROR_DIRNAME / ctm.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert written["report"]["status"] == "ok"


def test_an_unreadable_subagents_dir_still_mirrors_the_record(tmp_path, monkeypatch):
    """The record alone still yields the phase and agent tree; never lose it."""
    home = _home(tmp_path)
    hit = ctm.find_record([home], eval_dir="/runs/exp/eval_0")

    def _boom(self, *a, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "rglob", _boom)
    monkeypatch.setattr(Path, "glob", _boom)
    dest = tmp_path / "eval" / ctm.MIRROR_DIRNAME
    manifest = ctm.mirror(hit[0], hit[1], dest)
    assert [f["path"] for f in manifest["files"]][0].endswith("wf_abc123.json")


# --------------------------------------------------------------------------- #
# The runner-side helpers
# --------------------------------------------------------------------------- #
def _load_runner():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load_runner()


class _Msg:
    def __init__(self, sid):
        self.session_id = sid


def test_session_id_is_captured_from_an_sdk_message():
    rx._LAST_SDK_SESSION.clear()
    rx._note_session_id(_Msg("sess-42"))
    assert rx._LAST_SDK_SESSION["session_id"] == "sess-42"


def test_session_id_is_captured_from_the_cli_json_envelope():
    rx._LAST_SDK_SESSION.clear()
    rx._note_session_id({"session_id": "sess-cli", "result": "hi"})
    assert rx._LAST_SDK_SESSION["session_id"] == "sess-cli"


@pytest.mark.parametrize("msg", [_Msg(None), _Msg("  "), {}, object(), None])
def test_a_message_without_a_session_id_is_ignored(msg):
    rx._LAST_SDK_SESSION.clear()
    rx._note_session_id(msg)
    assert rx._LAST_SDK_SESSION == {}


def test_sdk_child_env_passes_the_config_dir_through_only_when_asked(monkeypatch):
    monkeypatch.delenv("GEAK_CLAUDE_CONFIG_DIR", raising=False)
    assert "CLAUDE_CONFIG_DIR" not in rx._sdk_child_env()
    monkeypatch.setenv("GEAK_CLAUDE_CONFIG_DIR", "/durable/.claude")
    assert rx._sdk_child_env()["CLAUDE_CONFIG_DIR"] == "/durable/.claude"


def test_sdk_child_env_sandboxes_only_under_root(monkeypatch):
    monkeypatch.delenv("GEAK_CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(rx.os, "geteuid", lambda: 0)
    assert rx._sdk_child_env()["IS_SANDBOX"] == "1"
    monkeypatch.setattr(rx.os, "geteuid", lambda: 1000)
    assert "IS_SANDBOX" not in rx._sdk_child_env()


def test_mirror_trace_without_an_eval_dir_is_a_no_op():
    assert rx._mirror_trace("")["status"] == "no_eval_dir"


def test_mirror_trace_throttles_the_mid_run_pass(monkeypatch, tmp_path):
    calls: list[bool] = []
    monkeypatch.setattr(
        rx.claude_trace_mirror,
        "mirror_run_trace",
        lambda *a, **kw: calls.append(kw["render"]) or {"status": "ok"},
    )
    monkeypatch.setattr(rx, "TRACE_MIRROR_EVERY_S", 10_000.0)
    rx._MIRROR_STATE["t"] = 0.0
    assert rx._mirror_trace(tmp_path, throttle=True)["status"] == "ok"
    assert rx._mirror_trace(tmp_path, throttle=True)["status"] == "throttled"
    assert calls == [False], "the mid-run pass never renders a report"

    # The guaranteed final pass is never throttled, and does render.
    assert rx._mirror_trace(tmp_path)["status"] == "ok"
    assert calls == [False, True]


def test_mirror_trace_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(rx, "TRACE_MIRROR_EVERY_S", 0.0)
    assert rx._mirror_trace(tmp_path, throttle=True)["status"] == "disabled"


def test_mirror_trace_never_lets_telemetry_kill_a_run(monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise RuntimeError("mirror exploded")

    monkeypatch.setattr(rx.claude_trace_mirror, "mirror_run_trace", _boom)
    out = rx._mirror_trace(tmp_path)
    assert out["status"] == "error" and "mirror exploded" in out["error"]


# --------------------------------------------------------------------------- #
# install_skill — the report's own instructions travel with the artifacts
# --------------------------------------------------------------------------- #
def test_install_skill_copies_the_shipped_skill(tmp_path):
    out = tmp_path / "reports"
    result = ctm.install_skill(out)
    assert result["status"] == "ok"
    text = (out / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: run-report" in text


def test_install_skill_skips_when_the_source_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ctm, "SKILL_RELPATH", Path("no") / "such" / "SKILL.md")
    result = ctm.install_skill(tmp_path / "reports")
    assert result["status"] == "skipped"
    assert "not found" in result["reason"]


def test_install_skill_reports_an_oserror(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(ctm.Path, "mkdir", boom)
    result = ctm.install_skill(tmp_path / "reports")
    assert result["status"] == "error"
    assert "OSError" in result["error"]
