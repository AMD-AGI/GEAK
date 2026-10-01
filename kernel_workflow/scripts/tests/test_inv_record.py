"""Offline acceptance cases for inv_record.py (Eikos phase 2a design rev 3, sec 3b). No GPU work:
the gpu_lock case uses a fake GPU id with the idle check, arch probe and reaper switched off."""
import json
import os
import signal
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REC = os.path.join(SCRIPTS, "inv_record.py")
GPU_LOCK = os.path.join(SCRIPTS, "gpu_lock.sh")
sys.path.insert(0, SCRIPTS)
import inv_record as ir  # noqa: E402

DOMAIN = {"exclude": [".torch_ext", "out"], "include_ignored": [], "external_deps": []}


def git(ws, *args):
    subprocess.run(["git", "-C", str(ws)] + list(args), check=True, capture_output=True)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    git(w, "init", "-q")
    git(w, "config", "user.email", "t@t")
    git(w, "config", "user.name", "t")
    (w / "kernel.py").write_text("x = 1\n")
    (w / "keep.py").write_text("k = 1\n")
    (w / ".gitignore").write_text(".torch_ext/\nout/\n*.cache\n")
    git(w, "add", "-A")
    git(w, "commit", "-qm", "base")
    d = tmp_path / "domain.json"
    d.write_text(json.dumps(DOMAIN))
    return w


def run(ws, tmp_path, *cmd, mode="benchmark", domain=True, env=None, extra=()):
    args = [sys.executable, "-B", REC, "run", "--rec-dir", str(tmp_path / "rec"), "--workspace", str(ws),
            "--mode", mode, "--engineer-id", "r1_d1"] + list(extra)
    if domain:
        args += ["--domain", str(tmp_path / "domain.json")]
    return subprocess.run(args + ["--"] + list(cmd), capture_output=True, env=env, timeout=120)


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "rec" / "invocations.jsonl").read_text().splitlines()]


def fp(ws, tmp_path, domain=DOMAIN):
    return ir.fingerprint(str(ws), "HEAD", domain, None, str(tmp_path / "rec"))


# 1. pass-through + recorder failure isolation
def direct(*cmd):
    return subprocess.run(list(cmd), capture_output=True, timeout=60)


def test_child_streams_and_exit_pass_through_unchanged(ws, tmp_path):
    cmd = ("bash", "-c", "printf 'a\\nGEAK_RESULT_LATENCY_MS=1.5 case0\\n'; printf 'err\\n' >&2; exit 7")
    r, d = run(ws, tmp_path, *cmd), direct(*cmd)                       # identical to running it directly
    assert (r.returncode, r.stdout, r.stderr) == (d.returncode, d.stdout, d.stderr) == (7, d.stdout, d.stderr)
    assert r.stdout == b"a\nGEAK_RESULT_LATENCY_MS=1.5 case0\n" and r.stderr.endswith(b"err\n")
    start, done = events(tmp_path)
    assert done["child"]["exit"] == 7 and done["recorder_error"] is None
    raw = tmp_path / "rec" / "raw" / (start["inv"] + ".stdout")
    assert raw.read_bytes() == r.stdout


def test_a_signal_death_is_passed_back_as_a_signal(ws, tmp_path):
    r = run(ws, tmp_path, "bash", "-c", "kill -SEGV $$")
    assert r.returncode == -signal.SIGSEGV
    assert events(tmp_path)[1]["child"] == {"exit": None, "signal": signal.SIGSEGV, "forwarded_signals": [],
                                            "wall_s": events(tmp_path)[1]["child"]["wall_s"]}


def test_recorder_failure_never_changes_the_child(ws, tmp_path):
    (tmp_path / "rec").write_text("not a dir")                          # every recorder write fails
    r, d = run(ws, tmp_path, "bash", "-c", "echo hi; exit 3"), direct("bash", "-c", "echo hi; exit 3")
    assert (r.returncode, r.stdout, r.stderr) == (d.returncode, d.stdout, d.stderr) and r.returncode == 3


def test_unexecutable_command_is_127_and_recorded(ws, tmp_path):
    r = run(ws, tmp_path, "/nonexistent/cmd")
    assert r.returncode == 127 and events(tmp_path)[1]["child"]["exit"] == 127


# 2. start without completion
def test_start_without_completion_is_incomplete_cause_unknown(ws, tmp_path):
    run(ws, tmp_path, "true")
    lines = (tmp_path / "rec" / "invocations.jsonl").read_text().splitlines()
    (tmp_path / "rec" / "invocations.jsonl").write_text(lines[0] + "\n")  # completion append lost
    s = ir.summarize(str(tmp_path / "rec"))
    assert s["invocations"] == 1 and s["completed"] == 0 and s["incomplete_cause_unknown"] == 1


# 3. source change during the command
def test_source_edited_during_command_is_not_attributed(ws, tmp_path):
    run(ws, tmp_path, "bash", "-c", "echo 'x = 2' > '%s'" % (ws / "kernel.py"))
    assert events(tmp_path)[1]["source_status"] == "source_changed_during_command"


def test_unchanged_source_is_no_net_change(ws, tmp_path):
    run(ws, tmp_path, "true")
    assert events(tmp_path)[1]["source_status"] == "no_net_change"


# 4. fingerprint coverage, and the real index/objects untouched
@pytest.mark.parametrize("edit", ["staged", "unstaged", "untracked", "deleted", "mode", "symlink"])
def test_each_kind_of_source_change_moves_the_hash(ws, tmp_path, edit):
    base = fp(ws, tmp_path)
    if edit == "staged":
        (ws / "kernel.py").write_text("x = 3\n")
        git(ws, "add", "kernel.py")
    elif edit == "unstaged":
        (ws / "kernel.py").write_text("x = 4\n")
    elif edit == "untracked":
        (ws / "new.py").write_text("n = 1\n")
    elif edit == "deleted":
        (ws / "keep.py").unlink()
    elif edit == "mode":
        os.chmod(ws / "kernel.py", 0o755)
    elif edit == "symlink":
        os.symlink("kernel.py", ws / "alias.py")
    after = fp(ws, tmp_path)
    assert base["complete"] and after["complete"] and base["identity"] != after["identity"]


def test_build_cache_and_recorder_output_do_not_move_the_hash(ws, tmp_path):
    base = fp(ws, tmp_path)
    (ws / ".torch_ext").mkdir()
    (ws / ".torch_ext" / "k.so").write_bytes(b"\0" * 10)
    (ws / "out").mkdir()
    (ws / "out" / "log.txt").write_text("x")
    assert fp(ws, tmp_path)["identity"] == base["identity"]


def test_real_index_and_objects_are_byte_identical(ws, tmp_path):
    (ws / "kernel.py").write_text("x = 9\n")
    (ws / "new.py").write_text("n\n")
    idx = (ws / ".git" / "index").read_bytes()
    objs = sorted(str(p) for p in (ws / ".git" / "objects").rglob("*"))
    run(ws, tmp_path, "true")
    assert (ws / ".git" / "index").read_bytes() == idx
    assert sorted(str(p) for p in (ws / ".git" / "objects").rglob("*")) == objs


def test_recorder_output_inside_the_domain_makes_identity_incomplete(ws, tmp_path):
    f = ir.fingerprint(str(ws), "HEAD", DOMAIN, None, str(ws / "rec_here"))
    assert not f["complete"] and any("inside the source domain" in g for g in f["gaps"])


def test_a_symlink_to_an_undeclared_outside_target_is_incomplete(ws, tmp_path):
    (tmp_path / "dep.py").write_text("d\n")
    os.symlink(str(tmp_path / "dep.py"), ws / "dep.py")
    assert not fp(ws, tmp_path)["complete"]
    declared = dict(DOMAIN, external_deps=[str(tmp_path / "dep.py")])
    ok = fp(ws, tmp_path, declared)
    assert ok["complete"]
    (tmp_path / "dep.py").write_text("changed\n")                        # target content is covered
    assert fp(ws, tmp_path, declared)["identity"] != ok["identity"]


def test_a_symlink_into_an_excluded_dir_is_incomplete(ws, tmp_path):
    (ws / ".torch_ext").mkdir()
    (ws / ".torch_ext" / "gen.py").write_text("g\n")
    os.symlink(".torch_ext/gen.py", ws / "gen.py")                     # target is in the tree's blind spot
    f = fp(ws, tmp_path)
    assert not f["complete"] and any("gen.py" in g for g in f["gaps"])


def test_no_declared_domain_is_incomplete(ws, tmp_path):
    run(ws, tmp_path, "true", domain=False)
    s = events(tmp_path)[0]["source_before"]
    assert not s["complete"] and "not declared" in s["gaps"][0]
    assert events(tmp_path)[1]["source_status"] == "incomplete"


# 5. ignored-but-read content
def test_ignored_file_is_covered_only_when_declared(ws, tmp_path):
    (ws / "tune.cache").write_text("a\n")
    base = fp(ws, tmp_path)
    (ws / "tune.cache").write_text("b\n")
    assert fp(ws, tmp_path)["identity"] == base["identity"]             # undeclared: not covered
    declared = dict(DOMAIN, include_ignored=["tune.cache"])
    one = fp(ws, tmp_path, declared)
    (ws / "tune.cache").write_text("c\n")
    assert one["complete"] and fp(ws, tmp_path, declared)["identity"] != one["identity"]
    (ws / "tune.cache").unlink()
    assert not fp(ws, tmp_path, declared)["complete"]                   # declared but missing


# 6. parsing vs correctness
def test_unparsable_output_and_exit_zero_are_not_correctness(ws, tmp_path):
    run(ws, tmp_path, "bash", "-c", "echo 'latency: 1.2ms'")
    done = events(tmp_path)[1]
    assert done["measurement"]["parsed"] is False and done["correctness"] == {"evidence": "none"}


def test_correctness_mode_records_exit_against_source_and_oracle(ws, tmp_path):
    (tmp_path / "oracle.py").write_text("o\n")
    run(ws, tmp_path, "bash", "-c", "exit 1", mode="correctness", extra=["--identity-file", "oracle=%s" % (tmp_path / "oracle.py")])
    start, done = events(tmp_path)
    c = done["correctness"]
    assert c["evidence"] == "correctness_command" and c["exit"] == 1
    assert c["source_identity"] == start["source_before"]["identity"] and c["identity_files"]["oracle"]


def test_parsed_latencies(ws, tmp_path):
    run(ws, tmp_path, "bash", "-c", "echo 'GEAK_RESULT_LATENCY_MS=0.25 m=1'; echo 'GEAK_RESULT_LATENCY_MS=1e-1 m=2'")
    m = events(tmp_path)[1]["measurement"]
    assert m["parsed"] and [c["latency_ms"] for c in m["cases"]] == [0.25, 0.1] and m["cases"][0]["case"] == "m=1"


# 7. measurement identity
def test_measurement_identity_tracks_mode_files_and_gpu(ws, tmp_path):
    cm = tmp_path / "COMMANDMENT.md"
    cm.write_text("v1\n")
    run(ws, tmp_path, "true", extra=["--identity-file", "commandment=%s" % cm, "--gpu-spec", "0"])
    cm.write_text("v2\n")
    run(ws, tmp_path, "true", mode="full_benchmark", extra=["--identity-file", "commandment=%s" % cm, "--gpu-spec", "1"])
    a, b = [e["measurement"] for e in events(tmp_path) if e["event"] == "start"]
    assert a["identity_files"]["commandment"] != b["identity_files"]["commandment"]
    assert (a["mode"], a["gpu_spec"]) != (b["mode"], b["gpu_spec"])


# 8. gpu_lock correlation
def _lock_env(tmp_path, **kw):
    env = dict(os.environ, GEAK_GPU_USE_LOG=str(tmp_path / "use.log"), GEAK_GPU_REQUIRE_IDLE="0",
               KERNEL_ENV_SKIP_ENUM_REAP="1", KERNEL_ENV_KEEP_ARCH="1")
    for k in ("GEAK_ENGINEER_ID", "GEAK_RECORDER_INV_ID", "GEAK_GPU_ALLOWED"):
        env.pop(k, None)
    env.update(kw)
    return env


def test_gpu_lock_log_carries_recorder_ids_allocated_outside(ws, tmp_path):
    r = run(ws, tmp_path, "bash", GPU_LOCK, "97", "true", env=_lock_env(tmp_path))
    assert r.returncode == 0
    line = json.loads((tmp_path / "use.log").read_text().splitlines()[-1])
    assert line["recorder_inv_id"] == events(tmp_path)[0]["inv"] and line["engineer_id"] == "r1_d1"


def test_gpu_lock_log_is_unchanged_without_ids(tmp_path):
    subprocess.run(["bash", GPU_LOCK, "97", "true"], cwd=tmp_path, env=_lock_env(tmp_path), check=True,
                   capture_output=True)
    line = json.loads((tmp_path / "use.log").read_text().splitlines()[-1])
    assert set(line) == {"t", "gpu", "pool", "pid", "mode", "wait_s"}


def test_gpu_lock_drops_a_malformed_id(tmp_path):
    subprocess.run(["bash", GPU_LOCK, "97", "true"], cwd=tmp_path, check=True, capture_output=True,
                   env=_lock_env(tmp_path, GEAK_RECORDER_INV_ID='x","pwn":"1'))
    line = json.loads((tmp_path / "use.log").read_text().splitlines()[-1])
    assert "recorder_inv_id" not in line and "pwn" not in line


# 9. declarations apart from observations; no inference from hashes
def _decl(tmp_path, *args):
    out = subprocess.run([sys.executable, "-B", REC, "declare", "--rec-dir", str(tmp_path / "rec"),
                          "--engineer-id", "r1_d1"] + list(args), capture_output=True, timeout=60)
    return json.loads(out.stdout)


def test_histories_without_declarations_stay_unknown(ws, tmp_path):
    for content in ("A", "B", "A"):                                     # A,B,A: revert OR re-timed reference
        (ws / "kernel.py").write_text(content)
        run(ws, tmp_path, "true")
    s = ir.summarize(str(tmp_path / "rec"))
    assert s["declarations"] == [] and s["distinct_complete_sources"] == 2
    assert "never inferred" in s["note"]


def test_a_declared_revert_is_joined_as_consistency_evidence(ws, tmp_path):
    for content, decl in (("A", None), ("B", "revert"), ("A", None)):
        (ws / "kernel.py").write_text(content)
        run(ws, tmp_path, "true")
        if decl:
            assert _decl(tmp_path, "--kind", decl)["status"] == "recorded"
    (j,) = ir.summarize(str(tmp_path / "rec"))["declarations"]
    assert j["kind"] == "revert" and j["consistency"] == "consistent"
    assert json.loads((tmp_path / "rec" / "declarations.jsonl").read_text())["reported_by"] == "agent"


# 10. first commitment wins
def test_a_conflicting_next_declaration_does_not_overwrite(ws, tmp_path):
    run(ws, tmp_path, "true")
    assert _decl(tmp_path, "--kind", "next", "--value", "continue", "--seq", "1")["status"] == "recorded"
    assert _decl(tmp_path, "--kind", "next", "--value", "submit", "--seq", "1")["status"] == "conflict_not_overwritten"
    kept = json.loads((tmp_path / "rec" / "commitments" / "r1_d1__1.json").read_text())
    assert kept["value"] == "continue"
    assert _decl(tmp_path, "--kind", "next", "--value", "maybe", "--seq", "2")["status"] == "rejected"
