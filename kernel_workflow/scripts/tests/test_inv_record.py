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
    c = events(tmp_path)[1]["child"]
    assert (c["exit"], c["signal"], c["forwarded_signals"], c["descendants_held_streams"]) == (None, signal.SIGSEGV, [], False)


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
    s = ir.summarize(str(tmp_path / "rec"))["engineers"]["r1_d1"]
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
    assert not base["complete"] and any("tune.cache" in g for g in base["gaps"])  # undeclared -> incomplete
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
    assert c["evidence"] == "correctness_mode_command_exit" and c["exit"] == 1 and c["source_complete"]
    assert c["source_identity"] == start["source_before"]["identity"] and c["identity_files"]["oracle"]


def test_parsed_latencies(ws, tmp_path):
    run(ws, tmp_path, "bash", "-c", "echo 'GEAK_RESULT_LATENCY_MS=0.25 m=1'; echo 'GEAK_RESULT_LATENCY_MS=1e-1 m=2'")
    m = events(tmp_path)[1]["measurement"]
    assert m["parsed"] and m["status"] == "usable" and [c["latency_ms"] for c in m["cases"]] == [0.25, 0.1] and m["cases"][0]["case"] == "m=1"


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
    e = s["engineers"]["r1_d1"]
    assert e["declarations"] == [] and e["distinct_complete_sources"] == 2
    assert "never inferred" in s["note"]


def test_a_declared_revert_is_joined_as_consistency_evidence(ws, tmp_path):
    for content, decl in (("A", None), ("B", "revert"), ("A", None)):
        (ws / "kernel.py").write_text(content)
        run(ws, tmp_path, "true")
        if decl:
            assert _decl(tmp_path, "--kind", decl)["status"] == "recorded"
    (j,) = ir.summarize(str(tmp_path / "rec"))["engineers"]["r1_d1"]["declarations"]
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



# ------------------------------------------------------------- review of 9d3aed66 (2026-10-05)
# 1. ignored runtime content and links to it
def test_undeclared_ignored_content_and_a_link_to_it_are_incomplete(ws, tmp_path):
    (ws / "runtime.cache").write_text("1.0\n")
    os.symlink("runtime.cache", ws / "alias.py")
    git(ws, "add", "alias.py")
    f = fp(ws, tmp_path)
    assert not f["complete"]
    assert any("runtime.cache" in g and "neither declared nor excluded" in g for g in f["gaps"])
    assert any("alias.py" in g for g in f["gaps"])                       # link target not captured
    declared = fp(ws, tmp_path, dict(DOMAIN, include_ignored=["runtime.cache"]))
    assert declared["complete"], declared["gaps"]                         # captured -> link covered


# 2. a workspace that is a repo subdirectory
def test_a_subdirectory_workspace_hashes_only_its_subtree(ws, tmp_path):
    task = ws / "task"
    task.mkdir()
    (task / "k.py").write_text("t\n")
    git(ws, "add", "-A")
    git(ws, "commit", "-qm", "task")
    a = fp(task, tmp_path)
    (ws / "unrelated.py").write_text("outside\n")
    b = fp(task, tmp_path)
    assert a["complete"] and b["complete"] and a["identity"] == b["identity"] and a["workspace_prefix"] == "task/"
    (task / "k.py").write_text("changed\n")
    assert fp(task, tmp_path)["identity"] != a["identity"]
    tree = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD:task"], capture_output=True).stdout.decode().strip()
    assert a["tree"] == tree                                               # exactly the committed subtree


# 3a. recorder failure on close/flush stays out of the child's streams
def test_a_raw_file_failure_at_close_never_reaches_the_child_stream(ws, tmp_path):
    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    os.symlink("/dev/full", rec / "raw" / "fixed.stdout")
    code = ("import sys,types;sys.path.insert(0,sys.argv[1]);import inv_record as m;"
            "m.uuid.uuid4=lambda:types.SimpleNamespace(hex='fixed');sys.exit(m.main(sys.argv[2:]))")
    cmd = ["bash", "-c", "echo child-output"]
    r = subprocess.run([sys.executable, "-B", "-c", code, SCRIPTS, "run", "--rec-dir", str(rec), "--workspace",
                        str(ws), "--domain", str(tmp_path / "domain.json"), "--mode", "benchmark", "--"] + cmd,
                       capture_output=True, timeout=60)
    d = direct(*cmd)
    assert (r.returncode, r.stdout, r.stderr) == (d.returncode, d.stdout, d.stderr)
    done = events(tmp_path)[1]
    assert done["recorder_error"] and any("fixed.stdout" in e for e in done["recorder_error"])
    # the raw record is incomplete, though the child's streams and exit were fully preserved
    assert done["raw_streams"] == "incomplete" and done["raw_capture"]["stdout"]["status"] == "incomplete"
    assert set(done["raw_capture"]["stdout"]["problems"]) & {"raw_write_failed", "raw_close_failed"}
    assert done["raw_capture"]["stderr"] == {"status": "complete", "problems": []}


# 3b. descendants holding the streams; the outer-recorder -> lock -> shell shape
def test_descendants_holding_the_streams_do_not_hang_the_recording(ws, tmp_path):
    t0 = __import__("time").monotonic()
    r = run(ws, tmp_path, "bash", "-c", "sleep 30 & echo hi")
    try:
        assert r.returncode == 0 and r.stdout == b"hi\n"
        assert __import__("time").monotonic() - t0 < 20
        done = events(tmp_path)[1]
        assert done["child"]["descendants_held_streams"] is True
    finally:
        try:
            os.killpg(events(tmp_path)[1]["child"]["pgid"], signal.SIGKILL)
        except (ProcessLookupError, KeyError, IndexError):
            pass


def _sigterm_after_ready(argv, env):
    import time
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
    try:
        assert p.stdout.readline() == b"READY\n"
        t = time.monotonic()
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=15)
        return rc, time.monotonic() - t
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate(timeout=5)


def test_sigterm_through_lock_and_shell_matches_running_it_directly(ws, tmp_path):
    shape = ["bash", GPU_LOCK, "97", "bash", "-c", "sleep 30 & echo READY; wait"]
    env = _lock_env(tmp_path)
    d_rc, _ = _sigterm_after_ready(shape, env)                            # the lock script alone gets SIGTERM
    w_rc, w_s = _sigterm_after_ready([sys.executable, "-B", REC, "run", "--rec-dir", str(tmp_path / "rec"),
                                      "--workspace", str(ws), "--domain", str(tmp_path / "domain.json"),
                                      "--mode", "benchmark", "--engineer-id", "r1_d1", "--"] + shape, env)
    assert w_rc == d_rc and w_s < 10
    done = events(tmp_path)[1]
    assert done["event"] == "completion" and done["child"]["forwarded_signals"] == [signal.SIGTERM]


# 4. declarations stay with their engineer
def test_declarations_attach_only_to_the_same_engineers_invocations(ws, tmp_path):
    run(ws, tmp_path, "true")                                              # r1_d1
    subprocess.run([sys.executable, "-B", REC, "run", "--rec-dir", str(tmp_path / "rec"), "--workspace", str(ws),
                    "--mode", "other", "--engineer-id", "r1_d2", "--domain", str(tmp_path / "domain.json"), "--", "true"],
                   check=True, capture_output=True)
    starts = [e for e in events(tmp_path) if e["event"] == "start"]
    mine, other = starts[0]["inv"], starts[1]["inv"]
    assert _decl(tmp_path, "--kind", "revert")["status"] == "recorded"   # r1_d1 declares after r1_d2 ran
    d = json.loads((tmp_path / "rec" / "declarations.jsonl").read_text().splitlines()[-1])
    assert d["inv"] == mine and d["inv_source"] == "latest_of_engineer"
    assert _decl(tmp_path, "--kind", "keep", "--inv", other)["status"] == "rejected"
    s = ir.summarize(str(tmp_path / "rec"))["engineers"]
    assert s["r1_d1"]["invocations"] == 1 and s["r1_d2"]["invocations"] == 1
    assert len(s["r1_d1"]["declarations"]) == 1 and s["r1_d2"]["declarations"] == []


# 5. torn records
def test_torn_lines_keep_the_valid_prefix_and_are_counted(ws, tmp_path):
    run(ws, tmp_path, "true")
    run(ws, tmp_path, "true")
    p = tmp_path / "rec" / "invocations.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:3]) + '\n{"event":"completion","inv":')   # 2nd completion torn
    (tmp_path / "rec" / "declarations.jsonl").write_text('{"decl":')
    s = ir.summarize(str(tmp_path / "rec"))
    e = s["engineers"]["r1_d1"]
    assert e["invocations"] == 2 and e["completed"] == 1 and e["incomplete_cause_unknown"] == 1
    assert s["malformed_records"] == {"invocations": 1, "declarations": 1}


# 6. latency validation
def test_bad_latency_values_are_not_usable_measurements():
    m = ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=-1 c0\nGEAK_RESULT_LATENCY_MS=1e999 c1\n"
                           b"GEAK_RESULT_LATENCY_MS=0 c2\nGEAK_RESULT_LATENCY_MS=nan c3\nGEAK_RESULT_LATENCY_MS=0.5 c4\n")
    assert m["status"] == "partial" and not m["parsed"] and [c["case"] for c in m["cases"]] == ["c4"]
    assert len(m["invalid"]) == 4
    ir.canonical(m)                                                        # strict JSON: no Infinity/NaN
    assert ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=0.5\n")["status"] == "partial"        # no case id
    assert ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=1 a\nGEAK_RESULT_LATENCY_MS=2 a\n")["duplicate_cases"] == ["a"]
    assert ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=1 a\n", truncated=True)["status"] == "partial"
    assert ir.parse_latencies(b"nothing\n")["status"] == "none"


def test_parse_input_truncation_is_recorded(monkeypatch, tmp_path):
    import threading
    monkeypatch.setattr(ir, "STDOUT_PARSE_CAP", 4)
    r, w = os.pipe()
    os.write(w, b"0123456789")
    os.close(w)
    keep, rec, sink = [], ir.Recorder(str(tmp_path)), open(os.devnull, "wb")
    ir._tee(r, sink, str(tmp_path / "raw.out"), keep, rec, threading.Event(), [])
    assert keep == [b"0123", None] and (tmp_path / "raw.out").read_bytes() == b"0123456789"


# coverage join with the gpu_lock use-log
def test_summarize_joins_the_lock_log_by_recorder_id_only(ws, tmp_path):
    r = run(ws, tmp_path, "bash", GPU_LOCK, "97", "true", env=_lock_env(tmp_path))
    assert r.returncode == 0
    with open(tmp_path / "use.log", "a") as fh:
        fh.write('{"t":1,"gpu":3,"pool":"3","pid":1,"mode":"pin","wait_s":0}\n')
    g = ir.summarize(str(tmp_path / "rec"), str(tmp_path / "use.log"))["gpu_lock"]
    inv = events(tmp_path)[0]["inv"]
    assert g["gpu_by_invocation"] == {inv: [97]} and g["lock_lines_without_ids"] == 1
    assert g["outside_observed_population"] == "unknown"


# ------------------------------------------------------------- review of 167c4713 (2026-10-05)
def _run_patched(ws, tmp_path, patch, *cmd, env=None, mode="other"):
    code = ("import sys,subprocess,os;sys.path.insert(0,sys.argv[1]);import inv_record as m;" + patch +
            ";sys.exit(m.main(sys.argv[2:]))")
    return subprocess.run([sys.executable, "-B", "-c", code, SCRIPTS, "run", "--rec-dir", str(tmp_path / "rec"),
                           "--workspace", str(ws), "--domain", str(tmp_path / "domain.json"), "--mode", mode,
                           "--engineer-id", "r1_d1", "--"] + list(cmd), capture_output=True, timeout=60, env=env)


PGRP_PROBE = ["python3", "-c", "import os; print(os.getpgrp() == os.getpid())"]


def test_older_python_path_still_runs_the_child_in_its_own_group(ws, tmp_path):
    """GEAK supports Python >= 3.8; process_group= exists only from 3.11."""
    r = _run_patched(ws, tmp_path, "m.sys.version_info=(3,8,0)", *PGRP_PROBE)
    assert r.returncode == 0 and r.stdout == b"True\n"
    c = events(tmp_path)[1]["child"]
    assert c["grouped"] is True and events(tmp_path)[1]["recorder_error"] is None


def test_a_refused_group_runs_the_child_ungrouped_and_says_so(ws, tmp_path):
    patch = ("real=subprocess.Popen\n"
             "def P(*a,**k):\n"
             " if 'process_group' in k or 'preexec_fn' in k: raise TypeError('no group here')\n"
             " return real(*a,**k)\n"
             "m.subprocess.Popen=P")
    r = _run_patched(ws, tmp_path, patch, *PGRP_PROBE)
    assert r.returncode == 0 and r.stdout == b"False\n"                   # the child still ran
    done = events(tmp_path)[1]
    assert done["child"]["grouped"] is False and "no group here" in done["recorder_error"][0]


def _launch_descendant(tmp_path, body):
    """A parent that starts a descendant (same process group) and exits at once."""
    script = tmp_path / "launch.py"
    script.write_text("import subprocess,sys\nsubprocess.Popen([sys.executable,'-c',%r])\n%s" % body)
    return script


def _kill_group(tmp_path):
    try:
        os.killpg(events(tmp_path)[1]["child"]["pgid"], signal.SIGKILL)
    except (ProcessLookupError, KeyError, IndexError, FileNotFoundError):
        pass


def test_a_continuously_writing_descendant_cannot_keep_the_recorder_alive(ws, tmp_path):
    import time
    script = tmp_path / "launch.py"
    script.write_text("import subprocess,sys\n"
                      "subprocess.Popen([sys.executable,'-c','import os,time\\nwhile True:\\n os.write(1,b\"tick\\\\n\");time.sleep(0.02)'])\n"
                      "print('GEAK_RESULT_LATENCY_MS=0.5 caseA', flush=True)\n")
    env = dict(os.environ, GEAK_INV_RECORD_DRAIN_GRACE_S="0.2")
    t = time.monotonic()
    try:
        r = run(ws, tmp_path, sys.executable, str(script), env=env)
        assert time.monotonic() - t < 8 and r.returncode == 0
        done = events(tmp_path)[1]
        assert done["child"]["descendants_held_streams"] is True and done["raw_streams"] == "incomplete"
        assert "drain_cutoff" in done["raw_capture"]["stdout"]["problems"]
        m = done["measurement"]
        assert m["stream_cutoff"] is True and m["status"] == "partial" and not m["parsed"]
    finally:
        _kill_group(tmp_path)


def test_a_quiet_descendant_cutoff_prefix_is_not_usable(ws, tmp_path):
    script = tmp_path / "launch.py"
    script.write_text("import subprocess,sys\n"
                      "subprocess.Popen([sys.executable,'-c','import time\\ntime.sleep(5)\\nprint(\"GEAK_RESULT_LATENCY_MS=0.7 caseB\")'])\n"
                      "print('GEAK_RESULT_LATENCY_MS=0.5 caseA', flush=True)\n")
    try:
        run(ws, tmp_path, sys.executable, str(script), env=dict(os.environ, GEAK_INV_RECORD_DRAIN_GRACE_S="0.2"))
        m = events(tmp_path)[1]["measurement"]
        assert [c["case"] for c in m["cases"]] == ["caseA"] and m["status"] == "partial" and m["stream_cutoff"]
    finally:
        _kill_group(tmp_path)


def test_the_parse_cap_and_the_stream_cutoff_stay_distinct():
    a = ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=1 a\n", truncated=True)
    b = ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=1 a\n", stream_cutoff=True)
    assert (a["stdout_truncated_for_parse"], a["stream_cutoff"]) == (True, False)
    assert (b["stdout_truncated_for_parse"], b["stream_cutoff"]) == (False, True)
    assert a["status"] == b["status"] == "partial"
    assert ir.parse_latencies(b"", stream_cutoff=True)["status"] == "partial"


def _dep_layout(ws, tmp_path):
    dep, actual = tmp_path / "declared-dep", tmp_path / "indirect-dep"
    dep.mkdir()
    actual.mkdir()
    (actual / "code.py").write_text("v=1\n")
    os.symlink(str(actual), dep / "nested", target_is_directory=True)
    os.symlink(str(dep), ws / "dep", target_is_directory=True)
    return dep, actual, dict(DOMAIN, external_deps=[str(dep)])


def test_content_behind_a_nested_link_in_a_declared_dependency_is_covered(ws, tmp_path):
    dep, actual, declared = _dep_layout(ws, tmp_path)
    a = fp(ws, tmp_path, declared)
    (actual / "code.py").write_text("v=2\n")
    b = fp(ws, tmp_path, declared)
    assert a["complete"] and b["complete"] and a["identity"] != b["identity"]


def test_a_dependency_link_cycle_terminates_and_a_dangling_link_is_a_gap(ws, tmp_path):
    dep, actual, declared = _dep_layout(ws, tmp_path)
    os.symlink(str(dep), actual / "loop", target_is_directory=True)         # dep/nested/loop -> dep
    f = fp(ws, tmp_path, declared)
    assert f["complete"], f["gaps"]
    os.symlink(str(tmp_path / "missing"), dep / "dangling")
    g = fp(ws, tmp_path, declared)
    assert not g["complete"] and any("dangles" in x for x in g["gaps"])


# ------------------------------------------------------------- review of 66c18c74 (2026-10-05)
def test_a_complete_capture_says_complete(ws, tmp_path):
    run(ws, tmp_path, "bash", "-c", "echo GEAK_RESULT_LATENCY_MS=1 a")
    done = events(tmp_path)[1]
    assert done["raw_streams"] == "complete" and done["raw_capture"]["stdout"] == {"status": "complete", "problems": []}


def test_a_raw_write_failure_alone_keeps_a_fully_read_parse_usable(ws, tmp_path):
    rec = tmp_path / "rec"
    (rec / "raw").mkdir(parents=True)
    os.symlink("/dev/full", rec / "raw" / "fixed.stdout")
    code = ("import sys,types;sys.path.insert(0,sys.argv[1]);import inv_record as m;"
            "m.uuid.uuid4=lambda:types.SimpleNamespace(hex='fixed');sys.exit(m.main(sys.argv[2:]))")
    subprocess.run([sys.executable, "-B", "-c", code, SCRIPTS, "run", "--rec-dir", str(rec), "--workspace", str(ws),
                    "--domain", str(tmp_path / "domain.json"), "--mode", "benchmark", "--",
                    "bash", "-c", "echo GEAK_RESULT_LATENCY_MS=1 a"], capture_output=True, timeout=60)
    done = events(tmp_path)[1]
    assert done["raw_streams"] == "incomplete" and done["measurement"]["status"] == "usable"


def test_a_read_failure_makes_the_parse_partial():
    m = ir.parse_latencies(b"GEAK_RESULT_LATENCY_MS=1 a\n", read_failed=True)
    assert m["status"] == "partial" and m["stream_read_failed"] is True


def test_a_read_failure_is_a_capture_problem(tmp_path):
    import threading
    problems, rec = [], ir.Recorder(str(tmp_path))
    r, w = os.pipe()
    os.close(r)                                                            # reading a closed fd fails
    ir._tee(r, open(os.devnull, "wb"), str(tmp_path / "raw.out"), [], rec, threading.Event(), problems)
    os.close(w)
    assert "read_failed" in problems and rec.errors


def test_a_special_file_in_a_declared_dependency_is_a_gap(ws, tmp_path):
    dep, actual, declared = _dep_layout(ws, tmp_path)
    os.mkfifo(str(actual / "config.fifo"))                                  # reached through dep/nested
    f = fp(ws, tmp_path, declared)
    assert not f["complete"] and any("special file" in g and "config.fifo" in g for g in f["gaps"])


# ------------------------------------------------------------- review of 0832e98c (2026-10-05)
def test_a_real_short_raw_write_is_a_capture_problem(tmp_path):
    """A real OS short write (file-size limit in a child process), not a patched write()."""
    code = r"""
import io,json,os,resource,signal,sys,threading
sys.path.insert(0, sys.argv[1]); import inv_record as ir
r, w = os.pipe(); payload = b'GEAK_RESULT_LATENCY_MS=1 caseA\n' + b'x' * 227
os.write(w, payload); os.close(w)
rec, keep, problems, sink = ir.Recorder(sys.argv[2]), [], [], io.BytesIO()
raw = os.path.join(sys.argv[2], 'raw.out')
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
prev = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (64, prev[1]))
try: ir._tee(r, sink, raw, keep, rec, threading.Event(), problems)
finally: resource.setrlimit(resource.RLIMIT_FSIZE, prev)
print(json.dumps({'raw': os.path.getsize(raw), 'fwd': len(sink.getvalue()),
                  'parse': sum(len(x) for x in keep if x is not None), 'problems': problems, 'errors': rec.errors}))
"""
    out = subprocess.run([sys.executable, "-B", "-c", code, SCRIPTS, str(tmp_path)], capture_output=True, timeout=60)
    d = json.loads(out.stdout)
    assert d["raw"] == 64 and d["fwd"] == 258 and d["parse"] == 258           # stream and parse still whole
    assert set(d["problems"]) & {"raw_write_failed", "raw_write_short"} and d["errors"]


def test_partial_writes_are_completed(tmp_path):
    class Dribble:
        def __init__(self):
            self.data = b""

        def write(self, b):
            self.data += bytes(b[:3])                                      # 3 bytes per call, no error
            return min(3, len(b))
    raw, problems, rec = Dribble(), [], ir.Recorder(str(tmp_path))
    assert ir._write_all(raw, b"0123456789", "x", rec, problems) and raw.data == b"0123456789" and not problems


def test_a_write_that_stores_nothing_is_reported(tmp_path):
    class Stuck:
        def write(self, b):
            return 0
    problems, rec = [], ir.Recorder(str(tmp_path))
    assert not ir._write_all(Stuck(), b"abc", "x", rec, problems) and problems == ["raw_write_short"]
