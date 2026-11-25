import importlib.util
import json
import shutil
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "detect_gpu_requirement.py"
UT_SCRIPTS = Path(__file__).resolve().parents[3] / "e2e_workflow" / "scripts"


def load():
    spec = importlib.util.spec_from_file_location("detect_gpu_requirement", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("files", "kind", "count"),
    [
        ({"unittest.py": "import torch\nout = a @ b\n"}, "single", 1),
        ({"run.sh": "torchrun --nproc_per_node=8 unittest.py\n"}, "fixed", 8),
        (
            {"unittest.py": "import torch.multiprocessing as mp\nmp.spawn(run, nprocs=4)\n"},
            "fixed",
            4,
        ),
        (
            {"unittest.py": "import torch.distributed as dist\nWORLD_SIZE = 2\n"
             "dist.init_process_group('nccl')\n"},
            "fixed",
            2,
        ),
        (
            {"bench.py": "import argparse, torch.distributed\n"
             "p.add_argument('--world-size', type=int, default=8)\n"},
            "fixed",
            8,
        ),
        ({"meta.json": json.dumps({"resource": {"gpus_per_job": 8}})}, "fixed", 8),
        (
            {"config.yaml": "correctness_command:\n- torchrun --nproc_per_node=8 run.py correctness\n"},
            "fixed",
            8,
        ),
        (
            {"unittest.py": "import torch\nimport torch.multiprocessing as mp\n"
             "mp.spawn(run, nprocs=torch.cuda.device_count())\n"},
            "scalable",
            0,
        ),
        (
            {"unittest.py": "import torch.distributed as dist\n"
             "world = int(os.environ.get('WORLD_SIZE', '1'))\n"
             "dist.init_process_group('nccl')\n"},
            "unknown",
            0,
        ),
    ],
)
def test_detects_rank_count(tmp_path, files, kind, count):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    result = load().detect([str(tmp_path)])
    assert result["kind"] == kind
    assert result["gpus_per_job"] == count
    assert bool(result["evidence"]) == (kind != "single")


def test_counts_in_single_process_files_are_ignored(tmp_path):
    (tmp_path / "unittest.py").write_text(
        "from multiprocessing import Pool\npool = Pool(processes=16)\nnum_gpus = 4\n"
    )
    assert load().detect([str(tmp_path)])["kind"] == "single"


def test_largest_count_wins_and_scan_is_shallow(tmp_path):
    (tmp_path / "a.sh").write_text("torchrun --nproc_per_node=2 t.py\n")
    (tmp_path / "b.sh").write_text("torchrun --nproc_per_node=8 t.py\n")
    deep = tmp_path / "x" / "y" / "z"
    deep.mkdir(parents=True)
    (deep / "c.sh").write_text("torchrun --nproc_per_node=64 t.py\n")
    result = load().detect([str(tmp_path)])
    assert (result["kind"], result["gpus_per_job"]) == ("fixed", 8)


def test_e2e_capture_artifacts_are_skipped(tmp_path):
    # An extracted task keeps what was captured from the serving process; a TP8 server records 8.
    trace = json.dumps({"distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 8}})
    (tmp_path / "unittest.py").write_text("import torch\nout = a @ b\n")
    (tmp_path / "meta.json").write_text(json.dumps({"baseline_callable": "pkg.mod:fn"}))
    (tmp_path / "selection_trace.pid-1.rank-unknown.call-0.json").write_text(trace)
    (tmp_path / "capture_meta.promoted.json").write_text(trace)
    for sub in ("_capture", "capture.pid-1.rank-0"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "meta.json").write_text(trace)
    module = load()
    assert module.detect([str(tmp_path)])["kind"] == "single"
    explicit = tmp_path / "selection_trace.pid-1.rank-unknown.call-0.json"
    assert module.detect([str(explicit)])["kind"] == "single"


def test_capture_traces_do_not_crowd_the_test_out_of_the_scan(tmp_path):
    module = load()
    for i in range(module.MAX_FILES + 1):
        (tmp_path / f"selection_trace.pid-1.rank-unknown.call-{i}.json").write_text("{}")
    (tmp_path / "unittest.py").write_text("import torch.multiprocessing as mp\nmp.spawn(run, nprocs=2)\n")
    assert module.detect([str(tmp_path)])["gpus_per_job"] == 2


def test_the_runtime_a_task_carries_is_not_its_code(tmp_path):
    # leg_runner.py (serve mode), ut_runner.py (rank pools) and overlay_setup.py mention torch.distributed: copied
    # into a single-GPU task without a declared count, they made it `unknown`
    module = load()
    (tmp_path / "unittest.py").write_text("import torch\nout = a @ b\n")
    for name in module.RUNTIME_FILES:
        if (UT_SCRIPTS / name).exists():    # this branch ships no ut_runner.py
            shutil.copy(UT_SCRIPTS / name, tmp_path / name)
    assert module.detect([str(tmp_path)]) == {"kind": "single", "gpus_per_job": 1, "evidence": []}
    assert module.detect([str(tmp_path / "leg_runner.py"), str(tmp_path / "unittest.py")])["kind"] == "single"


def test_only_meta_json_counts_as_json_metadata(tmp_path):
    (tmp_path / "workload.json").write_text(json.dumps({"world_size": 8}))
    assert load().detect([str(tmp_path)])["kind"] == "single"
    (tmp_path / "meta.json").write_text(json.dumps({"world_size": 4}))
    assert load().detect([str(tmp_path)])["gpus_per_job"] == 4


def _task(tmp_path, files):
    for name, data in files.items():
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(data if isinstance(data, str) else json.dumps(data))
    return tmp_path


# The task's own GPU count: what dispatch sizes the job by when the lane gets no gpus_per_job.
DISTRIBUTED_NO_COUNT = ("import torch.distributed as dist\n"
                        "world = int(os.environ['WORLD_SIZE'])\ndist.init_process_group('gloo')\n")


@pytest.mark.parametrize(
    "files",
    [
        {"run/requested_kernel.json": {"required_gpu_count": 4}},
        {"run/RETURN.json": {"required_gpu_count": 4, "unittest_smoke": "pass"}},
        {"meta.json": {"capabilities": {"multi_rank": {"world_size": 4, "comm": "peer_memory"}}}},
    ],
)
def test_declared_count_is_used(tmp_path, files):
    task = _task(tmp_path, {"unittest.py": DISTRIBUTED_NO_COUNT, **files})
    result = load().detect([str(task)])
    assert (result["kind"], result["gpus_per_job"]) == ("fixed", 4)
    assert any("(declared)" in e for e in result["evidence"])


def test_declared_count_found_from_task_files(tmp_path):
    # The Director passes the task's test file and meta.json; run/ records sit next to them.
    task = _task(tmp_path, {"unittest.py": "import torch\nout = a @ b\n", "meta.json": {"name": "t"},
                            "run/requested_kernel.json": {"required_gpu_count": 2}})
    module = load()
    for args in ([task / "unittest.py", task / "meta.json"], [task / "run" / "requested_kernel.json"]):
        result = module.detect([str(a) for a in args])
        assert (result["kind"], result["gpus_per_job"]) == ("fixed", 2)


def test_declared_count_wins_over_the_scan(tmp_path):
    task = _task(tmp_path, {"run.sh": "torchrun --nproc_per_node=2 unittest.py\n",
                            "run/RETURN.json": {"required_gpu_count": 4}})
    assert load().detect([str(task)])["gpus_per_job"] == 4


def test_disagreeing_declarations_take_the_largest_and_say_so(tmp_path):
    task = _task(tmp_path, {"run/requested_kernel.json": {"required_gpu_count": 2},
                            "run/RETURN.json": {"required_gpu_count": 4},
                            "meta.json": {"capabilities": {"multi_rank": {"world_size": 4}}}})
    result = load().detect([str(task)])
    assert (result["kind"], result["gpus_per_job"]) == ("fixed", 4)
    assert "disagree" in result["evidence"][0]


def test_declared_one_settles_unknown_but_not_a_scanned_count(tmp_path):
    module = load()
    task = _task(tmp_path / "a", {"unittest.py": DISTRIBUTED_NO_COUNT,
                                  "meta.json": {"capabilities": {"multi_rank": {"world_size": 1}}}})
    result = module.detect([str(task)])
    assert (result["kind"], result["gpus_per_job"]) == ("single", 1)
    assert "(declared)" in result["evidence"][0]
    # A count the code states is kept (main's behavior), as is a plain single-GPU task.
    task = _task(tmp_path / "b", {"run.sh": "torchrun --nproc_per_node=8 t.py\n",
                                  "run/RETURN.json": {"required_gpu_count": 1}})
    assert module.detect([str(task)])["gpus_per_job"] == 8
    task = _task(tmp_path / "c", {"unittest.py": "import torch\nout = a @ b\n",
                                  "run/requested_kernel.json": {"required_gpu_count": 1}})
    assert module.detect([str(task)])["kind"] == "single"


@pytest.mark.parametrize("value", [0, -2, "4", True, 2.5, None, [4]])
def test_unusable_declarations_are_ignored(tmp_path, value):
    task = _task(tmp_path, {"unittest.py": "import torch\nout = a @ b\n",
                            "run/requested_kernel.json": {"required_gpu_count": value},
                            "run/RETURN.json": "{not json"})
    assert load().detect([str(task)]) == {"kind": "single", "gpus_per_job": 1, "evidence": []}


def test_capture_world_size_is_not_a_declaration(tmp_path):
    # Only capabilities.multi_rank.world_size declares; a captured server's TP8 stays out.
    task = _task(tmp_path, {"unittest.py": "import torch\nout = a @ b\n",
                            "meta.json": {"capabilities": {"multi_rank": {"comm": "collective_api"}},
                                          "deployment": {"tp": 8}},
                            "selection_trace.pid-1.rank-0.call-0.json": {"world_size": 8}})
    assert load().detect([str(task)])["kind"] == "single"


# --declaration: what the task declares, as deterministic JSON for the lane's Setup (task_declaration).
UT_RUNNER_PATH = Path(__file__).resolve().parents[3] / "e2e_workflow" / "scripts" / "ut_runner.py"


def ut_runner():
    spec = importlib.util.spec_from_file_location("ut_runner_for_declaration", UT_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declare(tmp_path, meta=None, request=None, result=None):
    files = {"unittest.py": "import torch\nout = a @ b\n"}
    if meta is not None:
        files["meta.json"] = meta
    if request is not None:
        files["run/requested_kernel.json"] = request
    if result is not None:
        files["run/RETURN.json"] = result
    task = _task(tmp_path, files)
    return task, load().declaration(str(task))


WORLD4 = {"capabilities": {"multi_rank": {"world_size": 4}}}


@pytest.mark.parametrize(
    ("timeouts", "total"),
    [
        ({"total_s": 2400}, 2400),
        ({"total_s": 2400.0}, 2400),
        ({"total_s": 1234.5}, 1234.5),
        ({"total_s": "1800"}, 1800),        # float("1800") in ut_runner as well
        ({"setup_s": 60, "command_s": 600}, 3600),  # no total_s: ut_runner's multi-rank default
    ],
)
def test_declaration_ut_total_is_ut_runners(tmp_path, timeouts, total):
    meta = {**WORLD4, "timeouts": timeouts}
    task, decl = _declare(tmp_path, meta=meta)
    assert decl["required_gpu_count"] == 4
    assert decl["ut_total_s"] == total
    assert decl["evidence"][0] == f"{task}/meta.json: capabilities.multi_rank.world_size = 4 (declared)"
    assert decl["evidence"][-1].startswith(f"ut_total_s = {total}: {task}/meta.json: meta.timeouts.total_s ")


@pytest.mark.parametrize(
    ("meta", "why"),
    [
        (WORLD4, "meta.timeouts.total_s absent: ut_runner's default for world_size 4 is 3600"),
        ({**WORLD4, "timeouts": None}, "meta.timeouts.total_s absent: ut_runner's default for world_size 4 is 3600"),
        ({**WORLD4, "timeouts": {"total_s": None}}, "meta.timeouts.total_s is null: ut_runner's default for world_size 4 is 3600"),
    ],
)
def test_declaration_without_a_limit_takes_ut_runners_multi_rank_default(tmp_path, meta, why):
    task, decl = _declare(tmp_path, meta=meta)
    assert (decl["required_gpu_count"], decl["ut_total_s"]) == (4, 3600)
    assert decl["evidence"][-1] == f"ut_total_s = 3600: {task}/meta.json: {why}"


@pytest.mark.parametrize(
    ("timeouts", "why"),
    [
        ({"total_s": "abc"}, 'meta.timeouts.total_s is not a number ("abc")'),
        ({"total_s": [3600]}, "meta.timeouts.total_s is not a number ([3600])"),
        ({"total_s": {"s": 1}}, 'meta.timeouts.total_s is not a number ({"s": 1})'),
        ({"total_s": True}, "meta.timeouts.total_s is not a number (true)"),
        ({"total_s": 0}, "meta.timeouts.total_s = 0 (ut_runner timeouts, world_size 4), not a positive number of seconds"),
        ({"total_s": -5}, "meta.timeouts.total_s = -5 (ut_runner timeouts, world_size 4), not a positive number of seconds"),
        ("x", 'meta.timeouts is not an object ("x")'),
    ],
)
def test_declaration_with_an_unusable_limit_takes_3600_and_says_why(tmp_path, timeouts, why):
    task, decl = _declare(tmp_path, meta={**WORLD4, "timeouts": timeouts})
    assert (decl["required_gpu_count"], decl["ut_total_s"]) == (4, 3600)
    assert decl["evidence"][-1] == f"ut_total_s = 3600: {task}/meta.json: {why}; the multi-rank default 3600 is used"


def test_declaration_of_an_old_multi_rank_task_and_of_an_undeclared_one(tmp_path):
    # wf_neg_mimo_tp4: the count in both run/ records, no meta.timeouts (a top-level count in meta is not one).
    task, decl = _declare(tmp_path / "wf_neg", meta={"required_gpu_count": 4},
                          request={"required_gpu_count": 4}, result={"required_gpu_count": 4, "unittest_smoke": "pass"})
    assert decl == {"required_gpu_count": 4, "ut_total_s": 3600, "evidence": [
        f"{task}/run/requested_kernel.json: required_gpu_count = 4 (declared)",
        f"{task}/run/RETURN.json: required_gpu_count = 4 (declared)",
        f"ut_total_s = 3600: {task}/meta.json: meta.timeouts.total_s absent: ut_runner's default for world_size 4 is 3600"]}
    # No declaration: null count; ut_runner's single-GPU total_s 0 means "no limit", which no lease can take.
    task, decl = _declare(tmp_path / "plain", meta={"name": "t"})
    assert decl == {"required_gpu_count": None, "ut_total_s": 3600, "evidence": [
        f"{task}: no GPU count declared (run/requested_kernel.json or run/RETURN.json required_gpu_count, "
        "meta.json capabilities.multi_rank.world_size)",
        f"ut_total_s = 3600: {task}/meta.json: meta.timeouts.total_s absent: ut_runner's default for world_size 1 is 0, "
        "not a positive number of seconds; the multi-rank default 3600 is used"]}
    # A single-GPU task's own limit is used as is.
    _, decl = _declare(tmp_path / "limited", meta={"timeouts": {"total_s": 900}})
    assert (decl["required_gpu_count"], decl["ut_total_s"]) == (None, 900)


def test_declaration_disagreeing_and_unusable_counts(tmp_path):
    task, decl = _declare(tmp_path / "a", meta=WORLD4, request={"required_gpu_count": 2}, result={"required_gpu_count": "8"})
    assert decl["required_gpu_count"] == 4
    assert decl["evidence"][:4] == [
        "declared GPU counts disagree (2, 4); the largest is used",
        f"{task}/run/requested_kernel.json: required_gpu_count = 2 (declared)",
        f'{task}/run/RETURN.json: required_gpu_count = "8" (not a positive integer: ignored)',
        f"{task}/meta.json: capabilities.multi_rank.world_size = 4 (declared)"]
    task, decl = _declare(tmp_path / "b", meta={"capabilities": {"multi_rank": {"world_size": 1}}})
    assert decl["required_gpu_count"] == 1
    # A declared 1 runs ut_runner's single-GPU defaults.
    assert decl["evidence"][-1].endswith("ut_runner's default for world_size 1 is 0, not a positive number of seconds; "
                                         "the multi-rank default 3600 is used")


def test_declaration_cli_is_deterministic_json(tmp_path):
    import subprocess
    import sys
    task = _task(tmp_path, {"meta.json": {**WORLD4, "timeouts": {"total_s": 2400}},
                            "run/requested_kernel.json": {"required_gpu_count": 4}})
    run = lambda *a: subprocess.run([sys.executable, str(MODULE_PATH), *a], capture_output=True, text=True)  # noqa: E731
    first, second = run("--declaration", str(task)), run("--declaration", str(task / "meta.json"))
    assert first.returncode == 0 and first.stdout == second.stdout
    assert json.loads(first.stdout) == load().declaration(str(task))
    assert list(json.loads(first.stdout)) == ["required_gpu_count", "ut_total_s", "evidence"]
    missing = run("--declaration", str(tmp_path / "nope"))
    assert missing.returncode != 0 and "no task dir" in missing.stderr
    # The scan (no flag) is unchanged by the task's timeouts.
    assert json.loads(run(str(task)).stdout)["gpus_per_job"] == 4


def _tree(root):
    """root and everything under it, hidden entries too: {relative path: (mode, mtime_ns, a file's bytes)}."""
    tree = {}
    for p in [root, *root.rglob("*")]:
        st = p.lstat()
        content = p.read_bytes() if p.is_file() and not p.is_symlink() else None
        tree[p.relative_to(root).as_posix()] = (st.st_mode, st.st_mtime_ns, content)
    return tree


def test_declaration_only_reads_the_task_dir(tmp_path):
    # Every lane's Setup, single-GPU ones too, runs --declaration on the original task dir
    # ($KERNEL_PATH_ORIG), not the lane's copy: afterwards the dir and run/ hold the same files with the
    # same bytes, and nothing new (no __pycache__). The task's .py leaves a file behind if it is imported or run.
    import os
    import subprocess
    import sys
    task = tmp_path / "task"
    _task(task, {"meta.json": {**WORLD4, "timeouts": {"total_s": 2400}},
                 "run/requested_kernel.json": {"required_gpu_count": 4},
                 "run/RETURN.json": {"required_gpu_count": 4, "unittest_smoke": "pass"},
                 "unittest.py": f"open({str(task / 'IMPORTED')!r}, 'w').write(__name__)\n"})
    # Bytecode writing stays on, so importing the task's code would add __pycache__ to the task dir.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    before = _tree(task)
    assert sorted(before) == [".", "meta.json", "run", "run/RETURN.json", "run/requested_kernel.json", "unittest.py"]
    for arg in (task, task / "unittest.py", task / "run" / "RETURN.json"):
        run = subprocess.run([sys.executable, str(MODULE_PATH), "--declaration", str(arg)],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        decl = json.loads(run.stdout)
        assert (decl["required_gpu_count"], decl["ut_total_s"]) == (4, 2400)
        assert decl["evidence"][:3] == [f"{task}/run/requested_kernel.json: required_gpu_count = 4 (declared)",
                                        f"{task}/run/RETURN.json: required_gpu_count = 4 (declared)",
                                        f"{task}/meta.json: capabilities.multi_rank.world_size = 4 (declared)"]
        assert _tree(task) == before, f"--declaration {arg} changed the task dir"
    # The comparison does see an import of the task's code: it adds __pycache__ and the file the .py writes.
    subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import unittest", str(task)],
                   cwd=tmp_path, env=env, check=True)
    added = set(_tree(task)) - set(before)
    assert {"IMPORTED", "__pycache__"} <= added
