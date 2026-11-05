import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "detect_gpu_requirement.py"


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


def test_only_meta_json_counts_as_json_metadata(tmp_path):
    (tmp_path / "workload.json").write_text(json.dumps({"world_size": 8}))
    assert load().detect([str(tmp_path)])["kind"] == "single"
    (tmp_path / "meta.json").write_text(json.dumps({"world_size": 4}))
    assert load().detect([str(tmp_path)])["gpus_per_job"] == 4
