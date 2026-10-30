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
