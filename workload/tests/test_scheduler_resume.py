import json
from pathlib import Path
import sys
from types import SimpleNamespace

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "scripts"))
import scheduler


def test_resume_credit_never_regresses_to_an_older_heartbeat(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    credit, status = state / "a.credit", state / "a.status.json"
    status.write_text('{"credited_seconds": 43190}')
    credit.write_text('43180')
    assert scheduler.read_credit(str(tmp_path), 'a') == 43190
    credit.write_text('')
    assert scheduler.read_credit(str(tmp_path), 'a') == 43190
    credit.write_text('43210')
    assert scheduler.read_credit(str(tmp_path), 'a') == 43210


def test_full_budget_does_not_accept_early_done_or_interrupt_final_attempt(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "a.done").touch()
    (state / "a.credit").write_text("43201")
    status = state / "a.status.json"
    status.write_text('{"status": "done", "credited_seconds": 41000}')
    assert not scheduler.is_done(str(tmp_path), "a", 43200, True)
    status.write_text('{"status": "running", "credited_seconds": 43201}')
    assert not scheduler.is_done(str(tmp_path), "a", 43200, True)
    status.write_text('{"status": "done", "credited_seconds": 43400}')
    assert scheduler.is_done(str(tmp_path), "a", 43200, True)


def test_restart_adopts_live_worker_before_dispatch(tmp_path):
    args = SimpleNamespace(work_root=str(tmp_path), target=100, slots=[[0], [1]],
        slot_of={"gpu0": [0], "gpu1": [1]}, started_at={}, startup_grace=300,
        stale_sec=900, bad_node_limit=2, worker_env={}, tasks_dir=str(tmp_path),
        keep_overrides={}, measure_timeout=60)
    starts = []
    backend = SimpleNamespace(alive=lambda kernel, gpus: kernel == "b" and gpus == [1],
        start=lambda kernel, gpus, env: starts.append((kernel, gpus)) or True,
        stop=lambda *args: None)
    assign, _, _ = scheduler.tick(args, backend, ["a", "b"], {}, {})
    assert starts == [("a", [0])]
    assert assign == {"gpu1": "b", "gpu0": "a"}


def test_roofline_declared_task_directory_controls_order(tmp_path):
    for name in ("campaign_a", "campaign_b"):
        (tmp_path / name / "kernel_src").mkdir(parents=True)
    ranking = tmp_path / "roofline.json"
    ranking.write_text(json.dumps({"kernels": [
        {"kernel": "unrelated_native_symbol", "task_dir": "tasks/campaign_b", "expected_e2e_gain_pct": 9}]}))
    assert scheduler.discover_tasks(str(tmp_path), str(ranking)) == ["campaign_b", "campaign_a"]
