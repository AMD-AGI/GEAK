"""A resumed run reads list fields of STATE.json back in chunks; the chunks must add up to the field."""

import importlib.util
import json
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "state_chunks.py"
SPEC = importlib.util.spec_from_file_location("state_chunks", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _state(tmp_path):
    rows = [{"id": f"c{i}", "source": "workflow", "notes": "n" * 3000, "per_case": [{"x": i}] * 200}
            for i in range(20)]
    state = {"candidate_registry": rows, "insights": [f"insight {i}" for i in range(50)], "ledger": []}
    path = tmp_path / "STATE.json"
    path.write_text(json.dumps(state))
    return path, state


def _read(path, field, capsys):
    assert MODULE.main([str(path), field]) == 0
    head = json.loads(capsys.readouterr().out)
    values = []
    for i in range(head["chunks"]):
        assert MODULE.main([str(path), field, "--chunk", str(i)]) == 0
        out = capsys.readouterr().out
        assert len(out) <= MODULE.CHUNK_CHARS + 1000
        values += json.loads(out)["items"]
    assert len(values) == head["length"]
    return values


def test_registry_round_trips_without_notes(tmp_path, capsys):
    path, state = _state(tmp_path)
    rows = _read(path, "candidate_registry", capsys)
    assert rows == [{k: v for k, v in r.items() if k != "notes"} for r in state["candidate_registry"]]


def test_contract_failures_keep_ids_without_messages(tmp_path, capsys):
    failure = {"id": "fused_host_abi", "category": "abi", "severity": "required", "tier": "semantic",
               "messages": ["m" * 1000, "second"]}
    row = {"id": "c0", "source": "search", "contract_failures": [failure] * 40,
           "working_snapshot": {"head": "abc", "notes": "n" * 3000, "contract_failures": [failure] * 40}}
    path = tmp_path / "STATE.json"
    path.write_text(json.dumps({"candidate_registry": [row]}))
    [got] = _read(path, "candidate_registry", capsys)
    short = {k: v for k, v in failure.items() if k != "messages"}
    assert got["contract_failures"] == [short] * 40
    assert got["working_snapshot"] == {"head": "abc", "contract_failures": [short] * 40}


def test_other_fields_round_trip_verbatim(tmp_path, capsys):
    path, state = _state(tmp_path)
    assert _read(path, "insights", capsys) == state["insights"]
    assert _read(path, "ledger", capsys) == []
    assert _read(path, "missing", capsys) == []


def test_out_of_range_chunk_fails(tmp_path, capsys):
    path, _ = _state(tmp_path)
    assert MODULE.main([str(path), "insights", "--chunk", "99"]) == 2
