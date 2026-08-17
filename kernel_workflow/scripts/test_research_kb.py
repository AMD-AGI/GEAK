from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("research_kb.py")
SPEC = importlib.util.spec_from_file_location("research_kb", MODULE_PATH)
assert SPEC and SPEC.loader
research_kb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(research_kb)


GRAPH_DIRECTION = {
    "id": "D1",
    "title": "Wrapper-level HIP graph capture",
    "specialty": "host_runtime",
    "bottleneck": "launch overhead",
    "mechanism": (
        "Capture the full repeated kernel dispatch sequence and replay it to "
        "remove Python and launch overhead."
    ),
    "expected_upside": "2-4x on launch-bound shapes",
    "implementation_cost": "medium",
    "confidence": "high",
    "kill_criterion": "Graph replay does not beat eager execution.",
    "rank_score": 9.2,
    "evidence": [
        {
            "title": "HIP Graph documentation",
            "url": "https://rocm.docs.amd.com/hipgraph",
            "kind": "docs",
            "note": "Defines capture and replay.",
        }
    ],
}

GRAPH_DIRECTION_WEEK_TWO = {
    "id": "D3",
    "title": "Full-wrapper graph replay",
    "specialty": "host_runtime",
    "bottleneck": "dispatch and launch floor",
    "mechanism": (
        "Use HIP graph capture for the entire repeated dispatch sequence, then "
        "replay the graph to collapse launch and Python overhead."
    ),
    "expected_upside": "1.8-3x on small shapes",
    "implementation_cost": "medium",
    "confidence": "medium",
    "kill_criterion": "Measured replay latency is not below eager latency.",
    "rank_score": 8.8,
    "evidence": [
        {
            "title": "PyTorch graph API",
            "url": "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs",
            "kind": "docs",
            "note": "Static-buffer replay pattern.",
        }
    ],
}

TILING_DIRECTION = {
    "id": "D2",
    "title": "Cooperative LDS input tiling",
    "specialty": "memory",
    "bottleneck": "memory",
    "mechanism": (
        "Stage coalesced input tiles in LDS and reuse them across the workgroup "
        "to reduce repeated global loads."
    ),
    "expected_upside": "1.2-1.4x",
    "implementation_cost": "medium",
    "confidence": "medium",
    "kill_criterion": "L2 traffic and latency do not fall.",
    "rank_score": 6.5,
    "evidence": [],
}


def _write_run(
    root: Path,
    name: str,
    directions: list[dict],
    *,
    rejected: list[object] | None = None,
) -> Path:
    run = root / name
    (run / "research").mkdir(parents=True)
    (run / "deep_search.json").write_text(
        json.dumps(
            {
                "intro": "Synthetic research.",
                "directions": directions,
                "open_measurements": ["Measure launch floor."],
                "rejected_directions": rejected
                if rejected is not None
                else ["Do not use an approximate algorithm."],
            }
        ),
        encoding="utf-8",
    )
    (run / "research" / "facts.json").write_text(
        json.dumps(
            {
                "kernel_language": "hip",
                "kernel_backend": "HIP on gfx950",
                "bottleneck_type": "latency",
            }
        ),
        encoding="utf-8",
    )
    return run


def _kernel(root: Path, name: str = "kernel") -> Path:
    path = root / name
    path.mkdir()
    (path / "kernel.hip").write_text(
        'extern "C" __global__ void kernel(float* x) { x[0] += 1; }\n',
        encoding="utf-8",
    )
    return path


def _ingest(run: Path, kb: Path, kernel: Path, *, gfx: str = "gfx950"):
    return research_kb.ingest(
        eval_dir=run,
        kb_dir=kb,
        kernel_path=kernel,
        operator="reduction",
        language="hip",
        backend="hip",
        gfx=gfx,
        dtype="bf16",
        regime="decode",
        bottleneck="latency",
        kernel_name="demo",
    )


def test_ingest_is_immediate_idempotent_and_snapshot_backed(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    run = _write_run(tmp_path, "week1", [GRAPH_DIRECTION, TILING_DIRECTION])

    first = _ingest(run, kb, kernel)
    second = _ingest(run, kb, kernel)

    assert first["cards_created"] == 2
    assert first["card_count"] == 2
    assert second["cards_created"] == 0
    assert second["observations_unchanged"] == 2
    assert second["snapshot_id"] == first["snapshot_id"]
    assert len(list((kb / "cards" / "reduction").glob("*.json"))) == 2
    assert len(list((kb / "cards" / "reduction").glob("*.md"))) == 2
    assert (kb / "observations" / f"{first['run_id']}.json").exists()
    assert (kb / "snapshots" / f"{first['snapshot_id']}.json").exists()
    assert json.loads((kb / "channels" / "latest.json").read_text())[
        "snapshot_id"
    ] == first["snapshot_id"]


def test_weekly_wording_variation_merges_into_one_canonical_card(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    week1 = _write_run(tmp_path, "week1", [GRAPH_DIRECTION])
    week2 = _write_run(tmp_path, "week2", [GRAPH_DIRECTION_WEEK_TWO])

    first = _ingest(week1, kb, kernel)
    second = _ingest(week2, kb, kernel)

    assert first["cards_created"] == 1
    assert second["cards_created"] == 0
    assert second["cards_merged"] == 1
    cards = list((kb / "cards" / "reduction").glob("*.json"))
    assert len(cards) == 1
    card = json.loads(cards[0].read_text())
    assert card["support_count"] == 2
    assert len(card["source_runs"]) == 2
    assert {item["url"] for item in card["evidence"]} == {
        "https://rocm.docs.amd.com/hipgraph",
        "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs",
    }
    # The first Researcher statement remains canonical; later runs add evidence
    # and observed ranges instead of continually rewriting the KB.
    assert card["mechanism"] == GRAPH_DIRECTION["mechanism"]
    assert card["expected_upside_observed"] == [
        GRAPH_DIRECTION["expected_upside"],
        GRAPH_DIRECTION_WEEK_TWO["expected_upside"],
    ]
    old_snapshot, old_cards = research_kb._load_snapshot_cards(
        kb, first["snapshot_id"]
    )
    assert old_snapshot == first["snapshot_id"]
    assert old_cards[0]["support_count"] == 1
    assert len(old_cards[0]["evidence"]) == 1


def test_scope_boundary_prevents_cross_arch_merge(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    week1 = _write_run(tmp_path, "week1", [GRAPH_DIRECTION])
    week2 = _write_run(tmp_path, "week2", [GRAPH_DIRECTION_WEEK_TWO])

    _ingest(week1, kb, kernel, gfx="gfx950")
    second = _ingest(week2, kb, kernel, gfx="gfx942")

    assert second["cards_created"] == 1
    assert second["cards_merged"] == 0
    assert second["card_count"] == 2


def test_later_rejection_marks_existing_card_contested_without_duplication(
    tmp_path: Path,
):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    week1 = _write_run(tmp_path, "week1", [GRAPH_DIRECTION])
    week2 = _write_run(
        tmp_path,
        "week2",
        [],
        rejected=[
            {
                "title": "Wrapper-level HIP graph capture",
                "reason": "Replay remained slower than eager execution on this regime.",
            }
        ],
    )

    _ingest(week1, kb, kernel)
    second = _ingest(week2, kb, kernel)

    assert second["cards_created"] == 0
    assert second["cards_contested"] == 1
    cards = list((kb / "cards" / "reduction").glob("*.json"))
    assert len(cards) == 1
    card = json.loads(cards[0].read_text())
    assert card["contested"] is True
    assert len(card["contested_observations"]) == 1
    output = tmp_path / "offline.md"
    research_kb.retrieve(
        kb_dir=kb,
        output=output,
        kernel_path=kernel,
        operator="reduction",
        language="hip",
        backend="hip",
        gfx="gfx950",
        dtype="bf16",
        regime="decode",
        kernel_name="demo",
    )
    assert "Conflicting Researcher observations exist" in output.read_text()


def test_retrieve_materializes_online_compatible_brief(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    run = _write_run(tmp_path, "week1", [GRAPH_DIRECTION, TILING_DIRECTION])
    ingested = _ingest(run, kb, kernel)
    output = tmp_path / "offline" / "deep_search_brief.offline.md"

    result = research_kb.retrieve(
        kb_dir=kb,
        output=output,
        kernel_path=kernel,
        operator="reduction",
        language="hip",
        backend="hip",
        gfx="MI350X / gfx950",
        dtype="bf16",
        regime="decode",
        bottleneck="latency",
        kernel_name="demo",
        snapshot_id=ingested["snapshot_id"],
        max_directions=8,
    )

    assert result["cards_retrieved"] == 2
    assert result["snapshot_id"] == ingested["snapshot_id"]
    brief = output.read_text()
    assert "Deep Search Brief — offline Researcher knowledge" in brief
    assert "Wrapper-level HIP graph capture" in brief
    assert "Cooperative LDS input tiling" in brief
    assert "**Specialty:** host_runtime" in brief
    retrieval = json.loads(
        (output.parent / "research_kb_retrieval.json").read_text()
    )
    assert retrieval["snapshot_id"] == ingested["snapshot_id"]
    assert len(retrieval["cards"]) == 2
    comparison_path = tmp_path / "online_offline_comparison.json"
    comparison = research_kb.compare_online_offline(
        kb_dir=kb,
        online_json=run / "deep_search.json",
        offline_retrieval=output.parent / "research_kb_retrieval.json",
        output=comparison_path,
    )
    assert comparison["equivalent"] is True
    assert comparison["direction_recall"] == 1.0
    assert comparison["mean_mechanism_similarity"] == 1.0
    assert comparison["specialty_agreement"] == 1.0
    assert comparison_path.exists()


def test_retrieve_with_incompatible_scope_returns_no_brief(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path, "source")
    run = _write_run(tmp_path, "week1", [GRAPH_DIRECTION])
    _ingest(run, kb, kernel)
    different = _kernel(tmp_path, "different")
    output = tmp_path / "offline.md"

    result = research_kb.retrieve(
        kb_dir=kb,
        output=output,
        kernel_path=different,
        operator="attention_decode_paged",
        language="triton",
        gfx="gfx942",
        dtype="fp16",
        regime="prefill",
        kernel_name="attention",
    )

    assert result["cards_retrieved"] == 0
    assert result["brief_path"] == ""
    assert not output.exists()


def test_validation_events_are_append_only_and_idempotent(tmp_path: Path):
    kb = tmp_path / "kb"
    kernel = _kernel(tmp_path)
    run = _write_run(tmp_path, "week1", [GRAPH_DIRECTION])
    ingested = _ingest(run, kb, kernel)

    kwargs = {
        "kb_dir": kb,
        "snapshot_id": ingested["snapshot_id"],
        "eval_dir": run,
        "kernel_path": kernel,
        "kernel_name": "demo",
        "dra_mode": "offline",
        "card_ids": ",".join(ingested["card_ids"]),
        "final_speedup": 1.18,
        "validation_status": "accepted",
        "correctness": "pass",
    }
    first = research_kb.record_validation(**kwargs)
    second = research_kb.record_validation(**kwargs)

    assert first["validation_recorded"] is True
    assert second["validation_recorded"] is False
    assert first["validation_event_id"] == second["validation_event_id"]
    events = [
        json.loads(line)
        for line in (kb / "validation" / "events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["card_ids"] == ingested["card_ids"]
    assert events[0]["final_speedup"] == 1.18
