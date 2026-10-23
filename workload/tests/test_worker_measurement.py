"""Exercise the worker's actual measurement transport without GPUs or an agent.

The seven observed rows below are compact projections of qualified task receipts:
their aggregate location, score values and native {ok: bool} gate form are retained. Timing
cases are not reproduced because the worker must transport the harness aggregate,
not derive a different result from those cases. No campaign paths are required.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OBSERVED = [
    ("custom_allreduce", 1.0000728737935418, 0.9885973159038647, 31),
    ("unified_attention", 1.000147704535395, 0.9986655024704032, 34),
    ("draft_attention", 0.9992403942007807, None, 9),
    ("norm_residual", 0.9911625966771236, 1.0011308409443607, 17),
    ("kv_indices", 0.9899329305592746, 0.9691591510402655, 18),
    ("fused_moe", 0.9979070696432663, 0.9773535830674343, 18),
    ("routing", 0.9995486302962227, 0.9845003286618085, 17),
]


def receipt(primary=1.2, *, top_level=False):
    result = {"ok": True, "exit_code": 0,
              "gates": {"binding": {"ok": True}, "correctness": {"ok": True}}}
    aggregate = {"primary": primary}
    if top_level:
        result["gates"]["weighted_timing"] = {"ok": True}
        result["serving_weighted_speedup"] = aggregate
        result["per_case"] = []
    else:
        result["gates"]["timing"] = {"ok": True, "aggregate": aggregate, "per_case": []}
    return result


def measure(tmp_path, *, result=None, output="", print_result=False, entry_exit=0):
    """Run default_measure unchanged, with a CPU entry that emits a frozen receipt."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPTS / "run_task_entry.py", scripts / "run_task_entry.py")
    worker = (SCRIPTS / "worker.sh").read_text()
    function = worker.split("default_measure() {", 1)[1].split("\nmeasure() {", 1)[0]
    parser = worker.split("  parse_measurement() {", 1)[1].split("\n  }\n", 1)[0]
    shell = scripts / "measure.sh"
    shell.write_text("default_measure() {" + function + "\nparse_measurement() {" + parser +
                     '\n}\nvalue="$(default_measure)"\nparse_measurement "$value"\n'
                     'echo "STATUS=$OK SPEEDUP=$SP CLEAN=$CLEAN SINGLE=$SINGLE"\n')
    task, run = tmp_path / "task", tmp_path / "worker_run"
    task.mkdir()
    run.mkdir()
    body = "import json, pathlib, time\n"
    if result is not None:
        body += "result = json.loads(" + repr(json.dumps(result)) + ")\n"
        body += "result['started'] = time.time()\n"
        body += "pathlib.Path('run').mkdir()\n"
        body += "pathlib.Path('run/ut_result.json').write_text(json.dumps(result))\n"
        if print_result:
            body += "print(json.dumps(result))\n"
    body += "print(" + repr(output) + ")\n"
    body += "raise SystemExit(" + repr(entry_exit) + ")\n"
    (task / "unittest.py").write_text(body)
    env = dict(os.environ, TASK_DIR=str(task), RUN=str(run), MEASURE_TIMEOUT="5")
    process = subprocess.run(["bash", str(shell)], env=env, text=True,
                             capture_output=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert process.stderr == "", process.stderr
    return dict(field.split("=", 1) for field in process.stdout.strip().split())


@pytest.mark.parametrize("name,primary,geomean,included", OBSERVED,
                         ids=[row[0] for row in OBSERVED])
def test_observed_task_receipt_transports_authoritative_aggregate(tmp_path, name, primary, geomean, included):
    top_level = name == "draft_attention"
    result = receipt(primary, top_level=top_level)
    if top_level:
        result["gates"] = {key: {"ok": True} for key in (
            "binding", "captured_oracles", "random_values", "poisoned_destinations",
            "output_independence", "graph_input_refresh", "leaf_execution", "identity_control",
            "numerical_negative_control", "restoration", "weighted_timing")}
        aggregate = result["serving_weighted_speedup"]
    else:
        aggregate = result["gates"]["timing"]["aggregate"]
    aggregate.update(weighted=primary, geomean=geomean, included=included)
    fields = measure(tmp_path, result=result, print_result=top_level)
    assert fields["STATUS"] == "1"
    assert float(fields["SPEEDUP"]) == primary
    assert fields["CLEAN"] == ""
    assert fields["SINGLE"] == "0"


@pytest.mark.parametrize("top_level", [False, True])
@pytest.mark.parametrize("score", [None, "", "broken", True, False, 0, -1,
                                  float("nan"), float("inf"), -float("inf"), [], {}])
def test_missing_or_malformed_receipt_score_is_not_zero_measurement(tmp_path, top_level, score):
    fields = measure(tmp_path, result=receipt(score, top_level=top_level))
    assert fields["STATUS"] == "0"
    assert fields["SPEEDUP"] == ""
    assert fields["SINGLE"] == "0"


def test_top_level_score_does_not_override_failed_gate(tmp_path):
    result = receipt(1.8, top_level=True)
    result["gates"]["correctness"]["ok"] = False
    fields = measure(tmp_path, result=result)
    assert fields["STATUS"] == "0"


def test_failed_entry_cannot_publish_a_passing_score(tmp_path):
    fields = measure(tmp_path, output="CORRECTNESS: PASS\nGEAK_GEOMEAN_SPEEDUP 1.8", entry_exit=1)
    assert fields == {"STATUS": "0", "SPEEDUP": "", "CLEAN": "", "SINGLE": "0"}


@pytest.mark.parametrize("output", [
    "CORRECTNESS: PASS\nGEAK_GEOMEAN_SPEEDUP 1.25",
    '{"exit": 0, "weighted": 1.25}',
    '{"exit": 0, "weighted": "1.25"}',
])
def test_existing_text_and_json_contracts(tmp_path, output):
    fields = measure(tmp_path, output=output)
    assert fields["STATUS"] == "1"
    assert float(fields["SPEEDUP"]) == 1.25


@pytest.mark.parametrize("contract", ["text", "json", "receipt"])
def test_finite_scientific_score_survives_the_worker_handoff(tmp_path, contract):
    score = 1.25e-6
    result = receipt(score) if contract == "receipt" else None
    output = ("CORRECTNESS: PASS\nGEAK_GEOMEAN_SPEEDUP 1.25e-6" if contract == "text"
              else json.dumps({"exit": 0, "weighted": score}) if contract == "json" else "")
    fields = measure(tmp_path, result=result, output=output)
    assert fields["STATUS"] == "1"
    assert float(fields["SPEEDUP"]) == score


@pytest.mark.parametrize("score", ["nan", "inf", "1e309", "-1", "0", "1.2.3", "1.25suffix"])
@pytest.mark.parametrize("contract", ["text", "json"])
def test_invalid_text_or_scalar_json_score_is_no_measurement(tmp_path, score, contract):
    output = ("CORRECTNESS: PASS\nGEAK_GEOMEAN_SPEEDUP " + score if contract == "text"
              else json.dumps({"exit": 0, "weighted": score}))
    fields = measure(tmp_path, output=output)
    assert fields["STATUS"] == "0"
    assert fields["SPEEDUP"] == ""


def test_interval_contract_keeps_the_lower_endpoint(tmp_path):
    result = receipt()
    result["gates"]["timing"] = {
        "ok": True, "serving_weighted_speedup_interval": [1.4, 1.1],
        "rows": [{"sig": "captured", "reps": 1}],
    }
    fields = measure(tmp_path, result=result)
    assert float(fields["SPEEDUP"]) == 1.1
    assert fields["CLEAN"] == ""
    assert fields["SINGLE"] == "1"


def test_existing_timing_contract_takes_precedence_over_top_level_score(tmp_path):
    result = receipt()
    result["gates"]["timing"] = {"ok": True, "serving_weighted_speedup_interval": [1.4, 1.1]}
    result["serving_weighted_speedup"] = {"primary": 2.0}
    fields = measure(tmp_path, result=result)
    assert float(fields["SPEEDUP"]) == 1.1


def test_bare_gate_map_contract(tmp_path):
    fields = measure(tmp_path, result=receipt(1.25), output='UT PASS\n{"correctness": true}')
    assert fields["STATUS"] == "1"
    assert float(fields["SPEEDUP"]) == 1.25


def test_empty_primary_field_does_not_shift_a_secondary_statistic(tmp_path):
    result = receipt()
    result["gates"]["timing"]["aggregate"] = {
        "per_case": [{"sig": "a", "included": True, "weight": 3, "speedup": 1.4}]}
    result["gates"]["timing"]["per_case"] = [{"sig": "a", "reps": 2}]
    fields = measure(tmp_path, result=result)
    assert fields["STATUS"] == "0"
    assert fields["SPEEDUP"] == ""
    assert float(fields["CLEAN"]) == pytest.approx(1.4)
    assert fields["SINGLE"] == "0"
