"""Golden test of archive.cells (drift test 8: reduce_cells/select_cell and helpers).

GOLDEN holds the SHA-256 of the canonical JSON output (or error type and text)
of the ORIGINAL pre-port functions (oct15 bridge/native_cells.py) on the
synthetic inputs below. The port must give byte-identical outputs.

Run from the repo root:  python3 -B tests/geak_evo/test_cells_golden.py
"""

import copy
import hashlib
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

SPEC = {"schema": "native-categorical-cells-v1", "descriptor_id": "layout", "revision": 1,
        "axes": [{"name": "tiling", "meaning": "tile shape of the main loop", "values": ["small", "large"]},
                 {"name": "memory", "meaning": "operand staging path", "values": ["lds", "direct"]}]}
CASES_IDS = ["case-a", "case-b", "case-c"]
COMPARISON = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1", "benchmark_case_ids": CASES_IDS,
              "statistic": "median-of-complete-trial-equal-case-geomeans-ms-v1"}
REQUEST = {"contract": {"timing_policy": {"warmup": 5, "iterations": 50}, "task": "gemm"},
           "benchmark_case_ids": CASES_IDS,
           "benchmark_report_ids": {"case-a": "r1", "case-b": "r2", "case-c": "r3"},
           "correctness_case_ids": ["c1", "c2"], "phases": ["correctness", "benchmark"],
           "evaluator_sha256": "2" * 64, "binding": {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}}
HOST = {"runtime": {"image": "sha256:" + "3" * 64, "config_digest": "4" * 64,
                    "physical_devices": ["GPU-0001"], "extra": "ignored"}}


def _spec(**changes):
    spec = copy.deepcopy(SPEC)
    spec.update(changes)
    return spec


def _axes(*axes):
    return _spec(axes=list(axes))


def _assign(status="supported", refs=("ref-1",), **descriptors):
    return {"status": status, "descriptors": descriptors or {"tiling": "large", "memory": "lds"},
            "evidence_refs": list(refs)}


def _obs(operation, times, source="src-1", comparison=COMPARISON):
    return {"operation_id": operation, "source_id": source, "comparison": comparison,
            "per_case_times_ms": dict(zip(CASES_IDS, times))}


def _request(**changes):
    request = copy.deepcopy(REQUEST)
    request.update(changes)
    return request


def _host(**changes):
    host = copy.deepcopy(HOST)
    host["runtime"].update(changes)
    return host


def _member(cell, source, candidate, value, direction="d1", island="island-1", membership="m1"):
    return {"cell_id": cell, "source_id": source, "candidate_id": candidate, "direction_id": direction,
            "island_id": island, "membership_ref": {"path": "memberships/" + membership, "sha256": "5" * 64},
            "retained_roles": ["parent"], "charter_ref": {"path": "charters/" + cell, "sha256": "6" * 64},
            "descriptor_spec": SPEC, "coordinates": ["large", "lds"], "comparison": COMPARISON,
            "quality": {"value_ms": value, "statistic": COMPARISON["statistic"]}}


def _members():
    return [_member("cell-b", "src-3", "cand-3", 2.0),
            _member("cell-a", "src-2", "cand-2", 1.5, direction="d2"),
            _member("cell-a", "src-1", "cand-1", 1.5),
            _member("cell-a", "src-1", "cand-1b", 1.5, direction="d3", membership="m2"),
            _member("cell-a", "src-1", "cand-1b", 1.5, direction="d3", membership="m2"),
            _member("cell-a", "src-4", "cand-4", 1.2, island="island-2")]


def _view(api):
    return {"cells": api["reduce_cells"](_members())}


CASES = {
    "spec_ok": lambda api: api["validate_descriptor_spec"](SPEC),
    "spec_bad_schema": lambda api: api["validate_descriptor_spec"](_spec(schema="other")),
    "spec_bad_revision": lambda api: api["validate_descriptor_spec"](_spec(revision=0)),
    "spec_extra_key": lambda api: api["validate_descriptor_spec"](_spec(note="x")),
    "spec_no_axes": lambda api: api["validate_descriptor_spec"](_axes()),
    "spec_axis_no_meaning": lambda api: api["validate_descriptor_spec"](
        _axes({"name": "tiling", "meaning": " ", "values": ["a"]})),
    "spec_duplicate_values": lambda api: api["validate_descriptor_spec"](
        _axes({"name": "tiling", "meaning": "m", "values": ["a", "a"]})),
    "spec_duplicate_axis": lambda api: api["validate_descriptor_spec"](
        _axes({"name": "tiling", "meaning": "m", "values": ["a"]},
              {"name": "tiling", "meaning": "n", "values": ["b"]})),
    "coords_ok": lambda api: api["descriptor_coordinates"](SPEC, _assign()),
    "coords_unsupported": lambda api: api["descriptor_coordinates"](SPEC, _assign(status="unsupported")),
    "coords_missing_axis": lambda api: api["descriptor_coordinates"](SPEC, _assign(tiling="large")),
    "coords_unknown_value": lambda api: api["descriptor_coordinates"](
        SPEC, _assign(tiling="huge", memory="lds")),
    "coords_no_evidence": lambda api: api["descriptor_coordinates"](SPEC, _assign(refs=())),
    "coords_bad_spec": lambda api: api["descriptor_coordinates"](_spec(revision=-1), _assign()),
    "context_ok": lambda api: api["_quality_context"](REQUEST, HOST),
    "context_no_runtime": lambda api: api["_quality_context"](REQUEST, {}),
    "context_no_image": lambda api: api["_quality_context"](REQUEST, _host(image="")),
    "context_bad_devices": lambda api: api["_quality_context"](REQUEST, _host(physical_devices=["", "x"])),
    "context_no_evaluator": lambda api: api["_quality_context"](_request(evaluator_sha256=None), HOST),
    "context_no_timing_policy": lambda api: api["_quality_context"](_request(contract={"task": "gemm"}), HOST),
    "context_mapping_mismatch": lambda api: api["_quality_context"](
        _request(benchmark_report_ids={"case-a": "r1", "case-b": "r2"}), HOST),
    "context_mapping_ambiguous": lambda api: api["_quality_context"](
        _request(benchmark_report_ids={"case-a": "r1", "case-b": "r1", "case-c": "r3"}), HOST),
    "quality_ok": lambda api: api["development_quality"](
        [_obs("op-2", [1.0, 2.0, 4.0]), _obs("op-1", [1.5, 1.5, 1.5]), _obs("op-3", [3.0, 1.0, 0.5]),
         _obs("op-2", [1.0, 2.0, 4.0])]),
    "quality_even_count": lambda api: api["development_quality"](
        [_obs("op-1", [1.0, 1.0, 1.0]), _obs("op-2", [2.0, 2.0, 2.0])]),
    "quality_empty": lambda api: api["development_quality"]([]),
    "quality_other_source": lambda api: api["development_quality"](
        [_obs("op-1", [1.0, 1.0, 1.0]), _obs("op-2", [1.0, 1.0, 1.0], source="src-2")]),
    "quality_conflicting_duplicate": lambda api: api["development_quality"](
        [_obs("op-1", [1.0, 1.0, 1.0]), _obs("op-1", [1.0, 1.0, 2.0])]),
    "quality_missing_case": lambda api: api["development_quality"]([_obs("op-1", [1.0, 1.0])]),
    "quality_nonpositive": lambda api: api["development_quality"]([_obs("op-1", [1.0, 0.0, 1.0])]),
    "reduce_two_cells": lambda api: api["reduce_cells"](_members()),
    "reduce_tie_by_source": lambda api: api["reduce_cells"](_members()[:5]),
    "reduce_empty": lambda api: api["reduce_cells"]([]),
    "reduce_conflicting_quality": lambda api: api["reduce_cells"](
        [_member("cell-a", "src-1", "cand-1", 1.5), _member("cell-a", "src-1", "cand-1b", 1.7)]),
    "select_ok": lambda api: api["select_cell"](_view(api), "cell-a", "src-4"),
    "select_other_cell": lambda api: api["select_cell"](_view(api), "cell-b", "src-3"),
    "select_elite_changed": lambda api: api["select_cell"](_view(api), "cell-a", "src-1"),
    "select_absent": lambda api: api["select_cell"](_view(api), "cell-z", "src-1"),
    "select_ambiguous": lambda api: api["select_cell"](
        {"cells": _view(api)["cells"] * 2}, "cell-a", "src-4"),
    "bind_none": lambda api: api["bind_engineer_identity"](None, {}, None),
    "bind_kind": lambda api: api["bind_engineer_identity"](None, {}, {"kind": "stop", "reason": "done"}),
    "bind_unassigned": lambda api: api["bind_engineer_identity"](
        None, {}, {"direction_id": "d1", "question": "q", "repeat_delta": "more unroll"}),
    "bind_explicit_island": lambda api: api["bind_engineer_identity"](
        None, {}, {"direction_id": "d1", "island_id": "island-2", "membership_ref": None}),
}

GOLDEN = {
    "bind_explicit_island": "4ec61dd939f7b5b2431613ec15f37f306f311fad95f664a4baba2d5a9f2b398c",
    "bind_kind": "de5f6c4cde54788922fe6a0e25aa7103a9cd1510ef140dd638fda4b24737f6ec",
    "bind_none": "78fa2886f5e9367182124d9465d8ad8827856f6bce3db3b362e0e90257cf6733",
    "bind_unassigned": "c419628c97d0f300685687aeea5b45d198c1fab7e41fb35522362c39802ccb93",
    "context_bad_devices": "caa91f43b62449e0ceb7ac10f3e235bf6fd1690e56e69a7306b6ebb03f599691",
    "context_mapping_ambiguous": "75cb0a539dd3eebecad0d1e5c836014456851d44511c00ca1ac605f3c0597113",
    "context_mapping_mismatch": "75cb0a539dd3eebecad0d1e5c836014456851d44511c00ca1ac605f3c0597113",
    "context_no_evaluator": "8c32a7ba081311a23ef1ac6f460b01756a5efda254da27b7a2e3066a158eccb7",
    "context_no_image": "96ea786abadab83409d31860e7bec9ba6597e176dc3c0558bdb83b490d348c95",
    "context_no_runtime": "96ea786abadab83409d31860e7bec9ba6597e176dc3c0558bdb83b490d348c95",
    "context_no_timing_policy": "8c32a7ba081311a23ef1ac6f460b01756a5efda254da27b7a2e3066a158eccb7",
    "context_ok": "915bacd846cb9a66d5821a1290af6828f9537df2bad2cbeb5a539327bc29063d",
    "coords_bad_spec": "2f0f21f470ecfe8aa7930b569c489be4ad1750236d3931f1c3131103b7b9f1ca",
    "coords_missing_axis": "164fcc7580e1620d75e29bfa06e2a7930ab652811c45bfe0127d58ab5e248726",
    "coords_no_evidence": "e43b134a659ce68e81c5bcb2f6dfefc533016f406c6c4b562d3eeb9d32fdf132",
    "coords_ok": "14c054b24e4a1f436493da4869b0640f0f38cd7ff2ce0436aaebe78b6f8f7a21",
    "coords_unknown_value": "a39afb6dc67b819478b9276ee09e032ed58bf2531967e180d9a523d07f229eec",
    "coords_unsupported": "3a70085a099c148bbab2bb50156974d4a1810adc9fdd46335468a0af9784d763",
    "quality_conflicting_duplicate": "737801fea4f3061d50f4115e4257c08cb7d4d4b54131220f6fa5d386934901e8",
    "quality_empty": "03590405bffd4ad9ab1b05028b53eabeabfee0526839272b3f4066e3f31be672",
    "quality_even_count": "32dd66f46b1b373cbdfcc35722ff711c1ec5c1e36e1097ee503c08998f5d71a4",
    "quality_missing_case": "8b5bcb93980f447cc4fce9779b2b8581e024e6cef7110775c70d290ccf18b20e",
    "quality_nonpositive": "8b5bcb93980f447cc4fce9779b2b8581e024e6cef7110775c70d290ccf18b20e",
    "quality_ok": "24e78078fd2d8942b2028d208da8b52728a7e8ad1caada9513eee7b7e107f559",
    "quality_other_source": "44011de9b4383c84393b234d843db51170924101aeca2ed73dcbfab5f167a67c",
    "reduce_conflicting_quality": "3ba1644700c22d6e3ccac9af6083842f6e90563450afe060f91553d63045d804",
    "reduce_empty": "79f889f1b8ec725e8664679b972e031a687ac0b0ef9c489f183be0a8b83e39d4",
    "reduce_tie_by_source": "ea0e3c4dd3646f63dcb204b4a663b9bd8f6e5fb36ee87c54bfac99d8d7c0eebd",
    "reduce_two_cells": "06a2adb847030e8af95a5d814d513c4eac2bb9caf77140e5c3368e2a41d98980",
    "select_absent": "c945b3f27316578d29118f08ee0336c41ed76ddc7cb38c78260840197bb49440",
    "select_ambiguous": "c945b3f27316578d29118f08ee0336c41ed76ddc7cb38c78260840197bb49440",
    "select_elite_changed": "92b5e469699f93ba723a98a11d8d6db27b4bdb06aa07cac4b6d7da901224e451",
    "select_ok": "ebe470795d2245809daf23a77377619454a9eb6eb62744c645f74d56346cac4d",
    "select_other_cell": "4995069b70cdc1a272d6341b2a4977ad6160aa55219c892afddd6c12d66e1fbd",
    "spec_axis_no_meaning": "754a29f730c8b2d4c4a24647e048449abf159578ffd75966aa25ccffbe00b251",
    "spec_bad_revision": "2f0f21f470ecfe8aa7930b569c489be4ad1750236d3931f1c3131103b7b9f1ca",
    "spec_bad_schema": "2f0f21f470ecfe8aa7930b569c489be4ad1750236d3931f1c3131103b7b9f1ca",
    "spec_duplicate_axis": "4449702fc8f5d88f1328fa2b63c2d2e8347618a56ad8a3057cd4467d584d9b85",
    "spec_duplicate_values": "98970c87a95b445f10372c11c91da73b34ee87c8941e202152b9cf4944aa6ed7",
    "spec_extra_key": "2f0f21f470ecfe8aa7930b569c489be4ad1750236d3931f1c3131103b7b9f1ca",
    "spec_no_axes": "a745ec5dd7833e06e441d63aa3ea1a0f00e8c1504b678afdaf34bf52d021a28b",
    "spec_ok": "27095948c2cd6a4c268ef576d8f9e2b16bb96fda8c3a1de2807be49b612584b7",
}


def outputs(api):
    """Map case name -> sha256 of canonical JSON of its output or error."""
    result = {}
    for name, case in CASES.items():
        try:
            value = {"ok": case(api)}
        except api["CellError"] as error:
            value = {"error": type(error).__name__, "message": str(error)}
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result[name] = hashlib.sha256(data).hexdigest()
    return result


NAMES = ("validate_descriptor_spec", "descriptor_coordinates", "_quality_context", "development_quality",
         "reduce_cells", "select_cell", "bind_engineer_identity", "CellError")


def port_api():
    from geak_evo.archive import cells
    return {name: getattr(cells, name) for name in NAMES}


class CellsGoldenTest(unittest.TestCase):
    def test_outputs_match_original(self):
        actual = outputs(port_api())
        self.assertEqual(set(actual), set(GOLDEN))
        for name in sorted(GOLDEN):
            with self.subTest(case=name):
                self.assertEqual(actual[name], GOLDEN[name])

    def test_error_classes(self):
        from geak_evo._util import EvoError
        from geak_evo.archive import cells
        self.assertTrue(issubclass(cells.CellError, EvoError))
        self.assertTrue(issubclass(cells.CellError, ValueError))
        self.assertTrue(issubclass(cells.CellProposalError, cells.CellError))


if __name__ == "__main__":
    unittest.main()
