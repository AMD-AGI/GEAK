"""Hermetic tests for the upstream-knob enumerator.

The failure modes worth testing are all forms of MISREPORTING, because the output is read as a list
of things to try and a wrong entry costs a round:

  * missing the pin that closed the axis (the whole point),
  * calling a consistency check (`assert BN == BK`) a pin,
  * reporting a parameter as narrowed when the candidate takes it under `**kwargs`,
  * pairing a function against a same-named coincidence in an unrelated subsystem and showing only
    that one,
  * needing the candidate to be importable -- these modules do not import without a GPU.

The last group runs against the real GLM-5.2 seam, because the ranking is only worth anything on a
file with 4866 upstream functions to collide with, and a fixture cannot catch that.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("upstream_knobs", HERE / "upstream_knobs.py")
UK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UK)

SEAM = Path("/sgl-workspace/ut_glm52_conc10_0917/glm52_mxfp4_fused_moe_task/baseline_src/"
            "fused_moe_seam.py")
AITER = Path("/sgl-workspace/geak_flydsl/aiter_main/aiter/aiter")


def run(tmp_path, candidate_src, upstream_src, up_name="up.py"):
    cand = tmp_path / "cand.py"
    cand.write_text(candidate_src)
    up_dir = tmp_path / "upstream"
    up_dir.mkdir(exist_ok=True)
    (up_dir / up_name).write_text(upstream_src)
    index, _ = UK.index_upstream([up_dir])
    findings, errors = UK.compare(cand, index)
    assert errors == []
    return findings


def test_reports_a_parameter_upstream_has_and_the_candidate_does_not(tmp_path):
    f = run(tmp_path,
            "def k(BM=32, *, BN=256):\n    return BM\n",
            "def k(BM=32, *, BN=256, num_waves=4, k_wave=1):\n    return BM\n")
    assert len(f) == 1
    assert [n["param"] for n in f[0]["narrowed"]] == ["num_waves", "k_wave"]


def test_reports_a_parameter_the_candidate_pins_to_a_constant(tmp_path):
    f = run(tmp_path,
            "def k(*, BN=256, BK=256):\n"
            "    assert BN == 256 and BK == 256, 'only 256'\n",
            "def k(*, BN=256, BK=256):\n    return BN\n")
    assert len(f) == 1
    pinned = {p["param"]: p for p in f[0]["pinned"]}
    assert pinned["BN"]["to"] == 256 and pinned["BN"]["upstream_has"] is True
    assert pinned["BK"]["to"] == 256


def test_a_raise_guard_pins_just_like_an_assert(tmp_path):
    f = run(tmp_path,
            "def k(*, BN=256):\n"
            "    if BN != 256:\n        raise AssertionError('no')\n",
            "def k(*, BN=256):\n    return BN\n")
    assert [p["param"] for p in f[0]["pinned"]] == ["BN"]


def test_membership_pins_to_the_allowed_set(tmp_path):
    f = run(tmp_path,
            "def k(*, row_bytes=128):\n    assert row_bytes in (64, 128)\n",
            "def k(*, row_bytes=128, extra=1):\n    return row_bytes\n")
    assert f[0]["pinned"][0]["to"] == (64, 128)


def test_comparing_two_parameters_is_not_a_pin(tmp_path):
    """`assert BN == BK` constrains their relationship; it closes neither axis."""
    f = run(tmp_path,
            "def k(*, BN=256, BK=256, z=0):\n    assert BN == BK\n",
            "def k(*, BN=256, BK=256, z=0, num_waves=4):\n    return BN\n")
    assert f[0]["pinned"] == []
    assert [n["param"] for n in f[0]["narrowed"]] == ["num_waves"]


def test_kwargs_does_not_absorb_upstream_parameters(tmp_path):
    """A candidate taking `**kw` still does not EXPOSE the knob -- a caller cannot discover it.

    Reported as narrowed on purpose: the question is what the author can turn, and a parameter that
    only exists if you already know its name is not a knob the next round will find.
    """
    f = run(tmp_path,
            "def k(BM=32, **kw):\n    return BM\n",
            "def k(BM=32, num_waves=4):\n    return BM\n")
    assert [n["param"] for n in f[0]["narrowed"]] == ["num_waves"]


def test_identical_signatures_produce_no_finding(tmp_path):
    assert run(tmp_path,
               "def k(a, b=1):\n    return a\n",
               "def k(a, b=1):\n    return a\n") == []


def test_a_function_with_no_upstream_counterpart_is_silent(tmp_path):
    assert run(tmp_path,
               "def only_here(a, b=1):\n    return a\n",
               "def something_else(a):\n    return a\n") == []


def test_a_vendoring_suffix_matches_fuzzily_and_says_so(tmp_path):
    f = run(tmp_path,
            "def gemm1_fold(a):\n    return a\n",
            "def gemm1(a, num_waves=4):\n    return a\n")
    assert f[0]["match"] == "fuzzy"
    assert f[0]["upstream_function"] == "gemm1"


def test_self_is_not_counted_as_a_narrowed_parameter(tmp_path):
    assert run(tmp_path,
               "def m(self, a):\n    return a\n",
               "def m(self, a):\n    return a\n") == []


def test_a_syntactically_broken_candidate_reports_rather_than_raises(tmp_path):
    cand = tmp_path / "bad.py"
    cand.write_text("def k(:\n")
    findings, errors = UK.compare(cand, {})
    assert findings == [] and len(errors) == 1 and "SyntaxError" in errors[0]


def test_every_same_named_upstream_is_listed_not_just_the_chosen_one(tmp_path):
    """Two upstream files define the name. Showing one and hiding the other is the failure."""
    cand = tmp_path / "cand.py"
    cand.write_text("from pkg.dispatcher import helper\n\ndef k(a):\n    return a\n")
    up = tmp_path / "upstream" / "pkg"
    up.mkdir(parents=True)
    (up / "__init__.py").write_text("")
    (up / "dispatcher.py").write_text("def helper():\n    pass\n\ndef k(a, g2_spart=0):\n    return a\n")
    (up / "kernel.py").write_text("def k(a, num_waves=4, k_wave=1):\n    return a\n")
    index, _ = UK.index_upstream([tmp_path / "upstream"])
    findings, _ = UK.compare(cand, index)
    assert len(findings) == 1
    seen = {findings[0]["upstream_module"]} | {o["module"] for o in findings[0]["other_counterparts"]}
    assert seen == {"pkg.dispatcher", "pkg.kernel"}
    # The module the candidate actually imports from leads.
    assert findings[0]["upstream_module"] == "pkg.dispatcher"


def test_runs_without_importing_the_candidate(tmp_path):
    """The candidate imports a module that does not exist. AST-only, so this must still work."""
    f = run(tmp_path,
            "import flydsl_that_is_not_installed as fx\n\ndef k(a):\n    return fx.go(a)\n",
            "def k(a, num_waves=4):\n    return a\n")
    assert [n["param"] for n in f[0]["narrowed"]] == ["num_waves"]


# --- against the real tree -----------------------------------------------------------------------

needs_box = pytest.mark.skipif(not (SEAM.is_file() and AITER.is_dir()),
                               reason="GLM-5.2 seam / aiter_main not on this box")


@needs_box
def test_finds_the_bn_pin_that_cost_the_glm52_campaign_seventeen_points():
    out = subprocess.run(
        [sys.executable, str(HERE / "upstream_knobs.py"), str(SEAM),
         "--upstream", str(AITER), "--json"],
        capture_output=True, text=True, check=True, timeout=900)
    data = json.loads(out.stdout)
    g1 = [f for f in data["findings"] if f["function"] == "compile_gemm1_a4w4_port"]
    assert g1, "the seam's gemm1 entry point must be paired with its upstream"
    pinned = {p["param"]: p for p in g1[0]["pinned"]}
    assert pinned["BN"]["to"] == 256 and pinned["BN"]["upstream_has"] is True
    narrowed = {n["param"] for n in g1[0]["narrowed"]}
    assert {"num_waves", "k_wave"} <= narrowed


@needs_box
def test_the_two_moe_gemms_rank_above_the_noise():
    """Ranking is the difference between a usable report and 31 findings to skim.

    Both GLM-5.2 MoE GEMM entry points carry a pin, and a pin outranks a bare absence, so both must
    land in the top few lines of a scan over a 6901-line file against 4866 upstream functions.
    """
    out = subprocess.run(
        [sys.executable, str(HERE / "upstream_knobs.py"), str(SEAM),
         "--upstream", str(AITER), "--json"],
        capture_output=True, text=True, check=True, timeout=900)
    data = json.loads(out.stdout)
    ranked = sorted(data["findings"], key=lambda f: (-len(f["pinned"]), -len(f["narrowed"])))
    top = [f["function"] for f in ranked[:3]]
    assert "compile_gemm1_a4w4_port" in top and "compile_gemm2_a4w4_port" in top


# --------------------------------------------------------------------------------------------
# Coverage: is the tree we were pointed at even the one this kernel was written against?
#
# This check's own failure mode is crying wolf, and it is the worse one: a warning that fires on a
# correct tree teaches the reader to ignore it, and the one time it is right it is ignored too.
# Three separate false alarms were found and fixed while writing it -- package `__init__`, PEP 420
# namespace packages, and `try/except ImportError` probes -- so all three are pinned here.
# --------------------------------------------------------------------------------------------

def test_namespace_package_counts_as_present(tmp_path):
    """PEP 420: a package directory with no `__init__.py` is still importable."""
    root = tmp_path / "up"
    (root / "pkg" / "nspkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "nspkg" / "mod.py").write_text("")  # nspkg has NO __init__.py
    assert UK.module_in_roots("pkg.nspkg", [root])
    assert UK.module_in_roots("pkg.nspkg.mod", [root])
    assert not UK.module_in_roots("pkg.absent", [root])


def test_compiled_extension_counts_as_present(tmp_path):
    """`_mlir_libs` ships as `.so`; importing one is not evidence of a wrong revision."""
    root = tmp_path / "up"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "_native.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    assert UK.module_in_roots("pkg._native", [root])


def test_guarded_import_is_not_counted_against_coverage(tmp_path):
    """An import the file itself wraps in `except ImportError` is declared optional by its author."""
    root = tmp_path / "up"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    cand = tmp_path / "cand.py"
    cand.write_text("import pkg\n"
                    "try:\n"
                    "    from pkg.optional_probe import thing\n"
                    "except ImportError:\n"
                    "    thing = None\n")
    want, missing = UK.stack_coverage([cand], [root])
    assert missing == set(), "a guarded probe must not be reported missing"
    assert "pkg" in want and "pkg.optional_probe" not in want


def test_unguarded_absent_import_is_reported(tmp_path):
    """The case the check exists for: the tree lacks what the candidate hard-imports."""
    root = tmp_path / "up"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    cand = tmp_path / "cand.py"
    cand.write_text("import pkg\nfrom pkg.kernels.mxfp4 import compile_it\nimport torch\n")
    want, missing = UK.stack_coverage([cand], [root])
    assert missing == {"pkg.kernels.mxfp4"}
    assert "torch" not in want, "a foreign dependency says nothing about this tree's revision"


@needs_box
def test_the_stale_ambient_checkout_is_called_out():
    """The measured trap: the box's ambient aiter predates the seam's whole mxfp4 family.

    Pointed at it the tool reports ~2k upstream functions and ZERO pins -- indistinguishable from a
    clean result, and exactly the false all-clear that let three rounds conclude `BN` was inert.
    Pointed at the stack the task pins it reports the pin. Coverage is what separates the two.
    """
    stale = Path("/sgl-workspace/aiter")
    if not stale.is_dir():
        pytest.skip("ambient checkout absent")
    out = subprocess.run(
        [sys.executable, str(HERE / "upstream_knobs.py"), str(SEAM),
         "--upstream", str(stale), "--json"],
        capture_output=True, text=True, check=True, timeout=900)
    data = json.loads(out.stdout)
    assert data["stack_coverage"]["missing"], "the stale tree must not look complete"
    assert "WARNING" in out.stderr and "wrong upstream tree" in out.stderr

    good = subprocess.run(
        [sys.executable, str(HERE / "upstream_knobs.py"), str(SEAM),
         "--upstream", str(AITER), "--json"],
        capture_output=True, text=True, check=True, timeout=900)
    assert not json.loads(good.stdout)["stack_coverage"]["missing"], \
        "the correct tree must be silent -- a check that cries wolf is worse than none"
