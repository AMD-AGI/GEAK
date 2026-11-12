"""fusion_overlay_check.py: the accepted KernelFusion overlay keeps the overlay it was built on and is the one its A/B measured."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPTS_DIR / "fusion_overlay_check.py"
_spec = importlib.util.spec_from_file_location("fusion_overlay_check", SCRIPT)
foc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(foc)


def make_overlay(run_dir, initial=None, site="import _kf_c1\n", extra=None):
    """An accepted overlay as the integrator leaves it: sources + the integrate_result.json of its A/B."""
    ov = Path(run_dir) / "overlay" / "cand_kf_c1"
    ov.mkdir(parents=True, exist_ok=True)
    if initial:
        for rel, text in initial.items():
            (ov / rel).parent.mkdir(parents=True, exist_ok=True)
            (ov / rel).write_text(text)
    (ov / "_kf_c1.py").write_text("X = 1\n")
    (ov / "sitecustomize.py").write_text(site)
    for rel, text in (extra or {}).items():
        (ov / rel).write_text(text)
    (ov / "cand").mkdir(exist_ok=True)
    (ov / "cand" / "bench_summary.json").write_text("{}")
    (Path(run_dir) / "integrate_kf_c1").mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "integrate_kf_c1" / "integrate_result.json").write_text(json.dumps(
        {"ab_complete": True, "accepted_overlay": str(ov),
         "measurement_evidence": {"cand_summary": str(ov / "cand" / "bench_summary.json")}}))
    return str(ov)


def _initial(tmp_path, files):
    d = tmp_path / "initial_overlay"
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return str(d)


def test_validate_overlay_accepts_an_overlay_that_keeps_the_initial_overlay(tmp_path):
    run_dir = tmp_path / "e2e_x"
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n", "pkg/a.py": "A = 1\n"})
    ov = make_overlay(run_dir, initial={"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n",
                                        "pkg/a.py": "A = 1\n"},
                      site="import _initial_sitecustomize\nimport _kf_c1\n")
    check = foc.validate_overlay(ov, init, run_dir)
    assert check["ok"], check["problems"]
    assert len(check["manifest_sha256"]) == 64


def test_validate_overlay_rejects_what_would_lose_or_break_an_optimization(tmp_path):
    init_files = {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    kept = {"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"}
    good_site = "import _initial_sitecustomize\nimport _kf_c1\n"
    cases = {
        "missing": ({"initial": {"_initial_sitecustomize.py": "import tuned\n"}, "site": good_site}, "tuned.py"),
        "changed": ({"initial": {**kept, "tuned.py": "T = 3\n"}, "site": good_site}, "was changed"),
        "not imported": ({"initial": kept, "site": "import _kf_c1\n"}, "does not import it"),
        "dropped site": ({"initial": {"tuned.py": "T = 2\n"}, "site": good_site}, "is not kept"),
        "syntax": ({"initial": kept, "site": good_site, "extra": {"_kf_bad.py": "def (:\n"}}, "does not compile"),
        "dangling kernel": ({"initial": kept, "site": good_site,
                             "extra": {"_kf_k.py": "K = '{kf}/kernels/c1/kernel'\n"}}, "does not exist"),
    }
    for i, (name, (kw, needle)) in enumerate(cases.items()):
        root = tmp_path / str(i)
        run_dir = root / "e2e_x"
        if "extra" in kw:
            kw["extra"] = {k: v.replace("{kf}", str(run_dir)) for k, v in kw["extra"].items()}
        ov = make_overlay(run_dir, **kw)
        check = foc.validate_overlay(ov, _initial(root, init_files), run_dir)
        assert not check["ok"] and any(needle in p for p in check["problems"]), (name, check["problems"])


def test_validate_overlay_ties_the_overlay_to_the_ab_that_accepted_it(tmp_path):
    run_dir = tmp_path / "e2e_x"
    ov = make_overlay(run_dir)
    assert foc.validate_overlay(ov, "", run_dir)["ok"]
    first = foc.validate_overlay(ov, "", run_dir)["manifest_sha256"]
    # the integrator's A/B records land in the candidate dir after the A/B (DSv4.1-Flash c64: 4 min later):
    # they are records, not overlay content -- neither a change nor part of the manifest
    later = os.path.getmtime(Path(ov) / "cand" / "bench_summary.json") + 240
    for rec in ("e2e_validation.json", "manifests.json"):
        (Path(ov) / rec).write_text("{}")
        os.utime(Path(ov) / rec, (later, later))
    check = foc.validate_overlay(ov, "", run_dir)
    assert check["ok"], check["problems"]
    assert check["manifest_sha256"] == first
    os.utime(Path(ov) / "_kf_c1.py", (later, later))                       # code edited after the A/B
    assert any("after the A/B" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])
    (run_dir / "integrate_kf_c1" / "integrate_result.json").unlink()
    assert any("no integrate_result.json" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])
    elsewhere = make_overlay(tmp_path / "other_run")
    assert any("outside the run dir" in p for p in foc.validate_overlay(elsewhere, "", run_dir)["problems"])


def test_validate_overlay_edge_cases(tmp_path):
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"})
    # the overlay loads its kernels without touching sitecustomize.py; binaries are not compiled as Python;
    # an initial overlay that is gone has nothing left to keep; a corrupt record elsewhere is ignored
    run_dir = tmp_path / "e2e_a"
    ov = make_overlay(run_dir, initial={"tuned.py": "T = 2\n"}, site="import tuned\n",
                      extra={"_kf_c1.hsaco": "\x7fELF"})
    (run_dir / "integrate_aborted").mkdir()                                  # read before integrate_kf_c1
    (run_dir / "integrate_aborted" / "integrate_result.json").write_text("{not json")
    check = foc.validate_overlay(ov, os.pathsep.join([init, str(tmp_path / "gone")]), run_dir)
    assert check["ok"], check["problems"]
    assert check["files"] == 4                                                # tuned, sitecustomize, _kf_c1 .py + .hsaco

    run_dir = tmp_path / "e2e_b"
    ov = make_overlay(run_dir)
    (Path(ov) / "sitecustomize.py").unlink()
    assert "no sitecustomize.py at the overlay root" in foc.validate_overlay(ov, "", run_dir)["problems"]

    run_dir = tmp_path / "e2e_c"
    ov = make_overlay(run_dir)
    rec = run_dir / "integrate_kf_c1" / "integrate_result.json"
    rec.write_text(json.dumps({**json.loads(rec.read_text()), "ab_complete": False}))
    assert any("the A/B is not complete" in p for p in foc.validate_overlay(ov, "", run_dir)["problems"])


def _kept_overlay(tmp_path):
    run_dir = tmp_path / "e2e_x"
    init = _initial(tmp_path, {"sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"})
    ov = make_overlay(run_dir, initial={"_initial_sitecustomize.py": "import tuned\n", "tuned.py": "T = 2\n"},
                      site="import _initial_sitecustomize\nimport _kf_c1\n")
    return run_dir, init, ov


def test_main_prints_one_json_object_and_exits_1_when_the_check_fails(tmp_path, capsys, monkeypatch):
    run_dir, init, ov = _kept_overlay(tmp_path)

    def cli(*argv):
        rc = foc.main(list(argv))
        return rc, json.loads(capsys.readouterr().out)

    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert rc == 0 and out["ok"] and out["problems"] == []
    (Path(ov) / "tuned.py").write_text("T = 3\n")
    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert rc == 1 and not out["ok"] and any("was changed" in p for p in out["problems"])
    rc, out = cli("--overlay", str(tmp_path / "nope"), "--root", str(run_dir))
    assert rc == 1 and not out["ok"] and out["problems"]

    def broken(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(foc, "validate_overlay", broken)                      # a check that cannot run never accepts
    rc, out = cli("--overlay", ov, "--initial", init, "--root", str(run_dir))
    assert (rc, out) == (1, {"ok": False, "problems": ["check failed: OSError: disk gone"]})


def test_the_script_runs_as_the_agent_runs_it(tmp_path):
    run_dir, init, ov = _kept_overlay(tmp_path)
    p = subprocess.run([sys.executable, str(SCRIPT), "--overlay", ov, "--initial", init, "--root", str(run_dir)],
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0 and json.loads(p.stdout)["ok"], p.stderr
