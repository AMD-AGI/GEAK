"""CPU-only tests for the paired CUDA-graph A/B tool (mega_graph_ab.py).

The torchrun leg is replaced by ``--fake-cmd``; no torch or GPU is required.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "mega_graph_ab.py")


def _load():
    spec = importlib.util.spec_from_file_location("mega_graph_ab", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(tmp_path, body):
    script = tmp_path / "fake_leg.py"
    script.write_text(textwrap.dedent(body))
    return f"{sys.executable} {script}"


def _run(tmp_path, fake, *extra):
    out = tmp_path / "ab.json"
    base, cand = tmp_path / "base_tree", tmp_path / "cand_tree"
    base.mkdir(exist_ok=True)
    cand.mkdir(exist_ok=True)
    cmd = [sys.executable, TOOL, "--arm", f"base={base}", "--arm", f"cand={cand}#MY_SWITCH=1",
           "--passes", "3", "--retry-sleep", "0", "--work-dir", str(tmp_path / "work"),
           "--jit-root", str(tmp_path / "jit"), "--json-output", str(out),
           "--fake-cmd", fake, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, json.loads(out.read_text())


def test_parse_arm_and_rotation():
    m = _load()
    arm = m.parse_arm("cand=/x/tree#A=1#B=two,three")
    assert arm == {"name": "cand", "tree": "/x/tree", "env": {"A": "1", "B": "two,three"}}
    assert m.parse_arm("base=/t")["env"] == {}
    arms = [{"name": n} for n in "abc"]
    assert [a["name"] for a in m.rotation(arms, 1)] == ["b", "c", "a"]
    assert [a["name"] for a in m.rotation(arms, 3)] == ["a", "b", "c"]


def test_paired_deltas_and_env(tmp_path):
    fake = _fake(tmp_path, """
        import json, os
        ms = 0.300 if os.environ.get("MY_SWITCH") == "1" else 0.305
        assert os.environ["FLYDSL_RUNTIME_CACHE_DIR"].endswith(os.environ["ARM_NAME"])
        assert os.environ["PYTHONPATH"].startswith(os.environ["ARM_TREE"])
        print("[megamoe] path=MEGA")
        print("MEGA_AB_RESULT " + json.dumps({"rounds_ms": [ms], "median_ms": ms, "finite": True}))
    """)
    proc, doc = _run(tmp_path, fake, "--frozen-baseline-ms", "0.3458")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    s = doc["summary"]["cand"]
    assert s["n_pairs"] == 3 and s["n_faster"] == 3
    assert abs(s["median_delta_us"] + 5.0) < 1e-6
    assert abs(s["speedup_vs_frozen"] - round(0.3458 / 0.300, 4)) < 1e-9
    assert {leg["arm"] for leg in doc["legs"]} == {"base", "cand"}
    # rotation: pass 1 starts with cand
    assert [leg["arm"] for leg in doc["legs"] if leg["pass"] == 1][0] == "cand"
    assert doc["legs"][0]["path_markers"] == {"MEGA": 1}


def test_failed_leg_is_retried_then_dropped(tmp_path):
    fake = _fake(tmp_path, """
        import json, os, sys
        if os.environ["ARM_NAME"] == "cand":
            sys.exit(3)
        print("MEGA_AB_RESULT " + json.dumps({"rounds_ms": [0.3], "median_ms": 0.3, "finite": True}))
    """)
    proc, doc = _run(tmp_path, fake, "--attempts", "2")
    assert proc.returncode == 1
    assert len(doc["dropped_legs"]) == 3
    assert all(len(d["attempts"]) == 2 for d in doc["dropped_legs"])
    assert doc["summary"]["cand"]["n_pairs"] == 0


def test_expect_path_voids_fallback_leg(tmp_path):
    fake = _fake(tmp_path, """
        import json
        print("[megamoe] path=FALLBACK")
        print("MEGA_AB_RESULT " + json.dumps({"rounds_ms": [0.3], "median_ms": 0.3, "finite": True}))
    """)
    proc, doc = _run(tmp_path, fake, "--attempts", "1", "--expect-path", "cand=MEGA")
    assert proc.returncode == 1
    assert {d["arm"] for d in doc["dropped_legs"]} == {"cand"}
    assert "path markers" in doc["dropped_legs"][0]["attempts"][0]["error"]


def test_jit_root_inside_tree_is_rejected(tmp_path):
    base = tmp_path / "base_tree"
    base.mkdir()
    proc = subprocess.run([sys.executable, TOOL, "--arm", f"base={base}", "--arm", f"cand={base}",
                           "--jit-root", str(base / "jit"), "--fake-cmd", "true"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "outside every tree" in proc.stderr
