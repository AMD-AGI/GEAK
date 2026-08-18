"""The warm-start loop, end to end, with no GPU and no network.

A lane's use of the experience KB is a cycle, and every piece of it is tested elsewhere in
isolation. What is only visible when the pieces run in order is whether the cycle CLOSES:

    read a key -> pick the top candidates -> land one on a workspace that does not match the
    layout it was recorded from -> optimize further on top of it -> write the result back under
    the same key -> read it again and get the improvement, not the starting point.

The one link this cannot cover is the measurement, which needs a card. Everything on either side
of it is here, so a regression in the plumbing fails in CI rather than half an hour into a run.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(SCRIPTS, "experience_store.py")
UPLOADER = os.path.join(SCRIPTS, "kb_remote_upload.py")
yaml = pytest.importorskip("yaml")
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="the loop lands a patch with git")

KERNEL = "fused_moe_kernel"
CID = "kernel:geak:fused_moe_kernel:rocm:7.2:triton:mi355x"
BASELINE = "import triton\n\nBLOCK = 64\nNUM_WARPS = 4\n"


def run(*args):
    p = subprocess.run([sys.executable, STORE] + [str(a) for a in args],
                       capture_output=True, text=True)
    assert p.returncode == 0, f"the store must never fail the lane: {p.stderr}"
    return json.loads(p.stdout)


def git(cwd, *args):
    p = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True)
    assert p.returncode == 0, f"git {' '.join(args)}: {p.stderr}"
    return p.stdout


def recorded_entry(root, exp_id, *, speedup, direction, patch, report):
    """One curated entry, as the imported backlog looks on disk."""
    d = os.path.join(root, "gfx950", "triton", f"{KERNEL}__triton__gfx950", exp_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.yaml"), "w") as f:
        yaml.safe_dump({
            "layer": "artifact", "lifecycle": "candidate", "gfx": "gfx950",
            "kernel_class": "triton", "kernel_name": KERNEL, "language": "triton",
            "direction": direction, "reproductions": 1, "verified_stack": {"rocm": "7.2"},
            "strategy": f"{direction}: what was tried and why",
            "metric": {"speedup": speedup, "gpu_arch": "gfx950", "bench_key": "b:imported",
                       "metric_kind": "geomean"},
        }, f)
    with open(os.path.join(d, "patch.diff"), "w") as f:
        f.write(patch)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(report)
    return d


def recorded_patch(new_value, path="source/triton_fused_moe_kernel.py"):
    """A patch as RECORDED: against the layout of the run that produced it, not this workspace."""
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n"
            f"--- a/{path}\n+++ b/{path}\n@@ -1,4 +1,4 @@\n"
            f" import triton\n \n-BLOCK = 64\n+BLOCK = {new_value}\n NUM_WARPS = 4\n")


def workspace(tmp_path, name="ws"):
    """A head-extracted layout: same file, a different place than the patch was recorded against."""
    ws = tmp_path / name
    src = ws / "kernel_src" / "sglang"
    src.mkdir(parents=True)
    (src / "triton_fused_moe_kernel.py").write_text(BASELINE)
    git(str(ws), "init", "-q")
    git(str(ws), "config", "user.email", "t@t")
    git(str(ws), "config", "user.name", "t")
    git(str(ws), "add", "-A")
    git(str(ws), "commit", "-qm", "baseline")
    return ws


def build_store(tmp_path, kb_root):
    jsonl = str(tmp_path / "records.jsonl")
    run("export-remote", "--root", kb_root, "--out", jsonl)
    store = str(tmp_path / "store")
    p = subprocess.run([sys.executable, UPLOADER, "--records", jsonl, "--local", store,
                        "--apply", "--quiet"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return store


def sessions_under(store, cid=CID):
    d = os.path.join(store, *cid.split(":"), "sessions")
    return sorted(n for n in os.listdir(d) if not n.startswith(".")) if os.path.isdir(d) else []


def champion_of(store, cid=CID):
    with open(os.path.join(store, *cid.split(":"), "champion.json")) as f:
        return json.load(f)


def test_the_warm_start_loop_closes(tmp_path):
    kb_root = str(tmp_path / "kb")
    recorded_entry(kb_root, "20260101_000000_a", speedup=3.0, direction="tile-retune",
                   patch=recorded_patch(128),
                   report="# tile retune\n## Key optimizations\nwidened the tile\n")
    recorded_entry(kb_root, "20260101_000000_b", speedup=1.8, direction="vectorize",
                   patch=recorded_patch(96, path="src/k.py"),
                   report="# vectorize\n## Key optimizations\nwider loads\n")
    recorded_entry(kb_root, "20260101_000000_c", speedup=1.01, direction="unroll",
                   patch=recorded_patch(72, path="src/k.py"), report="# unroll\n")
    store = build_store(tmp_path, kb_root)

    # 1. read the key. Ranked best first, the near-tie left out of the verify budget.
    read = run("resolve-remote", "--store", store, "--kernel-name", KERNEL, "--language", "triton",
               "--gfx", "gfx950", "--top-n", "3", "--min-speedup", "1.05",
               "--framework-version", "7.2", "--refs-dir", str(tmp_path / "refs"))
    assert read["read_reason"] == "read" and read["canonical_id"] == CID
    assert [c["speedup"] for c in read["candidates"]] == [3.0, 1.8]
    assert read["filtered"]["below_min_speedup"] == 1
    top = read["candidates"][0]
    assert os.path.getsize(top["patch_path"]) > 0
    assert "widened the tile" in open(top["prose_path"]).read()

    # 2. land it on a workspace whose layout the recorded patch never saw.
    ws = workspace(tmp_path)
    landed = str(tmp_path / "landed.diff")
    remapped = run("remap", "--patch", top["patch_path"], "--out", landed, "--workspace", str(ws),
                   "--editable", "kernel_src/sglang/triton_fused_moe_kernel.py")
    assert remapped["remapped"] is True
    git(str(ws), "apply", "--check", landed)
    git(str(ws), "apply", landed)
    assert "BLOCK = 128" in (ws / "kernel_src" / "sglang" / "triton_fused_moe_kernel.py").read_text()

    # 3. keep optimizing on top of what was adopted. (The measurement is the GPU's job; the number
    #    here stands in for it.) The patch written back is the FULL diff from the baseline, which is
    #    what the recorded speedup is measured against.
    path = ws / "kernel_src" / "sglang" / "triton_fused_moe_kernel.py"
    path.write_text(path.read_text().replace("BLOCK = 128", "BLOCK = 256")
                    .replace("NUM_WARPS = 4", "NUM_WARPS = 8"))
    improved = str(tmp_path / "improved.diff")
    with open(improved, "w") as f:
        f.write(git(str(ws), "diff"))

    # 4. write it back under the same key, deriving from the entry it was built on.
    wrote = run("write-remote", "--root", kb_root, "--store", store, "--kernel-name", KERNEL,
                "--language", "triton", "--gfx", "gfx950", "--kernel-class", "triton",
                "--speedup", "4.5", "--patch", improved, "--direction", "tile-retune",
                "--metric-kind", "geomean", "--framework-version", "7.2",
                "--parent", top["exp_dir"])
    assert wrote["written"] is True
    assert wrote["remote"]["written"] is True and wrote["remote"]["canonical_id"] == CID
    assert wrote["remote"]["replaced"] is False, "new code appends a candidate"
    assert wrote["remote"]["champion"] is True
    # All three recorded entries are still there, including the near-tie that was filtered out of
    # the read: the floor decides what is worth a verify slot, not what the store keeps.
    assert len(sessions_under(store)) == 4

    # 5. read the key again: the loop hands back the improvement, not the starting point.
    again = run("resolve-remote", "--store", store, "--kernel-name", KERNEL, "--language", "triton",
                "--gfx", "gfx950", "--top-n", "3", "--min-speedup", "1.05",
                "--framework-version", "7.2", "--refs-dir", str(tmp_path / "refs2"))
    assert [c["speedup"] for c in again["candidates"]] == [4.5, 1.8]
    assert again["candidates"][0]["session_id"] == wrote["remote"]["session_id"]
    assert again["candidates"][0]["is_champion"] is True
    assert champion_of(store)["session_id"] == wrote["remote"]["session_id"]
    # 3.0x and 4.5x share a direction, so the one that was superseded rides along as an alternate
    # rather than costing a second verify slot.
    assert again["filtered"]["same_direction_collapsed"] == 1
    assert [alt["speedup"] for alt in again["candidates"][0]["alternates"]] == [3.0]

    # and the improvement applies to the same baseline the run started from. It was recorded from
    # this very layout, so remap has nothing to rewrite and says so rather than emitting a copy.
    fresh = workspace(tmp_path, "ws2")
    relanded = str(tmp_path / "relanded.diff")
    fit = run("remap", "--patch", again["candidates"][0]["patch_path"], "--out", relanded,
              "--workspace", str(fresh), "--editable", "kernel_src/sglang/triton_fused_moe_kernel.py")
    assert fit["remapped"] is False and fit["reason"] == "no_change_needed"
    git(str(fresh), "apply", "--check", again["candidates"][0]["patch_path"])
    git(str(fresh), "apply", again["candidates"][0]["patch_path"])
    assert "BLOCK = 256" in (fresh / "kernel_src" / "sglang" / "triton_fused_moe_kernel.py").read_text()


def test_a_second_lap_over_the_same_code_does_not_grow_the_store(tmp_path):
    """A run that adopts a warm start and fails to beat it re-emits the patch it adopted. Recording
    that as a new candidate every time is how a store fills up with copies of one idea."""
    kb_root = str(tmp_path / "kb")
    recorded_entry(kb_root, "20260101_000000_a", speedup=3.0, direction="tile-retune",
                   patch=recorded_patch(128), report="# tile retune\n")
    store = build_store(tmp_path, kb_root)
    before = sessions_under(store)

    read = run("resolve-remote", "--store", store, "--kernel-name", KERNEL, "--language", "triton",
               "--gfx", "gfx950", "--framework-version", "7.2", "--refs-dir", str(tmp_path / "refs"))
    ws = workspace(tmp_path)
    landed = str(tmp_path / "landed.diff")
    run("remap", "--patch", read["candidates"][0]["patch_path"], "--out", landed,
        "--workspace", str(ws), "--editable", "kernel_src/sglang/triton_fused_moe_kernel.py")
    git(str(ws), "apply", landed)
    reemitted = str(tmp_path / "reemitted.diff")
    with open(reemitted, "w") as f:
        f.write(git(str(ws), "diff"))

    wrote = run("write-remote", "--root", kb_root, "--store", store, "--kernel-name", KERNEL,
                "--language", "triton", "--gfx", "gfx950", "--kernel-class", "triton",
                "--speedup", "2.95", "--patch", reemitted, "--direction", "tile-retune",
                "--framework-version", "7.2")
    assert wrote["written"] is False and wrote["reason"] == "duplicate_impl"
    assert wrote["remote"]["replaced"] is True, "the same code lands back on its own candidate"
    assert sessions_under(store) == before
    assert champion_of(store)["value"] == 3.0, "a slower remeasure does not take the pointer"
