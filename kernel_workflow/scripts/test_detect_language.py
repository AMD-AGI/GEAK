"""Contract tests for detect_language.py.

The detector exists because the lane's `TARGET_LANGUAGE` is a request, not an observation, and a
learned card labelled with a request is labelled wrong. So the bar here is not "usually right" --
it is "never confidently wrong". Each test below pins a case where an earlier version of this
detector WAS confidently wrong, measured against aiter:

  * a commented-out `ck_tile::` turned 15 `__global__` functions into a CK kernel;
  * one `cutlass::` mention outranked a whole HIP file;
  * treating `__global__` as a mere mention made HIP undetectable in all 33 files;
  * aggregating scores across a tree, then applying per-file host-language suppression, reported
    309 Triton kernels as Gluon because two files among them use Gluon.

The last test ties the vocabulary to kb.py: a language the detector can emit but the card lint
rejects would fail at the end of a run, after the expensive part.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "detect_language.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DL = load("detect_language", SCRIPT)


def write(tmp_path, name, src):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------------------------------
# The straightforward cases
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,src,expect", [
    ("k.py", "import flydsl\nfrom flydsl.compiler import flyc\n\n@flyc.kernel\ndef k():\n    pass\n",
     "flydsl"),
    ("k.py", "import triton\nimport triton.language as tl\n\n@triton.jit\ndef k():\n    tl.load(0)\n",
     "triton"),
    ("k.py", "import tilelang\n\n@tilelang.jit\ndef k():\n    pass\n", "tilelang"),
    ("k.cu", '#include <hip/hip_runtime.h>\n__global__ void k() {}\n', "hip"),
    ("k.cpp", '#include "ck/tensor_operation/gpu/device/x.hpp"\n__global__ void k() {}\n', "ck"),
    ("k.cpp", '#include <rocwmma/rocwmma.hpp>\n__global__ void k() {}\n', "rocwmma"),
    ("k.s", ".amdhsa_kernel k\n  v_mfma_f32_16x16x16f16 v[0:3], v4, v5, v[0:3]\n", "asm"),
])
def test_declared_language_is_detected(tmp_path, name, src, expect):
    assert DL.detect([write(tmp_path, name, src)])["language"] == expect


def test_a_file_with_no_marker_is_undecided(tmp_path):
    res = DL.detect([write(tmp_path, "plain.py", "import torch\n\ndef f(x):\n    return x + 1\n")])
    assert res["language"] is None
    assert "declares" in res["reason"] or "no authoring-language" in res["reason"]


# --------------------------------------------------------------------------------------------------
# Host-language suppression: a dialect's file carries its host's markers
# --------------------------------------------------------------------------------------------------

def test_gluon_outranks_the_triton_it_imports(tmp_path):
    src = """
        import triton
        import triton.language as tl
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl

        @gluon.jit
        def k():
            gl.alloc_shared(0)
    """
    res = DL.detect([write(tmp_path, "k.py", src)])
    assert res["language"] == "gluon", res["reason"]


def test_ck_outranks_the_hip_it_compiles_as(tmp_path):
    src = """
        #include <hip/hip_runtime.h>
        #include "ck/utility/data_type.hpp"
        __global__ void k() {}
    """
    assert DL.detect([write(tmp_path, "k.cu", src)])["language"] == "ck"


def test_a_namespace_mention_does_not_outrank_the_host_language(tmp_path):
    """One `cutlass::` in a HIP file is a call, not an authorship claim."""
    src = """
        #include <hip/hip_runtime.h>
        __global__ void k() { auto x = cutlass::something(); }
        __global__ void k2() {}
    """
    res = DL.detect([write(tmp_path, "k.cu", src)])
    assert res["language"] == "hip", res["reason"]


# --------------------------------------------------------------------------------------------------
# Comments are not evidence
# --------------------------------------------------------------------------------------------------

def test_a_commented_out_include_is_not_evidence(tmp_path):
    src = """
        #include <hip/hip_runtime.h>
        // #include "ck/utility/data_type.hpp"
        /* ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(p, oob); */
        __global__ void k() {}   // ck_tile::something
    """
    res = DL.detect([write(tmp_path, "k.cu", src)])
    assert res["language"] == "hip", res["reason"]
    assert "ck" not in res["votes"]


def test_a_commented_out_python_import_is_not_evidence(tmp_path):
    src = """
        import triton

        # import flydsl
        @triton.jit
        def k():
            pass
    """
    res = DL.detect([write(tmp_path, "k.py", src)])
    assert res["language"] == "triton", res["reason"]
    assert "flydsl" not in res["votes"]


def test_comment_stripping_keeps_line_numbers(tmp_path):
    src = """
        // pad
        /* a
           multi-line
           comment */
        import triton
    """
    path = write(tmp_path, "k.cpp", src)
    with open(path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    stripped = DL.strip_comments(raw, ".cpp")
    assert len(stripped) == len(raw), "evidence line numbers are indices into this list"
    assert "multi-line" not in "".join(stripped)
    assert "import triton" in "".join(stripped)


# --------------------------------------------------------------------------------------------------
# Aggregation across files
# --------------------------------------------------------------------------------------------------

def test_a_minority_dialect_does_not_capture_the_tree(tmp_path):
    """Per-file verdicts then vote. Summing scores first made 309 Triton files read as Gluon."""
    for i in range(8):
        write(tmp_path / "t", f"k{i}.py", "import triton\n\n@triton.jit\ndef k():\n    pass\n")
    write(tmp_path / "t", "odd.py",
          "from triton.experimental import gluon\n\n@gluon.jit\ndef k():\n    pass\n")
    res = DL.detect([str(tmp_path / "t")])
    assert res["language"] == "triton", res["reason"]
    assert res["votes"]["gluon"] == 1


def test_two_declared_languages_in_equal_measure_are_refused(tmp_path):
    write(tmp_path / "m", "a.py", "import triton\n\n@triton.jit\ndef k():\n    pass\n")
    write(tmp_path / "m", "b.py", "import flydsl\nfrom flydsl.compiler import flyc\n\n@flyc.kernel\ndef k():\n    pass\n")
    res = DL.detect([str(tmp_path / "m")])
    assert res["language"] is None
    assert "ambiguous" in res["reason"]


def test_a_refused_verdict_names_the_languages_it_was_refused_between(tmp_path):
    """A seam that vendors a dispatcher carries a FlyDSL path and a Triton fallback in one file.
    The verdict stays refused; `ambiguous_between` only says which requests it would confirm."""
    seam = write(tmp_path, "seam.py", """\
        import flydsl
        import triton
        from flydsl._mlir import ir
        from aiter.ops.triton.gemm import gemm_a16w16

        @flyc.kernel
        def hgemm():
            pass

        @triton.jit
        def fallback():
            tl.program_id(0)
        """)
    res = DL.detect([seam])
    assert res["language"] is None
    assert res["ambiguous_between"] == ["flydsl", "triton"]
    plain = write(tmp_path, "k.py", "import flydsl\nfrom flydsl.compiler import flyc\n\n@flyc.kernel\ndef k():\n    pass\n")
    assert DL.detect([plain])["ambiguous_between"] == []
    no_marker = write(tmp_path, "none.py", "def k():\n    return 1\n")
    assert DL.detect([no_marker])["ambiguous_between"] == []


def test_the_entry_file_outweighs_the_harness_around_it(tmp_path):
    write(tmp_path / "e", "kernel_impl.py",
          "import flydsl\nfrom flydsl.compiler import flyc\n\n@flyc.kernel\ndef my_op():\n    pass\n")
    write(tmp_path / "e", "test_my_op.py", "import triton\n\n@triton.jit\ndef ref():\n    pass\n")
    write(tmp_path / "e", "bench_my_op.py", "import triton\n\n@triton.jit\ndef ref2():\n    pass\n")
    res = DL.detect([str(tmp_path / "e")], entry="my_op")
    assert res["language"] == "flydsl", res["reason"]


# --------------------------------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------------------------------

def run_cli(*args):
    # check=False on purpose: the exit status IS what these tests assert on.
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def test_cli_exit_status_reports_whether_a_language_was_decided(tmp_path):
    decided = write(tmp_path, "k.py", "import flydsl\nfrom flydsl.compiler import flyc\n\n@flyc.kernel\ndef k():\n    pass\n")
    rc, out, _ = run_cli(decided, "--json")
    assert rc == 0 and json.loads(out)["language"] == "flydsl"

    undecided = write(tmp_path, "plain.py", "import torch\n")
    rc, out, _ = run_cli(undecided, "--json")
    assert rc == 1 and json.loads(out)["language"] is None


def test_cli_refuses_a_path_that_does_not_exist(tmp_path):
    rc, _, err = run_cli(str(tmp_path / "nope"), "--json")
    assert rc == 2 and "do not exist" in err, (
        "a typo'd path must not be reported as a kernel with no language markers"
    )


# --------------------------------------------------------------------------------------------------
# The vocabulary has to agree with the card lint, or a run fails after the expensive part
# --------------------------------------------------------------------------------------------------

def test_every_detectable_language_is_accepted_by_the_card_lint():
    kb = load("kb", os.path.join(HERE, "kb.py"))
    emitted = {lang for lang, *_ in DL.MARKERS} | set(DL.BY_EXT.values())
    unknown = emitted - set(kb.AUTHORING_LANGUAGES)
    assert not unknown, (
        f"detect_language can emit {sorted(unknown)}, which kb.py's card lint would reject; both "
        f"must track perf_knowledge/index/taxonomy.md"
    )
