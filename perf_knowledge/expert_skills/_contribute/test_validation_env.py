"""Contract tests for check_validation_env.py.

The check draws one line, and both sides of it matter:

  * `not_captured` PASSES. A run already in the past cannot grow a ROCm version, and demanding one
    would push a curator toward a plausible-looking invention. Two of the three real artifacts in
    this tree can never record more than they do.
  * an ABSENT field FAILS. Silence is what makes a stale number dangerous: the reader cannot tell
    "nobody wrote it down" from "same environment as mine".

And it must stay a report. `test_the_check_never_writes` is the guard that keeps this from becoming
a version gate, which the portability contract forbids.
"""
import importlib.util
import os
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "check_validation_env.py")
REAL_SKILLS = os.path.join(os.path.dirname(HERE), "skills")


def load():
    spec = importlib.util.spec_from_file_location("check_validation_env", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CV = load()


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """A skills/ tree the test owns, so the real one is neither read nor written."""
    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setattr(CV, "EXPERT_SKILLS", str(tmp_path))
    monkeypatch.setattr(CV, "SKILLS", str(skills))
    monkeypatch.setattr(CV, "REPO", str(tmp_path))

    def make(skill_id, frontmatter, artifact_body=None, artifact_name="validation.yaml",
             artifact_is_dir=False):
        d = skills / skill_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "skill.md").write_text(f"---\n{textwrap.dedent(frontmatter)}---\nbody\n",
                                    encoding="utf-8")
        if artifact_is_dir:
            (d / artifact_name).mkdir(exist_ok=True)
        elif artifact_body is not None:
            (d / artifact_name).write_text(textwrap.dedent(artifact_body), encoding="utf-8")
        return skill_id

    return make


FM = """\
scope: kernel
match:
  to_backend: flydsl
validation:
  status: validated
  artifact: skills/{sid}/validation.yaml
"""


def test_not_captured_is_accepted(fake_tree):
    sid = fake_tree("s1", FM.format(sid="s1"), """
        flydsl_version: 0.2.2
        rocm_version: not_captured
        aiter_commit: not_captured
        gpu: gfx950
    """)
    row = CV.check_skill(sid)
    assert row["problems"] == []
    assert set(row["not_captured"]) == {"rocm_version", "aiter_commit"}
    assert row["recorded"] == {"flydsl_version": "0.2.2", "gpu": "gfx950"}


def test_an_absent_field_is_reported(fake_tree):
    sid = fake_tree("s2", FM.format(sid="s2"), """
        flydsl_version: 0.2.2
        gpu: gfx950
    """)
    row = CV.check_skill(sid)
    assert len(row["problems"]) == 2
    assert all("Absent is not the same as unknown" in p for p in row["problems"])


def test_the_backend_decides_which_version_field_is_owed(fake_tree):
    assert CV.required_fields("flydsl")[0] == "flydsl_version"
    assert CV.required_fields("triton")[0] == "triton_version"
    assert "rocm_version" in CV.required_fields("")
    assert not any(f.endswith("_version") and f != "rocm_version"
                   for f in CV.required_fields("")), "no backend named, none demanded"


@pytest.mark.parametrize("status", ["draft", "failed", ""])
def test_an_unvalidated_skill_owes_nothing(fake_tree, status):
    fm = FM.format(sid="s3").replace("status: validated", f"status: {status}")
    row = CV.check_skill(fake_tree("s3", fm, "gpu: gfx950\n"))
    assert row["problems"] == []


def test_validated_with_no_artifact_is_reported(fake_tree):
    fm = """\
        scope: kernel
        match:
          to_backend: flydsl
        validation:
          status: validated
          artifact: ''
    """
    row = CV.check_skill(fake_tree("s4", fm))
    assert len(row["problems"]) == 1
    assert "has no record" in row["problems"][0]


def test_a_directory_artifact_is_reported(fake_tree):
    sid = fake_tree("s5", FM.format(sid="s5"), artifact_is_dir=True)
    row = CV.check_skill(sid)
    assert len(row["problems"]) == 1
    assert "is a directory" in row["problems"][0]


def test_a_missing_artifact_path_is_reported(fake_tree):
    sid = fake_tree("s6", FM.format(sid="s6"))
    row = CV.check_skill(sid)
    assert len(row["problems"]) == 1
    assert "does not exist" in row["problems"][0]


def test_the_check_never_writes(fake_tree):
    """A report that edits a skill is a gate. The portability contract forbids one."""
    sid = fake_tree("s7", FM.format(sid="s7"), "gpu: gfx950\n")
    skill_md = os.path.join(CV.SKILLS, sid, "skill.md")
    with open(skill_md, encoding="utf-8") as f:
        before = f.read()
    mtime = os.path.getmtime(skill_md)
    CV.check_skill(sid)
    with open(skill_md, encoding="utf-8") as f:
        assert f.read() == before
    assert os.path.getmtime(skill_md) == mtime


# --------------------------------------------------------------------------------------------------
# The real tree: locks in what was fixed, and names what is still missing
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("skill_id", [
    "flydsl_decode_moe_stage1_blkmap",
    "flydsl_prefill_moe_stage2_fp8partial",
    "flydsl_fp8_blockscale_gemm",
])
def test_the_three_real_measurement_records_state_their_environment(skill_id):
    row = CV.check_skill(skill_id)
    assert row["problems"] == [], row["problems"]
    assert row["recorded"] or row["not_captured"], "every field must say something"


@pytest.mark.parametrize("skill_id", [
    "flydsl_fp8_gemm_playbook",
    "flydsl_rewrite_quantized_moe",
    "apply_flydsl_moe_to_vllm",
])
def test_skills_validated_without_a_measurement_record_are_still_named(skill_id):
    """Not a fix -- a marker. These three claim `validated` with no readable record, and a record
    cannot be invented for a run nobody archived. The test exists so that supplying one deletes the
    test, rather than the gap being quietly forgotten."""
    row = CV.check_skill(skill_id)
    assert row["problems"], (
        f"{skill_id} now has a readable measurement record — good. Remove it from this list."
    )
