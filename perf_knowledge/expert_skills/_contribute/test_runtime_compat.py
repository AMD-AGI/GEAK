import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


HERE = Path(__file__).resolve().parent
RUNTIME_COMPAT = HERE / "runtime_compat.py"


def load_module(path, name):
    if not path.exists():
        pytest.fail(f"{path.name} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_compat():
    return load_module(RUNTIME_COMPAT, "runtime_compat")


def aiter_runtime_contract():
    common_imports = [
        "flydsl",
        "aiter.ops.flydsl",
        "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage",
    ]
    return {
        "language": "flydsl",
        "provider": "aiter_vendored_flydsl",
        "required_imports": common_imports,
        "profiles": [
            {
                "name": "current-0.2.2",
                "specifier": "==0.2.2",
                "validation_status": "validated",
                "required_symbols": [
                    "flydsl.expr.typing:as_ir_value",
                    "flydsl.expr.buffer_ops:create_buffer_resource_from_addr",
                ],
            },
            {
                "name": "future-0.2",
                "specifier": ">=0.2.3,<0.3",
                "validation_status": "revalidation_required",
                "required_symbols": [
                    "flydsl.expr.typing:as_ir_value",
                    "flydsl.expr.buffer_ops:create_buffer_resource_from_addr",
                ],
            },
        ],
        "provisioning": {"policy": "reuse_only"},
    }


def load_skill(skill_id):
    path = HERE.parent / "skills" / skill_id / "skill.md"
    text = path.read_text()
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_validates_provider_aware_aiter_contract():
    compat = load_runtime_compat()

    assert compat.validate_runtime_contract(aiter_runtime_contract()) == []


def test_rejects_unbounded_flydsl_version_range():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["profiles"][1]["specifier"] = ">=0.2.2"

    errors = compat.validate_runtime_contract(runtime)

    assert any("upper bound" in error for error in errors)


def test_rejects_standalone_kernel_requirement_for_aiter_provider():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["required_imports"].append("kernels.moe_gemm_2stage")

    errors = compat.validate_runtime_contract(runtime)

    assert any("standalone" in error for error in errors)


def test_rejects_profile_level_standalone_requirement_for_aiter_provider():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["profiles"][1]["required_imports"] = ["kernels.moe_gemm_2stage"]

    errors = compat.validate_runtime_contract(runtime)

    assert any("standalone" in error for error in errors)


def test_rejects_profile_symbol_from_standalone_namespace_for_aiter_provider():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["profiles"][1]["required_symbols"].append("kernels.moe_gemm_2stage:compile")

    errors = compat.validate_runtime_contract(runtime)

    assert any("standalone" in error for error in errors)


def test_aiter_provider_requires_exact_aiter_flydsl_namespace():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["required_imports"] = ["flydsl", "aiter.ops.flydsl_fake"]

    errors = compat.validate_runtime_contract(runtime)

    assert any("aiter.ops.flydsl import capability" in error for error in errors)


def test_each_profile_must_prove_its_effective_provider_capability():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["required_imports"] = ["flydsl"]
    runtime["profiles"][0]["required_imports"] = ["aiter.ops.flydsl"]

    errors = compat.validate_runtime_contract(runtime)

    assert any(
        "runtime.profiles[1]" in error
        and "aiter.ops.flydsl import capability" in error
        for error in errors
    )


def test_rejects_specifier_without_a_future_upper_bound():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["profiles"][1]["specifier"] = "!=0.3"

    errors = compat.validate_runtime_contract(runtime)

    assert any("upper bound" in error for error in errors)


def test_rejects_malformed_common_required_symbol():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["required_symbols"] = ["missing-colon"]

    errors = compat.validate_runtime_contract(runtime)

    assert any("module:attribute" in error for error in errors)


@pytest.mark.parametrize(
    ("version", "expected_name", "expected_status"),
    [
        ("0.2.2", "current-0.2.2", "validated"),
        ("0.2.3", "future-0.2", "revalidation_required"),
    ],
)
def test_selects_runtime_profile_with_pep440(version, expected_name, expected_status):
    compat = load_runtime_compat()

    profile = compat.select_runtime_profile(aiter_runtime_contract(), version)

    assert profile["name"] == expected_name
    assert profile["validation_status"] == expected_status


@pytest.mark.parametrize(
    "version", ["0.1.2", "0.2.0", "0.2.1", "0.2.9.dev1", "0.3.0"]
)
def test_returns_none_when_no_profile_matches(version):
    compat = load_runtime_compat()

    assert compat.select_runtime_profile(aiter_runtime_contract(), version) is None


def test_explicit_prerelease_profile_can_match():
    compat = load_runtime_compat()
    runtime = aiter_runtime_contract()
    runtime["profiles"] = [
        {
            "name": "explicit-release-candidate",
            "specifier": "==0.3.0rc1",
            "validation_status": "revalidation_required",
        }
    ]

    profile = compat.select_runtime_profile(runtime, "0.3.0rc1")

    assert profile["name"] == "explicit-release-candidate"


def test_probe_uses_aiter_vendored_modules_without_standalone_kernel():
    compat = load_runtime_compat()
    requested = []
    modules = {
        "flydsl": SimpleNamespace(__file__="/env/flydsl/__init__.py"),
        "aiter.ops.flydsl": SimpleNamespace(__file__="/repo/aiter/ops/flydsl/__init__.py"),
        "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage": SimpleNamespace(
            __file__="/repo/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"
        ),
        "flydsl.expr.typing": SimpleNamespace(
            __file__="/env/flydsl/expr/typing.py",
            as_ir_value=lambda value: value,
        ),
        "flydsl.expr.buffer_ops": SimpleNamespace(
            __file__="/env/flydsl/expr/buffer_ops.py",
            create_buffer_resource_from_addr=lambda address: address,
        ),
    }

    def importer(name):
        requested.append(name)
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    result = compat.probe_runtime(
        aiter_runtime_contract(),
        version_getter=lambda: "0.2.2",
        importer=importer,
    )

    assert result["compatible"] is True
    assert result["profile"] == "current-0.2.2"
    assert result["validation_status"] == "validated"
    assert "kernels.moe_gemm_2stage" not in requested


def test_probe_reports_missing_required_symbol():
    compat = load_runtime_compat()
    modules = {
        "flydsl": SimpleNamespace(__file__="/env/flydsl/__init__.py"),
        "aiter.ops.flydsl": SimpleNamespace(__file__="/repo/aiter/ops/flydsl/__init__.py"),
        "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage": SimpleNamespace(
            __file__="/repo/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"
        ),
        "flydsl.expr.typing": SimpleNamespace(__file__="/env/flydsl/expr/typing.py"),
        "flydsl.expr.buffer_ops": SimpleNamespace(
            __file__="/env/flydsl/expr/buffer_ops.py",
            create_buffer_resource_from_addr=lambda address: address,
        ),
    }

    result = compat.probe_runtime(
        aiter_runtime_contract(),
        version_getter=lambda: "0.2.2",
        importer=lambda name: modules[name],
    )

    assert result["compatible"] is False
    assert any("as_ir_value" in error for error in result["errors"])


def test_version_discovery_prefers_effective_import_over_distribution_metadata():
    compat = load_runtime_compat()

    version = compat.discover_flydsl_version(
        importer=lambda name: SimpleNamespace(__version__="0.2.3"),
    )

    assert version == "0.2.3"


def test_version_discovery_fails_closed_when_effective_module_has_no_version():
    compat = load_runtime_compat()

    with pytest.raises(RuntimeError, match="effective flydsl module"):
        compat.discover_flydsl_version(
            importer=lambda name: SimpleNamespace(),
        )


def test_scaffold_index_entry_preserves_nonempty_runtime_contract():
    scaffold = load_module(HERE / "scaffold.py", "expert_skill_scaffold")
    runtime = aiter_runtime_contract()
    frontmatter = {
        "id": "example",
        "scope": "kernel",
        "match": {"operator": "grouped_gemm_moe"},
        "expects": {"isolated_speedup_min": 1.0},
        "runtime": runtime,
        "validation": {"status": "validated"},
    }

    entry = scaffold.skill_index_entry("example", frontmatter)

    assert entry["runtime"] == runtime


def test_static_validation_rejects_invalid_runtime_contract():
    validator = load_module(HERE / "validate_skill.py", "expert_skill_validator")
    frontmatter = {
        "id": "example",
        "scope": "kernel",
        "match": {"operator": "grouped_gemm_moe"},
        "expects": {"isolated_speedup_min": 1.0},
        "runtime": {
            "language": "flydsl",
            "provider": "aiter_vendored_flydsl",
            "required_imports": ["aiter.ops.flydsl"],
            "profiles": [
                {
                    "name": "unsafe",
                    "specifier": ">=0.2.2",
                    "validation_status": "validated",
                }
            ],
            "provisioning": {"policy": "reuse_only"},
        },
    }
    body = "\n".join(
        f"## {section}\nfilled"
        for section in validator.REQUIRED_SECTIONS
    )

    errors = validator.static_check(frontmatter, body)

    assert any("upper bound" in error for error in errors)


def test_emit_plan_includes_runtime_probe(capsys):
    validator = load_module(HERE / "validate_skill.py", "expert_skill_validator_emit")
    frontmatter = {
        "scope": "kernel",
        "match": {"to_backend": "flydsl"},
        "runtime": aiter_runtime_contract(),
    }

    validator.emit_plan(
        "example",
        frontmatter,
        SimpleNamespace(model=""),
    )

    output = capsys.readouterr().out
    assert "runtime_compat.py example --json" in output


def test_skill_template_exposes_optional_runtime_contract():
    template = (HERE.parent / "_template" / "SKILL_TEMPLATE.md").read_text()

    assert "\nruntime:" in template


@pytest.mark.parametrize(
    "relative_path",
    [
        "e2e_workflow/roles/_fragments/expert_skills.md",
        "kernel_workflow/roles/_fragments/expert_skills.md",
    ],
)
def test_workflow_fragments_gate_skills_with_runtime_probe(relative_path):
    root = HERE.parents[2]
    fragment = (root / relative_path).read_text()

    assert "<EXPERT_SKILLS_DIR>/_contribute/runtime_compat.py" in fragment
    assert "backend_incompatible" in fragment
    assert "revalidation_required" in fragment
    assert "`revalidation_required` or `stale`" in fragment


def test_expert_skill_readme_documents_runtime_contract():
    readme = (HERE.parent / "README.md").read_text()

    assert "aiter_vendored_flydsl" in readme
    assert "standalone_source_flydsl" in readme
    assert "runtime_compat.py" in readme
    assert "does not install" in readme
    assert "Prerelease/dev builds do not match" in readme


@pytest.mark.parametrize(
    "skill_id",
    [
        "flydsl_decode_moe_stage1_blkmap",
        "flydsl_prefill_moe_stage2_fp8partial",
    ],
)
def test_gfx950_moe_skills_require_geak_flydsl_floor(skill_id):
    frontmatter, body = load_skill(skill_id)
    runtime = frontmatter["runtime"]
    compat = load_runtime_compat()

    assert runtime["provider"] == "aiter_vendored_flydsl"
    assert runtime["language"] == "flydsl"
    assert runtime["provisioning"]["policy"] == "reuse_only"
    assert "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage" in runtime["required_imports"]
    assert "kernels.moe_gemm_2stage" not in runtime["required_imports"]
    assert [
        (profile["specifier"], profile["validation_status"])
        for profile in runtime["profiles"]
    ] == [
        ("==0.2.2", "validated"),
        (">0.2.2,<0.3", "revalidation_required"),
    ]
    assert compat.select_runtime_profile(runtime, "0.2.0") is None
    assert compat.select_runtime_profile(runtime, "0.2.2")["name"] == "validated-0.2.2"
    assert compat.select_runtime_profile(runtime, "0.2.3")["name"] == "future-0.2"
    assert "<EXPERT_SKILLS_DIR>/_contribute/runtime_compat.py" in body
    assert "fx.Int32" in body
    assert "as_ir_value" in body
    assert "0.2.2 compatibility smoke" in body
    assert "0.2.0" not in body


def test_decode_skill_preserves_runtime_engagement_signatures():
    frontmatter, body = load_skill("flydsl_decode_moe_stage1_blkmap")

    assert "_am2_bmap" in body
    assert "_blkmap_kernel" in body
    assert "192.706" in body
    assert "174.498" in body
    assert "1.104x" in body
    artifact = HERE.parent / frontmatter["validation"]["artifact"]
    evidence = yaml.safe_load(artifact.read_text())
    assert evidence["flydsl_version"] == "0.2.2"
    assert evidence["speedup"] == pytest.approx(1.1043445)


def test_prefill_skill_requires_one_scale_compile_argument_and_cache_identity():
    frontmatter, body = load_skill("flydsl_prefill_moe_stage2_fp8partial")

    assert "compile argument" in body
    assert "cache identity" in body
    assert "producer and reducer" in body
    assert "1.206495" in body
    assert "0.997182" in body
    assert "1.210x" in body
    artifact = HERE.parent / frontmatter["validation"]["artifact"]
    evidence = yaml.safe_load(artifact.read_text())
    assert evidence["flydsl_version"] == "0.2.2"
    assert evidence["speedup"] == pytest.approx(1.2099041)


def test_generated_index_carries_moe_runtime_contracts():
    index = yaml.safe_load((HERE.parent / "index.yaml").read_text())
    by_id = {entry["id"]: entry for entry in index["skills"]}

    for skill_id in (
        "flydsl_decode_moe_stage1_blkmap",
        "flydsl_prefill_moe_stage2_fp8partial",
    ):
        runtime = by_id[skill_id]["runtime"]
        assert runtime["provider"] == "aiter_vendored_flydsl"
        assert [
            (profile["specifier"], profile["validation_status"])
            for profile in runtime["profiles"]
        ] == [
            ("==0.2.2", "validated"),
            (">0.2.2,<0.3", "revalidation_required"),
        ]
