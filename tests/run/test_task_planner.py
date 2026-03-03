"""Tests for the dynamic task planner."""

from pathlib import Path

from minisweagent.run.task_planner import _pick_fusion_target_lang, build_optimization_tasks
from minisweagent.tools.discovery_types import (
    DiscoveryResult,
    FusionOpportunity,
    KernelDependencyGraph,
    KernelInfo,
)


class FakeAgentClass:
    """Stand-in for an agent class in tests."""

    pass


def _make_kernel(
    kernel_type: str = "triton",
    kernel_language: str = "python",
    inner_kernel_path: Path | None = None,
) -> KernelInfo:
    return KernelInfo(
        file_path=Path("/workspace/kernel.py"),
        kernel_name="test_kernel",
        kernel_type=kernel_type,
        kernel_language=kernel_language,
        function_names=["kernel_fwd", "kernel_bwd"],
        has_jit_decorator=True,
        inner_kernel_path=inner_kernel_path,
    )


def _make_discovery(
    kernel_type: str = "triton",
    kernel_language: str = "python",
    inner_kernel_path: Path | None = None,
    fusion_opps: list[FusionOpportunity] | None = None,
) -> DiscoveryResult:
    kernel = _make_kernel(kernel_type, kernel_language, inner_kernel_path)
    dep_graphs: dict[str, KernelDependencyGraph] = {}
    if fusion_opps:
        graph = KernelDependencyGraph(root_name=kernel.kernel_name)
        graph.fusion_opportunities = fusion_opps
        dep_graphs[kernel.kernel_name] = graph
    return DiscoveryResult(
        kernels=[kernel],
        dependency_graphs=dep_graphs,
    )


# ---- Empty / no kernels ----


def test_no_kernels_returns_empty():
    result = DiscoveryResult()
    tasks = build_optimization_tasks(result, "context", FakeAgentClass)
    assert tasks == []


# ---- Triton tasks ----


def test_triton_generates_openevolve_autotune_and_algorithmic():
    dr = _make_discovery("triton", "python", inner_kernel_path=Path("/ws/inner.py"))
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    labels = [t.label for t in tasks]
    assert "openevolve-inner" in labels
    assert "triton-autotune" in labels
    assert "triton-algorithmic" in labels
    assert "profile-guided" in labels


def test_triton_without_inner_kernel_still_generates_tasks():
    dr = _make_discovery("triton", "python")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    labels = [t.label for t in tasks]
    # No openevolve-inner since no inner_kernel_path
    assert "openevolve-inner" not in labels
    assert "triton-autotune" in labels


# ---- HIP tasks ----


def test_hip_generates_launch_and_memory_tasks():
    dr = _make_discovery("hip", "cpp")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    labels = [t.label for t in tasks]
    assert "hip-launch-config" in labels
    assert "hip-memory" in labels
    assert "profile-guided" in labels


# ---- CK tasks ----


def test_ck_generates_template_tuning():
    dr = _make_discovery("ck", "cpp")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    labels = [t.label for t in tasks]
    assert "ck-template-tuning" in labels
    assert "ck-pipeline" in labels


# ---- ASM tasks ----


def test_asm_generates_wrapper_optimization():
    dr = _make_discovery("asm", "asm")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    labels = [t.label for t in tasks]
    assert "asm-launch-config" in labels
    assert "profile-guided" in labels


# ---- Fusion opportunity tasks ----


def test_fusion_opportunities_generate_tasks():
    fusion_opps = [
        FusionOpportunity(
            description="Fuse kernel_a + kernel_b",
            involved_nodes=["kernel_a", "kernel_b"],
            languages={"triton"},
            fusion_type="sequential_launch",
            estimated_benefit="high",
        ),
    ]
    dr = _make_discovery("triton", "python", fusion_opps=fusion_opps)
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    fusion_tasks = [t for t in tasks if t.label.startswith("fusion-")]
    assert len(fusion_tasks) == 1
    assert fusion_tasks[0].priority == 5


# ---- Priority ordering ----


def test_tasks_sorted_by_priority():
    dr = _make_discovery("triton", "python", inner_kernel_path=Path("/ws/inner.py"))
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    priorities = [t.priority for t in tasks]
    assert priorities == sorted(priorities)


# ---- kernel_language set correctly ----


def test_hip_tasks_have_cpp_language():
    dr = _make_discovery("hip", "cpp")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    for t in tasks:
        if t.label.startswith("hip-"):
            assert t.kernel_language == "cpp"


# ---- AgentTask fields ----


def test_agent_task_has_correct_class():
    dr = _make_discovery("triton", "python")
    tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
    for t in tasks:
        assert t.agent_class is FakeAgentClass


# ---- _pick_fusion_target_lang ----


def test_pick_fusion_target_lang_triton():
    assert _pick_fusion_target_lang({"triton"}) == "python"
    assert _pick_fusion_target_lang({"triton", "python"}) == "python"


def test_pick_fusion_target_lang_ck():
    assert _pick_fusion_target_lang({"ck"}) == "cpp"
    assert _pick_fusion_target_lang({"ck", "hip"}) == "cpp"


def test_pick_fusion_target_lang_asm():
    # ASM + triton => still cpp since triton check excludes asm
    assert _pick_fusion_target_lang({"triton", "asm"}) == "cpp"


# ---- Profile-guided task always present ----


def test_profile_guided_always_present():
    for ktype in ("triton", "hip", "ck", "asm"):
        dr = _make_discovery(ktype, "python" if ktype == "triton" else "cpp")
        tasks = build_optimization_tasks(dr, "ctx", FakeAgentClass)
        labels = [t.label for t in tasks]
        assert "profile-guided" in labels, f"Missing profile-guided for {ktype}"
