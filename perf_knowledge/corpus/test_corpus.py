#!/usr/bin/env python3
"""Invariants for the operator corpus.

Two kinds of test here, kept apart on purpose.

The hermetic ones build tiny source trees and assert on what the extractor does with them. They run
anywhere and they are where a rule's behaviour is pinned — a rule tested only against real aiter is
tested against a moving target, and when it breaks you cannot tell whether the rule regressed or the
upstream file changed.

The committed-artifact ones read `evidence/*.yaml` as it stands in the repo and assert the properties
the README promises: language vocabulary shared with `kb.py`, verbatim excerpts, no performance
numbers, gaps stated rather than dropped, and the rendered page matching its inputs. These are the
tests that catch a bad regeneration, which is the failure mode that actually happens — nobody
hand-edits a 17k-line YAML, but plenty of people run the extractor against the wrong tree.

There is no test asserting a particular count. `flydsl == 339` would fail on every aiter bump and
teach the next person to update the number rather than look at why it moved.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
EVIDENCE = os.path.join(HERE, "evidence")
FAMILY = os.path.join(EVIDENCE, "gemm_source.yaml")
TUNED = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
TUNED_DB = os.path.join(EVIDENCE, "gemm_csv_tuned_configs.yaml")
AXES = os.path.join(HERE, "performance_axes", "gemm.yaml")
DECISIONS = os.path.join(HERE, "decisions", "gemm.yaml")
SOURCE_DOC = os.path.join(HERE, "gemm_source_evidence.md")
DECISION_DOC = os.path.join(HERE, "gemm_decisions.md")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = _load("_extract_impl_facts", os.path.join(HERE, "_extract_impl_facts.py"))
RE_ = _load("_render_facts", os.path.join(HERE, "_render_facts.py"))
RD = _load("_render_decisions", os.path.join(HERE, "_render_decisions.py"))


@pytest.fixture(scope="module")
def source():
    if not os.path.exists(FAMILY):
        pytest.skip(f"{FAMILY} not generated")
    return RE_.load_yaml(FAMILY)


@pytest.fixture(scope="module")
def tuned():
    if not os.path.exists(TUNED):
        pytest.skip(f"{TUNED} not generated")
    return RE_.load_yaml(TUNED)


@pytest.fixture(scope="module")
def decisions():
    if not os.path.exists(DECISIONS):
        pytest.skip(f"{DECISIONS} not written")
    return RE_.load_yaml(DECISIONS)


@pytest.fixture(scope="module")
def tuned_db():
    if not os.path.exists(TUNED_DB):
        pytest.skip(f"{TUNED_DB} not generated")
    return RE_.load_yaml(TUNED_DB)


@pytest.fixture(scope="module")
def axes():
    return RE_.load_yaml(AXES)


def scan(tmp_path, name, body, language="flydsl"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    got, _, _ = EX.scan_file(str(p), name, language)
    return got


def cats(records):
    return {r["category"] for r in records}


# --------------------------------------------------------------------------------------------
# Rules, against trees we control
# --------------------------------------------------------------------------------------------
def test_a_commented_out_intrinsic_is_not_a_fact(tmp_path):
    """The failure `detect_language.py` hit, guarded here too: a mention is not a use."""
    got = scan(tmp_path, "k.py", """
        # rocdl.mfma_f32_16x16x16f16(a, b, c)   <- tried this, was slower
        x = 1
        """)
    assert not got, f"matched inside a comment: {got}"


def test_a_real_intrinsic_is_a_fact_with_its_line(tmp_path):
    got = scan(tmp_path, "k.py", """
        x = 1
        acc = rocdl.mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0)
        """)
    hits = [g for g in got if g["category"] == "mfma_intrinsic"]
    assert len(hits) == 1
    assert hits[0]["line"] == 3, "line number must survive comment stripping"
    assert hits[0]["captured"] == ["mfma_f32_16x16x32_bf16"]


def test_c_style_block_comments_are_stripped_without_shifting_lines(tmp_path):
    got = scan(tmp_path, "k.cu", """
        /* __shared__ float buf[256];
           more commentary */
        __shared__ float real[256];
        """, language="hip")
    hits = [g for g in got if g["category"] == "lds_alloc"]
    assert len(hits) == 1, "the commented allocation must not count"
    assert hits[0]["line"] == 4


def test_arch_scope_is_recorded_when_a_decision_sits_in_a_gfx_branch(tmp_path):
    got = scan(tmp_path, "k.py", """
        if arch == "gfx950":
            BLOCK_M = 256
        """)
    hits = [g for g in got if g["category"] == "tile_shape"]
    assert hits, "expected a tile fact"
    assert "gfx950" in (hits[0]["arch_scope"] or ""), hits[0]


def test_arch_scope_is_empty_for_an_unconditional_decision(tmp_path):
    got = scan(tmp_path, "k.py", "BLOCK_M = 128\n")
    hits = [g for g in got if g["category"] == "tile_shape"]
    assert hits and not hits[0]["arch_scope"]


def test_a_scalar_arch_scope_renders_as_one_label_not_six_characters():
    assert RE_.arch_scopes({"arch_scope": "gfx950"}) == ["gfx950"]
    page = "\n".join(RE_.question_section("tile_shape", [{
        "category": "tile_shape", "language": "flydsl", "file": "x.py", "line": 1,
        "captured": ["128"], "match": "BLOCK_M = 128", "arch_scope": "gfx950",
    }]))
    assert "[gfx950]" in page
    assert "[2, 4, 9, f, g, x]" not in page


def test_a_tunable_parameter_is_recorded_separately_from_a_tile_value(tmp_path):
    """The distinction the coverage table depends on: a knob's existence is not its value."""
    got = scan(tmp_path, "k.py", """
        def kernel(
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
        ):
            pass
        """, language="triton")
    assert cats(got) == {"tunable_param"}
    assert {g["captured"][0] for g in got} == {"BLOCK_SIZE_M", "BLOCK_SIZE_N"}


def test_the_asm_launcher_contract_is_captured(tmp_path):
    """What a hand-written assembly kernel publishes is its call shape, not its source."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        struct __attribute__((packed)) KernelArgs
        {
            void* ptr_c;
        };
        int blockSizeX = 256;
        gdx = (Ndim / SUBN) * blockSizeX;
        """, language="asm")
    assert {"asm_argblock", "asm_launch"} <= cats(got)


def test_a_grid_formula_inside_a_format_string_is_not_a_launch_fact(tmp_path):
    """`printf("gdx=%d, gdy=%d")` is a log line, not a grid. Found in the rendered page, where a
    fragment of a format string sat in the asm launch table looking like geometry."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        printf("gdx=%d, gdy=%d, gdz=%d\\n", gdx, gdy, gdz);
        """, language="asm")
    assert not [g for g in got if g["category"] == "asm_launch"], got


def test_a_zero_initialiser_is_not_a_grid_formula(tmp_path):
    """`int gdx = 0;` states no decision. Three of those per launcher crowded out the real formula."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        int gdx = 0;
        gdx = (Ndim / SUBN) * blockSizeX;
        """, language="asm")
    hits = [g for g in got if g["category"] == "asm_launch"]
    assert len(hits) == 1, hits
    assert hits[0]["captured"] == ["(Ndim / SUBN) * blockSizeX"]


def test_ck_records_the_named_pipeline_choice_and_not_the_positional_soup(tmp_path):
    got = scan(tmp_path, "g.cuh", """
        using Instance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
            <Row, Row, 256, 256, 128, 64, 16, 4, 32, 32, 4, 2,
             ck::BlockGemmPipelineScheduler::Interwave,
             ck::BlockGemmPipelineVersion::v1>;
        """, language="ck")
    assert "ck_instance" in cats(got)
    assert {g["captured"][0] for g in got if g["category"] == "ck_pipeline"} == {"Interwave", "v1"}
    # The 40-position argument list must not have been mined for tile evidence: a number at an unnamed
    # position is not a tile until somebody has counted the positions, and that is not this tool.
    assert "tile_shape" not in cats(got)


def test_evidence_is_verbatim_source_not_a_summary(tmp_path):
    body = "acc = rocdl.mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0)\n"
    got = scan(tmp_path, "k.py", body)
    ev = "\n".join(got[0]["evidence"])
    assert "rocdl.mfma_f32_16x16x32_bf16" in ev
    assert re.match(r"^\d+: ", got[0]["evidence"][0]), "excerpt lines carry their line number"


def test_a_missing_subtree_is_reported_not_skipped(tmp_path):
    _, _, _, seen, missing = EX.collect(str(tmp_path))
    assert not seen
    langs = {m["language"] for m in missing}
    assert {l for l, _, _ in EX.IMPLS} <= langs, f"every language must account for itself: {missing}"
    assert all(m["why"] for m in missing), "a gap without a reason is not a gap, it is a hole"


def test_a_declared_pattern_that_matches_nothing_is_reported_even_when_others_match(tmp_path,
                                                                                  monkeypatch):
    """HIP kept one matching file, so the whole `opus_gemm/` tree it was declared to read vanished
    upstream without a line in `missing`."""
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc" / "present.cu").write_text("__shared__ float x[4];\n")
    monkeypatch.setattr(EX, "IMPLS", [("hip", "csrc", ("present.cu", "gone/*.cuh"))])
    evidence, _, _, _, missing = EX.collect(str(tmp_path))
    assert evidence, "the matching pattern still produces evidence"
    assert any("gone/*.cuh" in m["why"] for m in missing), missing


def test_an_mfma_selected_as_a_function_object_is_a_fact_once(tmp_path):
    """The A8 kernel picks `mfma_fn = rocdl.mfma_…` per dtype and calls it later; the call-site rule
    saw none of those choices. A call must still produce exactly one record."""
    got = scan(tmp_path, "k.py", """
        mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
        acc = rocdl.mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0)
        op = getattr(rocdl, "mfma_i32_16x16x32i8", None)
        """)
    hits = sorted((g["line"], g["captured"][0]) for g in got if g["category"] == "mfma_intrinsic")
    assert hits == [(2, "mfma_f32_16x16x32_fp8_fp8"), (3, "mfma_f32_16x16x16f16"),
                    (4, "mfma_i32_16x16x32i8")]


def test_arch_gates_spelled_as_prefix_tests_or_hip_arch_are_recorded(tmp_path):
    got = scan(tmp_path, "k.py", """
        _is_gfx950 = str(gpu_arch).startswith("gfx950")
        if str(get_hip_arch()) != "gfx950":
            raise RuntimeError("x")
        """)
    gates = sorted((g["line"], g["captured"][0]) for g in got if g["category"] == "arch_gate")
    assert gates == [(2, "gfx950"), (3, "gfx950")]


def test_private_constants_tile_catalogs_and_kernel_lists_are_the_declared_space(tmp_path):
    got = scan(tmp_path, "tune.py", """
        _LDS_STAGES = (1, 2)
        _TILE_PRELOAD_TABLE = {
            (16, 64, 256): (2, 2),
        }
        _base_tiles_lds1 = [
            (16, 64, 256),
        ]
        kernels_list_942 = _build(_base_tiles_lds1)
        def f():
            _local_tiles = [1]
        """)
    names = {g["captured"][0] for g in got if g["category"] == "config_space"}
    assert names == {"_LDS_STAGES", "_TILE_PRELOAD_TABLE", "_base_tiles_lds1", "kernels_list_942"}


def test_a_legality_raise_is_recorded_on_its_raise_line(tmp_path):
    got = scan(tmp_path, "k.py", """
        if (tile_k_bytes % 64) != 0:
            raise ValueError(
                f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}"
            )
        raise ValueError("unrelated message")
        """)
    hits = [g for g in got if g["category"] == "config_validity"]
    assert [g["line"] for g in hits] == [3]
    assert "must be divisible by 64" in hits[0]["captured"][0]


def test_split_k_state_grid_mapping_scale_and_gluon_layouts_have_their_own_categories(tmp_path):
    got = scan(tmp_path, "k.py", """
        SPLIT_K_SIGNAL_STATE_COUNT = 3
        def _advance_split_k_signal_state(stream):
            pass
        def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
            pass
        _needs_per_token_scale = not is_fp4
        shared_a: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        """)
    by_cat = {}
    for g in got:
        by_cat.setdefault(g["category"], set()).add(g["captured"][0] if g["captured"] else g["match"])
    assert {"SPLIT_K_SIGNAL_STATE_COUNT", "_advance_split_k_signal_state"} <= by_cat["split_k"]
    assert "remap_xcd" in by_cat["grid_mapping"]
    assert "_needs_per_token_scale" in by_cat["scale_operand"]
    assert "SwizzledSharedLayout" in by_cat["lds_swizzle"]


def test_scale_application_is_recorded_by_where_it_meets_the_accumulator(tmp_path):
    """Block scale multiplies each K block's partial inside the loop; per-token multiplies the
    finished accumulator once. That placement is the math contract, so each has its own category."""
    got = scan(tmp_path, "k.py", """
        for k in range(n):
            accumulator += (
                tl.dot(a, b, input_precision="ieee")
                * a_scale[:, None]
                * b_scale[None, :]
            )
            acc += mfma_out * cur_a_scale[:, None] * cur_b_scale[None, :]
            acc += tl.dot(a, b) * other[:, None]
        accumulator *= a_scale[:, None] * b_scale[None, :]
        """, language="triton")
    inside = sorted((g["line"], g.get("end_line"), tuple(g["captured"]))
                    for g in got if g["category"] == "scale_in_k_loop")
    assert inside == [(3, 7, ("accumulator", "a_scale")), (8, None, ("acc", "cur_a_scale"))]
    after = [(g["line"], g["captured"]) for g in got if g["category"] == "scale_after_k_loop"]
    assert after == [(10, ["accumulator", "a_scale", "b_scale"])]


def test_the_split_k_combine_is_recorded_with_its_extent(tmp_path):
    """The host constants say a combine exists; the kernel lines say how it is written. A reader
    handed only the first line of a 50-line block does not know where it ends."""
    got = scan(tmp_path, "k.py", """
        def zero_c(bias_g, c_g):
            pass

        if const_expr(IS_SPLIT_K):
            split_k_barrier(COUNTER)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                ptr,
                val,
            )
        else:
            store()
        """)
    assert {(g["line"], g.get("end_line")) for g in got if g["category"] == "split_k"} == {
        (2, 3), (5, 13)}
    atomic = [(g["line"], g.get("end_line"), g["captured"])
              for g in got if g["category"] == "atomic_combine"]
    assert atomic == [(7, 11, ["fadd"])]


def test_a_kernel_name_serialiser_is_not_a_scale_operand(tmp_path):
    """`_gemm_a8w8_blockscale_repr = make_kernel_repr(…)` names a kernel and consumes no scale; ten
    of the eleven records this rule first produced were such names."""
    got = scan(tmp_path, "k.py", """
        _gemm_a8w8_blockscale_repr = make_kernel_repr("k", [])
        _needs_per_token_scale = not is_fp4
        if per_token == 1:
            pass
        """, language="triton")
    assert [g["captured"][0] for g in got if g["category"] == "scale_operand"] == [
        "_needs_per_token_scale"]


def test_end_line_is_location_not_identity(tmp_path):
    """An edit deep inside a long function must not invalidate every card citing its first line."""
    got = scan(tmp_path, "k.py", "def split_k_barrier(x):\n    return x\n")
    record = next(g for g in got if g["category"] == "split_k")
    assert record["end_line"] == 2
    without = {k: v for k, v in record.items() if k != "end_line"}
    assert EX.source_evidence_id(without) == record["evidence_id"]
    cuda = scan(tmp_path, "k.cu", "__shared__ float x[4];\n{\n}\n", language="hip")
    assert cuda and all("end_line" not in g for g in cuda)


def test_hand_written_hip_records_its_skinny_gemm_decisions(tmp_path):
    """The skinny family's knobs, its variant choice by activation size, its CU-count fill and its
    in-wave reduction are what a decode author needs from it; opus adds the buffer builtins."""
    got = scan(tmp_path, "k.cu", """
        template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int M>
        __global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitK_hf_(const int K) {
            sum = __shfl_xor(sum, 16);
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, off, 0, 0);
        }
        if((K_in * N_in <= 32 * 1024) && (M_in % _YTILEs == 0))
        {
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEs, _WvPrGrp);
        }
        """, language="hip")
    by_cat = {}
    for g in got:
        by_cat.setdefault(g["category"], []).append(g["captured"])
    assert by_cat["waves"] == [["WvPrGrp* THRDS"]]
    assert "YTILE" in by_cat["tunable_param"][0][0]
    assert by_cat["lane_reduction"] == [["__shfl_xor"]]
    assert by_cat["mfma_intrinsic"] == [["mfma_f32_16x16x16bf16_1k"]]
    assert by_cat["access_width"] == [["load", "b128"]]
    assert by_cat["kernel_family"] == [["K_in * N_in <= 32 * 1024"]]
    assert by_cat["grid_fill"] == [["__wvPrGrp", "mindiv(M_in, CuCount * _YTILEs, _WvPrGrp)"]]


def test_an_asm_round_count_against_the_cus_is_a_grid_fill_record(tmp_path):
    got = scan(tmp_path, "l.cu", """
        uint32_t tg_num = tg_num_M * tg_num_N * split_k;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
        """, language="asm")
    assert [g["captured"] for g in got if g["category"] == "grid_fill"] == [
        ["local_round", "tg_num", "num_cu"]]


def test_lane_mapping_jit_cache_and_test_gate_are_recorded(tmp_path):
    """What a port gets wrong without an error: which lane holds which row, which knobs force a new
    compile, and how far the implementation's own test relaxes where its numerics drift."""
    got = scan(tmp_path, "k.py", """
        ldmatrix_a_m_idx = w_tid % WMMA_M
        lane_div_16_mul4 = lane_div_16 * 4
        layout_lane16 = fx.make_layout((4, 16), (16, 1))

        @functools.lru_cache(maxsize=1024)
        def compile_hgemm(tile_m, tile_n):
            pass

        DEFAULT_PASS_PCT = 99.9
        CASES = [{"split_k": 16, "pass_pct": 99.0, "max_delta_limit": 32.0}]
        """, language="flydsl")
    by = {}
    for g in got:
        by.setdefault(g["category"], []).append(g["captured"][0])
    assert by["fragment_layout"] == ["ldmatrix_a_m_idx", "lane_div_16_mul4", "layout_lane16"]
    assert [g["captured"] for g in got if g["category"] == "jit_cache"] == [["1024", "compile_hgemm"]]
    assert set(by["parity_tolerance"]) == {"DEFAULT_PASS_PCT", "pass_pct", "max_delta_limit"}


def test_an_asm_inventory_is_read_by_header_and_per_architecture(tmp_path):
    """Two column layouts, one meaning. `splitK` is a capability flag — the launchers take a split
    above 1 only for a kernel whose flag is 1 — so it is not recorded as a split factor."""
    for name, body in (
        ("hsa/gfx942/fp8gemm_blockscale/a.csv",
         "tile_m,tile_n,splitK,bpreshuffle,knl_name,co_name\n32,128,1,1,_ZN5k,k.co\n"),
        ("hsa/gfx950/bf16gemm/b.csv",
         "knl_name,co_name,tn,tileM,tileN,pf,bPreshuffle,splitK,subK,bias\n_ZN5j,j.co,1,256,256,0,1,0,64,0\n"),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    a, _, _ = EX.scan_file(str(tmp_path / "hsa/gfx942/fp8gemm_blockscale/a.csv"),
                           "hsa/gfx942/fp8gemm_blockscale/a.csv", "asm")
    b, _, _ = EX.scan_file(str(tmp_path / "hsa/gfx950/bf16gemm/b.csv"),
                           "hsa/gfx950/bf16gemm/b.csv", "asm")
    assert [(r["category"], r["line"], r["arch_scope"]) for r in a + b] == [
        ("asm_tile", 2, "gfx942"), ("asm_tile", 2, "gfx950")]
    assert a[0]["captured"] == ["tile_m=32", "tile_n=128", "split_k_capable=1", "bpreshuffle=1"]
    assert "tile_m=256" in b[0]["captured"] and "split_k_capable=0" in b[0]["captured"]
    assert a[0]["evidence"] == ["1: tile_m,tile_n,splitK,bpreshuffle,knl_name,co_name",
                                "2: 32,128,1,1,_ZN5k,k.co"]


def test_a_routing_statement_is_filed_under_the_backend_it_routes_to(tmp_path):
    """Dispatch is library code, not a language; libraries (`hipblaslt`, `torch`) are not
    languages, and `block` must not read as CK."""
    path = tmp_path / "d.py"
    path.write_text(textwrap.dedent("""
        def is_skinny_default_shape(M, N):
            return M <= 16
        def route(cfg):
            if gfx == "gfx942":
                cfg["libtype"] = "hipblaslt"
            else:
                cfg["libtype"] = "asm"
            if cfg["libtype"] == "cktile":
                return gemm_a8w8_blockscale_cktile(x)
            return blockwise_reduce(x)
        """))
    got, _ = EX.routing_records(str(path), "d.py")
    assert sorted((r["line"], r["language"], r["captured"][0]) for r in got) == [
        (2, "hip", "is_skinny_default_shape"), (8, "asm", "asm"),
        (9, "ck", "cktile"), (10, "ck", "gemm_a8w8_blockscale_cktile")]
    assert all(r["category"] == "kernel_family" and r["arch_scope"] == "" for r in got)
    asm_route = next(r for r in got if r["language"] == "asm")
    assert any('gfx == "gfx942"' in line for line in asm_route["evidence"]), \
        "the excerpt must carry the condition above the route"


def test_both_architectures_have_inventory_and_routing_evidence(source):
    records = source["source_evidence"]
    inventory = {r["arch_scope"] for r in records
                 if r["category"] == "asm_tile" and r["file"].startswith("hsa/")}
    assert {"gfx942", "gfx950"} <= inventory
    routed = {r["file"] for r in records if r["file"] in EX.ROUTING_FILES}
    assert {"aiter/tuned_gemm.py", "aiter/ops/gemm_op_a8w8.py"} <= routed
    assert any(r["language"] == "hip" and r["file"] == "csrc/kernels/custom_kernels.cu"
               for r in records)


def test_the_page_counts_coverage_per_architecture(decisions, source, tuned, tuned_db, axes):
    body = RD.render(decisions, source, tuned, tuned_db, axes)
    section = body.split("## Coverage by architecture", 1)[1].split("\n## ", 1)[0]
    assert "| `gfx942` |" in section and "| `gfx950` |" in section


def test_extents_are_claimed_only_where_a_parser_gave_them(source):
    for record in source["source_evidence"]:
        if "end_line" in record:
            assert record["file"].endswith(".py"), record
            assert int(record["end_line"]) > int(record["line"]), record


def test_a_dispatch_m_padding_is_not_an_lds_pad(tmp_path):
    """`padded_m = 16` is the tuned-lookup granularity. Filed as `lds_pad`, it read as CK padding LDS."""
    got = scan(tmp_path, "g.cu", """
        int padded_m = 16;
        int pad_k = 8;
        """, language="ck")
    assert {g["captured"][0] for g in got if g["category"] == "lds_pad"} == {"pad_k"}
    assert {g["captured"][0] for g in got if g["category"] == "dispatch_padding"} == {"padded_m"}


# --------------------------------------------------------------------------------------------
# AITER's tuned GEMM databases
# --------------------------------------------------------------------------------------------
def test_both_shipped_flydsl_name_forms_decode_to_knobs():
    family, dtype, knobs = EX.decode_flydsl_kernel(
        "flydsl_gemm2_abf16_wbf16_bf16_t32x64x128_split_k8_block_m_warp2_block_n_warp2_"
        "async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950")
    assert (family, dtype) == ("hgemm", "abf16_wbf16_bf16")
    assert knobs["tile"] == "32x64x128" and knobs["split_k"] == 8 and knobs["warps"] == "2x2"
    assert knobs["b_to_lds"] is True and knobs["b_preshuffle"] is False
    family, dtype, knobs = EX.decode_flydsl_kernel(
        "flydsl_bpreshuflle_128x256x128_F8_F8_B16_2x1x1x2_default")
    assert (family, dtype) == ("a8_preshuffle", "af8_wf8_b16")
    assert knobs == {"tile": "128x256x128", "lds_stage": 2, "use_cshuffle_epilog": 1,
                     "use_async_copy": 1, "waves_per_eu": 2, "scheduler": "default"}
    assert EX.decode_flydsl_kernel("_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_16x128E") is None


def test_tuned_db_rows_keep_the_selection_and_drop_the_timings(tmp_path):
    db = tmp_path / "aiter" / "configs"
    db.mkdir(parents=True)
    (db / "bf16_tuned_gemm.csv").write_text(
        "gfx,cu_num,M,N,K,bias,libtype,us,kernelName,tflops,err_ratio\n"
        "gfx950,256,8,4096,4096,False,flydsl,12.5,"
        "flydsl_gemm2_abf16_wbf16_bf16_t32x64x128_split_k8_block_m_warp2_block_n_warp2_"
        "async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950,99.0,0.0\n"
        "gfx950,256,4096,4096,4096,False,asm,40.0,_ZN5aiter3asmE,900.0,0.0\n")
    (db / "a8w8_tuned_gemm.csv").write_text("gfx,cu_num,M,N,K,us,kernelName\ngfx950,256,1,1,1,1.0,k\n")
    rows, inventory, gaps = EX.tuned_db_rows(str(tmp_path))
    assert len(rows) == 3 and len(inventory) == 2
    flat = json.dumps(rows)
    assert "12.5" not in flat and "tflops" not in flat and '"us"' not in flat
    assert any("libtype" in g["why"] for g in gaps), "a DB without a backend column is a stated gap"
    groups, _ = EX.summarize_flydsl_tuned(rows)
    assert len(groups) == 1 and groups[0]["m_bucket"] == "M<=16"
    assert groups[0]["knobs"]["split_k"] == 8
    selection = EX.summarize_backend_selection(rows)
    assert {s["m_bucket"]: s["selected_backend_rows"] for s in selection} == {
        "M<=16": {"flydsl": 1}, "M1025-4096": {"asm": 1}}


def test_the_committed_tuned_db_carries_no_timing_and_unique_ids(tuned_db):
    text = open(TUNED_DB, encoding="utf-8").read()
    for column in EX.TUNED_DB_TIMING:
        assert not re.search(rf"^\s+{re.escape(column)}:", text, re.MULTILINE), column
    ids = [g["config_id"] for g in tuned_db["tuned_configs"]]
    ids += [s["selection_id"] for s in tuned_db["backend_selection"]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"(cfg|sel)_[0-9a-f]{16}", i) for i in ids)
    for group in tuned_db["tuned_configs"]:
        assert group["source_files"] and group["m_bucket"] in dict(
            (label, None) for _, label in EX.M_BUCKETS), group


# --------------------------------------------------------------------------------------------
# Filename parsing for the shipped sweep results
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,gfx,tags,shape", [
    ("gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json", "gfx942",
     ["GEMM", "A8W8_BLOCKSCALE"], {"n": 1024, "k": 8192}),
    ("gfx1250-FUSED-GEMM-AFP4WFP4-A16W16.json", "gfx1250",
     ["FUSED", "GEMM", "AFP4WFP4", "A16W16"], {}),
    ("gfx1250-FF-A16W16-fused.json", "gfx1250", ["FF", "A16W16", "fused"], {}),
    ("gfx950-GEMM-A16W16-ATOMIC.json", "gfx950", ["GEMM", "A16W16", "ATOMIC"], {}),
])
def test_tuned_names_split_into_gfx_tags_and_shape(name, gfx, tags, shape):
    """A splitter, not a grammar. An earlier regex insisting on kernel-then-variant dropped 239 of
    257 files into an empty-field bucket, and an empty field is indistinguishable from a real one."""
    got = EX.parse_tuned_name(name)
    assert got == {"gfx": gfx, "tags": tags, "shape": shape}


def test_a_name_without_a_gfx_prefix_is_refused_rather_than_guessed():
    assert EX.parse_tuned_name("GEMM-A8W8.json") is None


def test_a_knob_constant_across_shapes_is_separated_from_one_that_moves():
    rows = [
        {"file": "a.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 4}},
        {"file": "b.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 8}},
    ]
    g = EX.summarize_tuned(rows)
    assert len(g) == 1
    assert g[0]["same_across_configs"] == {"BLOCK_SIZE_K": 128}
    assert "num_warps" in g[0]["varies_by_shape"]
    assert g[0]["shape_configs"] == 2
    assert g[0]["source_files"] == ["a.json", "b.json"]


def test_a_single_shipped_shape_config_does_not_claim_agreement():
    """`same_across_configs` over one row would assert an invariant on evidence that cannot show it."""
    g = EX.summarize_tuned([{"file": "a.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {},
                             "m_bucket": "any", "knobs": {"BLOCK_SIZE_K": 64}}])
    assert g[0]["shape_configs"] == 1
    assert "same_across_configs" not in g[0]
    assert g[0]["knobs"] == {"BLOCK_SIZE_K": 64}


def test_varying_values_render_as_value_colon_count():
    """`.cgx23` reads as a cache modifier named `.cgx23`; the count has to be unambiguous."""
    g = EX.summarize_tuned([
        {"gfx": "g", "tags": [], "shape": {}, "m_bucket": "b", "knobs": {"cache_modifier": ".cg"}},
        {"gfx": "g", "tags": [], "shape": {}, "m_bucket": "b", "knobs": {"cache_modifier": None}},
    ])
    assert ": " in g[0]["varies_by_shape"]["cache_modifier"]


# --------------------------------------------------------------------------------------------
# The YAML round trip
# --------------------------------------------------------------------------------------------
def test_the_fallback_parser_agrees_with_pyyaml_on_the_real_files():
    """Two parse paths that disagree is worse than one that is absent: `--check` would pass or fail
    depending on whether PyYAML happened to be installed."""
    yaml = pytest.importorskip("yaml")
    for path in (FAMILY, TUNED, TUNED_DB, DECISIONS):
        if not os.path.exists(path):
            pytest.skip(f"{path} not generated")
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) == RE_._mini_yaml(path), f"parsers diverge on {path}"


def test_a_scalar_containing_a_newline_stays_on_one_line():
    """A CK template head matches across a line break. Emitted raw it put a second physical line
    inside a quoted scalar, which PyYAML folds back but which breaks the one-record-per-line property
    the whole hand-rolled format rests on, and misattributes the diff."""
    assert EX.yaml_quote("DeviceGemm\n<") == '"DeviceGemm\\n<"'
    assert "\n" not in EX.yaml_quote("a\tb\r\nc")


def test_escapes_survive_the_round_trip():
    for original in ('a"b', "a\\b", "a\nb", "a\\nb", "a\tb", 'q"\\n'):
        assert RE_._unquote(EX.yaml_quote(original)) == original, original


# --------------------------------------------------------------------------------------------
# The committed artifacts
# --------------------------------------------------------------------------------------------
def test_every_source_record_uses_the_language_vocabulary_kb_uses(source):
    """A corpus keyed on its own language names cannot be joined to a learned card, which is the
    entire point of putting `language` on cards in the first place."""
    kb = _load("kb", os.path.join(ROOT, "kernel_workflow", "scripts", "kb.py"))
    vocab = set(kb.AUTHORING_LANGUAGES)
    used = {e["language"] for e in source["source_evidence"]}
    assert used <= vocab, f"not in kb.AUTHORING_LANGUAGES: {sorted(used - vocab)}"
    assert {l for l, _, _ in EX.IMPLS} <= vocab, "a declared implementation language must be known"


def test_every_source_record_points_at_a_file_and_a_line(source):
    for record in source["source_evidence"]:
        assert record["file"] and not os.path.isabs(record["file"]), record
        assert isinstance(record["line"], int) and record["line"] > 0, record
        assert record["evidence"], f"source evidence with no verbatim excerpt: {record}"


def test_every_source_record_has_a_unique_content_bound_id(source):
    records = source["source_evidence"]
    ids = [record.get("evidence_id") for record in records]
    assert all(re.fullmatch(r"src_[0-9a-f]{16}", str(value)) for value in ids)
    assert len(ids) == len(set(ids))
    assert all(record["evidence_id"] == EX.source_evidence_id(record) for record in records)


def test_no_source_record_carries_a_performance_number(source):
    """A timing in source evidence is a measurement with no machine
    attached, which is the shape of an unfalsifiable claim."""
    banned = re.compile(r"\b\d+(?:\.\d+)?\s*(?:us|ms|ns|µs|TFLOPS|tflops|GB/s|gbps)\b|"
                        r"\b(?:speedup|faster|slower|x\s*speedup)\b", re.IGNORECASE)
    for record in source["source_evidence"]:
        blob = f"{record.get('match', '')} {' '.join(record.get('evidence') or [])}"
        assert not banned.search(blob), (
            f"performance language in source evidence: {record['file']}:{record['line']}"
        )


def test_provenance_names_the_commit_the_source_was_read_at(source):
    prov = source["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", str(prov["aiter_commit"])), prov
    assert prov["aiter_commit"] != "unknown", "source evidence without a commit has no expiry date"


def test_the_committed_corpus_resolves_at_the_commit_it_names(source):
    """The corpus's one promise, checked rather than assumed.

    An earlier version asserted only that a commit was PRESENT, and passed while 56 records pointed
    into a locally-modified kernel that the named commit does not contain — `file:line` resolved for
    the person who ran the extractor and nobody else. Present and identifying are different
    properties, and only the second one is worth anything in a citation index.
    """
    dirt = source["provenance"].get("aiter_dirty_sources")
    assert dirt == [] or dirt is None or not dirt, (
        f"extracted from a tree with uncommitted changes in {dirt}; re-extract from a clean "
        f"checkout (`git -C <aiter> worktree add /tmp/aiter-clean <commit>`)")
    assert not [e for e in source["source_evidence"] if e.get("unreproducible")], \
        "the committed corpus must not contain records flagged unreproducible"


def test_extraction_refuses_a_dirty_source_by_default(tmp_path, monkeypatch):
    """The refusal, not just the label. A marker on an artifact nobody re-reads is a footnote."""
    calls = {}

    def fake_dirty(path):
        calls["asked"] = path
        return {"aiter/ops/flydsl/kernels/splitk_hgemm.py"}

    monkeypatch.setattr(EX, "dirty_paths", fake_dirty)
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "aiter/ops/flydsl/kernels/splitk_hgemm.py", "line": 1, "language": "flydsl",
          "category": "mfma_intrinsic", "match": "x", "captured": [], "arch_scope": "",
          "evidence": ["1: x"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--emit"])
    assert EX.main() == 3, "a dirty contributing file must abort the emit"
    assert calls["asked"] == str(tmp_path)


def test_allow_dirty_records_but_marks_every_affected_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(EX, "dirty_paths", lambda a: {"dirty.py"})
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "dirty.py", "line": 1, "language": "flydsl", "category": "tile_shape",
          "match": "x", "captured": [], "arch_scope": "", "evidence": ["1: x"]},
         {"file": "clean.py", "line": 2, "language": "flydsl", "category": "tile_shape",
          "match": "y", "captured": [], "arch_scope": "", "evidence": ["2: y"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--json", "--allow-dirty"])
    assert EX.main() == 0
    payload = json.loads(capsys.readouterr().out)
    got = {e["file"]: e.get("unreproducible")
           for e in payload["evidence"]["source_evidence"]}
    assert got == {"dirty.py": True, "clean.py": None}, got
    assert payload["evidence"]["provenance"]["aiter_dirty_sources"] == ["dirty.py"]


def test_an_unrelated_dirty_file_does_not_block_extraction(tmp_path, monkeypatch, capsys):
    """Scoped to contributors. Refusing on any edit anywhere in aiter would be superstition — an
    unrelated change cannot make a citation unresolvable."""
    monkeypatch.setattr(EX, "dirty_paths", lambda a: {"docs/README.md", "aiter/unrelated.py"})
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "clean.py", "line": 1, "language": "flydsl", "category": "tile_shape",
          "match": "x", "captured": [], "arch_scope": "", "evidence": ["1: x"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--json"])
    assert EX.main() == 0
    assert json.loads(capsys.readouterr().out)["evidence"]["provenance"]["aiter_dirty_sources"] == []


def test_dirty_paths_parses_renames_and_quoted_names():
    """`XY old -> new`: the new name is what was read. A parser that kept the old one would clear a
    file that is in fact modified."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import subprocess, importlib.util, sys
        spec = importlib.util.spec_from_file_location("x", sys.argv[1])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        class R:
            stdout = ' M a/b.py\\nR  old/x.py -> new/y.py\\n?? "sp ace.py"\\n'
        subprocess.run = lambda *a, **k: R()
        print(sorted(m.dirty_paths("/nowhere")))
        """), os.path.join(HERE, "_extract_impl_facts.py")],
        capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "['a/b.py', 'new/y.py', 'sp ace.py']", out.stdout


def test_gaps_are_stated_with_a_reason(source):
    for m in source.get("missing") or []:
        assert m.get("language") and m.get("subtree") and m.get("why"), m


def test_every_language_declared_actually_produced_source_evidence(source):
    """A language present in `IMPLS` but absent from the output means the globs no longer match the
    upstream layout — which is how `ck` sat at zero while looking like a deliberate omission."""
    declared = {l for l, _, _ in EX.IMPLS}
    got = {e["language"] for e in source["source_evidence"]}
    assert declared == got, f"declared but empty: {sorted(declared - got)}"


def test_the_rendered_source_page_matches_the_evidence():
    if not os.path.exists(SOURCE_DOC):
        pytest.skip("doc not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py"), "--check"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"stale doc; re-run --emit\n{r.stdout}{r.stderr}"


def test_the_source_page_states_no_ranking(source):
    """The prose is hand-written, so it is the one place a claim could slip in. Checked as text."""
    if not os.path.exists(SOURCE_DOC):
        pytest.skip("doc not generated")
    with open(SOURCE_DOC, encoding="utf-8") as f:
        body = f.read()
    for phrase in ("is faster", "is better", "outperforms", "you should use", "the best "):
        assert phrase not in body.lower(), f"the page ranks: {phrase!r}"


def test_curated_decision_cards_are_actionable_and_grounded(decisions, source, tuned, tuned_db, axes):
    cards = decisions.get("cards") or []
    assert cards, "an evidence index without decision cards still leaves the author to infer the answer"
    assert RD.validate(cards, source["source_evidence"]) == []
    shipped = RD.shipped_index(tuned, tuned_db)
    assert RD.validate(cards, source["source_evidence"], RD.axis_ids_of(axes), shipped) == []
    for card in cards:
        assert card["conditions"], card
        assert card["actions"], card
        assert card["why"], card
        assert card["limitations"], card


def test_always_on_corpus_does_not_bypass_measured_knowledge_switches(decisions):
    levels = {card["evidence_level"] for card in decisions["cards"]}
    assert levels == {"source_observed"}
    for card in decisions["cards"]:
        assert card["source_evidence"] and not card["measurement_evidence"], card


def test_a_loose_file_line_reference_is_rejected(decisions, source):
    card = dict(decisions["cards"][0], source_evidence=["x.py:1"])
    errors = RD.validate([card], source["source_evidence"])
    assert any("content-bound" in error for error in errors)


def test_development_order_places_every_axis_once_and_every_card_somewhere(decisions, axes):
    """The page is ordered by `development_order`; a card whose first axis sat in no step would not
    be rendered at all."""
    steps, step_of, errors = RD.development_steps(axes)
    assert errors == []
    assert [step["step"] for step in steps] == list(range(1, len(steps) + 1))
    for card in decisions["cards"]:
        assert card["axes"][0] in step_of, card["id"]


def test_a_broken_development_order_is_refused(axes):
    moved = dict(axes, development_order=[dict(s) for s in axes["development_order"]])
    moved["development_order"][2]["axes"] = []
    assert any("axes in no step" in e for e in RD.development_steps(moved)[2])
    twice = dict(axes, development_order=[dict(s) for s in axes["development_order"]])
    twice["development_order"][0]["axes"] = ["workgroup_tile"]
    assert any("is in steps" in e for e in RD.development_steps(twice)[2])


def test_traits_are_defined_and_every_required_trait_is_one_of_them(decisions):
    """An undefined trait defers its card forever: no planner can know to pass it."""
    traits = decisions.get("traits") or {}
    assert RD.validate_traits(traits) == []
    required = {t for card in decisions["cards"] for t in card["match"].get("requires") or []}
    assert required and required <= set(traits), required - set(traits)


def test_the_page_follows_the_development_order_and_states_empty_steps(decisions, source, tuned,
                                                                        tuned_db, axes):
    body = RD.render(decisions, source, tuned, tuned_db, axes)
    headings = [line for line in body.splitlines() if re.match(r"^## \d+\. ", line)]
    assert [int(h.split()[1].rstrip(".")) for h in headings] == list(
        range(1, len(axes["development_order"]) + 1))
    assert "## Design traits" in body and "No card answers this step yet." in body


def test_new_card_fields_are_checked_not_trusted(decisions, source, tuned, tuned_db, axes):
    evidence = source["source_evidence"]
    base = dict(decisions["cards"][0])
    shipped = RD.shipped_index(tuned, tuned_db)
    axis_ids = RD.axis_ids_of(axes)
    flydsl_ref = base["source_evidence"][0]

    def errors_for(**changes):
        return RD.validate([dict(base, **changes)], evidence, axis_ids, shipped)

    assert any("unknown fields" in e for e in errors_for(problem="misc"))
    assert any("unknown performance axis" in e for e in errors_for(axes=["tile_magic"]))
    assert any("undefined trait" in e for e in RD.validate(
        [dict(base, match=dict(base["match"], requires=["no_such_trait"]))],
        evidence, axis_ids, shipped, decisions["traits"]))
    assert any("not triton" in e for e in errors_for(solutions=[
        {"backend": "triton", "mechanism": "x", "source_evidence": [flydsl_ref]}]))
    assert any("does not name a file" in e for e in errors_for(references=["hardware/nope.md"]))
    assert any("outside the file" in e for e in errors_for(
        references=["hardware/shared/dtype_numerics.md:1-99999"]))
    assert any("not present in tuned evidence" in e for e in errors_for(
        shipped_evidence=["cfg_0000000000000000"]))
    assert any("exact dtype" in e for e in errors_for(match=dict(base["match"], dtypes=["fp8*"])))


def test_the_rendered_pages_match_their_inputs():
    if not os.path.exists(DECISION_DOC):
        pytest.skip("decision doc not generated")
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "_render_decisions.py"), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"stale decision doc; re-run _render_decisions.py --emit\n{result.stdout}{result.stderr}"
    )


def test_shipped_configs_become_seed_candidate_and_vary_next_advice(decisions, source, tuned,
                                                                    tuned_db, axes):
    body = RD.render(decisions, source, tuned, tuned_db, axes)
    assert "## Shipped configuration seeds: FlyDSL" in body
    assert "seed candidate" in body.lower() and "vary next" in body
    assert "not measured winners" in body
    triton = RD.render_triton_page(decisions, source, tuned, tuned_db, axes)
    assert "## Shipped configuration seeds: Triton" in triton and "not measured winners" in triton
    # The Triton knob tables live on their own page; the decision page links to them.
    assert "BLOCK_SIZE_M=" not in body and "gemm_triton_seeds.md" in body


def test_tuned_groups_carry_arch_and_bucket(tuned):
    """A knob set without its gfx and its M bucket reads as a global recommendation. It is not one."""
    ids = []
    for g in tuned["tuned_configs"]:
        assert re.fullmatch(r"cfg_[0-9a-f]{16}", str(g["config_id"])), g
        ids.append(g["config_id"])
        assert str(g["gfx"]).startswith("gfx"), g
        assert g["m_bucket"], g
        assert int(g["shape_configs"]) >= 1, g
        assert g["source_files"], g
        assert ("same_across_configs" in g) != ("knobs" in g), \
            f"exactly one of the two forms, chosen by shape_configs: {g}"
    assert len(ids) == len(set(ids)), "config IDs must identify exactly one (gfx, variant, M bucket)"


def test_the_extractor_refuses_to_run_without_a_source_tree():
    """`--emit` with no `--aiter` would write an empty corpus over a good one."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "_extract_impl_facts.py"), "--emit"],
                       capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert "--aiter" in (r.stderr + r.stdout)


def test_the_extractor_requires_an_explicit_output_mode(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "_extract_impl_facts.py"), "--aiter", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--emit" in (result.stderr + result.stdout)
    assert "--json" in (result.stderr + result.stdout)


def test_the_renderer_runs_without_an_aiter_checkout():
    """The property that makes `--check` a usable gate: rendering reads only committed YAML."""
    if not os.path.exists(FAMILY):
        pytest.skip("source evidence not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py")],
                       capture_output=True, text=True, cwd="/", check=False)
    assert r.returncode == 0, r.stderr
    assert "## Source-evidence coverage" in r.stdout


def test_json_mode_is_machine_readable():
    if not os.path.exists(FAMILY):
        pytest.skip("source evidence not generated")
    aiter = os.environ.get("AITER_PATH", "/sgl-workspace/aiter")
    if not os.path.isdir(aiter):
        pytest.skip("no aiter checkout")
    # `--allow-dirty` because this asserts the OUTPUT PARSES, and whether the source tree is clean has
    # nothing to do with that. Without it the test asserts reproducibility twice and fails on any
    # developer box with a modified aiter, which teaches people to skip it.
    r = subprocess.run([sys.executable, os.path.join(HERE, "_extract_impl_facts.py"),
                        "--aiter", aiter, "--json", "--allow-dirty"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["evidence"]["source_evidence"] and payload["tuned"]["tuned_configs"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
