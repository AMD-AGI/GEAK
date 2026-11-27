#!/usr/bin/env python3
"""Coverage for tools/ep_abi.py, the frozen-contract check numeric-oracle tasks
run before building a candidate: which edits it rejects (signatures, dispatch
gates, constants and the variables gates read, ABI includes, macros on names
gates use, environment writes, exports) and which it must let through (kernel
bodies, comments, whitespace).

    python3 -m pytest tools/tests/test_ep_abi.py
"""
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ep_abi  # noqa: E402

KERNEL = r'''// Copyright
#include "hip_custom_kernels.h"
#include <hip/hip_runtime.h>

static constexpr int kBlockDim = 16;
#define GEMV_WAVES(TM) ((TM) > 1 ? 16 : 9)

template <int BLOCK_SIZE, int TILE_N, int TILE_M = 1>
__global__ __launch_bounds__(BLOCK_SIZE)
__attribute__((amdgpu_waves_per_eu(1, GEMV_WAVES(TILE_M))))
void gemv_kernel(const float* a, float* y, int n) {
    // body: the optimizer's surface
    const char* s = "}{ not code";
    const char c = '}';
    if (n > 0) { y[0] = a[0]; }
}

static bool wmmaEnabled() {
    static const bool on = true;   /* cached */
    return on;
}

extern "C" {
int hip_block_a(int x);
}

extern "C" const int kTable[2] = {1, 2};

extern "C" int hip_k(void* stream, const float* a, float* y,
                     int M, int n) {
    if (M <= kBlockDim && wmmaEnabled()) {
        hipLaunchKernelGGL(gemv_kernel<128, 4>, dim3(1), dim3(128), 0, 0, a, y, n);
        return 0;
    }
    while (wmmaEnabled()) { break; }
    return 1;
}

extern "C" int hip_k_supported(int M) { return M > 0; }
'''


def _manifest(src=KERNEL):
    return ep_abi.manifest(src, functions=["hip_k", "wmmaEnabled"], constants=["kBlockDim"])


# -- lexing ------------------------------------------------------------------

def test_strip_comments_keeps_offsets_and_blanks_comments():
    src = 'a // x\nb /* y\nz */ c "q // not" \'"\' R"(r)" d'
    code, mask = ep_abi.strip_comments(src)
    assert len(code) == len(mask) == len(src)
    assert code.count("\n") == src.count("\n")
    assert "x" not in code and "y" not in code and "z" not in code
    assert '"q // not"' in code              # strings survive in code ...
    assert "q" not in mask and "not" not in mask  # ... but not in the mask
    assert mask.rstrip().endswith("d")


def test_normalize_ignores_whitespace_only():
    assert ep_abi.normalize("if ( a <= b )\n {") == ep_abi.normalize("if(a<=b){")
    assert ep_abi.normalize("a <= b") != ep_abi.normalize("a < b")


def test_digit_separators_are_not_char_literals():
    src = "int a = 1'000;\nint b = 0x1'F;\nchar c = u8'a'; char d = L'}';\nint g() { return 0; }\n"
    _, mask = ep_abi.strip_comments(src)
    assert "1'000;" in mask and "0x1'F;" in mask
    assert "}" not in mask[mask.index("L'"):mask.index("int g")]
    assert len(ep_abi.function_definitions(src, "g")) == 1


def test_a_digit_separator_before_a_gate_keeps_the_gate():
    src = KERNEL.replace("static bool wmmaEnabled() {",
                         "static constexpr int kSep = 1'000;\nstatic bool wmmaEnabled() {")
    assert ep_abi.check_source(src, _manifest()) == []


# -- definitions -------------------------------------------------------------

@pytest.mark.parametrize("name", ["gemv_kernel", "wmmaEnabled", "hip_k", "hip_k_supported"])
def test_function_definitions_finds_each_definition_once(name):
    spans = ep_abi.function_definitions(KERNEL, name)
    assert len(spans) == 1
    start, end = spans[0]
    assert KERNEL[end] == "}"
    assert name in KERNEL[start:end]


def test_template_and_attribute_heads_are_part_of_the_definition():
    (start, _), = ep_abi.function_definitions(KERNEL, "gemv_kernel")
    assert KERNEL[start:].lstrip().startswith("template <int BLOCK_SIZE")


def test_multiline_macros_do_not_leak_into_the_next_definition():
    src = ("#define WG(A, B) \\\n    ((A) * (B) == 4)\n#define OPEN {\n"
           "template <int N = 2>\n__global__ __attribute__((amdgpu_flat_work_group_size(\n"
           "    WG(N, 2), WG(N, 2))))\nvoid k(float* y) { y[0] = 1.0f; }\n")
    (start, end), = ep_abi.function_definitions(src, "k")
    assert src[start:end + 1].lstrip().startswith("template <int N = 2>")
    assert src.endswith(src[start:end + 1] + "\n")


def test_calls_and_prototypes_are_not_definitions():
    src = "int f(int);\nint g() { return f(1); }\nint h() { while (f(2)) { } return 0; }\n"
    assert ep_abi.function_definitions(src, "f") == []


@pytest.mark.parametrize("src", [
    "int f() const noexcept { }",
    "int f() && { }",
    "auto f() -> int { }",
    "void f(int x) __attribute__((foo(1))) { }",
])
def test_qualifiers_attributes_and_trailing_returns_before_the_body(src):
    assert ep_abi.function_definitions(src, "f") == [(0, len(src) - 1)]


def test_an_attribute_after_a_prototype_does_not_reach_a_later_body():
    src = "void f(int x) __attribute__((x));\nint g() __attribute__((y)) { return 0; }\n"
    assert ep_abi.function_definitions(src, "f") == []
    assert len(ep_abi.function_definitions(src, "g")) == 1


# A single regex of repeated, overlapping alternatives backtracks exponentially
# on each of these (CodeQL py/redos).
@pytest.mark.parametrize("tail", [
    "&&" * 5000,
    "__attribute__((" + "))__attribute__((" * 5000,
    "->" + "\t&->" * 5000,
])
def test_long_runs_after_the_parameters_are_rejected_quickly(tail):
    assert ep_abi.function_definitions(f"int f() {tail} x", "f") == []


def test_function_hash_ignores_comments_and_whitespace():
    edited = KERNEL.replace("static const bool on = true;   /* cached */",
                            "static const bool on =   true; // cached")
    assert ep_abi.function_hash(edited, "wmmaEnabled") == ep_abi.function_hash(KERNEL, "wmmaEnabled")
    flipped = KERNEL.replace("static const bool on = true;", "static const bool on = false;")
    assert ep_abi.function_hash(flipped, "wmmaEnabled") != ep_abi.function_hash(KERNEL, "wmmaEnabled")


def test_extern_c_definitions_skip_data_prototypes_and_blocks():
    defs = ep_abi.extern_c_definitions(KERNEL)
    assert set(defs) == {"hip_k", "hip_k_supported"}
    assert defs["hip_k"].startswith('extern"C"int hip_k(void*stream')
    assert defs["hip_k"].endswith("int n)")


def test_constant_declaration_and_includes():
    assert ep_abi.constant_declaration(KERNEL, "kBlockDim") == "static constexpr int kBlockDim=16;"
    assert ep_abi.constant_declaration(KERNEL, "on") is None  # a function-local static
    assert ep_abi.quoted_includes(KERNEL) == ["hip_custom_kernels.h"]


def test_namespace_constants_skip_locals_members_parameters_and_arrays():
    src = """namespace detail {
static constexpr int kInNs = 1;
}
extern "C" {
const int kInBlock = 2;
}
void g(const int n = 4) { }
static constexpr int kAfterParam = 3;
static int f(int n) {
    constexpr int kLocal = 3;
    return n + kLocal;
}
struct S { static constexpr int kMember = 5; };
static constexpr bool kEq = kInNs == 1;
__device__ constexpr float kEps = 1e-5f;
static constexpr int kArr[] = {1, 2};
"""
    assert ep_abi.namespace_constants(src) == {
        "kInNs": "static constexpr int kInNs=1;",
        "kInBlock": "const int kInBlock=2;",
        "kAfterParam": "static constexpr int kAfterParam=3;",
        "kEq": "static constexpr bool kEq=kInNs==1;",
        "kEps": "constexpr float kEps=1e-5f;",
    }


def test_namespace_variables_cover_tables_state_and_objects():
    src = """namespace detail {
static int g_count = 0;
}
static constexpr int kArr[] = {1, 2};
static std::unordered_map<std::string, int> g_cache;
extern int g_ext;
int g_ext = 3;
__constant__ float c_tab[4][2];
using Alias = int;
struct S { int member = 1; } ;
template <int N = 2> void t();
void g(int n = 4) { static int local = 0; }
int S::shared = 5;
static auto f = [](int x) { return x; };
"""
    got = ep_abi.namespace_variables(src)
    assert set(got) == {"g_count", "kArr", "g_cache", "g_ext", "c_tab", "f"}
    assert got["kArr"] == ("static constexpr int kArr[]={1,2};", False)
    assert got["g_ext"] == ("extern int g_ext; int g_ext=3;", True)
    assert got["g_cache"][1] and got["c_tab"][1] and got["g_count"][1]


# -- manifest ----------------------------------------------------------------

def test_manifest_records_the_contract():
    man = _manifest()
    assert set(man["extern_c"]) == {"hip_k", "hip_k_supported"}
    assert set(man["functions"]) == {"hip_k", "wmmaEnabled"}
    assert man["constants"] == {"kBlockDim": "static constexpr int kBlockDim=16;"}
    assert man["required_includes"] == ["hip_custom_kernels.h"]
    assert man["exports"] is None


_READS_CONSTANT = KERNEL.replace(
    "static bool wmmaEnabled() {",
    "static constexpr int kUnused = 7;\nstatic constexpr int kWmmaMin = 8;\n"
    "static bool wmmaEnabled() {").replace("    return on;\n}", "    return on && kWmmaMin > 0;\n}")


def test_manifest_freezes_the_constants_frozen_functions_read():
    man = ep_abi.manifest(_READS_CONSTANT, functions=["hip_k", "wmmaEnabled"])
    assert man["constants"] == {"kBlockDim": "static constexpr int kBlockDim=16;",
                                "kWmmaMin": "static constexpr int kWmmaMin=8;"}
    assert ep_abi.check_source(_READS_CONSTANT.replace("kUnused = 7", "kUnused = 9"), man) == []
    violations = ep_abi.check_source(_READS_CONSTANT.replace("kWmmaMin = 8", "kWmmaMin = 1"), man)
    assert violations == ["frozen constant kWmmaMin changed"]


@pytest.mark.parametrize("kw,match", [
    ({"functions": ["no_such_gate"]}, "no_such_gate"),
    ({"constants": ["kNope"]}, "kNope"),
    ({"required_includes": ["missing.h"]}, "missing.h"),
])
def test_manifest_rejects_missing_items(kw, match):
    with pytest.raises(ValueError, match=match):
        ep_abi.manifest(KERNEL, **kw)


# -- check_source: what is editable vs frozen --------------------------------

@pytest.mark.parametrize("edit", [
    ("if (n > 0) { y[0] = a[0]; }", "for (int i = 0; i < n; ++i) { y[i] = 2 * a[i]; }"),
    ("__global__ __launch_bounds__(BLOCK_SIZE)", "__global__ __launch_bounds__(256)"),
    ("// Copyright", "// Copyright\n// tuned by GEAK"),
    ("    return 1;\n}", "    return   1;   // fallthrough\n}"),
])
def test_body_comment_and_whitespace_edits_pass(edit):
    assert ep_abi.check_source(KERNEL.replace(*edit), _manifest()) == []


def test_new_device_helper_passes():
    src = KERNEL.replace("static bool wmmaEnabled", "__device__ float helper(float x) { return x; }\n"
                         "static bool wmmaEnabled")
    assert ep_abi.check_source(src, _manifest()) == []


@pytest.mark.parametrize("edit,match", [
    (("int M, int n) {", "int64_t M, int n) {"), "hip_k changed its signature"),
    (("if (M <= kBlockDim && wmmaEnabled())", "if (M <= 2 * kBlockDim && wmmaEnabled())"),
     "dispatch gate hip_k()"),
    (("static const bool on = true;", "static const bool on = false;"), "dispatch gate wmmaEnabled()"),
    (("kBlockDim = 16;", "kBlockDim = 32;"), "frozen constant kBlockDim"),
    (('#include "hip_custom_kernels.h"\n', ""), "hip_custom_kernels.h"),
    (('extern "C" int hip_k_supported(int M) { return M > 0; }', ""), "hip_k_supported was removed"),
    (("int hip_block_a(int x);", "int hip_block_a(int x);\nint hip_block_b(int x);"),
     'extern "C" { ... } blocks changed'),
])
def test_frozen_edits_are_rejected(edit, match):
    violations = ep_abi.check_source(KERNEL.replace(*edit), _manifest())
    assert any(match in v for v in violations), violations


def test_new_extern_c_is_rejected():
    src = KERNEL + '\nextern "C" int hip_k_v2(int x) { return x; }\n'
    assert any("hip_k_v2 is new" in v for v in ep_abi.check_source(src, _manifest()))


def test_removed_gate_is_rejected():
    src = KERNEL.replace("static bool wmmaEnabled() {", "static bool wmmaOn() {")
    assert any("wmmaEnabled" in v for v in ep_abi.check_source(src, _manifest()))


@pytest.mark.parametrize("line", ["#define hip_k hip_k_fast", "#undef kBlockDim",
                                  "#  define wmmaEnabled() true"])
def test_macro_shadowing_a_frozen_name_is_rejected(line):
    src = KERNEL.replace("#include <hip/hip_runtime.h>", "#include <hip/hip_runtime.h>\n" + line)
    assert any("#define/#undef" in v for v in ep_abi.check_source(src, _manifest()))


# -- check_source: what frozen functions read --------------------------------

# Gates that read the environment, a latch, a config table and a cap constant,
# and an editable launcher that shares one counter with them.
GATED = r'''#include "hip_custom_kernels.h"
#include <cstdlib>
#include <mutex>

struct Cfg { int bm; bool fused; };
static constexpr Cfg kCfgs[] = {
    {64, true},
    {32, false},
};
static constexpr size_t kPadDefault = 256;
static int g_lookup = -1;   // latched on first read
static std::mutex g_lookup_mutex;
static unsigned char* g_buf = nullptr;
static int g_launches = 0;

static int minMOverride() {
    const char* v = getenv("MIN_M");
    return v ? atoi(v) : 8;
}

static size_t padMaxBytes() {
    const char* v = getenv("PAD_MAX");
    if (!v) return kPadDefault;
    return static_cast<size_t>(atoi(v));
}

static bool lookupEnabled() {
    std::lock_guard<std::mutex> lock(g_lookup_mutex);
    if (g_lookup < 0)
        g_lookup = getenv("LOOKUP") ? 1 : 0;
    return g_lookup == 1;
}

__global__ void k_body(float* y) { y[0] = 1.0f; }

static void launch(int cfg, float* y) {
    g_launches++;
    hipLaunchKernelGGL(k_body, dim3(kCfgs[cfg].bm), dim3(64), 0, 0, y);
}

extern "C" int hip_gate(int M, float* y) {
    if (M < minMOverride() || !lookupEnabled()) return -1;
    if (!kCfgs[0].fused) return -1;
    if (padMaxBytes() == 0) return -1;
    if (g_launches > 0) { }
    launch(0, y);
    return 0;
}
'''
GATED_FUNCTIONS = ["hip_gate", "minMOverride", "padMaxBytes", "lookupEnabled"]


def _gated_manifest(src=GATED):
    return ep_abi.manifest(src, functions=GATED_FUNCTIONS)


def test_manifest_freezes_the_variables_frozen_functions_read():
    man = _gated_manifest()
    assert set(man["constants"]) == {"kPadDefault"}
    assert set(man["variables"]) == {"kCfgs", "g_lookup", "g_lookup_mutex", "g_launches"}
    assert man["variables"]["g_lookup"] == "static int g_lookup=-1;"
    # launch() is editable and counts launches too, so only the declaration
    # of g_launches is frozen.
    assert man["mutable"] == ["g_lookup", "g_lookup_mutex"]
    assert "getenv" in man["guarded"] and "g_buf" not in man["guarded"]
    assert man["macros"] == {} and man["env_writes"] == 0
    assert ep_abi.check_source(GATED, man) == []


@pytest.mark.parametrize("edit,match", [
    # the review's probes, on the same shapes as the matmul_nbits kernel
    (("static int minMOverride() {", '#define getenv(s) ("1")\nstatic int minMOverride() {'),
     "#define/#undef of getenv"),
    (("static int g_lookup = -1;", "static int g_lookup = 0;"), "frozen variable g_lookup changed"),
    (("if (!v) return kPadDefault;", "if (!v) return 0;"), "dispatch gate padMaxBytes()"),
    # the ways around a frozen declaration
    (("    g_launches++;\n", "    g_launches++;\n    g_lookup = 0;\n"),
     "frozen variable g_lookup is used outside"),
    (("__global__ void k_body", "#define LATCH g_lookup\n__global__ void k_body"),
     "frozen variable g_lookup is used outside"),
    (("{32, false},", "{32, true},"), "frozen variable kCfgs changed"),
    (("static int g_launches = 0;", "static int g_launches = 5;"),
     "frozen variable g_launches changed"),
    (("__global__ void k_body", 'static int g_env = setenv("LOOKUP", "", 1);\n__global__ void k_body'),
     "sets an environment variable"),
])
def test_edits_that_reach_frozen_functions_from_outside_are_rejected(edit, match):
    src = GATED.replace(*edit)
    assert src != GATED
    violations = ep_abi.check_source(src, _gated_manifest())
    assert any(match in v for v in violations), violations


@pytest.mark.parametrize("edit", [
    ("y[0] = 1.0f; }", "y[0] = 2.0f; }"),
    ("    g_launches++;\n", "    g_launches += 2;\n"),
    ("static unsigned char* g_buf = nullptr;", "static unsigned char* g_buf = nullptr;\nstatic int g_new = 0;"),
    ("__global__ void k_body", "#define TILE 64\n__global__ void k_body"),
])
def test_edits_frozen_functions_cannot_see_pass(edit):
    src = GATED.replace(*edit)
    assert src != GATED
    assert ep_abi.check_source(src, _gated_manifest()) == []


def test_changing_a_macro_a_frozen_function_uses_is_rejected():
    src = (GATED.replace("static int minMOverride() {", "#define MIN_M_DEFAULT 8\nstatic int minMOverride() {")
           .replace("atoi(v) : 8;", "atoi(v) : MIN_M_DEFAULT;"))
    man = _gated_manifest(src)
    assert man["macros"] == {"MIN_M_DEFAULT": ["#define MIN_M_DEFAULT 8"]}
    assert ep_abi.check_source(src, man) == []
    assert ep_abi.check_source(src.replace("MIN_M_DEFAULT 8", "MIN_M_DEFAULT 1"), man) == [
        "#define/#undef of MIN_M_DEFAULT changed; a frozen function uses that name"]


def test_a_macro_reaching_an_ifdef_inside_a_frozen_function_is_rejected():
    src = GATED.replace('    const char* v = getenv("PAD_MAX");',
                        '#ifdef PAD_OFF\n    return 0;\n#endif\n    const char* v = getenv("PAD_MAX");')
    man = _gated_manifest(src)
    assert ep_abi.check_source(src, man) == []
    bad = src.replace("static size_t padMaxBytes() {", "#define PAD_OFF\nstatic size_t padMaxBytes() {")
    assert any("#define/#undef of PAD_OFF" in v for v in ep_abi.check_source(bad, man))


# -- exports -----------------------------------------------------------------

def _fake_nm(tmp_path, out):
    nm = tmp_path / "nm"
    nm.write_text(f"#!/bin/sh\ncat <<'EOF'\n{out}EOF\n")
    nm.chmod(nm.stat().st_mode | stat.S_IEXEC)
    return str(nm)


def test_exported_c_symbols_keeps_defined_unmangled_names(tmp_path):
    nm = _fake_nm(tmp_path, """hip_k T 0000000000001000 0000000000000010
hip_k_supported W 0000000000002000 0000000000000010
kTable R 0000000000003000 0000000000000008
_ZN3foo3barEv T 0000000000004000 0000000000000010
_init T 0000000000000500
hip_versioned@@V1 T 0000000000005000 0000000000000010
local_thing t 0000000000006000 0000000000000010
""")
    assert ep_abi.exported_c_symbols("lib.so", nm=nm) == [
        "hip_k", "hip_k_supported", "hip_versioned", "kTable"]


def test_check_exports_names_missing_and_new():
    got = ep_abi.check_exports(["a", "c"], ["a", "b"])
    assert any("b" in v and "no longer" in v for v in got)
    assert any("c" in v and "new" in v for v in got)
    assert ep_abi.check_exports(["a"], ["a"]) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
