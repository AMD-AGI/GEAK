# Compiler layouts and hardware facts: minimal interfaces

Use the installed compiler for Gluon layout semantics. Use shared ISA tools for
instruction facts or an independent hardware cross-check. Neither replaces a
compile/correctness check. These sources do not select an optimization tactic.

## Ownership

| Question | Interface | Why GEAK needs it |
| --- | --- | --- |
| Construct MFMA/WMMA data or scale layouts | Native Triton layout classes and `cdna4.get_mfma_scale_layout` | Already supplied by Triton; no GEAK generator |
| View a constructed layout | Native `format_hardware_view` / `format_tensor_view` | Reuse compiler semantics; views are logical slots, not physical register allocation |
| Recover/check layouts from actual TTGIR | [ttgir_bridge.py](../../scripts/ttgir_bridge.py) `recover/verify/view` | Adds use-site shape, role, provenance and equivalence checks around compiler APIs |
| Query opcode, operand bits, constraints or ISA source rules | Shared `amd_isa.py --arch TARGET` | Hardware facts for HIP, inline asm and other DSLs as well as Gluon |
| Query calculator-specific modifiers or sparse indices | Shared `hw_sources.sh layout TARGET OPCODE FLAGS...` | Thin passthrough to the optional pinned calculator; no second mapping algorithm |
| Check actual compiler registers/LDS and artifact identity | Shared `dump_ir.sh` / `probe.py measure` | Layout objects cannot report compiled resource use |

Shared interfaces and dependencies: [ISA tools](../../../../../../kernel_workflow/scripts/kernel_tools/isa_tools.md).
`gfx950_isa.py` and `layout_facts.py` remain compatibility entrypoints to the shared
ISA dispatcher. Do not install the calculator merely to construct a Gluon layout.

## Inspect the actual compiler first

Run from the GEAK root in the workload's Python environment, without GPU access:

```bash
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py capabilities --json
```

The command reports native API availability/signatures and probes distributed and
shared normalization in a child process. It ignores `TTGIR_BRIDGE_CAPS` overrides
so this inventory describes the actual installation. Missing Triton exits 3 with
a diagnostic. Missing individual APIs remain explicit; nothing is downloaded.
API presence is not proof that a particular target/dtype/layout can lower.

| Official baseline | Available interface / change |
| --- | --- |
| 3.6 | MFMA, RDNA3/4 WMMA, scale helper, distributed LinearLayout, inline asm already exist |
| 3.6 → 3.7 | WMMA changes from `warps_per_cta/tiles_per_warp` to `warp_bases/reg_bases`; layout views and shared normalization branch are added |
| 3.7 → 3.8 | CDNA4 padding helper and explicit `scale_factor=32` argument are added |

Vendor builds can backport these changes. Select APIs by the reported capability
and signature, not only `triton.__version__`. A missing view on 3.6 does not mean
its compiler lacks distributed layout mappings. The bridge reports that view gap;
it does not synthesize a replacement renderer.

## Native data/scale layout example

This is a **gfx950 packed MXFP4** example, not a target-independent template.
Instruction K counts FP4 elements; the A/B tensor shapes below count packed bytes.

```python
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna4

parent = AMDMFMALayout(version=4, instr_shape=[16, 16, 128],
                      transposed=True, warps_per_cta=[2, 2])
a_layout = gl.DotOperandLayout(0, parent, 16)
b_layout = gl.DotOperandLayout(1, parent, 16)
# Two positional arguments work on the checked 3.6/3.7/3.8 interfaces.
# The hardware scale group is 32; newer helpers expose this fixed contract.
a_scale = cdna4.get_mfma_scale_layout(a_layout, [32, 4])
b_scale = cdna4.get_mfma_scale_layout(b_layout, [32, 4])

if hasattr(a_layout, "format_hardware_view"):
    print(a_layout.format_hardware_view([32, 64]))
else:
    from triton._C.libtriton import gluon_ir, ir
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = gluon_ir.GluonOpBuilder(ctx)
    print(builder.to_linear_layout(a_layout._to_ir(builder), [32, 64]))
```

Choose WMMA constructor arguments from the installed signature. For padding,
use `compute_efficient_padded_shared_layout` only when available and applicable to
CDNA4. Treat a `None` result as unsupported for those inputs. Validate the selected
layout with the actual load/store/async-copy path; construction alone is insufficient.
The [gfx950 validation procedure](gfx950-validation.md) provides a runnable native
GEMM smoke and a separate shared-memory candidate survey with per-case results.

## Independent hardware queries

From the GEAK root:

```bash
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 facts v_mfma_scale_f32_16x16x128_f8f6f4 --as fp4
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 tables scale.16x16x128
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 locate v_mfma_scale_f32_16x16x128_f8f6f4 A 0 1 --as fp4
```

`facts --as` reports the register span from the existing hardware layout rules for
**equal A/B formats**. Mixed formats need separate operand queries. Without a
format, F8F6F4 A/B usage stays unknown; XML encoding width is labelled separately.
Neither number is actual compiler allocation. `tables scale.*` returns source
excerpts and their derived-placement caveats, not a second Gluon scale generator.

For gfx942/RDNA hardware matrix queries, the calculator remains optional. The
normalized `amd_isa.py` interface exposes basic A/B/C/D queries; the raw
`hw_sources.sh layout` entry forwards additional calculator flags. XML encodings
and committed gfx950 facts remain usable without Triton or the calculator.
