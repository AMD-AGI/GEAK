# Compiler layouts and hardware facts: minimal interfaces

Use the installed compiler for Gluon layout semantics and the shared ISA tools for instruction
facts. Neither replaces a compile + correctness check.

## Ownership

| Question | Interface |
| --- | --- |
| Construct MFMA data / scale layouts | Native Triton classes, `cdna4.get_mfma_scale_layout` (no GEAK generator) |
| View a constructed layout | Native `format_hardware_view` / `format_tensor_view` (logical slots, not register allocation) |
| Recover / check layouts from real TTGIR | [ttgir_bridge.py](../../scripts/ttgir_bridge.py) `recover` / `verify` / `view` |
| Opcode, operand bits, ISA constraints | `amd_isa.py --arch TARGET` ([ISA tools](../../../../../../kernel_workflow/scripts/kernel_tools/isa_tools.md)) |
| Compiled registers / LDS | `dump_ir.sh` / `probe.py measure` (layout objects cannot report them) |

## Inspect the actual compiler first

```bash
python perf_knowledge/expert_skills/skills/gluon_authoring/scripts/ttgir_bridge.py capabilities --json
```

Reports native API availability/signatures and probes shared/distributed normalization in a child
process (ignores `TTGIR_BRIDGE_CAPS`; no GPU; exit 3 if Triton is missing). API presence is not
proof a target/dtype/layout lowers. Select APIs by reported signature, not `triton.__version__`
(vendor builds backport). Version rows a 3.8.0 user can hit: `compute_efficient_padded_shared_layout`
and the explicit `scale_factor=32` argument are 3.8-only; layout views (`format_hardware_view`) are
absent on 3.6.

## Native data/scale layout example

**gfx950 packed MXFP4: instruction K counts FP4 elements; the A/B tensor shapes count packed bytes.**

```python
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna4

parent = AMDMFMALayout(version=4, instr_shape=[16, 16, 128],
                      transposed=True, warps_per_cta=[2, 2])
a_layout = gl.DotOperandLayout(0, parent, 16)
b_layout = gl.DotOperandLayout(1, parent, 16)
# Two positional arguments work on 3.6/3.7/3.8; the scale group is fixed at 32.
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

Validate any chosen layout on the real load/store/async-copy path; a `None` from the padding factory
means unsupported for those inputs (`layout-reference.md ### Contract points — places this goes
wrong quietly`). Runnable survey: [gfx950 validation procedure](gfx950-validation.md).

## Independent hardware queries

```bash
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 facts v_mfma_scale_f32_16x16x128_f8f6f4 --as fp4
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 tables scale.16x16x128
python kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 locate v_mfma_scale_f32_16x16x128_f8f6f4 A 0 1 --as fp4
```

`facts --as` gives register spans for **equal** A/B formats (query mixed formats per operand);
without `--as` F8F6F4 usage stays unknown. Neither is compiler allocation. `tables scale.*` returns
source excerpts, not a second scale generator.
