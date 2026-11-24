# AMD hardware primary sources (ISA queries)

**Do not assert a gfx950 hardware number from memory: query it.** Tools live in
`kernel_workflow/scripts/kernel_tools/` (`amd_isa.py --arch`). Databases: `perf_knowledge/hardware/data/`
(`gfx950-mfma-layout.json`, `gfx950-encoding.json.gz`, `gfx950-isa-facts.json.gz`,
`cdna3-encoding.json.gz`), offline, every record page-cited.

```bash
KT=kernel_workflow/scripts/kernel_tools
python3 $KT/amd_isa.py --arch gfx950 search "bank conflict"            # start here: all DBs at once
python3 $KT/amd_isa.py --arch gfx950 list | facts V_MFMA_F32_16X16X32_BF16   # shapes, cycles, dtypes
python3 $KT/amd_isa.py --arch gfx950 layout <V_MFMA_...> -A --as fp8   # lane x VGPR x bit table
python3 $KT/amd_isa.py --arch gfx950 locate <V_MFMA_...> A <i> <k>     # one element -> lane / bits
python3 $KT/amd_isa.py --arch gfx950 encoding|opcodes <instr>          # opcode, fields, ch.12 pseudo-code
python3 $KT/amd_isa.py --arch gfx950 lds|waitcnt|waits|hazard|mem|limits|regs|modifiers|sparse|errata
python3 $KT/amd_isa.py --arch gfx942 encoding v_mfma_f32_16x16x16_f16  # gfx942: instruction chapters only
python3 $KT/amd_isa.py --arch gfx942 layout v_mfma_f32_16x16x32_fp8_fp8 -A  # gfx942 operand layout (needs the Matrix Instruction Calculator)
echo '<insn>' | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950         # does the instruction exist on the target
```

gfx950 answers every query offline. gfx942 (`cdna3`) answers the instruction chapters (12-13);
operand layout needs the calculator (`SCOPED TOOL-GAP` when absent), and registers / waits / LDS
rules are not extracted - the tool says "missing knowledge", which is not missing hardware.

## Errata and trust levels

- **VOP3 opcode +64:** the ISA PDF prints the VOP3 opcode of VOP1-promoted instructions 64 too high
  (85 instructions on CDNA4, 81 on CDNA3, section 12.11). Never hand-assemble from the printed
  column; `amd_isa.py --arch <a> errata` has the XML-correct values.
- **`DS_READ_*_TR_*` (the MFMA-feeding transpose loads) have no pseudo-code box** in the document;
  their lane -> element mapping is not recoverable from it, and `encoding` says so.
- **CDNA4-new surfaces rest on a single reading of one document:** F8F6F4, E8M0 scaling, 6-bit
  packing, the K=128 shapes, the new sparse shapes. `sparse` and `hazard` are marked *needs review*.
  **Validate a low-precision layout numerically before shipping a kernel on it.** The tool exits 3
  (refuses) rather than guessing, e.g. for sparse A/B operands and runtime-selected F8F6F4 widths.
- Records marked `_conf: "uncertain"` mean re-read the cited page (the source contradicts itself,
  e.g. LDS "64 banks" vs "32 concurrent accesses").
- A wrong-numbers kernel that compiles is usually a data / scale **layout** mismatch: check `layout`
  first.
