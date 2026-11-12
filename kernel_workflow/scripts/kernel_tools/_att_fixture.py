"""Synthetic decoded-ATT fixture shared by the att_* / kernel_breakdown selftests (no GPU, no decoder).

Shapes mirror rocprofv3 v3.0.0 decoded output:
  code.json  {"header": "ISA,_,LineNumber,Source,Codeobj,Vaddr,Hit,Latency,Stall,Idle", "code": [rows]}
  se0_sm0_sl0_wv0.json  {"wave": {"begin", "end", "cu", "simd", "instructions": [[cycle, _, cost, _, code_idx], ...]}}
"""
from __future__ import annotations

import json
import os

HEADER = "ISA,_,LineNumber,Source,Codeobj,Vaddr,Hit,Latency,Stall,Idle"
CODE = [
    ["; kernel prologue", 0, 0, "", 0, 0, 0, 0, 0, 0],
    ["buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen", 0, 1, "k.py:3", 0, 4096, 4, 120, 20, 0],
    ["s_waitcnt vmcnt(0)", 0, 2, "k.py:4", 0, 4100, 4, 400, 380, 0],
    ["ds_read_b128 v[8:11], v5", 0, 3, "k.py:5", 0, 4104, 8, 96, 48, 0],
    ["v_mfma_f32_16x16x32_bf16 a[0:3], v[8:11], v[12:15], a[0:3]", 0, 4, "k.py:6", 0, 4108, 8, 128, 16, 0],
    ["v_add_f32 v20, v21, v22", 0, 5, "k.py:7", 0, 4112, 8, 32, 0, 0],
    ["s_barrier", 0, 6, "k.py:8", 0, 4116, 4, 60, 50, 0],
]
# (cycle, _, cost, _, code_idx): two passes over the loop body starting at cycle 1000
WAVE_INSTR = [
    [1000, 0, 20, 0, 1], [1020, 0, 380, 0, 2], [1400, 0, 12, 0, 3], [1412, 0, 16, 0, 4],
    [1428, 0, 4, 0, 5], [1432, 0, 16, 0, 4], [1448, 0, 4, 0, 5], [1452, 0, 50, 0, 6],
]


def write(td: str) -> tuple[str, str]:
    """Write code.json + one wave file under `td`; return (wave_path, code_path)."""
    code = os.path.join(td, "code.json")
    wave = os.path.join(td, "se0_sm0_sl0_wv0.json")
    with open(code, "w") as fh:
        json.dump({"header": HEADER, "code": CODE}, fh)
    with open(wave, "w") as fh:
        json.dump({"wave": {"begin": 1000, "end": 1502, "cu": 0, "simd": 0,
                            "instructions": WAVE_INSTR}}, fh)
    return wave, code
