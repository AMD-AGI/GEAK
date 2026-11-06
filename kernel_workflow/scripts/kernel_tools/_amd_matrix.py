"""Installed AMD Matrix Instruction Calculator adapter; no downloads or installs."""
import re
import subprocess
import sys

# byte width per calculator dtype token -> elems/lane = (GPRs*4 bytes) / bytes_per_elem
_DTYPE_BYTES = {"FP8": 1, "BF8": 1, "FP4": 0.5, "FP6": 0.75, "I8": 1, "INT8": 1,
                "FP16": 2, "BF16": 2, "F16": 2, "XF32": 4, "FP32": 4, "F32": 4, "FP64": 8, "F64": 8}


def _gap(msg):
    print(f"SCOPED TOOL-GAP: {msg}", file=sys.stderr)
    print("  -> fall back to distilled perf_knowledge/hardware/ with explicit source and uncertainty; never fabricate",
          file=sys.stderr)
    sys.exit(3)


def _calc_path():
    from amd_targets import calculator_path
    path = calculator_path()
    if path is None:
        _gap('Matrix Calculator absent; set TILE_HW_SEED or TILE_HW_TOOLS to preinstalled assets.')
    return str(path)


def _run(calc, *args):
    from matrix_dependency import interpreter
    try:
        r = subprocess.run([interpreter(calc), calc, *args], check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        _gap(f"calculator invocation failed: {e}")
    return r.returncode, r.stdout, r.stderr


def _detail(calc, arch, instr):
    rc, out, _err = _run(calc, "-a", arch, "-i", instr, "-d")
    if rc != 0 or "Matrix Dimensions" not in out:
        _gap(f"calculator has no '{instr}' for {arch} (list: layout_facts.py {arch} --list)")
    return out


def _grab(text, label):
    m = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _elems_per_lane(gprs, dtype_token):
    """GPRs*4 bytes/lane / bytes-per-elem. dtype_token like 'FP8 (AMD 4-bit...)'."""
    if gprs is None or dtype_token is None:
        return None
    key = dtype_token.split()[0].upper()
    bpe = _DTYPE_BYTES.get(key)
    if not bpe:
        return None
    return int(int(gprs) * 4 / bpe)


def facts(calc, arch, instr, want_layout):
    d = _detail(calc, arch, instr)
    M, N, K = _grab(d, "M"), _grab(d, "N"), _grab(d, "K")
    cyc = _grab(d, "Execution cycles")
    ga, gb = _grab(d, "GPRs required for A"), _grab(d, "GPRs required for B")
    gc, gd = _grab(d, "GPRs required for C"), _grab(d, "GPRs required for D")
    align = _grab(d, "GPR alignment requirement")
    coexec = _grab(d, "Can co-execute with VALU")
    src0, src1 = _grab(d, "Src0"), _grab(d, "Src1")
    epl_a = _elems_per_lane(ga, src0)
    epl_b = _elems_per_lane(gb, src1)

    print(f"## {instr.upper()}  ({arch})")
    print(f"  dims        M={M} N={N} K={K}   exec_cycles={cyc}   co-exec VALU={coexec}")
    print(f"  gpr/lane    A={ga} B={gb} C={gc} D={gd}   align={align}")
    print(f"  dtype       A(Src0)={src0}")
    print(f"              B(Src1)={src1}")
    if epl_a is not None:
        print(f"  elems/lane  A={epl_a}  B={epl_b}   "
              f"<- ds_read_tr fit: a ds_read_trN gives N/lane; need {epl_a} for A "
              f"(if trN < {epl_a}, transpose-read RAISES ds_read count -> disqualified)")
    if want_layout:
        rc, lay, err = _run(calc, "-a", arch, "-i", instr, "-R", "-A", "--markdown")
        if rc != 0:
            _gap(f'calculator layout failed for {arch}/{instr}: {err.strip()}')
        if rc == 0:
            print("  --- A register layout (v{lane}.[bits] holding A[m][k]) ---")
            print("\n".join("  " + ln for ln in lay.splitlines() if "|" in ln or ln.startswith("Block")))
    print(f"  # provenance: MI-calc {arch} {instr} -> distill into "
          f"your analysis record with this citation")
    print()
