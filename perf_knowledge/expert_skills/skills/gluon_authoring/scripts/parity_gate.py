#!/usr/bin/env python3
"""parity_gate.py - the anchor's transcription debt: is it paid, and if not, who owes it?

A faithful Gluon anchor is a REGRESSION you knowingly created, not a baseline to quietly climb
from. Climbing before the debt is paid caps the whole port: every later lever is measured against
a broken starting point, and the run closes below the champion while reporting a healthy-looking
gain "vs the anchor". That is the single most expensive procedural mistake available in a port,
and it is invisible without this gate, because the layout-equivalence checker says PASS and the
numeric oracle says PASS while the kernel runs 1.4x slower.

So: until `champion_ms / anchor_ms >= --threshold` (0.95 by default), a round's outcome is
`recovery` against the suspect it closed -- never a win. This tool decides that, and when the
gate is NOT cleared it attributes the gap across the three suspects, each from a signal in the
compiled artifacts rather than from a story:

  lost_pipeline  the champion's loop was software-pipelined and the anchor's is not.
                 Evidence: the champion's TTGIR carries `ttg.memdesc_index` / `ttg.local_store` /
                 `num_stages > 1` and the anchor's does not. Owned by the pipeline layer
                 (re-inject plain's pipeliner -- Route 1 -- before authoring anything).
                 NOTE the inverse trap: `max iter_args >= 2` is NOT evidence of pipelining, any
                 accumulator loop satisfies it. Only memdesc_index / local_store / a peeled
                 prologue are.

  lost_layout    a conversion was folded backwards into a load, or a staging buffer was
                 materialized that the champion left to the compiler. Evidence: the load-width
                 or LDS-op histogram shifted toward NARROWER operations (dwordx4 -> ushort,
                 ds_read_b128 -> ds_read_u16), or `shared` bytes/WG crossed an LDS/CU divisor.
                 Owned by the memory-path / shared-layout layers.

  lost_RA        the instruction multiset is essentially unchanged and the register allocator
                 serialized it anyway. Evidence: VGPR rose (especially across a wave
                 threshold), spill appeared, or the number of DISTINCT address registers
                 feeding the LDS read burst collapsed. This is the one a layout-equivalence
                 checker structurally cannot see: equivalent layouts, equal counters, unequal
                 address-register pressure. Read the `ds_read` OPERANDS, not just the count.

Usage:
  parity_gate.py --champion-ms 2.8966 --anchor-ms 4.0754 \
                 --champion-asm ir/champion/k.amdgcn --anchor-asm ir/anchor/k.amdgcn \
                 [--champion-ttgir ir/champion/k.ttgir] [--anchor-ttgir ir/anchor/k.ttgir] \
                 [--threshold 0.95] [--json parity.json]
  parity_gate.py --selftest

Exit status: 0 when the gate is CLEARED, 2 when it is not (so a round script can branch on it).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- what "narrower" means, per family. Ordered widest -> narrowest; the index IS the rank. ---
GLOBAL_WIDTHS = ["dwordx4", "dwordx3", "dwordx2", "dword", "ushort", "ubyte"]
LDS_WIDTHS = ["b128", "b96", "b64", "b32", "u16", "u8"]

# next_free_vgpr is the ONLY number the 256 cliff is measured against: ArchVGPR and AGPR share
# ONE 512-register file per SIMD, so `num_vgpr` / `.vgpr_count` (arch only) reads comfortably
# under 256 on a kernel that is actually past it. Keep both, decide on next_free.
_NEXTFREE_RE = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
_ARCHVGPR_RE = re.compile(r"(?:\.set\s+\S*num_vgpr,\s*|\.vgpr_count:\s*)(\d+)")
_ACCUM_RE = re.compile(r"(?:\.amdhsa_accum_offset\s+|;\s*AccumOffset\s*:\s*)(\d+)")
_OCC_RE = re.compile(r"^\s*;\s*Occupancy\s*:\s*(\d+)", re.M)
_SPILL_RE = re.compile(r"(?:\.vgpr_spill_count:\s*|;\s*ScratchSize\s*:\s*)(\d+)")
# `LDSByteSize` in the .amdgcn is a structural 0 on Triton kernels -- shared memory is sized
# dynamically at launch, so the compile-time field says "0 bytes/workgroup (compile time only)".
# The real figure is the `shared` field of the Triton cache metadata; pass it in with --*-lds.
_LDS_RE = re.compile(r"^\s*;\s*LDSByteSize:\s*(\d+)", re.M)


def _first_int(rx, text, default=None):
    m = rx.search(text)
    return int(m.group(1)) if m else default


def _hist(text: str, mnemonic_rx: str, widths: list[str]) -> dict[str, int]:
    """Count occurrences of each width suffix for the given mnemonic family."""
    out = {}
    for w in widths:
        n = len(re.findall(rf"\b{mnemonic_rx}[a-z0-9_]*{re.escape(w)}\b", text))
        if n:
            out[w] = n
    return out


def _weighted_rank(hist: dict[str, int], widths: list[str]) -> float | None:
    """Mean narrowness rank, 0 = all widest. A rise means the access got narrower."""
    total = sum(hist.values())
    if not total:
        return None
    return sum(widths.index(w) * n for w, n in hist.items()) / total


_DS_READ_RE = re.compile(r"^\s*(ds_read[a-z0-9_]*)\s+(?:v\[[\d:]+\]|v\d+)\s*,\s*(v\d+)")
_VALU_DEF_RE = re.compile(r"^\s*(v_[a-z0-9_]+)\s+(v\d+)\s*,")


def ds_addr_pressure(text: str) -> dict:
    """The address-register signature of the LDS read burst.

    This is the discriminator a layout-equivalence check structurally cannot make. A champion
    issuing its reads from N precomputed address registers has N mutually independent reads the
    hardware coalesces across banks. The same reads issued from a handful of registers, with the
    address rematerialized into one of them immediately before each read, are serialized by a
    WAR hazard on that register -- a dependency chain synthesised by the register allocator, not
    by the layout. Equivalent layouts, equal `ds_read` COUNTS, unequal address pressure.

    `remat` is the load-bearing number and it is measured locally (does the instruction
    immediately above this read define its address register?), so it does not need the hot loop
    to be delimited first. `distinct` and `max_per_reg` are whole-file and therefore diluted by
    prologue/epilogue code -- reported, but not decided on.
    """
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith((";", ".", "//"))]
    remat = 0
    per_reg: dict[str, int] = {}
    for i, ln in enumerate(lines):
        m = _DS_READ_RE.match(ln)
        if not m:
            continue
        addr = m.group(2)
        per_reg[addr] = per_reg.get(addr, 0) + 1
        for back in (1, 2):                     # the VALU write may sit 1-2 slots above
            if i - back < 0:
                break
            d = _VALU_DEF_RE.match(lines[i - back])
            if d and d.group(2) == addr:
                remat += 1
                break
    return {
        "n_reads": sum(per_reg.values()),
        "distinct": len(per_reg),
        "max_per_reg": max(per_reg.values()) if per_reg else 0,
        "remat": remat,
    }


def _count(text: str, pat: str) -> int:
    return len(re.findall(pat, text))


def asm_facts(text: str, lds_bytes: int | None = None) -> dict:
    lds_asm = _first_int(_LDS_RE, text)
    return {
        # the occupancy-relevant register count, and the arch-only one it is often confused with
        "next_free_vgpr": _first_int(_NEXTFREE_RE, text),
        "arch_vgpr": _first_int(_ARCHVGPR_RE, text),
        "accum_offset": _first_int(_ACCUM_RE, text),
        "occupancy_waves_per_simd": _first_int(_OCC_RE, text),
        "spill_bytes": _first_int(_SPILL_RE, text, 0),
        # None, not 0, when the asm's compile-time field is the structural zero
        "lds_bytes": lds_bytes if lds_bytes is not None else (lds_asm or None),
        "lds_source": ("caller (--*-lds, from Triton cache `shared`)" if lds_bytes is not None
                       else "asm LDSByteSize" if lds_asm else
                       "UNAVAILABLE -- the asm field is a structural 0 on Triton kernels; pass "
                       "--champion-lds/--anchor-lds from the cache metadata's `shared`"),
        "global_load_hist": _hist(text, r"(?:buffer|global|flat)_load_", GLOBAL_WIDTHS),
        "ds_read_hist": _hist(text, r"ds_read", LDS_WIDTHS),
        "ds_write_hist": _hist(text, r"ds_write", LDS_WIDTHS),
        "ds_addr": ds_addr_pressure(text),
        "n_ds_read": _count(text, r"\bds_read"),
        "n_ds_write": _count(text, r"\bds_write"),
        "n_mfma": _count(text, r"\bv_mfma"),
        "n_barrier": _count(text, r"\bs_barrier\b"),
        "n_valu": _count(text, r"\bv_(?!mfma)[a-z]"),
    }


def ttgir_pipeline_facts(text: str) -> dict:
    """Only the signals that actually evidence a software pipeline.

    `iter_args` is deliberately absent: reading `max iter_args >= 2` as pipelining is a
    confident false positive on every accumulator loop, including any online-softmax kernel.
    """
    num_stages = None
    m = re.search(r"tt\.num_stages\s*=\s*(\d+)", text) or \
        re.search(r"num_stages\s*=\s*(\d+)\s*:", text)
    if m:
        num_stages = int(m.group(1))
    return {
        "num_stages": num_stages,
        "memdesc_index": _count(text, r"\bttg\.memdesc_index\b"),
        "local_store": _count(text, r"\bttg\.local_store\b"),
        "local_alloc": _count(text, r"\bttg\.local_alloc\b"),
        "async_copy": _count(text, r"async_copy|async_commit|AsyncCopy"),
        "barrier": _count(text, r"\bttg\.barrier\b|\bgpu\.barrier\b"),
    }


def _is_pipelined(f: dict) -> bool:
    return bool(f["memdesc_index"] or f["local_store"] or f["async_copy"]
                or (f["num_stages"] or 1) > 1)


def attribute(champ_asm: dict | None, anch_asm: dict | None,
              champ_ttgir: dict | None, anch_ttgir: dict | None) -> list[dict]:
    """One verdict per suspect: suspected / cleared / unknown, each with its numbers."""
    out = []

    # --- lost_pipeline ---
    if champ_ttgir and anch_ttgir:
        cp, ap = _is_pipelined(champ_ttgir), _is_pipelined(anch_ttgir)
        if cp and not ap:
            out.append({"suspect": "lost_pipeline", "verdict": "SUSPECTED",
                        "owned_by": "pipeline layer -- re-inject plain's pipeliner (Route 1) "
                                    "BEFORE authoring overlap by hand",
                        "evidence": {"champion": champ_ttgir, "anchor": anch_ttgir}})
        else:
            why = ("the champion's own loop is not pipelined either (memdesc_index=%d, "
                   "local_store=%d, num_stages=%s), so there is no pipeline to lose"
                   % (champ_ttgir["memdesc_index"], champ_ttgir["local_store"],
                      champ_ttgir["num_stages"]))
            out.append({"suspect": "lost_pipeline", "verdict": "CLEARED",
                        "evidence": {"reason": why if not cp else "the anchor is pipelined too",
                                     "champion": champ_ttgir, "anchor": anch_ttgir}})
    else:
        out.append({"suspect": "lost_pipeline", "verdict": "UNKNOWN",
                    "evidence": {"reason": "pass --champion-ttgir and --anchor-ttgir; this "
                                           "suspect cannot be judged from the .amdgcn alone"}})

    if not (champ_asm and anch_asm):
        for s in ("lost_layout", "lost_RA"):
            out.append({"suspect": s, "verdict": "UNKNOWN",
                        "evidence": {"reason": "pass --champion-asm and --anchor-asm"}})
        return out

    # --- lost_layout: did any access family get NARROWER, or did shared cross a divisor? ---
    shifts, ev = [], {}
    for key, widths in (("global_load_hist", GLOBAL_WIDTHS),
                        ("ds_read_hist", LDS_WIDTHS),
                        ("ds_write_hist", LDS_WIDTHS)):
        rc, ra = _weighted_rank(champ_asm[key], widths), _weighted_rank(anch_asm[key], widths)
        ev[key] = {"champion": champ_asm[key], "anchor": anch_asm[key],
                   "narrowness_rank": {"champion": rc, "anchor": ra}}
        if rc is not None and ra is not None and ra > rc + 0.25:
            shifts.append(f"{key} narrowed (rank {rc:.2f} -> {ra:.2f})")
    lds_c, lds_a = champ_asm["lds_bytes"], anch_asm["lds_bytes"]
    if lds_c and lds_a and lds_a > lds_c:
        shifts.append(f"shared bytes/WG grew {lds_c} -> {lds_a}")
    ev["lds_bytes"] = {"champion": lds_c, "anchor": lds_a,
                       "source": {"champion": champ_asm["lds_source"],
                                  "anchor": anch_asm["lds_source"]}}
    if lds_c is None or lds_a is None:
        ev["lds_bytes"]["warning"] = (
            "LDS/WG unavailable, so the 'a materialized staging buffer cost an occupancy step' "
            "half of this suspect was NOT tested. That is the single most common lost_layout "
            "mechanism in a port -- pass --champion-lds/--anchor-lds before trusting a CLEARED.")
    out.append({"suspect": "lost_layout",
                "verdict": "SUSPECTED" if shifts else "CLEARED",
                "owned_by": "memory path / shared layout -- classify each ttg.local_alloc "
                            "(staged -> allocate_shared_memory; pass-through -> convert_layout, "
                            "buffer stays compiler-owned) before re-writing it",
                "evidence": {"shifts": shifts, **ev}})

    # --- lost_RA: same work, worse allocation ---
    reasons = {}
    inst_c = champ_asm["n_valu"] + champ_asm["n_mfma"] + champ_asm["n_ds_read"] + champ_asm["n_ds_write"]
    inst_a = anch_asm["n_valu"] + anch_asm["n_mfma"] + anch_asm["n_ds_read"] + anch_asm["n_ds_write"]
    multiset_same = inst_c and abs(inst_a - inst_c) / inst_c <= 0.05
    if anch_asm["spill_bytes"] and not champ_asm["spill_bytes"]:
        reasons["spill_appeared"] = anch_asm["spill_bytes"]
    vc, va = champ_asm["next_free_vgpr"], anch_asm["next_free_vgpr"]
    if vc and va and va > vc:
        reasons["next_free_vgpr_rose"] = {"champion": vc, "anchor": va}
        # the 256 cliff: ArchVGPR+AGPR share ONE 512/SIMD file, so crossing it costs a wave
        if vc <= 256 < va:
            reasons["crossed_256_wave_threshold"] = True
    oc, oa = champ_asm["occupancy_waves_per_simd"], anch_asm["occupancy_waves_per_simd"]
    if oc and oa and oa < oc:
        reasons["occupancy_dropped"] = {"champion": oc, "anchor": oa}
    ac, aa = champ_asm["ds_addr"], anch_asm["ds_addr"]
    # remat is the mechanism itself: an address recomputed into a register immediately before
    # the read that consumes it. A rise here IS the serial chain, whatever the counts say.
    if aa["remat"] > max(ac["remat"] * 1.5, ac["remat"] + 3):
        reasons["ds_address_rematerialization_rose"] = {
            "champion": ac["remat"], "anchor": aa["remat"],
            "mechanism": "each read is serialized behind a WAR hazard on an address register "
                         "recomputed immediately above it -- a dependency chain the allocator "
                         "synthesised. Read the ds_read OPERANDS, not just the count."}
    # `distinct` is a whole-file count, so it only means "the same reads lost their independent
    # addresses" when the reads ARE the same. On an anchor whose instruction stream genuinely
    # changed (u16 reads folded into b128, say) a drop here is arithmetic, not a regression --
    # firing on it would hand the author a serialization story about a kernel that got wider.
    if multiset_same and ac["distinct"] and aa["distinct"] < ac["distinct"] * 0.75:
        reasons["ds_address_registers_collapsed"] = {
            "champion": ac["distinct"], "anchor": aa["distinct"],
            "mechanism": "fewer distinct address registers feed the same reads, so they can no "
                         "longer issue independently. Whole-file count -- confirm in the hot loop."}
    ev_addr = {"champion": ac, "anchor": aa}
    out.append({"suspect": "lost_RA",
                "verdict": "SUSPECTED" if reasons else "CLEARED",
                "owned_by": "register/slicing layer -- and note a lower LDS instruction COUNT "
                            "can still be slower; the address-dependency chain is what binds",
                "evidence": {"instruction_multiset_unchanged": bool(multiset_same),
                             "instructions": {"champion": inst_c, "anchor": inst_a},
                             "ds_addr": ev_addr,
                             **reasons}})
    return out


def evaluate(champion_ms: float, anchor_ms: float, threshold: float, suspects: list[dict]) -> dict:
    ratio = champion_ms / anchor_ms if anchor_ms else None
    cleared = ratio is not None and ratio >= threshold
    named = [s["suspect"] for s in suspects if s["verdict"] == "SUSPECTED"]
    unknown = [s["suspect"] for s in suspects if s["verdict"] == "UNKNOWN"]
    res = {
        "champion_ms": champion_ms, "anchor_ms": anchor_ms,
        "ratio_champion_over_anchor": round(ratio, 4) if ratio else None,
        "threshold": threshold,
        "gate": "CLEARED" if cleared else "NOT CLEARED",
        "suspects": suspects,
        "suspected": named,
        "unattributed": unknown,
    }
    if cleared:
        res["round_outcome_allowed"] = "win"
        res["note"] = ("The debt is paid (or was never taken). Climbing is allowed, and a round "
                       "may be scored as a win. Reaching parity is NOT itself a win: it is "
                       "getting back to a number the front end already measured.")
        if ratio > 1.0:
            res["note"] += (" The anchor is FASTER than the champion -- attribute that too "
                            "rather than pocketing it; it is usually compiler-owned staging "
                            "buying an occupancy step, and knowing which one it is tells you "
                            "what the remaining levers can and cannot touch.")
    else:
        res["round_outcome_allowed"] = "recovery"
        res["note"] = (
            "DO NOT CLIMB YET. Until champion_ms/current_ms >= %.2f a round's outcome is "
            "`recovery` against the suspect it closed, never a win. Close the suspects above "
            "with the layer that OWNS each one, re-run this gate, and only then start the "
            "layer loop. Climbing from here caps the port: the best lever you find will be "
            "quoted against a broken anchor and the run will close below the champion while "
            "reporting a gain." % threshold)
        if not named:
            res["note"] += (" No suspect fired, which is itself a finding: pass the artifacts "
                            "(--champion-asm/--anchor-asm/--*-ttgir) if you have not, and if "
                            "they are all CLEARED then the gap is not in this taxonomy -- "
                            "profile it before authoring anything.")
    return res


def _fmt(res: dict) -> str:
    L = [f"=== parity gate: champion {res['champion_ms']:.4f} ms / anchor {res['anchor_ms']:.4f} ms"
         f" = {res['ratio_champion_over_anchor']}  (threshold {res['threshold']}) ===",
         f"  {res['gate']}   round outcome allowed: {res['round_outcome_allowed']}"]
    for s in res["suspects"]:
        L.append(f"  [{s['verdict']:9s}] {s['suspect']}")
        ev = s.get("evidence", {})
        for k in ("shifts", "reason"):
            if ev.get(k):
                L.append(f"       {k}: {ev[k]}")
        for k, v in ev.items():
            if k in ("shifts", "reason", "champion", "anchor"):
                continue
            if isinstance(v, dict) and "mechanism" in v:
                L.append(f"       {k}: {v['champion']} -> {v['anchor']}")
                L.append(f"           {v['mechanism']}")
            elif not isinstance(v, dict):
                L.append(f"       {k}: {v}")
        if s["verdict"] == "SUSPECTED" and s.get("owned_by"):
            L.append(f"       -> owned by: {s['owned_by']}")
    L.append("")
    L.append(f"  {res['note']}")
    return "\n".join(L)


def _selftest() -> int:
    # Fixtures shaped like a real .amdgcn: the register number that matters is
    # .amdhsa_next_free_vgpr (ArchVGPR+AGPR), NOT the arch-only num_vgpr beside it.
    champ = """
		.amdhsa_next_free_vgpr 257
	.set k.num_vgpr, 248
; Occupancy: 1
	buffer_load_dwordx4 v[10:13], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v5, s[0:3], 0 offen
	ds_read2_b64 v[34:37], v139 offset1:16
	ds_read2_b64 v[38:41], v141 offset1:16
	ds_read2_b64 v[42:45], v143 offset1:16
	ds_read2_b64 v[46:49], v146 offset1:16
	ds_write_b128 v200, v[10:13]
	v_mfma_f32_16x16x16_bf16 a[0:3], v[10:11], v[12:13], a[0:3]
	s_barrier
    """
    # the anchor rematerializes every address into ONE register right before its read
    anch = """
		.amdhsa_next_free_vgpr 202
	.set k.num_vgpr, 196
; Occupancy: 2
	buffer_load_dwordx4 v[10:13], v4, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v5, s[0:3], 0 offen
	v_add_u32_e32 v33, 0x2000, v139
	ds_read2_b64 v[34:37], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v141
	ds_read2_b64 v[38:41], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v143
	ds_read2_b64 v[42:45], v33 offset1:16
	v_add_u32_e32 v33, 0x2000, v146
	ds_read2_b64 v[46:49], v33 offset1:16
	ds_write_b128 v200, v[10:13]
	v_mfma_f32_16x16x16_bf16 a[0:3], v[10:11], v[12:13], a[0:3]
	s_barrier
    """
    cf, af = asm_facts(champ, 32768), asm_facts(anch, 16384)
    # next_free_vgpr is read, and is NOT confused with the arch-only count beside it
    assert cf["next_free_vgpr"] == 257 and cf["arch_vgpr"] == 248, cf
    assert af["next_free_vgpr"] == 202, af
    # the discriminator: 4 independent address regs, 0 remat -> 1 shared reg, 4 remats
    assert cf["ds_addr"]["distinct"] == 4 and cf["ds_addr"]["remat"] == 0, cf["ds_addr"]
    assert af["ds_addr"]["remat"] == 4, af["ds_addr"]
    # the asm LDS field is a structural 0 -> reported as UNAVAILABLE, never as 0
    assert asm_facts(anch)["lds_bytes"] is None, asm_facts(anch)
    assert "UNAVAILABLE" in asm_facts(anch)["lds_source"]

    # 1. an anchor FASTER than the champion clears, and is told to attribute the gain anyway
    res = evaluate(2.8966, 2.2255, 0.95, attribute(cf, af, None, None))
    assert res["gate"] == "CLEARED" and res["round_outcome_allowed"] == "win", res
    assert "FASTER than the champion" in res["note"], res["note"]

    # 2. the pa_decode-shaped failure: 0.71x, and lost_RA must fire on the address-reg collapse
    res = evaluate(2.8966, 4.0754, 0.95, attribute(cf, af, None, None))
    assert res["gate"] == "NOT CLEARED", res
    assert res["round_outcome_allowed"] == "recovery", res
    ra = next(s for s in res["suspects"] if s["suspect"] == "lost_RA")
    assert ra["verdict"] == "SUSPECTED", ra
    assert "ds_address_rematerialization_rose" in ra["evidence"], ra

    # 3. lost_layout fires when a load narrows dwordx4 -> ushort (a conversion folded into it)
    narrowed = anch.replace("buffer_load_dwordx4 v[10:13]", "buffer_load_ushort v10") \
                   .replace("buffer_load_dwordx4 v[14:17]", "buffer_load_ushort v14")
    res = evaluate(6.9035, 13.61, 0.95, attribute(cf, asm_facts(narrowed, 16384), None, None))
    lay = next(s for s in res["suspects"] if s["suspect"] == "lost_layout")
    assert lay["verdict"] == "SUSPECTED" and lay["evidence"]["shifts"], lay

    # 4. THE FALSE POSITIVE THIS TOOL MUST NOT MAKE: an accumulator loop with 3 iter_args and
    #    num_stages=1 is NOT pipelined, so lost_pipeline must be CLEARED, not suspected.
    champ_ttgir = """
      tt.num_stages = 1
      %0 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z, %p1 = %a, %p2 = %b) {
        %l = amdg.buffer_load %ptr[%off] : tensor<128x128xbf16>
      }
    """
    anch_ttgir = "%0 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) { }"
    ct, at = ttgir_pipeline_facts(champ_ttgir), ttgir_pipeline_facts(anch_ttgir)
    assert ct["memdesc_index"] == 0 and ct["local_store"] == 0 and ct["num_stages"] == 1, ct
    res = evaluate(2.8966, 4.0754, 0.95, attribute(cf, af, ct, at))
    pipe = next(s for s in res["suspects"] if s["suspect"] == "lost_pipeline")
    assert pipe["verdict"] == "CLEARED", pipe
    assert "no pipeline to lose" in pipe["evidence"]["reason"], pipe

    # 5. a genuinely pipelined champion against a flat anchor DOES fire
    piped = "tt.num_stages = 2\n ttg.memdesc_index %x\n ttg.local_store %y\n"
    res = evaluate(2.0, 4.0, 0.95, attribute(cf, af, ttgir_pipeline_facts(piped), at))
    pipe = next(s for s in res["suspects"] if s["suspect"] == "lost_pipeline")
    assert pipe["verdict"] == "SUSPECTED", pipe

    # 6. missing artifacts are UNKNOWN, never CLEARED -- a dark signal is not a clean one
    res = evaluate(2.0, 4.0, 0.95, attribute(None, None, None, None))
    assert res["unattributed"] == ["lost_pipeline", "lost_layout", "lost_RA"], res
    assert "No suspect fired" in res["note"], res["note"]

    print("parity_gate selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion-ms", type=float)
    ap.add_argument("--anchor-ms", type=float, help="the CURRENT Gluon number, not only round 1's")
    ap.add_argument("--champion-asm")
    ap.add_argument("--anchor-asm")
    ap.add_argument("--champion-ttgir")
    ap.add_argument("--anchor-ttgir")
    ap.add_argument("--champion-lds", type=int,
                    help="shared bytes/WG from the Triton cache metadata's `shared` field. The "
                         "asm's LDSByteSize is a structural 0 on Triton kernels, so without this "
                         "the LDS half of lost_layout is untested.")
    ap.add_argument("--anchor-lds", type=int)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.champion_ms is None or a.anchor_ms is None:
        ap.error("--champion-ms and --anchor-ms are required (or use --selftest)")

    def _read(p):
        return Path(p).read_text(errors="ignore") if p else None

    ca, aa = _read(a.champion_asm), _read(a.anchor_asm)
    ct, at = _read(a.champion_ttgir), _read(a.anchor_ttgir)
    suspects = attribute(asm_facts(ca, a.champion_lds) if ca else None,
                         asm_facts(aa, a.anchor_lds) if aa else None,
                         ttgir_pipeline_facts(ct) if ct else None,
                         ttgir_pipeline_facts(at) if at else None)
    res = evaluate(a.champion_ms, a.anchor_ms, a.threshold, suspects)
    print(_fmt(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.json}")
    return 0 if res["gate"] == "CLEARED" else 2


if __name__ == "__main__":
    sys.exit(main())
