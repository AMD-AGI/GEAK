#!/usr/bin/env python3
"""Shared parser for SGLang `step[<ForwardMode> bs=N ...]` profiler spans.

SGLang's ModelRunner.forward names every forward `step[{mode.name} bs=N]`
(`step[EXTEND bs=N toks=T]` for plain extend).  Without speculative decoding
only EXTEND and DECODE appear.  With EAGLE/NEXTN/MTP speculative decoding the
target model's per-token step is TARGET_VERIFY (it verifies bs x draft_tokens
positions through the full main-layer stack), and the draft model runs as
DRAFT_EXTEND / DRAFT_EXTEND_V2 or as unannotated draft CUDA-graph replays.

Everything that classifies a step into a model phase must use this module so a
speculative run cannot silently lose its decode phase again.
"""
import re


STEP_RE = re.compile(
    r"^step\[(?P<mode>[A-Z][A-Z0-9_]*)\s+bs=(?P<bs>\d+)"
    r"(?:\s+toks=(?P<toks>\d+))?\]$")

# Main-stack (target model) steps, keyed by ForwardMode name.  Without
# speculative decoding the target generates with DECODE; with it the target
# generates with TARGET_VERIFY (bs requests x draft tokens), phase "verify".
TARGET_PREFILL_MODES = ("EXTEND", "MIXED", "SPLIT_PREFILL")
TARGET_DECODE_MODES = ("DECODE",)
TARGET_VERIFY_MODES = ("TARGET_VERIFY",)
# Draft-model steps run the speculative draft stack (MTP/NEXTN/EAGLE head),
# not the main decoder layers.  They are recognised so they can be excluded
# from main-layer tables and reported, never mapped onto main layers.
DRAFT_MODES = ("DRAFT_EXTEND", "DRAFT_EXTEND_V2")
SPECULATIVE_MODES = ("TARGET_VERIFY",) + DRAFT_MODES

PHASE_BY_MODE = dict(
    [(mode, "prefill") for mode in TARGET_PREFILL_MODES]
    + [(mode, "decode") for mode in TARGET_DECODE_MODES]
    + [(mode, "verify") for mode in TARGET_VERIFY_MODES]
    + [(mode, "draft") for mode in DRAFT_MODES])


def parse_step(name):
    """Return a step description or None for non-step / unknown-mode names.

    Result: {"mode", "phase", "batch_size", "tokens", "speculative"} where
    phase is "prefill", "decode", "verify" or "draft".
    """
    if not isinstance(name, str):
        return None
    match = STEP_RE.match(name)
    if not match:
        return None
    mode = match.group("mode")
    phase = PHASE_BY_MODE.get(mode)
    if phase is None:
        return None
    return {
        "mode": mode,
        "phase": phase,
        "batch_size": int(match.group("bs")),
        "tokens": int(match.group("toks") or 0),
        "speculative": mode in SPECULATIVE_MODES,
    }


# The target's token-generation phase: plain "decode", or "verify" when
# speculative decoding runs the target as TARGET_VERIFY.
GENERATION_PHASES = ("decode", "verify")


def is_generation_phase(phase):
    return str(phase or "").lower() in GENERATION_PHASES


def generation_phase(phases):
    """The generation phase present in `phases` ("verify" wins), else "decode"."""
    phases = {str(p or "").lower() for p in phases or ()}
    return "verify" if "verify" in phases else "decode"


def target_phases(phases):
    """Required target phases for a run whose observed phases are `phases`."""
    return ["prefill", generation_phase(phases)]


def canonical_phase(value):
    """Map ForwardMode / legacy phase spellings onto prefill|decode|verify|draft."""
    value = str(value or "").strip()
    upper = value.upper()
    if upper in PHASE_BY_MODE:
        return PHASE_BY_MODE[upper]
    lower = value.lower()
    return {"extend": "prefill", "prompt": "prefill",
            "generation": "decode", "target_verify": "verify",
            "draft_extend": "draft", "draft_extend_v2": "draft"}.get(
                lower, lower)
