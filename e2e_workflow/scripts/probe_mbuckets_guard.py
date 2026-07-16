#!/usr/bin/env python3
"""Decide whether a kernel's decode m_buckets are REAL (probe-measured) or the
synthesized [1, CONC] fallback the workflow injects when no probe ran.

Pure stdlib. Mirrored by an inline check in e2e_workflow.js (same one-line rule);
kept here as an importable pure function so it is unit-testable without Node.
"""


def is_synthesized_fallback(decode_m_buckets, conc):
    """True iff decode_m_buckets carries NO measured M beyond the synthesized guess.
    The workflow injects DECODE_M_BUCKETS=[1, CONC]; a probe run adds real M values
    (e.g. conc*top_k). So: fallback == the bucket set is a subset of {1, conc} (incl. empty)."""
    if not decode_m_buckets:
        return True
    synth = {1, conc}
    return all(m in synth for m in decode_m_buckets)


def classify_mbuckets(meta, conc):
    """meta: parsed meta.json dict. Returns 'measured' | 'synthesized_fallback' | 'missing'."""
    if "decode_m_buckets" not in meta:
        return "missing"
    return ("synthesized_fallback"
            if is_synthesized_fallback(meta["decode_m_buckets"], conc)
            else "measured")
