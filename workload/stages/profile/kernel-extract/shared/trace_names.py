"""Forward-span name dialects used by live trace verification."""
import re


def _seg(name, tag):
    """(lead, tokens, kv) for a 'context'/'generation' segment. Handles both annotation
    dialects: GEAK  context_<nreq>(<ntok>)  and  ..._<batch>(sq<q>sk<kv>...)."""
    m = re.search(tag + r"_(\d+)\(([^)]*)\)", name)
    if not m:
        return (0, 0, 0)
    lead = int(m.group(1))
    inner = m.group(2)
    sq = re.search(r"sq(\d+)", inner)
    sk = re.search(r"sk(\d+)", inner)
    if sq:
        return (lead, int(sq.group(1)), int(sk.group(1)) if sk else 0)
    return (lead, int(inner) if inner.isdigit() else 0, 0)


_SGLANG_STEP_RE = re.compile(
    r"step\[(?P<mode>[A-Z_0-9]+) bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?\]")


_SGLANG_MODE_PREFILL = {"EXTEND": True, "DRAFT_EXTEND_V2": True, "DRAFT_EXTEND": True,
                        "DECODE": False, "TARGET_VERIFY": False, "VERIFY": False}


def _parse_sglang_step(name):
    m = _SGLANG_STEP_RE.search(name)
    if not m:
        return None
    is_prefill = _SGLANG_MODE_PREFILL.get(m.group("mode"))
    if is_prefill is None:
        return None
    bs = int(m.group("bs"))
    toks = int(m.group("toks") or 0)
    # Prefill carries its token count; decode's "shape" is the batch.
    return (is_prefill, toks, 0 if is_prefill else bs, 0)


def _parse_vllm_detailed(name):
    if "context_" not in name or "generation_" not in name:
        return None
    ctx = _seg(name, "context")
    gen = _seg(name, "generation")
    return (ctx[1] > 0, ctx[1] + gen[1] if ctx[1] > 0 else 0,
            gen[0] or gen[1], gen[2])


STEP_DIALECTS = [
    ("sglang-step", lambda n: "step[" in n, _parse_sglang_step),
    ("vllm-detailed", lambda n: n.startswith("execute_"), _parse_vllm_detailed),
]


def _classify_step(name):
    """-> (is_prefill, ctx_tokens, decode_batch, kv) for a forward span, else None."""
    for _dname, detect, parse in STEP_DIALECTS:
        if detect(name):
            c = parse(name)
            if c is not None:
                return c
    return None
