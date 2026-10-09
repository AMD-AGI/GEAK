#!/usr/bin/env python3
"""Build Pattern/Phase/Layer ordered device-event tables from one clean trace."""
import argparse
import bisect
import collections
import difflib
import gzip
import hashlib
import json
import math
import os
import re
import statistics

import parse_profile
import triton_launch_probe
import sglang_step_modes


DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAYER_RE = re.compile(r"(?:layers?|h|blocks?)[./_\[](\d+)", re.IGNORECASE)
MODULE_LAYER_RE = re.compile(
    r"^nn\.Module:\s+.*DecoderLayer_(\d+)$", re.IGNORECASE)
# Python stack frames under SGLang's speculative workers.  The innermost such
# frame enclosing a step's CPU annotation says whether the step ran the draft
# model (e.g. `_draft_extend_for_prefill`, `draft_forward`) or the target model
# (`verify`, `forward_batch_generation`).  SGLang labels the draft prefill
# `step[EXTEND ...]`, so the step name alone cannot tell them apart.
SPECULATIVE_FRAME_RE = re.compile(
    r"/speculative/[^()]*\.py\(\d+\):\s*(?P<func>\w+)")
DRAFT_FUNC_RE = re.compile(r"draft", re.IGNORECASE)


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _layer_id(event):
    args = event.get("args") or {}
    fields = [event.get("name", "")]
    fields.extend(str(args.get(key, "")) for key in (
        "Module Hierarchy", "Call stack", "Python parent id", "op_path"))
    text = " ".join(fields)
    match = LAYER_RE.search(text)
    return int(match.group(1)) if match else None


def _phase_name(tag):
    return {"P": "prefill", "V": "verify"}.get(tag, "decode")


def _legacy_step_spans(events):
    """vLLM execute_* GPU windows, one per step: [(ts, end, tag, tokens, batch, name)].

    vLLM 0.30 emits the step's `gpu_user_annotation` once per GPU stream that ran work in
    it (MiniMax-M3 TP8: stream 4 every step, stream 2 on some), same name, overlapping
    windows. Read one per annotation, as parse_profile does, and a single step became two
    (legacy-0 held 4 of 60 prefill layers, legacy-1 the other 56) and nothing mapped.
    Same-name windows that overlap are the same step: take their union. Consecutive
    steps never overlap, so this cannot merge two real steps.
    """
    raw = []
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "gpu_user_annotation":
            continue
        name = event.get("name")
        if (not isinstance(name, str) or not name.startswith("execute_")
                or event.get("ts") is None or event.get("dur") is None):
            continue
        kind = parse_profile._classify_step(name)
        if kind:
            raw.append((event["ts"], event["ts"] + event["dur"],
                        "P" if kind[0] else "D", kind[1], kind[2], name))
    raw.sort()
    merged = []
    open_by_name = {}
    for span in raw:
        index = open_by_name.get(span[5])
        if index is not None and span[0] < merged[index][1]:
            lo, hi = merged[index][0], max(merged[index][1], span[1])
            merged[index] = (lo, hi) + merged[index][2:]
            continue
        open_by_name[span[5]] = len(merged)
        merged.append(span)
    merged.sort()
    return merged


def _launch_voted_host_windows(events, legacy_spans):
    """{span index: (lo, hi)}: the execute_* host window that launched a step's kernels.

    Each device event inside the step's device window votes, through its correlation
    id, for the host annotation its launch fell in; the window with a strict majority
    wins. Names cannot pair them: vLLM before 0.30 names a step only by its request
    counts, so a chunked prefill repeats `execute_context_2(8181)_generation_11(11)`
    step after step, and with the host more than a step ahead the latest same-name
    window before the device start is the NEXT step's (Kimi-K2.5 TP8: 4 of 17
    prefill steps shifted, each cutting one host pass across two device steps).
    """
    host = sorted(
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if isinstance(event, dict) and event.get("cat") == "user_annotation"
        and isinstance(event.get("name"), str)
        and event["name"].startswith("execute_")
        and event.get("ts") is not None and event.get("dur") is not None)
    if not host or not legacy_spans:
        return {}
    host_starts = [item[0] for item in host]
    launch_ts = {}
    for event in events:
        if (isinstance(event, dict)
                and event.get("cat") in ("cuda_runtime", "hip_runtime")
                and event.get("ts") is not None):
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None:
                launch_ts[correlation] = event["ts"]
    device_starts = [span[0] for span in legacy_spans]
    votes = [collections.Counter() for _ in legacy_spans]
    for event in events:
        if (not isinstance(event, dict) or event.get("cat") not in DEVICE_CATEGORIES
                or event.get("ts") is None):
            continue
        index = bisect.bisect_right(device_starts, event["ts"]) - 1
        if index < 0 or event["ts"] >= legacy_spans[index][1]:
            continue
        launched = launch_ts.get((event.get("args") or {}).get("correlation"))
        if launched is None:
            continue
        position = bisect.bisect_right(host_starts, launched) - 1
        if position >= 0 and launched < host[position][1]:
            votes[index][position] += 1
    paired = {}
    for index, counter in enumerate(votes):
        if counter:
            position, count = counter.most_common(1)[0]
            if 2 * count > sum(counter.values()):
                paired[index] = host[position]
    return paired


def _collect_step_spans(events):
    """Recognize both legacy execute_* and current SGLang step[...] spans."""
    return _collect_step_spans_with_draft(events)[0]


def _speculative_frames(events):
    """(start, end, tid, func) for python frames inside speculative workers."""
    frames = []
    for event in events:
        if (not isinstance(event, dict)
                or event.get("cat") != "python_function"
                or event.get("ts") is None or event.get("dur") is None):
            continue
        match = SPECULATIVE_FRAME_RE.search(str(event.get("name", "")))
        if match:
            frames.append((event["ts"], event["ts"] + event["dur"],
                           event.get("tid"), match.group("func")))
    return frames


def _draft_stack_evidence(cpu_window, frames):
    """Innermost speculative-worker frame enclosing a CPU step window."""
    if cpu_window is None:
        return None
    lo, hi, tid = cpu_window
    best = None
    for start, end, frame_tid, func in frames:
        if tid is not None and frame_tid is not None and frame_tid != tid:
            continue
        if start <= lo and hi <= end:
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end, func)
    return best[2] if best else None


def _collect_step_spans_with_draft(events):
    """Return (main_stack_spans, draft_spans).

    Draft spans are speculative-decoding draft-model forwards: an explicit
    DRAFT_EXTEND* mode, or any step whose innermost speculative-worker stack
    frame is a draft function.  They run the MTP/NEXTN/EAGLE head, never the
    main decoder stack, so they are kept out of the main-layer spans and
    reported separately.  TARGET_VERIFY is the target model's decode.
    """
    spans = []
    step_names = {}
    legacy_ids = set()
    legacy = _legacy_step_spans(events)
    voted = {"legacy-%d" % index: window for index, window
             in _launch_voted_host_windows(events, legacy).items()}
    for index, span in enumerate(legacy):
        spans.append((span[0], span[1], span[2], span[3], span[4],
                      "legacy-%d" % index, "legacy_execute"))
        step_names["legacy-%d" % index] = span[5]
        legacy_ids.add("legacy-%d" % index)
    for raw_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "gpu_user_annotation":
            continue
        name = event.get("name")
        step = sglang_step_modes.parse_step(name)
        if step is None or event.get("ts") is None or event.get("dur") is None:
            continue
        tag = {"prefill": "P", "decode": "D", "verify": "V",
               "draft": "S"}[step["phase"]]
        step_id = "step-%d" % raw_index
        step_names[step_id] = name
        spans.append((
            event["ts"], event["ts"] + event["dur"], tag,
            step["tokens"], step["batch_size"], step_id,
            "sglang_step_annotation" if step["mode"] in ("EXTEND", "DECODE")
            else "sglang_step_annotation:%s" % step["mode"]))

    # SGLang emits each step[...] annotation TWICE: a CPU-side `user_annotation`
    # spanning the host wall-clock of the step, and a GPU-side
    # `gpu_user_annotation` spanning only the device work.  The spans above are
    # the GPU ones, which is right for device rows but WRONG for CPU-side
    # python_function `nn.Module:` spans.
    #
    # In prefill the two windows are nearly the same length (the GPU is the
    # bottleneck), so using the GPU window to contain CPU module spans happened
    # to work.  In eager decode the host runs ~40x longer than the device
    # (1794ms CPU vs 45ms GPU on DSR1/MI308X), so 52 of 61 DecoderLayer spans
    # fall outside the GPU window and `full_passes` drops to 0. Historically
    # that triggered unsafe stage-recurrence segmentation; incomplete module
    # passes now stay unresolved.
    #
    # Carry the CPU window alongside each span (indices 7,8) so CPU-side
    # containment can use it.  Falls back to the GPU window when a trace has no
    # CPU-side annotation (older captures).  vLLM's execute_* has the same CPU/GPU
    # pair; an eager vLLM model (MiniMax-M3 on ROCm: no torch.compile) has
    # `nn.Module:` spans in prefill, which need the CPU window just the same.
    cpu_by_name = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "user_annotation":
            continue
        name = event.get("name")
        if (not isinstance(name, str)
                or not (sglang_step_modes.parse_step(name) is not None
                        or name.startswith("execute_"))
                or event.get("ts") is None or event.get("dur") is None):
            continue
        cpu_by_name.setdefault(name, []).append(
            (event["ts"], event["ts"] + event["dur"], event.get("tid")))
    for key in cpu_by_name:
        cpu_by_name[key].sort(key=lambda item: (item[0], item[1]))
    frames = _speculative_frames(events)
    # One CPU annotation belongs to one GPU span.  Same-named steps (target
    # and draft prefill are both `step[EXTEND bs=1 toks=N]`) pair in time
    # order, so each CPU window is consumed once.
    used_cpu = set()
    widened = []
    drafts = []
    for span in sorted(spans, key=lambda item: (item[0], item[1], item[5])):
        cpu_lo, cpu_hi = span[0], span[1]
        cpu_window = None
        name = step_names.get(span[5], _step_span_name(span))
        if span[5] in legacy_ids:
            # vLLM: the host annotation alone, paired by name (it carries the step's
            # cumulative sequence lengths) as the latest one opened before the device
            # window. Not by overlap: under --async-scheduling the host runs a whole
            # step ahead (~490ms in MiniMax-M3 prefill), so the pair never overlaps.
            # Not a union: eager decode's device work trails the host by ~40ms, and
            # a union reached into the next step's host window, so the last layer's
            # donor scope swallowed the next step's input prep, embedding and layer-0
            # head (MiniMax-M3 TP8) and no donor pass matched any decode step.
            # The launch vote decides when the trace carries correlations; the name
            # is the fallback (unique on 0.30, whose names carry sequence lengths).
            # vLLM has no speculative draft step annotation, so no draft check here.
            opened = [(lo, hi) for lo, hi, _tid in cpu_by_name.get(name, ())
                      if lo <= span[0]]
            if span[5] in voted:
                cpu_lo, cpu_hi = voted[span[5]]
            elif opened:
                cpu_lo, cpu_hi = opened[-1]
            widened.append(tuple(span) + (cpu_lo, cpu_hi))
            continue
        for index, (lo, hi, tid) in enumerate(cpu_by_name.get(name, ())):
            if (name, index) in used_cpu:
                continue
            # Pair a CPU window with the GPU window it encloses/overlaps, or
            # with the earliest unused CPU launch preceding the GPU work.
            if (lo <= span[0] < hi or (span[0] <= lo and hi <= span[1])
                    or hi <= span[0]):
                used_cpu.add((name, index))
                cpu_window = (lo, hi, tid)
                if lo <= span[0] < hi or (span[0] <= lo and hi <= span[1]):
                    cpu_lo, cpu_hi = min(cpu_lo, lo), max(cpu_hi, hi)
                break
        speculative_func = _draft_stack_evidence(cpu_window, frames)
        is_draft = span[2] == "S" or bool(
            speculative_func and DRAFT_FUNC_RE.search(speculative_func))
        if is_draft:
            drafts.append({
                "step_id": span[5],
                "name": name,
                "ts": span[0],
                "end": span[1],
                "batch_size": span[4],
                "input_tokens": span[3],
                "evidence": (
                    "draft_step_mode" if span[2] == "S"
                    else "speculative_stack_frame:%s" % speculative_func),
                "cpu_ts": cpu_window[0] if cpu_window else None,
                "cpu_end": cpu_window[1] if cpu_window else None,
            })
            continue
        widened.append(tuple(span) + (cpu_lo, cpu_hi))
    spans = widened
    spans.sort(key=lambda item: (item[0], item[1], item[5]))
    return spans, drafts


def _step_span_name(span):
    """Reconstruct the sglang annotation name of a legacy/plain step span."""
    kind = {"P": "EXTEND", "V": "TARGET_VERIFY"}.get(span[2], "DECODE")
    if kind == "EXTEND":
        return "step[EXTEND bs=%d toks=%d]" % (span[4], span[3])
    return "step[%s bs=%d]" % (kind, span[4])


def has_module_layer_spans(path):
    """True when a trace carries `nn.Module: ...DecoderLayer_N` python spans.

    CUDA-graph-replayed decode traces do not: the whole layer stack replays as
    one opaque graph launch, so there are no per-layer python frames and no
    External ids. Used to decide whether graph-construction capture can supply
    decode layer boundaries the clean trace cannot.
    """
    try:
        events = _load_events(path)
    except Exception:
        return False
    for event in events:
        if (isinstance(event, dict)
                and event.get("cat") == "python_function"
                and MODULE_LAYER_RE.match(str(event.get("name", "")))):
            return True
    return False


def _cpu_step_index(spans):
    """(spans, starts) sorted by CPU window start, for CPU-side containment."""
    usable = [span for span in spans if len(span) >= 9]
    usable.sort(key=lambda item: (item[7], item[8]))
    return usable, [span[7] for span in usable]


def _cpu_step_at(ts, spans, starts):
    if ts is None or not spans:
        return None
    pos = bisect.bisect_right(starts, ts) - 1
    while pos >= 0:
        if ts < spans[pos][8]:
            return spans[pos]
        pos -= 1
    return None


STAGE_RULESET_VERSION = "semantic-stage-v4"
# Kernel names are snake_case, and `\b` treats `_` as a word character, so `\bmoe\b` never
# matched fused_moe_kernel and `\bmm\b` never matched _w8a8_triton_block_scaled_mm: the
# largest expert and FP8 GEMM rows on both sglang and vllm landed as `unknown`. These
# tokens are bounded by any non-alphanumeric instead.
STAGE_RULES = (
    ("communication.collective", "communication",
     r"all.?reduce|reduce.?scatter|all.?gather|nccl|rccl|quickreduce|cross_device"),
    ("norm.layer", "norm", r"rms.?norm|layer.?norm|l2norm"),
    ("position.rope", "rope", r"rope|rotary"),
    ("attention.linear", "linear_attn",
     r"fused_qkvzba|causal_conv1d|gdn_gating|gated_delta|delta_rule|"
     r"recompute_w_u|chunk_local_cumsum|chunk_fwd_kernel_o|"
     r"state_passing|wv_splitk_small"),
    # SGLang triton extend/decode attention launch as bare `_fwd_kernel*`;
    # speculative TARGET_VERIFY decode runs the extend kernel.
    ("attention.full_or_mla", "attn",
     r"fmha|attention|attn|paged|mla_|^_fwd_kernel(_stage\d+)?$"),
    ("router.topk", "topk", r"topk|routing|router|gate_kernel"),
    ("experts.moe", "moe", r"(?<![a-z0-9])moe(?![a-z0-9])|expert|sorting|fmoe"),
    ("linear.gemm", "gemm", r"gemm|cijk|tensile|matmul|(?<![a-z0-9])mm(?![a-z0-9])"),
    ("cache.kv", "kv_cache", r"cache|index_put"),
    ("activation", "activation", r"silu|gelu|swiglu|act_and_mul"),
    ("quantization", "quant", r"quant|dequant|float8|fp8"),
    ("elementwise", "elementwise", r"elementwise|copy|cast|fill|add|mul"),
)


def _stage_detail(name, category, parent_name=""):
    if category in ("gpu_memcpy", "gpu_memset"):
        return "memory", "event.memory", "event_category"
    value = name.lower()
    for rule_id, stage, regex in STAGE_RULES:
        if re.search(regex, value):
            return stage, rule_id, "kernel_name"
    parent_value = parent_name.lower()
    # `gdn`: vLLM registers the whole gated-delta core as vllm::qwen_gdn_attention_core.
    if re.search(r"gated.?delta|linear.?attention|causal.?conv|(?<![a-z0-9])gdn(?![a-z0-9])",
                 parent_value):
        return "linear_attn", "attention.linear.parent", "parent_operator"
    # The kernel-name rules key off names like fmha/paged/mla_. A backend whose
    # kernel is generically named slips through: vLLM's TRITON_ATTN launches
    # `_fwd_kernel` under `vllm::unified_attention_with_output`, and that landed
    # as `unknown` -- so the model's LARGEST prefill row (298.5us on Qwen3.5-2B)
    # was not a donor, and Phase 2.1's escalation gate then demanded a fusion
    # candidate for the attention operator itself. The parent op is authoritative
    # evidence: it is the registered op the kernel actually ran under.
    if re.search(r"unified.?attention|attention.?with.?output|paged.?attention"
                 r"|flash.?attn|fmha", parent_value):
        return "attn", "attention.full.parent", "parent_operator"
    return "unknown", "unresolved", "unresolved"


def _stage(name, category):
    return _stage_detail(name, category)[0]


_PHASE_TAG_RE = re.compile(
    r"^(?P<stem>.*-TP-\d+)-(?P<phase>EXTEND|DECODE)(?P<suffix>\.trace\.json.*)$")


def _phase_tag(path):
    """Return the EXTEND/DECODE tag SGLang's profile_by_stage put in the name."""
    match = _PHASE_TAG_RE.match(os.path.basename(path))
    return match.group("phase") if match else None


def _sibling_phase_traces(path):
    """Sibling traces of the same rank and profiler session, by phase tag.

    SGLang's `profile_by_stage` writes `<stem>-TP-<rank>-EXTEND.trace.json.gz`
    and `<stem>-TP-<rank>-DECODE.trace.json.gz` as separate files.  Analysing
    one and never mentioning the other silently halves phase coverage (B2).
    """
    match = _PHASE_TAG_RE.match(os.path.basename(path))
    if not match:
        return {}
    directory = os.path.dirname(os.path.abspath(path))
    found = {}
    for phase in ("EXTEND", "DECODE"):
        candidate = os.path.join(directory, "%s-%s%s" % (
            match.group("stem"), phase, match.group("suffix")))
        if os.path.exists(candidate):
            found[phase] = candidate
    return found


def _resolve_trace_paths(trace_path, auto_sibling=True):
    """Return the exact trace set consumed by mapping and its adopted files."""
    trace_paths = (
        [trace_path] if isinstance(trace_path, str) else list(trace_path))
    if not trace_paths:
        raise ValueError("at least one trace is required")
    adopted_siblings = []
    if auto_sibling:
        known = {os.path.abspath(path) for path in trace_paths}
        for path in list(trace_paths):
            for phase, sibling in sorted(_sibling_phase_traces(path).items()):
                if os.path.abspath(sibling) not in known:
                    known.add(os.path.abspath(sibling))
                    trace_paths.append(sibling)
                    adopted_siblings.append({"phase": phase, "path": sibling})
    return trace_paths, adopted_siblings


def _load_events(path):
    with _open(path) as fh:
        data = json.load(fh)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def _load_events_multi(paths):
    """Concatenate several phase traces into one temporally ordered stream.

    Phase traces from a single profiler session are timestamp-disjoint and
    ordered (EXTEND strictly precedes DECODE), so concatenating by first
    timestamp preserves the real execution order.
    """
    streams = []
    for path in paths:
        events = _load_events(path)
        stamps = [event.get("ts") for event in events
                  if isinstance(event, dict) and event.get("ts") is not None]
        streams.append((min(stamps) if stamps else 0.0, path, events))
    streams.sort(key=lambda item: item[0])
    merged = []
    for _, _, events in streams:
        merged.extend(events)
    return merged


def _triton_launch_evidence(events):
    """{correlation: launch-probe evidence} for kernels a Triton launch probe annotated.

    The probe's annotation and the launch runtime event share a thread, and Kineto
    links neither the annotation nor the kernel by External id, so the kernel is
    attributed through its runtime event's correlation to the probe annotation
    whose span holds that launch. See triton_launch_probe.py.
    """
    by_tid = collections.defaultdict(list)
    for index, event in enumerate(events):
        if (not isinstance(event, dict) or event.get("cat") != "user_annotation"
                or event.get("ts") is None or event.get("dur") is None):
            continue
        parsed = triton_launch_probe.parse_annotation(event.get("name"))
        if parsed:
            by_tid[event.get("tid")].append(
                (event["ts"], event["ts"] + event["dur"], parsed, index))
    if not by_tid:
        return {}
    starts = {}
    for tid, spans in by_tid.items():
        spans.sort(key=lambda item: item[0])
        starts[tid] = [item[0] for item in spans]
    evidence = {}
    for event in events:
        if (not isinstance(event, dict)
                or event.get("cat") not in ("cuda_runtime", "hip_runtime")):
            continue
        correlation = (event.get("args") or {}).get("correlation")
        spans = by_tid.get(event.get("tid"))
        if correlation is None or not spans or event.get("ts") is None:
            continue
        pos = bisect.bisect_right(starts[event["tid"]], event["ts"]) - 1
        if pos < 0 or event["ts"] > spans[pos][1]:
            continue
        kernel_name, dims, types, names = spans[pos][2]
        evidence[correlation] = {"kernel_name": kernel_name, "input_dims": dims,
                                 "input_types": types, "operand_names": names,
                                 "event_index": spans[pos][3]}
    return evidence


def _launched_kernel_name(name):
    return name[:-len(".kd")] if name.endswith(".kd") else name


def _cpu_evidence(events):
    by_ext = {}
    scopes = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "cpu_op":
            continue
        args = event.get("args") or {}
        evidence = {
            "name": str(event.get("name", "unresolved")),
            "external_id": args.get("External id"),
            "input_dims": args.get("Input Dims") or [],
            "input_types": args.get("Input type") or [],
            "layer_id": _layer_id(event),
            "ts": event.get("ts"),
            "end": ((event.get("ts") or 0) + (event.get("dur") or 0)),
            "event_index": index,
        }
        ext = evidence["external_id"]
        if ext is not None:
            old = by_ext.get(ext)
            if old is None or (not old["input_dims"] and evidence["input_dims"]):
                by_ext[ext] = evidence
        if evidence["layer_id"] is not None and event.get("ts") is not None and event.get("dur") is not None:
            scopes.append(evidence)
    scopes.sort(key=lambda item: (item["ts"], item["end"]))
    return by_ext, scopes, [item["ts"] for item in scopes]


def _step_at(ts, spans, starts):
    if ts is None or not spans:
        return None
    pos = bisect.bisect_right(starts, ts) - 1
    if pos >= 0 and ts < spans[pos][1]:
        return spans[pos]
    return None


def _module_layer_scopes(events, spans, pattern_doc, draft_windows=()):
    """Resolve outer DecoderLayer python spans to global layer ordinals.

    Numeric suffixes are class-local for some hybrid model implementations, so
    execution order inside each full N-layer pass is authoritative. The suffix
    remains validation evidence only.
    """
    expected_count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    if expected_count <= 0:
        return [], []
    pattern_by_layer = _pattern_index(pattern_doc)
    # CPU-side module spans must be located against the CPU-side step window.
    cpu_spans, span_starts = _cpu_step_index(spans)
    if not cpu_spans:
        cpu_spans, span_starts = spans, [span[0] for span in spans]
    candidates = []
    for raw_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "python_function":
            continue
        name = str(event.get("name", ""))
        match = MODULE_LAYER_RE.match(name)
        if not match or event.get("ts") is None or event.get("dur") is None:
            continue
        # The CPU dispatches the draft forward while the target's GPU work
        # (whose window widens the target step) is still running; a draft
        # DecoderLayer span there would be counted as a 65th target layer.
        if any(lo <= event["ts"] < hi for lo, hi in draft_windows):
            continue
        step = _cpu_step_at(event["ts"], cpu_spans, span_starts)
        if step is None:
            continue
        candidates.append({
            "name": name,
            "class_local_id": int(match.group(1)),
            "ts": event["ts"],
            "end": event["ts"] + event["dur"],
            "event_index": raw_index,
            "step_id": step[5],
            "phase": _phase_name(step[2]),
        })
    by_step = {}
    for candidate in candidates:
        by_step.setdefault(candidate["step_id"], []).append(candidate)
    scopes = []
    diagnostics = []
    for step_id, values in sorted(by_step.items()):
        values.sort(key=lambda item: (item["ts"], item["end"]))
        full_passes = len(values) // expected_count
        remainder = len(values) % expected_count
        diagnostics.append({
            "step_id": step_id,
            "candidate_count": len(values),
            "expected_layer_count": expected_count,
            "full_passes": full_passes,
            "remainder": remainder,
        })
        for pass_index in range(full_passes):
            chunk = values[pass_index * expected_count:(pass_index + 1) * expected_count]
            for layer_id, item in enumerate(chunk):
                item = dict(item)
                item.update({
                    "layer_id": layer_id,
                    "pattern_id": (pattern_by_layer.get(layer_id) or {}).get("pattern_id"),
                    "pass_index": pass_index,
                    "layer_instance_id": "%s:pass-%d:layer-%d" % (
                        step_id, pass_index, layer_id),
                    # Class-local numeric suffixes and class names are not a
                    # model-independent Pattern taxonomy.  The complete count
                    # and execution order establish the global layer ordinal;
                    # the raw name remains available as audit evidence.
                    "type_validation": "not_applicable",
                })
                scopes.append(item)
    scopes.sort(key=lambda item: (item["ts"], item["end"]))
    if not scopes:
        # No module frame resolved a single layer (torch.compile erases them). Try the
        # anchor the Patterns themselves declare before leaving the step unresolved.
        fallback, fallback_diag = _dispatch_anchor_scopes(events, spans, pattern_doc)
        diagnostics.append(dict(fallback_diag,
                                source=("declared_dispatch_op_span" if fallback
                                        else "declared_dispatch_op_span_declined")))
        if fallback:
            return fallback, diagnostics
    return scopes, diagnostics


def _declared_dispatch_branches(pattern_doc):
    """{op_name: [pattern_id, ...]} from the AGENT's structural signatures.

    `runtime_dispatch_branch` is a REQUIRED signature field (see
    validate_structural_patterns.REQUIRED_SIGNATURE_FIELDS) that the agent fills from
    runtime SOURCE analysis. Using it as a layer anchor therefore keeps the Phase-1
    contract intact -- the agent defines structure, this code only validates it against
    the trace; the trace never defines a Pattern.
    """
    branches = {}
    for pattern in pattern_doc.get("patterns", []):
        signature = pattern.get("structural_signature") or {}
        branch = str(signature.get("runtime_dispatch_branch") or "").strip()
        if branch:
            branches.setdefault(branch, []).append(pattern.get("pattern_id"))
    return branches


def _dispatch_anchor_scopes(events, spans, pattern_doc):
    """Per-layer scopes cut at the dispatch op each Pattern declares it routes through.

    WHY THIS EXISTS. Under torch.compile the per-layer `nn.Module` python frames are
    gone -- the layer stack lives inside a compiled graph. vLLM's V1 engine compiles by
    default, so on vllm the module-span path finds NOTHING and every step is left
    boundary_unresolved. Kernel-stage recurrence cannot stand in for it: on a HYBRID
    model it is WRONG, not merely incomplete. Measured on Qwen3.5-2B (24 layers = 18
    gated-delta + 6 full-attention, full_attention_interval 4) the `attn` stage fires
    only in the 6 full-attention layers, at a perfectly regular spacing, so a repeat
    heuristic produced 6 "layer bodies" each holding FOUR real layers.

    The dispatch ops do not have that problem: they survive compilation precisely BECAUSE
    they are the graph's splitting points, and every layer kind has one. On that same
    trace `vllm::qwen_gdn_attention_core` (x18) + `vllm::unified_attention_with_output`
    (x6) = exactly 24 anchors per step.

    DECLINES rather than guesses. Returns [] unless ALL of:
      * every Pattern declares a dispatch branch -- one that does not is a layer kind with
        no anchor, which is precisely the 6-of-24 failure above;
      * the step carries exactly `num_hidden_layers_main` anchor events;
      * the anchor ORDER agrees with the declared per-layer Patterns (the i-th anchor's op
        is the one the Pattern owning layer i declared).
    That last check is what makes this evidence rather than an assumption: it fails loudly
    on a model whose layers do not execute in config order.

    Boundary semantics: layer i spans [anchor_i, anchor_i+1). Like the existing
    anchor_stage_rotation path this is phase-shifted within the layer (the segment holds
    layer i's core and tail plus layer i+1's head), so it is a partition into N correctly
    ORDERED and correctly CLASSIFIED segments -- not a claim about where nn.Module would
    have opened. Downstream reads it as `declared_dispatch_op_span`, never as a module span.
    """
    expected_count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    patterns = pattern_doc.get("patterns", [])
    if expected_count <= 0 or not patterns:
        return [], {"status": "no_pattern_doc"}
    undeclared = [p.get("pattern_id") for p in patterns
                  if not str((p.get("structural_signature") or {}).get(
                      "runtime_dispatch_branch") or "").strip()]
    if undeclared:
        return [], {"status": "patterns_without_dispatch_branch",
                    "patterns_missing_branch": undeclared}
    branches = _declared_dispatch_branches(pattern_doc)
    branch_by_pattern = {
        p.get("pattern_id"): str(
            (p.get("structural_signature") or {})["runtime_dispatch_branch"]).strip()
        for p in patterns}
    pattern_by_layer = _pattern_index(pattern_doc)

    cpu_spans, span_starts = _cpu_step_index(spans)
    if not cpu_spans:
        cpu_spans, span_starts = spans, [span[0] for span in spans]
    by_step = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "cpu_op":
            continue
        name = event.get("name")
        if not isinstance(name, str) or name not in branches:
            continue
        if event.get("ts") is None:
            continue
        step = _cpu_step_at(event["ts"], cpu_spans, span_starts)
        if step is None:
            continue
        by_step.setdefault(step[5], []).append(
            (event["ts"], name, step, event.get("pid"), event.get("tid")))

    scopes = []
    diagnostics = {"status": "mapped", "anchor_ops": sorted(branches),
                   "expected_layer_count": expected_count, "steps": []}
    for step_id, anchors in sorted(by_step.items()):
        anchors.sort(key=lambda item: item[0])
        record = {"step_id": step_id, "anchor_count": len(anchors)}
        if len(anchors) != expected_count:
            record["status"] = "anchor_count_mismatch"
            diagnostics["steps"].append(record)
            continue
        step = anchors[0][2]
        # The CPU-side step window closes the last layer; indices 7,8 carry it when the
        # trace has a CPU annotation, otherwise the GPU window end is the best available.
        step_end = step[8] if len(step) >= 9 else step[1]
        ordered = []
        for layer_id, (ts, name, _step, pid, tid) in enumerate(anchors):
            pattern = pattern_by_layer.get(layer_id) or {}
            if branch_by_pattern.get(pattern.get("pattern_id")) != name:
                ordered = None
                record["status"] = "anchor_order_disagrees_with_patterns"
                record["first_mismatch_layer"] = layer_id
                break
            end = anchors[layer_id + 1][0] if layer_id + 1 < len(anchors) else step_end
            ordered.append({
                "name": name,
                "class_local_id": layer_id,
                "ts": ts,
                "end": end,
                "event_index": layer_id,
                "step_id": step_id,
                "phase": _phase_name(step[2]),
                "layer_id": layer_id,
                "pattern_id": pattern.get("pattern_id"),
                "pass_index": 0,
                "layer_instance_id": "%s:pass-0:layer-%d" % (step_id, layer_id),
                "type_validation": "pass",
                "scope_source": "declared_dispatch_op",
                # The launching thread and the step bucket let a boundary transfer
                # use these scopes as a donor exactly like GEAK_LAYER_SCOPE markers.
                "pid": pid,
                "tid": tid,
                "batch_size": step[4],
                "input_tokens": step[3],
            })
        if ordered is None:
            diagnostics["steps"].append(record)
            continue
        record["status"] = "mapped"
        diagnostics["steps"].append(record)
        scopes.extend(ordered)
    scopes.sort(key=lambda item: (item["ts"], item["end"]))
    if not scopes:
        diagnostics["status"] = "no_usable_step"
    return scopes, diagnostics


def _module_scope_at(ts, scopes, starts):
    if ts is None or not scopes:
        return None
    pos = bisect.bisect_right(starts, ts)
    matches = []
    for item in reversed(scopes[max(0, pos - 256):pos]):
        if item["ts"] <= ts < item["end"]:
            matches.append(item)
    if not matches:
        return None
    # Prefer the narrowest enclosing DecoderLayer scope. Pattern identity was
    # already derived from config/runtime source and must not be re-inferred
    # from model-specific class-name substrings here.
    return min(matches, key=lambda item: item["end"] - item["ts"])


def _scope_at(ts, scopes, starts):
    if ts is None or not scopes:
        return None
    pos = bisect.bisect_right(starts, ts)
    best = None
    # Torch CPU scopes are nested and the relevant layer scope is normally among
    # the most recently opened scopes. The bound avoids O(device*cpu) traces.
    for item in reversed(scopes[max(0, pos - 256):pos]):
        if item["ts"] <= ts < item["end"]:
            if best is None or (item["end"] - item["ts"]) < (best["end"] - best["ts"]):
                best = item
    return best


def _flow_layer_index(events, module_scopes, module_starts):
    """Map GPU flow-marker timestamps from CPU-side DecoderLayer spans.

    Only the flow source ("s", the CPU launch) is located against the CPU
    module spans.  The flow targets ("t"/"f") are device timestamps: when the
    CPU runs ahead of the GPU they routinely fall inside a *later* layer's CPU
    span, which says nothing about ownership and must not veto the launch.
    """
    flows = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") not in ("s", "t", "f"):
            continue
        flow_id = event.get("id", (event.get("args") or {}).get("id"))
        if flow_id is not None and event.get("ts") is not None:
            flows.setdefault(str(flow_id), []).append(
                (event.get("ph"), event.get("ts")))
    result = {}
    for endpoints in flows.values():
        sources = [_module_scope_at(ts, module_scopes, module_starts)
                   for ph, ts in endpoints if ph == "s"]
        sources = [source for source in sources if source]
        instance_ids = set(source["layer_instance_id"] for source in sources)
        if len(instance_ids) != 1:
            continue
        for ph, ts in endpoints:
            if ph != "s":
                result[ts] = sources[0]
    return result


def _phase_at(ts, spans, starts):
    span = _step_at(ts, spans, starts)
    return _phase_name(span[2]) if span else None


def _pattern_index(pattern_doc):
    result = {}
    for pattern in pattern_doc.get("patterns", []):
        for layer_id in pattern.get("layer_ids", []):
            result[int(layer_id)] = pattern
    return result


def _event_rows(events, pattern_doc):
    patterns = _pattern_index(pattern_doc)
    spans, draft_spans = _collect_step_spans_with_draft(events)
    span_starts = [span[0] for span in spans]
    draft_starts = [item["ts"] for item in draft_spans]
    cpu_by_ext, scopes, scope_starts = _cpu_evidence(events)
    triton_launches = _triton_launch_evidence(events)
    module_scopes, module_diagnostics = _module_layer_scopes(
        events, spans, pattern_doc, draft_windows=[
            (item["cpu_ts"], item["cpu_end"]) for item in draft_spans
            if item.get("cpu_ts") is not None])
    module_starts = [scope["ts"] for scope in module_scopes]
    flow_layers = _flow_layer_index(events, module_scopes, module_starts)
    rows = []
    out_of_scope = {"count": 0, "duration_us": 0.0}
    draft_scope = {"step_count": len(draft_spans), "device_event_count": 0,
                   "duration_us": 0.0, "steps": draft_spans}
    device_sequence = 0
    device_events = [(raw_index, event) for raw_index, event in enumerate(events)
                     if isinstance(event, dict)
                     and event.get("cat") in DEVICE_CATEGORIES]
    device_events.sort(key=lambda item: (
        item[1].get("ts") is None, item[1].get("ts") or 0, item[0]))
    external_id_launch_count = {}
    for _, device_event in device_events:
        device_args = device_event.get("args") or {}
        device_external_id = device_args.get("External id")
        if device_external_id is not None:
            external_id_launch_count[device_external_id] = (
                external_id_launch_count.get(device_external_id, 0) + 1)
    for raw_index, event in device_events:
        ts = event.get("ts")
        step = _step_at(ts, spans, span_starts)
        phase = _phase_name(step[2]) if step else None
        if spans and phase is None:
            out_of_scope["count"] += 1
            out_of_scope["duration_us"] += float(event.get("dur", 0) or 0)
            pos = bisect.bisect_right(draft_starts, ts if ts is not None else -1) - 1
            if pos >= 0 and ts < draft_spans[pos]["end"]:
                draft_scope["device_event_count"] += 1
                draft_scope["duration_us"] += float(event.get("dur", 0) or 0)
            continue
        device_sequence += 1
        args = event.get("args") or {}
        ext = args.get("External id")
        parent = cpu_by_ext.get(ext)
        scope = _scope_at(ts, scopes, scope_starts)
        module_scope = _module_scope_at(
            (parent or {}).get("ts"), module_scopes, module_starts)
        flow_scope = flow_layers.get(ts)
        layer_instance_id = None
        if module_scope is not None:
            layer_id = module_scope["layer_id"]
            layer_instance_id = module_scope["layer_instance_id"]
            # Do not call a dispatch-op cut a python module span. Both are authoritative
            # per-layer scopes, but only one of them is a module frame, and a reader
            # auditing boundary provenance has to be able to tell them apart.
            layer_evidence = (
                "declared_dispatch_op_span"
                if module_scope.get("scope_source") == "declared_dispatch_op"
                else "python_module_span_external_id")
        elif flow_scope is not None:
            layer_id = flow_scope["layer_id"]
            layer_instance_id = flow_scope["layer_instance_id"]
            module_scope = flow_scope
            layer_evidence = "python_module_span_ac2g_flow"
        elif (parent or {}).get("layer_id") is not None:
            layer_id = parent["layer_id"]
            layer_evidence = "cpu_op_external_id"
        elif (scope or {}).get("layer_id") is not None:
            layer_id = scope["layer_id"]
            layer_evidence = "cpu_op_scope"
        elif ts in flow_layers:
            layer_id = flow_layers[ts]
            layer_evidence = "trace_flow"
        else:
            layer_id = None
            layer_evidence = "unresolved"
        parent = parent or scope
        pattern = patterns.get(layer_id)
        assignment = "layer_body" if pattern else (
            "transition_global" if parent else "concurrent_unresolved")
        name = str(event.get("name", "?"))
        classification, provider, _, _ = parse_profile.classify(name)
        dims = (parent or {}).get("input_dims") or []
        types = (parent or {}).get("input_types") or []
        one_to_one_launch = (
            ext is not None and external_id_launch_count.get(ext) == 1)
        shape_source = "kernel_exact" if dims and one_to_one_launch else (
            "parent_context" if parent else "unresolved")
        launch = triton_launches.get(args.get("correlation"))
        operand_names = None
        if (shape_source != "kernel_exact" and launch
                and launch["kernel_name"] == _launched_kernel_name(name)):
            # The kernel's own arguments beat its enclosing operator's inputs.
            dims, types = launch["input_dims"], launch["input_types"]
            operand_names = launch["operand_names"]
            shape_source = "triton_launch_args"
        stage, stage_rule_id, stage_source = _stage_detail(
            name, event.get("cat"), (parent or {}).get("name", ""))
        rows.append({
            "row_id": "event-%d" % raw_index,
            "raw_event_index": raw_index,
            "device_seq_index": device_sequence,
            "timestamp": ts,
            "duration_us": float(event.get("dur", 0) or 0),
            "stream": args.get("stream", args.get("Stream")),
            "external_id": ext,
            "correlation": args.get("correlation", args.get("Correlation ID")),
            "event_type": event.get("cat"),
            "phase": phase or "unresolved",
            "phase_source": step[6] if step else "unresolved",
            "step_id": step[5] if step else None,
            "step_batch_size": step[4] if step else None,
            "step_input_tokens": step[3] if step else None,
            "assignment": assignment,
            "layer_id": layer_id,
            "layer_instance_id": layer_instance_id,
            "layer_evidence": layer_evidence,
            "layer_region": None,
            "boundary_role": None,
            "pattern_id": pattern.get("pattern_id") if pattern else None,
            "raw_name": name,
            "short_name": parse_profile.short_name(name),
            "classification": classification,
            "stage": stage,
            "stage_rule_id": stage_rule_id,
            "stage_source": stage_source,
            "stage_ruleset_version": STAGE_RULESET_VERSION,
            "provider": provider,
            "parent_operator": {
                "op_instance_id": "ext-%s" % ext if ext is not None else None,
                "canonical_op": (parent or {}).get("name", "unresolved"),
                "mapping_level": "external_id" if ext in cpu_by_ext else (
                    "cpu_scope" if scope else "unresolved"),
                "mapping_cardinality": (
                    "1:1" if ext in cpu_by_ext and one_to_one_launch else
                    "1:N" if ext in cpu_by_ext else "unresolved"),
                "device_launch_count": external_id_launch_count.get(ext),
                "confidence": "high" if ext in cpu_by_ext else (
                    "medium" if scope else "low"),
                "evidence_event_index": (parent or {}).get("event_index"),
            },
            "shape": dict({
                "source": shape_source,
                "input_dims": dims,
                "input_types": types,
            }, **({"operand_names": operand_names} if operand_names else {})),
        })
    out_of_scope["duration_us"] = round(out_of_scope["duration_us"], 6)
    if draft_spans:
        draft_scope["duration_us"] = round(draft_scope["duration_us"], 6)
        # Speculative draft-model steps are excluded from main-layer tables by
        # construction; keep the count auditable instead of silently dropping.
        out_of_scope["speculative_draft"] = draft_scope
    return rows, spans, out_of_scope, module_scopes, module_diagnostics


def _boundary_identity(value, event_type=None):
    """Normalize profiler spelling differences, not operator semantics."""
    value = str(value or "")
    # Graph-construction traces expose HIP memset nodes using the generic
    # profiler label, while graph replay can expose the ROCclr implementation
    # kernel name.  They are the same device primitive and are safe to compare
    # as one identity; this rule is backend-generic and carries no layer/stage
    # meaning.
    if (value == "Memset (Device)"
            or "__amd_rocclr_fillBuffer" in value):
        return "__GEAK_DEVICE_MEMSET__"
    value = re.sub(r"GRID_MN_\d+", "GRID_MN_*", value)
    value = re.sub(r"(_grid_)\d+(?=_|$)", r"\1*", value, flags=re.I)
    return value


def _boundary_sequence_sha(rows):
    values = []
    for row in rows:
        values.append(_boundary_identity(
            row.get("raw_name"), row.get("event_type")))
    payload = json.dumps(
        values, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _apply_boundary_map(rows, map_path, pattern_doc, trace_paths=None,
                        pattern_path="", already_applied=None):
    """Apply a validated all-layer boundary artifact by step-local positions."""
    with open(map_path) as fh:
        document = json.load(fh)
    # Cuts from a dispatch-op donor are phase-shifted like direct dispatch cuts
    # (see _segment_pattern_doc), so the rows they place must say so.
    dispatch_donor = ((document.get("donor") or {}).get("scope_source")
                      == "declared_dispatch_op_span")
    if document.get("status") not in ("pass", "partial"):
        raise ValueError(
            "refusing non-passing layer boundary map %s: %s" % (
                map_path, document.get("failures", [])))
    expected = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    if int(document.get("expected_main_layers", 0) or 0) != expected:
        raise ValueError("layer boundary map main-layer count mismatch")
    if pattern_path:
        declared = (document.get("patterns") or {}).get("sha256")
        if declared != _sha(pattern_path):
            raise ValueError("layer boundary map structural-pattern hash mismatch")
    if trace_paths:
        actual = {
            (os.path.abspath(path), _sha(path)) for path in trace_paths}
        declared = {
            (item.get("path"), item.get("sha256"))
            for item in (document.get("recipient") or {}).get("traces", [])}
        if actual != declared:
            raise ValueError("layer boundary map recipient trace set mismatch")

    patterns = _pattern_index(pattern_doc)
    by_step = {}
    for row in rows:
        if row.get("step_id"):
            by_step.setdefault(row["step_id"], []).append(row)
    applied = already_applied if already_applied is not None else set()
    diagnostics = []
    for group in document.get("mapped_groups", []):
        step_id = group.get("recipient_step_id")
        if step_id in applied:
            continue
        step_rows = sorted(
            by_step.get(step_id, []),
            key=lambda row: row["device_seq_index"])
        if not step_rows:
            raise ValueError(
                "layer boundary map references absent step %s" % step_id)
        if len(step_rows) != int(group.get("recipient_row_count", -1)):
            raise ValueError(
                "layer boundary map row-count mismatch for %s" % step_id)
        if _boundary_sequence_sha(step_rows) != group.get(
                "recipient_sequence_sha256"):
            raise ValueError(
                "layer boundary map sequence hash mismatch for %s" % step_id)
        raw_ranges = group.get("layer_ranges") or []
        if raw_ranges:
            ranges = [{
                "layer_id": int(item.get("layer_id", -1)),
                "start_position": int(item.get("start_position", -1)),
                "end_position": int(item.get("end_position", -1)),
                "representative_eligible": bool(
                    item.get("representative_eligible", True)),
            } for item in raw_ranges]
        else:
            starts = [int(value) for value in group.get(
                "layer_start_positions", [])]
            end = int(group.get("body_end_position", -1))
            ranges = [{
                "layer_id": layer_id,
                "start_position": start,
                "end_position": (
                    starts[layer_id + 1]
                    if layer_id + 1 < len(starts) else end),
                "representative_eligible": True,
            } for layer_id, start in enumerate(starts)]
        if (len(ranges) != expected
                or [item["layer_id"] for item in ranges]
                != list(range(expected))
                or any(item["start_position"] < 0
                       or item["end_position"] > len(step_rows)
                       or item["end_position"] <= item["start_position"]
                       for item in ranges)
                or any(left["end_position"] > right["start_position"]
                       for left, right in zip(ranges, ranges[1:]))):
            raise ValueError(
                "invalid transferred layer cuts for %s" % step_id)

        _clear_step_layer_assignments(
            step_rows, "outside_validated_graph_capture_layer_scope")
        for item in group.get("residual_ranges") or []:
            start = int(item.get("start_position", -1))
            stop = int(item.get("end_position", -1))
            if start < 0 or stop > len(step_rows) or stop <= start:
                raise ValueError(
                    "invalid transferred residual range for %s" % step_id)
            for position in range(start, stop):
                row = step_rows[position]
                row["assignment"] = "transition_global"
                row["layer_region"] = "inter_layer_residual"
                row["layer_evidence"] = (
                    "validated_graph_capture_inter_layer_residual")
                row["boundary_residual"] = {
                    "layer_boundary": item.get("layer_boundary"),
                    "reason": item.get("reason"),
                }
        for item in ranges:
            layer_id = item["layer_id"]
            start = item["start_position"]
            stop = item["end_position"]
            instance_id = "%s:graph-capture-donor:layer-%d" % (
                step_id, layer_id)
            for position in range(start, stop):
                row = step_rows[position]
                row["assignment"] = "layer_body"
                row["layer_id"] = layer_id
                row["layer_instance_id"] = instance_id
                row["pattern_id"] = (patterns.get(layer_id) or {}).get(
                    "pattern_id")
                row["layer_evidence"] = (
                    "validated_graph_capture_layer_scope_transfer")
                row["layer_region"] = "layer_body"
                row["representative_eligible"] = item[
                    "representative_eligible"]
                if dispatch_donor:
                    row["boundary_alignment"] = "dispatch_op_span"
            step_rows[start]["boundary_role"] = "body_start_kernel"
            step_rows[stop - 1]["boundary_role"] = "end_kernel"
        # A graph-capture donor sets the cuts only.  Rows the capture launch
        # records did not name stay in their layer, marked for audit.
        for position in group.get("capture_unmatched_positions") or []:
            position = int(position)
            if position < 0 or position >= len(step_rows):
                raise ValueError(
                    "invalid capture-unmatched position for %s" % step_id)
            row = step_rows[position]
            if row.get("assignment") == "layer_body":
                row["layer_evidence"] = (
                    "validated_graph_capture_layer_scope_transfer"
                    "_not_in_capture_launches")
        applied.add(step_id)
        diagnostics.append({
            "map_path": os.path.abspath(map_path),
            "step_id": step_id,
            "phase": group.get("phase"),
            "batch_size": group.get("batch_size"),
            "input_tokens": group.get("input_tokens"),
            "body_start_position": ranges[0]["start_position"],
            "body_end_position": ranges[-1]["end_position"],
            "prefix_row_count": group.get(
                "prefix_row_count", ranges[0]["start_position"]),
            "suffix_row_count": group.get(
                "suffix_row_count",
                len(step_rows) - ranges[-1]["end_position"]),
            "layer_widths": group.get("layer_widths", []),
            "layer_ranges": ranges,
            "residual_ranges": group.get("residual_ranges") or [],
            "match_rule": group.get("match_rule"),
        })
    return diagnostics


def _sequence_ratio(left, right):
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(
        None, tuple(left), tuple(right), autojunk=False).ratio()


def _sequence_medoid(sequences):
    """Choose one observed sequence with minimum total edit-distance proxy."""
    counts = {}
    for sequence in sequences:
        key = tuple(sequence)
        if key:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    scored = []
    for candidate in sorted(counts):
        distance = sum(
            (1.0 - _sequence_ratio(candidate, other)) * count
            for other, count in counts.items())
        scored.append((round(distance, 12), len(candidate), candidate))
    return list(min(scored)[2])


_ANCHOR_MIN_BODIES = 2
_ANCHOR_MAX_GAP_CV = 0.35


def _anchor_candidates(runs, layer_count):
    """Report recurring stages for diagnostics; never assign layer identity."""
    positions = {}
    for index, run in enumerate(runs):
        positions.setdefault(run["stage"], []).append(index)
    candidates = []
    for stage, where in positions.items():
        if stage in ("unknown", "memory"):
            continue
        count = len(where)
        if count < _ANCHOR_MIN_BODIES or count > layer_count:
            continue
        gaps = [where[i + 1] - where[i] for i in range(count - 1)]
        mean = sum(gaps) / len(gaps) if gaps else 0
        if mean <= 0:
            continue
        variance = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
        cv = (variance ** 0.5) / mean
        if cv > _ANCHOR_MAX_GAP_CV:
            continue
        candidates.append({
            "anchor_stage": stage,
            "anchor_runs": where,
            "gap_cv": round(cv, 6),
            "gap_mean": round(mean, 6),
            "observed_layer_bodies": count,
        })
    return candidates


def _stage_runs(step_rows):
    """Return lossless row ranges for continuously deduplicated stages."""
    runs = []
    for index, row in enumerate(step_rows):
        if not runs or runs[-1]["stage"] != row["stage"]:
            runs.append({"stage": row["stage"], "start": index, "end": index})
        else:
            runs[-1]["end"] = index
    return runs


def _clear_step_layer_assignments(step_rows, evidence):
    for row in step_rows:
        row["assignment"] = "transition_global"
        row["layer_id"] = None
        row["layer_instance_id"] = None
        row["pattern_id"] = None
        row["layer_evidence"] = evidence
        row["layer_region"] = "transition_global"
        row["boundary_role"] = None


def _authoritative_instances(step_rows):
    grouped = {}
    for row in step_rows:
        instance_id = row.get("layer_instance_id")
        evidence = str(row.get("layer_evidence") or "")
        if (instance_id and (
                evidence.startswith("python_module_span")
                or evidence.startswith("explicit_layer_marker")
                or evidence == "declared_dispatch_op_span"
                or evidence.startswith(
                    "validated_graph_capture_layer_scope"))):
            grouped.setdefault(instance_id, []).append(row)
    instances = []
    for instance_id, group in grouped.items():
        group.sort(key=lambda row: row["device_seq_index"])
        instances.append({
            "instance_id": instance_id,
            "layer_id": group[0].get("layer_id"),
            "rows": group,
            "first": group[0]["device_seq_index"],
            "last": group[-1]["device_seq_index"],
        })
    instances.sort(key=lambda item: item["first"])
    return instances


def _anchor_cut(left, right, left_start):
    """First device index of `right`, cut from both layers' sorted anchors.

    Returns (start, misassigned_anchor_count), or (None, None) when no cut
    keeps both layers non-empty.
    """
    lowest = max(left[0], left_start) + 1
    best = None
    for cut in range(lowest, right[-1] + 1):
        misassigned = ((len(left) - bisect.bisect_left(left, cut))
                       + bisect.bisect_left(right, cut))
        if best is None or misassigned <= best[1]:
            best = (cut, misassigned)
    return best if best is not None else (None, None)


def _normalize_module_scope_cuts(
        step_rows, instances, layer_count, patterns):
    """Turn complete ordered module anchors into non-overlapping cuts.

    Raw module-span ownership can leave holes (a launch whose link to its
    span was not recovered) or, when independent streams interleave, let two
    adjacent layers overlap in device order.  Each boundary is cut from the
    two layers' own anchors only, so layers of different sizes (hybrid
    models) are cut where their anchors say, never at a size-dependent point:

    * anchors do not overlap: the next layer starts at its first anchor, so
      an unowned kernel inside a layer stays with it and one in the gap
      between two layers stays with the layer that launched before it;
    * anchors overlap: the cut inside the overlap that moves the fewest
      anchors to the other side (ties: the latest such cut).

    No stage or kernel identity participates.
    """
    if not instances or len(instances) % layer_count:
        return False, []
    if not all(all(str(row.get("layer_evidence") or "").startswith(
                       "python_module_span") for row in item["rows"])
               for item in instances):
        return False, []
    pass_count = len(instances) // layer_count
    specifications = []
    previous_end = None
    for pass_index in range(pass_count):
        chunk = instances[
            pass_index * layer_count:(pass_index + 1) * layer_count]
        if [item["layer_id"] for item in chunk] != list(range(layer_count)):
            return False, []
        anchors = [sorted(row["device_seq_index"] for row in item["rows"])
                   for item in chunk]
        if any(left[0] >= right[0]
               for left, right in zip(anchors, anchors[1:])):
            return False, []
        starts = [anchors[0][0]]
        boundaries = []
        for left, right in zip(anchors, anchors[1:]):
            start, misassigned = _anchor_cut(left, right, starts[-1])
            if start is None:
                return False, []
            starts.append(start)
            boundaries.append({
                "start_device_seq_index": start,
                "method": ("anchor_gap" if left[-1] < right[0]
                           else "anchor_overlap_min_misassigned"),
                "misassigned_anchor_count": misassigned,
            })
        end = anchors[-1][-1] + 1
        if (end <= starts[-1]
                or (previous_end is not None and starts[0] < previous_end)):
            return False, []
        specifications.append((pass_index, chunk, starts, end, boundaries))
        previous_end = end

    _clear_step_layer_assignments(
        step_rows, "outside_python_module_span_ordered_cut")
    audit = []
    for pass_index, chunk, starts, end, boundaries in specifications:
        for layer_id, start in enumerate(starts):
            stop = starts[layer_id + 1] if layer_id + 1 < layer_count else end
            instance_id = chunk[layer_id]["instance_id"]
            selected = [
                row for row in step_rows
                if start <= row["device_seq_index"] < stop]
            if not selected:
                return False, []
            for row in selected:
                row["assignment"] = "layer_body"
                row["layer_id"] = layer_id
                row["layer_instance_id"] = instance_id
                row["pattern_id"] = (patterns.get(layer_id) or {}).get(
                    "pattern_id")
                row["layer_evidence"] = "python_module_span_ordered_cut"
                row["layer_region"] = "layer_body"
            selected[0]["boundary_role"] = "body_start_kernel"
            selected[-1]["boundary_role"] = "end_kernel"
        audit.append({
            "pass_index": pass_index,
            "layer_start_device_seq_indices": starts,
            "layer_boundaries": boundaries,
            "misassigned_anchor_count": sum(
                item["misassigned_anchor_count"] for item in boundaries),
            "body_end_device_seq_index_exclusive": end,
        })
    return True, audit


def _authoritative_layer_partition(rows, pattern_doc):
    """Publish only layer ownership backed by an independent scope.

    Stage recurrence and sequence alignment are useful diagnostics, but a
    periodic stage can represent one Pattern rather than one layer.  They must
    therefore never assign layer IDs.  This function accepts only rows already
    carrying complete module/marker/donor instance IDs.
    """
    layer_count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    if layer_count <= 0:
        return [], {}
    template_evidence = {}
    by_step = {}
    for row in rows:
        if row.get("step_id"):
            by_step.setdefault(row["step_id"], []).append(row)
    diagnostics = []
    for step_id, step_rows in sorted(by_step.items()):
        step_rows.sort(key=lambda row: row["device_seq_index"])
        instances = _authoritative_instances(step_rows)
        actual_order = [item["layer_id"] for item in instances]
        pass_count = len(instances) // layer_count if layer_count else 0
        expected_order = list(range(layer_count)) * pass_count
        contiguous = all(
            [row["device_seq_index"] for row in item["rows"]]
            == list(range(item["first"], item["last"] + 1))
            for item in instances)
        non_overlapping = all(
            left["last"] < right["first"]
            for left, right in zip(instances, instances[1:]))
        complete = (
            bool(instances) and len(instances) % layer_count == 0
            and actual_order == expected_order
            and contiguous and non_overlapping)
        module_cut_audit = []
        if (not complete and instances
                and len(instances) % layer_count == 0
                and actual_order == expected_order):
            normalized, module_cut_audit = _normalize_module_scope_cuts(
                step_rows, instances, layer_count,
                _pattern_index(pattern_doc))
            if normalized:
                instances = _authoritative_instances(step_rows)
                actual_order = [item["layer_id"] for item in instances]
                pass_count = len(instances) // layer_count
                expected_order = list(range(layer_count)) * pass_count
                contiguous = all(
                    [row["device_seq_index"] for row in item["rows"]]
                    == list(range(item["first"], item["last"] + 1))
                    for item in instances)
                non_overlapping = all(
                    left["last"] < right["first"]
                    for left, right in zip(instances, instances[1:]))
                complete = (
                    actual_order == expected_order
                    and contiguous and non_overlapping)
        runs = _stage_runs(step_rows)
        recurring = _anchor_candidates(runs, layer_count)
        phase = step_rows[0].get("phase") if step_rows else "unresolved"
        if not complete:
            _clear_step_layer_assignments(
                step_rows, "boundary_unresolved_no_authoritative_scope")
            diagnostics.append({
                "step_id": step_id,
                "phase": phase,
                "status": "boundary_unresolved",
                "partition_method": "none",
                "configured_layer_count": layer_count,
                "module_instance_count": len(instances),
                "observed_stage_run_count": len(runs),
                "mapped_event_count": 0,
                "diagnostic_only_recurring_stages": recurring,
                "reason": (
                    "No complete per-layer module, explicit-marker, or "
                    "validated donor boundary. Recurring stages and sequence "
                    "alignment are diagnostic only."),
                "layer_boundaries": [],
            })
            continue
        cut_events = []
        for item in instances:
            item["rows"][0]["boundary_role"] = "body_start_kernel"
            item["rows"][-1]["boundary_role"] = "end_kernel"
            for row in item["rows"]:
                row["layer_region"] = "layer_body"
            cut_events.append({
                "layer_id": item["layer_id"],
                "body_start_event": item["rows"][0]["row_id"],
                "body_end_event": item["rows"][-1]["row_id"],
            })
        evidence = sorted({
            row.get("layer_evidence") for item in instances
            for row in item["rows"] if row.get("layer_evidence")})
        diagnostics.append({
            "step_id": step_id,
            "phase": phase,
            "status": "mapped",
            "partition_method": "authoritative_scope_ownership",
            "module_scope_ordered_cut": module_cut_audit,
            "boundary_evidence": evidence,
            "configured_layer_count": layer_count,
            "module_instance_count": len(instances),
            "mapped_pass_count": pass_count,
            "observed_stage_run_count": len(runs),
            "mapped_event_count": sum(len(item["rows"]) for item in instances),
            "diagnostic_only_recurring_stages": recurring,
            "layer_boundaries": cut_events,
        })
    return diagnostics, template_evidence


def _dispatch_aligned(row):
    return (row.get("layer_evidence") == "declared_dispatch_op_span"
            or row.get("boundary_alignment") == "dispatch_op_span")


def _segment_pattern_doc(pattern_doc, rows):
    """Key dispatch-cut layer bodies by (Pattern, successor Pattern).

    A declared dispatch op sits in the MIDDLE of a layer, so a dispatch cut
    [anchor_i, anchor_i+1) holds layer i's core and tail plus layer i+1's head.
    Its content therefore depends on the next layer's Pattern too: on a hybrid
    model the full-attention head (qkv GEMM, qk-norm, RoPE, KV write) only
    occurs in segments whose successor is a full-attention layer. Keyed by the
    core Pattern alone, a representative followed by a same-Pattern layer never
    contained it, and the highest-value fusion seams were in no table.

    The cut cannot simply be moved to the "real" layer start: in fused-residual
    models the previous layer's residual add and this layer's input norm are one
    kernel, so there is no clean boundary, and locating one would need stage
    inference, which may never define a boundary. Instead each segment kind
    (`P0>P1`, last layer `Pn>END`) becomes its own table Pattern. Every kernel
    lands in some table, each table is sequence-consistent, and layer-weighted
    time is conserved. Rows cut by module spans keep their Pattern.
    """
    aligned = [row for row in rows
               if row.get("assignment") == "layer_body" and _dispatch_aligned(row)]
    if not aligned:
        return pattern_doc
    patterns = _pattern_index(pattern_doc)
    count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)

    def kind(layer_id):
        core = (patterns.get(layer_id) or {}).get("pattern_id")
        succ = ((patterns.get(layer_id + 1) or {}).get("pattern_id")
                if layer_id + 1 < count else "END")
        return "%s>%s" % (core, succ)

    for row in aligned:
        row["core_pattern_id"] = row.get("pattern_id")
        row["pattern_id"] = kind(int(row["layer_id"]))
    used = {row.get("pattern_id") for row in rows
            if row.get("assignment") == "layer_body"}
    by_id = {pattern.get("pattern_id"): pattern
             for pattern in pattern_doc.get("patterns", [])}
    segments = {}
    for layer_id in range(count):
        segments.setdefault(kind(layer_id), []).append(layer_id)
    out_patterns = [pattern for pattern in pattern_doc.get("patterns", [])
                    if pattern.get("pattern_id") in used]
    for segment_id, layer_ids in sorted(segments.items()):
        if segment_id not in used:
            continue
        core_id, succ_id = segment_id.split(">", 1)
        core = by_id.get(core_id) or {}
        allowed = set(core.get("representative_candidates",
                               core.get("layer_ids", [])))
        candidates = [layer for layer in layer_ids if layer in allowed]
        out_patterns.append({
            **{key: value for key, value in core.items()
               if key not in ("pattern_id", "layer_ids",
                              "representative_candidates")},
            "pattern_id": segment_id,
            "core_pattern_id": core_id,
            "successor_pattern_id": succ_id,
            "segment_basis": "dispatch_op_span",
            "pattern_display_name": "%s -> %s head" % (
                core.get("pattern_display_name") or core_id,
                (by_id.get(succ_id) or {}).get("pattern_display_name") or succ_id),
            "layer_ids": layer_ids,
            "representative_candidates": candidates or layer_ids,
        })
    return {**pattern_doc, "patterns": out_patterns,
            "segment_patterns_from": "dispatch_op_span"}


def _layer_instances(rows):
    explicit = {}
    groups = []
    current = []
    current_key = None
    for row in sorted(rows, key=lambda item: item["device_seq_index"]):
        if row["assignment"] == "layer_body" and row.get("layer_instance_id"):
            explicit.setdefault(row["layer_instance_id"], []).append(row)
            if current:
                groups.append(current)
                current = []
                current_key = None
            continue
        key = (row["phase"], row["layer_id"]) if row["assignment"] == "layer_body" else None
        if key != current_key:
            if current:
                groups.append(current)
            current = []
            current_key = key
        if key is not None:
            current.append(row)
    if current:
        groups.append(current)
    groups.extend(explicit[key] for key in sorted(
        explicit, key=lambda key: min(
            row["device_seq_index"] for row in explicit[key])))
    instances = []
    occurrence = {}
    for group in groups:
        phase, layer_id = group[0]["phase"], group[0]["layer_id"]
        key = (phase, layer_id)
        occurrence[key] = occurrence.get(key, 0) + 1
        positions = [row["device_seq_index"] for row in group]
        explicit_instance = group[0].get("layer_instance_id")
        contiguous = positions == list(range(min(positions), max(positions) + 1))
        evidence_sources = sorted(set(row["layer_evidence"] for row in group))
        representative_eligible = all(
            row.get("representative_eligible", True) for row in group)
        boundary_complete = contiguous
        duration = sum(row["duration_us"] for row in group)
        signature = [
            ((row.get("parent_operator") or {}).get("canonical_op")
             if ((row.get("parent_operator") or {}).get("canonical_op")
                 not in (None, "", "unresolved"))
             else row.get("short_name"))
            for row in group]
        instances.append({
            "phase": phase,
            "layer_id": layer_id,
            "pattern_id": group[0]["pattern_id"],
            "step_id": group[0].get("step_id"),
            "step_batch_size": group[0].get("step_batch_size"),
            "step_input_tokens": group[0].get("step_input_tokens"),
            "layer_instance_id": explicit_instance,
            "occurrence": occurrence[key],
            "body_start_event": group[0]["row_id"],
            "body_end_event": group[-1]["row_id"],
            "first_device_seq_index": min(positions),
            "last_device_seq_index": max(positions),
            "event_count": len(group),
            "duration_us": round(duration, 6),
            "sequence_signature": hashlib.sha256(
                json.dumps(signature).encode()).hexdigest()[:16],
            "normalized_sequence": signature,
            "boundary_complete": boundary_complete,
            "representative_eligible": representative_eligible,
            "boundary_evidence": {
                "sources": evidence_sources,
                "continuity": "contiguous" if contiguous else "interleaved_or_unresolved",
                "end_anchor_required": False,
                "end_anchor_valid": True,
                "end_kernel": group[-1]["row_id"],
                "end_stage": group[-1]["stage"],
                "adjacent_to_boundary_residual": not representative_eligible,
            },
        })
    return instances


def _boundary_rank(instance):
    """Lower is a more trustworthy independently-owned layer boundary."""
    sources = (instance.get("boundary_evidence") or {}).get("sources") or []
    best = 9
    for src in sources:
        text = str(src)
        if text.startswith("python_module_span"):
            rank = 0
        elif text.startswith("validated_graph_capture_layer_scope"):
            rank = 0
        elif text.startswith("explicit_layer_marker"):
            rank = 1
        elif text == "declared_dispatch_op_span":
            # A runtime op boundary, but phase-shifted within the layer (see
            # _dispatch_anchor_scopes), so it ranks with explicit markers.
            rank = 1
        elif text == "module_sequence_interpolation":
            rank = 2
        else:
            rank = 9
        best = min(best, rank)
    return best


def _instance_context(pattern_doc, layer_id):
    contexts = pattern_doc.get("layer_contexts") or {}
    return contexts.get(str(layer_id), contexts.get(layer_id, {})) or {}


def _context_penalty(pattern_doc, layer_id):
    context = _instance_context(pattern_doc, layer_id)
    return int(any(bool(context.get(key)) for key in (
        "is_first_layer", "is_first_main_layer", "is_last_layer",
        "is_last_main_layer", "model_entry", "model_exit",
        "model_epilogue")))


def _select_analysis_steps(spans):
    """Pick ONE analysis step per phase from the captured target steps.

    A capture holds several steps (e.g. the client's single test prompt and a
    real 16x1024 prefill batch, or several steady verify steps).  Every table
    of a phase must describe the same step, otherwise Patterns end up in
    different buckets.  Prefill: most tokens, then batch size.  Decode/verify:
    largest batch, then tokens.  Ties: the step with the median duration, so a
    first/last-step outlier is not chosen.  Draft steps never reach `spans`.
    """
    by_phase = {}
    for span in spans:
        phase = _phase_name(span[2])
        by_phase.setdefault(phase, []).append({
            "step_id": span[5], "batch_size": int(span[4] or 0),
            "input_tokens": int(span[3] or 0),
            "duration_us": round(float(span[1]) - float(span[0]), 3),
            "source": span[6]})
    chosen = {}
    for phase, steps in by_phase.items():
        if phase == "prefill":
            key = lambda item: (item["input_tokens"], item["batch_size"])
            rule = "max_input_tokens_then_batch_then_median_duration"
        else:
            key = lambda item: (item["batch_size"], item["input_tokens"])
            rule = "max_batch_then_tokens_then_median_duration"
        best = max(key(item) for item in steps)
        bucket = sorted((item for item in steps if key(item) == best),
                        key=lambda item: (item["duration_us"], item["step_id"]))
        pick = bucket[(len(bucket) - 1) // 2]
        chosen[phase] = dict(pick, rule=rule, candidates=len(steps),
                             same_bucket_step_ids=[i["step_id"] for i in bucket],
                             other_bucket_step_ids=sorted(
                                 i["step_id"] for i in steps if key(i) != best))
    return chosen


def _analysis_candidates(values, phase, analysis_steps):
    """Restrict representative candidates to the phase's analysis step.

    Falls back to the same bucket, then to every step, only when the chosen
    step has no eligible instance; the returned tag records which applied.
    """
    chosen = (analysis_steps or {}).get(phase)
    if not chosen:
        return values, "no_analysis_step"
    exact = [item for item in values if item.get("step_id") == chosen["step_id"]]
    if exact:
        return exact, "analysis_step"
    same = set(chosen.get("same_bucket_step_ids") or [])
    bucket = [item for item in values if item.get("step_id") in same]
    if bucket:
        return bucket, "same_bucket_fallback"
    return values, "any_step_fallback"


def _representatives(pattern_doc, instances, layer_hints=None,
                     analysis_steps=None):
    """Choose one authoritative sequence medoid per (Pattern, phase)."""
    layer_hints = layer_hints or {}
    by_pattern = {}
    for instance in instances:
        if (instance["boundary_complete"]
                and instance.get("representative_eligible", True)
                and _boundary_rank(instance) < 9):
            by_pattern.setdefault(instance["pattern_id"], []).append(instance)
    selected = {}
    for pattern in pattern_doc.get("patterns", []):
        pid = pattern["pattern_id"]
        values = by_pattern.get(pid, [])
        phases = sorted({item["phase"] for item in values})
        medians = {}
        selected_instances = {}
        step_match = {}
        for phase in phases:
            phase_values = [item for item in values if item["phase"] == phase]
            phase_values, step_match[phase] = _analysis_candidates(
                phase_values, phase, analysis_steps)
            allowed_ids = set(pattern.get(
                "representative_candidates", pattern.get("layer_ids", [])))
            preferred = [
                item for item in phase_values
                if item["layer_id"] in allowed_ids]
            candidates = preferred or phase_values
            hinted_ids = set(layer_hints.get(pid, []))
            hinted = [
                item for item in candidates
                if item["layer_id"] in hinted_ids]
            if hinted:
                candidates = hinted
            medians[phase] = statistics.median(
                item["duration_us"] for item in candidates) if candidates else 0
            def score(item):
                sequence = item.get("normalized_sequence") or []
                distance = sum(
                    1.0 - _sequence_ratio(
                        sequence, other.get("normalized_sequence") or [])
                    for other in candidates)
                duration_deviation = (
                    abs(item["duration_us"] / medians[phase] - 1.0)
                    if medians[phase] else 0.0)
                return (
                    _boundary_rank(item),
                    _context_penalty(pattern_doc, item["layer_id"]),
                    round(distance, 12),
                    round(duration_deviation, 12),
                    item["layer_id"],
                    item["first_device_seq_index"],
                )
            if candidates:
                chosen = min(candidates, key=score)
                selected_instances[phase] = {
                    "layer_id": chosen["layer_id"],
                    "body_start_event": chosen["body_start_event"],
                    "body_end_event": chosen["body_end_event"],
                    "first_device_seq_index": chosen["first_device_seq_index"],
                    "last_device_seq_index": chosen["last_device_seq_index"],
                    "occurrence": chosen["occurrence"],
                    "boundary_rank": _boundary_rank(chosen),
                    "contextual_edge": bool(
                        _context_penalty(pattern_doc, chosen["layer_id"])),
                    "sequence_signature": chosen["sequence_signature"],
                    "boundary_evidence": chosen.get("boundary_evidence", {}),
                    "step_id": chosen.get("step_id"),
                    "analysis_step_match": step_match.get(phase),
                }
        selected_layers = sorted(set(
            value["layer_id"] for value in selected_instances.values()))
        selected[pid] = {
            # Compatibility field. Tables use the phase-local layer_id below.
            "layer_id": selected_layers[0] if len(selected_layers) == 1 else None,
            "phases": phases,
            "phase_median_duration_us": medians,
            "selected_instances": selected_instances,
            "selection_confidence": (
                "high" if selected_instances and phases != ["unresolved"]
                else "low"),
        }
    return selected


def _table(pattern_doc, rows, representatives, table_phases=None):
    pattern_meta = {
        pattern["pattern_id"]: pattern
        for pattern in pattern_doc.get("patterns", [])
    }
    grouped = {}
    for row in rows:
        if table_phases and row["phase"] not in table_phases:
            continue
        rep = representatives.get(row["pattern_id"], {})
        selected = (rep.get("selected_instances") or {}).get(row["phase"])
        if (row["assignment"] != "layer_body"
                or row["layer_id"] != (selected or {}).get("layer_id")
                or not selected
                or not (selected["first_device_seq_index"] <= row["device_seq_index"]
                        <= selected["last_device_seq_index"])):
            continue
        grouped.setdefault((row["phase"], row["pattern_id"]), []).append(row)
    tables = []
    phase_order = {"prefill": 0, "decode": 1, "verify": 1, "unresolved": 2}
    for (phase, pattern_id), group in sorted(
            grouped.items(),
            key=lambda item: (
                phase_order.get(item[0][0], 99), item[0][1])):
        selected = ((representatives.get(pattern_id, {}).get(
            "selected_instances") or {}).get(phase) or {})
        total = sum(row["duration_us"] for row in group)
        pattern_layer_ids = sorted(
            int(layer_id) for layer_id in
            pattern_meta.get(pattern_id, {}).get("layer_ids", []))
        output_rows = []
        for pos, row in enumerate(group):
            item = dict(row)
            item["pos"] = pos
            item["layer_total_pct"] = round(
                100.0 * row["duration_us"] / total, 6) if total else 0.0
            output_rows.append(item)
        tables.append({
            "phase": phase,
            "pattern_id": pattern_id,
            "pattern_display_name": pattern_meta.get(
                pattern_id, {}).get("pattern_display_name", pattern_id),
            "pattern_layer_ids": pattern_layer_ids,
            "pattern_layer_count": len(pattern_layer_ids),
            "representative_layer_id": selected["layer_id"],
            "selected_step_id": group[0].get("step_id"),
            "selected_bucket": {
                "phase": phase,
                "batch_size": group[0].get("step_batch_size"),
                "input_tokens": group[0].get("step_input_tokens"),
            },
            "structural_context": pattern_meta.get(
                pattern_id, {}).get(
                    "structural_context",
                    pattern_meta.get(pattern_id, {}).get(
                        "structural_signature", {})),
            "representative_instance_context": _instance_context(
                pattern_doc, selected["layer_id"]),
            "boundary_evidence": selected.get("boundary_evidence", {}),
            "event_count": len(group),
            "layer_total_us": round(total, 6),
            "rows": output_rows,
        })
    return tables


def _representative_integrity(rows, tables, representatives):
    """Verify only the Pattern representatives exported for downstream fusion."""
    by_position = {
        row["device_seq_index"]: row
        for row in rows
    }
    audits = []
    for table in tables:
        pattern_id = table["pattern_id"]
        phase = table["phase"]
        selected = (
            representatives.get(pattern_id, {}).get(
                "selected_instances", {}).get(phase) or {})
        first = selected.get("first_device_seq_index")
        last = selected.get("last_device_seq_index")
        expected = [
            by_position[position]
            for position in range(first, last + 1)
            if first is not None and last is not None and position in by_position
        ]
        actual = table.get("rows", [])
        expected_ids = [row["row_id"] for row in expected]
        actual_ids = [row["row_id"] for row in actual]
        duration_sum = round(sum(
            float(row.get("duration_us", 0) or 0) for row in actual), 6)
        exact_once = len(actual_ids) == len(set(actual_ids))
        ordered = [
            row["device_seq_index"] for row in actual
        ] == sorted(row["device_seq_index"] for row in actual)
        interval_complete = actual_ids == expected_ids
        duration_matches = math.isclose(
            duration_sum, float(table.get("layer_total_us", 0) or 0),
            rel_tol=0, abs_tol=1e-6)
        pattern_layer_ids = table.get("pattern_layer_ids", [])
        representative_in_pattern = (
            table["representative_layer_id"] in pattern_layer_ids)
        passed = (
            first is not None and last is not None and exact_once and ordered
            and interval_complete and duration_matches
            and representative_in_pattern)
        audits.append({
            "pattern_id": pattern_id,
            "phase": phase,
            "representative_layer_id": table["representative_layer_id"],
            "pattern_layer_ids": pattern_layer_ids,
            "representative_in_pattern": representative_in_pattern,
            "body_start_device_seq_index": first,
            "body_end_device_seq_index": last,
            "expected_event_count": len(expected_ids),
            "actual_event_count": len(actual_ids),
            "dropped_row_ids": [
                row_id for row_id in expected_ids if row_id not in set(actual_ids)],
            "duplicate_row_ids": sorted({
                row_id for row_id in actual_ids if actual_ids.count(row_id) > 1}),
            "exact_once": exact_once,
            "ordered": ordered,
            "interval_complete": interval_complete,
            "duration_sum_us": duration_sum,
            "declared_layer_total_us": table.get("layer_total_us"),
            "duration_matches": duration_matches,
            "status": "pass" if passed else "fail",
        })
    return {
        "status": "pass" if tables and all(
            item["status"] == "pass" for item in audits) else "fail",
        "table_count": len(tables),
        "tables": audits,
    }


def _trace_pattern_consistency(instances):
    """Validate runtime similarity without prescribing any semantic stage.

    This is deliberately diagnostic. Data-dependent routing, generated kernel
    specialisation, and different buckets may change an execution sequence even
    when config/runtime define one structural body. A disagreement asks the
    Agent to inspect a missed runtime branch; it does not invent a Pattern.
    """
    grouped = {}
    for instance in instances:
        if not instance.get("boundary_complete") or _boundary_rank(instance) >= 9:
            continue
        grouped.setdefault(
            (instance.get("phase"), instance.get("pattern_id")), []).append(
                instance)
    audits = []
    for (phase, pattern_id), values in sorted(grouped.items()):
        sequences = [value.get("normalized_sequence") or [] for value in values]
        medoid = _sequence_medoid(sequences)
        ratios = [_sequence_ratio(medoid, sequence) for sequence in sequences]
        audits.append({
            "phase": phase,
            "pattern_id": pattern_id,
            "instance_count": len(values),
            "distinct_sequence_signatures": len(set(
                value.get("sequence_signature") for value in values)),
            "minimum_medoid_similarity": (
                round(min(ratios), 6) if ratios else None),
            "median_medoid_similarity": (
                round(statistics.median(ratios), 6) if ratios else None),
            "status": "pass" if ratios else "unavailable",
        })
    return {
        "status": "pass" if audits else "unavailable",
        "gating": False,
        "scope": "authoritative_layer_instances",
        "tables": audits,
    }


def _step_link_audit(events, spans):
    """Per step: how intact the host-launch <-> device-record link is.

    {step_id: {"host_launches", "launches_without_device_record",
    "graph_replay_rows"}}. A launch with no device record means the profiler
    dropped GPU activity (Kimi-K2.5 TP8, vLLM 0.21: 356-968 of a prefill step's
    2703 launches lost in contiguous ~310ms holes; every Qwen3.5 / MiniMax-M3
    step lost 0). A device row carried by a graph launch has no per-kernel host
    launch, so a host scope cannot own it.
    """
    runtime = [event for event in events
               if isinstance(event, dict)
               and event.get("cat") in ("cuda_runtime", "hip_runtime")
               and event.get("ts") is not None]
    device = [event for event in events
              if isinstance(event, dict) and event.get("cat") in DEVICE_CATEGORIES
              and event.get("ts") is not None]
    recorded = {(event.get("args") or {}).get("correlation") for event in device}
    graph_launches = {(event.get("args") or {}).get("correlation")
                      for event in runtime if "GraphLaunch" in str(event.get("name"))}
    graph_launches.discard(None)
    audit = {span[5]: {"host_launches": 0, "launches_without_device_record": 0,
                       "graph_replay_rows": 0} for span in spans}
    host = sorted((span[7], span[8], span[5]) for span in spans if len(span) >= 9)
    host_starts = [item[0] for item in host]
    for event in runtime:
        name = str(event.get("name"))
        if "Launch" not in name or "Graph" in name:
            continue
        position = bisect.bisect_right(host_starts, event["ts"]) - 1
        if position < 0 or event["ts"] >= host[position][1]:
            continue
        entry = audit[host[position][2]]
        entry["host_launches"] += 1
        if (event.get("args") or {}).get("correlation") not in recorded:
            entry["launches_without_device_record"] += 1
    device_starts = [span[0] for span in spans]
    for event in device:
        index = bisect.bisect_right(device_starts, event["ts"]) - 1
        if index < 0 or event["ts"] >= spans[index][1]:
            continue
        if (event.get("args") or {}).get("correlation") in graph_launches:
            audit[spans[index][5]]["graph_replay_rows"] += 1
    return audit


def _excused_incomplete_steps(partition_diagnostics, link_audit):
    """Unresolved steps whose trace evidence cannot support layer ownership.

    {step_id: {"reason", ...counts}}. Excused only while the phase still has a
    mapped step to draw its representatives from:
      * device_records_dropped -- the profiler lost some of the step's GPU records,
        so its layers are missing kernels no mapping can restore;
      * graph_replay_rows_unowned -- every layer anchor is present, but part of the
        step replayed from a CUDA graph (a small mixed step under the capture size)
        and those rows have no host launch to own them. A step with no anchors at
        all (plain graph decode) is not excused here; it needs the boundary donor.
    """
    mapped = collections.Counter(
        item.get("phase") for item in partition_diagnostics
        if item.get("status") == "mapped")
    excused = {}
    for item in partition_diagnostics:
        if item.get("status") == "mapped" or not mapped[item.get("phase")]:
            continue
        link = link_audit.get(item["step_id"]) or {}
        if link.get("launches_without_device_record", 0) > 0:
            reason = "device_records_dropped"
        elif (link.get("graph_replay_rows", 0) > 0
              and item.get("module_instance_count")
              == item.get("configured_layer_count")):
            reason = "graph_replay_rows_unowned"
        else:
            continue
        excused[item["step_id"]] = dict(link, reason=reason)
    return excused


def _gating_step_audits(step_audits):
    """Mark extra unresolved steps non-gating and return the gating audits.

    A step with NO boundary evidence at all (not a contradicting one) does not
    gate when another step of the same phase mapped completely: that phase's
    tables already stand on authoritative scopes, and the extra step's rows stay
    unassigned rather than being guessed. A Profile trace can carry such a step,
    e.g. a prefill step inside its DECODE file. A phase with no mapped step still
    fails, exactly as before.
    """
    mapped_phases = {item.get("phase") for item in step_audits
                     if item["status"] == "pass"}
    for item in step_audits:
        if (item["status"] == "fail"
                and item.get("boundary_source_status") == "boundary_unresolved"
                and item.get("actual_instance_count") == 0
                and item.get("phase") in mapped_phases):
            item["status"] = "not_gating_unresolved_extra_step"
    return [item for item in step_audits
            if item["status"] != "not_gating_unresolved_extra_step"]


def _quality(
        pattern_doc, rows, instances, representatives, spans, out_of_scope,
        partition_diagnostics, tables, link_audit=None, analysis_steps=None):
    input_count = len(rows)
    assigned_count = sum(1 for row in rows if row["assignment"] in (
        "layer_body", "transition_global", "concurrent_unresolved"))
    input_duration = sum(row["duration_us"] for row in rows)
    assigned_duration = sum(row["duration_us"] for row in rows
                            if row["assignment"] in (
                                "layer_body", "transition_global", "concurrent_unresolved"))
    pattern_missing = [pid for pid, rep in representatives.items()
                       if not rep.get("selected_instances")]
    incomplete = [item for item in instances if not item["boundary_complete"]]
    instances_by_step = {}
    for instance in instances:
        instances_by_step.setdefault(instance.get("step_id"), []).append(instance)
    incomplete_evidence = _excused_incomplete_steps(
        partition_diagnostics, link_audit or {})
    step_audits = []
    for diagnostic in partition_diagnostics:
        step_instances = sorted(
            instances_by_step.get(diagnostic["step_id"], []),
            key=lambda item: item["first_device_seq_index"])
        layer_count = diagnostic["configured_layer_count"]
        pass_count = int(diagnostic.get("mapped_pass_count", 1) or 1)
        expected_order = list(range(layer_count)) * pass_count
        actual_order = [item["layer_id"] for item in step_instances]
        non_overlapping = all(
            left["last_device_seq_index"] < right["first_device_seq_index"]
            for left, right in zip(step_instances, step_instances[1:]))
        step_audits.append({
            "step_id": diagnostic["step_id"],
            "phase": diagnostic.get("phase"),
            "expected_instance_count": len(expected_order),
            "actual_instance_count": len(step_instances),
            "layer_order_valid": actual_order == expected_order,
            "non_overlapping": non_overlapping,
            "boundary_source_status": diagnostic.get("status"),
            "status": (
                "excused_incomplete_evidence"
                if diagnostic["step_id"] in incomplete_evidence else
                "pass" if (
                    diagnostic.get("status") == "mapped"
                    and actual_order == expected_order
                    and non_overlapping) else "fail"),
        })
    # Steps with no boundary evidence are marked first (an extra prefill step
    # inside a DECODE file).  Tables are then built from one analysis step per
    # phase; other captured steps (a test prompt, a step the profiler entered
    # mid-way, a ramp-up step) are audited but do not gate.  An analysis step
    # that is itself unresolved still gates: its rows are the tables.  Without
    # analysis steps every step gates except the marked extra ones.  A step
    # excused for incomplete link evidence never gates.
    _gating_step_audits(step_audits)
    analysis_ids = {item["step_id"]
                    for item in (analysis_steps or {}).values()}
    if analysis_ids:
        gating_audits = [item for item in step_audits
                         if item["step_id"] in analysis_ids]
    else:
        gating_audits = [item for item in step_audits
                         if item["status"] != "not_gating_unresolved_extra_step"]
    gating_audits = [item for item in gating_audits
                     if item["status"] != "excused_incomplete_evidence"]
    gated_step_ids = {item["step_id"] for item in gating_audits}
    non_gating_audits = [item for item in step_audits
                         if item["step_id"] not in gated_step_ids]
    mechanical_pass = (not partition_diagnostics or bool(gating_audits)) and all(
        item["status"] == "pass" for item in gating_audits)
    incomplete = [item for item in incomplete
                  if not analysis_ids or item.get("step_id") in gated_step_ids]
    phase_status = "pass" if spans else "partial"
    representative_integrity = _representative_integrity(
        rows, tables, representatives)
    trace_consistency = _trace_pattern_consistency(instances)
    conservation_pass = (
        input_count == assigned_count and
        math.isclose(input_duration, assigned_duration, rel_tol=0, abs_tol=1e-6))
    status = "fail" if (
        not tables or representative_integrity["status"] == "fail"
        or not conservation_pass) else (
        "partial" if pattern_missing or not mechanical_pass
        or phase_status == "partial"
        or pattern_doc.get("quality", {}).get("status") == "partial"
        else "pass")
    return {
        "schema_version": 1,
        "status": status,
        "gates": {
            "pattern_coverage": pattern_doc.get("coverage_check", {}),
            "phase": {
                "status": phase_status,
                "annotation_spans": len(spans),
                "reason": "" if spans else "no measured phase annotations",
            },
            "analysis_window_conservation": {
                "status": "pass" if conservation_pass else "fail",
                "input_event_count": input_count,
                "assigned_event_count": assigned_count,
                "input_duration_us": round(input_duration, 6),
                "assigned_duration_us": round(assigned_duration, 6),
                "out_of_scope": out_of_scope,
            },
            "layer_boundaries": {
                "status": "pass" if mechanical_pass and not incomplete else "fail",
                "gating": True,
                "scope": "analysis_steps" if analysis_ids else "all_required_steps",
                "incomplete_instances": len(incomplete),
                "patterns_without_representative": pattern_missing,
            },
            "step_layer_order": {
                "status": "pass" if mechanical_pass else "fail",
                "gating": True,
                "scope": "analysis_steps" if analysis_ids else "all_required_steps",
                "steps": gating_audits,
                "non_gating_steps": non_gating_audits,
                "non_gating_unresolved_steps": [
                    item["step_id"] for item in step_audits
                    if item["status"] == "not_gating_unresolved_extra_step"],
                "excused_incomplete_evidence_steps": [
                    dict(item, step_id=step_id)
                    for step_id, item in sorted(incomplete_evidence.items())],
            },
            "representative_layer_integrity": representative_integrity,
            "trace_pattern_consistency": trace_consistency,
        },
    }




def _phase_coverage(instances, tables, trace_paths, adopted_siblings,
                    table_phases, require_phases, step_phases=None,
                    observed_phases=()):
    """Describe what phase coverage this build actually achieved.

    Exists because `table_phases: ["all"]` used to be emitted whenever no
    filter was requested, which read as full coverage even when the only input
    was a single-phase EXTEND trace (B1).
    """
    def _norm(value):
        value = str(value or "").strip().lower()
        return {"extend": "prefill", "prompt": "prefill",
                "generation": "decode",
                "target_verify": "verify"}.get(value, value)

    in_tables = sorted({_norm(t.get("phase")) for t in tables if t.get("phase")})
    in_trace = sorted({_norm(i.get("phase")) for i in instances if i.get("phase")})
    # Phases of the annotated steps themselves, whether or not any layer in them was
    # resolved. `in_trace` only sees steps that produced layer instances, so a graph-
    # replayed decode step with no boundary looked like no decode step at all.
    with_steps = sorted({_norm(p) for p in (step_phases or ()) if p})
    tags = sorted({tag for tag in (_phase_tag(p) for p in trace_paths) if tag})
    # The trace decides the generation phase: "verify" when the target ran
    # TARGET_VERIFY steps (speculative decoding), else "decode".  A caller's
    # "decode"/"verify" requirement means "the generation phase".
    generation = sglang_step_modes.generation_phase(
        set(in_tables) | set(in_trace) | set(with_steps)
        | {_norm(p) for p in observed_phases})
    if require_phases is None:
        required = sglang_step_modes.target_phases([generation])
    else:
        required = sorted({
            generation if sglang_step_modes.is_generation_phase(_norm(v))
            else _norm(v) for v in require_phases if v})
    missing = [phase for phase in required if phase not in in_tables]

    # Sequence coverage and shape coverage fail independently, and conflating
    # them is what made the decode gap invisible.  A replay-mode DECODE trace
    # yields a complete ordered kernel sequence (it has kernel events) but no
    # module/cpu_op spans, so every shape comes back `unresolved`.
    shape_stats = {}
    for table in tables:
        phase = _norm(table.get("phase"))
        stat = shape_stats.setdefault(phase, {"rows": 0, "resolved": 0})
        for row in table.get("rows", []):
            stat["rows"] += 1
            if (row.get("shape") or {}).get("source") not in (None, "unresolved"):
                stat["resolved"] += 1
    for phase, stat in shape_stats.items():
        stat["resolved_fraction"] = (
            round(stat["resolved"] / stat["rows"], 4) if stat["rows"] else 0.0)
    # decode_* fields describe the generation phase (decode or verify).
    decode_stat = shape_stats.get(generation, {"rows": 0, "resolved": 0})
    decode_seq = generation in in_tables
    decode_shapes = decode_stat["resolved"] > 0
    return {
        "phases_in_tables": in_tables,
        "phases_in_trace": in_trace,
        "generation_phase": generation,
        "phases_absent_from_tables": [
            phase for phase in sglang_step_modes.target_phases([generation])
            if phase not in in_tables],
        "single_phase": len(in_tables) <= 1,
        "shape_resolution_by_phase": shape_stats,
        # sequence == "which kernels, in what order"; shapes == "with what
        # dtypes and dims".  Candidate DISCOVERY needs the first; candidate
        # GENERATION and benchmarking need the second.
        "decode_sequence_covered": decode_seq,
        "decode_shapes_covered": decode_shapes,
        "decode_covered": decode_seq and decode_shapes,
        "trace_phase_tags": tags,
        # A trace with NO step annotation at all cannot phase-tag a single row. On sglang
        # that never happens (its profiler always writes step[...]); on vllm it is the
        # DEFAULT unless the capture overlay's phase-annotation hook is armed, so make the
        # distinction machine-readable instead of leaving it to be inferred from an empty
        # phases_in_trace.
        "phase_annotation_present": bool(in_trace),
        "phases_with_steps": with_steps,
        # Decode steps are in the window but none got layer boundaries (on vLLM: graph
        # replay walks no per-layer op). The remedy is the Phase-1.2 boundary donor,
        # not a different capture window.
        "decode_requires_boundary_donor": (
            generation in with_steps and not decode_seq),
        "traces_analysed": [os.path.abspath(p) for p in trace_paths],
        "siblings_auto_adopted": adopted_siblings,
        "filter_requested": sorted(table_phases) if table_phases else None,
        "required_phases": required,
        "missing_required_phases": missing,
        # Under CUDA-graph replay the CPU never walks the module tree, so a
        # steady-state DECODE trace carries kernel events but essentially no
        # nn.Module spans.  The ordered sequence survives; the shapes do not.
        # Only the shape half needs a separate graph-construction capture.
        "decode_requires_graph_capture": decode_seq and not decode_shapes,
        # Naming WHY decode is missing, because the remedies are different and only one of
        # them is "capture again".  The last two arms exist for single-file mixed-phase
        # captures: sglang's profile_by_stage puts the phase in the FILENAME, so an absent
        # DECODE file is conclusive -- but vllm writes one un-split trace, where no phase
        # tag is normal and "no_decode_trace_analysed" would libel a capture that did in
        # fact cover decode.  Split that bucket by whether the trace carried step spans:
        #   no spans   -> the annotation itself is missing (on vllm: the capture overlay's
        #                 hook was not armed). Nothing is phase-tagged, prefill included.
        #   spans, but no decode -> annotation worked; the WINDOW held no decode step.
        "decode_evidence": (
            "sequence_and_shapes" if (decode_seq and decode_shapes) else
            "sequence_only_shapes_unresolved" if decode_seq else
            "trace_present_but_no_decode_tables" if "DECODE" in tags else
            "decode_steps_present_boundaries_unresolved" if (
                generation in with_steps) else
            "no_phase_annotation_in_trace" if (not tags and not in_trace) else
            "mixed_trace_no_decode_steps_in_window" if not tags else
            "no_decode_trace_analysed"),
    }


def _shape_capture_plan(tables, pattern_doc, trace_path,
                        trace_paths=None, coverage=None):
    needs = []
    target_layers = sorted({
        int(table["representative_layer_id"]) for table in tables
        if table.get("representative_layer_id") is not None})
    # Boundary capture needs one lightweight all-layer marker pass even when a
    # graph-erased phase has no table yet.  Keep tensor logging representative-
    # only by adding at most one deterministic, non-contextual candidate per
    # otherwise-unrepresented Pattern.
    represented_patterns = {table.get("pattern_id") for table in tables}
    for pattern in pattern_doc.get("patterns", []):
        if pattern.get("pattern_id") in represented_patterns:
            continue
        candidates = (pattern.get("representative_candidates")
                      or pattern.get("layer_ids") or [])
        if candidates:
            target_layers.append(int(candidates[0]))
    target_layers = sorted(set(target_layers))
    target_buckets = []
    for table in tables:
        bucket = dict(table.get("selected_bucket") or {})
        bucket.update({
            "pattern_id": table["pattern_id"],
            "representative_layer_id": table["representative_layer_id"],
            "step_id": table.get("selected_step_id"),
        })
        target_buckets.append(bucket)
        for row in table["rows"]:
            # Both carry the kernel's own inputs; nothing left to capture.
            if row["shape"]["source"] in ("kernel_exact", "triton_launch_args"):
                continue
            needs.append({
                "phase": table["phase"],
                "pattern_id": table["pattern_id"],
                "representative_layer_id": table["representative_layer_id"],
                "pos": row["pos"],
                "row_id": row["row_id"],
                "raw_event_index": row["raw_event_index"],
                "device_seq_index": row["device_seq_index"],
                "event_type": row["event_type"],
                "raw_name": row["raw_name"],
                "short_name": row["short_name"],
                "provider": row["provider"],
                "classification": row["classification"],
                "stage": row["stage"],
                "external_id": row.get("external_id"),
                "parent_operator": row["parent_operator"]["canonical_op"],
                "parent_mapping_level": row["parent_operator"]["mapping_level"],
                "parent_mapping_cardinality": row[
                    "parent_operator"].get("mapping_cardinality"),
                "parent_device_launch_count": row[
                    "parent_operator"].get("device_launch_count"),
                "candidate_op_path": None,
                "candidate_wrapper": None,
                "candidate_terminal_launcher": None,
                "mapping_cardinality": "unresolved",
                "source_evidence": [],
                "selected_bucket": table.get("selected_bucket", {}),
                "missing_fields": ["tensor roles", "input/output shapes", "dtype"],
                "current_source": row["shape"]["source"],
            })
    return {
        "schema_version": 2,
        "scope": "representative_layers_only",
        "analysis_rank": 0,
        "trace_path": os.path.abspath(trace_path),
        "trace_paths": [os.path.abspath(path)
                        for path in (trace_paths or [trace_path])],
        "trace_sha256": _sha(trace_path),
        "phase_coverage": coverage or {},
        "representative_layer_filter": target_layers,
        "target_buckets": target_buckets,
        "patterns": [{
            "pattern_id": table["pattern_id"],
            "pattern_layer_ids": table.get("pattern_layer_ids", []),
            "pattern_layer_count": table.get("pattern_layer_count", 0),
            "representative_layer_id": table["representative_layer_id"],
            "structural_context": table.get("structural_context", {}),
        } for table in tables],
        "capture_policy": {
            "rank": 0,
            "max_matched_forwards_per_bucket": 1,
            "metadata_only": True,
            "stdout": False,
            "unresolved_targets_only": True,
            "decode_capture_windows_implemented": ["graph_construction"],
            "decode_sequence_covered": bool(
                coverage and coverage.get("decode_sequence_covered")),
            "decode_shapes_covered": bool(
                coverage and coverage.get("decode_shapes_covered")),
            "decode_capture_requires": (
                [] if (coverage and coverage.get("decode_covered")) else
                ([] if (coverage and coverage.get("decode_sequence_covered"))
                 else ["run the Phase-1.2 boundary donor (on vLLM a "
                       "cudagraph_mode=NONE capture), then rebuild with "
                       "--layer-boundary-map: the trace holds decode steps "
                       "but none has layer boundaries"]
                 if (coverage and coverage.get(
                     "decode_requires_boundary_donor"))
                 else ["analyse the -TP-0-DECODE trace (auto-adopted by "
                       "default; --no-auto-sibling disables)"]) +
                ([] if (coverage and coverage.get("decode_shapes_covered"))
                 else ["run graph-construction shape capture for decode to "
                       "resolve decode shapes"])),
        },
        "capture_targets": needs,
        "target_count": len(needs),
    }


def _markdown(tables, quality):
    lines = ["# Ordered Unique Layer Kernel Tables", "",
             "Semantic mapping status: `%s`." % quality["status"], ""]
    for table in tables:
        lines.extend([
            "## %s — %s" % (table["phase"].upper(), table["pattern_display_name"]),
            "",
            "- pattern layers (%d): `%s`" % (
                int(table.get(
                    "pattern_layer_count",
                    len(table.get("pattern_layer_ids", [])))),
                json.dumps(table.get("pattern_layer_ids", []))),
            "- representative layer: `L%s`" % table["representative_layer_id"],
            "- selected bucket: `%s`" % json.dumps(
                table.get("selected_bucket", {}), sort_keys=True),
            "- complete-layer device event count: `%s`" % table["event_count"],
            "- raw one-layer device event total us: `%.3f`" % table["layer_total_us"],
            "- structural context: `%s`" % json.dumps(
                table.get("structural_context", {}), sort_keys=True),
            "",
            "| pos | stage | kernel | parent operator | shape source | duration us | layer total % |",
            "|---:|---|---|---|---|---:|---:|",
        ])
        for row in table["rows"]:
            lines.append("| %d | %s | `%s` | %s | %s | %.3f | %.3f |" % (
                row["pos"], row["stage"], row["short_name"],
                row["parent_operator"]["canonical_op"].replace("|", "\\|"),
                row["shape"]["source"], row["duration_us"], row["layer_total_pct"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def _step_consistency_warnings(instances, representatives, analysis_steps):
    """Warn when same-bucket steps disagree with the chosen representative."""
    warnings = []
    for pid, selected in (representatives or {}).items():
        for phase, chosen in (selected.get("selected_instances") or {}).items():
            same = set((analysis_steps.get(phase) or {}).get(
                "same_bucket_step_ids") or [])
            others = [item for item in instances
                      if item.get("pattern_id") == pid
                      and item.get("phase") == phase
                      and item.get("layer_id") == chosen["layer_id"]
                      and item.get("step_id") in same
                      and item.get("step_id") != chosen.get("step_id")]
            differing = sorted({item["step_id"] for item in others
                                if item["sequence_signature"]
                                != chosen["sequence_signature"]})
            if differing:
                warnings.append(
                    "step consistency: %s %s L%s kernel sequence differs in "
                    "same-bucket step(s) %s from the analysis step %s"
                    % (phase, pid, chosen["layer_id"], differing,
                       chosen.get("step_id")))
    return warnings


def _manifest_speculative(trace_paths):
    """Speculative facts from the capture's trace manifest, if present.

    bench_e2e.sh writes profile_trace_manifest.json beside the trace
    directory; it is optional enrichment (accept length, verify tokens).
    """
    for path in trace_paths:
        candidate = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(path))),
            "profile_trace_manifest.json")
        try:
            with open(candidate) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(doc.get("speculative"), dict):
            return dict(doc["speculative"], manifest=candidate)
    return None


def build(trace_path, pattern_path, out_dir, table_phases=None,
          auto_sibling=True, require_phases=None, boundary_map_paths=None,
          representative_layer_hints=None):
    """Build the semantic layer/kernel tables.

    `trace_path` may be a single path or a list of phase traces.  When
    `auto_sibling` is set, an unlisted EXTEND/DECODE sibling of the same rank
    and profiler session is pulled in automatically rather than silently
    ignored (B2).
    """
    trace_paths, adopted_siblings = _resolve_trace_paths(
        trace_path, auto_sibling=auto_sibling)
    primary_trace = trace_paths[0]
    with open(pattern_path) as fh:
        pattern_doc = json.load(fh)
    events = _load_events_multi(trace_paths)
    patterns = _pattern_index(pattern_doc)
    rows, spans, out_of_scope, module_scopes, module_diagnostics = _event_rows(
        events, pattern_doc)
    boundary_map_diagnostics = []
    applied_boundary_steps = set()
    for boundary_map_path in boundary_map_paths or []:
        boundary_map_diagnostics.extend(_apply_boundary_map(
            rows, boundary_map_path, pattern_doc, trace_paths,
            pattern_path, applied_boundary_steps))
    module_interpolated = 0
    partition_diagnostics, pattern_templates = _authoritative_layer_partition(
        rows, pattern_doc)
    # Historical code removed a first-layer prefix when its stage sequence was
    # absent from a dominant suffix. That mutates a trusted module boundary
    # using the same stage taxonomy we are trying to audit. Keep preparation
    # kernels outside a layer only when ownership evidence says so; do not
    # rewrite an authoritative boundary from sequence similarity.
    prefix_demotions = []
    pattern_doc = _segment_pattern_doc(pattern_doc, rows)
    instances = _layer_instances(rows)
    representative_instances = [
        instance for instance in instances
        if not table_phases or instance["phase"] in table_phases]
    analysis_steps = _select_analysis_steps(spans)
    representatives = _representatives(
        pattern_doc, representative_instances, representative_layer_hints,
        analysis_steps)
    tables = _table(pattern_doc, rows, representatives, table_phases)
    quality = _quality(
        pattern_doc, rows, instances, representatives, spans, out_of_scope,
        partition_diagnostics, tables, _step_link_audit(events, spans),
        analysis_steps)
    coverage = _phase_coverage(
        instances, tables, trace_paths, adopted_siblings, table_phases,
        require_phases,
        step_phases={row.get("phase") for row in rows if row.get("step_id")},
        observed_phases=analysis_steps.keys())
    coverage["analysis_steps"] = analysis_steps
    quality["phase_coverage"] = coverage
    quality.setdefault("warnings", []).extend(
        _step_consistency_warnings(instances, representatives, analysis_steps))
    manifest_speculative = _manifest_speculative(trace_paths)
    if manifest_speculative:
        coverage["speculative"] = manifest_speculative
    speculative_modes = sorted({
        span[6].split(":", 1)[1] for span in spans
        if str(span[6]).startswith("sglang_step_annotation:")})
    draft_scope = out_of_scope.get("speculative_draft")
    if speculative_modes or draft_scope:
        # Speculative decoding: decode is TARGET_VERIFY (bs requests x
        # draft-token positions through the main stack); draft-model steps
        # were excluded from the main-layer tables above.
        coverage["speculative_decoding"] = True
        coverage["decode_step_modes"] = [
            mode for mode in speculative_modes
            if sglang_step_modes.is_generation_phase(
                sglang_step_modes.PHASE_BY_MODE.get(mode))]
        coverage["draft_steps_excluded"] = (
            draft_scope["step_count"] if draft_scope else 0)
    if coverage["missing_required_phases"]:
        # A graph-replayed phase may legitimately need the Phase-1.2 donor
        # capture before it has authoritative layer boundaries. Preserve any
        # trustworthy phase tables and request completion; never publish the
        # missing phase as if sequence inference had recovered it.
        quality["status"] = "partial" if tables else "fail"
        quality.setdefault("failures", []).append(
            "phase coverage: required phase(s) %s absent from the tables; "
            "traces analysed: %s" % (
                ", ".join(coverage["missing_required_phases"]),
                ", ".join(os.path.basename(p) for p in trace_paths)))
    elif coverage["decode_requires_boundary_donor"]:
        quality.setdefault("warnings", []).append(
            "phase coverage: the trace holds decode steps but none has "
            "authoritative layer boundaries (graph replay walks no per-layer "
            "op). Run the Phase-1.2 boundary donor; widening the capture "
            "window will not help.")
    elif coverage["decode_requires_graph_capture"]:
        generation = coverage["generation_phase"]
        quality.setdefault("warnings", []).append(
            "phase coverage: %s kernel SEQUENCE is covered but 0/%d %s "
            "rows carry resolved shapes (CUDA-graph replay emits no module "
            "spans). Sequence is enough to discover %s fusion seams; "
            "generating or benchmarking one needs graph-construction shape capture."
            % (generation, coverage["shape_resolution_by_phase"]
               .get(generation, {}).get("rows", 0), generation, generation))
    elif coverage["single_phase"]:
        quality.setdefault("warnings", []).append(
            "phase coverage: tables contain only %s. Fusion candidates derived "
            "from this table apply to that phase alone." % (
                ", ".join(coverage["phases_in_tables"]) or "no phase"))
    capture_plan = _shape_capture_plan(
        tables, pattern_doc, primary_trace, trace_paths, coverage)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "semantic_event_audit_jsonl": os.path.join(out_dir, "semantic_event_audit.jsonl"),
        "layer_instance_audit_json": os.path.join(out_dir, "layer_instance_audit.json"),
        "semantic_table_json": os.path.join(out_dir, "pattern_layer_kernel_table.json"),
        "semantic_table_md": os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md"),
        "shape_capture_plan_json": os.path.join(out_dir, "SHAPE_CAPTURE_PLAN.json"),
        "quality_json": os.path.join(out_dir, "semantic_mapping_quality.json"),
    }
    with open(paths["semantic_event_audit_jsonl"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    docs = (
        (paths["layer_instance_audit_json"], {
            "schema_version": 1, "trace_sha256": _sha(primary_trace),
            "module_scope_diagnostics": module_diagnostics,
            "module_scope_count": len(module_scopes),
            "module_interpolated_event_count": module_interpolated,
            "boundary_maps": boundary_map_diagnostics,
            "boundary_partition_diagnostics": partition_diagnostics,
            "prefix_demotions": prefix_demotions,
            "pattern_stage_templates": pattern_templates,
            "instances": instances, "representatives": representatives}),
        (paths["semantic_table_json"], {
            "schema_version": 3,
            "trace_path": os.path.abspath(primary_trace),
            "trace_paths": [os.path.abspath(path) for path in trace_paths],
            "trace_sha256": _sha(primary_trace),
            "trace_sha256_by_path": {
                os.path.abspath(path): _sha(path) for path in trace_paths},
            "patterns_path": os.path.abspath(pattern_path),
            # B1: report the phases actually present, never the word "all".
            # table_phases=None means "no filter applied", which is not the
            # same as "every phase of the model was observed".
            "table_phases": coverage["phases_in_tables"],
            "table_phases_requested": (
                sorted(table_phases) if table_phases else "unfiltered"),
            "phase_coverage": coverage,
            "tables": tables}),
        (paths["shape_capture_plan_json"], capture_plan),
        (paths["quality_json"], quality),
    )
    for path, doc in docs:
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
    with open(paths["semantic_table_md"], "w") as fh:
        fh.write(_markdown(tables, quality))
    return {"schema_version": 1, "status": quality["status"], **paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, action="append",
                        help="phase trace; repeat or comma-separate for "
                             "EXTEND+DECODE coverage")
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    parser.add_argument("--table-phases", default="all",
                        help="comma-separated representative-table phases; default all, ordered prefill then decode")
    parser.add_argument("--require-phases", default="",
                        help="comma-separated phases that MUST appear in the "
                             "tables; build fails loudly if one is absent")
    parser.add_argument("--no-auto-sibling", action="store_true",
                        help="do not adopt an unlisted EXTEND/DECODE sibling "
                             "trace of the same rank")
    parser.add_argument("--layer-boundary-map", action="append", default=[],
                        help="validated graph-construction boundary artifact")
    args = parser.parse_args()
    traces = [item.strip() for entry in args.trace
              for item in entry.split(",") if item.strip()]
    requested_phases = set(
        value.strip() for value in args.table_phases.split(",") if value.strip())
    table_phases = None if "all" in requested_phases else requested_phases or None
    require_phases = [value.strip() for value in args.require_phases.split(",")
                      if value.strip()]
    result = build(traces, args.patterns, args.out_dir, table_phases,
                   auto_sibling=not args.no_auto_sibling,
                   require_phases=require_phases,
                   boundary_map_paths=args.layer_boundary_map)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
