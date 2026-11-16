#!/usr/bin/env python3
"""Read the device rows and step annotations out of a torch/kineto trace without
materializing the trace.

A decode window of an 8-rank MXFP4 deployment is ~280 MB of gzip per rank, which is
~4 GB / ~100 M lines of JSON. `json.load` on that costs minutes and tens of GB per rank,
and `serving_probe.py`'s verify step and every per-point census used to pay it -- in one
run the verify alone was 7.5 of the 16 minutes each decode point took.

Kineto writes one event per `{ ... }` block with a fixed field order:

    "ph": "X", "cat": "kernel", "name": "...", "pid": 2, "tid": 6,
    "ts": 1728908949092.539, "dur": 4.840,

so a byte regex over the decompressed stream recovers (cat, name, pid, tid, ts, dur) for
the categories asked for, at gzip speed. The regex is not trusted blindly: every chunk
also counts the raw `"cat": "<cat>"` occurrences, and if the two disagree for any
requested category (a different writer, a reordered field) the file is re-read with
`json.load` and `method` says so. A fast answer that silently drops rows would be worse
than the slow one.

    from trace_scan import scan
    s = scan(path)                 # s.events: [(cat, name, ts, dur, pid, tid, args)]
    s = scan(path, cats + ("cpu_op",), args_cats=("kernel", "cpu_op"))

`args` is None unless the event's category is in `args_cats`; then it is the parsed
`"args"` object (kernel grid/block/correlation, cpu_op Input Dims/type/Strides, ...).
Parsing args costs a json.loads per event, so ask only for the categories you need.

    python3 trace_scan.py <trace.json.gz> [--cats kernel,user_annotation] [--steps]

`--steps` lists the forward step spans in time order with the device time inside each,
which is what to look at when a census instance count looks wrong.

Stdlib only.
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter, namedtuple

DEVICE_CATS = ("kernel", "Kernel", "gpu_memcpy", "gpu_memset")
STEP_CATS = ("gpu_user_annotation", "user_annotation")
DEFAULT_CATS = DEVICE_CATS + STEP_CATS

_NUM = rb"(-?[0-9][0-9.]*(?:[eE][-+]?[0-9]+)?)"


def _event_re(cats, with_args=False):
    # The categories are part of the pattern: a generic `"cat": "<word>"` makes the
    # engine attempt every one of the ~15 M events in a decode trace (59 s per rank),
    # naming the wanted ones leaves only the device rows and step spans (2.4 s).
    alt = b"|".join(re.escape(c.encode()) for c in sorted(set(cats), key=len, reverse=True))
    tail = rb',\s*"args": \{(.*?)\n\s*\}' if with_args else b""
    return re.compile(
        rb'"cat": "(' + alt + rb')", "name": "([^"\\]*(?:\\.[^"\\]*)*)", '
        rb'"pid": ("?[^,"]*"?), "tid": ("?[^,"]*"?),\s*"ts": ' + _NUM + rb', "dur": ' + _NUM
        + tail, re.S if with_args else 0)
_CHUNK = 64 << 20

Scan = namedtuple("Scan", "events method counts")


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _id(raw):
    raw = raw.strip(b'"')
    try:
        return int(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


def _name(raw):
    if b"\\" not in raw:
        return raw.decode("utf-8", "replace")
    try:
        return json.loads(b'"' + raw + b'"')
    except ValueError:
        return raw.decode("utf-8", "replace")


def _args(raw):
    try:
        return json.loads(b"{" + raw + b"}")
    except ValueError:
        return None


def _scan_regex(path, cats, args_cats):
    want = {c.encode() for c in cats}
    with_args = [c for c in cats if c in args_cats]
    plain = [c for c in cats if c not in args_cats]
    passes = [(_event_re(plain), False)] if plain else []
    if with_args:
        passes.append((_event_re(with_args, True), True))
    needles = {c: b'"cat": "' + c + b'"' for c in want}
    events, seen, matched, bad_args = [], Counter(), Counter(), 0
    tail = b""
    with _open(path) as fh:
        while True:
            chunk = fh.read(_CHUNK)
            buf = tail + chunk
            if chunk:
                # Cut before the last event header so no event straddles two passes.
                ph = buf.rfind(b'"ph":')
                cut = buf.rfind(b"\n", 0, ph) if ph > 0 else -1
                if cut <= 0:
                    tail = buf
                    continue
                buf, tail = buf[:cut], buf[cut:]
            for c, needle in needles.items():
                seen[c] += buf.count(needle)
            for event_re, parse_args in passes:
                for m in event_re.finditer(buf):
                    cat = m.group(1)
                    matched[cat] += 1
                    args = None
                    if parse_args:
                        args = _args(m.group(7))
                        bad_args += args is None
                    events.append((cat.decode(), _name(m.group(2)), float(m.group(5)),
                                   float(m.group(6)), _id(m.group(3)), _id(m.group(4)),
                                   args))
            if not chunk:
                break
    counts = {c.decode(): seen[c] for c in want}
    ok = all(seen[c] == matched[c] for c in want) and not bad_args
    return events, counts, ok


def _scan_json(path, cats, args_cats=()):
    want = set(cats)
    with _open(path) as fh:
        data = json.load(fh)
    raw = data.get("traceEvents", data if isinstance(data, list) else [])
    events, counts = [], Counter()
    for e in raw:
        if not isinstance(e, dict) or e.get("cat") not in want:
            continue
        counts[e["cat"]] += 1
        if "ts" not in e or "dur" not in e:
            continue
        events.append((e["cat"], e.get("name", "?"), float(e["ts"]),
                       float(e.get("dur") or 0.0), e.get("pid"), e.get("tid"),
                       e.get("args") if e["cat"] in args_cats else None))
    return events, {c: counts[c] for c in want}


def scan(path, cats=DEFAULT_CATS, args_cats=()):
    """-> Scan(events sorted by ts, 'regex'|'json', {cat: count})."""
    args_cats = tuple(args_cats)
    events, counts, ok = _scan_regex(path, cats, args_cats)
    method = "regex"
    if not ok:
        events, counts = _scan_json(path, cats, args_cats)
        method = "json"
    events.sort(key=lambda e: e[2])
    return Scan(events, method, counts)


def steps(path):
    """Forward step spans in time order, each with the device microseconds inside it."""
    import bisect
    s = scan(path)
    dev = [(e[2], e[3]) for e in s.events if e[0] in DEVICE_CATS]
    starts = [d[0] for d in dev]
    out = []
    for cat, name, ts, dur, *_ in s.events:
        if cat in STEP_CATS and name.startswith("step["):
            lo, hi = bisect.bisect_left(starts, ts), bisect.bisect_left(starts, ts + dur)
            out.append({"timeline": "gpu" if cat == "gpu_user_annotation" else "cpu",
                        "name": name, "ts": ts, "dur_us": dur,
                        "device_us": round(sum(d for _t, d in dev[lo:hi]), 1),
                        "device_rows": hi - lo})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--cats", default=",".join(DEFAULT_CATS))
    ap.add_argument("--args-cats", default="")
    ap.add_argument("--steps", action="store_true", help="list forward step spans")
    ap.add_argument("--check-json", action="store_true",
                    help="also run the json.load path and compare (slow; for validation)")
    a = ap.parse_args()
    if a.steps:
        for r in steps(a.trace):
            print("%-4s %16.3f %10.1f us span %10.1f us device %6d rows  %s"
                  % (r["timeline"], r["ts"], r["dur_us"], r["device_us"], r["device_rows"],
                     r["name"]))
        return 0
    cats = tuple(c for c in a.cats.split(",") if c)
    args_cats = tuple(c for c in a.args_cats.split(",") if c)
    s = scan(a.trace, cats, args_cats)
    print(json.dumps({"method": s.method, "counts": s.counts, "events": len(s.events)}))
    if a.check_json:
        ref, _ = _scan_json(a.trace, cats, args_cats)
        ref.sort(key=lambda e: e[2])
        key = lambda e: (e[0], e[1], e[2], e[3], json.dumps(e[6], sort_keys=True))  # noqa: E731
        same = sorted(map(key, ref)) == sorted(map(key, s.events))
        print(json.dumps({"json_events": len(ref), "identical": same}))
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
