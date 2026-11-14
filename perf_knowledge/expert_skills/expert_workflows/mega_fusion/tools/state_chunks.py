#!/usr/bin/env python3
"""Print one list field of a mega STATE.json in chunks small enough for an agent reply.

A resumed run gets the prior state back through the setup agent's reply, and a grown candidate
registry (about 120 KB after 18 rounds) does not fit there. The orchestrator then reads each list
field a chunk at a time with this tool.

usage:
    state_chunks.py STATE.json FIELD            -> {"field": ..., "length": n, "chunks": k}
    state_chunks.py STATE.json FIELD --chunk i  -> {"field": ..., "chunk": i, "items": [...]}

Registry rows are copied without their free-text `notes`; every other field is kept as written.
"""
import argparse
import json
import sys

CHUNK_CHARS = 12000
DROP = {'candidate_registry': ('notes',)}


def items(state, field):
    value = state.get(field)
    if not isinstance(value, list):
        return []
    drop = DROP.get(field, ())
    return [{k: v for k, v in x.items() if k not in drop} if isinstance(x, dict) else x for x in value]


def chunks(values):
    out, cur, size = [], [], 0
    for x in values:
        n = len(json.dumps(x))
        if cur and size + n > CHUNK_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append(x)
        size += n
    if cur:
        out.append(cur)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('state')
    ap.add_argument('field')
    ap.add_argument('--chunk', type=int)
    a = ap.parse_args(argv)
    with open(a.state) as f:
        values = items(json.load(f), a.field)
    parts = chunks(values)
    if a.chunk is None:
        doc = {'field': a.field, 'length': len(values), 'chunks': len(parts)}
    elif 0 <= a.chunk < len(parts):
        doc = {'field': a.field, 'chunk': a.chunk, 'items': parts[a.chunk]}
    else:
        print(f'chunk {a.chunk} out of range (0..{len(parts) - 1})', file=sys.stderr)
        return 2
    print(json.dumps(doc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
