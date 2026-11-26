#!/usr/bin/env python3
"""Print one list field of a mega STATE.json in chunks small enough for an agent reply.

A resumed run gets the prior state back through the setup agent's reply, and a grown candidate
registry (about 120 KB after 18 rounds) does not fit there. The orchestrator then reads each list
field a chunk at a time with this tool.

usage:
    state_chunks.py STATE.json FIELD            -> {"field": ..., "length": n, "chunks": k}
    state_chunks.py STATE.json FIELD --chunk i  -> {"field": ..., "chunk": i, "items": [...]}

Registry rows are copied without their free-text `notes` (also in `working_snapshot`), and each
structural contract failure keeps its id, category, severity and tier but not its messages: one skill
run's row reached 62 KB, mostly failure messages, and the next structural check rewrites them anyway.
Every other field is kept as written.
"""
import argparse
import json
import sys

CHUNK_CHARS = 12000
DROP = {'candidate_registry': ('notes',)}


def short_failures(failures):
    if not isinstance(failures, list):
        return failures
    return [{k: v for k, v in f.items() if k != 'messages'} if isinstance(f, dict) else f
            for f in failures]


def compact(row):
    row = dict(row)
    if 'contract_failures' in row:
        row['contract_failures'] = short_failures(row['contract_failures'])
    snap = row.get('working_snapshot')
    if isinstance(snap, dict):
        snap = {k: v for k, v in snap.items() if k != 'notes'}
        if 'contract_failures' in snap:
            snap['contract_failures'] = short_failures(snap['contract_failures'])
        row['working_snapshot'] = snap
    return row


def items(state, field):
    value = state.get(field)
    if not isinstance(value, list):
        return []
    drop = DROP.get(field, ())
    out = [{k: v for k, v in x.items() if k not in drop} if isinstance(x, dict) else x for x in value]
    if field == 'candidate_registry':
        out = [compact(x) if isinstance(x, dict) else x for x in out]
    return out


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
