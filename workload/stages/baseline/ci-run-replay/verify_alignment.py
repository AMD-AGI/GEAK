#!/usr/bin/env python3
"""Prove the alignment held, from the run's own records. Requesting it is not having it.

`preflight.py` checks what we *asked* for. This checks what the run *did*, and the two
are not the same thing: on this cluster an environment variable has already failed to
reach a benchmark subprocess (a HIP-only device mask that never crossed into the
benchmark, so the pin silently landed on the wrong GPU). The same class of failure here
would leave the server compiling into a warm cache outside the run while every log line
says the cache was cold.

So each pin is re-read from an artifact the run itself produced:

| Pin | Proof |
| --- | --- |
| image digest | `docker inspect` of the container that actually served |
| cold compile caches | the cache directories were empty before and are **not** empty after. An empty cache after an hour of serving means the variable never arrived |
| benchmark_id | the client's own export; then written to the ledger so it is never reissued |
| cache-bust target | the client's recorded CLI/config |
| seed | the client's recorded CLI/config |
| CPU binding | `sched_getaffinity` sampled inside the container |
| the number | output throughput against the CI figure, within the declared band |

Anything that cannot be found is reported `unproven` and fails the run unless
`--allow-unproven` is given; "we could not check" must never render as a tick.

Usage: verify_alignment.py --params P --alignment A --out D --run-dir R
                           --container-info C [--affinity F] [--ledger L]
"""

import argparse
import json
import os
import sys

THROUGHPUT_PATHS = ("request_metrics.throughput.output.tokens_per_second",
                    "output_tok_per_s", "output_tokens_per_second",
                    "output_token_throughput")
BID_KEYS = ("benchmark_id",)
SEED_KEYS = ("random_seed", "seed")
BUST_KEYS = ("cache_bust", "cache_bust_target", "cache_bust_marker_target")


def rj(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def dig(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def find_key(node, keys, depth=0):
    """First value for any of `keys`, anywhere in the document."""
    if depth > 12:
        return None
    if isinstance(node, dict):
        for k in keys:
            v = node.get(k)
            if v not in (None, "", [], {}):
                return v
        for v in node.values():
            got = find_key(v, keys, depth + 1)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = find_key(v, keys, depth + 1)
            if got is not None:
                return got
    return None


def run_documents(run_dir, limit=60):
    docs = []
    for root, _dirs, files in os.walk(run_dir):
        for n in sorted(files):
            if n.endswith(".json") and len(docs) < limit:
                d = rj(os.path.join(root, n))
                if d is not None:
                    docs.append((os.path.join(root, n), d))
        if len(docs) >= limit:
            break
    return docs


def count_files(d):
    if not os.path.isdir(d):
        return -1
    return sum(len(files) for _r, _d, files in os.walk(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--container-info", default=None)
    ap.add_argument("--affinity", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--allow-unproven", action="store_true")
    a = ap.parse_args()

    p = json.load(open(a.params))
    align = json.load(open(a.alignment))
    pins = p.get("pins") or {}
    ci = p.get("ci") or {}
    checks, failures, unproven = [], [], []

    def record(name, ok, detail):
        checks.append({"check": name, "status": "ok" if ok else
                       ("unproven" if ok is None else "FAILED"), "detail": detail})
        if ok is None:
            unproven.append("%s: %s" % (name, detail))
        elif not ok:
            failures.append("%s: %s" % (name, detail))

    # --- 1. the image that actually served
    cinfo = rj(a.container_info) if a.container_info else None
    if not cinfo:
        record("image_digest", None, "no container_info.json was captured")
    else:
        c = cinfo[0] if isinstance(cinfo, list) else cinfo
        digests = (c.get("Image"), (c.get("Config") or {}).get("Image"))
        want = pins.get("image_digest") or ""
        hit = any(want and want in str(x) for x in digests)
        # The container is started by digest reference, so Config.Image carries it; Image
        # is the local image id, which is a different hash and is recorded, not compared.
        record("image_digest", bool(hit),
               "container ran %s; pinned %s" % (list(digests), want))

    # --- 2. the compile caches were cold and were used
    pre = align.get("cache_vars") or {}
    for var, d in sorted(pre.items()):
        n = count_files(d)
        if n < 0:
            record("cache_%s" % var, False, "%s does not exist after the run" % d)
        elif n == 0:
            record("cache_%s" % var, False,
                   "%s is still empty after the run. The server compiled into some other "
                   "cache, which means %s never reached it -- the measurement is not on a "
                   "cold cache and may not even be on this image." % (d, var))
        else:
            record("cache_%s" % var, True, "%s filled with %d files (cold at start)"
                   % (d, n))

    # --- 3/4/5. the client's own record of id, seed and cache-bust
    docs = run_documents(a.run_dir)
    bid = seed = bust = None
    for _path, d in docs:
        bid = bid or find_key(d, BID_KEYS)
        seed = seed if seed is not None else find_key(d, SEED_KEYS)
        bust = bust or find_key(d, BUST_KEYS)
    cli = ""
    for _path, d in docs:
        v = find_key(d, ("cli_command", "command", "replay_cmd"))
        if v:
            cli = v if isinstance(v, str) else " ".join(map(str, v))
            break

    want_bid = str(pins.get("benchmark_id") or "")
    if bid is None and want_bid and want_bid in cli:
        bid = want_bid
    if bid is None:
        record("benchmark_id", None, "no benchmark_id in any run document")
    else:
        record("benchmark_id", str(bid) == want_bid,
               "run recorded %r, pinned %r" % (bid, want_bid))

    want_seed = pins.get("seed")
    if seed is None:
        record("seed", None, "no seed in any run document")
    else:
        record("seed", str(seed) == str(want_seed),
               "run recorded %r, pinned %r" % (seed, want_seed))

    want_bust = pins.get("cache_bust")
    if bust is None and cli:
        bust = "first_turn_prefix" if "first_turn_prefix" in cli else None
    if bust is None:
        record("cache_bust", None, "no cache-bust target in any run document")
    else:
        record("cache_bust", str(bust).lower().endswith(str(want_bust).lower()),
               "run recorded %r, pinned %r" % (bust, want_bust))

    # --- 5b. the corpus generation. A loader name, not a file, on the AgentX harness:
    # 062126 and 061526 are different prompt-length mixes and therefore different numbers.
    corpus = pins.get("trace_corpus") or {}
    want_loader = corpus.get("loader")
    if want_loader:
        got_loader = None
        for _path, d in docs:
            got_loader = got_loader or find_key(d, ("public_dataset", "dataset", "loader"))
        if got_loader is None and cli:
            toks = cli.split()
            if "--public-dataset" in toks:
                got_loader = toks[toks.index("--public-dataset") + 1]
        if got_loader is None:
            record("trace_corpus", None, "no dataset/loader in any run document or CLI")
        else:
            record("trace_corpus", want_loader in str(got_loader),
                   "run recorded %r, pinned %r" % (got_loader, want_loader))

    # --- 6. the CPU mask the server actually had
    aff = rj(a.affinity) if a.affinity else None
    plan = sorted((align.get("cpu_plan") or {}).get("cpus") or [])
    if not aff:
        record("cpu_binding", None, "no affinity sample was taken inside the container")
    else:
        got = sorted(aff.get("cpus") or [])
        record("cpu_binding", got == plan,
               "container affinity %d cpus, plan %d cpus%s"
               % (len(got), len(plan),
                  "" if got == plan else " — first difference: %s"
                  % sorted(set(got) ^ set(plan))[:8]))

    # --- 7. the number
    tput = None
    for _path, d in docs:
        for path in THROUGHPUT_PATHS:
            v = dig(d, path)
            if isinstance(v, (int, float)):
                tput = float(v)
                break
        if tput is not None:
            break
    ref = ci.get("output_throughput")
    drift = None
    if tput is None:
        record("throughput", None, "no output throughput found under %s" % (THROUGHPUT_PATHS,))
    elif not ref:
        record("throughput", None, "measured %.5f but ci.output_throughput is unset" % tput)
    else:
        drift = (tput - ref) / ref * 100.0
        band = float(ci.get("max_drift_pct", 0.5))
        record("throughput", abs(drift) <= band,
               "measured %.5f vs CI %.5f = %+.4f%% (band ±%.2f%%)"
               % (tput, ref, drift, band))

    verdict = "FAILED" if failures else ("unproven" if unproven else "aligned")
    doc = {"verdict": verdict, "checks": checks, "failures": failures,
           "unproven": unproven, "throughput": tput, "ci_reference": ref,
           "drift_pct": None if drift is None else round(drift, 4),
           "benchmark_id": bid, "seed": seed, "cache_bust": bust}
    with open(os.path.join(a.out, "alignment_verified.json"), "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    # The id is spent whether or not the run reproduced: its cache-bust marker warmed a
    # keyspace, so it must never be issued again.
    if a.ledger and want_bid:
        led = rj(a.ledger) or {"used": []}
        led.setdefault("used", []).append(
            {"benchmark_id": want_bid, "run_dir": a.run_dir, "verdict": verdict})
        os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
        with open(a.ledger, "w") as f:
            json.dump(led, f, indent=2)

    for c in checks:
        print("  %-9s %-16s %s" % (c["status"], c["check"], c["detail"]))
    if failures:
        return 1
    if unproven and not a.allow_unproven:
        print("alignment could not be proven; 'not checked' is not a pass.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
