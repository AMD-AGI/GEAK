#!/usr/bin/env python3
"""The pins of a CI run, read from the run's own records, and the proof that a leg held them.

`baseline/measure` states the rule for any comparison against a named CI run: check every pin
before spending GPU time and prove every pin afterwards from the run's own records. This is that
rule as code, so every stage that measures against a CI number uses one definition of "aligned".

A CI point's records (adopt-ci-artifacts: baseline/out/points/<pid>/server_logs/logs/):

    image           config.yaml model.container (repo@sha256)
    model revisions every `models--<org>--<name>/snapshots/<sha>` the CI server loaded (model and
                    speculative draft); `main` moves upstream, the commit the CI resolved does not
    client          the `CLI Command:` aiperf recorded in agentic/benchmark.log: concurrency,
                    duration, dataset, tokenizer, every flag

    ci_pins.py pins   <point_dir>                    -> JSON of the above
    ci_pins.py hub    <point_dir> <hf_hub>           -> exit 1 unless <hf_hub>/models--*/refs/main
                                                        of every pinned repo is the pinned commit
                                                        and its snapshot exists (server is offline,
                                                        so `main` is what it loads)
    ci_pins.py prove  <point_dir> <server.log> <benchmark.log> <out.json>
                                                     -> exit 1 unless the leg's server loaded the
                                                        pinned commit of every repo and its client
                                                        ran the CI's argv (url/output dir aside,
                                                        plus --tokenizer-revision <pinned commit>)

Stdlib + PyYAML.
"""
import json
import os
import re
import shlex
import sys

import yaml

SNAP_RE = re.compile(r"models--([A-Za-z0-9._-]+?)--([A-Za-z0-9._-]+)/snapshots/([0-9a-f]{40})")
# Flags whose value is the run's own plumbing, not part of what was measured.
PLUMBING_FLAGS = {"--url", "--server-metrics", "--output-artifact-dir"}
# Flags a leg may add because they pin what the CI resolved implicitly at the time.
PIN_FLAGS = {"--tokenizer-revision"}


def logs_dir(point_dir):
    return os.path.join(point_dir, "server_logs", "logs")


def snapshots(text):
    """{'org/name': {commit, ...}} for every HF snapshot path in a log."""
    out = {}
    for org, name, sha in SNAP_RE.findall(text):
        out.setdefault("%s/%s" % (org, name), set()).add(sha)
    return out


def cli_command(text):
    """aiperf's recorded `CLI Command:` (it wraps lines) as argv tokens, or None."""
    m = re.search(r"CLI Command:\s*(.*?)\n(?:Benchmark Duration|CSV Export|JSON Export|\S+:\s)",
                  text, re.S)
    if not m:
        return None
    joined = " ".join(line.strip() for line in m.group(1).splitlines())
    toks = shlex.split(joined)
    return toks


def argv_map(toks):
    out, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 2
            else:
                out[t] = True
                i += 1
        else:
            out.setdefault("_positional", []).append(t)
            i += 1
    return out


def pins(point_dir):
    d = logs_dir(point_dir)
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
    server_text = "".join(open(os.path.join(d, f), errors="replace").read()
                          for f in sorted(os.listdir(d)) if f.endswith(".out"))
    snaps = snapshots(server_text)
    ambiguous = {k: sorted(v) for k, v in snaps.items() if len(v) != 1}
    bench = os.path.join(d, "agentic", "benchmark.log")
    cli = cli_command(open(bench, errors="replace").read()) if os.path.exists(bench) else None
    model = str((cfg.get("model") or {}).get("path") or "")
    model = model[3:] if model.startswith("hf:") else model
    doc = {"image": str((cfg.get("model") or {}).get("container") or "").strip(),
           "model": model,
           "revisions": {k: next(iter(v)) for k, v in snaps.items() if len(v) == 1},
           "ambiguous_revisions": ambiguous,
           "client_cli": cli}
    if cli:
        a = argv_map(cli)
        doc["client"] = {"dataset": a.get("--public-dataset"), "concurrency": a.get("--concurrency"),
                         "duration": a.get("--benchmark-duration"), "tokenizer": a.get("--tokenizer")}
    problems = []
    if not doc["image"] or "@sha256:" not in doc["image"]:
        problems.append("the CI image is not digest-pinned")
    if model not in doc["revisions"]:
        problems.append("the CI server log names no single snapshot of %s" % model)
    if ambiguous:
        problems.append("the CI server loaded several snapshots of %s" % sorted(ambiguous))
    if not cli:
        problems.append("the CI client log records no CLI Command")
    doc["problems"] = problems
    return doc


def check_hub(point_dir, hub):
    p = pins(point_dir)
    bad = list(p["problems"])
    for repo, sha in sorted(p["revisions"].items()):
        mdir = os.path.join(hub, "models--" + repo.replace("/", "--"))
        try:
            main = open(os.path.join(mdir, "refs", "main")).read().strip()
        except OSError:
            main = None
        if main != sha:
            bad.append("%s: %s/refs/main is %s, the CI loaded %s" % (repo, mdir, main, sha))
        if not os.path.isdir(os.path.join(mdir, "snapshots", sha)):
            bad.append("%s: snapshot %s is not in %s" % (repo, sha, mdir))
    for b in bad:
        print("PIN: " + b, file=sys.stderr)
    return 1 if bad else 0


def prove(point_dir, server_log, bench_log, out_path):
    p = pins(point_dir)
    leg_snaps = snapshots(open(server_log, errors="replace").read())
    rev = {}
    for repo, sha in sorted(p["revisions"].items()):
        got = sorted(leg_snaps.get(repo, set()))
        rev[repo] = {"ci": sha, "leg": got, "held": got == [sha]}
    leg_cli = cli_command(open(bench_log, errors="replace").read()) if os.path.exists(bench_log) else None
    client = {"held": False, "why": "the leg's client log records no CLI Command"}
    if p["client_cli"] and leg_cli:
        ci_a, leg_a = argv_map(p["client_cli"]), argv_map(leg_cli)
        keys = (set(ci_a) | set(leg_a)) - PLUMBING_FLAGS - PIN_FLAGS
        differ = sorted([k, ci_a.get(k), leg_a.get(k)] for k in keys if ci_a.get(k) != leg_a.get(k))
        model_rev = p["revisions"].get(p["model"])
        added = {k: leg_a.get(k) for k in PIN_FLAGS if k in leg_a}
        bad_pin = [k for k, v in added.items() if v != model_rev]
        client = {"held": not differ and not bad_pin, "differ": differ, "pin_flags": added,
                  "ci_cli": p["client_cli"], "leg_cli": leg_cli}
    doc = {"image": p["image"], "revisions": rev, "client": client, "problems": p["problems"]}
    doc["proven"] = (not p["problems"] and all(r["held"] for r in rev.values())
                     and client["held"])
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("pins: revisions %s, client %s -> %s" % (
        ", ".join("%s %s" % (k, "ok" if v["held"] else "DIFFERS %s vs %s" % (v["leg"], v["ci"]))
                  for k, v in rev.items()),
        "ok" if client["held"] else "DIFFERS %s" % client.get("differ") or client.get("why"),
        "PROVEN" if doc["proven"] else "NOT PROVEN"))
    return 0 if doc["proven"] else 1


def main(argv):
    if len(argv) == 2 and argv[0] == "pins":
        print(json.dumps(pins(argv[1]), indent=1, default=sorted))
        return 0
    if len(argv) == 3 and argv[0] == "hub":
        return check_hub(argv[1], argv[2])
    if len(argv) == 5 and argv[0] == "prove":
        return prove(*argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
