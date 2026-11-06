#!/usr/bin/env python3
"""A CI search-space point's own launch, as data, and the proof that a leg matched it.

The multi-config baseline (`adopt-ci-artifacts`) keeps, per point, the `config.yaml` the CI
job ran (`baseline/out/points/<pid>/server_logs/logs/config.yaml`). When applyback compares
against the CI number, that file is the specification of the environment, not a hint:

    ci_config.py launch <point_dir>                     -> JSON: image, model, server argv,
                                                           server env, client argv/env, conc,
                                                           duration, the fields the CI result
                                                           processor needs
    ci_config.py align  <point_dir> <inspect.json> <out>  -> alignment.json: the running
                                                           container's image, argv and env
                                                           against the point's config; exit 1
                                                           unless every item matches

`align` reads `docker inspect` of the server container, so it judges what was launched, not
what a script meant to launch. Extra server flags allowed: only the ones a launch needs and
the CI passes another way (`--model-path`, `--host`, `--port`). Extra env is allowed (the image
carries its own), a missing or different CI env var is not.

Stdlib + PyYAML.
"""
import json
import os
import shlex
import sys

import yaml

PLUMBING = {"model-path", "host", "port"}


def point_config(point_dir):
    path = os.path.join(point_dir, "server_logs", "logs", "config.yaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    roles = cfg.get("roles") or {}
    role = roles.get("agg") or next(iter(roles.values()))
    bench_env = (cfg.get("benchmark") or {}).get("env") or {}
    model = str((cfg.get("model") or {}).get("path") or "")
    model = model[3:] if model.startswith("hf:") else model
    base = json.load(open(os.path.join(point_dir, "baseline.json")))
    return {"config_yaml": path, "image": str((cfg.get("model") or {}).get("container") or "").strip(),
            "model": model, "args": role.get("args") or {}, "env": {k: str(v) for k, v in
                                                                   (role.get("env") or {}).items()},
            "gpus": role.get("gpus"), "bench_env": {k: str(v) for k, v in bench_env.items()},
            "conc": int(bench_env.get("CONC") or base["config"]["concurrency"]),
            "duration": int(bench_env.get("DURATION") or 3600),
            "point_config": base.get("config") or {}}


def server_argv(args):
    out = []
    for k, v in args.items():
        if v is True:
            out.append("--%s" % k)
        elif v is False or v is None:
            continue
        else:
            out += ["--%s" % k, str(v)]
    return out


def cmd_launch(point_dir):
    c = point_config(point_dir)
    c["server_argv"] = server_argv(c["args"])
    # The CI client argv (InferenceX benchmarks/srt_agentic.sh, as its log records it).
    c["client_argv"] = ["profile", "--model", c["model"], "--tokenizer", c["model"],
                        "--concurrency", str(c["conc"]), "--benchmark-duration",
                        str(c["duration"]), "--tokenizer-trust-remote-code"]
    c["client_env"] = {k: v for k, v in c["bench_env"].items() if k.startswith("AIPERF_")}
    print(json.dumps(c, indent=1))
    return 0


def parse_argv(tokens):
    got, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            k = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                got[k] = tokens[i + 1]
                i += 2
            else:
                got[k] = True
                i += 1
        else:
            i += 1
    return got


def norm(v):
    return "true" if v is True else str(v).strip().lower()


def cmd_align(point_dir, inspect_path, out_path):
    c = point_config(point_dir)
    ins = json.load(open(inspect_path))
    ins = ins[0] if isinstance(ins, list) else ins
    cfg = ins.get("Config") or {}
    image = cfg.get("Image") or ""
    cmd = cfg.get("Cmd") or []
    tokens = []
    for part in cmd:
        tokens += shlex.split(part) if " " in part else [part]
    got = parse_argv(tokens)
    want = {k: v for k, v in c["args"].items() if v is not False and v is not None}
    missing = sorted(k for k in want if k not in got)
    differ = sorted([k, want[k], got[k]] for k in want if k in got and norm(want[k]) != norm(got[k]))
    extra = sorted(k for k in got if k not in want and k not in PLUMBING)
    env = dict(e.split("=", 1) for e in (cfg.get("Env") or []) if "=" in e)
    env_missing = sorted(k for k in c["env"] if k not in env)
    env_differ = sorted([k, c["env"][k], env[k]] for k in c["env"] if k in env and env[k] != c["env"][k])
    doc = {"point": os.path.basename(point_dir.rstrip("/")), "config_yaml": c["config_yaml"],
           "image": {"ci": c["image"], "leg": image, "match": image == c["image"]},
           "args": {"ci": len(want), "matched": len(want) - len(missing) - len(differ),
                    "missing": missing, "differ": differ, "extra": extra},
           "env": {"ci": len(c["env"]), "matched": len(c["env"]) - len(env_missing) - len(env_differ),
                   "missing": env_missing, "differ": env_differ}}
    doc["aligned"] = (doc["image"]["match"] and not missing and not differ and not extra
                      and not env_missing and not env_differ)
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("%s: image %s, args %d/%d%s, env %d/%d%s -> %s" % (
        doc["point"], "ok" if doc["image"]["match"] else "DIFFERS",
        doc["args"]["matched"], doc["args"]["ci"],
        (" extra %s" % extra) if extra else "", doc["env"]["matched"], doc["env"]["ci"],
        (" missing %s differ %s" % (env_missing, env_differ)) if env_missing or env_differ else "",
        "ALIGNED" if doc["aligned"] else "NOT ALIGNED"))
    return 0 if doc["aligned"] else 1


def main(argv):
    if len(argv) >= 2 and argv[0] == "launch":
        return cmd_launch(argv[1])
    if len(argv) >= 4 and argv[0] == "align":
        return cmd_align(argv[1], argv[2], argv[3])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
