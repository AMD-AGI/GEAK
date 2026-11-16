#!/usr/bin/env python3
"""The recipe an InferenceX CI point ran (its `logs/config.yaml`), parsed once.

adopt-ci-artifacts (grouping by launch settings), applyback/ci_config.py (the leg's launch and
its alignment proof) and _shared/ci_pins.py (image/model pins) all read the same file; this is
the one reader, and the one argv convention they share.

    load(config_yaml)  -> {"path", "image", "model", "roles", "bench_env"}
    role(recipe)       -> the serving role {"args", "env", "gpus"}: `agg`, else the first one
    server_argv(args)  -> the role's args as sglang flags
    argv_map(tokens)   -> {"--flag": value | [values] | True, "_positional": [...]}

A flag's value may be a list in the recipe (`cuda-graph-bs: [1, 2, 4]`): it is passed as
`--cuda-graph-bs 1 2 4` and read back as the list ['1', '2', '4'].

PyYAML is imported on first use, so a caller that never reads a recipe does not need it.
"""
import os


def recipe_path(point_dir):
    return os.path.join(point_dir, "server_logs", "logs", "config.yaml")


def load(config_yaml):
    import yaml
    with open(config_yaml) as fh:
        doc = yaml.safe_load(fh) or {}
    model_doc = doc.get("model") or {}
    model = str(model_doc.get("path") or "")
    model = model[3:] if model.startswith("hf:") else model
    bench_env = (doc.get("benchmark") or {}).get("env") or {}
    return {"path": config_yaml, "image": str(model_doc.get("container") or "").strip(),
            "model": model, "roles": doc.get("roles") or {},
            "bench_env": {str(k): str(v) for k, v in bench_env.items()}}


def role(recipe, name="agg"):
    roles = recipe["roles"]
    r = roles.get(name) or next(iter(roles.values()), None) or {}
    return {"args": {str(k): v for k, v in (r.get("args") or {}).items()},
            "env": {str(k): str(v) for k, v in (r.get("env") or {}).items()},
            "gpus": r.get("gpus")}


def server_argv(args, skip=()):
    out = []
    for k, v in args.items():
        if k in skip or v is False or v is None:
            continue
        if v is True:
            out.append("--" + k)
        elif isinstance(v, (list, tuple)):
            out += ["--" + k] + [str(x) for x in v]
        else:
            out += ["--" + k, str(v)]
    return out


def argv_map(toks):
    """A flag takes every value up to the next flag (a list when there are several): the CI's
    `--server-metrics <server>/metrics <gpu-exporter>/metrics` is one flag, not a flag plus a
    positional (2026-11-09: the second URL read as a positional failed every leg's pins)."""
    out, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            j = i + 1
            while j < len(toks) and not toks[j].startswith("--"):
                j += 1
            vals = toks[i + 1:j]
            out[t] = True if not vals else vals[0] if len(vals) == 1 else vals
            i = j
        else:
            out.setdefault("_positional", []).append(t)
            i += 1
    return out
