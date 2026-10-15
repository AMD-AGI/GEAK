#!/usr/bin/env python3
"""What a run is optimizing, and the two checks that keep its name honest.

A run of this pipeline costs GPU-days and ends in a report. The report is read by people
who were not there, and the first thing they read is a name: the config file and the run
directory. In the first full campaign that name was `agentx_dsv4` while the stack under
test was `amd/GLM-5.2-MXFP4` -- a label from the planning phase that stayed after the
target changed. No stage had any reason to contradict it, so it propagated into every
artifact path, every log line and every table header. Nothing was measured wrong;
everything was labelled wrong, which is worse: a wrong number invites doubt, a wrong name
invites confidence.

Hence two checks, both free, both run before a single GPU hour is spent:

  * **naming** -- a scenario config must declare `target.model`, and the run directory
    must carry a recognisable token of it. `agentx_dsv4` and `amd/GLM-5.2-MXFP4` share no
    token, so `stage.py plan` refuses the run. An abbreviation the tokeniser cannot derive
    (`dsv4` for DeepSeek-V4) is still allowed, but only by writing it into
    `target.name_tokens` -- a deliberate declaration instead of an accident.
  * **identity** -- the declaration is cross-checked against what the stages actually
    observed and recorded in their own artifacts (image, component commits, model). A
    declaration that nobody corroborates is reported as `unverified` rather than trusted,
    and one the artifacts contradict fails the stage.

Neither check guesses. Fields only one side knows are skipped and listed as skipped.
"""

import os
import re

# Where a stage may record what it saw. A stage is free to write none of these -- then its
# identity verdict is `unverified`, which is a statement, not a pass.
IDENTITY_FILES = ("stack_identity.json", "identity.json", "stack.json", "baseline.json")

# `stack_identity.json` may carry several stacks (one of them explicitly superseded). When
# a document names the one in force, only that block is read.
ACTIVE_BLOCKS = ("active_stack", "stack_in_force", "target_stack")

MODEL_KEYS = ("model", "served_model", "served_model_name", "model_name", "model_path")
IMAGE_KEYS = ("image", "docker_image", "container_image")


# ---------------------------------------------------------------- tokens

def _words(s):
    return [w for w in re.split(r"[^0-9a-z]+", s.lower()) if w]


def name_tokens(model):
    """Tokens by which a run directory may legitimately refer to `model`.

    The organisation prefix is dropped: `amd/` describes who published the weights, not
    which stack is under test, and letting it count would bless `agentx_amd` for every
    model AMD ever uploaded. Version digits are folded onto the preceding word because
    `GLM-5.2` is written `glm52` and `glm5` at least as often as `glm-5.2`.
    """
    ws = _words(str(model).split("/")[-1])
    toks = set()
    for i, w in enumerate(ws):
        if w.isdigit():
            continue
        if len(w) >= 3:
            toks.add(w)
        acc = w
        for nxt in ws[i + 1:]:
            if not nxt.isdigit():
                break
            acc += nxt
            if len(acc) >= 3:
                toks.add(acc)
    return sorted(toks)


def check_name(name, model, extra=()):
    """(hits, tokens) -- which declared tokens the name carries, and all of them."""
    slug = "".join(ch for ch in str(name).lower() if ch.isalnum())
    toks = name_tokens(model)
    for e in extra or ():
        e = "".join(ch for ch in str(e).lower() if ch.isalnum())
        if e:
            toks.append(e)
    toks = sorted(set(toks))
    return [t for t in toks if t in slug], toks


# ---------------------------------------------------------------- declaration

def declared(cfg):
    t = cfg.get("target")
    if not isinstance(t, dict):
        return {}
    return dict((k, v) for k, v in t.items() if not k.startswith("_"))


def naming_problems(cfg, run_root, config_paths=()):
    """Problems to report before the run starts. Empty list means the name is defensible."""
    t = declared(cfg)
    if not t.get("enforce", True):
        return []
    model = t.get("model")
    run_name = os.path.basename(os.path.normpath(run_root))
    if not model:
        if run_name == "default" and not [p for p in config_paths
                                          if os.path.basename(p) != "default.json"]:
            return []          # the bare defaults are a template, not a run
        return ["this config does not declare target.model, so nothing can check that "
                "run_root %r names the stack it optimizes. Add a target block (see "
                "config/default.json) or set target.enforce=false and say why." % run_name]

    problems = []
    names = [("run_root", run_name)]
    names += [("config", os.path.basename(p)) for p in config_paths
              if os.path.basename(p) != "default.json"]
    for what, name in names:
        hits, toks = check_name(name, model, t.get("name_tokens"))
        if not hits:
            problems.append(
                "%s is named %r but target.model is %r, and the name carries none of %s. "
                "A run whose name disagrees with its stack misleads every reader of the "
                "report. Rename it, or -- if the name is a deliberate abbreviation -- add "
                "it to target.name_tokens." % (what, name, model, toks))

    # The config may also contradict itself: a stage pinned to an image other than the one
    # the run claims to be optimizing produces kernels for a system nobody is measuring.
    want_image = t.get("image")
    if want_image:
        for stage, spec in sorted((cfg.get("stages") or {}).items()):
            got = ((spec or {}).get("params") or {}).get("image")
            if got and got.strip() != want_image.strip():
                problems.append(
                    "stages.%s.params.image is %r but target.image is %r. One of the two "
                    "is wrong: the stage would work on a stack the run does not claim to "
                    "be optimizing." % (stage, got, want_image))
    return problems


# ---------------------------------------------------------------- observation

def _collect(node, into):
    if isinstance(node, dict):
        for k, v in node.items():
            kl = k.lower()
            if isinstance(v, str) and v.strip():
                if kl in MODEL_KEYS:
                    into.setdefault("model", v.strip())
                elif kl in IMAGE_KEYS and "/" in v:
                    into.setdefault("image", v.strip())
                elif kl.endswith("_commit"):
                    into.setdefault("commits", {}).setdefault(kl[: -len("_commit")], v.strip())
            elif isinstance(v, (dict, list)):
                _collect(v, into)
    elif isinstance(node, list):
        for v in node:
            _collect(v, into)
    return into


def observed_identity(out_dir, result=None, read_json=None):
    """What this stage recorded about the stack it ran against.

    Read from `result.json`'s own `stack` field first -- a stage that wants to be checked
    says so in the contract object -- then from the well-known artifact names. A document
    that names the stack in force (`active_stack`) is read only there: the same file also
    records superseded stacks, and reading those would compare against a stack this run
    deliberately abandoned.
    """
    if read_json is None:
        import json

        def read_json(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None

    docs = []
    if isinstance(result, dict) and isinstance(result.get("stack"), dict):
        docs.append(result["stack"])
    for name in IDENTITY_FILES:
        d = read_json(os.path.join(out_dir, name))
        if isinstance(d, dict):
            docs.append(d)

    found = {}
    for d in docs:
        for b in ACTIVE_BLOCKS:
            if isinstance(d.get(b), dict):
                d = d[b]
                break
        _collect(d, found)
    return found


def _same_commit(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def identity_report(cfg, observed):
    """{verdict, matched, mismatched, skipped} comparing declaration against observation."""
    t = declared(cfg)
    matched, mismatched, skipped = [], [], []

    def cmp(field, want, got, eq):
        if not want or not got:
            skipped.append(field + (" (not declared)" if not want else " (not observed)"))
        elif eq(str(want), str(got)):
            matched.append("%s = %s" % (field, got))
        else:
            mismatched.append("%s: the config declares %r, the run observed %r"
                              % (field, want, got))

    def model_eq(a, b):
        return a.strip().lower().split("/")[-1] == b.strip().lower().split("/")[-1]

    cmp("model", t.get("model"), observed.get("model"), model_eq)
    cmp("image", t.get("image"), observed.get("image"), lambda a, b: a.strip() == b.strip())
    want_c = t.get("commits") or {}
    got_c = observed.get("commits") or {}
    for k in sorted(set(want_c) | set(got_c)):
        cmp("commits.%s" % k, want_c.get(k), got_c.get(k), _same_commit)

    verdict = "mismatch" if mismatched else ("ok" if matched else "unverified")
    return {"verdict": verdict, "matched": matched, "mismatched": mismatched,
            "skipped": skipped}


def identity_problems(rep):
    if rep["verdict"] != "mismatch":
        return []
    return ["the stack this stage ran against is not the stack the config declares -- "
            "one of the two is wrong, and a report built on top of this would name the "
            "wrong system: " + "; ".join(rep["mismatched"])]
