#!/usr/bin/env python3
"""The stage driver for workload/.

It does three things and no more: resolve the config to locate a stage skill, prepare the
stage's run environment, and validate the `result.json` the stage hands back before
recording it in `STATE.json`. All stage logic lives in `stages/<stage>/<skill>/`; this
file knows no individual skill. Contract in ../stages/CONTRACT.md.

    stage.py plan                      # resolve and print where all five stages land (= --dry-run)
    stage.py prepare <stage>           # create OUT, write params.json/env.sh, print the SKILL.md path
    stage.py run <stage>               # prepare + execute + validate, script or agent mode
    stage.py all [stage ...]           # every stage in order, unattended; resumes by default
                                       #   (reports every 30 min via scripts/watch.py;
                                       #    --no-report opts out)
    stage.py validate <stage>          # validate $OUT/result.json on its own
    stage.py status                    # print STATE.json
    stage.py next                      # print the next pending stage (for resume); empty when done

Common arguments:
    --config F        layer in a config, repeatable (default.json is always at the bottom)
    --set a.b.c=v     point override, repeatable; v is parsed as JSON, falling back to a string
    --run-root DIR    override run_root
"""

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target  # noqa: E402  -- the naming/identity gate; see target.py for why it exists

WORKLOAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEAK_ROOT = os.path.dirname(WORKLOAD_ROOT)
DEFAULT_CONFIG = os.path.join(WORKLOAD_ROOT, "config", "default.json")
VALID_STATUS = ("ok", "blocked", "fail")


# ---------------------------------------------------------------- config

def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_set(cfg, expr):
    if "=" not in expr:
        raise SystemExit("--set expects a.b=value, got %r" % expr)
    path, raw = expr.split("=", 1)
    try:
        val = json.loads(raw)
    except ValueError:
        val = raw
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = val
    return cfg


def load_config(args):
    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    for path in args.config or []:
        p = resolve_path(path)
        if not os.path.exists(p):
            raise SystemExit("no such config: %s" % p)
        with open(p) as f:
            cfg = deep_merge(cfg, json.load(f))
    for expr in args.set or []:
        cfg = apply_set(cfg, expr)
    if args.run_root:
        cfg["run_root"] = args.run_root
    # Which files produced this configuration. Recorded as provenance, and read by the
    # naming gate: the config's own filename is part of what a reader sees.
    cfg["_config_files"] = [DEFAULT_CONFIG] + [resolve_path(p) for p in args.config or []]
    return cfg


def target_problems(cfg):
    """The naming half of the target gate (target.py). Runs before any GPU time."""
    return target.naming_problems(cfg, resolve_path(cfg["run_root"]),
                                  cfg.get("_config_files") or [])


def resolve_path(path):
    """Relative paths always resolve against the GEAK repository root. Site-local
    absolute paths may only come from config, never from code."""
    if not path:
        return path
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(GEAK_ROOT, path))


# ---------------------------------------------------------------- layout

def stage_layout(cfg, stage):
    order = cfg["order"]
    if stage not in order:
        raise SystemExit("unknown stage %r; order contains only %s" % (stage, order))
    spec = cfg["stages"].get(stage)
    if not spec:
        raise SystemExit("the config has no stages.%s" % stage)
    skill = spec.get("skill")
    if not skill:
        raise SystemExit("stages.%s does not name a skill" % stage)

    run_root = resolve_path(cfg["run_root"])
    skill_root = os.path.join(WORKLOAD_ROOT, "stages", stage, skill)
    out = os.path.join(run_root, stage, "out")

    idx = order.index(stage)
    prev_out = ""
    if idx > 0:
        prev_out = os.path.join(run_root, order[idx - 1], "out")

    runner = cfg.get("runner", "claude")
    return {
        "stage": stage,
        "skill": skill,
        "mode": spec.get("mode", "agent"),
        "params": spec.get("params", {}),
        "skill_root": skill_root,
        "skill_md": os.path.join(skill_root, "SKILL.md"),
        "run_sh": os.path.join(skill_root, "run.sh"),
        "run_root": run_root,
        "out": out,
        "in": prev_out,
        "runner": os.path.join(WORKLOAD_ROOT, "runners", runner + ".sh"),
    }


def stage_env(cfg, lay):
    env = dict(os.environ)
    env.update({
        "GEAK_ROOT": GEAK_ROOT,
        "WORKLOAD_ROOT": WORKLOAD_ROOT,
        "SKILL_ROOT": lay["skill_root"],
        "STAGE": lay["stage"],
        "STAGE_SKILL": lay["skill"],
        "RUN_ROOT": lay["run_root"],
        "IN": lay["in"],
        "OUT": lay["out"],
        "CONFIG": os.path.join(lay["run_root"], "config.resolved.json"),
        "PARAMS": os.path.join(lay["out"], "params.json"),
        "RUNNER": lay["runner"],
    })
    # runner_env knobs are exported as WORKLOAD_AGENT_* by convention; the runner reads
    # them itself. No credential appears here — only the path to an env file outside the
    # repository.
    re = cfg.get("runner_env") or {}
    if re.get("state_dir_base"):
        env["WORKLOAD_AGENT_STATE_BASE"] = re["state_dir_base"]
    if re.get("env_file"):
        env["WORKLOAD_AGENT_ENV_FILE"] = resolve_path(re["env_file"])
    if re.get("model"):
        env["WORKLOAD_AGENT_MODEL"] = re["model"]
    return env


# ---------------------------------------------------------------- state

def state_path(cfg):
    return os.path.join(resolve_path(cfg["run_root"]), "STATE.json")


def load_state(cfg):
    p = state_path(cfg)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"run_root": resolve_path(cfg["run_root"]), "stages": {}}


def save_state(cfg, state):
    p = state_path(cfg)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


# ---------------------------------------------------------------- checks

def check_skill(lay, problems):
    if not os.path.isdir(lay["skill_root"]):
        problems.append("%s: no such skill directory %s" % (lay["stage"], lay["skill_root"]))
        return
    if not os.path.exists(lay["skill_md"]):
        problems.append("%s: missing SKILL.md (contract rule 0) %s" % (lay["stage"], lay["skill_md"]))
    if lay["mode"] == "script" and not os.path.exists(lay["run_sh"]):
        problems.append("%s: mode=script but there is no run.sh %s" % (lay["stage"], lay["run_sh"]))


def validate_result(lay):
    """Returns (result_dict, [errors])."""
    rp = os.path.join(lay["out"], "result.json")
    if not os.path.exists(rp):
        return None, ["no result.json was written: %s" % rp]
    try:
        with open(rp) as f:
            res = json.load(f)
    except ValueError as e:
        return None, ["result.json is not valid JSON: %s" % e]

    errs = []
    if not isinstance(res, dict):
        return None, ["the top level of result.json must be an object"]
    if res.get("status") not in VALID_STATUS:
        errs.append("status must be one of %s, got %r" % (list(VALID_STATUS), res.get("status")))
    if res.get("stage") != lay["stage"]:
        errs.append("the stage field is %r, expected %r" % (res.get("stage"), lay["stage"]))
    if res.get("skill") != lay["skill"]:
        errs.append("the skill field is %r, expected %r" % (res.get("skill"), lay["skill"]))

    arts = res.get("artifacts") or {}
    if not isinstance(arts, dict):
        errs.append("artifacts must be an object")
        arts = {}
    for name, rel in arts.items():
        if not isinstance(rel, str):
            errs.append("artifacts.%s must be a string path relative to $OUT" % name)
            continue
        if os.path.isabs(rel):
            errs.append("artifacts.%s is the absolute path %r; the contract requires a path "
                        "relative to $OUT" % (name, rel))
            continue
        if not os.path.exists(os.path.join(lay["out"], rel)):
            errs.append("artifacts.%s points at %r, which does not exist" % (name, rel))

    metrics = res.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        errs.append("metrics must be an object")

    if res.get("status") == "ok" and not arts:
        errs.append("status=ok but artifacts is empty; an ok with no artifacts is "
                    "unusable downstream")
    if res.get("status") in ("blocked", "fail") and not (res.get("notes") or "").strip():
        errs.append("status=%s requires notes stating the cause and how to rerun once it "
                    "is resolved" % res["status"])
    return res, errs


# ---------------------------------------------------------------- commands

def cmd_plan(cfg, args):
    problems = list(target_problems(cfg))
    tgt = target.declared(cfg)
    print("GEAK_ROOT      %s" % GEAK_ROOT)
    print("run_root       %s" % resolve_path(cfg["run_root"]))
    print("runner         %s" % cfg.get("runner"))
    print("target         %s" % (tgt.get("model") or "(undeclared)"))
    if tgt.get("image"):
        print("               image %s" % tgt["image"])
    state = load_state(cfg)
    print()
    for stage in cfg["order"]:
        lay = stage_layout(cfg, stage)
        check_skill(lay, problems)
        st = state["stages"].get(stage, {}).get("status", "-")
        print("%-10s %-38s mode=%-6s state=%s" % (stage, lay["skill"], lay["mode"], st))
        print("    skill  %s" % os.path.relpath(lay["skill_root"], GEAK_ROOT))
        print("    in     %s" % (lay["in"] or "(none, first stage)"))
        print("    out    %s" % lay["out"])

    runner = os.path.join(WORKLOAD_ROOT, "runners", cfg.get("runner", "claude") + ".sh")
    if not os.path.exists(runner):
        problems.append("no such runner: %s" % runner)

    print()
    if problems:
        print("resolution problems:")
        for p in problems:
            print("  - %s" % p)
        return 1
    print("all five stages resolved successfully.")
    return 0


def cmd_prepare(cfg, args):
    lay = stage_layout(cfg, args.stage)
    problems = list(target_problems(cfg))
    check_skill(lay, problems)
    if problems:
        for p in problems:
            print("ERROR %s" % p, file=sys.stderr)
        return 1

    os.makedirs(lay["out"], exist_ok=True)
    os.makedirs(lay["run_root"], exist_ok=True)
    with open(os.path.join(lay["run_root"], "config.resolved.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(lay["out"], "params.json"), "w") as f:
        json.dump(lay["params"], f, indent=2, ensure_ascii=False)

    env = stage_env(cfg, lay)
    keys = ["GEAK_ROOT", "WORKLOAD_ROOT", "SKILL_ROOT", "STAGE", "STAGE_SKILL",
            "RUN_ROOT", "IN", "OUT", "CONFIG", "PARAMS", "RUNNER"]
    keys += [k for k in ("WORKLOAD_AGENT_STATE_BASE", "WORKLOAD_AGENT_ENV_FILE",
                         "WORKLOAD_AGENT_MODEL") if k in env]
    env_sh = os.path.join(lay["out"], "env.sh")
    with open(env_sh, "w") as f:
        for k in keys:
            f.write("export %s=%s\n" % (k, shlex.quote(env[k])))

    if args.quiet:
        print(env_sh)
        return 0
    for k in keys:
        print("export %s=%s" % (k, shlex.quote(env[k])))
    print()
    print("# mode:   %s" % lay["mode"])
    print("# follow: %s" % lay["skill_md"])
    if lay["mode"] == "script":
        print("# run:    source %s && bash %s" % (env_sh, lay["run_sh"]))
    print("# after:  python3 %s validate %s" % (__file__, lay["stage"]))
    return 0


def cmd_run(cfg, args):
    lay = stage_layout(cfg, args.stage)
    rc = cmd_prepare(cfg, argparse.Namespace(stage=args.stage, quiet=True))
    if rc:
        return rc
    env = stage_env(cfg, lay)

    # An agent-mode stage is dispatched through the runner rather than refused. The
    # judgement stays the agent's; only the dispatching is mechanical, and the runner
    # contract already isolates that from any particular CLI. See agent_stage.py.
    if lay["mode"] != "script":
        import agent_stage
        started = time.time()
        print("[stage] %s / %s (agent mode via %s)"
              % (lay["stage"], lay["skill"], os.path.basename(env["RUNNER"])))
        rc, note = agent_stage.dispatch(cfg, lay, env)
        print("[stage] agent stage finished rc=%d after %.0fs" % (rc, time.time() - started))
        if rc != 0:
            # No result.json, or an unusable one: say so here rather than letting
            # validate report a bare "missing file" with no history.
            status = "blocked" if rc == 78 else "fail"
            if not os.path.exists(os.path.join(lay["out"], "result.json")):
                with open(os.path.join(lay["out"], "result.json"), "w") as f:
                    json.dump({"stage": lay["stage"], "skill": lay["skill"],
                               "status": status, "artifacts": {}, "metrics": {},
                               "notes": note or "the agent session produced no result.json"},
                              f, indent=2, ensure_ascii=False)
        return cmd_validate(cfg, args, exec_rc=rc)

    log = os.path.join(lay["out"], "stage.log")
    started = time.time()
    print("[stage] %s / %s -> %s" % (lay["stage"], lay["skill"], log))
    with open(log, "w") as lf:
        proc = subprocess.Popen(["bash", lay["run_sh"]], cwd=lay["skill_root"],
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()
        proc.wait()
    print("[stage] run.sh exited %d after %.1fs" % (proc.returncode, time.time() - started))
    return cmd_validate(cfg, args, exec_rc=proc.returncode)


def cmd_validate(cfg, args, exec_rc=None):
    lay = stage_layout(cfg, args.stage)
    res, errs = validate_result(lay)
    state = load_state(cfg)
    entry = {
        "skill": lay["skill"],
        "mode": lay["mode"],
        "out": lay["out"],
        "updated": int(time.time()),
    }
    if exec_rc is not None:
        entry["exec_rc"] = exec_rc

    # The target gate. A stage that ran against a different stack than the config names
    # invalidates the labels on everything downstream, so it fails here rather than
    # surfacing as a confusing report later.
    errs = list(errs) + target_problems(cfg)
    observed = target.observed_identity(lay["out"], res)
    rep = target.identity_report(cfg, observed)
    entry["target_identity"] = rep
    errs += target.identity_problems(rep)
    # The run-level verdict aggregates the stages: most stages observe nothing about the
    # stack and report `unverified`, which must not erase the corroboration from the one
    # stage that did look.
    per_stage = dict((state.get("target") or {}).get("stages") or {})
    per_stage[lay["stage"]] = rep["verdict"]
    vs = set(per_stage.values())
    state["target"] = dict(target.declared(cfg), stages=per_stage,
                           verdict=("mismatch" if "mismatch" in vs
                                    else "ok" if "ok" in vs else "unverified"))

    if errs:
        entry["status"] = "fail"
        entry["errors"] = errs
        state["stages"][lay["stage"]] = entry
        save_state(cfg, state)
        print("stage %s failed validation:" % lay["stage"], file=sys.stderr)
        for e in errs:
            print("  - %s" % e, file=sys.stderr)
        return 1

    entry["status"] = res["status"]
    entry["metrics"] = res.get("metrics") or {}
    entry["artifacts"] = {k: os.path.join(lay["out"], v)
                          for k, v in (res.get("artifacts") or {}).items()}
    if res.get("notes"):
        entry["notes"] = res["notes"]
    state["stages"][lay["stage"]] = entry
    save_state(cfg, state)

    print("stage %s: %s (%s)" % (lay["stage"], res["status"], lay["skill"]))
    print("  stack identity: %s%s" % (rep["verdict"],
                                      (" — " + "; ".join(rep["matched"])) if rep["matched"] else ""))
    for k, v in (res.get("metrics") or {}).items():
        print("  %s = %s" % (k, v))
    if res.get("notes"):
        print("  notes: %s" % res["notes"])
    # blocked is not a crash, but downstream must not continue, so it too exits non-zero.
    return 0 if res["status"] == "ok" else 1


def cmd_all(cfg, args):
    """Every stage, in order, unattended -- the whole campaign from one command.

    Resume is the default and is not a special mode: a stage already recorded `ok` in
    STATE.json is skipped, so re-issuing this command after any interruption continues
    where it stopped. That is the same property `next` has always had, with the loop
    closed around it.

    It stops at the first stage that is not `ok`, and that is deliberate. `blocked` means
    a precondition was missing; carrying on would optimize kernels nobody profiled, or
    A/B a stack nothing was applied to. The run stops with the reason on screen and in
    STATE.json, and resumes with the same command once the precondition is met.

    It reports while it runs, by default: scripts/watch.py is started alongside and
    writes PROGRESS.md, progress.json and a refreshed REPORT.md every half hour. A day
    and a half of unattended GPU time that reports only at the end is how a stage that
    went wrong in its first ten minutes gets to spend the rest of its window going
    further wrong. `--no-report` (`--report-interval N` to change the cadence), or
    `report.enabled: false` in the config, turns it off.
    """
    order = list(cfg["order"])
    only = [s for s in (args.stages or []) if s]
    if only:
        bad = [s for s in only if s not in order]
        if bad:
            print("no such stage(s): %s (order is %s)" % (", ".join(bad), ", ".join(order)),
                  file=sys.stderr)
            return 2
        order = [s for s in order if s in only]

    state = load_state(cfg)
    print("run_root %s" % cfg["run_root"])
    print("stages   %s" % " -> ".join(order))
    reporter = start_reporter(cfg, args)
    try:
        return run_order(cfg, args, order, state)
    finally:
        stop_reporter(reporter)


def report_settings(cfg):
    """The `report` block, with the older `watch` key overlaid on top of it.

    `watch` was this block's name until reporting became the default and intervening
    became opt-in. A scenario config still using the old name has to keep working, and a
    plain `cfg.get("report") or cfg.get("watch")` would not do it: `default.json` always
    ships a `report` block, so the user's `watch` would sit beside it, ignored, with no
    symptom beyond a cadence quietly back at its default.
    """
    out = dict(cfg.get("report") or {})
    out.update(cfg.get("watch") or {})
    return out


def start_reporter(cfg, args):
    """The periodic reporter alongside the run. Its absence must never stop the campaign.

    On by default: an unattended run that says nothing until it finishes is one nobody
    can act on. Turning it off is a choice the caller makes (`--no-report`, or
    `report.enabled: false`), not something a missing config key does by accident --
    hence `is False` rather than a truthiness test.
    """
    rep = report_settings(cfg)
    if getattr(args, "no_report", False) or rep.get("enabled") is False:
        return None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch.py")
    run_root = resolve_path(cfg["run_root"])
    os.makedirs(run_root, exist_ok=True)
    cmd = [sys.executable, script, "--driver-pid", str(os.getpid())]
    for c in (getattr(args, "config", None) or []):
        cmd += ["--config", c]
    for s in (getattr(args, "set", None) or []):
        cmd += ["--set", s]
    if getattr(args, "run_root", None):
        cmd += ["--run-root", args.run_root]
    if getattr(args, "report_interval", None):
        cmd += ["--interval", str(args.report_interval)]
    log = os.path.join(run_root, "watch.log")
    try:
        lf = open(log, "a")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    except OSError as exc:
        print("[stage] could not start the reporter (%s); the run continues unreported"
              % exc, file=sys.stderr)
        return None
    print("[stage] reporting every %ss into %s/PROGRESS.md and REPORT.md (log %s; "
          "--no-report turns it off)"
          % (getattr(args, "report_interval", None) or rep.get("interval_sec") or 1800,
             os.path.basename(run_root), log))
    return proc


def stop_reporter(proc):
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_order(cfg, args, order, state):
    for stage in order:
        done = state["stages"].get(stage, {}).get("status") == "ok"
        if done and not args.force:
            print("\n=== %s: already ok, skipping (--force to re-run)" % stage)
            continue
        print("\n=== %s" % stage)
        sys.stdout.flush()
        rc = cmd_run(cfg, argparse.Namespace(stage=stage))
        state = load_state(cfg)
        if rc != 0:
            entry = state["stages"].get(stage) or {}
            print("\nstopped at %s (%s). %s"
                  % (stage, entry.get("status", "fail"),
                     entry.get("notes") or "; ".join(entry.get("errors") or []) or ""),
                  file=sys.stderr)
            print("Fix the cause and re-issue the same command; completed stages are "
                  "skipped.", file=sys.stderr)
            return rc
    print("\nall requested stages are ok.")
    return 0


def cmd_status(cfg, args):
    state = load_state(cfg)
    print("run_root %s" % state.get("run_root"))
    for stage in cfg["order"]:
        e = state["stages"].get(stage)
        if not e:
            print("%-10s -" % stage)
            continue
        line = "%-10s %-8s %s" % (stage, e.get("status"), e.get("skill"))
        print(line)
        for k, v in (e.get("metrics") or {}).items():
            print("           %s = %s" % (k, v))
        for err in e.get("errors") or []:
            print("           ! %s" % err)
    return 0


def cmd_next(cfg, args):
    state = load_state(cfg)
    for stage in cfg["order"]:
        if state["stages"].get(stage, {}).get("status") != "ok":
            print(stage)
            return 0
    print("")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", help="layer in a config, repeatable")
    ap.add_argument("--set", action="append", help="a.b=value point override, repeatable")
    ap.add_argument("--run-root", help="override run_root")
    ap.add_argument("--dry-run", action="store_true", help="equivalent to the plan subcommand")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("plan")
    p = sub.add_parser("prepare"); p.add_argument("stage"); p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("run"); p.add_argument("stage")
    p = sub.add_parser("all")
    p.add_argument("stages", nargs="*", help="limit to these stages (default: all)")
    p.add_argument("--force", action="store_true", help="re-run stages already recorded ok")
    # --no-watch is the previous spelling, kept because the flag appears in launch
    # scripts and in campaign notes; both names set the same dest.
    p.add_argument("--no-report", "--no-watch", dest="no_report", action="store_true",
                   help="do not report periodically (scripts/watch.py is not started)")
    p.add_argument("--report-interval", type=int, default=None, metavar="SEC",
                   help="seconds between progress reports (default from config: 1800)")
    p = sub.add_parser("validate"); p.add_argument("stage")
    sub.add_parser("status")
    sub.add_parser("next")

    args = ap.parse_args(argv)
    if args.dry_run or not args.cmd:
        args.cmd = "plan"
    cfg = load_config(args)

    return {
        "plan": cmd_plan,
        "prepare": cmd_prepare,
        "run": cmd_run,
        "all": cmd_all,
        "validate": cmd_validate,
        "status": cmd_status,
        "next": cmd_next,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
