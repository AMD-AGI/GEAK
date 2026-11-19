#!/usr/bin/env python3
"""Run an agent-mode stage unattended, through the runner.

Three of the five slots hold skills that are procedures rather than scripts: choosing a
kernel boundary, deciding which dispatch branch is one operator, reading a roofline and
saying what it implies. Those are judgement, and the contract (stages/CONTRACT.md §2 B)
says so by letting a skill ship a `SKILL.md` and no `run.sh`.

Until now that meant a human-attended session: `run.py run profile` printed "this is an
agent-mode stage" and stopped. The judgement is still the agent's, but *dispatching* it is
not judgement -- it is the same four steps every time, and the runner contract already
isolates "start one unattended agent session" from any particular CLI. So this module
does for an agent-mode stage exactly what `run.sh` does for a script-mode one, and the
pipeline runs end to end from one command.

What it refuses to do is decide whether the stage succeeded. The session ends, and then
the stage's own `result.json` is validated exactly as a script stage's would be. An agent
that talked at length and wrote nothing is a failed stage, not a passed one.

Two properties worth stating, because both are failure modes that have cost real time:

  * **A session that dies is resumed, not restarted.** The skills write their artifacts
    incrementally -- a census, then a ranking, then task directories. A restart from zero
    throws away hours of GPU profiling that is already on disk. The continuation prompt
    names what is already there and what is missing, so the agent continues rather than
    recomputes.
  * **`78` is not a retry.** The runner contract reserves it for an unmet precondition --
    no binary, no credentials, no sidecar. Retrying that ten thousand times produces the
    same result, so it stops immediately and reports `blocked` with the reason.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

# Everything the agent is told, other than the skill's own SKILL.md, is in prompt/. It is
# text written to be read and re-read by whoever is tuning how these sessions behave, and
# that person does not have to be someone who edits Python. Substitution is a literal
# `{{NAME}}` replace and nothing more: a template engine here would be a framework, and
# PRINCIPLES R2 says this pipeline does not have one.
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")

# Opt-in salvage (params.on_budget_exhausted="deliver_qualified"). When the normal
# sessions run out of budget or attempts without an `ok`, a bounded phase asks the agent
# to deliver only what already qualifies and to record the rest as blocked. The engine
# owns only the mechanism -- a fresh budget, the marker, the prompt frame; what "already
# qualifies" means is the slot's, in `stages/<stage>/salvage.md`, and its finalize.py
# judges the partial delivery it produces. The marker in $OUT is what makes a re-issued
# `run.py all` resume the salvage rather than open a fresh normal budget.
SALVAGE_MARKER = "salvage.json"
SALVAGE_SCHEMA = "workload/agent_salvage/v1"
SALVAGE_MODES = ("stop", "deliver_qualified")
SALVAGE_DEFAULT_BUDGET_SEC = 7200
SALVAGE_DEFAULT_ATTEMPTS = 3


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def prompt_text(name, **fields):
    """One of the prompt/*.md files, with every {{NAME}} replaced.

    A missing file is a defect in this directory, not something to paper over with a
    silent default: the session would run without the instruction that makes its output
    usable, and nothing downstream would say so.
    """
    path = os.path.join(PROMPT_DIR, name)
    try:
        with open(path) as f:
            text = f.read()
    except OSError as exc:
        raise SystemExit("the prompt text %s is missing or unreadable (%s); an agent "
                         "session without it is not worth starting" % (path, exc))
    for key, value in fields.items():
        text = text.replace("{{%s}}" % key, str(value))
    return text.rstrip("\n")


def inventory(out_dir, limit=60):
    """What is already on disk -- so a resumed session continues instead of recomputing."""
    rows = []
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            p = os.path.join(root, fn)
            try:
                rows.append((os.path.relpath(p, out_dir), os.path.getsize(p)))
            except OSError:
                pass
            if len(rows) >= limit:
                return rows, True
    return rows, False


def _positive_int(params, key, default):
    value = params.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer, got %r" % (key, value))
    return value


def salvage_policy(lay):
    """The stage's salvage settings, or None when it stops at an exhausted budget.

    Raises ValueError for a setting that cannot be honoured, so `run.py plan` refuses it
    before any GPU time rather than when the budget has already run out. The budget and
    attempts are read even when the mode is `stop`: a stage whose marker says it already
    entered salvage resumes it whatever the mode now says (see dispatch)."""
    params = lay.get("params") or {}
    mode = params.get("on_budget_exhausted", "stop")
    if mode is None:
        mode = "stop"
    if mode not in SALVAGE_MODES:
        raise ValueError("on_budget_exhausted must be one of %s, got %r"
                         % (list(SALVAGE_MODES), mode))
    policy = {"mode": mode,
              "budget": _positive_int(params, "salvage_budget_sec",
                                      SALVAGE_DEFAULT_BUDGET_SEC),
              "attempts": _positive_int(params, "salvage_attempts",
                                        SALVAGE_DEFAULT_ATTEMPTS),
              "instructions": os.path.join(lay.get("slot_root") or "", "salvage.md")}
    if mode != "stop" and not os.path.isfile(policy["instructions"]):
        raise ValueError("on_budget_exhausted=%s but the slot has no salvage.md (%s); "
                         "what may be delivered from a spent budget is the slot's to say"
                         % (mode, policy["instructions"]))
    return policy


def build_prompt(env, lay, params, attempt, why=None, repair=None, salvage=None):
    out = env["OUT"]
    lines = [prompt_text("contract.md"), "", "",
             "ENVIRONMENT (already exported in your shell; use these paths "
                         "verbatim rather than retyping them):", ""]
    for k in ("GEAK_ROOT", "WORKLOAD_ROOT", "SKILL_ROOT", "STAGE", "STAGE_SKILL",
              "RUN_ROOT", "IN", "OUT", "CONFIG", "PARAMS"):
        if k in env:
            lines.append("  %-13s %s" % (k, env[k]))

    lines += ["", "PARAMETERS ($PARAMS, reproduced here so there is no doubt what was "
                  "asked):", "", json.dumps(params, indent=2, ensure_ascii=False), ""]

    brief = params.get("agent_brief")
    if brief:
        lines += ["CAMPAIGN NOTES from the scenario config. These are facts about this "
                  "particular run that the skill cannot know:", "", str(brief), ""]

    if salvage:
        rows, truncated = inventory(out)
        lines += [prompt_text(
            "salvage.md",
            SESSION=salvage["session"],
            SESSIONS=salvage["attempts"],
            BUDGET=salvage["left"],
            MARKER=os.path.join(out, SALVAGE_MARKER),
            TRIGGER=str(salvage.get("trigger") or "").strip() or "(not recorded)",
            LAST=(str(why or "").strip() or "(no salvage session has run yet)"),
            SLOT_SALVAGE=read(salvage["instructions"]).strip()), ""]
    elif attempt > 1:
        rows, truncated = inventory(out)
        if repair:
            lines += [prompt_text(
                "repair.md",
                STATUS=repair.get("status"),
                PREV_ATTEMPT=attempt - 1,
                NOTES=(str(repair.get("notes") or "").strip()
                       or "(nothing -- the note was empty, which is itself the defect "
                          "described below)")), ""]
        else:
            lines += [prompt_text(
                "continuation.md",
                ATTEMPT=attempt,
                WHY=(": " + why) if why else ""), ""]
    if salvage or attempt > 1:
        for rel, size in rows:
            lines.append("  %10d  %s" % (size, rel))
        if truncated:
            lines.append("  ... (listing truncated; walk $OUT yourself for the rest)")
        if not rows:
            lines.append("  (nothing -- $OUT is empty, so there is nothing to salvage; "
                         "run the finalize and let it record that)" if salvage else
                         "  (nothing -- $OUT is empty, so begin at the start of SKILL.md)")
        lines.append("")

    lines += ["=" * 78,
              "THE SKILL -- %s" % lay["skill_md"],
              "(this file is on disk at the path above; re-read it there whenever you "
              "need the parts not quoted here, including anything it references under "
              "$SKILL_ROOT)",
              "=" * 78, "", read(lay["skill_md"], "(SKILL.md is missing)")]

    slot = os.path.join(env.get("WORKLOAD_ROOT", ""), "stages",
                        env.get("STAGE", ""), "README.md")
    if os.path.exists(slot):
        lines += ["", "=" * 78,
                  "THE SLOT -- %s" % slot,
                  "(what this stage owes the next one, how its artifacts map to "
                   "result.json, and what its verdicts mean. The slot outlives any "
                   "one implementation: where it and SKILL.md overlap, the slot "
                   "holds.)",
                  "=" * 78, "", read(slot, "")]
    return "\n".join(lines)


def finalize(env, log_cb=print):
    """Let the slot judge the session.

    The skill is an external archive that knows nothing about `result.json`, so the
    mapping -- and the acceptance criterion with it -- lives in the slot (PRINCIPLES R6).
    Running it here is what makes that true in practice: left to write its own verdict, a
    profile session once reported `ok` for a delivery of four tasks out of six declared,
    with single-task fields in `metrics`, and nothing downstream could tell.

    `stages/<stage>/finalize.py` reads `$OUT` and overwrites `result.json` last. Every
    slot has one and it runs unconditionally: calling it only where the file happened to
    exist is how three of the five slots ended up grading their own work, two of them
    under a different filename. `stages/CONTRACT.md` section 3.1 is the rule; a missing
    one is a defect in the slot, reported here and caught before the run by
    `run.py --dry-run`.
    """
    # engine/core/agent_stage.py -> engine/core -> engine -> workload
    workload_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stages_root = os.path.join(workload_root, "stages")
    script = os.path.join(stages_root, env.get("STAGE", ""), "finalize.py")
    # The envelope writer belongs to the stage contract, not to the plumbing, so it lives
    # with the contract in stages/_shared (stages/CONTRACT.md 3). The dependency points
    # that way round on purpose: the pipeline implements the contract, and a slot that
    # reached back up here for a tool would couple a replaceable skill to this file.
    sys.path.insert(0, os.path.join(stages_root, "_shared"))
    from emit_result import emit, read_json

    result_path = os.path.join(env["OUT"], "result.json")

    def archive(label):
        if os.path.lexists(result_path):
            os.replace(result_path, os.path.join(
                env["OUT"], "result.%s.%d.json" % (label, time.time_ns())))

    # The slot must produce its own receipt. Merely comparing mtimes can bless an
    # agent's old `ok` when a finalizer crashes before replacing it, especially on NFS.
    archive("before-finalize")
    log_cb("[agent] the slot's finalize has the last word: %s" % script)
    status, problem = "fail", None
    rc = None
    if not os.path.exists(script):
        status, problem = "blocked", ("the slot has no finalize.py, so nothing but the "
                                      "agent judged this stage (stages/CONTRACT.md 3.1)")
    else:
        try:
            rc = subprocess.call([sys.executable, script], env=env)
        except OSError as exc:
            status, problem = "blocked", "the slot finalizer could not run: %s" % exc
    if rc is not None:
        log_cb("[agent] finalize exited %d" % rc)
        receipt = read_json(result_path)
        if rc == 2:
            # emit_result reserves 2 for a missing environment or unusable receipt.
            status, problem = "blocked", "the slot finalizer refused to judge (exit 2)"
        elif rc not in (0, 1):
            problem = "the slot finalizer failed with exit %d" % rc
        elif not os.path.exists(result_path):
            problem = "the slot finalizer exited %d without a fresh result.json" % rc
        elif rc != 0 and isinstance(receipt, dict) and receipt.get("status") == "ok":
            problem = "the slot finalizer reported ok but failed with exit %d" % rc
    if problem:
        archive("rejected-finalize")
        if emit(status, notes=problem + "; repair the finalizer and resume this stage. "
                "Earlier session receipts are preserved.", out=env["OUT"],
                stage=env.get("STAGE"), skill=env.get("STAGE_SKILL")) == 2:
            raise RuntimeError(problem + "; no valid stage environment to record it")


def _last_session(out):
    """The highest session number already used in $OUT (0 when none)."""
    return max([int(match.group(1)) for name in os.listdir(out)
                if (match := re.fullmatch(r'agent_prompt\.(\d+)\.txt', name))] or [0])


def _run_session(env, attempt, prompt, left, log_cb, label):
    """Write the prompt, run one runner session under `timeout`, return (rc, seconds)."""
    out = env["OUT"]
    prompt_path = os.path.join(out, "agent_prompt.%d.txt" % attempt)
    with open(prompt_path, "w") as f:
        f.write(prompt)
    log = os.path.join(out, "agent.%d.log" % attempt)

    log_cb("[agent] %s  budget %ds left  prompt %s" % (label, left, prompt_path))
    log_cb("[agent] log %s" % log)
    started = time.time()
    # The budget is enforced from outside the runner, as the contract requires: a
    # runner that imposed its own would fight it.
    rc = subprocess.call(["timeout", str(left), "bash", env["RUNNER"], prompt_path,
                          env["GEAK_ROOT"], log], env=env)
    took = time.time() - started
    log_cb("[agent] attempt %d exited %d after %.0fs" % (attempt, rc, took))
    return rc, took


def _precondition(out, attempt):
    tail = "\n".join(read(os.path.join(out, "agent.%d.log" % attempt)).splitlines()[-5:])
    return 78, ("the runner reported an unmet precondition (78), which the "
                "contract says must not be retried: %s" % tail.strip())


def dispatch(cfg, lay, env, log_cb=print):
    """Run the stage's agent sessions until result.json validates or the budget is spent.

    Returns (exit_code, note). The caller validates result.json itself -- this function
    deliberately has no opinion about whether the stage succeeded.

    With `on_budget_exhausted: "deliver_qualified"`, a normal budget that runs out
    without an `ok` is followed by one bounded salvage phase (`salvage`). Once entered,
    `$OUT/salvage.json` makes every later invocation resume that phase instead.
    """
    import stage as stage_mod                      # circular only at call time

    params = lay["params"]
    out, runner = env["OUT"], env["RUNNER"]
    budget = int(params.get("agent_budget_sec") or cfg.get("agent_budget_sec") or 14400)
    attempts = int(params.get("agent_attempts") or cfg.get("agent_attempts") or 3)

    if not os.path.exists(runner):
        return 78, "runner %s does not exist" % runner

    try:
        policy = salvage_policy(lay)
    except ValueError as exc:
        return 1, str(exc)
    marker = os.path.join(out, SALVAGE_MARKER)
    if os.path.lexists(marker):
        # Resume, never restart: the partial delivery is already under way (or done),
        # and a fresh normal budget would redo the captures the salvage exists to skip.
        log_cb("[agent] %s exists: this stage already entered salvage; resuming the "
               "salvage phase, not the normal budget (move the marker aside to rerun "
               "the full stage)" % marker)
        return salvage(lay, env, policy, None, log_cb)

    def spent(rc, note):
        """The normal budget or attempts ran out without an `ok` verdict."""
        if policy["mode"] == "stop":
            return rc, note
        res, _ = stage_mod.validate_result(lay)
        trigger = note or ("the normal agent budget (%ds, %d attempts) ended with "
                           "status %s: %s" % (budget, attempts, (res or {}).get("status"),
                                              str((res or {}).get("notes") or "").strip()))
        log_cb("[agent] budget spent; entering salvage (%s, %ds, up to %d session(s)): %s"
               % (policy["mode"], policy["budget"], policy["attempts"], trigger))
        return salvage(lay, env, policy, trigger, log_cb)

    repairs = int(params.get("agent_repair_attempts")
                  or cfg.get("agent_repair_attempts") or 1)
    result_path = os.path.join(out, "result.json")
    deadline = time.time() + budget
    why, repair, recorded = None, None, None
    # Session numbers survive a new run.py invocation, so prompts/logs and the
    # continuation inventory survive node or controller recovery too.
    offset = _last_session(out)
    for local_attempt in range(1, attempts + 1):
        attempt = offset + local_attempt
        left = int(deadline - time.time())
        if left <= 60:
            return spent(1, "the %ds agent budget is spent after %d attempt(s); last "
                            "problem: %s" % (budget, local_attempt - 1, why or "none recorded"))

        if os.path.lexists(result_path):
            previous, _ = stage_mod.validate_result(lay)
            saved = os.path.join(out, 'result.before-session.%d.json' % attempt)
            os.replace(result_path, saved)
            if previous and previous.get('status') in ('blocked', 'fail'):
                recorded = saved

        rc, _ = _run_session(env, attempt,
                             build_prompt(env, lay, params, attempt, why, repair), left,
                             log_cb, "attempt %d/%d" % (attempt, offset + attempts))
        if rc == 78:
            return _precondition(out, attempt)

        # Whether the agent itself reported, before the slot overwrites the file.
        reported = os.path.exists(result_path)
        finalize(env, log_cb)

        res, errs = stage_mod.validate_result(lay)
        if not errs:
            status = (res or {}).get("status")
            if status == "ok":
                return 0, None
            if local_attempt == attempts or int(deadline - time.time()) <= 300:
                return spent(0, None)
            if repairs <= 0:
                return 0, None
            if not reported:
                # The session died mid-work and the slot graded what it left behind. That
                # is an interrupted stage, not one that was weighed and found wanting, so
                # it resumes on the normal attempt budget instead of spending the single
                # repair session -- the long stages legitimately outlive one window.
                repair = None
                why = ("it ended without reporting, and the slot's finalize says: %s"
                       % (str(res.get("notes") or "").strip() or status))
                log_cb("[agent] session did not report; finalize says %s, continuing"
                       % status)
                continue
            # A precondition the agent could satisfy itself is not a reason to stop the
            # campaign and wait for a person. Keep the diagnosis, then spend one session
            # on it; a second `blocked` is the stage's answer and the order stops there.
            recorded = os.path.join(out, "result.%d.json" % attempt)
            shutil.copy(result_path, recorded)
            repairs -= 1
            repair, why = res, "the stage reported %s" % status
            log_cb("[agent] %s reported %s; one repair session, record kept at %s"
                   % (env.get("STAGE", "stage"), status, os.path.basename(recorded)))
            continue
        repair = None
        why = "; ".join(errs)
        if recorded and not os.path.exists(result_path):
            # A repair session that produced nothing must not erase what it was given.
            shutil.copy(recorded, result_path)
        log_cb("[agent] result.json not yet valid: %s" % why)
        # An agent that exhausted its own window mid-work is the normal case for the long
        # stages; the next attempt resumes from the artifacts it left behind.

    _, errs = stage_mod.validate_result(lay)
    if errs and recorded:
        # A repair session that produced nothing must not erase the diagnosis it was
        # given. A repair session that produced a *verdict* keeps it: that one is newer.
        shutil.copy(recorded, result_path)
        _, errs = stage_mod.validate_result(lay)
    if not errs:
        return spent(0, None)
    return spent(1, "%d agent attempts ended without a valid result.json; last problem: %s"
                    % (attempts, why))


def _save_marker(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def salvage(lay, env, policy, trigger, log_cb=print):
    """The bounded salvage phase: deliver what already qualifies, record the rest.

    Entered once, from `dispatch`, when the normal budget is spent (`trigger` says why),
    and resumed from `$OUT/salvage.json` by every later invocation (`trigger` None). Its
    budget is its own and is cumulative across invocations: the marker records every
    session's seconds, so a re-issued `run.py all` continues the same `salvage_budget_sec`
    and `salvage_attempts` rather than granting them again. A session the controller never
    saw end (node loss) is charged up to its log's last write.

    The slot's finalize judges each session exactly as in the normal loop; an `ok` ends
    the phase. When nothing is left, the slot's verdict on what is on disk stands (one
    more finalize when this invocation ran no session, so a fix made by hand is judged).
    """
    import stage as stage_mod                      # circular only at call time

    out = env["OUT"]
    marker = os.path.join(out, SALVAGE_MARKER)
    if trigger is not None:
        doc = {"schema": SALVAGE_SCHEMA, "stage": env.get("STAGE"),
               "mode": policy["mode"],
               "entered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "entered_unix": int(time.time()), "trigger": trigger,
               "normal_sessions": _last_session(out), "sessions": []}
        _save_marker(marker, doc)
    else:
        try:
            with open(marker) as f:
                doc = json.load(f)
            if not isinstance(doc, dict) or not isinstance(doc.get("sessions"), list):
                raise ValueError("not a %s object" % SALVAGE_SCHEMA)
        except (OSError, ValueError) as exc:
            return 1, ("the salvage marker %s is unreadable (%s); it is the only record "
                       "of how much salvage budget is spent, so nothing was started. "
                       "Repair it or move it aside to rerun the full stage" % (marker, exc))
    if not os.path.isfile(policy["instructions"]):
        return 1, ("this stage entered salvage (%s) but the slot has no salvage.md (%s)"
                   % (marker, policy["instructions"]))

    sessions = doc["sessions"]
    for row in sessions:
        if isinstance(row, dict) and row.get("seconds") is None:
            try:
                last = os.path.getmtime(os.path.join(out, "agent.%d.log" % row["session"]))
            except (OSError, KeyError, TypeError):
                last = row.get("started_unix") or 0
            row["seconds"] = max(0, int(last - (row.get("started_unix") or last)))
            row["exit"] = "lost (the controller did not see this session end)"
    doc.update(budget_sec=policy["budget"], attempts=policy["attempts"])
    _save_marker(marker, doc)

    params = lay["params"]
    result_path = os.path.join(out, "result.json")
    why = next((r.get("finalize_notes") for r in reversed(sessions)
                if isinstance(r, dict) and r.get("finalize_notes")), None)
    ran = False
    while True:
        used = sum(int(r.get("seconds") or 0) for r in sessions if isinstance(r, dict))
        left = int(policy["budget"] - used)
        if len(sessions) >= policy["attempts"] or left <= 60:
            break
        attempt = _last_session(out) + 1
        if os.path.lexists(result_path):
            os.replace(result_path, os.path.join(
                out, "result.before-session.%d.json" % attempt))
        prompt = build_prompt(env, lay, params, attempt, why, salvage={
            "session": len(sessions) + 1, "attempts": policy["attempts"], "left": left,
            "trigger": doc.get("trigger"), "instructions": policy["instructions"]})
        row = {"session": attempt, "started_unix": int(time.time()), "seconds": None}
        sessions.append(row)
        _save_marker(marker, doc)
        rc, took = _run_session(env, attempt, prompt, left, log_cb,
                                "salvage %d/%d (session %d)"
                                % (len(sessions), policy["attempts"], attempt))
        ran = True
        row.update(seconds=int(round(took)), exit=rc)
        _save_marker(marker, doc)
        if rc == 78:
            return _precondition(out, attempt)
        finalize(env, log_cb)
        res, errs = stage_mod.validate_result(lay)
        row["status"] = (res or {}).get("status") if not errs else "invalid"
        why = ("; ".join(errs) if errs
               else str((res or {}).get("notes") or "").strip() or row["status"])
        row["finalize_notes"] = why
        _save_marker(marker, doc)
        if not errs and row["status"] == "ok":
            log_cb("[agent] salvage delivered: the slot's finalize accepts the partial "
                   "delivery")
            return 0, None
        log_cb("[agent] salvage session %d: finalize says %s" % (attempt, row["status"]))

    if not ran:
        finalize(env, log_cb)
    res, errs = stage_mod.validate_result(lay)
    if (res or {}).get("status") == "ok" and not errs:
        return 0, None
    log_cb("[agent] the salvage budget is spent (%d of %d session(s), %ds of %ds)"
           % (len(sessions), policy["attempts"],
              sum(int(r.get("seconds") or 0) for r in sessions if isinstance(r, dict)),
              policy["budget"]))
    if not errs:
        return 0, None
    return 1, ("the salvage phase (%ds, %d session(s)) ended without a valid result.json; "
               "last problem: %s" % (policy["budget"], policy["attempts"],
                                     "; ".join(errs)))
