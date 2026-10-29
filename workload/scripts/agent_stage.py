#!/usr/bin/env python3
"""Run an agent-mode stage unattended, through the runner.

Three of the five slots hold skills that are procedures rather than scripts: choosing a
kernel boundary, deciding which dispatch branch is one operator, reading a roofline and
saying what it implies. Those are judgement, and the contract (stages/CONTRACT.md §2 B)
says so by letting a skill ship a `SKILL.md` and no `run.sh`.

Until now that meant a human-attended session: `stage.py run profile` printed "this is an
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

CONTRACT = """You are running one stage of an unattended pipeline. No one will answer a
question, and nothing after this message is interactive: decide, act, and record what you
did. Work until the stage is finished or genuinely blocked.

THE CONTRACT (stages/CONTRACT.md), which is the whole of what this stage owes anybody:

  1. Read $IN (the previous stage's output directory) and $PARAMS (this stage's
     parameters). Neither is advisory. Where $PARAMS and the skill's SKILL.md disagree --
     a selection cutoff, a count, a threshold -- $PARAMS wins: it is this run's executable
     policy and the value in SKILL.md is the default it replaces. The slot checks
     afterwards that what ran is what was configured, so a stage that quietly used the
     skill's default fails even if everything it did was otherwise correct.
  2. Do the work described in the skill's SKILL.md, below. It is the procedure; follow it
     rather than improvising an equivalent.
  3. Write every artifact into $OUT.
  4. Finish by writing $OUT/result.json:
       {"stage": "<stage>", "skill": "<skill>", "status": "ok|blocked|fail",
        "artifacts": {"name": "relative/path"}, "metrics": {...}, "notes": "..."}
     Use $WORKLOAD_ROOT/scripts/emit_result.py to write it -- it validates the shape and
     refuses a malformed one.

status means exactly this:
  ok      -- the work described in SKILL.md was done and its artifacts are in $OUT.
  blocked -- a precondition was missing (no GPU, no server, no upstream artifact). Say in
             `notes` what was missing and what would unblock it. This is a legitimate,
             useful outcome; it is not a failure to admit one.
  fail    -- it was attempted and did not succeed.

Never report `ok` for work that was not done, and never report a number that was not
measured. A stage that reports a plausible figure nobody computed is the one outcome this
pipeline cannot recover from: everything downstream inherits it and nothing says so.

Write result.json even so -- but know that where the slot has a `finalize.py`, it runs
after this session, reads $OUT itself and has the last word. It is not a second opinion to
argue with: it is the stage's acceptance criterion, and it is in
$WORKLOAD_ROOT/stages/<stage>/finalize.py and README.md where you can read exactly what it
will require before you decide you are finished.
"""


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


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


def build_prompt(env, lay, params, attempt, why=None, repair=None):
    out = env["OUT"]
    lines = [CONTRACT, "", "ENVIRONMENT (already exported in your shell; use these paths "
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

    if attempt > 1:
        rows, truncated = inventory(out)
        if repair:
            lines += ["THIS STAGE REPORTED `%s` (attempt %d) AND THERE IS BUDGET LEFT, so "
                      "you get one more session. What it said was missing:"
                      % (repair.get("status"), attempt - 1), "",
                      "    " + (str(repair.get("notes") or "").strip()
                                or "(nothing -- the note was empty, which is itself the "
                                   "defect described below)"), "",
                      "Do one of two things and nothing else:", "",
                      "  * If that precondition is something you can satisfy here -- a "
                      "file to build, a path to locate, a container to start, a step of "
                      "SKILL.md that was skipped -- satisfy it and finish the stage.",
                      "  * If it genuinely cannot be satisfied on this machine, report "
                      "`blocked` again and make the note name exactly what is absent and "
                      "the command that shows it absent. A vague or empty note is the one "
                      "unacceptable outcome: nobody, human or agent, can act on "
                      "\"  is missing\".", "",
                      "Do not reinterpret the stage to get around the blocker, and do not "
                      "start over -- what is already in $OUT was expensive:", ""]
        else:
            lines += ["THIS IS A CONTINUATION (attempt %d). Earlier sessions left "
                      "artifacts%s." % (attempt, (": " + why) if why else ""),
                      "",
                      "Do not start over. These artifacts are already on disk and were "
                      "expensive to produce; read them, work out which step of SKILL.md "
                      "they correspond to, and carry on from there:", ""]
        for rel, size in rows:
            lines.append("  %10d  %s" % (size, rel))
        if truncated:
            lines.append("  ... (listing truncated; walk $OUT yourself for the rest)")
        if not rows:
            lines.append("  (nothing -- $OUT is empty, so begin at the start of SKILL.md)")
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
    """Let the slot judge the session. Returns True when slot finalization is required.

    The skill is an external archive that knows nothing about `result.json`, so the
    mapping -- and the acceptance criterion with it -- lives in the slot (PRINCIPLES R6).
    Running it here is what makes that true in practice: left to write its own verdict, a
    profile session once reported `ok` for a delivery of four tasks out of six declared,
    with single-task fields in `metrics`, and nothing downstream could tell.

    `stages/<stage>/finalize.py` reads `$OUT` and overwrites `result.json` last. A stage
    whose slot has no finalize keeps the agent's own result, unchanged -- this is the
    contract's "is the file there", not a plugin system.
    """
    script = os.path.join(env.get("WORKLOAD_ROOT", ""), "stages", env.get("STAGE", ""),
                          "finalize.py")
    if not os.path.exists(script):
        return False
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
    try:
        rc = subprocess.call([sys.executable, script], env=env)
    except OSError as exc:
        status, problem = "blocked", "the slot finalizer could not run: %s" % exc
    else:
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
    return True


def dispatch(cfg, lay, env, log_cb=print):
    """Run the stage's agent sessions until result.json validates or the budget is spent.

    Returns (exit_code, note). The caller validates result.json itself -- this function
    deliberately has no opinion about whether the stage succeeded.
    """
    import stage as stage_mod                      # circular only at call time

    params = lay["params"]
    out, runner = env["OUT"], env["RUNNER"]
    budget = int(params.get("agent_budget_sec") or cfg.get("agent_budget_sec") or 14400)
    attempts = int(params.get("agent_attempts") or cfg.get("agent_attempts") or 3)

    if not os.path.exists(runner):
        return 78, "runner %s does not exist" % runner

    repairs = int(params.get("agent_repair_attempts")
                  or cfg.get("agent_repair_attempts") or 1)
    result_path = os.path.join(out, "result.json")
    deadline = time.time() + budget
    why, repair, recorded = None, None, None
    # Session numbers survive a new stage.py invocation, so prompts/logs and the
    # continuation inventory survive node or controller recovery too.
    offset = max([int(match.group(1)) for name in os.listdir(out)
                  if (match := re.fullmatch(r'agent_prompt\.(\d+)\.txt', name))] or [0])
    for local_attempt in range(1, attempts + 1):
        attempt = offset + local_attempt
        left = int(deadline - time.time())
        if left <= 60:
            return 1, ("the %ds agent budget is spent after %d attempt(s); last problem: %s"
                       % (budget, local_attempt - 1, why or "none recorded"))

        if os.path.lexists(result_path):
            previous, _ = stage_mod.validate_result(lay)
            saved = os.path.join(out, 'result.before-session.%d.json' % attempt)
            os.replace(result_path, saved)
            if previous and previous.get('status') in ('blocked', 'fail'):
                recorded = saved

        prompt_path = os.path.join(out, "agent_prompt.%d.txt" % attempt)
        with open(prompt_path, "w") as f:
            f.write(build_prompt(env, lay, params, attempt, why, repair))
        log = os.path.join(out, "agent.%d.log" % attempt)

        log_cb("[agent] attempt %d/%d  budget %ds left  prompt %s"
               % (attempt, offset + attempts, left, prompt_path))
        log_cb("[agent] log %s" % log)
        started = time.time()
        # The budget is enforced from outside the runner, as the contract requires: a
        # runner that imposed its own would fight it.
        rc = subprocess.call(["timeout", str(left), "bash", runner, prompt_path,
                              env["GEAK_ROOT"], log], env=env)
        took = time.time() - started
        log_cb("[agent] attempt %d exited %d after %.0fs" % (attempt, rc, took))

        if rc == 78:
            tail = "\n".join(read(log).splitlines()[-5:])
            return 78, ("the runner reported an unmet precondition (78), which the "
                        "contract says must not be retried: %s" % tail.strip())

        # Whether the agent itself reported, before the slot overwrites the file.
        reported = os.path.exists(result_path)
        judged = finalize(env, log_cb)

        res, errs = stage_mod.validate_result(lay)
        if not errs:
            status = (res or {}).get("status")
            if (status == "ok" or repairs <= 0 or local_attempt == attempts
                    or int(deadline - time.time()) <= 300):
                return 0, None
            if judged and not reported:
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
        return 0, None
    return 1, ("%d agent attempts ended without a valid result.json; last problem: %s"
               % (attempts, why))
