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
import subprocess
import sys
import time

CONTRACT = """You are running one stage of an unattended pipeline. No one will answer a
question, and nothing after this message is interactive: decide, act, and record what you
did. Work until the stage is finished or genuinely blocked.

THE CONTRACT (stages/CONTRACT.md), which is the whole of what this stage owes anybody:

  1. Read $IN (the previous stage's output directory) and $PARAMS (this stage's
     parameters). Neither is advisory.
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


def build_prompt(env, lay, params, attempt, why=None):
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
        lines += ["THIS IS A CONTINUATION (attempt %d). The previous session ended "
                  "without a valid %s/result.json%s." % (attempt, out,
                                                         (": " + why) if why else ""),
                  "",
                  "Do not start over. These artifacts are already on disk and were "
                  "expensive to produce; read them, work out which step of SKILL.md they "
                  "correspond to, and carry on from there:", ""]
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
    return "\n".join(lines)


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

    deadline = time.time() + budget
    why = None
    for attempt in range(1, attempts + 1):
        left = int(deadline - time.time())
        if left <= 60:
            return 1, ("the %ds agent budget is spent after %d attempt(s); last problem: %s"
                       % (budget, attempt - 1, why or "none recorded"))

        prompt_path = os.path.join(out, "agent_prompt.%d.txt" % attempt)
        with open(prompt_path, "w") as f:
            f.write(build_prompt(env, lay, params, attempt, why))
        log = os.path.join(out, "agent.%d.log" % attempt)

        log_cb("[agent] attempt %d/%d  budget %ds left  prompt %s"
               % (attempt, attempts, left, prompt_path))
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

        res, errs = stage_mod.validate_result(lay)
        if not errs:
            return 0, None
        why = "; ".join(errs)
        log_cb("[agent] result.json not yet valid: %s" % why)
        # An agent that exhausted its own window mid-work is the normal case for the long
        # stages; the next attempt resumes from the artifacts it left behind.

    return 1, ("%d agent attempts ended without a valid result.json; last problem: %s"
               % (attempts, why))
