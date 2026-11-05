THIS STAGE REPORTED `{{STATUS}}` (attempt {{PREV_ATTEMPT}}) AND THERE IS BUDGET LEFT, so
you get one more session. What it said was missing:

    {{NOTES}}

Do one of two things and nothing else:

  * If that precondition is something you can satisfy here -- a file to build, a path to
    locate, a container to start, a step of SKILL.md that was skipped -- satisfy it and
    finish the stage.
  * If it genuinely cannot be satisfied on this machine, report `blocked` again and make
    the note name exactly what is absent and the command that shows it absent. A vague or
    empty note is the one unacceptable outcome: nobody, human or agent, can act on
    "  is missing".

Do not reinterpret the stage to get around the blocker, and do not start over -- what is
already in $OUT was expensive:
