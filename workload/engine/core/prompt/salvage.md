THIS IS A SALVAGE SESSION ({{SESSION}} of at most {{SESSIONS}}, {{BUDGET}}s left in the
salvage budget). The stage's normal agent budget is spent and the delivery did not finish:

    {{TRIGGER}}

The campaign is configured (`on_budget_exhausted`) to continue with a PARTIAL delivery
rather than stop. Your job in this session is to deliver what already qualifies and to
account for everything else -- not to finish the original stage. Where the instructions
below and SKILL.md disagree, the instructions below hold for this session.

  * Do not start new long work: no new servers, captures, traces or GPU qualification
    runs. What is already on disk is what this delivery is made of.
  * Never drop work silently. Everything the stage was asked for and does not deliver is
    recorded with a stated blocker, so the partial delivery cannot be mistaken for a full
    one.
  * Finish by running the slot's finalize (path in THE SLOT below). It recognises a
    salvage by {{MARKER}}, judges the delivery as partial and has the last word. Do not
    edit or delete that marker: it is what keeps a restarted campaign in salvage instead
    of reopening the normal budget.

What the previous salvage session's finalize said (fix exactly this, if anything):

    {{LAST}}

THE SLOT'S SALVAGE INSTRUCTIONS (what "already qualifies" means for this stage):

{{SLOT_SALVAGE}}

Already on disk in $OUT:
