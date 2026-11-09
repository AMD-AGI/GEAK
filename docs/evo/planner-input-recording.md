# Planner input recording

For prospective input audits, set the lane argument `record_planner_inputs:true`.
It records each initial plan and forced replan in
`EVAL_DIR/planner_rN_<attempt>_inputs.json`, alongside the existing proposed-plan
artifact. It defaults to off.

The snapshot's `inputs` field contains the argument object passed to `roleAgent`.
Its `role_prompt` contains the complete string returned by `roleAgent`, including
the injected `TASK` that may be absent from that argument object. The snapshot
also contains the response schema and supplied `prepared_original_ref` value.
It does not qualify that reference or override preparation, planning or scoring.
It is serialized before
the call, so later history changes and a failed planner return do not erase the
observation. Existing files are not overwritten, and recording failure is logged
without changing search decisions. The optional write consumes ordinary wall time.

The boundary is before `agentT`/backend prompt additions and excludes subsequent
tool reads. It does not establish identical complete model context or prove that
the model used any supplied evidence. The artifact describes a logical plan call;
retries within `agentT` or the backend use that same supplied role prompt and are
not separate input artifacts. Preserve raw artifacts when comparing paths, clocks
and substantive fields; a normalized comparison is not an actual-prompt hash.

This document is not injected into role prompts. Existing prepared-reference
qualification and local native validation obligations continue to apply.
