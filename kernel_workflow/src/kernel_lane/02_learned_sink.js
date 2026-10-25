// ---------------------------------------------------------------------------
// LEARNED-KNOWLEDGE SINK. A caller may supply private data without copying workflow code.
// The default remains kernel_workflow/knowledge/learned/, including when e2e spawns the lane.
// e2e's learned/ is separate memory owned by its system_architect step.
//   update_experience = on (default) | off | false | none
// The bake-off dispatcher passes `off` and curates once centrally instead (N parallel lanes would
// file N near-duplicate cards for one op).
const LEARNED_DIR = String(A.learned_dir || `${WORKFLOW_DIR}/knowledge/learned`);
// How much of ONE round the learned KB may steer. Measured, not guessed: an arm whose planner was
// handed at most 3 matched cards and could KB-seed at most one direction reached 4.45x geomean on 16
// kernels; the same 16 under a planner that reads the whole 84-line index and picks freely reached
// 3.44x — a 22.7% regression, with the losses concentrated on kernels where the bounded arm had found
// an unusual win (one went 25.77x -> 2.15x). Both arms averaged ~3 rounds, so it was not lost rounds.
// The index-reading design is better at RECALL; what it lost was the profile-driven direction that
// nothing from the KB had touched — the per-round control. `knowledge/learned/README.md` says the KB
// must be "an accelerant, NOT a cage", and this is the line that keeps that true.
//   KB_DIR_CAP        directions per round that may draw on cards (default 1)
//   KB_COLD_DIRECTION at least one direction per round must be profile-only (default on)
// Whether the planner is being pointed at the learned KB at all. Defined HERE because this lane is
// where it is USED: the constant of this name lives in the dispatcher, and referencing it from the
// lane threw `USE_LEARNED_READ is not defined` at the first plan step of every kernel — 13 of them
// before the run was stopped. A parse test cannot catch that: the file compiles, and the reference
// only resolves when the line runs. Default ON, matching this branch's design where the tech_lead
// role reads INDEX.md unconditionally; `use_learned_kb=false` turns the budget block off with it.
let USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';
// Did the CALLER freeze this kernel through oracle_freezer? Only the bake-off dispatcher does, so this
// tells director whether a GEAK_TIMING_RECEIPT is producible at all before it gates on one. Must be an
// ARG: this file never runs Freeze itself, and a workflow script has no filesystem access to go looking.
const FROZEN_ORACLE = String(A.frozen_oracle != null ? A.frozen_oracle : 'false') === 'true';

