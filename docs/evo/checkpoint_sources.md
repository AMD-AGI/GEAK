# Retaining selected sources at declared search times

These optional source hooks reuse the existing Store, commit frontier and GEAK external measurement wrapper. They do not start measurements, extend clocks or enable population search. The operational owner must install and register the exact source graph before a new study. CPU checks establish the source paths below, not production admission or scientific benefit.

## GEAK-EVO

Set `search.checkpoint_horizons` in the already admitted command config, for example:

```json
["H0.5", "H1", "H2", "H4", "H6", "H8"]
```

`run_admitted_task` forwards the option to `prepare_native_search`, which registers the existing `CommitFrontier` before `EvolutionLoop.create` makes its first journal commit. The host binding fixes the horizons and original clock. Resume reopens that same contract; a nonempty journal cannot acquire missing receipts retrospectively. Omitting the option preserves the existing behavior. `H12`, `H16` and `END` remain optional declared endpoints. H16 requires an originally declared sixteen-hour search policy; it does not extend an eight-hour clock.

H0 is the separately admitted and measured ORIGINAL. Do not manufacture an initial planner recommendation. Named hour checkpoints must fit inside the declared search interval after the final reserve; eight search hours therefore need 28,800 search seconds plus the final reserve. The owner must establish the first-counted-action origin separately.

The owner uses the existing `CommitFrontier.open` and `seal_horizon` APIs, with the original contract at `store/commit-frontier/CONTRACT.json`, the existing native episode reader and `authenticate_choice(frozen_store, context, choice, episode)`. A seal may be written after the cutoff or after an authentic early stop. It uses only the committed prefix whose receipt upper bounds are strictly before the cutoff. No timer or new search is required. Exact source bytes and validation witnesses are retained by the existing seal; the usual Store materializer restores the selected source.

Missing receipts or commits crossing the cutoff produce `unresolved`, with no fallback. The complete receipt inventory at sealing time is checked, so an unrelated later gap can invalidate delayed sealing. The ordinary final recommendation validator retains the distinction between no recommendation and an applied clear. An explicit clear needs the separately measured ORIGINAL if the registered outcome policy scores it. Identical later selected source IDs mean carry-forward or remeasurement, not discovery.

The existing population selector accepts the same finite horizon names. Its scoring, tie-breaking and unresolved-member rule are unchanged; it is used only if population selection is the separately declared treatment.

## GEAK

For the authentic GEAK lane with `external_primary`, set `capture_canonical_selections: true` in the common lane arguments. Its existing `external_measurement_command` must be the registered wrapper version supporting `measure --capture-selection`. Defaults and the search selection/stopping rules are unchanged.

The lane returns `canonical_selections` with rows for the initial search incumbent, before and after canonical commits, after recovery of an uncommitted round, and terminal selection with the actual stop reason. The rows retain exact source/candidate IDs and `capture_ref` evidence IDs; they are also in the existing workflow history. Nanosecond timestamps remain in the immutable Store artifact, avoiding JavaScript number rounding. Each artifact binds:

- The exact source closure and original clock reference, including retained clock bytes.
- Real boot/monotonic observations before and after the capture.
- The actual canonical Git commit, checked against every captured source byte and executable mode.
- The selected winner's existing settled native evidence, checked by `read_control_result` against that canonical source.

A dirty source or missing Git object leaves the commit join unresolved while retaining captured source bytes. A hand merge with different bytes cannot inherit the proposed winner's latency: `development_evidence.status` is `unresolved` and no checked result is attached. This capture requests no new native evaluation. Missing capture or an interrupted mutation remains missing/unresolved; it does not choose an earlier favorable source automatically.

These are ordinary-host provenance records. Hashes establish retained byte identity, not privileged chronology or protection against a hostile owner. The owner must retain the complete workflow/runProcess history and authenticate its order. A later result array alone cannot prove that an interrupted record was never omitted.

The opt-in host now uses `geak-canonical-source-capture-v2`. Its existing immutable artifact also contains `transition`: `kind`, `round`, attempted `sequence`, `previous_capture_ref` and `stopped_by`. Sequences start at one; the previous reference names the last successfully retained capture. A successful `before_commit` therefore survives loss of the in-memory workflow history. A crash before the matching `after_commit` or `after_restore` leaves an open mutation interval. No second journal or database is created.

Capture failures remain observational and do not stop GEAK search or change its winner. Attempt numbers still advance. Later source bytes may be retained with an explicit sequence gap; that gap is unresolved chronology, not permission to skip the failed observation. Contiguous captures validate transition order, matching round and original clock. A readout must require a contiguous chain, check retained source/evidence, and refuse to extrapolate an unknown tail. A qualifying terminal record is necessary to carry an early stop beyond the last observation. Neither the final history array nor the highest surviving sequence proves that an unobserved terminal tail never existed.

If the first capture fails, later captures retain their source with a sequence greater than one and no predecessor reference. Their history remains unanchored; the missing first observation does not prevent saving later programs.

Existing deep-mode resume, or an owner-supplied `canonical_selection_resume: true`, emits `resume_unknown` rather than claiming an authenticated continuation. The owner must set that flag for any restarted invocation on the original clock. The query must leave that segment unresolved; this source does not restore the prior host chain. The separate `--capture-selection` API still supports v1 capture without transition arguments for existing readers, but only v2 supports this chain query.

Per successful capture, the extension reads one existing predecessor artifact and adds a few transition fields to the existing capture artifact. It adds no model call or native evaluation and no additional capture subprocess beyond the original hook. Its actual overhead must be reported from the current run, not inferred from CPU fixtures.

For a conservative cutoff readout, join each `before_commit` with its `after_commit` or `after_restore`. The mutation's possible interval begins at the before-capture upper observation and ends at the after-capture upper observation. A cutoff inside that interval is unresolved; the later captured source becomes eligible only after its upper observation, with valid settled evidence. Before the initial capture, an unpaired mutation, an unexplained source change between captures, or a capture from another clock is unresolved. A terminal captured incumbent can be carried forward only when its source/evidence/stop record qualifies. H0 remains the separately admitted ORIGINAL. This document specifies the source join; it does not add a second checkpoint reducer or exporter.

Retain normal exports and complete owner cost/evaluation records. Later equal-allocation replay measures frozen selected sources and cannot alter earlier source selection or discovery time. The owner still needs to qualify complete capture/recovery coverage and preregister its cutoff reducer before claiming a complete seven-point production trajectory.
