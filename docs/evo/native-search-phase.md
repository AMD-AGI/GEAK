# Native search admission projection

The finite publisher can give generated engineer operations an earlier native
evaluation cutoff while the command's final measurements keep the protected
template's later cutoff. Add the optional `search_phase_deadline_seconds` to
the protected admission template alongside `phase_deadline_seconds`. Both must
be positive finite numbers (not booleans), and the native search value must not
exceed the inherited phase value.

When search accounting has a shorter deadline than the original clock, also set
the protected `search_deadline_seconds`. This separate positive finite number
must satisfy `search_phase_deadline_seconds <= search_deadline_seconds <=
phase_deadline_seconds`, and cannot exceed the original clock's `wall_seconds`.
The publisher constructor and each validation require the campaign state deadline
to equal the original wall origin plus this declared search duration. Without
`search_deadline_seconds`, they retain the original full-clock state deadline;
omitting both optional fields preserves existing behavior.

For example, a template with `phase_deadline_seconds: 6505` and
`search_phase_deadline_seconds: 3270` remains unchanged for command finals. A
separate `search_deadline_seconds: 3300` matches a 5700-second search reserve in
a 9000-second original arm; the native cutoff leaves the existing 30-second
turn-review allowance. Neither cutoff is inferred from the other.
The publisher first validates the ordinary host's five generated files against
the existing exact templates and dispatched values. Only after validation does
it replace `phase_deadline_seconds` with `3270` in the generated admission it
seals. The optional search field remains present as provenance. Ordinary hosts
must continue submitting the inherited value; a host cannot choose the cutoff.
Invalid projection values fail before the operation is marked used or sealed.

The projection changes no other file or admission field. In particular, the
host config and its hash binding, contract, source store, measurement epoch,
session, shared budget, and original clock remain paired as before. The native
helper still enforces the complete evaluation envelope using the sealed phase
offset from the original monotonic origin; use the companion `native-phase.patch`
and its pinned base. This projection does not create or extend a clock.

Apply `native-search-phase.patch` with `git apply` from the copied native root.
Its path is `bridge/finite_publisher.py`; verify the original and resulting
SHA256 values in `native-search-phase-base.json`. Register the patched copy in
the prospective immutable installation and source closure. The repository patch
does not modify an installed publisher.

The focused CPU test applies the patch to the hash-checked retained source and
executes its real constructor, `validate` state-authority checks and `publish`
method. Filesystem/clock custody, later dispatch reconstruction, and the sealing
writer are doubles. It tests deadline authority, projection and ordering, not a
live root install or native phase enforcement. Run:

```sh
python -m unittest discover -s tests/geak_evo -p test_native_search_phase_patch.py
```

The test skips when the pinned retained source is unavailable. No model, device,
privilege transition, or clock creation is exercised.
