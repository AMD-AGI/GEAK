# Local EVO archive CLI contract (schema 2)

`archive.py` uses standard-library Python 3.8+ and local Linux Git. It retains
bounded source snapshots and author-reported evidence. It performs no agent,
GPU, API, task execution, evaluation, performance calculation, ranking,
promotion, migration, or automatic parent/donor selection. It does not import
another EVO implementation. Main owns dispatch, materialization, benchmark
references, verification, promotion, cleanup and actual peer review.

Schema 2 uses Git-style regular-file modes throughout source identity. Schema-1
archives retain their original exact-POSIX-mode meaning and are rejected without
rewriting their objects or index. Use the original helper to read those archives;
new runs need fresh archives. No automatic migration or alternative backend is
introduced.

## Invocation and integration order

Run the helper through the existing Workflow agent plus Bash bridge and capture
the actual command stdout and exit code. Do not invoke it through host
`runProcess`. Shell-quote every argument, especially the JSON argument; never
interpolate report metadata as shell code.

```text
python3 -B evo/archive.py init --archive ABS_NEW_DIR --canonical ABS_GIT_WORKSPACE --source-path REL_FILE [--source-path REL_FILE ...]
python3 -B evo/archive.py query --archive PATH [--entry ID]
python3 -B evo/archive.py prepare --archive PATH --workspace ABS_EXISTING_WORKSPACE --canonical ABS_ROUND_CANONICAL [--parent ID] [--donor ID ...] [--initialize-index]
python3 -B evo/archive.py check-patch --archive PATH --canonical ABS_ROUND_CANONICAL --patch ABS_PATCH_FILE
python3 -B evo/archive.py record-batch --archive PATH --canonical ABS_ROUND_CANONICAL --report-json JSON_STRING [--report-sha256 HEX]
```

1. Finish ordinary main canonical preparation and warmstart, then `init` before
   round 1. Select the source paths explicitly as task owner. A path may name a
   regular file or a declared, currently absent future source file.
2. Materialize each engineer workspace using main's existing materializer.
   It excludes `.git`. Call `prepare --initialize-index --canonical ROUND_CANONICAL` before the engineer.
   The workflow always supplies the round canonical and requires its source-identity
   receipt. Declared file bytes, owner-executable status and absence must match before index creation
   or parent overlay. An excluded declared file, such as `build/kernel.py`, fails
   explicitly instead of silently becoming absent. Standalone helper calls may
   omit this comparison only when they intentionally manage their own baseline;
   the receipt then has null `materialized_canonical_source_id`, which the workflow rejects.
   This creates a Git index from only the materialized declared source bytes
   when `.git` is absent, with no commit, identity requirement or `git add -A`.
   Existing Git metadata/indexes are preserved. Override ordinary engineer
   initialization guidance: do not commit, reset, or reinitialize this baseline.
3. Engineers may run ordinary `git diff`/`git status`. To include a declared
   initially absent file, use `git add -N -- REL_FILE`, then `git diff --binary`.
   Do not stage source changes. Do not emit unrelated untracked files in the
   patch. Source restrictions still apply to all candidates.
4. Before main verifies or promotes a patch, require a successful `check-patch`.
   A rejected candidate fails closed independently of other candidates. Missing
   patches need no check when main skips verification, for example below-floor
   reports. The receipt identifies exact patch bytes, current canonical declared
   source bytes and reconstructed candidate bytes. It returns a frozen immutable
   `patch_object_path`: use that exact file for main verification, promotion and
   the subsequent record report's `patch` field. The original engineer patch
   path may subsequently change without changing this retained artifact. Keep
   the round canonical stable through verification/recording/promotion; recheck
   against the new canonical if it changes.
5. Call `record-batch` with the independent returned engineer/verifier values
   after results are available and before canonical promotion or workspace/patch
   cleanup. When `--report-sha256` is supplied, the helper refuses
   (`report_digest_mismatch`, nothing recorded) unless it equals the SHA-256 of the
   UTF-8 report JSON it received. Do not substitute unfinished workspace bytes for a best patch.
6. Main performs its own final canonical validation and promotion. An archive
   retention failure is not verification success or a promotion authorization.

Retention is deliberately fail-closed if the CURRENT workspace or its Git baseline
disappears before `record-batch`: raw reports and already-frozen patch objects remain,
but no discoverable entry is created without the parent-custody check. This differs
from reusing an already recorded source after its OLD workspace is cleaned up.
The latter is supported and tested. Recovering discoverable best-patch entries
with unknown ancestry after a current-workspace loss is not implemented here.

The archive and source workspaces must be separate directory trees. Absolute
canonical/workspace/patch paths are required; archive paths after initialization
may be relative. All existing directory components must be real directories,
not symlinks. The caller must keep declared files, Git baselines and patch files
quiescent during each command; the archive lock does not lock engineers.

## Common output and bridge validation

Successful commands write exactly one compact JSON object to stdout and exit 0.
The CLI envelope always includes `ok:true`, `command` equal to the invoked
operation and normalized absolute `archive`. Validate those fields as well as
the actual exit code; a syntactically valid JSON object alone is insufficient.
Argument, archive-integrity and command-level failures exit 2 with:

```json
{"ok":false,"error":{"code":"out_of_scope_patch","message":"..."},"verification":"not_performed"}
```

`--help` uses ordinary argparse help. Otherwise diagnostics remain JSON, not
tracebacks. No missing performance values are replaced with zero.

`record-batch` exit 0 means the batch index update committed, even if individual
reports failed. Inspect `has_failures`, `results[*].status`, `source_retention`
and `diagnostics`. A global bounds/integrity failure exits 2 and does not commit
a partial index. Source retention and verification are distinct fields.

## Operation results

### init

Returns `founder_source_id`, `source_retention:"retained"`,
`verification:"unknown"`, and `reused`. The archive directory must be new; an
exact completed initialization retry is allowed. Different initialization
canonical, scope or current founder bytes/executable status yields `init_conflict`.

`FOUNDER` means the canonical bytes captured at initialization. Those may
already contain warmstart changes. `original_benchmark.status` is
`external_not_captured` with a null source identity. Main's true ORIGINAL
benchmark remains separately main-owned; the helper neither reads arbitrary
benchmark metadata paths nor relabels FOUNDER as ORIGINAL. ORIGINAL is reserved
and is not a selectable captured source.

### query

Without `--entry`, returns the complete compact `index`: source scope, ORIGINAL
ownership marker, entries, original parsed reports, preparation receipts,
workspace-to-receipt references and batches. It never silently truncates.

With `--entry`, returns `entry`. For retained entries it also returns
`source_manifest`, `source_manifest_path`, and `blob_directory`. The manifest
maps every declared relative path to either `{"present":false}` or
`{"present":true,"sha256":"...","size":N,"mode":N}`. `mode` is the normalized
integer 0644 or 0755, selected using the owner-executable bit, not the full
filesystem permissions or Git mode string. Read a retained
file's bytes from `blob_directory/SHA256`; validate its SHA256 before use.
Donors therefore remain materializable after original workspaces are deleted.
Do not execute these bytes merely to inspect a donor.

### prepare

Returns `receipt_id`, `receipt`, `source_retention:"retained"` and
`verification:"not_performed"`. `--parent` defaults to `INCUMBENT`.

- `INCUMBENT` leaves existing materialized source bytes/modes/absence alone and
  captures their actual source identity in the receipt.
- `FOUNDER` or a retained entry ID overlays only the declared source files and
  their absences/modes. Unselected files remain alone. Donors are validated,
  retained-source references only; their contents are not automatically applied.
- Unknown, unretained, stale or corrupt parent/donor sources fail before overlay.
  Prepare refuses a workspace known to the archive as a canonical directory.
- `--initialize-index` initializes only when the workspace has no `.git`, seeding
  the pre-overlay declared source baseline using Git blob/index plumbing. No
  source code, filters or hooks execute. The new index is canonical-relative
  even when the selected parent differs from the current canonical.
- An existing `.git/index` is not written by the helper. The receipt contains
  `workspace_index_sha256` for the actual preserved index bytes and a semantic
  `workspace_index_baseline` containing `entries_sha256`, `head`, and
  `allowed_intent_to_add`. Subsequent stat-cache refreshes are acceptable.
  Staged mode/blob/stage changes or HEAD changes are rejected. Only real
  intent-to-add empty-blob placeholders for the receipt's declared initially
  absent files are ignored; normally staged empty files and undeclared
  intent-to-add paths are rejected.
- `index_initialization.status` is `created_declared_source_index` with
  `base_source_id`, or `existing_index_preserved`. Retries preserve the original
  initialization provenance when the semantic baseline still matches.

`receipt_id` is SHA256 of canonical JSON for the receipt, stored as an immutable
object. It binds workspace, requested parent, actual `prepared_source_id`,
donor IDs/source identities and index baseline. Record checks the most recent
receipt for that workspace. An explicit parent/donor report requires a matching
receipt. An ordinary INCUMBENT report without a receipt is retained with
`parent_binding.status:"unknown_no_prepare_receipt"`; no actual ancestry is
invented. File overlays are atomic per file, with rollback attempted on detected
failure. They are not a crash-atomic multi-file transaction; a crash without a
committed receipt requires fresh materialization/preparation.

### check-patch

Returns `source_scope_valid:true`, `source_retention:"retained"`,
`entry_recorded:false`, `verification:"not_performed"`, `receipt_id`, a top-level
absolute `patch_object_path`, top-level `patch_sha256`, `canonical_source_id`
and `source_id` aliases for the receipt fields, and:

```json
{"receipt":{"patch_sha256":"...","round_canonical_source_id":"...","canonical_source_id":"...","source_id":"...","source_paths":["src/kernel.py","new.py"],"canonical":"/absolute/canonical","patch":"/absolute/best.diff","patch_object_path":"/absolute/archive/objects/PATCH_SHA256"}}
```

The receipt is deterministic SHA256-bound canonical JSON. The receipt, patch,
base and reconstructed source are retained as immutable artifacts without entry
records or changes to the index. `canonical_source_id` aliases
`round_canonical_source_id`. The source/patch IDs match the corresponding best
entry from `record-batch` using the frozen patch and unchanged canonical. Scope
validity and source custody are not runtime correctness or measured performance.
Neither canonical source nor canonical Git metadata/index is modified.

### record-batch

`JSON_STRING` is a JSON list of report objects. An example:

```json
[{"id":"r1_d0","parent_id":"INCUMBENT","donor_ids":[],"title":"candidate","workspace":"/absolute/worker","patch":"/absolute/best.diff","engineer":{"status":"ok"},"verifier":{"status":"verified"}}]
```

Only `id` and an absolute `workspace` are required. Missing `parent_id` defaults
to INCUMBENT, missing `donor_ids` to `[]`. Missing/empty/null `patch` produces a
best-source diagnostic. `engineer`, `verifier` and `title` are optional; absent
evidence remains null/unknown. Additional report fields remain in the original
report without being interpreted or dereferenced. Independent returned JSON
values, including case weights, metric values and provenance, are preserved
unchanged as JSON values under labelled `author_reported_evidence`. The exact
UTF-8 input JSON string is retained as an immutable artifact too, preserving
original whitespace and number spellings. Metadata file references are not
opened or executed. Only the explicitly supplied workspace and patch are read.

For each valid report ID, two immutable logical references are attempted:

- `REPORTID-best`: reconstruct the supplied patch against the current round
  canonical's declared bytes in a new isolated temporary Git repository/index.
  This base is independent of FOUNDER, selected parent and canonical HEAD.
  Store exact `patch_sha256`, `round_canonical_source_id`, and `source_id`.
  Reject every out-of-scope path and symlink/submodule/nonregular patch mode.
  The verifier JSON is only author-reported evidence about this bound best
  source: `verification.status` is `author_reported_only` if supplied, otherwise
  `unknown`; `independently_verified` is always false.
- `REPORTID-development`: capture declared workspace bytes even for slower or
  failed engineer reports or missing/corrupt patches. Status is always
  `unverified_development`, with `evidence_applies_to:"no_measurement_binding"`
  and null `bound_source_id`. Copying the independent report does not transfer
  its measured-performance claim to unfinished development source.

Identical snapshots deduplicate artifacts but retain both logical references
and their distinct evidence bindings. A source capture failure records
`source_retention:"failed"`, null `source_id`, `verification.status:"unknown"`
and an explicit `diagnostic`. Parent/receipt/schema/ID failures can reject the
whole report; these are preserved in batch results even without assigned entry
IDs. Malformed or absent patches do not abort other reports in the batch.

The result contains `batch_id`, `has_failures`, and one result per input ordinal:
`id`, `status` (`recorded`, `partial`, `failed`, `conflict`), `entries`,
`source_retention` by source kind, `verification:"not_performed"` and
`diagnostics`. IDs are not rankings. All evidence includes the caveat that
only selected source bytes are bound: dependencies, runtime, benchmark inputs,
case weights, measurement validity and actual verifier consumption are not
independently established by this helper.

## Bounds, safety and retry semantics

- Entry/parent/donor IDs match `[A-Za-z0-9][A-Za-z0-9_-]{0,95}`. Report IDs have
  at most 64 characters; FOUNDER, INCUMBENT and ORIGINAL are reserved.
- Declare 1 to 256 unique ASCII relative paths, at most 1024 characters each.
  No traversal, backslashes, empty/dot components, `.git` components, symlink
  components, submodules, directories, devices, FIFOs or special permission bits.
- A snapshot's total declared bytes are at most 32 MiB. All immutable objects
  together, including source blobs/manifests, patches, raw reports and receipts,
  are also limited to 32 MiB and 32768 objects. No cap-increasing option exists.
  No automatic eviction or pruning occurs.
- A patch is at most 8 MiB. A batch JSON string is at most 1 MiB and 128 reports;
  an individual report is at most 256 KiB. Ordinary Linux argument-size limits
  may impose a smaller CLI input limit. Each index collection is at most 4096
  records, and the complete index is at most 8 MiB. JSON duplicate keys,
  nonfinite numbers and excessive nesting are rejected.
- Each Git subprocess has a 30-second wall timeout, 20-second CPU limit,
  512 MiB address-space limit and 64 MiB per-output-file limit. Git uses a
  sanitized environment, disabled hooks/fsmonitor/global/system config, no
  external diff/filter commands, no network and no task execution. Source
  reconstruction uses raw blob plumbing without content conversions.
- Source snapshots and reconstructed patches use the same regular-file modes:
  0644 or 0755 according to the owner-executable bit. Group/world write and read
  permissions are operational metadata, not source identity. Capturing an
  incumbent does not change its actual permissions; restoring another source
  uses the normalized mode. Exact historical POSIX-mode restoration is not claimed.
- The canonical source is captured from current file bytes, not assumed equal
  to `HEAD`. A dirty index/worktree or declared untracked source does not require
  a commit, reset or greedy fallback. The engineer index must still remain bound
  to the prepared canonical baseline for its final diff.
- Archive operations use an exclusive nonblocking `flock`; contention returns
  `archive_busy`. Objects have SHA256 filenames, are written immutable/read-only
  and verified on access. The index is fsynced and replaced atomically. Failures
  may leave bounded, unreferenced immutable objects, never a partly written
  committed index. Source workspaces still require caller-enforced quiescence.
- Exact observation retries return the same results/IDs. A changed patch,
  workspace snapshot, round canonical, metadata or binding under a reused report
  ID returns `id_conflict`; old entries are never silently overwritten. Duplicate
  report IDs within a batch all fail explicitly. After cleanup, a retry may
  conflict because its observed inputs are no longer the original inputs.
- Crash remnants from incomplete initialization require a different new archive
  path; initialization does not guess that such a directory is safe to replace.
  A full/corrupt archive fails explicitly rather than truncating query output,
  weakening source bounds or claiming verification.

## Offline validation

```text
python3 -B -m unittest discover -s evo/tests -p test_archive.py -v
```

The tests use real temporary repositories, file operations and fixture commits,
including main's actual materializer. They cover source/patch/evidence
separation, warmstart/ORIGINAL semantics, old-parent preparation, sequential
rounds, new/deleted/binary/mode-changed sources, semantic index receipts,
intent-to-add, corrupt/unknown inputs, traversal/symlink/submodule rejection,
bounded failures, exact retries, immutable IDs, metadata provenance and no
execution of hooks, filters or metadata-referenced files. No project commits,
branches or pushes are made by the helper or its tests.
