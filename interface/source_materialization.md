# Complete accepted Python source handoff

`run_e2e.py` supports the existing source-materialization schema version 1 in
a schema-version-2 handoff. Source-free handoffs retain their existing behavior.
Sparse `source_snapshots` alone are explicitly unresolved: independent sparse
packages cannot preserve cumulative edits, unchanged dependencies and deletions.

The producer supplies the ordered accepted layer IDs in `source_snapshots` and
this exact descriptor under `baseline_env_spec.source_materialization`:

```json
{
  "schema_version": 1,
  "status": "ready",
  "bundle_root": "/absolute/accepted-source",
  "manifest_path": "manifest.json",
  "manifest_sha256": "<64 lowercase hexadecimal characters>",
  "required_layer_ids": ["layer-1", "layer-2"],
  "pythonpath_prefixes": ["trees/framework/python"]
}
```

`manifest.json` contains exactly `schema_version`, `required_layer_ids`,
`pythonpath_prefixes`, `trees`, `files`, `deleted_paths`, `modules` and
`deleted_modules`. Tree rows contain `tree_id`, `root`, `accepted_commit` and
`layer_ids`. File rows contain `path`, `sha256` and the integer POSIX `mode`.
Module rows contain `name` and `path`. All manifest paths are relative to the
bundle root. The file list covers each complete accepted tree, including
unchanged dependencies and package data. `modules` identifies the affected
Python modules; deleted Python paths must exactly match `deleted_modules`.

GEAK verifies the manifest digest, file bytes and modes, exhaustive inventory,
layer order, import-root ownership, package initializers and absent deletions
without importing supplied code. It stages the complete bundle under
`<eval_dir>/source_materializations/<manifest_sha256>/`, together with the pure
verifier, `source_paths.sh` and `source_request.json`. Different pre-existing
sidecars or corrupted staged content are refused. `--dry-run` validates the
source without copying it. The effective configuration digest includes the
source manifest digest.

The normal `bench_e2e.sh` entry revalidates the per-run request before evaluating
a leg. It composes candidate overlay roots followed by accepted source roots;
the normal backend defaults and inherited paths follow. Accepted roots survive
when an engineer replaces `OVERLAY_PYTHONPATH`. SGLang's editable-install fallback
is moved behind accepted roots for source-bearing calls only. Python bytecode
writes are disabled for these calls so imports do not dirty the sealed inventory.
The current launcher, measurement protocol and scheduling remain authoritative.

Version 1 supports complete regular Python packages and data. It rejects
namespace packages, ambiguous ownership, package deletion, startup hooks and
runtime compiled artifacts inside the affected packages. A bundle can be
relocated by updating `bundle_root` and restaging it through `run_e2e.py`; its
manifest digest remains unchanged. A completed eval directory uses absolute
source-request paths and must be restaged after relocation.

These checks establish source transport and ordinary import precedence. They do
not prove actual serving-process engagement, compiler/native-code closure or
performance. A custom launch recipe that resets `PYTHONPATH`, an independently
executed adapter that bypasses `bench_e2e.sh`, or package code that rewrites its
own search path requires separate integration evidence. No historical source
observation, measurement attestation, launch-capture or reserve controller is
part of this port.

Deploy the compatible GEAK consumer before enabling producer handoffs containing
this descriptor. Source-bearing handoffs to an older consumer must be refused by
the producer's capability check; source-free controls need no new capability.
