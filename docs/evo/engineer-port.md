# Engineer port

Canonical input: `oct15_source/evolution/src/geak_evo/engineer.py` and `claude_transport.py`, plus `oct15_source/bridge/native_{assignment,investigation}.py` and `operation_files.py`, from the Oct 15 source snapshot recorded by `SOURCE_PINS.json`.

| Responsibility | Production module | Preserved behavior |
| --- | --- | --- |
| Exact request validation | `geak_evo/engineer/request.py` | Job, source, measurement epoch, evidence, continuation and finite budget binding |
| Continuing CLI worker | `geak_evo/engineer/runner.py` | Private ownership locks, durable attempt receipts, source capture after settlement, raw events, unknown cost, append-only recovery |
| Claude command and events | `geak_evo/engineer/backend.py` | Explicit provider route, native session/model identity, raw usage and cost |
| Native operation and collection | `geak_evo/engineer/investigation.py` | Six prepared refs, exact operation ID, original native receipts, same-ID reconciliation, archived source roles and measurements |
| Planner assignment to engineer | `geak_evo/engineer/assignment.py` | Admission, reservation, lease, review, own-history parents/donors, fresh native episode and pending review |
| Runtime operation files | `geak_evo/engineer/operations.py` | Exact supplied templates, workspace preparation, bounded deadline, explicit read material and native tool interface |
| Research review attachments | `geak_evo/engineer/research.py` | Existing claim append/correction and exact pending-review outcome |

Archive refs use the shared `archive.refs.EngineerError`, which remains catchable as `ValueError`. Job validation is provided by `evaluate.jobs.load_engineer_job`. Shared `canonical_json(..., ensure_ascii=False)` retains the engineer's UTF-8 record bytes. Native indented JSON retains its original formatting.

Runtime seams: existing launch/settle/capture callbacks remain required for CLI turns; native process execution and clocks are injectable. Configured roots/templates remain explicit. Boot identity checks use the shared `SystemClock.boot_id()` service or the injected clock. Publication uses the host callback already required by `NativeAssignment`; the unused `publish_sealed` wrapper is retired. Operation, source, measurement and deadline identities remain checked. No model or GPU invocation is part of the validation suite.

The default prompt uses the packaged `geak_evo/prompts/engineer.md`. `geak_evo/PROMPT_SOURCES.json` separates its current version and hash from historical import provenance. The concise revision applies to new work; saved runs keep their original instructions. Callers can still supply `prompt_path`; synthetic parity tests explicitly supply the same prompt to both versions.

Source mismatch found before port: `engineer.inspect_events(backend="claude")` omits the required `model_route` argument of `inspect_claude_events`. With valid synthetic Claude events and an explicit route in `backend_options`, the original runner records `event_read:TypeError`; original recovery raises the same uncaught `TypeError`. The isolated repair forwards the explicit route and saves a detached `model_route` in Claude receipts before launch, so recovery uses the original accepted model set. Codex receipt fields are unchanged. An old Claude receipt without a saved route remains unqualified; the port does not guess a provider identity.

Validation on the integrated engine (foundation through `8e1d923c`): 39 tests passed. The 28 inherited engineer fixtures, seven native assignment/investigation fixtures and two byte/error compatibility checks pass on both the Oct 15 source and the port. Two additional Claude route checks fail on the original as described above and pass after the isolated repair. This is 37 parity checks plus two explicit defect repairs, not a claim of identical outcomes across all 39 tests. The native fixtures inject a CPU-only entry and collector, retaining source roles, donor ancestry, raw usage/cost, no-repeat operation identity and pending review. The legacy CPU job fixture is unchanged; only imports and the explicit common prompt location differ.

Run the integrated checks with the engine and `tests/geak_evo` on `PYTHONPATH`:

```sh
python3 -m unittest -v test_engineer_port test_native_engineer_port test_engineer_bytes test_engineer_route
```

For the original baseline, use `EVO_ENGINEER_PARITY=original` with the Oct 15 `evolution/src`, `bridge` and the same test directory on `PYTHONPATH`. The route tests intentionally reproduce the two source defects; the other tests pass.

Runtime bindings still required by `NativeAssignment`: actual workspace helper, exact config/admission/entry/collection templates, collector reference, campaign clock and shared budget, publication callback, bindings/read material, and the admitted backend driver. The native entry and its provenance default to packaged `engineer/js` files; callers can override them with exact refs. `backend_driver_ref` can replace the legacy template-derived driver location. The unused `ClaudePlannerRuntime` wrapper is retired; Claude command/event/route functions remain. No fake model, collector or evaluator is provided in production code.

Reciprocal review: the planner owner compared assignment, investigation and operations against the pinned source and found no semantic blocker after the clock and publication guards were restored. Their request to avoid capturing `subprocess.run` at import time was applied by resolving the optional process runner in the constructor. Full model/GPU execution and production runtime qualification were outside this source-port validation.

October 16 prospective cleanup: repository and current-composition callers do
not use `completed_native_work`, `observation_material`, `archive_observation`
or `verify_observation`; that orphan helper chain is retired. `claim_update`
still binds claim append/correction requests to the exact settled outcome.
Native investigation, collection, archived measurements and lineage remain.
The counts above describe the initial port; old studies retain original source.
