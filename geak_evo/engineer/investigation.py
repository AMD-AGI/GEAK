"""Run one exact native operation and collect its saved source and observations.

The caller prepares runtime files and supplies execution and clock dependencies.
An uncertain operation remains pending for reconciliation under the same ID.
"""

import json
import os
from pathlib import Path
import subprocess

from ..clock import SystemClock
from .._util import require
from ..archive.refs import file_ref, read_ref as _read_ref
from ..archive.store import StateError, _write as _store_write
from .request import validate_request


def need(value, reason):
    require(value, reason, ValueError)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(ref):
    return json.loads(_read_ref(ref))


def write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def write_or_match(path, raw):
    """Keep an immutable input across collection retries; changed bytes fail."""
    try:
        _store_write(path, raw, immutable=True)
    except FileExistsError:
        need(_read_ref(file_ref(path)) == raw, "saved collection input changed")
    return file_ref(path)


def flat_action(proposal, policy=None):
    """Native bridge mechanisms supported by this prospective integration."""
    from ..archive.maturation import MaturationError, validate_maturation_decision
    from ..archive.policy import (ProposalError, native_enabled, program_lease_bound, proposal_need,
                                  proposal_ref, validate_lease, validate_raw_engineer_proposal)

    if proposal is None:
        return
    proposal_need(type(proposal) is dict, "proposal must be an object or explicit null")
    protection = (proposal.get('kind') == 'maturation_review' or
                  proposal.get('action') in ('nominate', 'revive') or proposal.get('protection_ref') is not None)
    cells = (proposal.get('action') in ('charter', 'membership') or
             proposal.get('membership_ref') is not None or proposal.get('island_id', 'unassigned') != 'unassigned')
    proposal_need(not protection or native_enabled(policy, 'protection'), 'protection is disabled by the registered native policy')
    proposal_need(not cells or native_enabled(policy, 'cells'), 'cells are disabled by the registered native policy')
    lease_enabled = native_enabled(policy, "program_lease")
    kinds = (None, "research_review", "research_action", "maturation_review")
    proposal_need(proposal.get("kind") in kinds + (("lease_release",) if lease_enabled else ()),
         "later mechanism is inactive")
    if proposal.get("kind") == "lease_release":
        proposal_need(set(proposal) == {"kind", "direction_id", "expected_revision", "reason"}
                      and type(proposal["reason"]) is str and proposal["reason"].strip(),
                      "lease_release requires direction, revision and a nonempty reason")
    if proposal.get("kind") == "research_action":
        required = {"kind", "action", "direction_id", "expected_revision", "candidate_id",
                    "expected_claim_head", "premises", "payload"}
        proposal_need(set(proposal) == required, "invalid research action proposal fields")
        proposal_need(proposal["action"] in ("accept_anchor", "nominate", "revive", "charter", "membership"), "formal exchange archive mechanism is inactive")
        proposal_need(type(proposal["candidate_id"]) is str and type(proposal["premises"]) is list
                      and type(proposal["payload"]) is dict, "invalid research action fields")
    if proposal.get("kind") == "research_review":
        required = {"kind", "direction_id", "expected_revision", "transport_ref", "collection_ref", "note", "claim_requests"}
        release = {"lease_release"} if lease_enabled else set()
        proposal_need(required <= set(proposal) <= required | release, "invalid planner review fields")
        proposal_need("lease_release" not in proposal or (type(proposal["lease_release"]) is str
                      and proposal["lease_release"].strip()), "lease_release must be a nonempty reason")
        for key in ("transport_ref", "collection_ref"):
            proposal_ref(proposal[key])
        note = proposal["note"]
        proposal_need(type(note) is dict and set(note) == {"claim", "scope", "next_test", "evidence_uptake"}
                      and all(type(value) is str and value.strip() for value in note.values()), "invalid research review note")
        proposal_need(type(proposal["claim_requests"]) is list, "claim_requests must be a list")
        for request in proposal["claim_requests"]:
            proposal_need(type(request) is dict, "claim request must be an object")
            action = request.get("action")
            fields = ({"action", "candidate_id", "record"} if action == "append" else
                      {"action", "candidate_id", "targets", "scope", "reason", "consequence", "replacement", "status"})
            proposal_need(action in ("append", "correct") and set(request) == fields,
                          "claim request must use exact ClaimLog append or correct fields")
    if proposal.get("kind") == "maturation_review":
        required = {"kind", "direction_id", "expected_revision", "maturation_ref", "decision"}
        proposal_need(set(proposal) == required and type(proposal["decision"]) is dict,
                      "invalid maturation review fields")
        proposal_ref(proposal["maturation_ref"])
        decision = proposal["decision"]
        try:
            validate_maturation_decision(decision)
        except MaturationError as exc:
            raise ProposalError(str(exc)) from exc
        proposal_need(decision.get("action") in ("observe", "review"), "native maturation requires observe or review")
        proposal_need(decision.get("outcome") not in ("realize", "refute"),
                      "scoped realization/refutation verifier is not configured")
    if proposal.get("kind") is None:
        raw_proposal = proposal
        if "lease" in proposal and lease_enabled:
            validate_lease(proposal, program_lease_bound(policy))
            raw_proposal = {key: value for key, value in proposal.items() if key != "lease"}
        validate_raw_engineer_proposal(raw_proposal)
        proposal_need(proposal.get("exchange_ref") is None, "formal exchange requires its separate native adapter")
        proposal_need((proposal["island_id"] == "unassigned" and proposal.get("membership_ref") is None)
             or (proposal["island_id"] != "unassigned" and type(proposal.get("membership_ref")) is dict),
             "assigned native work needs an exact membership reference")
        donors = proposal.get("semantic_donors", [])
        proposal_need(type(donors) is list and all(type(identity) is str and identity for identity in donors)
             and len(donors) == len(set(donors)) and proposal["parent_candidate_id"] not in donors,
             "explicit semantic donors must name distinct candidate IDs")
    proposal_need(type(proposal["direction_id"]) is str and proposal["direction_id"].strip()
                  and type(proposal["expected_revision"]) is int and proposal["expected_revision"] >= 0,
                  "invalid proposal direction or revision")


def inspect_prepared_identity(prepared, request, *, prompt_ref, operation_id):
    """Bind existing input/config/collector schemas to the reserved outer call.

    The runtime prepares these files after receiving the exact prompt reference
    and operation ID. Checks bind their bytes and identities to the reservation.
    """
    expected = {"entry_runner_ref", "entry_input_ref", "backend_binding_ref",
                "collection_config_ref", "collector_ref", "entry_provenance_ref"}
    need(set(prepared) == expected, "exact native host references required")
    for ref in prepared.values():
        _read_ref(ref)
    entry = read(prepared["entry_input_ref"])
    backend = read(prepared["backend_binding_ref"])
    config = read(backend["config_ref"])
    _read_ref(backend["driver_ref"])
    collect = read(prepared["collection_config_ref"])
    native = config["backend_config"]
    probe = native["probe_config"]
    context_policy = probe.get("report_context")
    if context_policy is not None:
        need(context_policy in ({"capture": "off", "delivery": "off"},
                                {"capture": "report-v1", "delivery": "history-v1"}),
             "report context requires explicit history delivery; private capture-only is unsupported")
        if context_policy["capture"] != "off":
            output = str(Path(request["workspace"]) / ".idea-output" / operation_id)
            need(entry.get("output_dir") == collect.get("output_dir") == output,
                 "trial report must use the host-assigned engineer output directory")
    need(entry["protocol"] == "geak-one-engineer-v1"
         and entry["turn_id"] == probe["turn_id"] == collect["turn_id"] == operation_id,
         "native turn differs from reserved operation")
    need(entry["idea_id"] == native["direction_id"] == collect["direction_id"] == request["direction_id"]
         and entry["workspace"] == probe["workspace"] == collect["workspace"] == request["workspace"],
         "native direction/workspace differs")
    need(probe["job_id"] == collect["job_id"] == request["job"]["job_id"]
         and probe["measurement_epoch_id"] == collect["measurement_epoch_id"] == request["measurement_epoch_id"]
         and entry["starting_source_id"] == collect["starting_source_id"] == request["tip"]["source_id"],
         "native parent/domain differs")
    need(native["continuation"] == entry["continuation"] == {"mode": "fresh", "prior_receipt_ref": None}
         and request["continuation"]["mode"] != "resume", "native idea uses a fresh engineer with explicit history")
    need(entry["direction"]["prompt"] == request["next_question"], "native question differs from planner choice")
    if request["donors"]:
        need(entry["parents"] == collect["parents"]
             and [row["candidate_id"] for row in entry["parents"]] ==
                 [request["tip"]["candidate_id"]] + [row["candidate_id"] for row in request["donors"]],
             "native semantic parent material differs from exact selected sources")
    # The factory must deliver the exact outer assignment through an ordinary
    # existing native input, not substitute a new prompt after its seal.
    delivered = [ref for ref in entry.get("insights", []) if type(ref) is dict
                 and set(ref) == {"path", "sha256"} and ref["sha256"] == prompt_ref["sha256"]]
    need(len(delivered) == 1 and _read_ref(delivered[0]) == _read_ref(prompt_ref),
         "outer prompt absent from native context")
    roots = [Path(path) for path in native["model_config"]["read_roots"]]
    need(any(Path(delivered[0]["path"]) == root or root in Path(delivered[0]["path"]).parents for root in roots),
         "outer assignment absent from native read grants")
    need(config["entry_input_ref"] == prepared["entry_input_ref"], "driver pins another native entry")
    return entry, config, collect


def inspect_prepared(prepared, request, *, prompt_ref, operation_id, clock=None):
    entry, config, collect = inspect_prepared_identity(prepared, request, prompt_ref=prompt_ref, operation_id=operation_id)
    native, probe = config["backend_config"], config["backend_config"]["probe_config"]
    clock_record = read(native["clock_ref"])
    clock = SystemClock() if clock is None else clock
    now = clock.monotonic_ns()
    need(clock_record["boot_id"] == clock.boot_id()
         and now < probe["turn_deadline_ns"] <= probe["budget"]["deadline_ns"]
         and (probe["turn_deadline_ns"] - now) / 1e9 <= request["budget"]["wall_seconds"]
         and entry["timeout_ms"] / 1000 <= request["budget"]["wall_seconds"],
         "native original clock exceeds the outer work reservation")
    return entry, config, collect


def transport_envelope(request, native, native_ref, *, operation_id, command, elapsed_seconds):
    """Adapt receipt fields without inventing Codex events or native success."""
    need(native["schema"] == "geak-native-idea-model-receipt-v1"
         and native["invocation_id"] == operation_id
         and native["direction_id"] == request["direction_id"]
         and native["workspace"] == request["workspace"]
         and native["job_id"] == request["job"]["job_id"]
         and native["measurement_epoch_id"] == request["measurement_epoch_id"]
         and native["initial_source_id"] == request["tip"]["source_id"], "native receipt differs from intent")
    settled = native["settlement"].get("settled") is True
    observed = native.get("observed") or {}
    terminal = native.get("terminal_snapshot") if settled else None
    complete = settled and native["status"] == "completed"
    need(not complete or (observed.get("identity_valid") is True
         and observed.get("terminal") == "turn.completed" and not observed.get("protocol_errors")),
         "completed native receipt lacks authenticated model completion")
    return {"schema_version": 1, "attempt_id": operation_id, "request": request,
        "backend": "native_idea", "command": command, "elapsed_seconds": elapsed_seconds,
        "process_status": "exited" if settled else "unsettled", "exit_code": native.get("returncode"),
        "process_reaped": settled, "settlement": native["settlement"],
        "captured_tip_source_id": terminal["source_id"] if terminal else None,
        "transport_complete": complete, "identity_valid": observed.get("identity_valid", False),
        "thread_id": native.get("thread_id"), "terminal": observed.get("terminal"),
        "protocol_errors": observed.get("protocol_errors", []),
        "errors": [native["error_kind"]] if native.get("error_kind") else [],
        "native_receipt_ref": native_ref,
        "scope": "Outer lifecycle adapter; native_receipt_ref is authoritative. No synthetic CLI event stream."}


def observation_context(manifest):
    """Copy source-bound native observations and raw timing method fields only."""
    rows = []
    for trial in manifest["trials"]:
        value = trial.get("verified")
        if not value or trial.get("error"):
            continue
        row = {key: value.get(key) for key in ("source_id", "operation_id", "per_case_times_ms",
                                                "measurement_valid", "custody_proven", "reward_eligible")}
        row["feedback_ref"] = trial["archive_ref"]
        row["raw_benchmarks"] = []
        for artifact in trial["artifacts"]:
            name = artifact["input_ref"]["name"]
            if artifact["error"] is None and name.endswith("/_bench_raw.json"):
                raw = read(artifact["archive_ref"])
                # Preserve exact method/call population fields; no profiling
                # subtraction, weighted aggregate or confidence is inferred.
                fields = ("case_id", "method", "event_placement", "mean_ms", "iters", "graph",
                          "checked_measured_calls", "checked_warmup_calls", "every_timed_call_checked",
                          "same_call_checks_passed")
                row["raw_benchmarks"].append({"ref": artifact["archive_ref"],
                    "cases": [{key: case[key] for key in fields if key in case} for case in raw.get("cases", [])]})
        rows.append(row)
    return rows


def harvested_idea(manifest, request):
    """Keep the assigned question separate from the harvest's implementation report.

    A failed or slower realization does not refute the idea. Source-bound
    strategy, selection and failures remain in the linked, unchanged harvest.
    """
    return {"idea_id": request["direction_id"], "statement": request["next_question"],
            "status": "unassessed"}


def retain_source_roles(store, manifest, receipt, native, candidate_id):
    """Archive actual source roles; donor ancestry belongs to every retained role."""
    roles = {}
    for role in ("selected", "faithful", "unfinished", "terminal"):
        row = manifest.get(role)
        if row:
            source = read(row["manifest_ref"])
            need(store.capture_source(row["source_dir"], [x["path"] for x in
                source["files"]]) == row["source_id"], "retained source changed")
            roles[role] = candidate_id if row["source_id"] == receipt["captured_tip_source_id"] else store.candidate(
                native["job_id"], native["measurement_epoch_id"], row["source_id"],
                parents=[receipt["request"]["tip"]["candidate_id"]],
                donors=[item["candidate_id"] for item in receipt["request"]["donors"]])
    return roles


def retain_trial_sources(store, manifest, receipt, native):
    """Index every supplied trial without granting correctness or fitness.

    The external collector authenticates source_archive independently of a
    trial's outcome, including failed and incomplete evaluations. Older
    collectors may supply only a checked observation whose source is already
    in the Store. Raw feedback alone is insufficient to assign a source.
    """
    current_turn = receipt["attempt_id"]
    rows = []
    for index, trial in enumerate(manifest["trials"]):
        observation = trial.get("verified") or trial.get("diagnostic") or {}
        snapshot = trial.get("source_archive")
        source_id = snapshot.get("source_id") if snapshot else observation.get("source_id")
        row = {"trial_index": index, "operation_id": observation.get("operation_id"),
               "source_id": source_id, "candidate_id": None,
               "feedback_ref": trial.get("archive_ref"), "source_retention_error": None,
               "feedback_error": None, "owner_turn_id": observation.get("turn_id")}
        feedback = None
        try:
            if row["feedback_ref"] is not None:
                feedback = read(row["feedback_ref"])
                owner = feedback.get("turn_id")
                need(row["owner_turn_id"] is None or row["owner_turn_id"] == owner,
                     "trial observation and feedback owners differ")
                row["owner_turn_id"] = owner
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row["feedback_error"] = type(exc).__name__ + ": " + str(exc)
        try:
            need(source_id is not None, "collector supplied no authenticated trial source")
            if snapshot:
                source = read(snapshot["manifest_ref"])
                captured = store.capture_source(snapshot["source_dir"], [item["path"] for item in source["files"]])
                need(captured == source_id, "retained trial source changed")
            else:
                store.source(source_id)  # Checks the manifest and every source blob.
            need(row["owner_turn_id"] is not None and row["owner_turn_id"] == current_turn
                 and row["feedback_error"] is None,
                 "trial owning turn is missing or differs from this attempt; original lineage is not reassigned")
            row["candidate_id"] = store.candidate(
                native["job_id"], native["measurement_epoch_id"], source_id,
                parents=[receipt["request"]["tip"]["candidate_id"]],
                donors=[item["candidate_id"] for item in receipt["request"]["donors"]])
        except StateError:
            raise  # Corrupt archive state is not a missing collector snapshot.
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row["source_retention_error"] = type(exc).__name__ + ": " + str(exc)
        try:
            # A broken measurement association must not discard exact source
            # bytes. Keep it explicit and withhold the operation ID on conflict.
            need(observation.get("source_id", source_id) == source_id,
                 "trial observation and source archive differ")
            if feedback is not None:
                need(feedback.get("source_id") == source_id, "trial feedback and retained source differ")
                operation_id = feedback.get("operation_id")
                need(row["operation_id"] is None or row["operation_id"] == operation_id,
                     "trial feedback and observation operations differ")
                row["operation_id"] = operation_id
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row["operation_id"] = None
            row["feedback_error"] = type(exc).__name__ + ": " + str(exc)
        rows.append(row)
    return {"schema_version": 1, "attempt_id": current_turn, "trials": rows}


class NativeInvestigation:
    """Actual native entry + actual harvest, for a host-prepared opportunity.

    prepare_native(request, prompt_ref, operation_id) returns the six existing
    file refs consumed by inspect_prepared. It is trusted host code and must
    meter preparation, copy only admitted own history, and preserve the original
    campaign deadline. It must not call any model or GPU itself.
    """

    def __init__(self, store, prepare_native, *, began=None, quantities=None,
                 process_runner=None, clock=None, node_executable="node", python_executable="python3"):
        need(callable(prepare_native), "explicit trusted native admission factory required")
        self.store, self.prepare_native = store, prepare_native
        self.began = began
        self.quantities = quantities
        need(process_runner is None or callable(process_runner), "native process runner must be callable")
        self.process_runner = subprocess.run if process_runner is None else process_runner
        self.clock = SystemClock() if clock is None else clock
        self.node_executable, self.python_executable = node_executable, python_executable

    def run(self, request, attempts_root, *, prompt_path, operation_id):
        request = validate_request(request)
        began = self.clock.monotonic() if self.began is None else self.began
        attempt = Path(attempts_root) / operation_id
        attempt.mkdir(mode=0o700)  # Existing reservation ID is also the no-replay fence.
        prompt_ref = write(attempt / "prompt.txt", Path(prompt_path).read_bytes())
        prepared = self.prepare_native(request, prompt_ref, operation_id)
        entry, config, collect = inspect_prepared(prepared, request, prompt_ref=prompt_ref, operation_id=operation_id, clock=self.clock)
        need(Path(config["store_root"]) == self.store.root, "native and outer source stores must be identical")
        write(attempt / "native_prepared.json", encoded(prepared))
        output = attempt / "native_entry_result.json"
        command = [self.node_executable, prepared["entry_runner_ref"]["path"], "--input", prepared["entry_input_ref"]["path"],
                   "--backend", prepared["backend_binding_ref"]["path"], "--output", str(output)]
        # The native host's absolute clocks, CPU envelope, wait_with_probes and
        # finish_after_worker own termination/cleanup. An outer timeout must not
        # strand that host while its native operation is still being settled.
        with (attempt / "native_entry.stdout").open("xb") as stdout, (attempt / "stderr.txt").open("xb") as stderr:
            completed = self.process_runner(command, cwd=request["workspace"], stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, shell=False)
        need(completed.returncode == 0 and output.stat().st_size > 0,
             "native entry failed; reconcile the exact native receipt, never retry")
        result = read(file_ref(output))
        native_ref = result["transport_result"]["receipt_ref"]
        native = read(native_ref)
        write(attempt / "native_receipt.json", _read_ref(native_ref))
        events = Path(native_ref["path"]).parent / "events.jsonl"
        if events.is_file():
            write(attempt / "native_events.jsonl", events.read_bytes())
        receipt = transport_envelope(request, native, native_ref, operation_id=operation_id,
            command=command, elapsed_seconds=self.clock.monotonic() - began)
        ref = write(attempt / "receipt.json", encoded(receipt))
        return {"attempt_dir": str(attempt), "receipt": receipt, "receipt_ref": ref}

    def collect(self, result, candidate_id):
        """Reuse saved local native trials; never append a post-turn evaluation."""
        began_ns = self.clock.monotonic_ns()
        attempt = Path(result["attempt_dir"])
        receipt = result["receipt"]
        native = read(receipt["native_receipt_ref"])
        prepared = read(file_ref(attempt / "native_prepared.json"))
        costs = {"wall_seconds": receipt["elapsed_seconds"],
                 "evaluations": len(native["native_results"]) if receipt["settlement"].get("settled") else None}
        if not receipt["settlement"].get("settled") or receipt["captured_tip_source_id"] is None:
            return {"costs": costs, "artifacts": {"native_receipt.json": encoded(native)},
                    "card": None, "exchange_receipt": None}
        need(read(result["receipt_ref"]) == receipt, "collection receipt differs from saved bytes")
        inspect_prepared_identity(prepared, receipt["request"],
            prompt_ref=file_ref(attempt / "prompt.txt"), operation_id=receipt["attempt_id"])
        candidate = self.store.get_candidate(candidate_id)
        need(all(candidate[key] == value for key, value in {
            "job_id": native["job_id"], "measurement_epoch_id": native["measurement_epoch_id"],
            "source_id": receipt["captured_tip_source_id"]}.items()), "collection candidate differs")
        binding = {"entry_result_ref": file_ref(attempt / "native_entry_result.json"),
            "entry_input_ref": prepared["entry_input_ref"], "entry_provenance_ref": prepared["entry_provenance_ref"]}
        inputs = {"receipt_ref": result["receipt_ref"], "prepared_ref": file_ref(attempt / "native_prepared.json"),
                  "entry_result_ref": binding["entry_result_ref"], "candidate_id": candidate_id,
                  "boot_id": self.clock.boot_id()}
        intent_path = attempt / "collection_intent.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent = read(file_ref(intent_path))
            need(set(intent) == set(inputs) | {"started_monotonic_ns"}
                 and all(intent[key] == value for key, value in inputs.items()),
                 "collection retry changed its saved inputs or boot")
            began_ns = intent["started_monotonic_ns"]
            need(type(began_ns) is int and 0 <= began_ns <= self.clock.monotonic_ns(),
                 "invalid original collection clock")
        else:
            previous = (attempt / "native_entry_binding.json").exists() or any(
                path.name == "collection" or path.name.startswith("collection-")
                or path.name in ("collection.stdout", "collection.stderr", "collection.status.json")
                for path in attempt.iterdir())
            need(not previous, "saved collection lacks its original intent; owner reconciliation required")
            intent = dict(inputs, started_monotonic_ns=began_ns)
            write_or_match(intent_path, encoded(intent))
        binding_ref = write_or_match(attempt / "native_entry_binding.json", encoded(binding))
        # Re-run only the pinned CPU collector, including all its validators.
        # Partial or complete earlier output is preserved, never inferred successful.
        revision = 0
        while True:
            name = "collection" if revision == 0 else "collection-%04d" % revision
            destination = attempt / name
            stdout_path, stderr_path = attempt / (name + ".stdout"), attempt / (name + ".stderr")
            status_path = attempt / (name + ".status.json")
            paths = (destination, stdout_path, stderr_path, status_path)
            need(not any(path.is_symlink() for path in paths), "unsafe saved collection path")
            if not any(path.exists() for path in paths):
                break
            revision += 1
        command = [self.python_executable, "-I", "-B", prepared["collector_ref"]["path"],
            "--config", prepared["collection_config_ref"]["path"], "--native-entry-binding", binding_ref["path"],
            "--output", str(destination)]
        _read_ref(prepared["collector_ref"])
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = self.process_runner(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), shell=False)
        status = {"command": command, "returncode": completed.returncode,
                  "collection_intent_ref": file_ref(intent_path)}
        status_ref = write_or_match(status_path, encoded(status))
        need(completed.returncode == 0, "collection failed; retain collecting phase for same-ID reconciliation")
        manifest = read(file_ref(destination / "archive/manifest.json"))
        need(manifest["terminal"]["source_id"] == receipt["captured_tip_source_id"], "harvest terminal differs")
        artifacts = {"native_idea_harvest.json": encoded(manifest), "native_receipt.json": encoded(native)}
        artifacts["native_collection_attempt.json"] = encoded({"revision": revision,
            "intent_ref": status["collection_intent_ref"], "intent": intent,
            "status_ref": status_ref, "status": status,
            "manifest_ref": file_ref(destination / "archive/manifest.json")})
        observations = observation_context(manifest)
        # This is an evidence view, not a replacement claim schema or verifier.
        # Scientific claims remain unverified; the separate quantity index has exact scope.
        artifacts["native_idea_context.json"] = encoded({
            "binding": manifest["binding"], "selected": manifest.get("selected"),
            "faithful": manifest.get("faithful"), "unfinished": manifest.get("unfinished"),
            "terminal": manifest["terminal"],
            "native_observations": observations,
            "explanation_status": manifest["explanation"]["explanation_status"],
            "explanation_errors": manifest["explanation"].get("errors", []),
            "raw_explanation_ref": file_ref(destination / "archive/manifest.json"),
            "instruction": "Engineer strategy, notes and causal stories are unverified claims. Read exact raw methods, "
                "source bindings and supplied corrections. Use existing ClaimLog scientific_claim records for review; "
                "Only a listed native_quantity_index structured timing scalar has host producer authority. "
                "No copied prose becomes a fact by appearing in this archive. "
                "A native pass or generic verified status records that check's outcome. Read retained warnings and "
                "unresolved semantic/source constraints separately; such a status cannot authorize eligible fitness. "
                "Per-case timings are separate native development observations, not independently qualified endpoint "
                "fitness or a noise floor. Missing selected-source diagnostics remain unknown."})
        # Raw selected, faithful, unfinished, terminal and alternate sources remain in the
        # same content-addressed store. Only the investigation is a direction.
        roles = retain_source_roles(self.store, manifest, receipt, native, candidate_id)
        artifacts["native_source_roles.json"] = encoded(roles)
        trials = retain_trial_sources(self.store, manifest, receipt, native)
        artifacts["native_trial_sources.json"] = encoded(trials)
        # Publish the authenticated collector tree into the EXISTING evidence
        # store, so the next planner's normal evidence-group traversal reaches
        # the real report, exact selected/faithful code and raw diagnostics.
        # Preserve bytes and original path references; the delivery index maps
        # those historical paths to the current immutable evidence objects.
        raw_artifacts, original_files = {}, []
        archive_root = destination / "archive"
        for path in sorted(archive_root.rglob("*")):
            need(not path.is_symlink(), "collector archive contains a symlink")
            if path.is_file():
                name = "archive/" + path.relative_to(archive_root).as_posix()
                raw_artifacts[name] = path.read_bytes()
                original_files.append({"original_ref": file_ref(path), "artifact": name})
        raw_artifacts["native_source_roles.json"] = encoded(roles)
        raw_artifacts["native_trial_sources.json"] = encoded(trials)
        raw_id = self.store.archive_evidence(candidate_id, raw_artifacts)
        files = [{"original_ref": row["original_ref"],
                  "evidence_ref": {"evidence_id": raw_id, "artifact": row["artifact"]}}
                 for row in original_files]
        path_map = {row["original_ref"]["path"]: row["artifact"] for row in original_files}
        for trial in manifest["trials"]:
            pairs = [(trial.get("feedback_ref"), trial.get("archive_ref"))]
            pairs += [(item.get("input_ref"), item.get("archive_ref")) for item in trial["artifacts"]]
            for original, retained in pairs:
                if original and retained and retained["path"] in path_map:
                    files.append({"original_ref": {key: original[key] for key in ("path", "sha256")},
                        "evidence_ref": {"evidence_id": raw_id, "artifact": path_map[retained["path"]]}})
        artifacts["native_evidence_index.json"] = encoded(files)
        if self.quantities is not None:
            artifacts["native_quantity_index.json"] = encoded(self.quantities.publish(
                manifest, roles, files, operation_id=receipt["attempt_id"]))
        card = {"idea": harvested_idea(manifest, receipt["request"]),
            "observation_refs": [{"evidence_id": raw_id, "artifact": "archive/manifest.json"}]}
        if roles.get("faithful"):
            evidence_id = self.store.archive_evidence(roles["faithful"], {"native_idea_harvest.json": encoded(manifest)})
            card["last_faithful"] = {"candidate_id": roles["faithful"],
                "claim": "Engineer-designated faithful source; semantic fidelity and explanation remain unreviewed.",
                "evidence_refs": [{"evidence_id": evidence_id, "artifact": "native_idea_harvest.json"}]}
        costs["wall_seconds"] += (self.clock.monotonic_ns() - began_ns) / 1e9
        # Omitting research_update intentionally invokes EvolutionLoop's existing
        # pending_review gate. The next metered planner reviews exact raw outcome
        # + corrections through research.claim_update and ClaimLog.
        return {"costs": costs, "artifacts": artifacts, "card": card, "exchange_receipt": None}
