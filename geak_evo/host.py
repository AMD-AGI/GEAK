"""Compose the cleaned search with the execution services already owned by GEAK.

The caller admits the task, installs and verifies runtime services, opens the
store (including its frontier), and supplies the original clock and templates.
This module neither discovers imports nor allocates devices. The external lane
still owns final selection, paired measurement, export and workload validation.
"""

from dataclasses import dataclass
import copy
import math
from pathlib import Path

from ._util import require
from .archive.cells import NativeCellProjector
from .archive.episodes import NativeEvidence
from .archive.islands import group_for_policy
from .archive.policy import native_enabled, program_lease_bound, validate_policy
from .archive.refs import file_ref, read_ref
from .archive.succession import episode_review
from .clock import SystemClock
from .engineer.assignment import NativeAssignment
from .engineer.investigation import encoded, write
from .evaluate.evaluator import ArenaEvaluator
from .evaluate.jobs import job_digest, validate_arena_job
from .evaluate.quantities import NativeQuantities, allocation
from .loop import EvolutionLoop, PROMPT_DIR
from .model_profile import ModelProfile
from .planner import model
from .planner.native import ENGINEER_RESERVATION, NativePlanner, planner_model_profile
from .selection.advice import (NativeAdviceInputs, NativeAdviceReview,
                              NativeDevelopmentAdvice, native_advice_config)


def _document(reference):
    return model.decode(read_ref(reference))


@dataclass(frozen=True)
class HostServices:
    """Existing, admitted services; constructing this value starts no work.

    ``seal_operation`` is the owner's existing publication channel. The driver
    and collector references point to that same native host installation.
    ``model_runtime`` supplies its qualified route, process boundary and events.
    """

    model_runtime: model.ModelRuntime
    workspace: object
    harvest: object
    episode_material: object
    seal_operation: object
    collector_ref: dict
    backend_driver_ref: dict


def make_native_evaluator(store, config, *, callbacks):
    """Bind actual NativeCallbacks.launch/settle to the unchanged Arena protocol.

    The owner creates NativeCallbacks with its existing helper, admission and
    wait bound. Recovery passes None to settle, through ArenaEvaluator, and
    never calls launch again. Request/receipt/source/device checks stay there.
    """
    require(callable(getattr(callbacks, "launch", None))
            and callable(getattr(callbacks, "settle", None)),
            "existing owned native callbacks required", ValueError)
    probe = config["backend_config"]["probe_config"]
    return ArenaEvaluator(store, contract=_document(config["contract_ref"]),
        contract_id=probe["job_id"], artifact_root=config["artifact_root"],
        output_root=config["evaluation_root"], measurement_epoch_id=probe["measurement_epoch_id"],
        launch=callbacks.launch, settle=callbacks.settle)


class NativeSearch:
    """A prepared search. Each run has an explicit decision bound.

    Interrupted planner/engineer operations remain pending in the loop. Their
    existing recovery APIs must authenticate the same operation before another
    run. A returned snapshot is search history, not a measured lane result.
    """

    def __init__(self, loop, planner, assignment, *, profile, planner_seconds,
                 planner_decision_limit, advice, advice_inputs, after_review):
        self.loop, self.planner, self.assignment = loop, planner, assignment
        self.profile = profile
        self.planner_seconds, self.planner_decision_limit = planner_seconds, planner_decision_limit
        self.advice, self.advice_inputs, self.after_review = advice, advice_inputs, after_review

    def run(self, *, max_steps):
        self.profile.verify()
        state = self.loop.snapshot()["state"]
        require(state["phase"] in ("idle", "planned", "stopped"),
                "pending operation needs same-ID reconciliation before search", ValueError)
        if state["phase"] == "stopped":
            return self.loop.snapshot()
        return self.loop.run(self.planner, self.assignment,
            planning_reservation={"wall_seconds": self.planner_seconds + 60, "evaluations": 0},
            planning_seconds=self.planner_seconds + 60, advice_config=self.advice,
            max_steps=max_steps, planner_decision_limit=self.planner_decision_limit,
            advice_inputs=self.advice_inputs, after_review=self.after_review)


def prepare_native_search(store, campaign_id, seed_candidate_id, *, campaign_root,
                          templates, clock_ref, search_policy, shared_budget, profile,
                          services, case_id, evaluation_limit, planner_seconds,
                          planner_decision_limit, sealed_root, root_state_root,
                          common_read_roots, common_refs=(), common_tool_addendum="",
                          review_objective="ordinary", clock=None, prompt_dir=None):
    """Compose one member from the existing config/admission/entry/collection refs.

    This is the callable replacement for the construction inside the old
    coordinator.host. The surrounding owner keeps task adoption, frontier
    registration, sealing, population selection and final completion unchanged.
    New campaigns persist their exact profile and original limits. Opening an
    old campaign without this binding requires a separately reviewed migration.
    """
    require(isinstance(services, HostServices) and isinstance(profile, ModelProfile),
            "explicit host services and model profile required", ValueError)
    require(isinstance(services.model_runtime, model.ModelRuntime),
            "existing model runtime services required", ValueError)
    require(set(templates) == {"config", "admission", "entry", "collection"},
            "existing four native template references required", ValueError)
    docs = {key: _document(ref) for key, ref in templates.items()}
    config, admission, entry = docs["config"], docs["admission"], docs["entry"]
    probe = config["backend_config"]["probe_config"]
    contract = validate_arena_job(_document(config["contract_ref"]), config["artifact_root"])
    require(job_digest(contract) == probe["job_id"], "native task identity differs", ValueError)
    seed = store.get_candidate(seed_candidate_id)
    require(seed["job_id"] == probe["job_id"]
            and seed["measurement_epoch_id"] == probe["measurement_epoch_id"]
            and entry["source_paths"] == contract["source_paths"],
            "seed, task, epoch or editable source closure differs", ValueError)
    allocation(admission)  # Timing authority needs the full admitted physical identity.
    require(callable(getattr(services.workspace, "fresh_workspace", None))
            and callable(services.episode_material) and callable(services.seal_operation),
            "existing workspace, episode and owner publication services required", ValueError)
    for ref in (services.collector_ref, services.backend_driver_ref):
        read_ref(ref)
    root = Path(campaign_root)
    require(root.is_absolute() and root.resolve(strict=True) == root
            and root in store.root.resolve().parents,
            "campaign and store must have the same canonical root", ValueError)
    policy = validate_policy(search_policy, root, require_native_capabilities=True)
    prompt_dir = Path(PROMPT_DIR if prompt_dir is None else prompt_dir)
    prompt_names = ["planner.md", "engineer.md", "crossover.md", "program_lease_engineer.md"]
    if program_lease_bound(policy):
        prompt_names.append("program_lease_planner.md")
    prompt_refs = {name: file_ref(prompt_dir / name) for name in prompt_names}
    clock = clock or services.model_runtime.clock or SystemClock()
    original = _document(clock_ref)
    require(type(original["wall_seconds"]) in (int, float)
            and math.isfinite(original["wall_seconds"]) and original["wall_seconds"] > 0
            and type(original["started_monotonic_ns"]) is int,
            "finite original clock duration required", ValueError)
    deadline_ns = original["started_monotonic_ns"] + int(original["wall_seconds"] * 1e9)
    # The existing native clock derives this value; projected owner clocks may
    # additionally carry it. An explicit value may not change that same bound.
    require(original.get("deadline_ns", deadline_ns) == deadline_ns,
            "original clock duration and absolute deadline differ", ValueError)
    require(original["boot_id"] == clock.boot_id()
            and original["started_monotonic_ns"] <= clock.monotonic_ns() < deadline_ns,
            "original clock is expired or belongs to another boot", ValueError)
    require(type(planner_decision_limit) is int and planner_decision_limit > 0,
            "original total planner decision ceiling required", ValueError)
    model_config = copy.deepcopy(config["backend_config"]["model_config"])
    require(model_config["clock_path"] == clock_ref["path"], "model original clock differs", ValueError)
    if profile.workflow_backend == "codex":
        for key, value in {"tool_audit_ref": config["tool_audit_ref"],
                           "codex_native_root": config["imports"]["codex_native_root"]}.items():
            require(model_config.get(key, value) == value, "model authority differs: " + key, ValueError)
            model_config[key] = value
    model.validate_runtime_dependencies(model_config, runtime=services.model_runtime)
    profile.verify()
    require(profile == planner_model_profile(model_config, runtime=services.model_runtime),
            "planner and engineer model profiles differ", ValueError)
    descriptor_factory = (services.model_runtime.transport.derive_descriptor
                          if "codex_runtime_family_ref" in model_config else None)
    now = lambda: original["started_unix"] + (clock.monotonic_ns() - original["started_monotonic_ns"]) / 1e9
    deadline = original["started_unix"] + original["wall_seconds"]
    limits = {"wall_seconds": original["wall_seconds"], "evaluations": evaluation_limit}
    packaged_entry = Path(__file__).parent / "engineer" / "js"
    runtime_refs = {"entry_runner_ref": file_ref(packaged_entry / "run_one_engineer.mjs"),
        "entry_provenance_ref": file_ref(packaged_entry / "entry-provenance.json"),
        "collector_ref": services.collector_ref}
    binding = {"campaign_id": campaign_id, "seed_candidate_id": seed_candidate_id,
        "model_profile": profile.as_dict(), "clock_ref": clock_ref, "templates": templates,
        "limits": limits, "planner_decision_limit": planner_decision_limit,
        "planner_seconds": planner_seconds, "search_policy": policy,
        "shared_budget": shared_budget, "review_objective": review_objective,
        "case_id": case_id, "common_refs": list(common_refs),
        "common_read_roots": list(map(str, common_read_roots)), "common_tool_addendum": common_tool_addendum,
        "sealed_root": str(sealed_root), "root_state_root": str(root_state_root),
        "runtime_refs": runtime_refs, "host_ref": file_ref(Path(__file__)),
        "prompt_refs": prompt_refs,
        "model_backend_ref": file_ref(Path(model.__file__)), "backend_driver_ref": services.backend_driver_ref}
    binding_path = root / "host-binding.json"
    existing = campaign_id in store.get()
    if binding_path.exists():
        require(_document(file_ref(binding_path)) == binding, "saved host binding changed", ValueError)
    else:
        require(not existing, "saved campaign has no shared host binding", ValueError)
        write(binding_path, encoded(binding))
    quantities = NativeQuantities(store, campaign_id, contract=contract, admission=admission,
        case_id=case_id, producer_ref=file_ref(Path(__file__).with_name("evaluate") / "quantities.py"))
    options = dict(clock=now, monotonic=clock.monotonic, verify_quantity=quantities,
                   program_lease=program_lease_bound(policy), prompt_dir=prompt_dir)
    loop = (EvolutionLoop(store, campaign_id, **options) if existing else
            EvolutionLoop.create(store, campaign_id, seed_candidate_id, limits=limits,
                                 deadline=deadline, **options))
    initial = loop.snapshot()
    require(initial["state"]["limits"] == limits and initial["state"]["deadline"] == deadline,
            "saved campaign bounds changed", ValueError)
    scratch = root / "native-evidence"
    scratch.mkdir(mode=0o700, exist_ok=True)
    evidence = NativeEvidence(loop, root, harvest=services.harvest,
        episode_material=services.episode_material, scratch_root=scratch, runtime_refs=runtime_refs)
    cells = (NativeCellProjector(store, episode_material=evidence.episode, episode_review=episode_review)
             if native_enabled(policy, "cells") else None)
    islands = group_for_policy(policy, root, probe["job_id"])

    def bindings(candidate_id):
        candidate = store.get_candidate(candidate_id)
        attempt = {key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")}
        attempt.update(case_id=case_id, inputs_id=contract["case_bindings"][case_id]["sha256"],
            configuration_id=config["contract_ref"]["sha256"], diagnostic_id=config["helper_ref"]["sha256"])
        return quantities.binding(candidate_id), attempt

    assignment = NativeAssignment(loop, campaign_root=root,
        job={"path": config["contract_ref"]["path"], "job_id": probe["job_id"], "artifact_root": config["artifact_root"]},
        workspace_helper=services.workspace, templates=templates, clock_ref=clock_ref,
        shared_budget=shared_budget, common_read_roots=common_read_roots, search_policy=policy,
        sealed_root=sealed_root, root_state_root=root_state_root, seal_operation=services.seal_operation,
        bindings=bindings, read_only=lambda snapshot: list(common_refs), parent_material=lambda proposal, snapshot: [],
        native_collector_ref=services.collector_ref, native_evidence=evidence,
        common_tool_addendum=common_tool_addendum, island_group=islands,
        descriptor_factory=descriptor_factory, clock=clock, backend_driver_ref=services.backend_driver_ref)
    capabilities = store.archive_evidence(seed_candidate_id, {
        "Host capability templates/native-host-config.json": read_ref(templates["config"]),
        "Host capability templates/native-entry.json": read_ref(templates["entry"]),
        "Host capability templates/original-clock.json": read_ref(clock_ref)})
    planner = NativePlanner(loop, root / "planner", model_backend=model, model_config=model_config,
        runtime=services.model_runtime, clock=clock, clock_ref=clock_ref, read_only=common_refs,
        evidence_refs=[{"evidence_id": capabilities, "artifact": "Host capability templates/native-host-config.json"}],
        search_policy=policy, campaign_root=root, wall_seconds=planner_seconds,
        native_envelope_seconds=probe["native_envelope_seconds"],
        current_measurement=entry.get("kb_inputs", {}).get("CURRENT_MEASUREMENT"),
        native_cells=cells, island_group=islands, native_evidence=evidence,
        review_objective=review_objective, descriptor_factory=descriptor_factory,
        program_lease_prompt_ref=prompt_refs.get("program_lease_planner.md"))
    advice = {"weights": {"wall_seconds": 0.5, "evaluations": 0.5}, "c_ref": 1,
              "half_life": 1, "epsilon": 0.01, "beta": 0}
    advice_inputs, progress = None, evidence.record_reviewed_progress
    if policy["arm"] == "EVO" and native_enabled(policy, "puct"):
        development = NativeDevelopmentAdvice(loop, evidence)
        advice = native_advice_config(limits, ENGINEER_RESERVATION)
        advice_inputs = NativeAdviceInputs(loop, development, clock_ref=clock_ref,
            native_envelope_seconds=probe["native_envelope_seconds"], engineer_limit=ENGINEER_RESERVATION)
        progress = NativeAdviceReview(loop, development)

    def after_review(snapshot):
        if islands is not None:
            islands.publish(store, campaign_id)
        progress(snapshot)

    return NativeSearch(loop, planner, assignment, profile=profile, planner_seconds=planner_seconds,
        planner_decision_limit=planner_decision_limit, advice=advice,
        advice_inputs=advice_inputs, after_review=after_review)
