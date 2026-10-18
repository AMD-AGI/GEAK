"""Host-assignment wiring for the existing EvolutionLoop.run callback.

Uses a fresh native engineer per idea and the existing metered research_review
action between settled investigations. No new planner or search algorithm.
"""
import copy
import json
from pathlib import Path

from ..clock import SystemClock
from ..archive.refs import source_reference, file_ref
from ..archive.succession import archive_files
from .investigation import NativeInvestigation, flat_action, need, read
from .operations import build_files, prepare_workspace


QUANTITY_CATALOG_BYTES = 16 * 1024


def quantity_catalog_text(catalog, limit=QUANTITY_CATALOG_BYTES):
    """Catalog JSON for the outer assignment: the recent suffix of each stored list within ``limit`` bytes.

    Every row stays in its collection's native_quantity_index.json (``index_refs``);
    the ordinary read grant publishes those collections, listed in READ_INDEX.
    """
    text = json.dumps(catalog, sort_keys=True)
    if len(text.encode()) <= limit:
        return text
    lists = ("index_refs", "quantities", "unavailable")
    view = dict(catalog, **{key: [] for key in lists}, inline_bound={"bytes": limit,
        "totals": {key: len(catalog[key]) for key in lists},
        "omitted": "earlier rows in stored order; each is in a native_quantity_index.json named by the full index_refs, "
                   "whose collection evidence is in READ_INDEX"})
    used = len(json.dumps(view, sort_keys=True).encode())
    pending = {key: len(catalog[key]) - 1 for key in lists if catalog[key]}
    while pending:
        for key in tuple(pending):
            item = catalog[key][pending[key]]
            size = len(json.dumps(item, sort_keys=True).encode()) + 2
            if used + size > limit:
                del pending[key]
                continue
            view[key].insert(0, item)
            used += size
            pending[key] -= 1
            if pending[key] < 0:
                del pending[key]
    text = json.dumps(view, sort_keys=True)
    need(len(text.encode()) <= limit, "bounded quantity catalog exceeds its inline bound")
    return text


def reservation_preflight(policy, snapshot, proposal):
    """Check the actual admission on a private copy; never recredit a snapshot.

    An existing E2 is already held outside free allowance. Use the same pure
    transition as the loop and publisher so only its exact reviewed direction
    can draw it. The real opportunity remains uncreated until loop admission.
    """
    from .. import budget as program_lease
    from ..archive.policy import program_lease_bound

    need(all(value > 0 for value in proposal["reservation"].values()),
         "native opportunity requires positive resource allowances")
    state = snapshot["state"]
    bound = program_lease_bound(policy)
    if bound and ("lease" in proposal or program_lease.active_lease(state) is not None):
        need(state["phase"] == "planned" and state["pending"]["proposal"] == proposal,
             "commitment preflight requires the exact selected planner proposal")
        row = dict(state["binding"],
            opportunity_id="lease-preflight:" + state["pending"]["operation_id"],
            direction_id=proposal["direction_id"], island_id=proposal["island_id"],
            engineer_id=proposal["engineer_id"], question=proposal["question"],
            opened_at=snapshot["accounting"]["now"], review_at=proposal["review_at"],
            requested=copy.deepcopy(proposal["reservation"]), reserved=copy.deepcopy(proposal["reservation"]),
            receipts=[], settled=False, completed_sequence=None, progress_complete=False)
        program_lease.admit_engineer(copy.deepcopy(state), proposal, row, bound=bound, now=row["opened_at"])
    else:
        need(all(value <= snapshot["accounting"]["available"][key]
                 for key, value in proposal["reservation"].items()),
             "native opportunity exceeds remaining campaign allowance")


def selected_immigrants(immigrants, proposal):
    """Group immigrants named by this proposal and their local import evidence and copies."""
    chosen = [identity for identity in [proposal["parent_candidate_id"]] + proposal.get("semantic_donors", [])
              if identity in immigrants]
    return chosen, [identity for candidate in chosen for record in immigrants[candidate]
                    for identity in [record["evidence_id"], *record["origin_evidence_map"].values()]]

class NativeAssignment:
    def __init__(self, loop, *, campaign_root, job, workspace_helper, templates,
                 clock_ref, shared_budget, common_read_roots, search_policy,
                 sealed_root, root_state_root, seal_operation, bindings,
                 read_only, parent_material, native_entry_runner_ref=None,
                 native_collector_ref, native_entry_provenance_ref=None, native_evidence=None,
                 common_tool_addendum="", island_group=None, descriptor_factory=None,
                 clock=None, process_runner=None, node_executable="node", python_executable="python3",
                 backend_driver_ref=None):
        from ..archive.policy import validate_policy

        search_policy = validate_policy(search_policy, campaign_root, require_native_capabilities=True)
        self.loop, self.store = loop, loop.store
        self.root, self.job = Path(campaign_root), job
        self.workspace_helper, self.templates = workspace_helper, templates
        self.clock_ref, self.shared_budget = clock_ref, shared_budget
        self.common_read_roots, self.search_policy = common_read_roots, search_policy
        self.sealed_root, self.root_state_root = Path(sealed_root), root_state_root
        self.seal_operation, self.bindings = seal_operation, bindings
        self.read_only, self.parent_material = read_only, parent_material
        packaged_entry = Path(__file__).resolve().with_name("js")
        self.entry_runner = (file_ref(packaged_entry / "run_one_engineer.mjs")
                             if native_entry_runner_ref is None else native_entry_runner_ref)
        self.collector = native_collector_ref
        self.provenance = (file_ref(packaged_entry / "entry-provenance.json")
                           if native_entry_provenance_ref is None else native_entry_provenance_ref)
        self.native_evidence = native_evidence
        verifier = getattr(loop, "verify_quantity", None)
        self.quantities = verifier if callable(getattr(verifier, "catalog", None)) else None
        self.common_tool_addendum = common_tool_addendum
        self.island_group = island_group
        self.descriptor_factory = descriptor_factory
        self.clock = SystemClock() if clock is None else clock
        self.process_runner = process_runner
        self.node_executable, self.python_executable = node_executable, python_executable
        self.backend_driver_ref = backend_driver_ref
        (self.root / "attempts").mkdir(mode=0o700, exist_ok=True)

    def __call__(self, proposal, snapshot):
        from ..archive.policy import native_enabled, validate_action
        from ..archive.cells import validate_cell_action
        from ..archive.succession import (episode_review, parent_outcomes, parent_prompt_context,
                                         selected_parent_material)
        from .research import claim_update

        began = self.clock.monotonic()
        flat_action(proposal, self.search_policy)
        validate_action(self.search_policy, snapshot, proposal, store=self.store)
        if proposal.get("kind") == "lease_release":
            return {}  # The loop checks and commits release; no native work is prepared.
        validate_cell_action(self.store, snapshot, proposal,
            episode_material=self.native_evidence.episode if self.native_evidence is not None else None,
            episode_review=episode_review)
        if proposal.get("kind") == "research_review":
            return {"research_update": claim_update(self.store, proposal, snapshot, self.bindings,
                                                    verify_quantity=getattr(self.loop, "verify_quantity", None))}
        if proposal.get("kind") == "research_action":
            need(proposal["action"] != "accept_anchor" or callable(self.store.verify_evidence),
                 "validated anchor verifier is not configured")
            return {"binding": self.bindings(proposal["candidate_id"])[0]}
        if proposal.get("kind") == "maturation_review":
            need(self.native_evidence is not None, "native observation adapter is not configured")
            return self.native_evidence.review_assignment(snapshot, proposal)
        direction = self.store.get().get(proposal["direction_id"])
        prior = snapshot["state"]["latest_receipts"].get(proposal["direction_id"])
        if direction is None:
            need(proposal["expected_revision"] == 0 and prior is None, "fresh direction has prior state")
            continuation, predecessor = {"mode": "fresh", "prior_receipt": None}, None
        else:
            need(native_enabled(self.search_policy, "continuation"), "continuation is disabled by the registered native policy")
            need(self.search_policy is not None and self.search_policy["arm"] == "EVO"
                 and proposal["expected_revision"] == direction["revision"]
                 and proposal["parent_candidate_id"] == direction["tip"] and prior is not None,
                 "native continuation requires the current direction, tip and settled receipt")
            continuation = {"mode": "handoff", "prior_receipt": prior}
            previous = read(prior)
            predecessor = previous["attempt_id"]
            histories = [row for row in snapshot["state"]["history"] if row.get("attempt_id") == predecessor]
            need(len(histories) == 1 and histories[0].get("settled") is True
                 and histories[0]["candidate_id"] == direction["tip"] and self.native_evidence is not None,
                 "native continuation needs its exact committed terminal episode")
            checked, _ = self.native_evidence.episode(snapshot, histories[0])
            need(checked == previous, "native continuation predecessor differs from authenticated episode")
        clock = read(self.clock_ref)
        template = read(self.templates["config"])
        need(type(proposal["reservation"]["evaluations"]) is int
              and proposal["reservation"]["evaluations"] > 0,
             "reserve a positive whole native request allowance")
        reservation_preflight(self.search_policy, snapshot, proposal)
        # Translate the reserved review time with the supplied campaign clock.
        # Cleanup remains inside that reservation.
        review_ns = clock["started_monotonic_ns"] + int((proposal["review_at"] - clock["started_unix"]) * 1e9)
        search_end_ns = clock["started_monotonic_ns"] + int(
            (snapshot["state"]["deadline"] - clock["started_unix"]) * 1e9)
        review_ns = min(review_ns, search_end_ns,
                        clock["started_monotonic_ns"] + int(clock["wall_seconds"] * 1e9))
        envelope_ns = int(template["backend_config"]["probe_config"]["native_envelope_seconds"] * 1e9)
        need(self.clock.monotonic_ns() + envelope_ns < review_ns - 30_000_000_000,
             "fixed native opportunity cannot fit its native envelope before workspace preparation")
        parent = proposal["parent_candidate_id"]
        # A selected/faithful source can differ from a direction's terminal tip.
        # Review all exact native roles before workspace preparation or dispatch.
        parent_outcomes(self.store, snapshot, parent)
        immigrants = (self.island_group.lookup(self.store, snapshot["state"]["binding"]["measurement_epoch_id"],
                      self.store.get(snapshot["campaign_id"])["tip"]) if self.island_group else {})
        imported, import_evidence = selected_immigrants(immigrants, proposal)
        parents = selected_parent_material(self.store, snapshot, proposal, immigrants)
        extra_parents = self.parent_material(proposal, snapshot)
        need(not extra_parents, "flat succession accepts only explicit agent-selected own-history parents")
        prepared = prepare_workspace(self.store, self.workspace_helper, campaign_root=self.root,
            direction_id=proposal["direction_id"], job_id=self.job["job_id"], epoch=snapshot["state"]["binding"]["measurement_epoch_id"],
            session_id=clock["session_id"], parent=parent, predecessor=predecessor,
            opportunity=snapshot["state"]["pending"]["operation_id"])
        pins = read(prepared["workspace_ref"])
        request = {"schema_version": 1, "direction_id": proposal["direction_id"], "job": self.job,
            "measurement_epoch_id": snapshot["state"]["binding"]["measurement_epoch_id"], "workspace": pins["workspace"],
            "anchor": source_reference(self.store, direction["anchor"]["candidate_id"])
                      if direction and direction["anchor"] else None,
            "tip": source_reference(self.store, parent),
            "donors": [source_reference(self.store, identity) for identity in proposal.get("semantic_donors", [])],
            "read_only": self.read_only(snapshot) + archive_files(self.store, snapshot, [parent] + imported, import_evidence), "next_question": proposal["question"],
            "budget": {"wall_seconds": proposal["reservation"]["wall_seconds"], "max_tokens": None},
            "continuation": continuation}
        def prepare_native(actual, prompt_ref, operation_id):
            sealed = self.sealed_root / operation_id
            files = build_files(templates=self.templates, request=actual, operation_id=operation_id,
                prompt_ref=prompt_ref, workspace_ref=prepared["workspace_ref"], store_root=self.store.root,
                campaign_root=self.root, sealed_operation_root=sealed, clock_ref=self.clock_ref,
                turn_deadline_ns=review_ns - 30_000_000_000, shared_budget=self.shared_budget,
                common_read_roots=self.common_read_roots, parents=parents,
                search_policy=self.search_policy, root_state_root=self.root_state_root,
                common_tool_addendum=self.common_tool_addendum, proposal=proposal,
                descriptor_factory=self.descriptor_factory, clock=self.clock,
                backend_driver_ref=self.backend_driver_ref)
            refs = self.seal_operation(sealed, files)
            return {"entry_runner_ref": self.entry_runner, "entry_input_ref": refs["entry-input.json"],
                "backend_binding_ref": refs["backend-binding.json"], "collection_config_ref": refs["collection-config.json"],
                "collector_ref": self.collector, "entry_provenance_ref": self.provenance}
        runtime = {"clock": self.clock, "node_executable": self.node_executable,
                   "python_executable": self.python_executable}
        if self.process_runner is not None:
            runtime["process_runner"] = self.process_runner
        native = NativeInvestigation(self.store, prepare_native, began=began, quantities=self.quantities, **runtime)
        binding, attempt_binding = self.bindings(parent)
        def context(current):
            need(selected_parent_material(self.store, current, proposal, immigrants) == parents,
                 "selected native parent source or review changed before admission")
            result = parent_prompt_context(self.store, current, parent,
                direction_id=proposal["direction_id"], search_policy=self.search_policy, campaign_root=self.root)
            if self.quantities is not None:
                text, ref = result
                catalog = self.quantities.catalog(current)
                # These exact quantity objects also enter the ordinary read grant below.
                text = "Host-recorded quantity catalog; free-text rationale remains unverified.\n" + quantity_catalog_text(catalog) + "\n" + text
                result = text, ref
            need(self.clock.monotonic_ns() + envelope_ns < review_ns - 30_000_000_000,
                 "fixed native opportunity cannot fit its native envelope before admission")
            return result
        return {"request": request, "binding": binding, "attempt_binding": attempt_binding,
                "run_investigation": native.run, "collect": native.collect, "attempts_root": self.root / "attempts",
                "prompt_context": context}
