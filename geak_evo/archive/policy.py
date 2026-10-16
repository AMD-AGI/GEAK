"""Search-policy checks: which arm (G or EVO) a campaign runs, and which files it may read.

These checks limit host actions and the files offered to an existing sandbox.
They cannot erase a model's prior knowledge, prove the contents of supplied
evidence, or isolate a privileged filesystem writer.

``ProposalError`` marks a fault in an agent's proposal. A bad host policy or
host state raises a plain ``ValueError``.
"""

import copy
import json
from pathlib import Path

from .._util import EvoError, require
from .claims import ClaimError, _ref



class ProposalError(EvoError, ValueError):
    """A deterministic error in an agent proposal, never host/store corruption."""


NATIVE_CAPABILITIES = ("continuation", "cells", "protection", "puct", "program_lease")
POLICY_KEYS = {"arm", "study_root", "shared_read_roots", "native_capabilities", "island_group",
               "program_lease_max_episodes"}
RAW_PROPOSAL_KEYS = {"direction_id", "expected_revision", "parent_candidate_id", "island_id", "engineer_id",
                     "question", "reason", "evidence_uptake", "premises", "repeat_delta", "exchange_ref",
                     "protection_ref", "override_reason", "membership_ref"}
RAW_PROPOSAL_OPTIONAL = {"reservation", "review_at", "grant_intent", "semantic_donors"}
REPEAT_KINDS = ("new_evidence", "resolved_prerequisite", "replication", "new_question")


def _need(condition, message):
    require(condition, message, ValueError)


def proposal_need(condition, message):
    require(condition, message, ProposalError)


def proposal_ref(value):
    """Check only submitted reference syntax; do not catch evidence/store errors."""
    try:
        return _ref(value)
    except ClaimError as exc:
        raise ProposalError(str(exc)) from exc


def _canonical(value, *, exists=True):
    _need(isinstance(value, (str, Path)) and bool(str(value).strip()), "explicit read-scope path required")
    path = Path(value)
    _need(path.is_absolute(), "read-scope paths must be absolute")
    resolved = path.resolve(strict=exists)
    _need(path == resolved, "read-scope paths must be canonical, without symlink aliases")
    return resolved


def _within(path, root):
    return path == root or root in path.parents


def _check_capabilities(policy):
    if "native_capabilities" in policy:
        capabilities = policy["native_capabilities"]
        _need(type(capabilities) is list and all(type(value) is str and value in NATIVE_CAPABILITIES
                                                 for value in capabilities)
              and len(capabilities) == len(set(capabilities)), "invalid enabled native capabilities")
    if "program_lease" in policy.get("native_capabilities", ()) or "program_lease_max_episodes" in policy:
        bound = policy.get("program_lease_max_episodes", 2)
        _need("program_lease" in policy.get("native_capabilities", ()) and policy["arm"] == "EVO"
              and "continuation" in policy["native_capabilities"] and type(bound) is int and bound == 2,
              "program_lease requires EVO continuation and exactly two episodes")


def _check_island_group(policy, study, campaign):
    group = policy["island_group"]
    _need(policy["arm"] == "EVO", "only EVO campaigns may join an island group")
    _need(type(group) is dict and set(group) == {"group_id", "registry_root", "members"}
          and type(group["group_id"]) is str and group["group_id"].strip()
          and type(group["members"]) is list, "island_group requires group_id, registry_root and members")
    registry = _canonical(group["registry_root"])
    members = [_canonical(root, exists=False) for root in group["members"]]
    _need(len(members) >= 2 and len(set(members)) == len(members), "island group needs distinct members")
    _need(registry.is_dir() and _within(registry, study)
          and all(member != study and _within(member, study) and not _within(registry, member)
                  and not _within(member, registry) for member in members),
          "island group members and registry must be disjoint parts of the study root")
    _need(all(not _within(a, b) for a in members for b in members if a != b), "nested island members")
    if campaign is not None:
        _need(campaign in members, "campaign is not a declared island group member")


def validate_policy(policy, campaign_root=None, *, require_native_capabilities=False):
    """Check a policy and return a copy; native entry points need an explicit capability list."""
    if require_native_capabilities:
        _need(type(policy) is dict and "native_capabilities" in policy,
              "native_capabilities must be explicit; use [] to disable all native capabilities")
    if policy is None:
        return None
    _need(type(policy) is dict and {"arm", "study_root", "shared_read_roots"} <= set(policy) <= POLICY_KEYS,
          "search_policy requires arm, study_root and shared_read_roots")
    _need(policy["arm"] in ("G", "EVO"), "search_policy arm must be G or EVO")
    _check_capabilities(policy)
    study = _canonical(policy["study_root"])
    _need(study.is_dir(), "study_root must be an existing directory")
    roots = policy["shared_read_roots"]
    _need(type(roots) is list, "shared_read_roots must be an explicit list")
    shared = [_canonical(root) for root in roots]
    _need(len(set(shared)) == len(shared), "duplicate shared read root")
    _need(all(not _within(root, study) and not _within(study, root) for root in shared),
          "shared read roots must be disjoint from the entire study root")
    campaign = None
    if campaign_root is not None:
        campaign = _canonical(campaign_root, exists=False)
        _need(campaign != study and _within(campaign, study), "campaign must be below its registered study root")
    if "island_group" in policy:
        _check_island_group(policy, study, campaign)
    return copy.deepcopy(policy)


def native_enabled(policy, capability):
    """With no capability list every capability is on (except program_lease); an explicit list is exact."""
    if capability == "program_lease":
        return policy is not None and capability in policy.get("native_capabilities", ())
    return policy is None or "native_capabilities" not in policy or capability in policy["native_capabilities"]


def program_lease_bound(policy):
    """The fixed two-episode bound, or zero without explicit opt-in."""
    return policy.get("program_lease_max_episodes", 2) if native_enabled(policy, "program_lease") else 0


def validate_lease(proposal, bound):
    """Check the agent's finite allocation terms; the ordinary proposal checks still apply."""
    lease = proposal.get("lease")
    proposal_need(type(bound) is int and bound == 2 and type(lease) is dict
                  and set(lease) == {"episodes", "release_condition", "displaced_work"}
                  and type(lease["episodes"]) is int and lease["episodes"] == 2
                  and all(type(lease[key]) is str and lease[key].strip()
                          for key in ("release_condition", "displaced_work")),
                  "program lease requires exactly two episodes, release_condition and displaced_work")


def native_capability_context(policy):
    if policy is None or "native_capabilities" not in policy:
        return ""
    return ("Registered outer native capabilities: " + json.dumps(policy["native_capabilities"]) +
            ". Native capabilities omitted from this list are disabled. Group transfer follows its "
            "separately registered group policy. The native local edit/test/repair loop remains available.\n")


def validate_read_scope(policy, campaign_root, paths):
    """Reject inputs outside the campaign and the shared read roots, before copying or dispatch."""
    if policy is None:
        return
    validate_policy(policy, campaign_root)
    campaign = _canonical(campaign_root, exists=False)
    shared = [_canonical(root) for root in policy["shared_read_roots"]]
    for item in paths:
        path = _canonical(item)
        _need(_within(path, campaign) or any(_within(path, root) for root in shared),
              "read scope excludes sibling campaigns and undeclared inputs: " + str(path))


_SHARED_CAPABILITIES = (
    "Continue implementation, diagnosis and repair across episodes. Retain the valid anchor, "
    "current unfinished or slower tip, exact own source history and notes independently. "
    "You may restore an earlier implementation, change algorithms or IO, revisit an idea and "
    "combine your own local experiments within an ordinary engineer turn. No requirement "
    "that intermediate sources be correct, faster, or the accepted anchor. The common evaluator "
    "still governs correctness and promotion. Use the declared horizon and renewal opportunities; "
    "a short episode or a plateau alone is not an exhausted research budget."
)
_G_TREATMENT = (
    "One serial continuing research line. Use fresh direction IDs only as native episode "
    "records, never as independently allocated alternatives. The first episode starts from "
    "the campaign seed. Each later episode starts from an exact reviewed selected, faithful, unfinished "
    "or terminal source of the immediately preceding episode. All earlier own code, ideas, "
    "failures and evidence remain readable: restore, rederive and compose them freely inside "
    "the native engineer loop. No semantic_donors or portfolio actions. Research review and "
    "explicit stop remain available under the same finite resources. Fresh model contexts "
    "receive retained history; episode IDs do not imply scientific branching."
)
_EVO_TREATMENT = (
    "The continuing engineer may maintain alternative directions and use the existing "
    "island, exchange and finite maturation interfaces under the same total resources. "
    "The agent chooses whether these actions are useful; availability is not benefit."
)
_ISLAND_GROUP_TREATMENT = (
    " This campaign is one concurrent island of a declared group. Reviewed sibling sources "
    "are importable exact code with foreign-epoch observations; only local measurements rate them."
)
_READ_SCOPE_LIMIT = (
    "Only this campaign and the registered common inputs may be granted to workers. "
    "This does not establish model-prior independence or protect against a privileged host writer."
)


def policy_contract(policy):
    """Describe the arm to the agent: shared capabilities plus the arm's treatment text."""
    if policy is None:
        return {"arm": "unregistered", "claim": "No G/EVO policy comparison registered."}
    validate_policy(policy)
    result = copy.deepcopy(policy)
    result["shared_capabilities"] = _SHARED_CAPABILITIES
    if policy["arm"] == "G":
        result["treatment"] = _G_TREATMENT
    else:
        result["treatment"] = _EVO_TREATMENT
        if "island_group" in policy:
            result["treatment"] += _ISLAND_GROUP_TREATMENT
    result["read_scope_limit"] = _READ_SCOPE_LIMIT
    return result


def _check_raw_premises(premises):
    for premise in premises:
        proposal_need(type(premise) is dict and set(premise) == {"claim", "use", "reason", "next_test"},
                      "invalid premise request")
        proposal_ref(premise["claim"])
        proposal_need(premise["use"] in ("fact", "hypothesis"), "unknown premise use")
        if premise["use"] == "hypothesis":
            proposal_need(all(type(premise[field]) is str and premise[field].strip()
                              for field in ("reason", "next_test")), "hypothesis needs reason and next test")


def _check_raw_repeat_delta(delta):
    proposal_need(set(delta) == {"kind", "reason", "expected_observation", "reviewed_attempts", "evidence"}
                  and delta["kind"] in REPEAT_KINDS, "invalid repeat delta")
    proposal_need(all(type(delta[field]) is str and delta[field].strip()
                      for field in ("reason", "expected_observation")),
                  "repeat delta needs reason and expected observation")
    proposal_need(type(delta["reviewed_attempts"]) is list
                  and all(type(item) is str and item.strip() for item in delta["reviewed_attempts"])
                  and type(delta["evidence"]) is list, "invalid repeat delta lists")
    proposal_need(delta["kind"] in ("replication", "new_question") or bool(delta["evidence"]),
                  "new evidence needs exact references")
    for ref in delta["evidence"]:
        proposal_ref(ref)


def validate_raw_engineer_proposal(proposal):
    """The existing scientific-envelope checks, before any source or store read."""
    proposal_need(type(proposal) is dict, "raw engineer proposal must be an object")
    missing = sorted(RAW_PROPOSAL_KEYS - set(proposal))
    unexpected = sorted(set(proposal) - RAW_PROPOSAL_KEYS - RAW_PROPOSAL_OPTIONAL)
    proposal_need(not missing and not unexpected,
                  "invalid raw engineer proposal fields: missing %s, unexpected %s" % (missing, unexpected))
    for key in ("direction_id", "parent_candidate_id", "island_id", "engineer_id", "question", "reason",
                "evidence_uptake"):
        proposal_need(type(proposal[key]) is str and bool(proposal[key].strip()), "invalid raw engineer field: " + key)
    proposal_need(type(proposal["expected_revision"]) is int and proposal["expected_revision"] >= 0
                  and type(proposal["premises"]) is list
                  and (proposal["repeat_delta"] is None or type(proposal["repeat_delta"]) is dict),
                  "invalid raw engineer revision, premises or repeat delta")
    proposal_need(proposal["override_reason"] is None or (type(proposal["override_reason"]) is str
                  and bool(proposal["override_reason"].strip())), "invalid override reason")
    donors = proposal.get("semantic_donors", [])
    proposal_need(type(donors) is list and all(type(item) is str and item for item in donors)
                  and len(donors) == len(set(donors)) and proposal["parent_candidate_id"] not in donors,
                  "invalid raw semantic donor IDs")
    _check_raw_premises(proposal["premises"])
    if proposal["repeat_delta"] is not None:
        _check_raw_repeat_delta(proposal["repeat_delta"])


def _serial_g_parents(snapshot, store):
    """Sources a serial-G episode may start from: the seed, or the previous episode's reviewed source roles."""
    _need(store is not None, "serial G parent admission requires the exact campaign store")
    history = [row for row in snapshot["state"]["history"]
               if row.get("attempt_id") and row.get("collection_ref")]
    if not history:
        seed_id = snapshot["archive"]["directions"][snapshot["campaign_id"]]["tip"]
        seed = store.get_candidate(seed_id)
        _need(all(seed[key] == value for key, value in snapshot["state"]["binding"].items()),
              "serial G seed belongs to another job/epoch")
        return {seed_id}
    previous = history[-1]
    _need(previous.get("settled") is True, "serial G predecessor must be settled")
    envelope, artifacts = store.evidence(previous["collection_ref"]["evidence_id"])
    _need(envelope["candidate_id"] == previous["candidate_id"], "G predecessor collection owner differs")
    roles = json.loads(artifacts["native_source_roles.json"])
    _need(type(roles) is dict and roles and set(roles) <= {"selected", "faithful", "unfinished", "terminal"},
          "G predecessor needs exact retained native source roles")
    owners = {row["opportunity_id"]: row["direction_id"] for row in snapshot["state"]["opportunities"]}
    note = snapshot["direction_notes"][owners[previous["attempt_id"]]]["note"]
    _need(note.get("research_status") == "reported" and note.get("research_review_id")
          and note.get("reviewed_outcome_refs") == [previous["transport_ref"], previous["collection_ref"]]
          and note.get("research_update_ref") == previous["collection_ref"]
          and note.get("latest_transport_ref") == previous["transport_ref"],
          "serial G predecessor needs its exact scientific review first")
    return set(roles.values())


def validate_action(policy, snapshot, proposal, *, store=None):
    """Serial-G restriction on proposals; it never ranks sources or selects ideas."""
    validate_policy(policy)
    if policy is None or policy["arm"] == "EVO" or proposal is None:
        return
    proposal_need(type(proposal) is dict, "G requires a proposal object or explicit stop")
    proposal_need(not proposal.get("semantic_donors"), "G has no explicit semantic donors")
    if proposal.get("kind") == "research_review":
        return
    proposal_need(proposal.get("kind") is None, "G has no portfolio or maturation actions")
    proposal_need(proposal.get("island_id") == "unassigned"
                  and all(proposal.get(key) is None for key in
                          ("membership_ref", "exchange_ref", "protection_ref", "grant_intent")),
                  "G has no island membership, donor exchange or protected-portfolio grant")
    proposal_need(proposal.get("expected_revision") == 0
                  and proposal.get("direction_id") not in snapshot["archive"]["directions"]
                  and proposal.get("direction_id") != snapshot["campaign_id"],
                  "serial G uses a fresh native episode ID, not a maintained alternative")
    allowed = _serial_g_parents(snapshot, store)
    proposal_need(proposal.get("parent_candidate_id") in allowed,
                  "serial G parent must be an exact source role of its preceding episode, or the initial seed")


def lineage_candidates(store, candidate_ids):
    """All local ancestor sources (parents and donors), including slow or broken historical tips."""
    pending, visited = list(candidate_ids), set()
    while pending:
        identity = pending.pop()
        if identity in visited:
            continue
        candidate = store.get_candidate(identity)
        visited.add(identity)
        pending.extend(candidate["parents"])
        pending.extend(candidate["donors"])
    return sorted(visited)
