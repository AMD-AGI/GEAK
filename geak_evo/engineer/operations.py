"""Prepare native operation files from explicit runtime templates and roots.

This module prepares source, request and deadline bindings. It does not launch
models or evaluators; the runtime owns execution and worker containment.
"""

import copy
from pathlib import Path
import re

from ..clock import SystemClock
from .._util import sha256_hex
from ..archive.refs import read_ref as _read_ref, file_ref
from .investigation import encoded, flat_action, need, read, write
from .instructions import tool_addendum


NAMES = ("entry-input.json", "host-config.json", "admission.json", "backend-binding.json", "collection-config.json")


def native_only_mode(config, admission, config_ref):
    """Bind an opt-in capability only from the protected stable templates."""
    probe = config["backend_config"]["probe_config"]
    modes = [value.get("native_only", False) for value in (config, probe, admission)]
    need(all(type(value) is bool for value in modes) and modes[0] == modes[1] == modes[2],
         "native-only capability differs between protected templates")
    if modes[0]:
        need(admission.get("schema") == "geak-dense-native-admission-v1"
             and "diagnostic" not in admission
             and type(admission["arms"]) is list
             and [arm["name"] for arm in admission["arms"]] == ["native"]
             and admission.get("host_config_refs") == {"native": config_ref},
             "native-only admission scope or host binding differs")
        need(config["hardware_profile"] is None and config["hardware_evaluation_root"] is None
             and config["helper_ref"]["path"] == str(Path(config["imports"]["host_root"]).parent / "root_helper.py"),
             "native-only capability requires the native helper without hardware configuration")
    return modes[0]


def future_ref(path, value):
    return {"path": str(path), "sha256": sha256_hex(encoded(value))}


def episode_output(campaign_root, direction_id, predecessor=None, *, opportunity):
    """Bind a workspace to its exact previous host attempt and to the planner opportunity that
    prepared it, never a model counter. A rejected proposal leaves its own inert preparation;
    a later opportunity from the same predecessor gets a new directory, never a reused one."""
    need(type(direction_id) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", direction_id)
         and direction_id not in (".", ".."), "invalid idea identity")
    need(type(opportunity) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", opportunity),
         "exact host planner opportunity required")
    if predecessor is None:
        return Path(campaign_root) / "ideas" / direction_id / opportunity
    need(type(predecessor) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", predecessor),
         "exact host predecessor attempt required")
    return Path(campaign_root) / "episodes" / direction_id / predecessor / opportunity


def prepare_workspace(store, workspace_helper, *, campaign_root, direction_id, job_id, epoch, session_id, parent,
                      opportunity, predecessor=None):
    """One fresh native workspace per episode, copied by the existing source helper."""
    need(re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", direction_id), "invalid idea identity")
    candidate = store.get_candidate(parent)
    need(candidate["job_id"] == job_id and candidate["measurement_epoch_id"] == epoch, "parent domain differs")
    output = episode_output(campaign_root, direction_id, predecessor, opportunity=opportunity)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return workspace_helper.fresh_workspace({"source_objects_root": str(store.root / "objects"),
        "selected_source_id": candidate["source_id"], "idea_id": direction_id,
        "session_id": session_id, "job_id": job_id, "measurement_epoch_id": epoch}, output)


def build_files(*, templates, request, operation_id, prompt_ref, workspace_ref, store_root,
                campaign_root, sealed_operation_root, clock_ref, turn_deadline_ns,
                shared_budget, common_read_roots, parents, search_policy, root_state_root, common_tool_addendum="", proposal=None,
                descriptor_factory=None, clock=None, backend_driver_ref=None):
    """Produce five existing schemas for one reserved call, with no install copy.

    templates contains exact refs named config/admission/entry/collection.
    Their code closure, runtime, owner, contract, profiles, device and source
    paths are fixed by the stable installation. Caller supplies only host-bound
    identity, finite reservation, original clock, and previously admitted history.
    Native shell commands and model/GPU settings never come from the planner.
    """
    from ..archive.policy import native_enabled, validate_read_scope

    need(search_policy is None or 'native_capabilities' not in search_policy or proposal is not None,
         "registered native capability policy requires the exact proposal")
    flat_action(proposal, search_policy)
    need(request['continuation']['mode'] == 'fresh' or native_enabled(search_policy, 'continuation'),
         "continuation is disabled by the registered native policy")
    need(set(templates) == {"config", "admission", "entry", "collection"}, "native template refs required")
    config, admission, entry, collection = [copy.deepcopy(read(templates[k])) for k in
                                           ("config", "admission", "entry", "collection")]
    native_only = native_only_mode(config, admission, templates["config"])
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", operation_id), "invalid operation identity")
    pins, clock_record = read(workspace_ref), read(clock_ref)
    campaign, sealed = Path(campaign_root), Path(sealed_operation_root)
    need(campaign.is_absolute() and campaign.resolve() == campaign and sealed.is_absolute()
         and sealed.name == operation_id and not sealed.exists(), "new canonical operation roots required")
    need(pins["workspace"] == request["workspace"] and pins["idea_id"] == request["direction_id"]
         and pins["job_id"] == request["job"]["job_id"]
         and pins["measurement_epoch_id"] == request["measurement_epoch_id"]
         and pins["starting_source_id"] == pins["patch_base_source_id"] == request["tip"]["source_id"]
         and pins["source_objects_root"] == str(Path(store_root) / "objects"), "prepared native workspace differs")
    clock = SystemClock() if clock is None else clock
    need(clock_record["boot_id"] == clock.boot_id()
         and clock_record["session_id"] == pins["session_id"] == shared_budget["session_id"], "campaign clock differs")
    end_ns = clock_record["started_monotonic_ns"] + int(clock_record["wall_seconds"] * 1e9)
    now = clock.monotonic_ns()
    need(shared_budget["boot_id"] == clock_record["boot_id"] and shared_budget["deadline_ns"] == end_ns
         and now < turn_deadline_ns < end_ns
         and (turn_deadline_ns - now) / 1e9 <= request["budget"]["wall_seconds"],
         "reserved investigation exceeds original campaign deadline")
    operation = campaign / "native-operations" / operation_id
    operation.mkdir(mode=0o700, parents=True)  # No reuse, including partial preparation.
    directories = ("model", "probe-turns", "evaluations", "readonly", "replies") if native_only else (
        "model", "probe-turns", "evaluations", "hardware-evaluations", "readonly", "replies")
    for name in directories:
        (operation / name).mkdir(mode=0o700)
    public = operation / "readonly"
    published_prompt = write(public / "outer-assignment.txt", _read_ref(prompt_ref))
    # Exact allowed refs are copied into one public directory, never broad store
    # grants. Raw source blobs referenced inside the original assignment stay
    # indexed below, even when their original private paths are unreadable.
    validate_read_scope(search_policy, campaign, [ref["path"] for ref in request["read_only"]])
    published = []
    for index, ref in enumerate(request["read_only"]):
        actual = write(public / ("evidence-%04d" % index), _read_ref(ref))
        published.append({"original_ref": ref, "readable_ref": actual})
    # Parent packets must already be admitted own-run source/evidence material;
    # this routine never searches other campaigns, including Greedy histories.
    def refs(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256"}:
                yield value
            else:
                for child in value.values():
                    yield from refs(child)
        elif type(value) is list:
            for child in value:
                yield from refs(child)
    parent_refs = list(refs(parents))
    validate_read_scope(search_policy, campaign, [ref["path"] for ref in parent_refs] + list(common_read_roots))
    for ref in parent_refs:
        _read_ref(ref)
    def publish_parent(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256"}:
                raw = _read_ref(value)
                destination = public / value["sha256"]
                if not destination.exists():
                    actual = write(destination, raw)
                else:
                    actual = file_ref(destination)
                need(actual["sha256"] == value["sha256"], "parent copy differs")
                published.append({"original_ref": value, "readable_ref": actual})
                return actual
            return {key: publish_parent(child) for key, child in value.items()}
        if type(value) is list:
            return [publish_parent(child) for child in value]
        return value
    parents = publish_parent(parents)
    write(public / "READ_INDEX.json", encoded(published))
    parent_file = write(public / "PARENTS.json", encoded(parents))
    native, probe = config["backend_config"], config["backend_config"]["probe_config"]
    need(type(proposal["reservation"]["evaluations"]) is int
         and proposal["reservation"]["evaluations"] > 0, "positive whole native request reservation required")
    probe["max_calls_per_turn"] = min(probe["max_calls_per_turn"], proposal["reservation"]["evaluations"])
    work = Path(request["workspace"])
    requests, output = work / ".native-requests" / operation_id, work / ".idea-output" / operation_id
    requests.mkdir(mode=0o700, parents=True)
    output.mkdir(mode=0o700, parents=True)
    entry.update(idea_id=request["direction_id"], turn_id=operation_id, workspace=str(work),
        initial_workspace=pins["initial_workspace"], output_dir=str(output),
        starting_source_id=request["tip"]["source_id"], patch_base_source_id=request["tip"]["source_id"],
        source_paths=pins["source_paths"], continuation={"mode": "fresh", "prior_receipt_ref": None},
        parents=parents, insights=[published_prompt, parent_file, file_ref(public / "READ_INDEX.json")],
        timeout_ms=int(min(14400, request["budget"]["wall_seconds"]) * 1000),
        direction=dict(entry["direction"], id=request["direction_id"], title=request["next_question"],
                       prompt=request["next_question"]))
    native.update(direction_id=request["direction_id"], clock_ref=clock_ref,
                  continuation={"mode": "fresh", "prior_receipt_ref": None})
    # The caller's public own-history directories are explicit. Read-only roots
    # must be directories because the existing native CPU transport requires it.
    roots = list(common_read_roots) + [str(public), pins["initial_workspace"], str(operation / "replies")]
    native["model_config"].update(clock_path=clock_ref["path"], state_root=str(operation / "model"),
        work_root=str(work), read_roots=list(dict.fromkeys(roots)))
    probe.update(session_id=clock_record["session_id"], turn_id=operation_id, workspace=str(work),
        request_root=str(requests), reply_root=str(operation / "replies"), state_root=str(operation / "probe-turns"),
        job_id=request["job"]["job_id"], measurement_epoch_id=request["measurement_epoch_id"],
        turn_deadline_ns=turn_deadline_ns, budget=copy.deepcopy(shared_budget))
    entry["tool_addendum"] = tool_addendum(config=config, admission=admission, request=request,
        operation=operation, turn_deadline_ns=turn_deadline_ns, native_only=native_only,
        search_policy=search_policy, common_tool_addendum=common_tool_addendum)
    config.update(store_root=str(store_root), evaluation_root=str(operation / "evaluations"),
        hardware_evaluation_root=None if native_only else str(operation / "hardware-evaluations"), workspace_ref=workspace_ref,
        entry_input_ref=future_ref(sealed / "entry-input.json", entry))
    if native['model_config'].get('codex_runtime_family_ref') is not None:
        need(native['model_config'].get('workflow_backend') == 'codex' and callable(descriptor_factory)
             and native['model_config']['tool_audit_ref'] == config['tool_audit_ref'],
             'Codex family requires an explicit trusted descriptor factory and the host seed audit')
        private_roots = [native['model_config']['state_root'], probe['state_root'], probe['budget']['root'],
                         config['store_root'], config['evaluation_root']]
        if not native_only:
            private_roots.append(config['hardware_evaluation_root'])
        descriptor = descriptor_factory(native['model_config'], workspace=work,
            read_roots=native['model_config']['read_roots'], state_root=native['model_config']['state_root'],
            operational_roots=private_roots)
        descriptor_path = operation / 'model' / 'codex-runtime.json'
        descriptor_ref = write(descriptor_path, encoded(descriptor))
        descriptor_path.chmod(0o444)
        native['model_config']['codex_runtime_ref'] = descriptor_ref
    collection.update(session_id=clock_record["session_id"], turn_id=operation_id,
        direction_id=request["direction_id"], job_id=request["job"]["job_id"],
        measurement_epoch_id=request["measurement_epoch_id"], workspace=str(work), output_dir=str(output),
        source_objects_root=str(Path(store_root) / "objects"), parents=parents,
        starting_source_id=request["tip"]["source_id"], patch_base_source_id=request["tip"]["source_id"])
    config_ref = future_ref(sealed / "host-config.json", config)
    admission.update(study_id=clock_record["session_id"], arm_seconds=clock_record["wall_seconds"], prospective_unsealed=False,
                     state_root=str(Path(root_state_root) / operation_id))
    if native_only:
        admission["host_config_refs"] = {"native": config_ref}
    else:
        admission["diagnostic"]["host_config_refs"] = {name: config_ref for name in ("native", "hardware")}
    for arm in admission["arms"]:
        arm.update(request_root=config["evaluation_root" if arm["name"] == "native" else "hardware_evaluation_root"],
            source_object_root=str(Path(store_root) / "objects"), study_clock_path=clock_ref["path"],
            measurement_epoch_id=request["measurement_epoch_id"])
    # Preserve the runtime's device pool lock and owner bindings. Every native
    # request still acquires the configured physical lock.
    driver_ref = backend_driver_ref or file_ref(Path(config["imports"]["host_root"]).parent / "host_driver.py")
    _read_ref(driver_ref)
    binding = {"driver_ref": driver_ref, "config_ref": config_ref}
    return dict(zip(NAMES, (entry, config, admission, binding, collection)))
