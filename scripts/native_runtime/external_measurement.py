"""Ordinary GEAK command over an existing admitted native evaluator.

The operational owner installs/registers this file and supplies an existing
admission and clock. This command creates no admission, clock or GPU lifecycle.
"""

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def protected(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "canonical protected path required")
    for item in (path, *path.parents):
        info = item.lstat()
        need(not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
             "root-controlled source/admission required: " + str(item))
    return path


def checked(ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256"}, "exact file reference required")
    path = protected(ref["path"])
    need(path.is_file(), "regular source/admission required")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref["sha256"], "registered file changed: " + str(path))
    return raw


def ordinary_directory(value):
    path = Path(value)
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
         "existing canonical ordinary-host directory required")
    info = path.stat()
    need(info.st_uid == os.geteuid() != 0 and not info.st_mode & 0o022,
         "existing private ordinary-host state required")
    return path


def ordinary_registration(admission, native, wrapper):
    """Authenticate the ordinary reader; root_helper checks the full graph."""
    registration = admission["registration_sha256"]
    need(type(registration) is dict and registration, "existing registered source graph required")
    assets = admission["assets_sha256"]
    need(type(assets) is dict and assets, "existing native asset graph required")
    result = {}
    for relative, digest in assets.items():
        path = native / relative
        need(path.is_relative_to(native) and ".." not in Path(relative).parts
             and registration.get(str(path)) == digest, "native asset differs from registration")
        result[str(path)] = digest
    need(str(wrapper) in registration,
         "measurement wrapper is absent from admitted code closure")
    result[str(wrapper)] = registration[str(wrapper)]
    for path, digest in result.items():
        checked({"path": path, "sha256": digest})
    return result


def bind_services(admission_ref):
    """Bind the same source graph and callback ABI as the ORIGINAL v2 recipe."""
    need(os.geteuid() != 0 and sys.flags.isolated and sys.flags.dont_write_bytecode
         and sys.pycache_prefix is None, "ordinary isolated Python -I -B required")
    admission = json.loads(checked(admission_ref))
    need(admission["schema"] == "geak-dense-native-admission-v1"
         and admission["native_only"] is True and admission["prospective_unsealed"] is False
         and [arm["name"] for arm in admission["arms"]] == ["native"],
         "existing sealed native-only admission required")
    arm = admission["arms"][0]
    native = protected(admission["intended_install_root"])
    artifact_root = protected(admission["request_artifact_root"])
    need(artifact_root == native / "task", "admitted task root differs from native installation")
    wrapper = Path(__file__).resolve(strict=True)
    # Owner-only prerequisites remain in the sealed admission and are checked
    # by the privileged helper on every launch/collect. This process reads only
    # the positive native asset closure and its own registered entrypoint.
    registration = ordinary_registration(admission, native, wrapper)
    package = native / "cleaned/geak_evo"
    need(not list(package.rglob("*.pyc")) and not list((native / "__pycache__").glob("*.pyc")),
         "registered source graph must have no importable bytecode cache")
    need(not any(path.is_file() and any(str(path).endswith(suffix)
             for suffix in importlib.machinery.EXTENSION_SUFFIXES) for path in package.rglob("*")),
         "native extensions are outside the registered Python reader graph")
    for path in package.rglob("*.py"):
        need(str(path) in registration, "unregistered cleaned Python source")
    for name in ("adapter.py", "contract.json", "root_helper.py"):
        need(str(native / name) in registration, "native entry source is not registered")
    need(not any(name == "geak_evo" or name.startswith("geak_evo.") or name == "adapter"
                 for name in sys.modules), "fresh admitted import namespace required")
    # Give only the named package its reviewed search path. Adding the native
    # root to sys.path could let an unrelated file shadow a standard library.
    for name, path, locations in (("geak_evo", package / "__init__.py", [str(package)]),
                                  ("adapter", native / "adapter.py", None)):
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    imported = {name: importlib.import_module(name) for name in (
        "geak_evo.archive.store", "geak_evo.evaluate.evaluator", "geak_evo.evaluate.jobs",
        "geak_evo.control_result", "adapter")}
    for name, module in list(sys.modules.items()):
        if name == "geak_evo" or name.startswith("geak_evo.") or name == "adapter":
            path = Path(module.__file__)
            need(path.suffix == ".py" and (path.is_relative_to(package) if name != "adapter"
                                          else path == native / "adapter.py"),
                 "native import escaped admitted source graph")
            checked({"path": str(path), "sha256": registration[str(path)]})
    objects = ordinary_directory(arm["source_object_root"])
    need(objects.name == "objects", "existing ResearchStore objects directory required")
    store_root = ordinary_directory(objects.parent)
    output_root = ordinary_directory(arm["request_root"])
    # Reading the existing owner clock is a prerequisite. No clock is created,
    # extended or reset here; NativeCallbacks/root_helper retain enforcement.
    clock_path = protected(arm["study_clock_path"])
    need(clock_path.is_file(), "existing admitted study clock required")
    clock_ref = {"path": str(clock_path), "sha256": hashlib.sha256(clock_path.read_bytes()).hexdigest()}
    jobs = imported["geak_evo.evaluate.jobs"]
    contract = jobs.load_arena_job(native / "contract.json", artifact_root=artifact_root,
                                   expected_digest=admission["contract_id"])
    need(contract["schema"] == "geak-arena-external-task-v1", "external task contract required")
    job, epoch = admission["contract_id"], arm["measurement_epoch_id"]
    store_module = imported["geak_evo.archive.store"]
    store = store_module.ResearchStore(store_root, allowed_epochs={(job, epoch)},
                                       source_paths={job: contract["source_paths"]})
    wait = admission["evaluation_seconds"] + admission["settlement_reserve_seconds"]
    need(type(wait) in (int, float) and math.isfinite(wait) and wait > 0, "finite native wait required")
    callbacks = imported["adapter"].NativeCallbacks(native / "root_helper.py",
        admission_path=admission_ref["path"], use_sudo=True, wait_seconds=wait)
    evaluator = imported["geak_evo.evaluate.evaluator"].ArenaEvaluator(store,
        contract=contract, contract_id=job, artifact_root=artifact_root, output_root=output_root,
        measurement_epoch_id=epoch, launch=callbacks.launch, settle=callbacks.settle)
    need(evaluator.evaluator_sha256 == admission["evaluator_sha256"] == arm["evaluator_sha256"],
         "evaluator differs from existing admission")
    return SimpleNamespace(store=store, evaluator=evaluator, store_module=store_module,
                           reader=imported["geak_evo.control_result"], job_id=job, epoch=epoch,
                           clock_ref=clock_ref)


def verify_protected_workspace(services, workspace):
    root = Path(workspace).absolute()
    need(root.resolve(strict=True) == root and root.is_dir(), "canonical candidate workspace required")
    contract = services.evaluator.contract
    for name, digest in contract["artifacts"].items():
        if name not in contract["source_paths"]:
            raw, _ = services.store_module._read(root, name)
            need(hashlib.sha256(raw).hexdigest() == digest, "protected workspace artifact changed: " + name)
    return root


def sample_clock():
    path = Path("/proc/sys/kernel/random/boot_id")
    boot = path.read_text().strip()
    instant = time.monotonic_ns()
    need(path.read_text().strip() == boot, "capture clock boot changed")
    return boot, instant


def committed_source(workspace, source):
    """Join every captured byte/mode to one actual canonical Git commit."""
    def git(*args):
        result = subprocess.run(["git", *args], cwd=workspace, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CEILING_DIRECTORIES": str(workspace.parent)})
        need(result.returncode == 0, "canonical Git source is unavailable")
        return result.stdout
    head = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = {}
    for entry in git("ls-tree", "-rz", head, "--", *[row["path"] for row in source["files"]]).split(b"\0"):
        if entry:
            mode_kind_id, path = entry.split(b"\t", 1)
            tree[path.decode()] = mode_kind_id.split()
    need(set(tree) == {row["path"] for row in source["files"]}, "canonical Git source closure differs")
    for row in source["files"]:
        mode, kind, blob = tree[row["path"]]
        need(kind == b"blob" and mode in (b"100644", b"100755")
             and (0o111 if mode == b"100755" else 0) == row["executable"]
             and hashlib.sha256(git("cat-file", "blob", blob.decode())).hexdigest() == row["blob_id"],
             "canonical source differs from Git commit: " + row["path"])
    need(git("rev-parse", "--verify", "HEAD^{commit}").decode().strip() == head,
         "canonical Git commit changed during capture")
    return head


def capture_selection(services, workspace, *, selected=None, transition=None, clock=sample_clock):
    """Retain canonical bytes and their original-clock observation, without evaluation.

    The host brackets canonical mutation with two captures. This receipt alone
    observes source availability; it is not a Git-commit timestamp or privileged
    attestation. A proposed winner's settled evidence is checked against these
    exact bytes; a hand merge cannot inherit metrics for a different source.
    """
    clock_raw = Path(services.clock_ref["path"]).read_bytes()
    need(hashlib.sha256(clock_raw).hexdigest() == services.clock_ref["sha256"], "original clock changed")
    original = json.loads(clock_raw)
    parent = None
    if transition is not None:
        need(set(transition) == {"kind", "round", "sequence", "previous_capture_ref", "stopped_by"},
             "exact canonical transition required")
        kind, sequence = transition["kind"], transition["sequence"]
        need(kind in ("initial", "resume_unknown", "before_commit", "after_commit", "after_restore", "terminal")
             and type(sequence) is int and sequence > 0
             and type(transition["round"]) is int and transition["round"] >= 0,
             "valid canonical transition required")
        need((kind == "terminal" and type(transition["stopped_by"]) is str and transition["stopped_by"])
             or (kind != "terminal" and transition["stopped_by"] is None), "canonical stop reason differs")
        previous = transition["previous_capture_ref"]
        if kind in ("initial", "resume_unknown"):
            need(sequence == 1 and previous is None and transition["round"] == 0,
                 "initial canonical transition differs")
        elif previous is None:
            # Keep later source bytes after a failed first observation. The
            # missing anchor remains an explicit gap for the cutoff reader.
            need(sequence > 1, "noninitial canonical capture requires an attempted predecessor")
        else:
            need(type(previous) is dict and set(previous) == {"evidence_id", "artifact"}
                 and previous["artifact"] == "canonical-selection.json", "previous canonical capture required")
            envelope, raw = services.store.evidence_artifact(previous["evidence_id"], previous["artifact"])
            parent = json.loads(raw)
            need(parent["schema"] == "geak-canonical-source-capture-v2"
                 and parent["clock_ref"] == services.clock_ref
                 and parent["transition"]["sequence"] < sequence
                 and envelope["candidate_id"] == parent["candidate_id"]
                 and envelope["binding"] == {"job_id": services.job_id,
                    "measurement_epoch_id": services.epoch, "candidate_source_id": parent["source_id"]},
                 "canonical capture sequence or clock differs")
            before = parent["transition"]
            if sequence != before["sequence"] + 1:
                pass  # Retain later bytes; the explicit sequence gap makes the timeline unresolved.
            elif kind in ("after_commit", "after_restore"):
                need(before["kind"] == "before_commit" and before["round"] == transition["round"],
                     "canonical mutation has no matching before capture")
            else:
                need(before["kind"] in ("initial", "resume_unknown", "after_commit", "after_restore")
                     and (transition["round"] > before["round"] if kind == "before_commit"
                          else transition["round"] >= before["round"]),
                     "canonical transition order differs")
    boot, lower = clock()
    need(boot == original["boot_id"] and lower >= original["started_monotonic_ns"],
         "capture differs from original clock")
    need(parent is None or lower >= parent["upper_monotonic_ns"], "canonical capture clock went backwards")
    root = verify_protected_workspace(services, workspace)
    source = services.store.capture_source(root, services.evaluator.contract["source_paths"])
    need(all(row["blob_id"] is not None for row in services.store.source(source)["files"]),
         "canonical source closure is incomplete")
    candidate = services.store.candidate(services.job_id, services.epoch, source)
    try:
        commit = {"status": "committed", "git_commit": committed_source(root, services.store.source(source))}
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        commit = {"status": "unresolved", "error_kind": type(error).__name__, "reason": str(error)}
    evidence = {"status": "missing", "reason": "no_selected_evaluation"}
    if selected is not None:
        try:
            checked_result = services.reader.read_control_result(services.evaluator, selected["candidate_id"],
                evaluation_id=selected["evaluation_id"], workspace=root)
            # Source IDs, rather than candidate IDs, define program identity.
            need(services.store.get_candidate(selected["candidate_id"])["source_id"] == source,
                 "selected evaluation source differs from canonical capture")
            evidence = {"status": "validated", "result": checked_result}
        except (ValueError, OSError, KeyError, TypeError) as error:
            evidence = {"status": "unresolved", "selected_evaluation": selected,
                        "error_kind": type(error).__name__, "reason": str(error)}
    verify_protected_workspace(services, root)
    need(services.store.capture_source(root, services.evaluator.contract["source_paths"]) == source,
         "canonical source changed during capture")
    end_boot, upper = clock()
    need(end_boot == boot and lower <= upper, "capture clock changed")
    need(Path(services.clock_ref["path"]).read_bytes() == clock_raw, "original clock changed")
    capture = {"schema": "geak-canonical-source-capture-v1", "candidate_id": candidate, "source_id": source,
               "clock_ref": services.clock_ref, "boot_id": boot,
               "lower_monotonic_ns": lower, "upper_monotonic_ns": upper,
               "canonical_commit": commit, "development_evidence": evidence}
    if transition is not None:
        capture.update(schema="geak-canonical-source-capture-v2", transition=transition)
    identity = services.store.archive_evidence(candidate, {
        "canonical-selection.json": canonical(capture).encode(), "original-clock.json": clock_raw})
    print(canonical({**capture, "capture_ref": {"evidence_id": identity,
                                              "artifact": "canonical-selection.json"}}), flush=True)
    return 0


def measure(services, workspace, evaluation_id, *, candidate_id=None, capture_only=False):
    root = verify_protected_workspace(services, workspace)
    paths = services.evaluator.contract["source_paths"]
    source = services.store.capture_source(root, paths)
    need(all(row["blob_id"] is not None for row in services.store.source(source)["files"]),
         "candidate source closure is incomplete")
    verify_protected_workspace(services, root)
    candidate = services.store.candidate(services.job_id, services.epoch, source)
    need(candidate_id is None or candidate == candidate_id,
         "workspace differs from the candidate selected before final validation")
    identity = {"candidate_id": candidate, "evaluation_id": evaluation_id, "source_id": source}
    if capture_only:
        # Validate the planned identity without prepare(): a prepared operation
        # is intentionally not launchable. The final call creates its request.
        services.evaluator._request(candidate, evaluation_id)
        need(not (services.evaluator.root / evaluation_id).exists(),
             "final evaluation identity is already in use")
        print(canonical({"schema": "geak-native-measurement-identity-v1",
                         "capture_only": True, **identity}), flush=True)
        return 0
    # Expose the exact identity before the existing callback may block. The
    # measurement argv is used by the engineer, not the JSON-only lane reader.
    print(canonical({"schema": "geak-native-measurement-identity-v1", **identity}), flush=True)
    response = services.evaluator.evaluate(candidate, evaluation_id=evaluation_id)
    print(canonical({"schema": "geak-native-measurement-result-v1", **identity,
                     "external_evaluation": {key: identity[key] for key in ("candidate_id", "evaluation_id")},
                     "receipt": response["receipt"], "costs": response["costs"]}), flush=True)
    return 0 if response["receipt"]["settled"] and response["receipt"]["native_suite_pass_observed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("action", choices=("measure", "read"))
    options, remaining = parser.parse_known_args(argv)
    services = bind_services(dict(zip(("path", "sha256"), options.admission)))
    arguments = argparse.ArgumentParser(add_help=False)
    arguments.add_argument("--workspace", required=True, type=Path)
    if options.action == "read":
        args, _ = arguments.parse_known_args(remaining)
        verify_protected_workspace(services, args.workspace)
        return services.reader.main(remaining, evaluator=services.evaluator)
    arguments.add_argument("--evaluation-id", default=None)
    arguments.add_argument("--candidate-id", default=None)
    arguments.add_argument("--capture-only", action="store_true")
    arguments.add_argument("--capture-selection", action="store_true")
    arguments.add_argument("--selected-candidate-id")
    arguments.add_argument("--selected-evaluation-id")
    arguments.add_argument("--selection-kind")
    arguments.add_argument("--selection-round", type=int)
    arguments.add_argument("--selection-sequence", type=int)
    arguments.add_argument("--previous-selection")
    arguments.add_argument("--selection-stop-reason")
    args = arguments.parse_args(remaining)
    need(bool(args.selected_candidate_id) == bool(args.selected_evaluation_id),
         "selected candidate and evaluation IDs must be supplied together")
    need(args.capture_selection or args.selected_candidate_id is None, "selected evidence requires source capture")
    if args.capture_selection:
        need(not args.capture_only and args.evaluation_id is None and args.candidate_id is None,
             "canonical capture does not allocate an evaluation")
        selected = ({"candidate_id": args.selected_candidate_id, "evaluation_id": args.selected_evaluation_id}
                    if args.selected_candidate_id else None)
        transition = None
        if args.selection_kind is not None:
            transition = {"kind": args.selection_kind, "round": args.selection_round,
                "sequence": args.selection_sequence, "stopped_by": args.selection_stop_reason,
                "previous_capture_ref": ({"evidence_id": args.previous_selection,
                    "artifact": "canonical-selection.json"} if args.previous_selection else None)}
        return capture_selection(services, args.workspace, selected=selected, transition=transition)
    need(not args.capture_only or (args.evaluation_id is None and args.candidate_id is None),
         "capture allocates a fresh final evaluation identity")
    return measure(services, args.workspace, args.evaluation_id or "geak-" + uuid.uuid4().hex,
                   candidate_id=args.candidate_id, capture_only=args.capture_only)


if __name__ == "__main__":
    raise SystemExit(main())
