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
import sys
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
    registration = admission["registration_sha256"]
    need(type(registration) is dict and registration, "existing registered source graph required")
    for path, digest in registration.items():
        checked({"path": path, "sha256": digest})
    wrapper = Path(__file__).resolve(strict=True)
    need(registration.get(str(wrapper)) == hashlib.sha256(wrapper.read_bytes()).hexdigest(),
         "measurement wrapper is absent from admitted code closure")
    for relative, digest in admission["assets_sha256"].items():
        path = native / relative
        need(path.is_relative_to(native) and ".." not in Path(relative).parts
             and registration.get(str(path)) == digest, "native asset differs from registration")
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
    protected(arm["study_clock_path"])
    need(Path(arm["study_clock_path"]).is_file(), "existing admitted study clock required")
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
                           reader=imported["geak_evo.control_result"], job_id=job, epoch=epoch)


def verify_protected_workspace(services, workspace):
    root = Path(workspace).absolute()
    need(root.resolve(strict=True) == root and root.is_dir(), "canonical candidate workspace required")
    contract = services.evaluator.contract
    for name, digest in contract["artifacts"].items():
        if name not in contract["source_paths"]:
            raw, _ = services.store_module._read(root, name)
            need(hashlib.sha256(raw).hexdigest() == digest, "protected workspace artifact changed: " + name)
    return root


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
    args = arguments.parse_args(remaining)
    need(not args.capture_only or (args.evaluation_id is None and args.candidate_id is None),
         "capture allocates a fresh final evaluation identity")
    return measure(services, args.workspace, args.evaluation_id or "geak-" + uuid.uuid4().hex,
                   candidate_id=args.candidate_id, capture_only=args.capture_only)


if __name__ == "__main__":
    raise SystemExit(main())
