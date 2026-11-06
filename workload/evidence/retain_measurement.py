#!/usr/bin/env python3
"""Copy one worker measurement's evidence before its working files can change."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import tempfile
import time

ARTIFACT_ROOTS = ("run", "kernel_src", "ut_result.json", "_baseline_random.pt")


def reuse_helper():
    """Load only this file's optional sibling; preserve exact-module test patches."""
    import importlib.util
    path = Path(__file__).resolve().with_name('retention_reuse.py')
    loaded = sys.modules.get('retention_reuse')
    loaded_path = getattr(loaded, '__file__', None)
    if isinstance(loaded_path, str) and Path(loaded_path).resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location('_retention_reuse_sibling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_identity(pid):
    try:
        fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else fields[19]
    except (OSError, IndexError):
        return None


def manual_processes(record):
    manual = record.get("manual_measurement")
    if not manual:
        return {}
    token = manual.get("token", "")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("invalid manual measurement ownership token")
    required = {("WORKLOAD_MANUAL_TOKEN=" + token).encode(),
                ("WORKLOAD_MANUAL_TASK=" + record["task_dir"]).encode()}
    owned = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if required <= set((path / "environ").read_bytes().split(b"\0")):
                identity = process_identity(int(path.name))
                if identity is not None:
                    owned[int(path.name)] = identity
        except (OSError, ProcessLookupError):
            continue
    return owned


def stop_manual_processes(record):
    """Stop only this invocation's marked descendants, including detached ranks.

    The marker survives the frozen collective's copied environment and new sessions.
    It also lets restart recovery find orphan ranks after their wrapper was killed.
    PID generations (and pidfds where supported) prevent signalling a reused PID.
    """
    stopped = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 1.0  # Bounded operational cleanup grace, not a scoring threshold.
        sent = set()
        while True:
            owned = manual_processes(record)
            if not owned:
                return sorted(stopped)
            for pid, identity in owned.items():
                if (pid, identity) in sent:
                    continue
                descriptor = None
                try:
                    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                        descriptor = os.pidfd_open(pid)
                    if process_identity(pid) != identity:
                        continue
                    if descriptor is None:
                        os.kill(pid, sig)
                    else:
                        signal.pidfd_send_signal(descriptor, sig)
                    stopped.add(pid)
                    sent.add((pid, identity))
                except ProcessLookupError:
                    pass
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise ValueError("manual measurement descendants remain alive; working evidence is not stable")


def write_record(directory, record):
    temporary = directory / "retention.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, directory / "retention.json")


def write_pointer(path, directory):
    # Never truncate the last usable pointer. Failed/interrupted publications
    # leave it intact and retain the temporary file for inspection.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        stream.write(str(directory) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def outside_task(task, path):
    if path.resolve().is_relative_to(task.resolve()):
        raise ValueError("evidence archive must be outside the entire task directory")


def evidence_parent(task, parent, pending, evidence_root=None):
    """Redirect only a new begin; preserve its existing per-task relative layout."""
    parent = Path(parent)
    if evidence_root is None:
        return parent, None
    root = Path(evidence_root)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('evidence_root must be a normalized absolute path')
    if pending is None:
        raise ValueError('external evidence_root requires the existing per-task pending pointer')
    task = Path(task).resolve()
    owner = Path(pending).parent.resolve()
    if owner.name != task.name or owner.parent.name != 'runs':
        raise ValueError('pending pointer does not identify this task owner')
    try:
        relative = parent.resolve().relative_to(owner)
    except ValueError:
        raise ValueError('new evidence parent is outside the pending task owner') from None
    destination = root.resolve() / task.name / relative
    outside_task(task, destination)
    return destination, {'root': str(root.resolve()), 'owner': str(owner),
                         'original_parent': str(parent.resolve()), 'relative_parent': relative.as_posix(),
                         'actual_parent': str(destination)}


def begin(task, parent, label, pending=None, evidence_root=None):
    outside_task(task, parent)
    if pending is not None:
        outside_task(task, pending)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("invalid evidence archive label")
    parent, placement = evidence_parent(task, parent, pending, evidence_root)
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=label + "-", dir=parent))
    record = {"complete": False, "kind": label, "started": time.time(),
              "task_dir": str(task.resolve()), "artifact_roots": list(ARTIFACT_ROOTS)}
    if placement is not None:
        record['evidence_placement'] = placement
    write_record(directory, record)
    if pending is not None:
        write_pointer(pending, directory)
    return directory


def reuse_with_retry(helper, task, directory, lock_timeout=None, cancelled=None):
    """Retry only acquisition contention, before any journal or payload exists."""
    deadline = None
    if lock_timeout is not None:
        if not math.isfinite(lock_timeout) or lock_timeout <= 0:
            raise ValueError("retention lock timeout must be finite and positive")
        deadline = time.monotonic() + lock_timeout
    while True:
        if cancelled is not None and cancelled():
            raise InterruptedError("retention lock wait cancelled by caller")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("retention maintenance lock wait exceeded caller timeout")
        try:
            reuse = helper.Reuse(task, directory)
        except helper.MaintenanceLockBusy:
            if deadline is None:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("retention maintenance lock wait exceeded caller timeout") from None
            # Match the existing manual-wrapper polling cadence, not a new policy bound.
            time.sleep(min(0.05, remaining))
        else:
            if cancelled is not None and cancelled():
                reuse.close()
                raise InterruptedError("retention lock wait cancelled by caller")
            return reuse


def retain(task, directory, raw_log=None, measure_exit=None, native_exit=None, complete_pointer=None,
           lock_timeout=None, cancelled=None):
    outside_task(task, directory)
    if complete_pointer is not None:
        outside_task(task, complete_pointer)
    record = json.loads((directory / "retention.json").read_text())
    if record.get("complete"):
        raise ValueError("completed evidence is immutable")
    if (directory / 'retention-files.jsonl').exists() or (directory / 'retention-files.jsonl').is_symlink():
        raise ValueError('existing reuse journal; recover into a fresh archive instead')
    record.update(measure_exit_code=measure_exit, native_exit_code=native_exit,
                  copied_files=0, copied_bytes=0, missing=[], artifact_roots=list(ARTIFACT_ROOTS))
    write_record(directory, record)
    helper = reuse_helper() if os.environ.get('WORKLOAD_RETENTION_REUSE') == '1' else None
    reuse = None

    def copy(source, destination):
        mode = source.lstat().st_mode
        if stat.S_ISDIR(mode):
            destination.mkdir()
            for child in sorted(source.iterdir()):
                copy(child, destination / child.name)
            if reuse is not None:
                fd = helper.open_path(destination, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
        elif stat.S_ISREG(mode):
            if destination.exists() or destination.is_symlink():
                raise ValueError('destination already exists; preserve partial evidence: ' + str(destination))
            if reuse is None:
                shutil.copy2(source, destination)
            else:
                reuse.copy(source, destination)
            record["copied_files"] += 1
            record["copied_bytes"] += destination.stat().st_size
        else:
            # Following links can copy mutable external data or recurse into this
            # archive. A partial copy is retained and the worker must stop instead.
            raise ValueError("non-regular evidence entry: " + str(source))

    try:
        if helper is not None:
            reuse = reuse_with_retry(helper, task, directory, lock_timeout, cancelled)
        for name in ARTIFACT_ROOTS:
            source = task / name
            if source.exists() or source.is_symlink():
                copy(source, directory / name)
            else:
                record["missing"].append(name)
        if raw_log is not None and (raw_log.exists() or raw_log.is_symlink()):
            copy(raw_log, directory / "native.log")
        record.update(complete=True, finished=time.time())
    except (OSError, ValueError) as exc:
        record.update(error=str(exc), finished=time.time())
        if reuse is not None:
            try: record['reuse'] = reuse.finish()
            finally: reuse.close()
        write_record(directory, record)
        raise
    if reuse is not None:
        try:
            record['reuse'] = reuse.finish()
            if record['reuse'].get('incomplete_extra_records'):
                raise ValueError('completed provenance exceeds the bounded reader contract')
            write_record(directory, record)
            for path, flags in ((directory / 'retention.json', os.O_RDONLY),
                                (directory, os.O_RDONLY | os.O_DIRECTORY),
                                (directory.parent, os.O_RDONLY | os.O_DIRECTORY)):
                fd = helper.open_path(path, flags)
                try: os.fsync(fd)
                finally: os.close(fd)
            try: reuse.publish_hint()
            except OSError as exc:
                print('retention reuse hint not advanced: ' + str(exc), file=sys.stderr)
        except (OSError, ValueError) as exc:
            record.update(complete=False, error=str(exc), finished=time.time())
            write_record(directory, record)
            raise
        finally: reuse.close()
    else:
        write_record(directory, record)
    if complete_pointer is not None:
        write_pointer(complete_pointer, directory)


def recover(task, pending, raw_log, lock_timeout=None, cancelled=None, evidence_root=None):
    """Finish retention on restart without rerunning or judging an interrupted UT."""
    outside_task(task, pending)
    previous = Path(pending.read_text().strip())
    outside_task(task, previous)
    record = json.loads((previous / "retention.json").read_text())
    if record.get("task_dir") != str(task.resolve()):
        raise ValueError("pending evidence belongs to another task")
    parent, placement = previous.parent, None
    if evidence_root is not None:
        parent, placement = evidence_parent(task, pending.parent / 'recovery', pending, evidence_root)
    stop_manual_processes(record)
    directory = previous
    if not record.get("complete"):
        # Never merge into or overwrite the partial archive. Preserve transport
        # logs, then copy the still-unmodified working evidence to a fresh sibling.
        directory = begin(task, parent, record["kind"])
        new = json.loads((directory / "retention.json").read_text())
        new.update(recovered_from=str(previous), recovery="retention only; no verdict inferred")
        if placement is not None:
            new['evidence_placement'] = placement
            if record.get('evidence_placement'):
                new['source_evidence_placement'] = record['evidence_placement']
        elif record.get('evidence_placement'):
            new['evidence_placement'] = record['evidence_placement']
        if record.get("manual_measurement"):
            new["manual_measurement"] = record["manual_measurement"]
        write_record(directory, new)
        for name in ("measure.stdout.log", "measure.stderr.log", "previous_measure.log"):
            source = previous / name
            if source.exists():
                if source.is_symlink() or not source.is_file():
                    raise ValueError("invalid pending transport log: " + str(source))
                shutil.copy2(source, directory / name)
        write_pointer(pending, directory)
        retain(task, directory, None if record.get("manual_measurement") else raw_log,
               record.get("measure_exit_code"), record.get("native_exit_code"),
               lock_timeout=lock_timeout, cancelled=cancelled)
    if record["kind"] == "pre-search":
        write_pointer(pending.parent / "pre_search_evidence.path", directory)
    pending.unlink()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("begin", "retain", "recover"))
    parser.add_argument("task", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--measure-exit", type=int)
    parser.add_argument("--native-exit", type=int)
    parser.add_argument("--pending", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=os.environ.get('WORKLOAD_EVIDENCE_ROOT'),
                        help="optional root for new begin/recovery archives; old archives and task ownership are preserved")
    parser.add_argument("--complete-pointer", type=Path)
    parser.add_argument("--lock-timeout", type=float,
                        help="finite lock-wait budget supplied by the caller; omitted means fail-fast")
    args = parser.parse_args()
    try:
        if args.action == "begin":
            print(begin(args.task, args.destination, args.label, args.pending, args.evidence_root))
        elif args.action == "recover":
            print(recover(args.task, args.destination, args.raw_log, lock_timeout=args.lock_timeout,
                          evidence_root=args.evidence_root))
        else:
            retain(args.task, args.destination, args.raw_log, args.measure_exit, args.native_exit,
                   args.complete_pointer, lock_timeout=args.lock_timeout)
    except (OSError, ValueError) as exc:
        print("evidence retention failed: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
