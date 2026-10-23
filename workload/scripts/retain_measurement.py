#!/usr/bin/env python3
"""Copy one worker measurement's evidence before its working files can change."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time

ARTIFACT_ROOTS = ("run", "kernel_src", "ut_result.json", "_baseline_random.pt")


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


def begin(task, parent, label, pending=None):
    outside_task(task, parent)
    if pending is not None:
        outside_task(task, pending)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("invalid evidence archive label")
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=label + "-", dir=parent))
    write_record(directory, {"complete": False, "kind": label, "started": time.time(),
                             "task_dir": str(task.resolve()), "artifact_roots": list(ARTIFACT_ROOTS)})
    if pending is not None:
        write_pointer(pending, directory)
    return directory


def retain(task, directory, raw_log=None, measure_exit=None, native_exit=None, complete_pointer=None):
    outside_task(task, directory)
    if complete_pointer is not None:
        outside_task(task, complete_pointer)
    record = json.loads((directory / "retention.json").read_text())
    if record.get("complete"):
        raise ValueError("completed evidence is immutable")
    record.update(measure_exit_code=measure_exit, native_exit_code=native_exit,
                  copied_files=0, copied_bytes=0, missing=[], artifact_roots=list(ARTIFACT_ROOTS))
    write_record(directory, record)

    def copy(source, destination):
        mode = source.lstat().st_mode
        if stat.S_ISDIR(mode):
            destination.mkdir()
            for child in sorted(source.iterdir()):
                copy(child, destination / child.name)
        elif stat.S_ISREG(mode):
            shutil.copy2(source, destination)
            record["copied_files"] += 1
            record["copied_bytes"] += destination.stat().st_size
        else:
            # Following links can copy mutable external data or recurse into this
            # archive. A partial copy is retained and the worker must stop instead.
            raise ValueError("non-regular evidence entry: " + str(source))

    try:
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
        write_record(directory, record)
        raise
    write_record(directory, record)
    if complete_pointer is not None:
        write_pointer(complete_pointer, directory)


def recover(task, pending, raw_log):
    """Finish retention on restart without rerunning or judging an interrupted UT."""
    outside_task(task, pending)
    previous = Path(pending.read_text().strip())
    outside_task(task, previous)
    record = json.loads((previous / "retention.json").read_text())
    if record.get("task_dir") != str(task.resolve()):
        raise ValueError("pending evidence belongs to another task")
    directory = previous
    if not record.get("complete"):
        # Never merge into or overwrite the partial archive. Preserve transport
        # logs, then copy the still-unmodified working evidence to a fresh sibling.
        directory = begin(task, previous.parent, record["kind"])
        new = json.loads((directory / "retention.json").read_text())
        new.update(recovered_from=str(previous), recovery="retention only; no verdict inferred")
        write_record(directory, new)
        for name in ("measure.stdout.log", "measure.stderr.log", "previous_measure.log"):
            source = previous / name
            if source.exists():
                if source.is_symlink() or not source.is_file():
                    raise ValueError("invalid pending transport log: " + str(source))
                shutil.copy2(source, directory / name)
        write_pointer(pending, directory)
        retain(task, directory, raw_log, record.get("measure_exit_code"), record.get("native_exit_code"))
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
    parser.add_argument("--complete-pointer", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "begin":
            print(begin(args.task, args.destination, args.label, args.pending))
        elif args.action == "recover":
            print(recover(args.task, args.destination, args.raw_log))
        else:
            retain(args.task, args.destination, args.raw_log, args.measure_exit, args.native_exit,
                   args.complete_pointer)
    except (OSError, ValueError) as exc:
        print("evidence retention failed: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
