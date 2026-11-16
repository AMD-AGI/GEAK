#!/usr/bin/env python3
"""Lease every requested GPU using the existing single-device lock contract.

Run from the task directory. Unlike gpu_lock.sh's comma-separated pool, this
wrapper holds the entire set until the command exits. GPU IDs retain the
caller's allocation mapping; an inherited ROCR mask is not replaced.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_set", help="comma-separated IDs from this run's allocated set")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raw = args.gpu_set.split(",")
    if not raw or any(not value.strip().isascii() or not value.strip().isdigit() for value in raw):
        parser.error("GPU IDs must be nonnegative integers")
    ids = [str(int(value)) for value in raw]
    if len(set(ids)) != len(ids):
        parser.error("GPU set contains duplicate IDs")
    allowed = os.environ.get("GEAK_GPU_ALLOWED", "")
    if allowed and not set(ids).issubset(value.strip() for value in allowed.split(",")):
        parser.error(f"requested set {','.join(ids)} is outside GEAK_GPU_ALLOWED={allowed}")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a foreground command is required")
    lock = Path(__file__).resolve().parent / "gpu_lock.sh"
    # The wrappers stay alive, retaining their individual flock descriptors and
    # source-provenance exit checks. A common acquisition order prevents cycles.
    # Restore visibility only inside the innermost, fully leased command.
    requested = ",".join(ids)
    nested = ["env", f"HIP_VISIBLE_DEVICES={requested}", f"GEAK_GPU_LOCKED_SET={requested}", *command]
    for gpu in reversed(sorted(ids, key=int)):
        nested = ["bash", str(lock), gpu, *nested]
    os.execvp(nested[0], nested)


if __name__ == "__main__":
    main()
