#!/usr/bin/env python3
"""Host-local Linux flock files shared by root containers and host users.

Existing files are opened read-only: Linux flock needs no write permission.
Never truncate, unlink or replace a lock inode, including while it is held.
"""
import os
import sys


def open_lock(path):
    # These launchers are single-threaded. Publish new files with shared access
    # even under a restrictive caller umask; existing permissions stay intact.
    previous = os.umask(0)
    try:
        os.makedirs(os.path.dirname(path), mode=0o1777, exist_ok=True)
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o666)
        except FileExistsError:
            return os.open(path, os.O_RDONLY)
    except PermissionError as exc:
        raise PermissionError(
            "%s: GPU locks need a searchable shared directory and readable files; "
            "ask the owner to repair permissions in place, never replace lock files" % path) from exc
    finally:
        os.umask(previous)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        os.close(open_lock(path))
