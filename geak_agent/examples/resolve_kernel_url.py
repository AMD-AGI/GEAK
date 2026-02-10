#!/usr/bin/env python3
"""Resolve a kernel spec (local path or GitHub URL) to a local path. Requires git for GitHub URLs."""
import sys
from pathlib import Path

# Ensure geak_agent is importable when run as script (e.g. from repo root)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from geak_agent.resolve_kernel_url import cleanup_resolved_path, resolve_kernel_url


def main():
    if len(sys.argv) < 2:
        print("Usage: python geak_agent/examples/resolve_kernel_url.py <spec>")
        print("       ./geak_agent/examples/resolve_kernel_url.py <spec>")
        print("  spec: local path or GitHub URL (e.g. https://github.com/OWNER/REPO/blob/BRANCH/path/to/file.py)")
        sys.exit(1)
    spec = sys.argv[1].strip()
    result = resolve_kernel_url(spec)
    print("is_weblink:", result["is_weblink"])
    print("local_file_path:", result["local_file_path"])
    if result.get("local_repo_path"):
        print("local_repo_path:", result["local_repo_path"])
    if result.get("error"):
        print("error:", result["error"])
        sys.exit(2)
    if result["is_weblink"] and result.get("local_file_path"):
        path = Path(result["local_file_path"])
        if path.exists():
            print("(First 3 lines of file)")
            with open(path) as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    print(" ", line.rstrip())
        # if result.get("local_repo_path"):
        #     cleanup_resolved_path(result["local_repo_path"])


if __name__ == "__main__":
    main()
