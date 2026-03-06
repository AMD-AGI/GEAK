# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Example: Resolve a GitHub kernel URL to a local file path.

Usage:
    python examples/resolve_kernel_url/resolve_kernel_url.py \
        "https://github.com/org/repo/blob/main/kernel.py#L106"
"""

import sys

from minisweagent.tools.resolve_kernel_url_impl import (
    get_kernel_name_at_line,
    resolve_kernel_url,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: resolve_kernel_url.py <github_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Resolving: {url}")

    result = resolve_kernel_url(url)

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)

    path = result["local_file_path"]
    line = result.get("line_number")
    print(f"Local path: {path}")

    if line:
        print(f"Line number: {line}")
        name = get_kernel_name_at_line(path, line)
        if name:
            print(f"Kernel name: {name}")


if __name__ == "__main__":
    main()
