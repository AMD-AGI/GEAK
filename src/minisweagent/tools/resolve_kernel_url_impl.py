"""
Resolve kernel specs: detect web links (e.g. GitHub) and clone repo to a local temp path.
Used by examples/resolve_kernel_url/resolve_kernel_url.py (script entrypoint).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Canonical name of the directory used to cache cloned repos.
# Other modules (discovery, mini.py) import this constant to detect
# whether a kernel path lives inside a resolved clone.
RESOLVED_DIR_NAME = ".geak_resolved"


def find_resolved_clone_root(file_path: str | Path) -> Path | None:
    """Return the clone-root directory if *file_path* lives inside a resolved clone.

    For example, given ``/workspace/.geak_resolved/owner_repo/sub/kernel.py``,
    this returns ``Path('/workspace/.geak_resolved/owner_repo')``.

    Returns ``None`` when the path is not inside a resolved clone.
    """
    path = Path(file_path).resolve()
    parts = path.parts
    try:
        idx = parts.index(RESOLVED_DIR_NAME)
    except ValueError:
        return None
    # The clone root is the directory immediately after RESOLVED_DIR_NAME
    if idx + 1 < len(parts):
        return Path(*parts[: idx + 2])
    return None


def is_weblink(s: str) -> bool:
    """Return True if s looks like an http(s) URL."""
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")


def _parse_fragment(spec: str) -> tuple[int | None, int | None]:
    """Parse #L106 or #L106-L108 from spec. Returns (line_start, line_end) or (None, None)."""
    if "#" not in spec:
        return (None, None)
    frag = spec.split("#", 1)[1].strip()
    if not frag.startswith("L"):
        return (None, None)
    part = frag[1:].strip()
    if "-" in part:
        a, b = part.split("-", 1)
        try:
            return (int(a.strip()), int(b.strip().lstrip("L")))
        except ValueError:
            return (None, None)
    try:
        return (int(part), int(part))
    except ValueError:
        return (None, None)


def _strip_fragment(spec: str) -> str:
    """Return spec without #L123 fragment."""
    return spec.split("#", 1)[0].rstrip()


def _parse_github_blob(url: str) -> tuple[str, str, str, str] | None:
    """
    Parse GitHub blob URL: https://github.com/OWNER/REPO/blob/BRANCH/PATH
    Returns (owner, repo, branch, file_path) or None.
    """
    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "raw.githubusercontent.com"):
        return None
    path = parsed.path.strip("/")
    if "raw.githubusercontent.com" in parsed.netloc:
        parts = path.split("/", 3)
        if len(parts) >= 4:
            return (parts[0], parts[1], parts[2], parts[3])
        return None
    match = re.match(r"([^/]+)/([^/]+)/blob/([^/]+)/(.+)", path)
    if match:
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    return None


def resolve_kernel_url(spec: str, clone_into: str | Path | None = None) -> dict:
    """
    If spec is a web link (e.g. GitHub file URL), clone the repo to a temp dir
    (or into clone_into/{RESOLVED_DIR_NAME}/<repo> if clone_into is set) and return local paths.
    Otherwise return the spec as a local path.

    Returns dict with: is_weblink, local_repo_path (or None), local_file_path,
    original_spec, line_number (int or None), line_end (int or None), error (or None).
    """
    spec = (spec or "").strip()
    line_start, line_end = _parse_fragment(spec)
    spec_no_frag = _strip_fragment(spec)
    out = {
        "is_weblink": False,
        "local_repo_path": None,
        "local_file_path": spec_no_frag,
        "original_spec": spec,
        "line_number": line_start,
        "line_end": line_end,
        "error": None,
    }
    if not spec_no_frag:
        out["error"] = "Empty spec"
        return out
    if not is_weblink(spec_no_frag):
        out["local_file_path"] = spec_no_frag
        out["line_number"] = line_start
        out["line_end"] = line_end
        return out

    out["is_weblink"] = True
    parsed = _parse_github_blob(spec_no_frag)
    if not parsed:
        out["error"] = "Only GitHub blob or raw URLs are supported"
        return out

    owner, repo, branch, file_path = parsed
    clone_url = f"https://github.com/{owner}/{repo}.git"
    try:
        if clone_into is not None:
            base = Path(clone_into)
            base.mkdir(parents=True, exist_ok=True)
            tmpdir_path = base / RESOLVED_DIR_NAME / f"{owner}_{repo}"
            # If the target file already exists from a previous clone, reuse it
            if tmpdir_path.exists() and (tmpdir_path / file_path).exists():
                out["local_repo_path"] = str(tmpdir_path)
                out["local_file_path"] = str((tmpdir_path / file_path).resolve())
                out["line_number"] = line_start
                out["line_end"] = line_end
                return out
            # Remove any leftover partial clone before re-cloning
            if tmpdir_path.exists():
                shutil.rmtree(tmpdir_path, ignore_errors=True)
            tmpdir = str(tmpdir_path)
        else:
            tmpdir = tempfile.mkdtemp(prefix=f"geak_kernel_{repo}_")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, clone_url, tmpdir],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            out["error"] = result.stderr or result.stdout or "git clone failed"
            return out
        local_file = Path(tmpdir) / file_path
        if not local_file.exists():
            out["error"] = f"File not found in repo: {file_path}"
            return out
        out["local_repo_path"] = tmpdir
        out["local_file_path"] = str(local_file.resolve())
        out["line_number"] = line_start
        out["line_end"] = line_end
        return out
    except subprocess.TimeoutExpired:
        out["error"] = "git clone timed out"
        return out
    except FileNotFoundError:
        out["error"] = "git not found; install git to clone from URLs"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def get_kernel_name_at_line(file_path: str | Path, line_number: int) -> str | None:
    """
    Return the name of the kernel (e.g. @triton.jit function or def) that contains the given line.
    Scans the file for def/async def; returns the innermost function name that spans line_number.
    Returns None if not found.
    """
    path = Path(file_path)
    if not path.exists() or line_number < 1:
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    # Build (name, start_line, end_line) for each top-level def
    funcs: list[tuple[str, int, int]] = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            match = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", stripped)
            if match:
                if funcs:
                    prev_name, prev_start, _ = funcs[-1]
                    funcs[-1] = (prev_name, prev_start, i)
                funcs.append((match.group(1), i, len(lines) + 1))
    if funcs and len(funcs) > 1:
        funcs[-1] = (funcs[-1][0], funcs[-1][1], len(lines) + 1)
    for name, start, end in reversed(funcs):
        if start <= line_number < end:
            return name
    return None


def cleanup_resolved_path(local_repo_path: str | None) -> None:
    """Remove a previously cloned temp repo."""
    if not local_repo_path:
        return
    path = Path(local_repo_path)
    if path.is_dir() and "geak_kernel_" in path.name:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


# ============================================================================
# CLI
# ============================================================================

def main():
    """Resolve a GitHub kernel URL to a local file path."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Clone a GitHub repo and resolve a kernel URL to a local path",
    )
    parser.add_argument("url", help="GitHub file URL (blob link)")
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output result as JSON (for piping to test-discovery --from-resolved)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write output to file instead of stdout (implies --json)",
    )
    args = parser.parse_args()

    use_json = args.output_json or args.output is not None

    print(f"Resolving: {args.url}", file=sys.stderr)
    result = resolve_kernel_url(args.url)

    if result.get("error"):
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if use_json:
        output_text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output_text + "\n")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(output_text)
    else:
        local_path = result["local_file_path"]
        line = result.get("line_number")
        repo_root = result.get("local_repo_path")

        print(f"Local path:  {local_path}")
        if repo_root:
            print(f"Repo root:   {repo_root}")
        if line:
            print(f"Line number: {line}")
            name = get_kernel_name_at_line(local_path, line)
            if name:
                print(f"Kernel name: {name}")


if __name__ == "__main__":
    main()
