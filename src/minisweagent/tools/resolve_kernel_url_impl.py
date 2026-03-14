"""
Resolve kernel specs: detect web links (e.g. GitHub) and clone repo to a local temp path.
Used by examples/resolve_kernel_url/resolve_kernel_url.py (script entrypoint).
"""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from minisweagent.run.git_safe_env import get_git_safe_env

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


def _parse_github_source_parts(url: str) -> tuple[str, str, str] | None:
    """Return ``(owner, repo, ref_and_path)`` for GitHub blob/raw URLs."""
    parsed = urlparse(url)
    if parsed.netloc == "github.com":
        match = re.match(r"([^/]+)/([^/]+)/blob/(.+)", parsed.path.strip("/"))
        if match:
            return (match.group(1), match.group(2), match.group(3))
        return None
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/", 2)
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    return None


def _looks_like_commitish(ref: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", ref or ""))


def _github_clone_urls(owner: str, repo: str) -> list[str]:
    ssh_url = f"git@github.com:{owner}/{repo}.git"
    https_url = f"https://github.com/{owner}/{repo}.git"
    prefer_https = os.getenv("GEAK_GITHUB_PREFER_HTTPS", "").strip().lower() in {"1", "true", "yes"}
    return [https_url, ssh_url] if prefer_https else [ssh_url, https_url]


def _list_remote_refs(owner: str, repo: str) -> tuple[list[str], list[str]]:
    """Return branch/tag names from the remote, trying SSH first."""
    errors: list[str] = []
    git_env = get_git_safe_env(None)
    for clone_url in _github_clone_urls(owner, repo):
        try:
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "--tags", clone_url],
                capture_output=True,
                text=True,
                timeout=180,
                env=git_env,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"git ls-remote timed out for {clone_url}")
            continue
        except FileNotFoundError:
            errors.append("git not found")
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue
        if result.returncode != 0:
            errors.append(result.stderr or result.stdout or f"git ls-remote failed for {clone_url}")
            continue
        refs: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            refname = parts[1]
            if refname.startswith("refs/heads/"):
                refs.add(refname.removeprefix("refs/heads/"))
            elif refname.startswith("refs/tags/"):
                refs.add(refname.removeprefix("refs/tags/").removesuffix("^{}"))
        return sorted(refs, key=len, reverse=True), errors
    return [], errors


def _split_github_ref_and_path(owner: str, repo: str, ref_and_path: str) -> tuple[str, str] | None:
    if not ref_and_path or "/" not in ref_and_path:
        return None

    first, remainder = ref_and_path.split("/", 1)
    if remainder and _looks_like_commitish(first):
        return first, remainder

    refs, _errors = _list_remote_refs(owner, repo)
    for ref in refs:
        prefix = f"{ref}/"
        if ref_and_path.startswith(prefix):
            file_path = ref_and_path[len(prefix):]
            if file_path:
                return ref, file_path

    if remainder:
        return first, remainder
    return None


def parse_github_source_url(url: str) -> dict[str, str] | None:
    """Parse a GitHub blob/raw URL and resolve ``ref`` even when it contains ``/``."""
    parts = _parse_github_source_parts(url)
    if not parts:
        return None
    owner, repo, ref_and_path = parts
    resolved = _split_github_ref_and_path(owner, repo, ref_and_path)
    if not resolved:
        return None
    ref, file_path = resolved
    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "file_path": file_path,
    }


def _resolved_clone_dir(base: Path, owner: str, repo: str, ref: str) -> Path:
    ref_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ref).strip("._-") or "ref"
    ref_hash = hashlib.sha1(f"{owner}/{repo}@{ref}".encode("utf-8")).hexdigest()[:12]
    return base / RESOLVED_DIR_NAME / f"{owner}_{repo}" / f"{ref_slug}-{ref_hash}"


def _clone_remote_repo(owner: str, repo: str, ref: str, target_dir: str) -> tuple[str | None, str | None]:
    """Clone the remote repo into ``target_dir`` and return ``(clone_url, error)``."""
    errors: list[str] = []
    git_env = get_git_safe_env(Path(target_dir).parent)
    for clone_url in _github_clone_urls(owner, repo):
        if _looks_like_commitish(ref):
            clone_cmd = ["git", "clone", clone_url, target_dir]
        else:
            clone_cmd = ["git", "clone", "--depth", "1", "--branch", ref, clone_url, target_dir]

        result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=git_env,
        )
        if result.returncode != 0:
            errors.append(result.stderr or result.stdout or f"git clone failed for {clone_url}")
            shutil.rmtree(target_dir, ignore_errors=True)
            continue

        if _looks_like_commitish(ref):
            checkout = subprocess.run(
                ["git", "-C", target_dir, "checkout", ref],
                capture_output=True,
                text=True,
                timeout=180,
                env=git_env,
            )
            if checkout.returncode != 0:
                errors.append(checkout.stderr or checkout.stdout or f"git checkout failed for {ref}")
                shutil.rmtree(target_dir, ignore_errors=True)
                continue

        return clone_url, None

    error = " ; ".join(err for err in errors if err.strip()) or "git clone failed"
    return None, error


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
        "github_owner": None,
        "github_repo": None,
        "github_ref": None,
        "github_file_path": None,
        "remote_clone_url": None,
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
    parsed = parse_github_source_url(spec_no_frag)
    if not parsed:
        out["error"] = "Only GitHub blob or raw URLs are supported"
        return out

    owner = parsed["owner"]
    repo = parsed["repo"]
    branch = parsed["ref"]
    file_path = parsed["file_path"]
    out["github_owner"] = owner
    out["github_repo"] = repo
    out["github_ref"] = branch
    out["github_file_path"] = file_path
    try:
        if clone_into is not None:
            base = Path(clone_into)
            base.mkdir(parents=True, exist_ok=True)
            tmpdir_path = _resolved_clone_dir(base, owner, repo, branch)
            # Always refresh remote clones inside the target tree so a GitHub URL
            # reflects the current remote ref rather than stale local resolver state.
            if tmpdir_path.exists():
                shutil.rmtree(tmpdir_path, ignore_errors=True)
            tmpdir_path.parent.mkdir(parents=True, exist_ok=True)
            tmpdir = str(tmpdir_path)
        else:
            tmpdir = tempfile.mkdtemp(prefix=f"geak_kernel_{repo}_")
        clone_url, clone_error = _clone_remote_repo(owner, repo, branch, tmpdir)
        if clone_error:
            out["error"] = clone_error
            return out
        out["remote_clone_url"] = clone_url
        local_file = Path(tmpdir) / file_path
        if not local_file.exists():
            out["error"] = f"File not found in repo: {file_path}"
            return out
        out["local_repo_path"] = str(Path(tmpdir).resolve())
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
        "--json",
        action="store_true",
        dest="output_json",
        help="Output result as JSON (for piping to test-discovery --from-resolved)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
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
