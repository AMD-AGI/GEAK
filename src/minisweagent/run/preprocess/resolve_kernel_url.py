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

from minisweagent.run.utils.git_safe_env import get_git_safe_env

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
    ref_hash = hashlib.sha1(f"{owner}/{repo}@{ref}".encode()).hexdigest()[:12]
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


def _parse_github_repo_url(url: str) -> dict[str, str] | None:
    """Parse a plain GitHub repo URL like ``https://github.com/org/repo[/tree/ref]``."""
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return None
    owner, repo_name = parts[0], parts[1]
    ref = "main"
    if len(parts) >= 4 and parts[2] == "tree":
        ref = "/".join(parts[3:])
    return {"owner": owner, "repo": repo_name, "ref": ref}


def _clone_github_and_find(
    out: dict,
    owner: str,
    repo_name: str,
    ref: str,
    file_path: str,
    clone_into: str | Path | None,
    line_start: int | None,
    line_end: int | None,
) -> dict:
    """Clone a GitHub repo and locate *file_path* within it.

    Populates *out* with GitHub metadata, clone paths, and the resolved
    local file path.  Returns *out* (mutated).
    """
    out.update(
        is_weblink=True,
        github_owner=owner,
        github_repo=repo_name,
        github_ref=ref,
        github_file_path=file_path,
    )
    try:
        if clone_into is not None:
            base = Path(clone_into)
            base.mkdir(parents=True, exist_ok=True)
            tmpdir_path = _resolved_clone_dir(base, owner, repo_name, ref)
            if tmpdir_path.exists():
                shutil.rmtree(tmpdir_path, ignore_errors=True)
            tmpdir_path.parent.mkdir(parents=True, exist_ok=True)
            tmpdir = str(tmpdir_path)
        else:
            tmpdir = tempfile.mkdtemp(prefix=f"geak_kernel_{repo_name}_")
        clone_url, clone_error = _clone_remote_repo(owner, repo_name, ref, tmpdir)
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

def resolve_kernel_url(
    spec: str,
    repo: str | None = None,
    clone_into: str | Path | None = None,
) -> dict:
    """Resolve a kernel file specification to local paths.

    Parameters
    ----------
    spec:
        Kernel identifier -- a GitHub blob/raw URL
        (e.g. ``https://github.com/org/repo/blob/main/ops/kernel.py#L42``)
        or a local file path (absolute, or relative to *repo* / CWD).
    repo:
        Optional repository root -- a local directory path or a GitHub URL.
        When provided, *spec* is resolved relative to *repo* for local
        paths, or within the cloned checkout for remote URLs.  This
        ensures ``local_repo_path`` is always populated when *repo* is
        given, avoiding the fallback to ``spec``'s parent directory.
    clone_into:
        Directory for cloning remote repositories.  When set, clones land
        in ``clone_into/.geak_resolved/<repo>/`` and are refreshed on
        every call.  When ``None``, a temporary directory is used.

    Returns
    -------
    dict with keys:
        ``is_weblink``, ``local_repo_path``, ``local_file_path``,
        ``original_spec``, ``line_number``, ``line_end``, ``error``,
        ``github_owner``, ``github_repo``, ``github_ref``,
        ``github_file_path``, ``remote_clone_url``.
    """
    import logging

    logger = logging.getLogger(__name__)

    spec = str(spec).strip() if spec else ""
    repo = str(repo).strip() if repo else ""

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

    # ── classify inputs ───────────────────────────────────────────
    spec_is_url = is_weblink(spec_no_frag)
    repo_is_url = is_weblink(repo) if repo else False
    spec_is_relative = not spec_is_url and not Path(spec_no_frag).is_absolute()

    # ── validate impossible combinations ──────────────────────────
    if repo_is_url and not spec_is_url and not spec_is_relative:
        out["error"] = (
            "Cannot combine a remote --repo URL with an absolute local "
            f"kernel path: {spec_no_frag}"
        )
        return out

    if repo_is_url and spec_is_url:
        logger.info("Both --repo and kernel spec are URLs; ignoring --repo")
        repo = None
        repo_is_url = False

    # ── Case 1: spec is a URL ────────────────────────────────────
    if spec_is_url:
        if repo and not repo_is_url:
            logger.warning(
                "--repo is a local path but kernel spec is a remote URL; "
                "ignoring --repo for resolution"
            )
        out["is_weblink"] = True
        parsed = parse_github_source_url(spec_no_frag)
        if not parsed:
            out["error"] = "Only GitHub blob or raw URLs are supported"
            return out
        return _clone_github_and_find(
            out, parsed["owner"], parsed["repo"], parsed["ref"],
            parsed["file_path"], clone_into, line_start, line_end,
        )

    # ── Case 2: repo is a URL, spec is relative ──────────────────
    if repo_is_url:
        parsed_repo = _parse_github_repo_url(repo)
        if not parsed_repo:
            out["error"] = f"Unsupported --repo URL: {repo}"
            return out
        return _clone_github_and_find(
            out, parsed_repo["owner"], parsed_repo["repo"],
            parsed_repo["ref"], spec_no_frag, clone_into,
            line_start, line_end,
        )

    # ── Case 3: both local ───────────────────────────────────────
    base = Path(repo).resolve() if repo else Path.cwd()
    if repo and not base.is_dir():
        out["error"] = f"--repo directory not found: {base}"
        return out

    kernel_path = Path(spec_no_frag)
    if kernel_path.is_absolute():
        kernel_path = kernel_path.resolve()
        if repo:
            try:
                kernel_path.relative_to(base)
            except ValueError:
                logger.warning(
                    "Kernel %s is outside --repo %s", kernel_path, base,
                )
    else:
        kernel_path = (base / kernel_path).resolve()

    if not kernel_path.is_file():
        out["error"] = f"Kernel file not found: {kernel_path}"
        return out

    out["local_repo_path"] = str(base) if repo else None
    out["local_file_path"] = str(kernel_path)
    out["line_number"] = line_start
    out["line_end"] = line_end
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
