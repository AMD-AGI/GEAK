"""
Resolve kernel specs: detect web links (e.g. GitHub) and clone repo to a local temp path.
Used by geak_agent/examples/resolve_kernel_url.py (script entrypoint).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def is_weblink(s: str) -> bool:
    """Return True if s looks like an http(s) URL."""
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")


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


def resolve_kernel_url(spec: str) -> dict:
    """
    If spec is a web link (e.g. GitHub file URL), clone the repo to a temp dir
    and return local paths. Otherwise return the spec as a local path.

    Returns dict with: is_weblink, local_repo_path (or None), local_file_path,
    original_spec, error (or None).
    """
    spec = (spec or "").strip()
    out = {
        "is_weblink": False,
        "local_repo_path": None,
        "local_file_path": spec,
        "original_spec": spec,
        "error": None,
    }
    if not spec:
        out["error"] = "Empty spec"
        return out
    if not is_weblink(spec):
        out["local_file_path"] = spec
        return out

    out["is_weblink"] = True
    parsed = _parse_github_blob(spec)
    if not parsed:
        out["error"] = "Only GitHub blob or raw URLs are supported"
        return out

    owner, repo, branch, file_path = parsed
    clone_url = f"https://github.com/{owner}/{repo}.git"
    try:
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
