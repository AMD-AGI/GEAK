"""GEAK v4 bootstrap — clone the repo locally and install the Claude Code CLI.

GEAK v4 is not a package you `import`; its Workflows run *inside* Claude Code
from a repo checkout. So `pip install git+https://github.com/AMD-AGI/GEAK` does
three things:

  1. pip installs the Python runtime deps (pyproject.toml [project.dependencies]).
  2. This bootstrap clones the full GEAK repo to a working dir ($GEAK_HOME).
  3. This bootstrap installs the Claude Code CLI (native installer; npm fallback).

Everything here is best-effort: a failure warns but never aborts the install.
It runs once, during the wheel build — there is no separate re-run command.

Env knobs:
  GEAK_HOME        where to clone the repo         (default: ./GEAK, under the
                   directory you run pip install from — or the current checkout
                   itself when run from inside one)
  GEAK_REPO_URL    repo to clone                   (default: the AMD-AGI repo)
  GEAK_REF         branch/tag to clone             (default: repo default branch)
  CLAUDE_VERSION   native-installer target         (default: latest)
  CLAUDE_BIN_DIR   where the CLI lands             (default: ~/.local/bin)
  GEAK_SKIP_BOOTSTRAP  set to skip step 2+3 (CI/docker image builds)

This module is imported at build time, so it must use the stdlib only.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

CLAUDE_MIN_VERSION = "2.1.177"


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


def _invocation_dir() -> str:
    """The directory the user launched from — where GEAK gets cloned by default.

    Plain os.getcwd() is wrong at wheel-build time: during `pip install git+...`
    pip has chdir'd into a throwaway temp source tree, so cwd points at something
    like /tmp/pip-req-build-xxxx. The shell still exports PWD pointing at the
    user's real directory, so prefer that when it names a real dir; fall back to
    cwd when PWD is unset.
    """
    pwd = os.environ.get("PWD", "")
    if pwd and os.path.isdir(pwd):
        return os.path.abspath(pwd)
    return os.getcwd()


def _is_geak_checkout(path: str) -> bool:
    """True when `path` is itself a GEAK repo checkout (not just any directory)."""
    return (
        os.path.isdir(os.path.join(path, ".git"))
        and os.path.isfile(os.path.join(path, "geak", "bootstrap.py"))
        and os.path.isdir(os.path.join(path, "kernel_workflow"))
    )


def _default_geak_home() -> str:
    """Where to put the repo when GEAK_HOME is not set.

    - Run from *inside* a manual clone (`git clone ... && cd GEAK && pip install .`):
      use that checkout in place, so we don't nest a redundant ./GEAK/GEAK.
    - Otherwise (`pip install git+...` from an arbitrary dir): clone into ./GEAK
      under the invocation directory.
    """
    inv = _invocation_dir()
    return inv if _is_geak_checkout(inv) else os.path.join(inv, "GEAK")


REPO_URL = _env("GEAK_REPO_URL", "https://github.com/AMD-AGI/GEAK.git")
REPO_REF = _env("GEAK_REF", "")  # empty -> the repo's default branch
# Default: ./GEAK under the directory you run pip install from — or the current
# checkout itself when run from inside one. Override with GEAK_HOME.
GEAK_HOME = os.path.abspath(os.path.expanduser(
    _env("GEAK_HOME", _default_geak_home())))
CLAUDE_VERSION = _env("CLAUDE_VERSION", "latest")
CLAUDE_BIN_DIR = os.path.abspath(os.path.expanduser(_env("CLAUDE_BIN_DIR", os.path.join("~", ".local", "bin"))))

# Bold-green styling for the copy-paste commands in the printed next-steps, but
# only when stdout is a real terminal (keep piped logs free of escape junk).
if sys.stdout.isatty():
    C_CMD, C_OFF = "\033[1;32m", "\033[0m"
else:
    C_CMD, C_OFF = "", ""


def log(msg: str) -> None:
    print("[geak-bootstrap] %s" % msg, flush=True)


def warn(msg: str) -> None:
    print("[geak-bootstrap WARN] %s" % msg, file=sys.stderr, flush=True)


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd, shell: bool = False):
    log(cmd if shell else " ".join(cmd))
    return subprocess.run(cmd, shell=shell)


# ver_ge(a, b) -> True when semver a >= b, comparing dotted fields numerically.
def _ver_tuple(s: str):
    out = []
    for part in s.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _ver_ge(a: str, b: str) -> bool:
    return _ver_tuple(a) >= _ver_tuple(b)


def claude_version() -> str:
    """Leading semver from `claude --version` (e.g. '2.1.206 (Claude Code)')."""
    try:
        out = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True
        ).stdout
    except Exception:
        return ""
    m = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", out or "")
    return m.group(0) if m else ""


# --- 1. Download the repo -------------------------------------------------

def clone_repo() -> None:
    if not _has("git"):
        warn("git not found; cannot download the GEAK repo to %s. "
             "Install git first." % GEAK_HOME)
        return

    # Running from inside the checkout itself (manual `cd GEAK && pip install .`):
    # use it in place, do not clone — respect the user's branch/worktree.
    if GEAK_HOME == _invocation_dir() and _is_geak_checkout(GEAK_HOME):
        log("already inside a GEAK checkout at %s; using it in place "
            "(skipping clone)" % GEAK_HOME)
        return

    if os.path.isdir(GEAK_HOME) and os.listdir(GEAK_HOME):
        warn("%s already exists and is not empty; leaving it untouched. "
             "Remove it or set GEAK_HOME to another path." % GEAK_HOME)
        return

    parent = os.path.dirname(GEAK_HOME) or "."
    os.makedirs(parent, exist_ok=True)
    # Shallow clone: GEAK is used as a runtime checkout, not for history, and the
    # repo is large (perf_knowledge + example artifacts). --depth 1 keeps it fast.
    cmd = ["git", "clone", "--depth", "1"]
    if REPO_REF:
        cmd += ["--branch", REPO_REF]
    cmd += [REPO_URL, GEAK_HOME]
    if _run(cmd).returncode == 0:
        log("GEAK repo downloaded to %s" % GEAK_HOME)
    else:
        warn("git clone failed; check network/credentials.")


# --- 2. Claude Code CLI (native installer; npm fallback) ------------------

def _install_claude_native() -> bool:
    if not (_has("curl") or _has("npm")):
        warn("need curl (native installer) or npm to install Claude Code. "
             "Install one, or install manually: https://code.claude.com/docs/en/setup")
        return False

    if _has("curl"):
        # The native installer pulls a ~260MB binary via a silent curl, so on a
        # slow link it can sit for minutes with no output — a slow download, NOT
        # a hang. We deliberately do not cap its total time.
        log("installing Claude Code (%s) via the native installer" % CLAUDE_VERSION)
        cmd = ("curl -fsSL --connect-timeout 20 https://claude.ai/install.sh "
               "| bash -s %s" % CLAUDE_VERSION)
        if _run(cmd, shell=True).returncode == 0:
            return True
        warn("native installer failed")

    if _has("npm"):
        warn("falling back to npm: npm install -g @anthropic-ai/claude-code")
        if _run(["npm", "install", "-g", "@anthropic-ai/claude-code"]).returncode == 0:
            return True

    warn("could not install Claude Code. Check network access or install "
         "manually: https://code.claude.com/docs/en/setup")
    return False


def ensure_claude_code() -> None:
    cur = claude_version()
    if cur and _ver_ge(cur, CLAUDE_MIN_VERSION):
        log("Claude Code present (%s) >= %s" % (cur, CLAUDE_MIN_VERSION))
        return

    if cur:
        warn("Claude Code %s is older than %s; updating" % (cur, CLAUDE_MIN_VERSION))
        _run(["claude", "update"])
        cur = claude_version()
        if cur and _ver_ge(cur, CLAUDE_MIN_VERSION):
            log("Claude Code updated to %s" % cur)
            return
    else:
        warn("Claude Code CLI not found")

    _install_claude_native()

    cur = claude_version()
    if not cur:
        warn("claude not on PATH after install; ensure '%s' is on your PATH" % CLAUDE_BIN_DIR)
    elif not _ver_ge(cur, CLAUDE_MIN_VERSION):
        warn("installed Claude Code %s is still < %s; run 'claude update' or set "
             "CLAUDE_VERSION" % (cur, CLAUDE_MIN_VERSION))


# --- 3. Environment prerequisites (detect only) --------------------------

def check_environment() -> None:
    log("checking ROCm / profiler / serving-backend prerequisites (detect only)")

    if _has("rocminfo") or _has("rocm-smi"):
        log("  ROCm: present")
    else:
        warn("  ROCm not detected (rocminfo/rocm-smi missing). GEAK targets AMD "
             "Instinct MI GPUs; install ROCm 6+.")

    profiler = next((p for p in ("rocprof-compute", "rocprofv3", "rocprof", "metrix") if _has(p)), "")
    if profiler:
        log("  profiler: %s" % profiler)
    else:
        warn("  no profiler found (rocprof-compute/rocprofv3/rocprof/metrix). "
             "Profiling steps will be degraded.")

    backend = ""
    for mod in ("sglang", "vllm"):
        if subprocess.run([sys.executable, "-c", "import %s" % mod],
                          capture_output=True).returncode == 0:
            backend = mod
            break
    if backend:
        log("  serving backend: %s" % backend)
    else:
        warn("  no serving backend (sglang/vllm) importable. Required for e2e_workflow only.")


# --- 4. Next steps -------------------------------------------------------

def print_next_steps() -> None:
    on_path = any(os.path.abspath(p) == CLAUDE_BIN_DIR for p in os.environ.get("PATH", "").split(os.pathsep) if p)
    if not on_path and os.path.isfile(os.path.join(CLAUDE_BIN_DIR, "claude")):
        print(
            "\n[geak-bootstrap] NOTE: Claude Code is installed at %s, which is not on\n"
            "your PATH. Add it (use ~/.zshrc for zsh):\n"
            "    %secho 'export PATH=\"%s:$PATH\"' >> ~/.bashrc && source ~/.bashrc%s"
            % (CLAUDE_BIN_DIR, C_CMD, CLAUDE_BIN_DIR, C_OFF)
        )

    print(
        "\n[geak-bootstrap] setup complete.\n\n"
        "Next steps — configure Claude Code, then launch it:\n\n"
        "1) Give Claude Code API access (pick ONE):\n\n"
        "   a. Anthropic API directly:\n"
        "        %sexport ANTHROPIC_API_KEY=sk-ant-...%s\n\n"
        "   b. A gateway / proxy (OpenAI- or Anthropic-compatible):\n"
        "        %sexport ANTHROPIC_BASE_URL=https://your-gateway.example.com%s\n"
        "        %sexport ANTHROPIC_AUTH_TOKEN=your-token%s\n\n"
        "   c. Interactive login (Claude / Anthropic Console account):\n"
        "        %sclaude%s            # then run: /login   and follow the browser flow\n\n"
        "   (Persist your choice in ~/.bashrc so future shells inherit it.)\n\n"
        "2) Launch Claude Code in auto-approve mode from the repo root:\n"
        "     %scd %s%s\n"
        "     %sIS_SANDBOX=1 claude --dangerously-skip-permissions%s"
        % (C_CMD, C_OFF, C_CMD, C_OFF, C_CMD, C_OFF, C_CMD, C_OFF,
           C_CMD, GEAK_HOME, C_OFF, C_CMD, C_OFF)
    )


def main() -> None:
    if os.environ.get("GEAK_SKIP_BOOTSTRAP"):
        log("GEAK_SKIP_BOOTSTRAP set; skipping repo clone and Claude Code install")
        return
    log("GEAK_HOME=%s" % GEAK_HOME)
    clone_repo()
    ensure_claude_code()
    check_environment()
    print_next_steps()
