#!/usr/bin/env python3
"""Run reverse-knowledge InteractiveAgent in yolo mode (loaded from mini_reverse_kl.yaml).

Workspace is always GEAK_ROOT / REVERSE_KL_WORKSPACE_REL (AMD user-case knowledge folder).
That resolved path is the agent cwd and is injected at the top of the task text.

Environment (set by run-reverse_knowledge.sh):
  GEAK_ROOT               Path to GEAK package root
  GEAK_REVERSE_KL_CONFIG  Absolute path to mini_reverse_kl.yaml
  GEAK_REVERSE_KL_TASK_FILE  Path to a temp file containing the task body (after bash preamble)

API keys are not read from the YAML (model.api_key is null). For model_class amd_llm, export
AMD_LLM_API_KEY or LLM_GATEWAY_KEY or MSWEA_MODEL_API_KEY (see GEAK/README.md). ~/.config/mini-swe-agent/.env
is still loaded by minisweagent for dotenv.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import fields
from pathlib import Path

import yaml

# Must match DEFAULT_REL_WS in run-reverse_knowledge.sh — only outputs under this tree.
REVERSE_KL_WORKSPACE_REL = (
    "mcp_tools/rag-mcp/knowledge-base/amd-knowledge-base/"
    "layer-6-extended/optimize-guides/user-case/user"
)


def _apply_tool_disables(config: dict) -> None:
    tools_cfg = config.get("tools") or {}
    disabled_tools: list[str] = []
    if tools_cfg.get("bash") is False:
        disabled_tools.append("bash")
    if tools_cfg.get("profiling") is False:
        disabled_tools.append("profiling")
        disabled_tools.append("profile_kernel")
    rag_enabled = tools_cfg.get("rag", False)
    if rag_enabled:
        try:
            import rag_mcp  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "RAG is enabled in config but rag-mcp is not installed.\n"
                "  pip install -e mcp_tools/rag-mcp"
            ) from e
        _index_path = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
        _has_faiss = (_index_path / "index.faiss").exists() or (_index_path / "faiss.index").exists()
        _has_pkl = bool(list(_index_path.glob("*.pkl"))) if _index_path.exists() else False
        if not (_has_faiss and _has_pkl):
            raise RuntimeError(
                f"RAG enabled but index missing at {_index_path}. Build index first:\n"
                "  python scripts/build_index.py --force"
            )
    if not rag_enabled:
        disabled_tools.append("query")
        disabled_tools.append("optimize")
    if disabled_tools:
        config.setdefault("agent", {}).setdefault("disabled_tools", [])
        config["agent"]["disabled_tools"] = list(set(config["agent"]["disabled_tools"]) | set(disabled_tools))


def _canonical_knowledge_workspace(geak_root: Path) -> Path:
    return (geak_root / REVERSE_KL_WORKSPACE_REL).resolve()


def _strip_placeholder_api_key(model_cfg: dict) -> None:
    """Remove api_key from YAML when it is null, empty, or the literal 'None'/'null'.

    Secrets must come from the environment (GEAK/README.md), not the checked-in file.
    """
    if "api_key" not in model_cfg:
        return
    v = model_cfg["api_key"]
    if v is None:
        del model_cfg["api_key"]
        return
    if isinstance(v, str) and (not v.strip() or v.strip().lower() in ("none", "null")):
        del model_cfg["api_key"]


def _preflight_model_credentials(model_cfg: dict) -> None:
    """Ensure amd_llm has a key source before constructing the model (README env vars)."""
    model_class = str(model_cfg.get("model_class") or "").strip().lower()
    if model_class not in ("", "amd_llm"):
        return
    mk = model_cfg.get("model_kwargs") or {}
    if model_cfg.get("api_key"):
        return
    if mk.get("api_key"):
        return
    if os.getenv("MSWEA_MODEL_API_KEY"):
        return
    if os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY"):
        return
    print(
        "error: no API credentials for amd_llm (api_key is null in mini_reverse_kl.yaml).\n"
        "Export one of the following (see GEAK/README.md):\n"
        "  AMD_LLM_API_KEY or LLM_GATEWAY_KEY  — AMD LLM gateway\n"
        "  MSWEA_MODEL_API_KEY                 — generic override used by get_model()\n",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    geak_root = Path(os.environ["GEAK_ROOT"]).resolve()
    cfg_path = Path(os.environ["GEAK_REVERSE_KL_CONFIG"]).resolve()
    task_path = Path(os.environ["GEAK_REVERSE_KL_TASK_FILE"])

    repo = _canonical_knowledge_workspace(geak_root)
    env_ws = os.environ.get("REVERSE_KL_PATH")
    if env_ws:
        try:
            if Path(env_ws).resolve() != repo:
                print(
                    f"[reverse_knowledge] warning: REVERSE_KL_PATH={env_ws!r} ignored; "
                    f"using canonical workspace {repo}",
                    file=sys.stderr,
                )
        except OSError:
            print(
                f"[reverse_knowledge] warning: invalid REVERSE_KL_PATH={env_ws!r}; using {repo}",
                file=sys.stderr,
            )

    src = geak_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from minisweagent.run.extra.config import configure_if_first_time
    from minisweagent.models import get_model
    from minisweagent.environments import get_environment_class
    from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig

    configure_if_first_time()

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    print(f"[reverse_knowledge] using agent YAML: {cfg_path}", file=sys.stderr)
    _apply_tool_disables(config)

    model_cfg = config.setdefault("model", {})
    _strip_placeholder_api_key(model_cfg)
    _preflight_model_credentials(model_cfg)

    task_body = task_path.read_text(encoding="utf-8")
    if not task_body.strip():
        print("error: empty task", file=sys.stderr)
        sys.exit(1)

    repo.mkdir(parents=True, exist_ok=True)

    _q = shlex.quote(str(repo))
    preamble = (
        "## Mandatory knowledge output directory\n\n"
        "Create **all** per-kernel directories, `Report_*.md`, `Report_*_simplified.md`, and `Path_*.txt` "
        "**only** under this absolute path (no other root for deliverables):\n\n"
        f"    {repo}\n\n"
        "Your shell cwd for this task is that directory. Each subshell starts fresh: when writing files, "
        f"use either paths under that directory or prefix commands with `cd {_q} &&`.\n\n"
        "---\n\n"
    )
    task = preamble + task_body

    model = get_model(None, config.get("model", {}))

    _env_kwargs = dict(config.get("env", {}))
    _env_kwargs.setdefault("cwd", str(repo))
    env_type = str(_env_kwargs.pop("type", _env_kwargs.pop("environment_class", "local"))).strip().lower() or "local"
    env_class = get_environment_class(env_type)
    env = env_class(**_env_kwargs)

    agent_section = dict(config.get("agent") or {})
    allowed = {f.name for f in fields(InteractiveAgentConfig)}
    agent_kwargs = {k: v for k, v in agent_section.items() if k in allowed}
    if "step_limit" in agent_kwargs and agent_kwargs["step_limit"] is not None:
        agent_kwargs["step_limit"] = int(float(agent_kwargs["step_limit"]))
    agent_kwargs["save_patch"] = False
    agent_kwargs["patch_output_dir"] = None
    agent_kwargs["use_strategy_manager"] = False
    agent_kwargs["test_command"] = None
    agent_kwargs["metric"] = None
    # InteractiveAgent prints each turn to the terminal; yolo skips action/exit prompts (like geak -y).
    agent_kwargs["mode"] = "yolo"

    agent = InteractiveAgent(model, env, **agent_kwargs)
    exit_status, msg = agent.run(task.strip())
    print(f"[run-reverse_knowledge] finished: {exit_status}", file=sys.stderr)
    if msg:
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()