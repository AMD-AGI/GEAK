from pathlib import Path

from minisweagent.tools.registry import EnvRegistry


def test_env_registry_default_path_is_user_writable(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_ENV_FILE", raising=False)

    registry = EnvRegistry()
    path = registry.env_file

    assert path.name.startswith("swe-agent-env-")
    assert path != Path("/root/.swe-agent-env")
    assert path.exists()
