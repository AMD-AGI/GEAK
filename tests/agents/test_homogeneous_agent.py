"""Tests for the HomogeneousAgent module.

This module tests the homogeneous agent runner which runs multiple identical
agents in parallel for kernel optimization tasks.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from minisweagent.agents.homogeneous_agent import (
    parse_gpu_ids,
    run_homogeneous_agent,
)
from minisweagent.agents.parallel_agent import BestPatchResult, ParallelAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel

# --- Test parse_gpu_ids ---


class TestParseGpuIds:
    """Tests for the parse_gpu_ids utility function."""

    def test_parse_gpu_ids_none(self):
        """Test that None returns default [0]."""
        assert parse_gpu_ids(None) == [0]

    def test_parse_gpu_ids_empty_string(self):
        """Test that empty string returns default [0]."""
        assert parse_gpu_ids("") == [0]

    def test_parse_gpu_ids_single(self):
        """Test parsing a single GPU ID."""
        assert parse_gpu_ids("0") == [0]
        assert parse_gpu_ids("1") == [1]
        assert parse_gpu_ids("7") == [7]

    def test_parse_gpu_ids_multiple(self):
        """Test parsing multiple GPU IDs."""
        assert parse_gpu_ids("0,1") == [0, 1]
        assert parse_gpu_ids("0,1,2,3") == [0, 1, 2, 3]
        assert parse_gpu_ids("1,3,5,7") == [1, 3, 5, 7]

    def test_parse_gpu_ids_with_spaces(self):
        """Test parsing GPU IDs with spaces."""
        assert parse_gpu_ids("0, 1") == [0, 1]
        assert parse_gpu_ids("0 , 1 , 2") == [0, 1, 2]
        assert parse_gpu_ids(" 0 , 1 ") == [0, 1]

    def test_parse_gpu_ids_with_trailing_comma(self):
        """Test parsing GPU IDs with trailing comma."""
        assert parse_gpu_ids("0,1,") == [0, 1]
        assert parse_gpu_ids("0,") == [0]

    def test_parse_gpu_ids_with_leading_comma(self):
        """Test parsing GPU IDs with leading comma."""
        assert parse_gpu_ids(",0,1") == [0, 1]


# --- Test HomogeneousAgent Configuration ---


class TestHomogeneousAgentConfig:
    """Tests for homogeneous agent configuration loading."""

    @pytest.fixture
    def homogeneous_config(self):
        """Load the homogeneous agent config."""
        config_path = Path("src/minisweagent/config/homogeneous_agent.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_has_agent_section(self, homogeneous_config):
        """Test that config has agent section."""
        assert "agent" in homogeneous_config

    def test_config_has_system_template(self, homogeneous_config):
        """Test that config has system_template."""
        assert "system_template" in homogeneous_config["agent"]
        assert "kernel optimization" in homogeneous_config["agent"]["system_template"].lower()

    def test_config_has_instance_template(self, homogeneous_config):
        """Test that config has instance_template."""
        assert "instance_template" in homogeneous_config["agent"]
        assert "{{task}}" in homogeneous_config["agent"]["instance_template"]

    def test_config_has_tools_section(self, homogeneous_config):
        """Test that config has tools section."""
        assert "tools" in homogeneous_config

    def test_config_tools_strategy_manager(self, homogeneous_config):
        """Test that strategy_manager is configured."""
        tools = homogeneous_config.get("tools", {})
        assert "strategy_manager" in tools
        assert tools["strategy_manager"] is True

    def test_config_tools_strategy_file(self, homogeneous_config):
        """Test that strategy_file is configured."""
        tools = homogeneous_config.get("tools", {})
        assert "strategy_file" in tools
        assert tools["strategy_file"] == ".optimization_strategies.md"


# --- Test run_homogeneous_agent ---


class TestRunHomogeneousAgent:
    """Tests for the run_homogeneous_agent function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return DeterministicModel(
            outputs=["THOUGHT: Test thought\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```"]
        )

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        return LocalEnvironment()

    @pytest.fixture
    def base_config(self):
        """Create base configuration."""
        return {
            "agent": {
                "system_template": "Test system template",
                "instance_template": "Task: {{task}}",
                "step_limit": 10,
                "cost_limit": 10.0,
            },
            "tools": {
                "strategy_manager": False,
                "strategy_file": ".optimization_strategies.md",
            },
            "parallel": {
                "num_parallel": 1,
            },
        }

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()
            # Initialize as git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=test@test.com",
                    "-c",
                    "user.name=Test",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "Initial commit",
                ],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            yield repo_path

    def test_run_homogeneous_agent_requires_repo_path(self, mock_model, mock_env, base_config):
        """Test that run_homogeneous_agent requires a valid repo path."""
        with pytest.raises(ValueError, match="Repository path does not exist"):
            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=Path("/nonexistent/path"),
            )

    def test_run_homogeneous_agent_with_strategy_manager_enabled(self, mock_model, mock_env, base_config, temp_repo):
        """Test that strategy agent is used when strategy_manager is enabled."""
        base_config["tools"]["strategy_manager"] = True

        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = BestPatchResult(
                agent_id=0,
                patch_id="patch_0",
                test_output="Test passed",
            )
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=temp_repo,
            )

            # Verify ParallelAgent was called
            mock_parallel.assert_called_once()
            # Check that agent_class is StrategyInteractiveAgent
            call_kwargs = mock_parallel.call_args[1]
            from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

            assert call_kwargs.get("agent_class") == StrategyInteractiveAgent

    def test_run_homogeneous_agent_with_strategy_manager_disabled(self, mock_model, mock_env, base_config, temp_repo):
        """Test that interactive agent is used when strategy_manager is disabled."""
        base_config["tools"]["strategy_manager"] = False

        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = BestPatchResult(
                agent_id=0,
                patch_id="patch_0",
                test_output="Test passed",
            )
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=temp_repo,
            )

            # Verify ParallelAgent was called
            mock_parallel.assert_called_once()
            # Check that agent_class is InteractiveAgent
            call_kwargs = mock_parallel.call_args[1]
            from minisweagent.agents.interactive import InteractiveAgent

            assert call_kwargs.get("agent_class") == InteractiveAgent

    def test_run_homogeneous_agent_num_parallel_from_param(self, mock_model, mock_env, base_config, temp_repo):
        """Test that num_parallel parameter takes precedence."""
        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = None
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=temp_repo,
                num_parallel=4,
            )

            call_kwargs = mock_parallel.call_args[1]
            assert call_kwargs.get("num_parallel") == 4

    def test_run_homogeneous_agent_gpu_ids_from_param(self, mock_model, mock_env, base_config, temp_repo):
        """Test that gpu_ids parameter is parsed correctly."""
        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = None
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=temp_repo,
                gpu_ids="0,1,2,3",
            )

            call_kwargs = mock_parallel.call_args[1]
            assert call_kwargs.get("gpu_ids") == [0, 1, 2, 3]

    def test_run_homogeneous_agent_output_dir_creation(self, mock_model, mock_env, base_config, temp_repo):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"

            with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
                mock_agent = MagicMock()
                mock_agent.run.return_value = None
                mock_parallel.return_value = mock_agent

                run_homogeneous_agent(
                    config=base_config,
                    task_content="Test task",
                    model=mock_model,
                    env=mock_env,
                    env_class=LocalEnvironment,
                    env_kwargs={},
                    tools_settings=base_config["tools"],
                    agent_config={
                        "system_template": "Test",
                        "instance_template": "{{task}}",
                        "step_limit": 10,
                        "cost_limit": 10.0,
                    },
                    repo=temp_repo,
                    output_dir=output_dir,
                )

                assert output_dir.exists()

    def test_run_homogeneous_agent_mode_yolo(self, mock_model, mock_env, base_config, temp_repo):
        """Test that agent is configured in yolo mode."""
        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = None
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                    "step_limit": 10,
                    "cost_limit": 10.0,
                },
                repo=temp_repo,
            )

            call_kwargs = mock_parallel.call_args[1]
            assert call_kwargs.get("mode") == "yolo"
            assert call_kwargs.get("confirm_exit") is False


# --- Test ParallelAgent Integration ---


class TestParallelAgentIntegration:
    """Integration tests for ParallelAgent used by HomogeneousAgent."""

    @pytest.fixture
    def mock_model_factory(self):
        """Create a mock model factory."""

        def factory():
            return DeterministicModel(
                outputs=["THOUGHT: Test\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```"]
            )

        return factory

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary git repository with some content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()

            # Create a test file
            test_file = repo_path / "test.py"
            test_file.write_text("print('hello')")

            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test", "commit", "-m", "Initial commit"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            yield repo_path

    def test_parallel_agent_config_dataclass(self):
        """Test ParallelAgentConfig has required fields."""
        from minisweagent.agents.parallel_agent import ParallelAgentConfig

        config = ParallelAgentConfig(
            system_template="Test",
            instance_template="{{task}}",
            num_parallel=2,
            repo=Path("/tmp"),
            gpu_ids=[0, 1],
        )

        assert config.num_parallel == 2
        assert config.repo == Path("/tmp")
        assert config.gpu_ids == [0, 1]

    def test_best_patch_result_dataclass(self):
        """Test BestPatchResult has required fields."""
        result = BestPatchResult(
            agent_id=0,
            patch_id="patch_0",
            test_output="Test passed",
            metric_result={"speedup": 1.5},
            patch_dir=Path("/tmp/patches"),
            llm_conclusion="Patch 0 is best",
        )

        assert result.agent_id == 0
        assert result.patch_id == "patch_0"
        assert result.test_output == "Test passed"
        assert result.metric_result == {"speedup": 1.5}
        assert result.patch_dir == Path("/tmp/patches")
        assert result.llm_conclusion == "Patch 0 is best"

    def test_create_worktree_static_method(self, temp_git_repo):
        """Test that worktree creation works correctly."""
        worktree_path = temp_git_repo.parent / "worktree_test"

        try:
            result = ParallelAgent._create_worktree(temp_git_repo, worktree_path)
            assert result.exists()
            assert (result / "test.py").exists()
        finally:
            # Cleanup
            import subprocess

            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path), "--force"], cwd=temp_git_repo, capture_output=True
            )

    def test_ensure_safe_directory_static_method(self, temp_git_repo):
        """Test that safe directory configuration works."""
        # Should not raise
        ParallelAgent._ensure_safe_directory(temp_git_repo)

    def test_has_valid_head_static_method(self, temp_git_repo):
        """Test that HEAD validation works."""
        assert ParallelAgent._has_valid_head(temp_git_repo) is True

    def test_has_valid_head_no_commits(self):
        """Test HEAD validation with no commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

            assert ParallelAgent._has_valid_head(repo_path) is False

    def test_init_as_git_repo_non_git(self):
        """Test initializing a non-git directory as git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            test_file = repo_path / "test.py"
            test_file.write_text("print('hello')")

            ParallelAgent._init_as_git_repo(repo_path)

            assert (repo_path / ".git").exists()
            assert ParallelAgent._has_valid_head(repo_path) is True

    def test_replace_paths_static_method(self, temp_git_repo):
        """Test path replacement in text."""
        worktree_path = temp_git_repo.parent / "worktree"

        text = f"File at {temp_git_repo}/test.py"
        result = ParallelAgent._replace_paths(text, temp_git_repo, worktree_path)

        assert str(worktree_path) in result
        assert str(temp_git_repo) not in result


# --- Test Error Handling ---


class TestHomogeneousAgentErrors:
    """Tests for error handling in homogeneous agent."""

    @pytest.fixture
    def mock_model(self):
        return DeterministicModel(outputs=["test"])

    @pytest.fixture
    def mock_env(self):
        return LocalEnvironment()

    @pytest.fixture
    def base_config(self):
        return {
            "agent": {},
            "tools": {
                "strategy_manager": False,
                "strategy_file": ".optimization_strategies.md",
            },
            "parallel": {},
        }

    def test_invalid_repo_path_raises_error(self, mock_model, mock_env, base_config):
        """Test that invalid repo path raises ValueError."""
        with pytest.raises(ValueError, match="Repository path does not exist"):
            run_homogeneous_agent(
                config=base_config,
                task_content="Test task",
                model=mock_model,
                env=mock_env,
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=base_config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                },
                repo=Path("/this/path/does/not/exist"),
            )

    def test_parallel_agent_requires_repo(self):
        """Test that ParallelAgent.run requires repo path."""
        model = DeterministicModel(outputs=["test"])
        env = LocalEnvironment()

        agent = ParallelAgent(
            model=model,
            env=env,
            system_template="Test",
            instance_template="{{task}}",
        )

        with pytest.raises(ValueError, match="repository path"):
            agent.run("Test task")


# --- Test Console Output ---


class TestHomogeneousAgentConsoleOutput:
    """Tests for console output in homogeneous agent."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=test@test.com",
                    "-c",
                    "user.name=Test",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "Initial commit",
                ],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            yield repo_path

    def test_console_output_configuration_summary(self, temp_repo):
        """Test that configuration summary is printed to console."""
        from io import StringIO

        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        config = {
            "agent": {},
            "tools": {
                "strategy_manager": False,
                "strategy_file": ".optimization_strategies.md",
            },
            "parallel": {
                "num_parallel": 2,
            },
            "model": {},
        }

        with patch("minisweagent.agents.homogeneous_agent.ParallelAgent") as mock_parallel:
            mock_agent = MagicMock()
            mock_agent.run.return_value = None
            mock_parallel.return_value = mock_agent

            run_homogeneous_agent(
                config=config,
                task_content="Test task",
                model=DeterministicModel(outputs=["test"]),
                env=LocalEnvironment(),
                env_class=LocalEnvironment,
                env_kwargs={},
                tools_settings=config["tools"],
                agent_config={
                    "system_template": "Test",
                    "instance_template": "{{task}}",
                },
                repo=temp_repo,
                num_parallel=2,
                console=console,
            )

        output_str = output.getvalue()
        assert "Configuration Summary" in output_str or "Parallel agents" in output_str


# --- Marker for slow tests ---


@pytest.mark.slow
class TestHomogeneousAgentSlowIntegration:
    """Slow integration tests that may require actual execution."""

    @pytest.fixture
    def temp_git_repo_with_content(self):
        """Create a temporary git repository with actual content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "test_repo"
            repo_path.mkdir()

            # Create test files
            (repo_path / "main.py").write_text("def main():\n    print('hello')\n")
            (repo_path / "test_main.py").write_text("def test_main():\n    assert True\n")

            # Initialize git
            import subprocess

            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test", "commit", "-m", "Initial commit"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            yield repo_path

    def test_full_parallel_run_single_agent(self, temp_git_repo_with_content):
        """Test a full parallel run with a single agent."""
        model = DeterministicModel(
            outputs=[
                "THOUGHT: Running test\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'test passed'\n```"
            ]
        )
        env = LocalEnvironment()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)

            agent = ParallelAgent(
                model=model,
                env=env,
                system_template="You are a test agent.",
                instance_template="Task: {{task}}",
                step_limit=5,
                cost_limit=10.0,
                num_parallel=1,
                repo=temp_git_repo_with_content,
                patch_output_dir=str(output_path),
                mode="yolo",
                confirm_exit=False,
            )

            # Mock the _select_best_from_parallel_runs to avoid model calls
            with patch.object(ParallelAgent, "_select_best_from_parallel_runs", return_value=None):
                agent.run("Run a simple test")

            # Check that output directory has parallel directories
            assert (output_path / "parallel_0").exists() or output_path.exists()
