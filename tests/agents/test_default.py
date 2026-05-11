from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel

# --- Helpers ---


def make_output(content: str | None, actions: list[dict]) -> str:
    """Build a deterministic model output string with embedded bash code blocks."""
    parts = []
    if content:
        parts.append(content)
    for action in actions:
        parts.append(f"```bash\n{action['command']}\n```")
    return "\n".join(parts)


def _make_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples."""
    return DeterministicModel(outputs=[make_output(content, actions) for content, actions in outputs_spec], **kwargs)


def get_text(msg: dict) -> str:
    """Extract text content from a message."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    return ""


# --- Fixtures ---


@pytest.fixture
def default_config():
    """Load default agent config from config/mini.yaml"""
    config_path = Path("src/minisweagent/config/mini.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # mini.yaml may include InteractiveAgent-only keys (e.g. `mode`).
    # Keep only keys accepted by AgentConfig for DefaultAgent tests.
    allowed = {f.name for f in fields(AgentConfig)}
    return {k: v for k, v in config["agent"].items() if k in allowed}


@pytest.fixture
def model_factory(default_config):
    """Returns (factory_fn, config) for creating test models."""
    return _make_model, default_config


# --- Tests ---


def test_successful_completion(model_factory):
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("I'll echo a message", [{"command": "echo 'hello world'"}]),
                (
                    "Now finishing",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed successfully'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    exit_status, submission = agent.run("Echo hello world then finish")
    assert exit_status == "Submitted"
    assert submission == "Task completed successfully\n"
    assert agent.model.n_calls == 2


def test_step_limit_enforcement(model_factory):
    """Test agent stops when step limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("First command", [{"command": "echo 'step1'"}]),
                ("Second command", [{"command": "echo 'step2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "step_limit": 1},
    )

    exit_status, _ = agent.run("Run multiple commands")
    assert exit_status == "LimitsExceeded"
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement(model_factory):
    """Test agent stops when cost limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test", [{"command": "echo 'test'"}])]),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 0.5},
    )

    exit_status, _ = agent.run("Test cost limit")
    assert exit_status == "LimitsExceeded"


def test_timeout_handling(model_factory):
    """Test agent handles command timeouts properly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Long sleep", [{"command": "sleep 5"}]),
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **config,
    )

    exit_status, submission = agent.run("Test timeout handling")
    assert exit_status == "Submitted"
    assert submission == "recovered\n"
    timed_out = [msg for msg in agent.messages if "timed out" in get_text(msg)]
    assert len(timed_out) == 1


def test_timeout_captures_partial_output(model_factory):
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    factory, config = model_factory
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=factory(
            [
                ("Output then sleep", [{"command": calculation_command}]),
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **config,
    )
    exit_status, submission = agent.run("Test timeout with partial output")
    assert exit_status == "Submitted"
    assert submission == "recovered\n"
    timed_out = [msg for msg in agent.messages if "timed out" in get_text(msg)]
    assert len(timed_out) == 1
    assert expected_output in get_text(timed_out[0])


def test_multiple_steps_before_completion(model_factory):
    """Test agent can handle multiple steps before finding completion signal."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Step 3", [{"command": "echo 'third'"}]),
                (
                    "Final step",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed all steps'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},
    )

    exit_status, submission = agent.run("Multi-step task")
    assert exit_status == "Submitted"
    assert submission == "completed all steps\n"
    assert agent.model.n_calls == 4


def test_custom_config(model_factory):
    """Test agent works with custom configuration."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                (
                    "Test response",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom config works'"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "system_template": "You are a test assistant.",
            "instance_template": "Task: {{task}}. Return bash command.",
            "step_limit": 2,
            "cost_limit": 1.0,
        },
    )

    exit_status, submission = agent.run("Test custom config")
    assert exit_status == "Submitted"
    assert submission == "custom config works\n"
    assert get_text(agent.messages[0]) == "You are a test assistant."
    assert "Test custom config" in get_text(agent.messages[1])


def test_render_template_model_stats(model_factory):
    """Test that render_template has access to n_model_calls and model_cost from agent."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Test 1", [{"command": "echo 'test1'"}]),
                ("Test 2", [{"command": "echo 'test2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    agent.add_message("system", "test")
    agent.add_message("user", "test")
    agent.query()
    agent.query()

    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    assert agent.render_template(template) == "Calls: 2, Cost: 2.0"


def test_message_history_tracking(model_factory):
    """Test that messages are properly added and tracked."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Response 1", [{"command": "echo 'test1'"}]),
                ("Response 2", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    exit_status, submission = agent.run("Track messages")
    assert exit_status == "Submitted"
    assert submission == "done\n"

    # Should have 6 messages: system, user, assistant, observation, assistant, exit
    assert len(agent.messages) == 6
    assert get_text(agent.messages[0])  # system has content
    assert get_text(agent.messages[1])  # user has content
    assert agent.messages[2]["role"] == "assistant"
    assert agent.messages[3]["role"] == "user"
    assert "returncode" in get_text(agent.messages[3])
    assert agent.messages[4]["role"] == "assistant"


def test_step_adds_messages(model_factory):
    """Test that step adds assistant and observation messages."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test command", [{"command": "echo 'hello'"}])]),
        env=LocalEnvironment(),
        **config,
    )

    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    initial_count = len(agent.messages)
    agent.step()

    # step() should add assistant message + observation message
    assert len(agent.messages) == initial_count + 2
    assert agent.messages[-2]["role"] == "assistant"
    assert "echo 'hello'" in get_text(agent.messages[-2])
    assert agent.messages[-1]["role"] == "user"
    assert "returncode" in get_text(agent.messages[-1])


def test_observations_captured(model_factory):
    """Test intermediate outputs are captured correctly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Final", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},
    )

    agent.run("Multi-step task")
    observations = [
        get_text(msg) for msg in agent.messages
        if msg.get("role") == "user" and "returncode" in get_text(msg)
    ]
    assert len(observations) == 2
    assert "first" in observations[0]
    assert "second" in observations[1]


def test_empty_actions_handling(model_factory):
    """Test agent handles empty actions (continues without error)."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("No actions here", []),
                ("Now with action", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    exit_status, submission = agent.run("Test empty actions")
    assert exit_status == "Submitted"
    assert submission == "done\n"
    assert agent.model.n_calls == 2
