from __future__ import annotations

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.dispatch_plan import DispatchPlan, DispatchPlanItem


class _Agent:
    pass


def test_dispatch_plan_round_trips_agent_tasks() -> None:
    tasks = [
        AgentTask(
            agent_class=_Agent,
            task="Optimize the kernel.",
            label="opt",
            priority=3,
            kernel_language="triton",
            config={"agent_name": "general-kernel-optimization"},
            num_gpus=2,
        )
    ]

    plan = DispatchPlan.from_agent_tasks(round_num=2, mode="mixed", tasks=tasks)
    rebuilt = plan.to_agent_tasks(_Agent)

    assert plan.round_num == 2
    assert plan.mode == "mixed"
    assert plan.items[0].agent_name == "general-kernel-optimization"
    assert rebuilt[0].agent_class is _Agent
    assert rebuilt[0].config["agent_name"] == "general-kernel-optimization"
    assert rebuilt[0].num_gpus == 2


def test_dispatch_plan_dict_shape_uses_task_prompt_key() -> None:
    plan = DispatchPlan(
        round_num=1,
        mode="planned",
        items=(
            DispatchPlanItem(
                label="x",
                task="Do x",
                agent_name="harness-generator",
                kernel_language="hip",
            ),
        ),
    )

    data = plan.to_dict()

    assert data["round"] == 1
    assert data["mode"] == "planned"
    assert data["tasks"][0]["task_prompt"] == "Do x"
    assert data["tasks"][0]["agent_name"] == "harness-generator"

