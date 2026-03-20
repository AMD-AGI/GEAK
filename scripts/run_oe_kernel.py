#!/usr/bin/env python3
"""Standalone OpenEvolve runner for a single kernel.
Usage: python run_oe_kernel.py <task_dir> <workspace_dir>

task_dir:      e.g., /home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/fused_rms_fp8
workspace_dir: e.g., /home/sdubagun/work/repos/AIG-Eval/workspace_MI325X_openevolve/fused_rms_fp8_20260311
"""
import asyncio
import logging
import os
import shutil
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "AIG-Eval" / "agents" / "openevolve"))
from launch_agent import create_kernel_with_separator, create_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_oe_kernel")


def main():
    task_dir = os.path.abspath(sys.argv[1])
    workspace = os.path.abspath(sys.argv[2])
    os.makedirs(workspace, exist_ok=True)

    task_config_path = os.path.join(task_dir, "config.yaml")
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)

    agent_config_path = Path(__file__).resolve().parent.parent.parent / "AIG-Eval" / "agents" / "openevolve" / "agent_config.yaml"
    with open(agent_config_path) as f:
        agent_config = yaml.safe_load(f)

    kernel_src = os.path.join(task_dir, "kernel.py")
    kernel_dst = os.path.join(workspace, "kernel.py")
    shutil.copy2(kernel_src, kernel_dst)
    shutil.copy2(task_config_path, os.path.join(workspace, "config.yaml"))

    formatted_kernel = create_kernel_with_separator(kernel_dst)
    evaluator_path = create_evaluator(workspace, task_config)
    logger.info("Formatted kernel: %s", formatted_kernel)
    logger.info("Evaluator: %s", evaluator_path)

    from openevolve import OpenEvolve
    from openevolve.config import Config, LLMModelConfig

    oe_config = Config()
    oe_config.max_iterations = agent_config.get("max_iterations", 20)
    oe_config.checkpoint_interval = agent_config.get("checkpoint_interval", 5)
    oe_config.max_code_length = 50000
    oe_config.llm.sampling = {"fn": "random"}

    api_key = os.environ.get("AMD_API_KEY") or os.environ.get("OPENAI_API_KEY")
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)

    if "llm" in agent_config:
        if "api_base" in agent_config["llm"]:
            api_base = agent_config["llm"]["api_base"]
        if "models" in agent_config["llm"] and agent_config["llm"]["models"]:
            mc = agent_config["llm"]["models"][0]
            model_name = mc.get("name", "gpt-4o-mini")
            model_api_base = mc.get("api_base", api_base)
            if "claude" in model_name.lower() and api_base == DEFAULT_API_BASE and "api_base" not in mc:
                model_api_base = f"https://llm-api.amd.com/claude3/deployments/{model_name}"
                api_base = model_api_base
            if len(oe_config.llm.models) == 0:
                oe_config.llm.models.append(LLMModelConfig(
                    name=model_name, api_base=model_api_base,
                    api_key=api_key, weight=mc.get("weight", 1.0),
                ))
            else:
                oe_config.llm.models[0].name = model_name
                oe_config.llm.models[0].api_base = model_api_base
                oe_config.llm.models[0].api_key = api_key
            logger.info("Model: %s @ %s", model_name, model_api_base)
        oe_config.llm.api_base = api_base
        oe_config.llm.temperature = agent_config["llm"].get("temperature", 0.7)
    else:
        oe_config.llm.api_base = api_base

    if api_key:
        oe_config.llm.api_key = api_key

    if "database" in agent_config:
        oe_config.database.population_size = agent_config["database"].get("population_size", 50)
        oe_config.database.num_islands = agent_config["database"].get("num_islands", 2)
    oe_config.database.db_path = os.path.join(workspace, "database")

    if "evaluator" in agent_config:
        oe_config.evaluator.timeout = agent_config["evaluator"].get("timeout", 120)
        oe_config.evaluator.verbose = agent_config["evaluator"].get("verbose", False)
    oe_config.evaluator.eval_dir = os.path.join(workspace, "evals")
    os.makedirs(oe_config.evaluator.eval_dir, exist_ok=True)

    output_dir = os.path.join(workspace, "openevolve_output")
    os.makedirs(output_dir, exist_ok=True)

    oe = OpenEvolve(
        initial_program_path=formatted_kernel,
        evaluation_file=evaluator_path,
        config=oe_config,
        output_dir=output_dir,
    )
    logger.info("Starting evolution for %d iterations...", oe_config.max_iterations)
    best = asyncio.run(oe.run())

    if best is None:
        logger.error("No valid program produced")
        sys.exit(1)

    logger.info("Best metrics: %s", best.metrics)
    sep = "#" * 146
    best_kernel = best.code.split(sep)[0].strip() if sep in best.code else best.code

    result = {
        "task_name": Path(workspace).name,
        "pass_compilation": best.metrics.get("success", 0) >= 0.2,
        "pass_correctness": best.metrics.get("correctness_score", 0) >= 1.0,
        "speedup_ratio": best.metrics.get("speedup", 0),
        "optimization_summary": "OpenEvolve optimization complete",
    }
    with open(os.path.join(workspace, "task_result.yaml"), "w") as f:
        yaml.dump(result, f)

    logger.info("Done. Speedup: %.2fx", result["speedup_ratio"])


if __name__ == "__main__":
    main()
