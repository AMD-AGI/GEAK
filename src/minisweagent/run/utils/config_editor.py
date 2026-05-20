"""Interactive configuration editor for auto-detected settings."""

import logging
import sys
from select import select

logger = logging.getLogger(__name__)


def input_with_timeout(prompt: str, timeout_s: float, default: str) -> tuple[str, bool]:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select([sys.stdin], [], [], timeout_s)
    if ready:
        return sys.stdin.readline().rstrip("\n"), False
    return default, True


def prompt_missing_pipeline_params(
    pipeline_params: dict,
    console,
    yolo: bool,
) -> tuple[dict, bool]:
    """Prompt the user for missing required pipeline parameters.

    Args:
        pipeline_params: Dict from parse_pipeline_params (may have None values).
        console: Rich console for output.
        yolo: If True, skip prompting and return as-is.

    Returns:
        (updated_params, should_use_pipeline):
        - updated_params: pipeline_params with user-provided values filled in.
        - should_use_pipeline: True if we have enough info to trigger pipeline mode.
    """
    kernel_url = pipeline_params.get("kernel_url")
    preprocess_dir = pipeline_params.get("preprocess_dir")
    pipeline_intent = pipeline_params.get("pipeline_intent", False)

    # Already have what we need
    if kernel_url or preprocess_dir:
        logger.info(
            "Pipeline params: kernel_url or preprocess_dir already set; skipping missing-param prompt.",
        )
        _display_pipeline_params(pipeline_params, console)
        return pipeline_params, True

    # No pipeline intent detected
    if not pipeline_intent:
        logger.debug("Pipeline params: no pipeline_intent in task; not prompting for kernel path.")
        return pipeline_params, False

    # Pipeline intent detected but kernel_url is missing
    if yolo:
        logger.info(
            "Pipeline intent detected but kernel_url missing; yolo mode cannot prompt — using legacy agent path.",
        )
        return pipeline_params, False

    # Show what was extracted and prompt for kernel path
    logger.info("Pipeline mode: prompting for missing kernel_url (interactive).")
    console.print("\n[bold cyan]Pipeline optimization detected from your task.[/bold cyan]")
    _display_pipeline_params(pipeline_params, console)
    console.print("[bold yellow]Kernel file path is required to run the pipeline.[/bold yellow]")

    answer, timed_out = input_with_timeout(
        "Enter kernel file path or URL (press Enter for legacy mode): ",
        timeout_s=60.0,
        default="",
    )
    logger.info("Kernel path prompt: answer=%r, timed_out=%s", answer, timed_out)

    if timed_out or not answer.strip():
        if timed_out:
            logger.info("Pipeline kernel path prompt timed out; using legacy agent mode.")
        else:
            logger.info("Pipeline kernel path empty; using legacy agent mode.")
        console.print("[dim]No kernel path provided — using legacy agent mode.[/dim]")
        return pipeline_params, False

    pipeline_params["kernel_url"] = answer.strip()
    logger.info("Pipeline kernel_url set from user input; proceeding in pipeline mode.")
    return pipeline_params, True


def _display_pipeline_params(params: dict, console) -> None:
    """Display extracted pipeline parameters."""
    fields = [
        ("kernel_url", params.get("kernel_url") or "[dim]not detected[/dim]"),
        ("preprocess_dir", params.get("preprocess_dir") or "[dim]not set[/dim]"),
        ("max_rounds", str(params.get("max_rounds")) if params.get("max_rounds") is not None else "[dim]default[/dim]"),
        ("start_round", str(params.get("start_round")) if params.get("start_round") is not None else "[dim]1[/dim]"),
    ]
    console.print("[dim]Pipeline parameters:[/dim]")
    for key, value in fields:
        console.print(f"  [dim]{key}:[/dim] {value}")
    logger.info("Pipeline parameters: %s", {k: v for k, v in fields})
