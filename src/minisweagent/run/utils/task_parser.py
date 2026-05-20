"""Parse optimization task information from user input."""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from minisweagent.run.utils.prompts import (
    EXTRACT_USER_CONSTRAINTS_TEMPLATE,
    JSON_EXTRACTION_SYSTEM_PROMPT,
    PARSE_PIPELINE_PARAMS_USER_TEMPLATE,
    PARSE_TASK_INFO_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_EMPTY_TASK_INFO: dict = {
    "kernel_name": None,
    "kernel_url": None,
    "kernel_type": "other",
    "repo": None,
    "test_command": None,
    "metric": None,
    "num_parallel": None,
    "gpu_ids": None,
    "output_dir": None,
    "model": None,
    "config": None,
}

_EMPTY_PIPELINE_PARAMS: dict = {
    "kernel_url": None,
    "preprocess_dir": None,
    "max_rounds": None,
    "start_round": None,
    "pipeline_intent": False,
}


def _resolve_path_case(path: Path) -> Path | None:
    """Resolve path to filesystem case (e.g. geak -> GEAK on case-sensitive filesystems).
    Walks each component and matches case-insensitively against directory listing.
    Returns None if any component is not found.
    """
    if not path.is_absolute():
        logger.debug("_resolve_path_case: non-absolute path %r", path)
        return None
    parts = path.parts[1:]  # drop leading /
    resolved = Path(path.anchor)
    for name in parts:
        if not resolved.is_dir():
            logger.debug("_resolve_path_case: not a directory: %r", resolved)
            return None
        found = None
        for entry in resolved.iterdir():
            if entry.name.lower() == name.lower():
                found = entry
                break
        if found is None:
            logger.debug("_resolve_path_case: no match for %r", name)
            return None
        resolved = found
    if str(path) != str(resolved):
        logger.debug("_resolve_path_case: case-insensitive match: %s -> %s", path, resolved)
    return resolved


def _normalize_path(path_str: str) -> str | None:
    """Normalize a path string: resolve if exists, try case-insensitive resolution otherwise."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        out = str(p.resolve())
        if out != path_str:
            logger.debug("_normalize_path: resolved existing path %r -> %s", path_str, out)
        return out
    resolved = _resolve_path_case(p)
    if resolved is not None:
        out = str(resolved.resolve())
        logger.debug("_normalize_path: case-insensitive resolution %r -> %s", path_str, out)
        return out
    return path_str  # return as-is if can't resolve


_JSON_OBJECT_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _load_json_object_from_model_response(response: dict, *, log_prefix: str) -> dict:
    """Strip optional ```json fence and parse the model's JSON object.

    On ``JSONDecodeError``, log a truncated view of the raw response BEFORE
    re-raising. Without this the only diagnostic users get from the warning
    handler is "Expecting value: line 1 column 1 (char 0)", which is
    indistinguishable across any non-JSON failure mode (model returned a
    plain-text apology, returned nothing, returned markdown without a
    fenced JSON block, etc.).
    """
    content = response.get("content", "").strip()
    m = _JSON_OBJECT_FENCE.search(content)
    if m:
        logger.debug("%s: extracted JSON from markdown fence in model response", log_prefix)
        content = m.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Show what the model actually produced so the user can tell whether
        # this was a prompt issue, a model regression, or a transport problem.
        # Truncate to keep logs manageable; full content lives in trajectory.
        preview = (response.get("content") or "").strip()
        if len(preview) > 500:
            preview = preview[:500] + "... [truncated]"
        logger.warning(
            "%s: model response was not valid JSON. Raw response (truncated): %r",
            log_prefix,
            preview or "<empty>",
        )
        raise


# Kernel-source filename heuristics, ranked. When the LLM returned a
# directory as ``kernel_url`` we look inside it for the first match. Keep
# this conservative -- most users will have ``kernel.py`` or
# ``<kernel_name>.<ext>``; if that's not the layout we leave the path
# alone and let downstream resolution surface the original error with the
# (now slightly more actionable) message we log here.
_KERNEL_FILENAME_GUESSES: tuple[str, ...] = (
    "kernel.py",
    "kernel.hip",
    "kernel.cu",
    "kernel.flydsl",
)
_KERNEL_TYPE_TO_EXT: dict[str, tuple[str, ...]] = {
    "triton": (".py",),
    "hip": (".hip", ".cu", ".cpp"),
    "flydsl": (".flydsl", ".py"),
    "pytorch2flydsl": (".py",),
}


# Safety bound on the cheap "exactly one matching file" fallback inside the
# directory: we don't want to walk a HIP repo with thousands of files just
# because the LLM echoed a wrong directory back to us, especially on shared
# filesystems where ``iterdir`` can be slow and racy.
_PROMOTE_DIR_MAX_ENTRIES = 32


def _promote_kernel_url_dir_to_file(
    kernel_url: str,
    *,
    kernel_name_hint: str | None,
    kernel_type: str,
) -> str:
    """Return ``kernel_url`` unchanged unless it points at an existing
    *directory*, in which case search inside for a kernel file and promote.

    Search order, first match wins:
      1. ``<kernel_name_hint>.<kernel-type extension>`` if both hints set.
      2. Hard-coded ``kernel.{py,hip,cu,flydsl}`` (most common GEAK layout).
      3. The single file in the directory matching the kernel-type
         extensions, if exactly one exists *and* the directory is small
         (<= ``_PROMOTE_DIR_MAX_ENTRIES`` entries).

    Promotion only runs when ``kernel_type`` is set to a known, kernel-bearing
    value. The LLM extractor is explicitly prompted not to investigate the
    filesystem; this helper does, but we keep its scope tight: we never walk
    large directories, and we never guess on an "unknown"/"other" kernel
    type. Without those guardrails a stale ``kernel_url`` like
    ``/home/.../some/cached/dir`` could silently get promoted to whatever
    ``.py`` file happens to live there.

    If nothing matches, return the original ``kernel_url`` (the existing
    error will fire downstream); we only log a warning so users know we
    tried and saw a directory.
    """
    p = Path(kernel_url)
    if not p.is_dir():
        return kernel_url

    if kernel_type not in _KERNEL_TYPE_TO_EXT and not kernel_name_hint:
        # Unknown / "other" kernel types don't have a reliable extension
        # signal, and without a name hint we cannot tell which file to
        # promote. Refuse with a warning so the user passes the file path
        # explicitly rather than risk silently promoting a stale directory's
        # arbitrary ``.py`` file.
        logger.warning(
            "parse_task_info: kernel_url %s is a directory and kernel_type=%r "
            "with no kernel_name hint is not promotable; pass the kernel file "
            "path explicitly.",
            kernel_url,
            kernel_type,
        )
        return kernel_url

    # Resolve to absolute for clearer logs.
    p = p.resolve()

    candidates: list[Path] = []

    if kernel_name_hint:
        for ext in _KERNEL_TYPE_TO_EXT.get(kernel_type, (".py",)):
            cand = p / f"{kernel_name_hint}{ext}"
            candidates.append(cand)

    candidates.extend(p / name for name in _KERNEL_FILENAME_GUESSES)

    for cand in candidates:
        if cand.is_file():
            promoted = str(cand)
            logger.info(
                "parse_task_info: kernel_url was a directory; promoted to %s (was %s)",
                promoted,
                kernel_url,
            )
            return promoted

    # Last resort: if exactly one file in the directory matches the type's
    # extensions, take it -- but only if the directory is small enough that
    # iterating it is cheap and predictable.
    exts = _KERNEL_TYPE_TO_EXT.get(kernel_type, (".py",))
    try:
        entries: list[Path] = []
        for i, entry in enumerate(p.iterdir()):
            if i >= _PROMOTE_DIR_MAX_ENTRIES:
                logger.warning(
                    "parse_task_info: kernel_url %s contains > %d entries; refusing to "
                    "scan for a kernel file. Pass the kernel file path explicitly.",
                    kernel_url,
                    _PROMOTE_DIR_MAX_ENTRIES,
                )
                return kernel_url
            entries.append(entry)
    except OSError as exc:
        logger.debug("parse_task_info: iterdir failed on %s: %s", p, exc)
        return kernel_url

    matches = [f for f in entries if f.is_file() and f.suffix in exts]
    if len(matches) == 1:
        promoted = str(matches[0])
        logger.info(
            "parse_task_info: kernel_url was a directory with a single %s file; promoted to %s (was %s)",
            "/".join(exts),
            promoted,
            kernel_url,
        )
        return promoted

    logger.warning(
        "parse_task_info: kernel_url %s is a directory and no obvious kernel "
        "file (%s, kernel.{py,hip,cu,flydsl}, single %s match) was found "
        "inside it. Pass the kernel file path explicitly.",
        kernel_url,
        f"{kernel_name_hint}.<ext>" if kernel_name_hint else "<kernel_name>.<ext>",
        "/".join(exts),
    )
    return kernel_url


def _normalize_parsed_task_info(parsed: dict) -> dict:
    """Validate and normalize fields after JSON parse (paths, kernel_type)."""
    raw_kernel_type = parsed.get("kernel_type")
    kernel_type = str(raw_kernel_type or "").strip().lower()
    logger.info("parse_task_info: kernel_type: %s", kernel_type)

    if kernel_type not in {"hip", "triton", "pytorch2flydsl", "flydsl", "other"}:
        if raw_kernel_type not in (None, ""):
            logger.warning(
                "parse_task_info: invalid kernel_type %r; normalizing to 'other'.",
                raw_kernel_type,
            )
        kernel_type = "other"
    result = {
        "kernel_name": parsed.get("kernel_name"),
        "kernel_url": parsed.get("kernel_url"),
        "kernel_type": kernel_type,
        "repo": parsed.get("repo"),
        "test_command": parsed.get("test_command"),
        "metric": parsed.get("metric"),
        "num_parallel": parsed.get("num_parallel"),
        "gpu_ids": parsed.get("gpu_ids"),
        "output_dir": parsed.get("output_dir"),
        "model": parsed.get("model"),
        "config": parsed.get("config"),
    }

    # Normalize repo path and preserve filesystem case (LLM often returns lowercase)
    if result["repo"]:
        original_repo = result["repo"]
        repo_path = Path(result["repo"])
        if repo_path.exists():
            result["repo"] = str(repo_path.resolve())
            if result["repo"] != original_repo:
                logger.debug("parse_task_info: repo path resolved: %s -> %s", original_repo, result["repo"])
        else:
            resolved = _resolve_path_case(repo_path)
            if resolved is not None:
                result["repo"] = str(resolved.resolve())
                logger.debug("parse_task_info: repo path case-corrected: %s -> %s", original_repo, result["repo"])

    # Promote a directory kernel_url to a kernel file inside it. Users
    # frequently say "the kernel is in <DIR>" rather than handing us the file
    # path directly; the LLM extractor tends to echo the directory verbatim.
    # Without this rescue, ``resolve_kernel_url`` later fails with the
    # opaque "Kernel file not found: <DIR>" error.
    if result["kernel_url"]:
        result["kernel_url"] = _promote_kernel_url_dir_to_file(
            result["kernel_url"],
            kernel_name_hint=result.get("kernel_name"),
            kernel_type=kernel_type,
        )

    if result["output_dir"]:
        result["output_dir"] = _normalize_path(result["output_dir"])
    if result["config"]:
        result["config"] = _normalize_path(result["config"])

    populated = sorted(k for k, v in result.items() if v is not None and v != "")
    logger.debug("parse_task_info: extracted non-empty fields: %s", populated)

    return result


def _normalize_pipeline_params_from_parsed(parsed: dict) -> dict:
    """Normalize paths and integer fields after JSON parse."""
    result = {
        "kernel_url": parsed.get("kernel_url"),
        "preprocess_dir": parsed.get("preprocess_dir"),
        "max_rounds": parsed.get("max_rounds"),
        "start_round": parsed.get("start_round"),
        "pipeline_intent": bool(parsed.get("pipeline_intent", False)),
    }

    if result["kernel_url"]:
        result["kernel_url"] = _normalize_path(result["kernel_url"])
    if result["preprocess_dir"]:
        result["preprocess_dir"] = _normalize_path(result["preprocess_dir"])

    for field in ("max_rounds", "start_round"):
        if result[field] is not None:
            raw = result[field]
            try:
                result[field] = int(result[field])
            except (ValueError, TypeError):
                logger.debug(
                    "parse_pipeline_params: invalid %s value %r; clearing.",
                    field,
                    raw,
                )
                result[field] = None

    populated = sorted(k for k, v in result.items() if v is not None)
    logger.debug("parse_pipeline_params: extracted non-null fields: %s", populated)

    return result


def parse_task_info(task_content: str, model) -> dict:
    """Parse task content to extract optimization configuration.

    Extracts:
    - kernel_name: Name of the kernel being optimized
    - kernel_url: URL/path of the kernel being optimized
    - kernel_type: One of hip/triton/flydsl/other
    - repo: Repository path
    - test_command: Command to test the optimization
    - metric: Performance metric to extract
    - num_parallel: Number of parallel agents
    - gpu_ids: GPU IDs for parallel execution
    - output_dir: Output directory for logs/artifacts
    - model: Model name/identifier to use
    - config: Path to a config YAML file

    Returns dict with extracted values (None if not found).
    """
    prompt = PARSE_TASK_INFO_USER_TEMPLATE.format(task_content=task_content)

    logger.debug("parse_task_info: querying model (task_content length=%d chars)", len(task_content))

    try:
        response = model.query(
            [
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _load_json_object_from_model_response(response, log_prefix="parse_task_info")
    except json.JSONDecodeError as e:
        logger.warning("parse_task_info: model response JSON decode failed: %s", e)
        return _EMPTY_TASK_INFO.copy()
    except Exception as e:
        logger.warning(
            "parse_task_info: unexpected error (%s): %s",
            type(e).__name__,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _EMPTY_TASK_INFO.copy()

    try:
        return _normalize_parsed_task_info(parsed)
    except Exception as e:
        logger.warning(
            "parse_task_info: normalization failed after successful JSON parse (%s): %s",
            type(e).__name__,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _EMPTY_TASK_INFO.copy()


def parse_pipeline_params(task_content: str, model) -> dict:
    """Extract pipeline orchestration parameters from task text via LLM.

    Extracts:
    - kernel_url: Path or URL to the specific kernel file to optimize
    - preprocess_dir: Path to existing preprocessing artifacts
    - max_rounds: Maximum optimization rounds
    - start_round: Round to resume from
    - pipeline_intent: Whether the task describes kernel optimization work

    Returns dict with extracted values (None if not found).
    """
    prompt = PARSE_PIPELINE_PARAMS_USER_TEMPLATE.format(task_content=task_content)

    logger.debug("parse_pipeline_params: querying model (task_content length=%d chars)", len(task_content))

    try:
        response = model.query(
            [
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _load_json_object_from_model_response(response, log_prefix="parse_pipeline_params")
    except json.JSONDecodeError as e:
        logger.warning("parse_pipeline_params: model response JSON decode failed: %s", e)
        return _EMPTY_PIPELINE_PARAMS.copy()
    except Exception as e:
        logger.warning(
            "parse_pipeline_params: unexpected error (%s): %s",
            type(e).__name__,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _EMPTY_PIPELINE_PARAMS.copy()

    try:
        return _normalize_pipeline_params_from_parsed(parsed)
    except Exception as e:
        logger.warning(
            "parse_pipeline_params: normalization failed after successful JSON parse (%s): %s",
            type(e).__name__,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _EMPTY_PIPELINE_PARAMS.copy()


_EMPTY_USER_CONSTRAINTS: dict[str, list[str]] = {"constraints": [], "directives": []}


def extract_user_constraints(task_content: str, model) -> dict[str, list[str]]:
    """Extract mandatory constraints and optimization directives from task text via LLM.

    Returns dict with:
        "constraints": list of hard rules (violation = rejection)
        "directives": list of prescribed optimization strategies (should follow, may explore beyond)
    """
    prompt = EXTRACT_USER_CONSTRAINTS_TEMPLATE.format(task_content=task_content)
    logger.debug("extract_user_constraints: querying model (task_content length=%d chars)", len(task_content))

    try:
        response = model.query(
            [
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _load_json_object_from_model_response(response, log_prefix="extract_user_constraints")
    except json.JSONDecodeError as e:
        logger.warning("extract_user_constraints: model response JSON decode failed: %s", e)
        return _EMPTY_USER_CONSTRAINTS.copy()
    except Exception as e:
        logger.warning(
            "extract_user_constraints: unexpected error (%s): %s",
            type(e).__name__,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _EMPTY_USER_CONSTRAINTS.copy()

    constraints = parsed.get("constraints", [])
    directives = parsed.get("directives", [])
    if not isinstance(constraints, list):
        constraints = []
    if not isinstance(directives, list):
        directives = []
    result = {
        "constraints": [str(c) for c in constraints if c],
        "directives": [str(d) for d in directives if d],
    }
    logger.debug(
        "extract_user_constraints: extracted %d constraints, %d directives.",
        len(result["constraints"]),
        len(result["directives"]),
    )
    return result


# Max length of the kernel-derived path segment (before ``_<YYYYMMDD>_<HHMMSS>``).
# Long symbols (e.g. hipBLASLt ``Cijk_*``) are shortened to keep the leaf short.
#
# 48 chars balances filesystem-friendly path length against the ergonomics of
# scanning logs by eye: at 20 a Cijk-style kernel was reduced to ~11 chars +
# hash and effectively unrecognisable; at 48 the human-meaningful prefix is
# preserved before the disambiguating SHA-256 digest is appended.
_MAX_KERNEL_DIR_STEM_LEN = 48


def _sanitize_kernel_name_for_patch_dir(kernel_name: str) -> str:
    """Return a filesystem-safe, bounded-length stem for log / output directories."""
    clean_name = re.sub(r"[^\w\-]", "_", kernel_name)
    if len(clean_name) <= _MAX_KERNEL_DIR_STEM_LEN:
        return clean_name
    digest = hashlib.sha256(clean_name.encode("utf-8")).hexdigest()[:8]
    take = _MAX_KERNEL_DIR_STEM_LEN - 1 - len(digest)
    take = max(1, take)
    shortened = f"{clean_name[:take]}_{digest}"
    logger.debug(
        "Sanitized long kernel_name for patch dir (%d -> %d chars).",
        len(clean_name),
        len(shortened),
    )
    return shortened


def generate_patch_output_dir(kernel_name: str | None, base_dir: str = "optimization_logs") -> str:
    """Generate patch output directory based on kernel name and timestamp.

    Format: optimization_logs/kernelname_timestamp
    If kernel_name is None, use "optimization_timestamp"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if kernel_name:
        dir_stem = _sanitize_kernel_name_for_patch_dir(str(kernel_name))
        dir_name = f"{dir_stem}_{timestamp}"
    else:
        dir_name = f"optimization_{timestamp}"

    out = str(Path(base_dir) / dir_name)
    logger.debug("generate_patch_output_dir: %s (kernel_name=%r)", out, kernel_name)
    return out


def display_parsed_config(parsed_info: dict, patch_output_dir: str) -> str:
    """Display parsed configuration in a formatted way for user confirmation."""
    lines = [
        "\n" + "=" * 70,
        "Resolved Configuration (CLI overrides auto-detection):",
        "=" * 70,
    ]

    fields: list[tuple[str, str]] = [
        (
            "kernel_type",
            parsed_info.get("kernel_type") or "Not detected. Default to other.",
        ),
        (
            "kernel_name",
            parsed_info.get("kernel_name")
            or "Not detected. Please provide --kernel-url or include kernel name in the task",
        ),
        (
            "kernel_url",
            parsed_info.get("kernel_url") or "Not detected. Please use --kernel-url to specify the kernel target",
        ),
        ("repo", parsed_info["repo"] or "Not detected. Please use --repo to specify the repository path"),
        (
            "test_command",
            parsed_info["test_command"]
            or "Not detected. Automatically search or create the test command via UnitTestAgent",
        ),
        (
            "metric",
            parsed_info["metric"] or "Not detected. Automatically extract the metric from the test output",
        ),
        ("num_parallel", str(parsed_info["num_parallel"] or "Not detected. Default to 1.")),
        ("gpu_ids", parsed_info["gpu_ids"] or "Not detected. Default to 0."),
        ("model", parsed_info.get("model") or "Not detected. Using default."),
        ("config", parsed_info.get("config") or "Not detected. Using default."),
        ("patch_output_dir", patch_output_dir),
    ]
    key_width = max(len(k) for k, _ in fields)
    for key, value in fields:
        lines.append(f"  {key + ':':<{key_width + 1}}  {value}")
    lines.append("=" * 70)

    return "\n".join(lines)
