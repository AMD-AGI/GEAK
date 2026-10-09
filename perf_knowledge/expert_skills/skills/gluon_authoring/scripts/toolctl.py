#!/usr/bin/env python3
"""Use the generated tool registry without claiming a sandbox.

Usage:
  toolctl.py describe <tool> --json
  toolctl.py stage --work <dir> --role <role> --json
  toolctl.py exec [--work <dir>] [--role <role>] [--stage <stage>] <tool> -- <argv...>
  toolctl.py --describe --format=json
  toolctl.py --selftest

``exec`` launches the registered command directly on the current host.  It does
not isolate, authorize, or sandbox the child process; its receipt records only
what was requested and the child result.

IN GEAK this is optional bookkeeping a deep_engineer may use for its records, not a run mode.
Role ids such as `captain` / `deep` / `skeptic` are upstream record-schema identifiers (`deep` = the
deep_engineer, pass `--role deep`); nothing spawns them, and final arbitration in GEAK is Director's.
GEAK's phases (kernel_workflow/kernel_lane.js) are the stage machine; a stage card here is a
record index for the deep_engineer, not an assignment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CATALOG_SCHEMA = "toolctl.catalog/1"
STAGE_SCHEMA = "toolctl.stage-card/1"
ROLE_SURFACE_SCHEMA = "toolctl.role-surface/1"
REGISTRY_SCHEMA = "toolctl.run-registry/1"
RECEIPT_SCHEMA = "toolctl.command-receipt/1"
TOOLCTL_NAME = "toolctl"


class ToolctlError(RuntimeError):
    """An input or contract failure that is safe to show to the caller."""


def _run_state_module():
    """Load the optional state primitive only in packs that ship lifecycle control."""
    try:
        import run_state
    except ModuleNotFoundError:
        return None
    return run_state


def _context_modules():
    try:
        import context_query
        import stage_context
    except ModuleNotFoundError as exc:
        raise ToolctlError(f"this pack does not ship context lifecycle support: {exc}") from exc
    return stage_context, context_query


def _journal_lifecycle(work: Path) -> dict[str, Any]:
    try:
        import round_record
    except ModuleNotFoundError as exc:
        raise ToolctlError(f"this pack does not ship journal lifecycle support: {exc}") from exc
    return round_record.lifecycle_check(
        str(work / round_record.DEFAULT_JOURNAL),
        str(work / round_record.DEFAULT_MEASUREMENT_EVENTS),
        str(work / round_record.DEFAULT_RECALL_EVENTS),
    )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _native_description() -> dict[str, Any]:
    return {
        "schema": "toolctl.describe/1",
        "name": TOOLCTL_NAME,
        "classification": "public",
        "path": "scripts/toolctl.py",
        "flags": ["--help", "--catalog", "--describe", "--format", "--selftest"],
        "commands": ["describe", "stage", "context", "exec"],
        "describe_argv": ["--describe", "--format=json"],
        "execution_model": "direct_host_subprocess_with_receipt",
    }


def _default_catalog_path() -> Path:
    """Locate the pack catalog for both direct and router-namespaced scripts."""
    override = os.environ.get("TOOLCTL_CATALOG")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "runtime" / "tool-catalog.json"
        if candidate.is_file():
            return candidate
    # Keep the error deterministic even before a pack has been composed.
    return here.parent.parent / "runtime" / "tool-catalog.json"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ToolctlError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ToolctlError(f"{label} is not valid JSON: {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ToolctlError(f"{label} must be a JSON object: {path}")
    return value


def _load_catalog(path: Path) -> dict[str, Any]:
    catalog = _read_json(path, "tool catalog")
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ToolctlError(f"tool catalog schema must be {CATALOG_SCHEMA!r}: {path}")
    tools = catalog.get("tools")
    if not isinstance(tools, list):
        raise ToolctlError(f"tool catalog tools must be a list: {path}")
    seen = set()
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ToolctlError(f"tool catalog has an unnamed tool: {path}")
        if tool["name"] in seen:
            raise ToolctlError(f"tool catalog has duplicate tool {tool['name']!r}: {path}")
        seen.add(tool["name"])
    return catalog


def _load_role_surface(catalog_path: Path, role: str) -> dict[str, Any] | None:
    """Load a dedicated role surface without first reading the full tool catalog."""
    path = catalog_path.parent / "roles" / f"{role}.json"
    if not path.is_file():
        return None
    surface = _read_json(path, "role surface")
    tools = surface.get("tools")
    if (surface.get("schema") != ROLE_SURFACE_SCHEMA or surface.get("role") != role
            or not isinstance(surface.get("stage"), str) or not isinstance(tools, list)):
        raise ToolctlError(f"invalid role surface for {role!r}: {path}")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ToolctlError(f"role surface has an unnamed tool: {path}")
        if tool["name"] in names:
            raise ToolctlError(f"role surface has duplicate tool {tool['name']!r}: {path}")
        names.add(tool["name"])
    return surface


def _catalog_root(catalog_path: Path) -> Path:
    # Catalogs are always emitted at <pack>/runtime/tool-catalog.json.
    return catalog_path.parent.parent.resolve()


def _tool(catalog: dict[str, Any], name: str) -> dict[str, Any]:
    match = next((tool for tool in catalog["tools"] if tool["name"] == name), None)
    if match is None:
        raise ToolctlError(f"unknown tool {name!r}")
    return match


def _surface_tool(surface: dict[str, Any], name: str) -> dict[str, Any]:
    match = next((tool for tool in surface["tools"] if tool["name"] == name), None)
    if match is None:
        raise ToolctlError(f"tool {name!r} is not exposed to role {surface['role']!r}")
    return match


def _safe_tool_path(root: Path, tool: dict[str, Any]) -> Path:
    rel = tool.get("path")
    if not isinstance(rel, str) or not rel:
        raise ToolctlError(f"tool {tool.get('name')!r} has no relative path")
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ToolctlError(f"tool {tool.get('name')!r} path escapes the pack: {rel!r}") from exc
    if not target.is_file():
        raise ToolctlError(f"tool {tool.get('name')!r} does not exist: {rel}")
    return target


def _load_optional_json(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path, label)


def _run_registry(work: Path) -> dict[str, Any]:
    """Read an optional, fixed-name registry without walking experiment trees."""
    for name in ("toolctl-registry.json", "run_registry.json", "registry.json"):
        doc = _load_optional_json(work / name, "run registry")
        if doc is None:
            continue
        schema = doc.get("schema")
        if schema not in (None, REGISTRY_SCHEMA):
            raise ToolctlError(f"run registry schema must be {REGISTRY_SCHEMA!r}: {work / name}")
        return doc
    return {}


def _run_state(work: Path) -> dict[str, Any]:
    return _load_optional_json(work / "run_state.json", "run state") or {}


def _stage_names(catalog_path: Path) -> list[str]:
    directory = catalog_path.parent / "stages"
    if not directory.is_dir():
        raise ToolctlError(f"stage directory is missing: {directory}")
    cards = []
    for path in sorted(directory.glob("*.json")):
        card = _read_json(path, "stage card")
        if card.get("schema") != STAGE_SCHEMA or card.get("stage") != path.stem:
            raise ToolctlError(f"invalid stage card: {path}")
        order = card.get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ToolctlError(f"stage card has invalid order: {path}")
        cards.append((order, path.stem, card))
    if not cards:
        raise ToolctlError(f"stage directory has no cards: {directory}")
    cards.sort(key=lambda item: item[0])
    orders = [order for order, _, _ in cards]
    if orders != list(range(len(cards))):
        raise ToolctlError(f"stage cards must have contiguous explicit order 0..{len(cards) - 1}: "
                           f"{directory}")
    names = [name for _, name, _ in cards]
    if names[0] != "entry":
        raise ToolctlError(f"first explicit stage must be 'entry', not {names[0]!r}: {directory}")
    for index, (_, name, card) in enumerate(cards):
        expected_next = names[index + 1] if index + 1 < len(names) else None
        if card.get("next") != expected_next:
            raise ToolctlError(f"stage card {name!r} next={card.get('next')!r}; "
                               f"explicit order requires {expected_next!r}")
    return names


def _infer_stage(catalog_path: Path, registry: dict[str, Any], state: dict[str, Any],
                 role: str | None = None) -> tuple[str, str]:
    """Return the active stage AND the evidence that chose it.

    The provenance is not diagnostics.  Every branch below returns a stage the caller
    cannot distinguish from any other, and the last one is a fallback that fires
    precisely when nothing upstream has told this run anything.  A worker spawned
    mid-pipeline against a work root whose `run_state.json` was never written was
    handed `entry` -- the captain's own resolve/preflight stage -- and had no way to
    see that it was a default rather than an instruction, so it went and did it.
    """
    stages = _stage_names(catalog_path)
    # run_state is the durable authority.  The registry remains a convenience index for
    # legacy artifact references and must never move a workflow backwards.
    authoritative = state.get("stage")
    if isinstance(authoritative, str) and authoritative in stages:
        return authoritative, "run_state"
    current = registry.get("current_stage", registry.get("stage"))
    if isinstance(current, str) and current in stages:
        return current, "registry"
    # A bounded worker (currently triton_sweep) has one dedicated operating
    # stage. Without an explicit registry its old default was `entry`, where
    # its sweep facade is intentionally absent; it then looked like a
    # role-scope denial rather than a missing stage marker.
    # A dedicated stage need not be a stage_graph node -- `sweep` is not -- so this
    # must NOT be filtered by membership in `stages`.  With that filter the branch
    # never fired for the one role it was written for.
    dedicated = catalog_path.parent / "tool-catalog.json"
    catalog = _load_catalog(dedicated)
    if role:
        for stage, roles in (catalog.get("dedicated_stages") or {}).items():
            if role in (roles or []):
                return stage, "dedicated"
    state_name = state.get("state")
    if state_name in ("finalizing", "closed"):
        return ("close" if "close" in stages else stages[-1]), "run_state"
    completed = registry.get("completed_stages", registry.get("completed", []))
    if isinstance(completed, list) and completed:
        for stage in stages:
            if stage not in completed:
                return stage, "registry"
    return stages[0], "default"


def _stage_card(catalog_path: Path, name: str) -> dict[str, Any]:
    card = _read_json(catalog_path.parent / "stages" / f"{name}.json", "stage card")
    if card.get("schema") != STAGE_SCHEMA or card.get("stage") != name:
        raise ToolctlError(f"invalid stage card for {name!r}")
    return card


def _role_policy(catalog: dict[str, Any], role: str) -> dict[str, Any]:
    policy = catalog.get("role_policy")
    roles = policy.get("roles") if isinstance(policy, dict) else None
    if not isinstance(roles, dict) or role not in roles or not isinstance(roles[role], dict):
        known = sorted(roles) if isinstance(roles, dict) else []
        raise ToolctlError(f"unknown role {role!r}; known roles: {', '.join(known)}")
    return roles[role]


def _forbidden_actions(policy: dict[str, Any]) -> list[str]:
    forbidden = list(policy.get("forbidden_actions") or [])
    if policy.get("may_edit_kernel") is False:
        forbidden.append("edit_kernel")
    if policy.get("may_spawn") is False:
        forbidden.append("spawn_subagent")
    if policy.get("may_measure") is False:
        forbidden.append("measure")
    return sorted(set(str(item) for item in forbidden))


def _open_obligations(state: dict[str, Any]) -> list[dict[str, Any]]:
    obligations = state.get("obligations")
    if not isinstance(obligations, list):
        return []
    return [
        {"id": item.get("obligation_id"), "kind": item.get("kind"), "reason": item.get("reason")}
        for item in obligations
        if isinstance(item, dict) and item.get("status") == "open"
    ]


def _artifact_refs(registry: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    refs = registry.get("artifact_refs", registry.get("artifacts", {}))
    if not isinstance(refs, dict):
        refs = {}
    # Evidence references carried by completed state obligations remain useful after a registry
    # writer has not copied them into artifact_refs.
    for obligation in state.get("obligations") or []:
        if not isinstance(obligation, dict) or not obligation.get("evidence_ref"):
            continue
        key = str(obligation.get("obligation_id") or obligation.get("kind") or "obligation")
        refs.setdefault(key, obligation["evidence_ref"])
    return refs


def _allowed_for_role(card: dict[str, Any], role: str) -> list[str]:
    """Return this role's stage-scoped public tool names, never a global catalog inventory."""
    tools = card.get("tools")
    if not isinstance(tools, dict):
        raise ToolctlError(f"stage card {card.get('stage')!r} has no tools object")
    allowed = tools.get("allowed")
    by_role = tools.get("allowed_by_role")
    if not isinstance(allowed, list) or not isinstance(by_role, dict):
        raise ToolctlError(f"stage card {card.get('stage')!r} has no role-scoped allow-list")
    role_allowed = by_role.get(role)
    if not isinstance(role_allowed, list):
        raise ToolctlError(f"role {role!r} is not permitted to operate stage {card.get('stage')!r}")
    if any(not isinstance(name, str) or name not in allowed for name in role_allowed):
        raise ToolctlError(f"stage card {card.get('stage')!r} has an invalid allow-list for role {role!r}")
    return list(role_allowed)


def stage_status(catalog_path: Path, work: Path, role: str) -> dict[str, Any]:
    surface = _load_role_surface(catalog_path, role)
    if surface is not None:
        # A dedicated worker must not learn the pack inventory merely to discover its one facade.
        # Its assigned stage is explicit and does not wait for the owner's global run registry.
        return {
            "schema": "toolctl.stage/1",
            "run": work.name,
            "role": role,
            "current": surface["stage"],
            "stage_source": "role_surface",
            "required": dict(surface.get("required") or {}),
            "next": None,
            "allowed": [tool["name"] for tool in surface["tools"]],
            "forbidden": list(surface.get("forbidden") or []),
            "artifact_refs": {},
            "state": "dedicated",
            "enforcement": dict(surface.get("enforcement") or {}),
            "reference": surface.get("reference"),
            "standing_references": dict(surface.get("standing_references") or {}),
            "evidence_roles": dict(surface.get("evidence_roles") or {}),
        }
    catalog = _load_catalog(catalog_path)
    policy = _role_policy(catalog, role)
    registry, state = _run_registry(work), _run_state(work)
    current, stage_source = _infer_stage(catalog_path, registry, state, role)
    card = _stage_card(catalog_path, current)
    allowed = _allowed_for_role(card, role)
    required = dict(card.get("required") or {})
    open_obligations = _open_obligations(state)
    if open_obligations:
        required["open_obligations"] = open_obligations
    run = registry.get("run", registry.get("run_id", state.get("run_id", work.name)))
    return {
        "schema": "toolctl.stage/1",
        "run": run,
        "role": role,
        "current": current,
        # How `current` was chosen. `default` means NOTHING in this work root named a stage: the
        # value is the graph's first stage, not an instruction. A role that is not the pipeline's
        # entry owner should treat `default` as missing context rather than as its assignment.
        "stage_source": stage_source,
        "required": required,
        "next": card.get("next"),
        "allowed": allowed,
        "forbidden": _forbidden_actions(policy),
        "artifact_refs": _artifact_refs(registry, state),
        "state": state.get("state", "untracked"),
        "enforcement": (catalog.get("role_policy") or {}).get("enforcement", {}),
        # The pack routes one document per stage and carries a standing set on every card. Neither
        # was in this projection, so the only way to learn either was to open the card JSON -- which
        # nothing tells a worker to do. A routing contract the routed party cannot read is prose.
        "reference": (card.get("stage_spec") or {}).get("reference"),
        "standing_references": dict(card.get("standing_references") or {}),
        # Measurement tools are classified `internal`, so `exec` refuses them by design and the
        # stage allow-list never names one. This map is the supported path: topic -> script to run
        # as direct-host Bash. Without it the worker concludes it may not measure at all.
        "evidence_roles": dict(card.get("evidence_roles") or {}),
    }


def _explicit_artifacts(values: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise ToolctlError("--artifact must use NAME=REF")
        name, ref = item.split("=", 1)
        if not name or not ref:
            raise ToolctlError("--artifact NAME and REF must be non-empty")
        out.append((name, ref))
    return out


def acquire_for_stage(
    catalog_path: Path,
    work: Path,
    role: str,
    artifacts: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage_module, _ = _context_modules()
    projection = stage_status(catalog_path, work, role)
    try:
        context, lease = stage_module.acquire_context(
            work, _catalog_root(catalog_path), projection,
            _explicit_artifacts(artifacts or []),
        )
        refs = stage_module.context_projection(
            work, _catalog_root(catalog_path), context, lease
        )
    except (OSError, ValueError, KeyError) as exc:
        raise ToolctlError(f"context acquire failed: {exc}") from exc
    return projection, context, dict(lease, **refs)


def stage_with_context(catalog_path: Path, work: Path, role: str) -> dict[str, Any]:
    projection, context, lease = acquire_for_stage(catalog_path, work, role)
    projection.update({
        "context_ref": lease["context_ref"],
        "context_hash": context["context_hash"],
        "lease_ref": lease["lease_ref"],
    })
    return projection


def _receipt_path(work: Path, receipt: str | None) -> Path:
    if receipt:
        candidate = Path(receipt).expanduser()
        return candidate if candidate.is_absolute() else work / candidate
    stamp = _now().replace(":", "").replace("+", "_")
    return work / "toolctl-receipts" / f"{stamp}-{os.getpid()}.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_PROVENANCE_SCHEMA_ROLES = {
    "kernel_opt.sweep_result/1": "sweep",
    "kernel_opt.branch_converge/1": "branch",
    "kernel_opt.arm_result/1": "arm",
    "kernel_opt.final_report/3": "final_report",
}


def _validate_transition_artifact(path: Path, observed_sha256: str) -> dict[str, Any] | None:
    """Verify schema, producer sidecar, hashes, original exit, and process-time freshness."""
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("schema") == RECEIPT_SCHEMA:
        result = document.get("result") or {}
        if result.get("exit_code") != 0 or result.get("status") != "ok":
            raise ToolctlError(
                f"transition refuses failed command receipt {path}; pipelines may not mask child exit")
        return None
    role = _PROVENANCE_SCHEMA_ROLES.get(document.get("schema"))
    if role is None:
        if path.name == "close_audit.json":
            role = "audit"
        elif path.name in ("plain_champion.json", "champion.json"):
            role = "champion"
        else:
            return None
    receipt_path = Path(str(path) + ".receipt.json")
    receipt = _read_json(receipt_path, f"{role} producer receipt")
    try:
        import canonical_record
    except ModuleNotFoundError as exc:
        raise ToolctlError(f"producer receipt validator unavailable: {exc}") from exc
    findings = canonical_record.validate_producer_receipt(receipt)
    if findings:
        raise ToolctlError(
            f"invalid producer receipt for {path.name}: {findings[0]['field']}: "
            f"{findings[0]['detail']}")
    if receipt.get("artifact_role") != role:
        raise ToolctlError(
            f"producer receipt role {receipt.get('artifact_role')!r} != expected {role!r}")
    artifact = receipt.get("artifact") or {}
    if artifact.get("sha256") != observed_sha256:
        raise ToolctlError(f"producer receipt hash does not match {path}")
    if artifact.get("schema") != str(document.get("schema") or "legacy.unspecified/1"):
        raise ToolctlError(f"producer receipt schema does not match {path}")
    if receipt.get("timing") != "process_time":
        raise ToolctlError(
            f"{path.name} receipt is late_reconstruction; it is evidence but cannot advance clean state")
    if receipt_path.stat().st_mtime + 1e-6 < path.stat().st_mtime:
        raise ToolctlError(f"producer receipt predates artifact bytes: {path}")
    produced_at = receipt.get("produced_at")
    try:
        produced_epoch = dt.datetime.fromisoformat(
            produced_at.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError) as exc:
        raise ToolctlError(f"producer receipt has invalid produced_at: {receipt_path}") from exc
    # One-second tolerance covers filesystems whose mtime has coarser precision than ISO timestamps.
    if produced_epoch + 1.0 < path.stat().st_mtime:
        raise ToolctlError(f"producer receipt is stale for artifact bytes: {path}")
    return receipt


def _enforce_deadline(work: Path, state: dict[str, Any], stage: str) -> float | None:
    """Fail closed at the task deadline and reserve close time for reconciliation."""
    module = _run_state_module()
    if module is None:
        return None
    remaining = module.remaining_seconds(state)
    if remaining is None:
        return None
    if remaining <= 0:
        try:
            module.set_terminal_state(
                str(work / "run_state.json"), state["generation"], "partial_time_limit",
                "task deadline reached before a new operation could start")
        except (OSError, ValueError):
            pass
        raise ToolctlError("task deadline reached; run is partial_time_limit, not a search conclusion")
    reserve = ((state.get("deadline") or {}).get("close_reserve_s") or 0)
    if stage != "close" and remaining <= reserve:
        raise ToolctlError(
            f"only {remaining:.1f}s remain, inside the {reserve}s close reserve; do not start "
            "new optimization work")
    return remaining


def transition_stage(catalog_path: Path, work: Path, role: str, next_stage: str,
                     artifacts: list[str], receipt: str | None) -> dict[str, Any]:
    """Atomically advance the run-state after validating the current stage contract.

    ``--artifact NAME=PATH`` intentionally makes the artifact mapping explicit: a transition
    cannot succeed because a same-named file happened to exist elsewhere in the work tree.
    """
    catalog = _load_catalog(catalog_path)
    _role_policy(catalog, role)
    state_path = work / "run_state.json"
    state = _run_state(work)
    if not state:
        raise ToolctlError("workflow transition requires run_state.json; initialize it first")
    module = _run_state_module()
    if module is None:
        raise ToolctlError("this pack does not ship run_state.py and cannot transition workflow state")
    state_errors = module.validate_state(state)
    if state_errors:
        raise ToolctlError(f"invalid run state: {state_errors[0]['field']}: "
                           f"{state_errors[0]['detail']}")
    current, _ = _infer_stage(catalog_path, _run_registry(work), state, role)
    _enforce_deadline(work, state, current)
    card = _stage_card(catalog_path, current)
    if card.get("next") != next_stage:
        raise ToolctlError(
            f"invalid transition {current!r} -> {next_stage!r}; expected {card.get('next')!r}")
    # A transition is a toolctl control-plane operation, not a stage tool.  The role is
    # validated against the catalog above; stages that delegate all work to a child may have
    # an intentionally empty public-tool surface and must still be able to advance after
    # validating the child's declared artifacts.
    supplied: dict[str, Path] = {}
    for item in artifacts:
        if "=" not in item:
            raise ToolctlError("--artifact must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser()
        path = path if path.is_absolute() else work / path
        path = path.resolve()
        if not name or name in supplied:
            raise ToolctlError("transition artifact names must be unique and non-empty")
        if not path.is_file():
            raise ToolctlError(f"transition artifact {name!r} is not a file: {path}")
        try:
            path.relative_to(work.resolve())
        except ValueError as exc:
            raise ToolctlError(
                f"transition artifact {name!r} is outside the durable work root: {path}") from exc
        supplied[name] = path
    required = card.get("required") or {}
    missing = [name for name in required.get("artifacts", ()) if name not in supplied]
    if missing:
        raise ToolctlError(f"transition missing required artifacts for {current}: {missing}")
    lifecycle = _journal_lifecycle(work)
    if not lifecycle["ok"]:
        first = lifecycle["findings"][0]
        raise ToolctlError(f"journal lifecycle blocks transition: {first['kind']}: "
                           f"{first['detail']}")
    artifact_hashes = {name: _sha256_file(path) for name, path in supplied.items()}
    # The binding is named, not walrus-bound. A walrus inside a comprehension binds in the
    # ENCLOSING scope on purpose (the loop variables are the only scoped names), so spelling this
    # `(receipt := ...)` overwrote the `receipt` parameter declared above with whatever the last
    # artifact validated to -- a dict when one carried a producer receipt, which then reached
    # `_receipt_path` and raised `TypeError` on `Path(dict)`, and `None` when none did, which
    # silently discarded the caller's `--receipt` and wrote to the timestamp fallback instead.
    producer_receipts = {}
    for name, path in supplied.items():
        artifact_receipt = _validate_transition_artifact(path, artifact_hashes[name])
        if artifact_receipt is not None:
            producer_receipts[name] = artifact_receipt
    _, acquired_context, acquired_lease = acquire_for_stage(catalog_path, work, role)
    stage_module, _ = _context_modules()
    roots = stage_module.artifact_roots(work, _catalog_root(catalog_path))
    receipt_path = _receipt_path(work, receipt)
    receipt_doc = {
        "schema": RECEIPT_SCHEMA,
        "run": state["run_id"],
        "role": role,
        "stage": current,
        "tool": "stage-transition",
        "timestamp": _now(),
        "result": {"status": "ok", "exit_code": 0},
        "ref": str(receipt_path),
        "execution_model": "state_transition",
        "next_stage": next_stage,
        "deadline_remaining_s": _enforce_deadline(work, state, current),
        "context_hash": acquired_context["context_hash"],
        "lease_id": acquired_lease["lease_id"],
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": artifact_hashes[name],
                "artifact_ref": stage_module.contracts.normalize_artifact_ref(str(path), roots),
            }
            for name, path in supplied.items()
        },
        "producer_receipts": producer_receipts,
    }
    _write_json(receipt_path, receipt_doc)
    try:
        module.transition(
            str(state_path), state["generation"], current, next_stage, role,
            str(receipt_path), artifact_hashes)
    except (OSError, ValueError) as exc:
        raise ToolctlError(f"stage transition refused: {exc}") from exc
    return receipt_doc


def execute(catalog_path: Path, work: Path, role: str, stage: str | None, name: str,
            argv: list[str], receipt: str | None) -> tuple[int, dict[str, Any]]:
    surface = _load_role_surface(catalog_path, role)
    if surface is not None:
        effective_stage = surface["stage"]
        if stage is not None and stage != effective_stage:
            raise ToolctlError(f"requested stage {stage!r} does not match dedicated stage {effective_stage!r}")
        tool = _surface_tool(surface, name)
        allowed = [item["name"] for item in surface["tools"]]
        run = work.name
    else:
        catalog = _load_catalog(catalog_path)
        _role_policy(catalog, role)
        registry, state = _run_registry(work), _run_state(work)
        current_stage, _ = _infer_stage(catalog_path, registry, state, role)
        if stage is not None and stage != current_stage:
            raise ToolctlError(f"requested stage {stage!r} does not match current stage {current_stage!r}")
        effective_stage = current_stage
        card = _stage_card(catalog_path, effective_stage)
        allowed = _allowed_for_role(card, role)
        tool = _tool(catalog, name)
        run = registry.get("run", registry.get("run_id", state.get("run_id", work.name)))
    classification = tool.get("classification")
    if classification != "public":
        raise ToolctlError(f"tool {name!r} is {classification!r}, not a public executable tool")
    if name not in allowed:
        raise ToolctlError(f"tool {name!r} is not allowed for role {role!r} in stage {effective_stage!r}")
    state_for_deadline = _run_state(work)
    deadline_remaining = _enforce_deadline(work, state_for_deadline, effective_stage) if state_for_deadline else None
    root = _catalog_root(catalog_path)
    target = _safe_tool_path(root, tool)
    command = tool.get("command")
    if not isinstance(command, list) or not command:
        raise ToolctlError(f"tool {name!r} has no command")
    executable = command[0]
    command = [sys.executable if executable == "python3" else executable]
    command += [str(target) if part == tool.get("path") else str(part) for part in tool.get("command", [])[1:]]
    command.extend(argv)
    _, acquired_context, acquired_lease = acquire_for_stage(catalog_path, work, role)
    resolved_executable = Path(command[0])
    if not resolved_executable.is_absolute():
        located = shutil.which(command[0])
        if located is None:
            raise ToolctlError(f"executable is not available on PATH: {command[0]!r}")
        resolved_executable = Path(located)
    resolved_executable = resolved_executable.resolve()
    executable_sha256 = _sha256_file(resolved_executable)
    stage_module, _ = _context_modules()
    tool_ref = stage_module.contracts.normalize_artifact_ref(
        str(target), stage_module.artifact_roots(work, root)
    )
    started = _now()
    argv_sha256 = hashlib.sha256(
        json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        result = subprocess.run(command, cwd=str(work), check=False)
        exit_code = result.returncode
        status = "ok" if exit_code == 0 else "failed"
    except OSError as exc:
        exit_code = 127
        status = "launch_error"
        launch_error = str(exc)
    receipt_path = _receipt_path(work, receipt)
    receipt_doc: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "run": run,
        "role": role,
        "stage": effective_stage,
        "tool": name,
        "argv_hash": {"algorithm": "sha256", "value": argv_sha256},
        "argv_sha256": argv_sha256,
        "timestamp": started,
        "result": {"status": status, "exit_code": exit_code},
        "ref": str(receipt_path),
        "execution_model": "direct_host_subprocess",
        "command": command,
        "executable": {
            "name": resolved_executable.name,
            "sha256": executable_sha256,
        },
        "executable_sha256": executable_sha256,
        "tool_ref": tool_ref,
        "context_hash": acquired_context["context_hash"],
        "lease_id": acquired_lease["lease_id"],
        "deadline_remaining_s": deadline_remaining,
    }
    if "launch_error" in locals():
        receipt_doc["result"]["error"] = launch_error
    _write_json(receipt_path, receipt_doc)
    return exit_code, receipt_doc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, help="generated runtime/tool-catalog.json to use")
    parser.add_argument("--describe", action="store_true",
                        help="emit toolctl's own machine-readable metadata and exit")
    parser.add_argument("--format", choices=["json"],
                        help="output format for --describe (JSON is the stable machine form)")
    parser.add_argument("--selftest", action="store_true", help="run the offline registry self-test")
    sub = parser.add_subparsers(dest="command")

    describe = sub.add_parser("describe", help="render one catalog tool record")
    describe.add_argument("tool")
    describe.add_argument("--role", help="use a role-scoped surface when one is emitted")
    describe.add_argument("--json", action="store_true", help="emit JSON (the stable form)")
    describe.add_argument("--format", dest="describe_format", choices=["json"],
                          help="emit JSON (equivalent to --json)")

    stage = sub.add_parser("stage", help="render the active stage for a work directory")
    stage.add_argument("--work", required=True, type=Path)
    stage.add_argument("--role", required=True)
    stage.add_argument("--json", action="store_true", help="emit JSON (the stable form)")

    context_parser = sub.add_parser("context", help="acquire, query, rotate, or resume context")
    context_sub = context_parser.add_subparsers(dest="context_command", required=True)
    acquire = context_sub.add_parser("acquire", help="create or reuse the active context lease")
    acquire.add_argument("--work", required=True, type=Path)
    acquire.add_argument("--role", required=True)
    acquire.add_argument("--artifact", action="append", default=[], metavar="NAME=REF")
    acquire.add_argument("--json", action="store_true")
    query_parser = context_sub.add_parser("query", help="run one lease-bound bounded query")
    query_parser.add_argument("--work", required=True, type=Path)
    query_parser.add_argument("--role", required=True)
    query_parser.add_argument("--selector", choices=("current-stage", "identity", "measurement",
                                                     "obligations", "recall", "actions", "artifacts"))
    query_parser.add_argument("--ref", help="exact context artifact URI")
    query_parser.add_argument("--artifact", help="artifact name or typed URI for content queries")
    query_parser.add_argument("--json-pointer", "--pointer", dest="json_pointer")
    query_parser.add_argument("--markdown-heading", "--heading", dest="markdown_heading")
    query_parser.add_argument("--python-symbol", "--symbol", dest="python_symbol")
    query_parser.add_argument("--text-window", metavar="START:COUNT")
    query_parser.add_argument("--max-bytes", type=int)
    query_parser.add_argument("--json", action="store_true")
    rotate = context_sub.add_parser("rotate", help="write a resume capsule and settle the lease")
    rotate.add_argument("--work", required=True, type=Path)
    rotate.add_argument("--role", required=True)
    rotate.add_argument("--json", action="store_true")
    resume = context_sub.add_parser("resume", help="verify a capsule and reacquire context")
    resume.add_argument("--work", required=True, type=Path)
    resume.add_argument("--role", required=True)
    resume.add_argument("--capsule", type=Path)
    resume.add_argument("--json", action="store_true")

    execute_parser = sub.add_parser("exec", help="run a registry tool directly and write a receipt")
    execute_parser.add_argument("--work", default=Path("."), type=Path)
    execute_parser.add_argument("--role", required=True,
                                help="role declared by the current catalog's role policy")
    execute_parser.add_argument("--stage")
    execute_parser.add_argument("--receipt",
                                help="receipt path, relative to --work unless absolute")
    execute_parser.add_argument("tool")
    execute_parser.add_argument("argv", nargs=argparse.REMAINDER)
    transition_parser = sub.add_parser("transition",
                                       help="validate artifacts and advance authoritative run state")
    transition_parser.add_argument("--work", default=Path("."), type=Path)
    transition_parser.add_argument("--role", required=True)
    transition_parser.add_argument("--to", required=True, dest="next_stage")
    transition_parser.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    transition_parser.add_argument("--receipt",
                                   help="receipt path, relative to --work unless absolute")
    return parser


def _selftest() -> int:
    # NO WALRUS MAY REBIND A PARAMETER. Checked on the syntax rather than through a transition
    # because the bug was invisible at the call site: a walrus inside a comprehension binds in the
    # enclosing function scope, so `(receipt := ...)` in `transition_stage`'s producer-receipt
    # comprehension silently replaced the `receipt` argument -- crashing on `Path(dict)` when an
    # artifact carried a producer receipt, and discarding the caller's `--receipt` when none did.
    # A shape check is the cheap guard for a whole class, and this class shipped in every pack.
    import ast as _ast
    _tree = _ast.parse(Path(__file__).read_text())
    for _fn in (n for n in _ast.walk(_tree) if isinstance(n, _ast.FunctionDef)):
        _params = {a.arg for a in (*_fn.args.posonlyargs, *_fn.args.args, *_fn.args.kwonlyargs)}
        _clobbered = sorted({
            n.target.id for n in _ast.walk(_fn)
            if isinstance(n, _ast.NamedExpr) and isinstance(n.target, _ast.Name)
            and n.target.id in _params})
        assert not _clobbered, f"{_fn.name} rebinds its parameter(s) {_clobbered} with a walrus"

    root = Path(tempfile.mkdtemp(prefix="toolctl_selftest_"))
    try:
        (root / "runtime" / "stages").mkdir(parents=True)
        (root / "scripts").mkdir()
        noop = root / "scripts" / "noop.py"
        noop.write_text("import sys\nprint('noop', *sys.argv[1:])\n")
        internal = root / "scripts" / "internal.py"
        internal.write_text("raise SystemExit('internal command must not run')\n")
        other = root / "scripts" / "other.py"
        other.write_text("raise SystemExit('wrong-stage command must not run')\n")
        catalog = {
            "schema": CATALOG_SCHEMA,
            "tools": [
                {
                    "name": "noop",
                    "classification": "public",
                    "path": "scripts/noop.py",
                    "command": ["python3", "scripts/noop.py"],
                    "flags": ["--help"],
                },
                {
                    "name": "internal",
                    "classification": "internal",
                    "path": "scripts/internal.py",
                    "command": ["python3", "scripts/internal.py"],
                    "flags": [],
                },
                {
                    "name": "other",
                    "classification": "public",
                    "path": "scripts/other.py",
                    "command": ["python3", "scripts/other.py"],
                    "flags": [],
                },
            ],
            "role_policy": {
                "enforcement": {"local": "advisory"},
                "roles": {
                    "skeptic": {"may_edit_kernel": False, "may_spawn": False, "may_measure": False},
                    "observer": {"may_edit_kernel": False, "may_spawn": False, "may_measure": False},
                },
            },
        }
        catalog_path = root / "runtime" / "tool-catalog.json"
        _write_json(catalog_path, catalog)
        card = {
            "$schema": "../stage-card.schema.json",
            "schema": STAGE_SCHEMA,
            "pack": {"name": "test-pack", "dsl": None, "vendor": None, "platform": "test"},
            "stage": "entry",
            "order": 0,
            "next": "close",
            "required": {"artifacts": ["input.json"], "optional_artifacts": [],
                         "conditional_artifacts": [], "entry_gate": None, "anchor": None,
                         "evidence_roles": []},
            "tools": {"allowed": ["noop", "internal"], "allowed_by_role": {"skeptic": ["noop"],
                      "observer": []}, "public": ["noop"], "internal": ["internal"],
                      "developer_only": []},
            "artifacts": {"produces": {}, "consumes": {}},
            "evidence_roles": {},
            "role_policy": catalog["role_policy"],
        }
        _write_json(root / "runtime" / "stages" / "entry.json", card)
        _write_json(root / "runtime" / "stages" / "close.json", dict(card, stage="close",
                                                                        order=1, next=None))
        _write_json(root / "toolctl-registry.json", {
            "schema": REGISTRY_SCHEMA,
            "run": "run-selftest",
            "current_stage": "entry",
            "artifact_refs": {"input.json": "input.json"},
        })
        (root / "input.json").write_text('{"input": true}\n')
        status = stage_status(catalog_path, root, "skeptic")
        assert status["current"] == "entry"
        assert status["forbidden"] == ["edit_kernel", "measure", "spawn_subagent"]
        rc, receipt = execute(catalog_path, root, "skeptic", None, "noop", ["value"], None)
        assert rc == 0 and receipt["run"] == "run-selftest"
        assert Path(receipt["ref"]).is_file()
        assert receipt["execution_model"] == "direct_host_subprocess"
        (root / "toolctl-registry.json").unlink()
        status = stage_status(catalog_path, root, "skeptic")
        assert status["current"] == "entry", status
        for forbidden_name, expected in (("internal", "not a public executable"),
                                         ("other", "not allowed for role"),
                                         ("noop", "does not match current stage")):
            try:
                execute(catalog_path, root, "skeptic",
                        "close" if forbidden_name == "noop" else None,
                        forbidden_name, [], None)
                raise AssertionError(f"{forbidden_name} execution was not rejected")
            except ToolctlError as exc:
                assert expected in str(exc), exc
        try:
            execute(catalog_path, root, "observer", None, "noop", [], None)
            raise AssertionError("role without stage permission was not rejected")
        except ToolctlError as exc:
            assert "not allowed for role" in str(exc), exc
        _write_json(root / "run_state.json", {
            "run_id": "run-from-state",
            "state": "closed",
            "obligations": [{
                "obligation_id": "report",
                "kind": "report",
                "status": "done",
                "evidence_ref": "final_report.json",
            }],
        })
        status = stage_status(catalog_path, root, "skeptic")
        assert status["run"] == "run-from-state" and status["current"] == "close"
        assert status["artifact_refs"]["report"] == "final_report.json"

        # The dedicated role surface remains operable after the full catalog is removed:
        # this proves stage / describe / exec do not need to open the general inventory.
        sweep = root / "scripts" / "sweepctl.py"
        sweep.write_text("import sys\nprint('sweepctl', *sys.argv[1:])\n")
        role_dir = root / "runtime" / "roles"
        role_dir.mkdir()
        surface = {
            "schema": ROLE_SURFACE_SCHEMA,
            "role": "triton_sweep",
            "stage": "sweep",
            "required": {"artifacts": ["preflight-ready.json"]},
            "forbidden": ["edit_kernel", "spawn_subagent"],
            "enforcement": {"local": "advisory"},
            "tools": [{
                "name": "sweepctl",
                "classification": "public",
                "path": "scripts/sweepctl.py",
                "command": ["python3", "scripts/sweepctl.py"],
            }],
        }
        _write_json(role_dir / "triton_sweep.json", surface)
        catalog_path.unlink()
        dedicated = stage_status(catalog_path, root, "triton_sweep")
        assert dedicated["current"] == "sweep" and dedicated["allowed"] == ["sweepctl"]
        assert _surface_tool(_load_role_surface(catalog_path, "triton_sweep"), "sweepctl")["name"] == "sweepctl"
        rc, receipt = execute(catalog_path, root, "triton_sweep", "sweep", "sweepctl",
                              ["--help"], None)
        assert rc == 0 and receipt["stage"] == "sweep"
        try:
            execute(catalog_path, root, "triton_sweep", None, "noop", [], None)
            raise AssertionError("dedicated role could access a non-surface tool")
        except ToolctlError as exc:
            assert "not exposed" in str(exc), exc
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("[toolctl] SELFTEST PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.describe:
        print(json.dumps(_native_description(), indent=2))
        return 0
    if args.selftest:
        return _selftest()
    if not args.command:
        _parser().print_help(sys.stderr)
        return 2
    catalog_path = (args.catalog or _default_catalog_path()).expanduser().resolve()
    try:
        if args.command == "describe":
            surface = _load_role_surface(catalog_path, args.role) if args.role else None
            print(json.dumps(_surface_tool(surface, args.tool) if surface is not None
                             else _tool(_load_catalog(catalog_path), args.tool), indent=2))
            return 0
        if args.command == "stage":
            print(json.dumps(stage_with_context(catalog_path, args.work.expanduser().resolve(), args.role),
                             indent=2))
            return 0
        if args.command == "context":
            stage_module, query_module = _context_modules()
            work = args.work.expanduser().resolve()
            pack = _catalog_root(catalog_path)
            if args.context_command == "acquire":
                _, context, lease = acquire_for_stage(
                    catalog_path, work, args.role, args.artifact
                )
                print(json.dumps({
                    "schema": "toolctl.context-acquire/1",
                    "run_id": context["run_id"],
                    "generation": context["generation"],
                    "role": context["role"],
                    "stage": context["current_stage"],
                    "context_ref": lease["context_ref"],
                    "context_hash": context["context_hash"],
                    "lease_ref": lease["lease_ref"],
                    "lease_id": lease["lease_id"],
                    "status": lease["status"],
                }, indent=2))
                return 0
            projection = stage_status(catalog_path, work, args.role)
            if args.context_command == "query":
                paths = stage_module.context_paths(work)
                if not paths["context"].is_file() or not paths["lease"].is_file():
                    acquire_for_stage(catalog_path, work, args.role)
                window = None
                if args.text_window is not None:
                    try:
                        start, count = args.text_window.split(":", 1)
                        window = (int(start), int(count))
                    except ValueError as exc:
                        raise ToolctlError("--text-window must use START:COUNT integers") from exc
                result = query_module.leased_query(
                    work, pack, args.role, selector=args.selector, ref=args.ref,
                    artifact=args.artifact, json_pointer=args.json_pointer,
                    markdown_heading=args.markdown_heading, python_symbol=args.python_symbol,
                    text_window=window, max_bytes=args.max_bytes,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            if args.context_command == "rotate":
                capsule = stage_module.rotate_context(work, pack, projection)
                print(json.dumps({
                    "schema": "toolctl.context-rotate/1",
                    "capsule_ref": stage_module.contracts.normalize_artifact_ref(
                        str(stage_module.context_paths(work)["capsule"]),
                        stage_module.artifact_roots(work, pack),
                    ),
                    "capsule_sha256": capsule["capsule_sha256"],
                    "lease_status": "settled",
                }, indent=2))
                return 0
            if args.context_command == "resume":
                capsule_path = args.capsule or stage_module.context_paths(work)["capsule"]
                if not capsule_path.is_absolute():
                    capsule_path = work / capsule_path
                context, lease = stage_module.resume_context(
                    work, pack, args.role, projection, capsule_path
                )
                refs = stage_module.context_projection(work, pack, context, lease)
                print(json.dumps({
                    "schema": "toolctl.context-resume/1",
                    "run_id": context["run_id"],
                    "generation": context["generation"],
                    "role": context["role"],
                    "stage": context["current_stage"],
                    **refs,
                }, indent=2))
                return 0
            raise AssertionError(f"unhandled context command {args.context_command!r}")
        if args.command == "exec":
            forwarded = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            code, receipt = execute(catalog_path, args.work.expanduser().resolve(), args.role,
                                    args.stage, args.tool, forwarded, args.receipt)
            print(json.dumps(receipt, indent=2))
            return code
        if args.command == "transition":
            receipt = transition_stage(
                catalog_path, args.work.expanduser().resolve(), args.role, args.next_stage,
                args.artifact, args.receipt)
            print(json.dumps(receipt, indent=2))
            return 0
    except (ToolctlError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[toolctl] ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
