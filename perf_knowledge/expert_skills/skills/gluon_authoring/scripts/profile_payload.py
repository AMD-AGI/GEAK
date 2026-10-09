#!/usr/bin/env python3
"""Validate and query a shell-free AMD profiling payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "tile.profile-payload/1"
FORBIDDEN_ARGV = {"bash", "sh", "zsh", "fish"}
PCI_BDF = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


def _allowed_host_path(value: Any, roots: tuple[str, ...]) -> bool:
    return isinstance(value, str) and any(value == root or value.startswith(root.rstrip("/") + "/")
                                         for root in roots)


def validate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        return [f"schema must be {SCHEMA}"]
    locus, device, source, execution, outputs, policy = (
        payload.get("locus"), payload.get("device"), payload.get("source"),
        payload.get("execution"), payload.get("outputs"), payload.get("policy"),
    )
    roots = tuple(policy.get("host_allowed_roots") or []) if isinstance(policy, dict) else ()
    if not roots or not all(isinstance(root, str) and root.startswith("/") for root in roots):
        errors.append("policy.host_allowed_roots must be a non-empty absolute path list")
    if not isinstance(locus, dict) or locus.get("kind") not in {"host", "container"}:
        errors.append("locus.kind must be host or container")
    elif not isinstance(locus.get("image_inventory_sha256"), str) or not locus["image_inventory_sha256"]:
        errors.append("locus.image_inventory_sha256 is required")
    if not isinstance(device, dict) or not isinstance(device.get("logical"), str):
        errors.append("device.logical is required")
    elif not isinstance(device.get("pci_bdf"), str) or not PCI_BDF.fullmatch(device["pci_bdf"]):
        errors.append("device.pci_bdf is invalid")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        errors.append("source.path is required")
    elif not re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256", ""))):
        errors.append("source.sha256 must be a lowercase SHA-256")
    if not isinstance(execution, dict):
        errors.append("execution is required")
        execution = {}
    cwd = execution.get("cwd")
    if not isinstance(cwd, dict) or not _allowed_host_path(cwd.get("host"), roots):
        errors.append("execution.cwd.host must be below policy.host_allowed_roots")
    if not isinstance(cwd, dict) or not isinstance(cwd.get("container"), str) or not cwd["container"].startswith("/"):
        errors.append("execution.cwd.container must be absolute")
    argv = execution.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        errors.append("execution.argv must be a non-empty string array")
    else:
        if argv[0] in FORBIDDEN_ARGV or argv[:2] in (["bash", "-lc"], ["sh", "-lc"]):
            errors.append("execution.argv may not invoke a shell")
        if any(item.lstrip().startswith("{") and item.rstrip().endswith("}") for item in argv):
            errors.append("execution.argv may not carry an inline JSON object; use a config file")
    allowlist = execution.get("env_allowlist")
    if not isinstance(allowlist, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                   for k, v in allowlist.items()):
        errors.append("execution.env_allowlist must be a string map")
    if not isinstance(outputs, dict) or not _allowed_host_path(outputs.get("root"), roots):
        errors.append("outputs.root must be below policy.host_allowed_roots")
    return errors


def load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("payload must be a JSON object")
    return value, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "emit"):
        item = sub.add_parser(command)
        item.add_argument("--payload", type=Path, required=True)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        payload = {
            "schema": SCHEMA,
            "policy": {"host_allowed_roots": ["/host/work"]},
            "locus": {"kind": "container", "image_inventory_sha256": "a" * 64},
            "device": {"logical": "0", "pci_bdf": "0000:01:00.0"},
            "source": {"path": "/host/work/source.py", "sha256": "b" * 64},
            "execution": {
                "cwd": {"host": "/host/work", "container": "/work"},
                "argv": ["python3", "bench.py", "--config-file", "/work/config.json"],
                "env_allowlist": {"PYTHONPATH": "/work"},
            },
            "outputs": {"root": "/host/work/profile"},
        }
        assert not validate(payload), validate(payload)
        payload["execution"]["argv"] = ["bash", "-lc", "python3 bench.py"]
        assert any("shell" in error for error in validate(payload))
        print("[profile_payload] SELFTEST PASS")
        return 0
    try:
        payload, raw = load(args.payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"profile_payload: {exc}", file=sys.stderr)
        return 2
    errors = validate(payload)
    if args.command == "validate":
        print(json.dumps({"ok": not errors, "payload_sha256": hashlib.sha256(raw).hexdigest(),
                          "errors": errors}, indent=2))
        return 0 if not errors else 1
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({
        "source": payload["source"],
        "device": payload["device"],
        "locus": payload["locus"],
        "cwd": payload["execution"]["cwd"],
        "argv": payload["execution"]["argv"],
        "env": payload["execution"]["env_allowlist"],
        "outputs": payload["outputs"],
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
