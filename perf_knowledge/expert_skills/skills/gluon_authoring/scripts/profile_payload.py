#!/usr/bin/env python3
"""Validate and query a shell-free AMD profiling payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
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
    capture = payload.get("capture", {})
    if not isinstance(capture, dict) or capture.get("mode", "full") not in ("full", "lite", "isa-only"):
        errors.append("capture.mode must be full, lite or isa-only")
    selector = payload.get("kernel_selector", {})
    if not isinstance(selector, dict) or not isinstance(selector.get("regex", ""), str):
        errors.append("kernel_selector.regex must be a string")
    return errors


def capture_payload(payload: dict[str, Any], raw: bytes, script: Path) -> int:
    """Adapt the pack contract to shared capture; the shared tool owns execution."""
    execution = payload["execution"]
    env = dict(os.environ)
    kind = payload["locus"]["kind"]
    if (kind == "container") != bool(env.get("TILE_KERNEL_CONTAINER")):
        raise ValueError("payload locus.kind disagrees with TILE_KERNEL_CONTAINER; configure the execution locus")
    for key, value in execution["env_allowlist"].items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\0" in value:
            raise ValueError(f"invalid payload environment entry: {key!r}")
        if key in {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LOCUS_ENV"} or key.startswith(("GEAK_GPU_", "TILE_KERNEL_")):
            raise ValueError(f"payload environment may not override execution/lease control: {key}")
        env[key] = value
    host_cwd = Path(execution["cwd"]["host"]).resolve()
    if not host_cwd.is_dir():
        raise ValueError(f"payload host working directory is unavailable: {host_cwd}")
    source = Path(payload["source"]["path"])
    if not source.is_absolute():
        source = host_cwd / source
    env.update(TILE_KERNEL_HOST_WORKDIR=str(host_cwd),
               TILE_KERNEL_CONTAINER_WORKDIR=execution["cwd"]["container"],
               GEAK_PROFILE_PAYLOAD_SHA256=hashlib.sha256(raw).hexdigest(),
               LOCUS_ENV=env.get("LOCUS_ENV", "") + " " + " ".join(execution["env_allowlist"]))
    if kind == "container":
        env["TILE_KERNEL_CONTAINER_CWD"] = execution["cwd"]["container"]
    else:
        for key in ("TILE_KERNEL_CONTAINER_CWD", "TILE_KERNEL_HOST_WORKDIR", "TILE_KERNEL_CONTAINER_WORKDIR"):
            env.pop(key, None)
    out = Path(payload["outputs"]["root"])
    mode = (payload.get("capture") or {}).get("mode", "full")
    mode_args = {"full": [], "lite": ["--lite"], "isa-only": ["--isa-only"]}
    if mode not in mode_args:
        raise ValueError(f"unknown capture.mode: {mode}")
    command = ["bash", str(script.resolve()), out.name, str(out),
               "--source", str(source), "--source-sha256", payload["source"]["sha256"],
               "--dev", payload["device"]["logical"],
               "--kernel", (payload.get("kernel_selector") or {}).get("regex") or out.name,
               *mode_args[mode], "--", *execution["argv"]]
    return subprocess.call(command, cwd=host_cwd, env=env)


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
    capture = sub.add_parser("capture", help="execute the payload through GEAK shared capture")
    capture.add_argument("--payload", type=Path, required=True)
    capture.add_argument("--capture-script", type=Path, required=True)
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
    if args.command == "capture":
        try:
            return capture_payload(payload, raw, args.capture_script)
        except (ValueError, OSError) as exc:
            print(f"profile_payload: {exc}", file=sys.stderr)
            return 2
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
