# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared server-flag semantics and fail-closed live launch verification.

This file is also staged beside extra_env.py and executed directly by benchmark
scripts. It depends only on the standard library and that literal-token helper.
Live proof is local-only: active removals require a loopback benchmark host.
Declared configurations retain last-assignment semantics. Live argv proof is
more conservative: a repeated option containing a removed value is ambiguous
because an unknown backend option may append values instead of replacing them.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, MutableMapping

if __package__:
    from . import extra_env as _extra_env
else:
    import extra_env as _extra_env

_shell_tokens = _extra_env._shell_tokens

@dataclass(frozen=True)
class _Flag:
    name: str
    value: str | None


def _looks_like_flag(token: str) -> bool:
    if token == "-" or not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _parse_flags(text: Any) -> list[_Flag]:
    """Parse long/unknown flags, equals forms, booleans, and JSON values."""

    return _parse_flag_tokens(_shell_tokens(text))


def _parse_flag_tokens(tokens: list[str]) -> list[_Flag]:
    """Parse already-delimited argv with the same semantics as declared flags."""
    flags: list[_Flag] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not _looks_like_flag(token):
            raise ValueError(f"server argument has no flag: {token!r}")
        if "=" in token:
            name, value = token.split("=", 1)
            flags.append(_Flag(name, _canonical_value(value)))
            index += 1
            continue
        value: str | None = None
        if index + 1 < len(tokens) and not _looks_like_flag(tokens[index + 1]):
            value = _canonical_value(tokens[index + 1])
            index += 1
        flags.append(_Flag(token, value))
        index += 1
    return flags


def _canonical_value(value: str) -> str:
    stripped = value.strip()
    if stripped[:1] in "[{" and stripped[-1:] in "]}":
        try:
            return json.dumps(
                json.loads(stripped),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except json.JSONDecodeError:
            pass
    return value


def _flag_map(text: Any) -> OrderedDict[str, _Flag]:
    result: OrderedDict[str, _Flag] = OrderedDict()
    for flag in _parse_flags(text):
        result[flag.name] = flag
    return result


def _render_flags(flags: Iterable[_Flag]) -> str:
    tokens: list[str] = []
    for flag in flags:
        tokens.append(flag.name)
        if flag.value is not None:
            tokens.append(flag.value)
    return shlex.join(tokens)


def _remove_flags(flags: MutableMapping[str, _Flag], specs: Any) -> None:
    """Apply explicit key or key/value removals before current assignments."""
    if specs is None:
        return
    if isinstance(specs, str):
        specs = [specs]
    if not isinstance(specs, (list, tuple)):
        raise TypeError("remove_args must be a string or list of flag specs")
    for spec in specs:
        if not isinstance(spec, str):
            raise TypeError("remove_args entries must be strings")
        for flag in _parse_flags(spec):
            if flag.value is None or flags.get(flag.name) == flag:
                flags.pop(flag.name, None)


def resolve_remove_args(specs: Any, *assignments: Any) -> tuple[str, ...]:
    """Keep removals that explicit current assignments have not re-enabled."""
    if specs is None:
        return ()
    if isinstance(specs, str):
        specs = [specs]
    if not isinstance(specs, (list, tuple)) or any(not isinstance(s, str) for s in specs):
        raise TypeError("remove_args must be a string or list of flag specs")
    current = OrderedDict()
    for args in assignments:
        current.update(_flag_map(args))
    return tuple(sorted({
        _render_flags([flag]) for spec in specs for flag in _parse_flags(spec)
        if flag.name not in current or (flag.value is not None and current[flag.name] != flag)
    }))


SCHEMA = "geak.server_args_validation.v1"
REFERENCE_LOCAL_FLAGS = frozenset({
    "--model-path", "--model", "--tokenizer", "--tokenizer-path", "--served-model-name",
    "--host", "--port", "--nccl-port", "--dist-init-addr", "--base-gpu-id", "--gpu-id-step",
    "--node-rank", "--nnodes", "--tensor-parallel-size", "--tp-size", "--tp",
    "--data-parallel-size", "--dp-size", "--pipeline-parallel-size", "--pp-size",
    "--random-seed", "--download-dir", "--pid", "--profiler-config",
    "--enable-profile-cuda-graph", "--enable-shape-discovery-for-cuda-graph-profile",
    "--enable-profile", "--enable-torch-compile-debug-mode", "--debug-cuda-graph",
})
SEMANTIC_FLAGS = {
    "--model": "model", "--model-path": "model", "--tokenizer": "tokenizer", "--tokenizer-path": "tokenizer",
    "--served-model-name": "served_model_name", "--tensor-parallel-size": "tp", "--tp-size": "tp", "--tp": "tp",
    "--data-parallel-size": "dp", "--dp-size": "dp", "--pipeline-parallel-size": "pp", "--pp-size": "pp",
    "--random-seed": "seed", "--seed": "seed", "--nnodes": "nnodes", "--node-rank": "node_rank",
}
CREDENTIAL_FLAGS = frozenset({"--api-key", "--api_key", "--auth-token", "--password", "--hf-token"})


def serving_env(values: dict[str, str]) -> dict[str, str]:
    prefixes = ("VLLM_", "SGLANG_", "AITER_", "TORCH_", "TORCHINDUCTOR_", "TORCHDYNAMO_", "PYTORCH_", "TRITON_", "NCCL_", "OMP_", "KMP_", "MIOPEN_", "ROCM_", "MORI_", "RCCL_", "HSA_", "HIP_")
    exact = {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM"}
    return {k: v for k, v in sorted(values.items()) if (k.startswith(prefixes) or k in exact)
            and not re.search(r"API_?KEY|AUTH|HEADERS|SECRET|PASSWORD|CREDENTIAL|TOKEN(?!IZER)", k)
            and not k.endswith(("PATH", "_DIR", "_FILE", "_PORT", "_URL", "_ENDPOINT", "_ROOT", "_HOME", "_VISIBLE_DEVICES"))
            and not any(word in k for word in ("PROFILER", "PROFILE", "TRACE", "DUMP"))}


def _env_map(value: str) -> dict[str, str]:
    if value.lstrip().startswith('{'):
        env = json.loads(value)
    else:
        env = dict(token.split('=', 1) for token in _extra_env._shell_tokens(value, canonicalize_json=False))
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("required serving environment must contain string assignments")
    return serving_env(env)


class VerificationError(ValueError):
    """A launch cannot establish the requested removal contract."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "server_args.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extra_env.py": hashlib.sha256(Path(_extra_env.__file__).read_bytes()).hexdigest(),
    }


def _prepare(backend: str, port: Any, remove_args: Any, current_args: str,
             host: str, required_args: str | None = None, required_env: str | None = None,
             required_semantics: str | None = None) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "failed",
        "reason": "invalid_controls",
        "backend": str(backend).lower(),
        "port": str(port),
        "host": host,
        "source_hashes": _source_hashes(),
        "validated_at_unix": time.time(),
    }
    try:
        if required_args is not None and any(token.split("=", 1)[0] in CREDENTIAL_FLAGS
                                             for text in (current_args, required_args)
                                             for token in _shell_tokens(text)):
            raise ValueError("credential-bearing argv cannot be recorded as serving evidence")
        receipt["current_args"] = current_args
        requested = resolve_remove_args(remove_args)
        receipt["requested_remove_args"] = list(requested)
        # Preserve legacy launches with no controls, including older flag syntax.
        if not requested and required_args is None and required_env is None and required_semantics is None:
            receipt.update(status="not_required", reason="no_active_removals", active_remove_args=[])
            return receipt
        active = resolve_remove_args(requested, current_args)
        current_flags = _render_flags(_flag_map(current_args).values())
        receipt.update(active_remove_args=list(active), current_flags=current_flags)
        if not active and required_args is None and required_env is None and required_semantics is None:
            receipt.update(status="not_required", reason="explicit_assignments_reenabled_flags")
            return receipt
        if required_args is not None:
            required = _flag_map(required_args)
            if not required:
                raise ValueError("strict launch requires complete nonempty accepted server flags")
            receipt["required_args"] = _render_flags(required.values())
        if required_env is not None:
            receipt["required_env"] = _env_map(required_env)
        if required_semantics is not None:
            semantic = json.loads(required_semantics)
            receipt["required_semantics"] = semantic
        # A local PID cannot attest a remote endpoint even when ports match.
        # No hostname lookup or DNS-based inference belongs in this proof.
        if host.lower() == "localhost":
            receipt["host"] = "localhost"
        else:
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is None or not address.is_loopback:
                receipt.update(reason="unsupported_host", detail="active removals require a loopback benchmark host")
                return receipt
            receipt["host"] = str(address)
        if not str(port).isdigit() or not 0 < int(port) < 65536:
            raise ValueError("expected port must be an integer between 1 and 65535")
        receipt["port"] = str(int(port))
        receipt["control_digest"] = _digest({
            key: receipt[key] for key in (
                "backend", "host", "port", "requested_remove_args", "active_remove_args", "current_flags"
            )
        } | {key: receipt[key] for key in ("required_args", "required_env", "required_semantics") if key in receipt})
    except (TypeError, ValueError) as error:
        receipt["detail"] = str(error)
    return receipt


def _python_prefix(argv: list[str]) -> int:
    """Return the first Python program operand, rejecting unknown launch forms."""
    if not argv or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(argv[0]).name):
        raise VerificationError("unsupported_entrypoint", "expected a Python server entrypoint")
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in ("-B", "-u", "-E", "-s", "-S", "-I", "-O", "-OO", "-q"):
            index += 1
        elif token in ("-W", "-X") and index + 1 < len(argv):
            index += 2
        elif (token.startswith(("-W", "-X"))) and len(token) > 2:
            index += 1
        else:
            return index
    raise VerificationError("unsupported_entrypoint", "Python command has no server entrypoint")


def server_flag_tokens(argv: list[str], backend: str) -> list[str]:
    """Recognize direct SGLang/vLLM entrypoints; never infer them from substrings."""
    backend = backend.lower()
    if backend not in ("sglang", "vllm"):
        raise VerificationError("unsupported_entrypoint", f"unsupported backend {backend!r}")
    index = 0
    if argv and Path(argv[0]).name == "vllm" and backend == "vllm":
        index = 1
    else:
        index = _python_prefix(argv)
        expected = "sglang.launch_server" if backend == "sglang" else "vllm.entrypoints.openai.api_server"
        if argv[index:index + 2] == ["-m", expected]:
            return argv[index + 2:]
        # Console-script shebangs appear as python /venv/bin/vllm serve MODEL.
        if backend != "vllm" or index >= len(argv) or Path(argv[index]).name != "vllm":
            raise VerificationError("unsupported_entrypoint", f"unrecognized {backend} server command")
        index += 1
    if argv[index:index + 1] != ["serve"]:
        raise VerificationError("unsupported_entrypoint", "vllm command is not 'serve'")
    index += 1
    if index < len(argv) and not _looks_like_flag(argv[index]):
        index += 1  # vllm serve's optional positional model, not a server flag.
    return argv[index:]


def server_semantics(argv: list[str], backend: str) -> dict[str, str | None]:
    flags = _flag_map(shlex.join(server_flag_tokens(argv, backend)))
    out = {"model": None, "tokenizer": None, "served_model_name": None, "tp": "1", "dp": "1", "pp": "1", "seed": None, "nnodes": "1", "node_rank": "0"}
    if backend == "vllm":
        index = 0 if Path(argv[0]).name == "vllm" else _python_prefix(argv)
        if (index < len(argv) and Path(argv[index]).name == "vllm"
                and argv[index + 1:index + 2] == ["serve"]
                and index + 2 < len(argv) and not _looks_like_flag(argv[index + 2])):
            out["model"] = argv[index + 2]
    out.update({SEMANTIC_FLAGS[k]: flag.value for k, flag in flags.items() if k in SEMANTIC_FLAGS})
    if not out["model"]:
        raise ValueError("server argv does not identify a model")
    out["tokenizer"] = out["tokenizer"] or out["model"]
    out["served_model_name"] = out["served_model_name"] or out["model"]
    return out


def _start_ticks(proc_root: Path, pid: int) -> str:
    text = (proc_root / str(pid) / "stat").read_text()
    # comm can contain spaces and ')'; fields after its final ') ' begin at state.
    fields = text.rsplit(") ", 1)[1].split()
    if len(fields) < 20 or fields[0] in ("Z", "X", "x") or not fields[19].isdigit():
        raise VerificationError("unverified_process", "server stat is incomplete or process is not live")
    if int(fields[19]) <= 0:
        raise VerificationError("unverified_process", "server start ticks are empty or invalid")
    return fields[19]


def _boot_id(proc_root: Path) -> str:
    value = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    if not value:
        raise VerificationError("unverified_process", "boot identity is unavailable")
    return value


def _observe(receipt: dict[str, Any], pid: Any, expected_ticks: Any, proc_root: Path) -> None:
    if not str(pid).isdigit() or int(pid) <= 1:
        raise VerificationError("unverified_process", "expected a live server PID greater than one")
    if not str(expected_ticks).isdigit() or int(expected_ticks) <= 0:
        raise VerificationError("unverified_process", "launch-time start ticks are required")
    pid = int(pid)
    receipt.update(pid=pid, expected_start_ticks=str(expected_ticks), boot_id=_boot_id(proc_root))
    uid = (proc_root / str(pid)).stat().st_uid
    if uid != os.geteuid():
        raise VerificationError("unverified_process", "server PID belongs to another effective user")
    before = _start_ticks(proc_root, pid)
    receipt["observed_start_ticks"] = before
    if before != str(expected_ticks):
        raise VerificationError("process_identity_mismatch", "server PID no longer matches launch-time start ticks")
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    if not raw or not raw.endswith(b"\0"):
        raise VerificationError("unverified_process", "server argv is empty or not NUL-delimited")
    argv = [token.decode("utf-8", "strict") for token in raw[:-1].split(b"\0")]
    if "required_args" in receipt and any(
            token.split("=", 1)[0] in CREDENTIAL_FLAGS
            for token in argv):
        raise VerificationError("unsupported_argv", "credential-bearing argv cannot be recorded as serving evidence")
    after = _start_ticks(proc_root, pid)
    if before != after:
        raise VerificationError("process_identity_mismatch", "server identity changed while reading argv")
    receipt.update(argv=argv, argv_digest=_digest(argv), observed_start_ticks=after, process_uid=uid)
    if "required_semantics" in receipt:
        observed_semantics = server_semantics(argv, receipt["backend"])
        receipt["observed_semantics"] = observed_semantics
        if observed_semantics != receipt["required_semantics"]:
            raise VerificationError("reference_semantics_mismatch", "live model, tokenizer, topology or seed differs from the intended launch")
    if "required_env" in receipt:
        env = {}
        for entry in (proc_root / str(pid) / "environ").read_bytes().split(b"\0"):
            key, equal, value = entry.partition(b"=")
            if equal:
                name = key.decode("utf-8")
                if serving_env({name: ""}):
                    env[name] = value.decode("utf-8")
        receipt["observed_serving_env"] = env
        if env != receipt["required_env"]:
            receipt["different_env_keys"] = sorted(k for k in env.keys() | receipt["required_env"].keys()
                                                  if env.get(k) != receipt["required_env"].get(k))
            raise VerificationError("reference_env_mismatch", "live serving environment differs from the intended launch")
    tokens = server_flag_tokens(argv, receipt["backend"])
    if "--" in tokens:
        raise VerificationError("unsupported_argv", "server argv contains an ambiguous option terminator")
    flags: OrderedDict[str, _Flag] = OrderedDict()
    occurrences: dict[str, list[_Flag]] = {}
    for flag in _parse_flag_tokens(tokens):
        flags[flag.name] = flag
        occurrences.setdefault(flag.name, []).append(flag)
    receipt["effective_flags"] = _render_flags(flags.values())
    actual_port = flags.get("--port")
    if (actual_port is None or actual_port.value is None or not actual_port.value.isdigit()
            or int(actual_port.value) != int(receipt["port"])):
        raise VerificationError("port_mismatch", "server argv does not identify the expected port")
    violations = []
    ambiguous = []
    for spec in receipt["active_remove_args"]:
        for flag in _parse_flags(spec):
            values = occurrences.get(flag.name, [])
            if flag.value is not None and len(values) > 1 and flag in values:
                ambiguous.append(_render_flags([flag]))
            if flag.name in flags and (flag.value is None or flag == flags[flag.name]):
                violations.append(_render_flags([flag]))
    if ambiguous:
        receipt["ambiguous_repeated_options"] = ambiguous
        raise VerificationError(
            "ambiguous_repeated_option",
            "a removed value occurs in a repeated option; overwrite semantics are unverified",
        )
    if violations:
        receipt["violations"] = violations
        raise VerificationError("removal_mismatch", "removed flags remain effective in the launched server argv")
    if "required_args" in receipt:
        required = {k: v for k, v in _flag_map(receipt["required_args"]).items() if k not in REFERENCE_LOCAL_FLAGS}
        actual = {k: v for k, v in flags.items() if k not in REFERENCE_LOCAL_FLAGS}
        if actual != required:
            receipt["missing_flags"] = sorted(set(required) - set(actual))
            receipt["unexpected_flags"] = sorted(set(actual) - set(required))
            receipt["different_flags"] = sorted(k for k in required.keys() & actual.keys() if required[k] != actual[k])
            raise VerificationError("reference_args_mismatch", "live serving flags differ from the complete intended launch")
    # Do not accept a process that exited or was recycled during interpretation.
    if _start_ticks(proc_root, pid) != after:
        raise VerificationError("process_identity_mismatch", "server identity changed during validation")
    receipt.update(status="verified", reason="removals_verified")


def validate_launch(*, pid: Any, start_ticks: Any, backend: str, port: Any,
                    remove_args: Any, current_args: str = "", host: str = "127.0.0.1",
                    required_args: str | None = None,
                    required_env: str | None = None, required_semantics: str | None = None,
                    proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Observe an owned launch and return a structured, source-bound result."""
    receipt = _prepare(backend, port, remove_args, current_args, host, required_args, required_env, required_semantics)
    receipt["validation_kind"] = "launch"
    if receipt["status"] == "not_required" or "detail" in receipt:
        return receipt
    try:
        _observe(receipt, pid, start_ticks, Path(proc_root))
    except VerificationError as error:
        receipt.update(status="failed", reason=error.reason, detail=str(error))
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        receipt.update(status="failed", reason="unverified_process", detail=str(error))
    return receipt


def validate_reuse(*, launch_receipt: Any, backend: str, port: Any, remove_args: Any,
                   current_args: str = "", host: str = "127.0.0.1",
                   required_args: str | None = None,
                   required_env: str | None = None, required_semantics: str | None = None,
                   proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Require matching previous proof and re-observe its still-live process."""
    receipt = _prepare(backend, port, remove_args, current_args, host, required_args, required_env, required_semantics)
    receipt["validation_kind"] = "reuse"
    if receipt["status"] == "not_required" or "detail" in receipt:
        return receipt
    try:
        previous = json.loads(Path(launch_receipt).read_text())
        if not isinstance(previous, dict) or previous.get("schema_version") != SCHEMA or previous.get("status") != "verified":
            raise VerificationError("reuse_receipt_mismatch", "reuse requires a previously verified launch receipt")
        for key in ("source_hashes", "control_digest", "backend", "host", "port"):
            if previous.get(key) != receipt[key]:
                raise VerificationError("reuse_receipt_mismatch", f"reuse receipt has different {key}")
        if previous.get("boot_id") != _boot_id(Path(proc_root)):
            raise VerificationError("reuse_receipt_mismatch", "reuse receipt belongs to another boot")
        receipt["launch_receipt"] = str(launch_receipt)
        _observe(receipt, previous.get("pid"), previous.get("observed_start_ticks"), Path(proc_root))
        if receipt.get("argv_digest") != previous.get("argv_digest"):
            raise VerificationError("reuse_receipt_mismatch", "server argv changed since the verified launch")
    except VerificationError as error:
        receipt.update(status="failed", reason=error.reason, detail=str(error))
    except (OSError, UnicodeError, ValueError, TypeError, IndexError) as error:
        receipt.update(status="failed", reason="unverified_reuse", detail=str(error))
    return receipt


def _write_receipt(path: str, receipt: dict[str, Any]) -> None:
    destination = Path(path)
    # Atomic replacement prevents reuse from consuming a partially written proof.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "tokens", "validate", "validate-reuse"):
        command = commands.add_parser(name)
        command.add_argument("--remove-args", default="[]")
        command.add_argument("--current-args", default="")
        if name in ("resolve", "tokens"):
            continue
        command.add_argument("--required-args", default=None)
        command.add_argument("--required-env", default=None)
        command.add_argument("--required-semantics", default=None)
        command.add_argument("--backend", required=True)
        command.add_argument("--host", default="127.0.0.1", help="Loopback benchmark host; remote attestation is unsupported")
        command.add_argument("--port", required=True)
        command.add_argument("--receipt", help="Write this validation's JSON result atomically")
        if name == "validate":
            command.add_argument("--pid", default="")
            command.add_argument("--start-ticks", default="")
        else:
            command.add_argument("--launch-receipt", default="", help="Prior verified launch receipt")
    args = parser.parse_args(argv)
    if args.command == "tokens":
        try:
            tokens = _extra_env._shell_tokens(args.current_args, canonicalize_json=False)
            if any("\0" in token for token in tokens):
                raise ValueError("server arguments cannot contain NUL")
        except (TypeError, ValueError) as error:
            print(f"invalid server arguments: {error}", file=sys.stderr)
            return 2
        for token in tokens:
            sys.stdout.buffer.write(token.encode() + b"\0")
        return 0
    if args.command == "resolve":
        try:
            requested = resolve_remove_args(json.loads(args.remove_args))
            active = resolve_remove_args(requested, args.current_args) if requested else ()
        except (TypeError, ValueError) as error:
            print(f"invalid server argument controls: {error}", file=sys.stderr)
            return 2
        print(json.dumps(list(active), ensure_ascii=False))
        return 0
    try:
        removals = json.loads(args.remove_args)
    except ValueError as error:
        result = {"schema_version": SCHEMA, "status": "failed", "reason": "invalid_controls", "detail": str(error)}
    else:
        kwargs = {"backend": args.backend, "host": args.host, "port": args.port,
                      "remove_args": removals, "current_args": args.current_args,
                      "required_args": args.required_args, "required_env": args.required_env,
                      "required_semantics": args.required_semantics}
        if args.command == "validate":
            result = validate_launch(pid=args.pid, start_ticks=args.start_ticks, **kwargs)
        else:
            result = validate_reuse(launch_receipt=args.launch_receipt, **kwargs)
    if args.receipt:
        try:
            _write_receipt(args.receipt, result)
        except OSError as error:
            print(f"server argument validation: cannot write receipt: {error}", file=sys.stderr)
            return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] in ("verified", "not_required") else 2


if __name__ == "__main__":
    sys.exit(main())
