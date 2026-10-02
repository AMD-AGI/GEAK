# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a pinned external Hyperloom specialist command against a local stub.

GEAK does not include Hyperloom. The caller supplies the reviewed checkout and
an independently pinned manifest. This profile executes selected source methods.
It does not execute the specialist factory, actor, runner, or leaf agent.
"""

import __future__

import ast
import json
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from .native_prefix_marker import DECODER, sha, wire

PREFIX_COUNT = 18
TOTAL_TOOLS = 23
SOURCE_SCHEMA = "geak-hyperloom-source-v1"
DISPATCHER = "src/hyperloom/orchestrator/specialists/subprocess_.py"
RUNNER = "src/hyperloom/orchestrator/specialists/runner.py"
LEAF = "src/hyperloom/orchestrator/specialists/leaf.py"
PROMPT = "src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py"
FACTORY = "src/hyperloom/inference_optimizer/cli/executors.py"
SOURCE_FILES = (DISPATCHER, RUNNER, LEAF, PROMPT, FACTORY)
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def add_arguments(parser):
    parser.add_argument("--hyperloom-source", type=Path,
                        help="Use this reviewed external Hyperloom checkout.")
    parser.add_argument("--hyperloom-source-manifest", type=Path,
                        help="Read the Hyperloom source file hashes from this manifest.")
    parser.add_argument("--hyperloom-source-manifest-sha256",
                        help="Require this independently recorded source manifest SHA-256 hash.")
    parser.add_argument("--hyperloom-framework-root", type=Path, action="append", default=[],
                        help="Supply a framework root to the source directory builder. Repeat as needed.")


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("A Hyperloom source input exceeds the byte limit.")
    return raw


def _sources(options):
    if options.prefix_count != PREFIX_COUNT:
        raise ValueError("The Hyperloom specialist profile requires a prefix count of 18.")
    if getattr(options, "effort", None) not in (None, "ultracode"):
        raise ValueError("The Hyperloom specialist profile supports only the source default effort.")
    source = getattr(options, "hyperloom_source", None)
    manifest_path = getattr(options, "hyperloom_source_manifest", None)
    expected = getattr(options, "hyperloom_source_manifest_sha256", None)
    if not source or not manifest_path or not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
        raise ValueError("Supply the Hyperloom checkout, source manifest, and manifest hash.")
    root = Path(source).resolve(strict=True)
    manifest_raw = _read(manifest_path)
    if sha(manifest_raw) != expected:
        raise ValueError("The Hyperloom source manifest hash does not match the explicit pin.")
    manifest = DECODER.decode(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("The Hyperloom source manifest schema is unsupported.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(SOURCE_FILES):
        raise ValueError("The Hyperloom source manifest must pin all five profile source files.")
    sources = {}
    for name in SOURCE_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("A Hyperloom source file escapes the selected checkout.")
        raw = _read(path)
        if not isinstance(files[name], str) or not re.fullmatch(r"[0-9a-f]{64}", files[name]) or sha(raw) != files[name]:
            raise ValueError("A Hyperloom source file differs from the pinned manifest.")
        sources[name] = raw
    roots = []
    for value in getattr(options, "hyperloom_framework_root", []):
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Each Hyperloom framework root must be an existing absolute directory.")
        roots.append(str(path.resolve(strict=True)))
    return sources, roots


def bindings(options):
    """Verify every selected source before any source code or CLI can execute."""
    sources, roots = _sources(options)
    return {"source_manifest_sha256": options.hyperloom_source_manifest_sha256,
            "profile_module_sha256": sha(_read(Path(__file__))),
            "source_sha256": {name: sha(raw) for name, raw in sources.items()},
            "framework_roots_sha256": sha(wire(roots)), "framework_root_count": len(roots)}


def _named(nodes, name):
    matches = []
    for node in nodes:
        if ((isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
                or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                       for target in node.targets))
                or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name)):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError("A required Hyperloom source definition is absent or ambiguous.")
    return matches[0]


def _literal(node):
    value = node.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset":
        if len(value.args) != 1 or value.keywords:
            raise ValueError("The Hyperloom source literal is unsupported.")
        return frozenset(ast.literal_eval(value.args[0]))
    return ast.literal_eval(value)


def _execute(nodes, namespace, filename):
    # Keep source nodes unchanged. Supply only the dependencies of these methods.
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    # Execute only selected AST nodes from the verified source manifest.
    exec(compile(module, filename, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)  # noqa: S102


def _source_command(options, temporary):
    sources, roots = _sources(options)
    trees = {name: ast.parse(raw, filename=name) for name, raw in sources.items()}
    config_node = _named(trees[DISPATCHER].body, "SpecialistSubprocessConfig")
    fields = ("permission_mode", "output_format", "extra_claude_args", "mcp_config_path", "leaf_agents_json")
    defaults = {name: _literal(_named(config_node.body, name)) for name in fields}
    if (defaults["permission_mode"] != "bypassPermissions" or defaults["output_format"] != "stream-json"
            or defaults["extra_claude_args"] != () or defaults["mcp_config_path"] is not None
            or defaults["leaf_agents_json"] is not None):
        raise ValueError("The Hyperloom source defaults differ from the supported specialist profile.")
    denylist = _literal(_named(trees[RUNNER].body, "SPECIALIST_TOOL_DENYLIST"))
    if denylist != frozenset({"KillShell", "SlashCommand"}):
        raise ValueError("The Hyperloom source tool filter differs from the supported specialist profile.")
    preamble = _literal(_named(trees[PROMPT].body, "BASH_KILL_SAFETY_PREAMBLE"))
    safe_builtins = {"list": list, "str": str, "frozenset": frozenset, "sorted": sorted,
                     "BaseException": BaseException, "OSError": OSError}
    leaf_namespace = {"__builtins__": safe_builtins, "json": json,
                      "BASH_KILL_SAFETY_PREAMBLE": preamble}
    leaf_names = ("LEAF_AGENT_NAME", "LEAF_AGENT_TOOLS", "_LEAF_AGENT_PROMPT",
                  "_LEAF_AGENT_DESCRIPTION", "build_leaf_agents_json")
    leaf_nodes = [_named(trees[LEAF].body, name) for name in leaf_names]
    _execute(leaf_nodes, leaf_namespace, LEAF)
    leaf_builder = leaf_namespace["build_leaf_agents_json"]
    leaf_json = leaf_builder()
    agents = DECODER.decode(leaf_json)
    if not isinstance(agents, dict) or set(agents) != {"hyperloom-leaf"}:
        raise ValueError("The Hyperloom source leaf definitions are unsupported.")
    leaf = SimpleNamespace(build_leaf_agents_json=leaf_builder)

    def source_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name != "leaf" or level != 1 or tuple(fromlist) != ("build_leaf_agents_json",):
            raise ValueError("The selected Hyperloom method imports an unsupported dependency.")
        return leaf

    namespace = {"__builtins__": {**safe_builtins, "__import__": source_import}, "Path": Path, "os": os}
    dispatcher_node = _named(trees[DISPATCHER].body, "SpecialistSubprocessDispatcher")
    methods = [_named(dispatcher_node.body, name) for name in ("_writable_dirs", "_build_claude_cmd")]
    _execute(methods, namespace, DISPATCHER)
    config = SimpleNamespace(**defaults, model=options.model, claude_executable=str(options.cli),
                             framework_source_roots=tuple(roots))
    dispatcher = SimpleNamespace(config=config)
    dispatcher._writable_dirs = lambda workspace, worktree: namespace["_writable_dirs"](dispatcher, workspace, worktree)
    work = temporary / "work"
    work.mkdir()
    command = namespace["_build_claude_cmd"](dispatcher, system_prompt_file=temporary / "system.txt",
        system_prompt="Synthetic catalog capture. Return terminal text. Do not use any tool.",
        workspace=work, worktree=None, disallowed_tools=denylist)
    add_dirs = dispatcher._writable_dirs(work, None)
    expected_dirs = [str(work), *dict.fromkeys(roots)]
    expected_command = [str(options.cli), "--print", "--output-format", defaults["output_format"],
                        "--verbose", "--permission-mode", defaults["permission_mode"],
                        "--system-prompt-file", str(temporary / "system.txt"), "--model", options.model,
                        "--disallowedTools", ",".join(sorted(denylist)), "--agents", leaf_json]
    for directory in expected_dirs:
        expected_command.extend(["--add-dir", directory])
    if add_dirs != expected_dirs or command != expected_command:
        raise ValueError("The Hyperloom source command differs from the supported pinned CLI profile.")
    source_command_sha256 = sha(wire(command))
    # Preserve source defaults. The following flags only isolate this capture.
    session_id = str(uuid4())
    command.extend(["--session-id", session_id, "--setting-sources", "", "--settings",
                    '{"disableAllHooks":true}', "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'])
    profile = {"scope": "hyperloom_specialist_main_source_command_builder", "role": "specialist_main",
               "source_entrypoint": DISPATCHER,
               "source_method": "SpecialistSubprocessDispatcher._build_claude_cmd",
               "source_directory_method": "SpecialistSubprocessDispatcher._writable_dirs",
               "source_command_sha256": source_command_sha256,
               "source_defaults": defaults, "model": options.model, "effort_override": None,
               "disallowed_tools": sorted(denylist), "allowed_tools": None,
               "tool_filters": {"disallowed_tools": sorted(denylist), "leaf_agents_sha256": sha(leaf_json.encode())},
               "leaf_agents_sha256": sha(leaf_json.encode()),
               "add_dirs_sha256": sha(wire(add_dirs)), "add_dir_count": len(add_dirs),
               "framework_roots_sha256": sha(wire(roots)),
               "factory_executed": False, "actor_executed": False, "leaf_agent_executed": False,
               "source_execution": "unchanged_selected_ast_methods_and_leaf_definitions",
               "directory_input": "explicit_framework_roots_and_empty_temporary_workspace",
               "isolation_flags": ["session-id", "setting-sources=empty", "disableAllHooks",
                                   "strict-mcp-config", "mcp-config=empty", "IS_SANDBOX=1"],
               "differences_from_default_runner": [
                   "Execute the selected source command builder with an empty temporary workspace.",
                   "Supply explicit framework roots instead of executing source directory discovery.",
                   "Use a fixed prompt and one terminal text response without tool calls.",
                   "Set an explicit native session ID for request selection.",
                   "Disable filesystem settings, hooks, and MCP servers.",
                   "Use a private configuration directory, local endpoint, and fake credentials.",
                   "Set IS_SANDBOX=1 inside the private namespace for the CLI permission check.",
                   "Do not execute the factory, actor, runner, or leaf agent."],
               "limits": ["This capture describes a specialist main request, not a leaf request.",
                          "The source factory, actor, runner, and directory discovery do not execute.",
                          "The caller supplies the framework roots used by the source directory builder.",
                          "The tool count alone does not identify a Hyperloom catalog."]}
    return command, work, session_id, profile


def _specialist_capture(session_id):
    from .produce_catalog import SyntheticCapture

    class SpecialistCapture(SyntheticCapture):
        def validate_headers(self, headers):
            values = {name.lower(): value for name, value in headers.items()}
            if values.get("x-claude-code-agent-id"):
                raise ValueError("The Hyperloom capture received a leaf request.")
            if values.get("x-claude-code-session-id") != self.expected_session_id:
                raise ValueError("The Hyperloom request does not match the selected native session.")
            super().validate_headers(values)

        def message(self, raw):
            body = DECODER.decode(raw.decode("utf-8"))
            tools = body.get("tools")
            if (not isinstance(tools, list) or len(tools) != TOTAL_TOOLS
                    or any(not isinstance(tool, dict) or tool.get("name") == "StructuredOutput" for tool in tools)):
                raise ValueError("The Hyperloom specialist request requires 23 tools without StructuredOutput.")
            metadata = DECODER.decode(body.get("metadata", {}).get("user_id", ""))
            if self.session_id != self.expected_session_id or metadata.get("session_id") != self.expected_session_id:
                raise ValueError("The Hyperloom request body does not match the selected native session.")
            return super().message(raw)

    capture = SpecialistCapture(PREFIX_COUNT)
    capture.expected_session_id = session_id
    return capture


def _native_result(output, terminal):
    results = []
    for raw in output.splitlines():
        value = DECODER.decode(raw.decode("utf-8"))
        if not isinstance(value, dict):
            # Malformed native output retains the existing protocol error type.
            raise RuntimeError("The synthetic Hyperloom output contains an unsupported record.")  # noqa: TRY004
        message = value.get("message")
        if isinstance(message, dict) and any(block.get("type") == "tool_use"
                for block in message.get("content", []) if isinstance(block, dict)):
            raise RuntimeError("The synthetic Hyperloom output contains a tool call.")
        if value.get("type") == "result":
            results.append(value)
    if len(results) != 1 or results[0].get("is_error") or results[0].get("result", "").strip() != terminal:
        raise RuntimeError("The synthetic Hyperloom session did not return the terminal text.")


def capture(options):
    """Run only inside the producer's private network, PID, and device mounts."""
    from .produce_catalog import TERMINAL
    before = bindings(options)
    with tempfile.TemporaryDirectory(prefix="geak-hyperloom-catalog-") as directory:
        temporary = Path(directory)
        command, work, session_id, profile = _source_command(options, temporary)
        recording = _specialist_capture(session_id)
        with recording.server() as endpoint, patch.dict(os.environ, {
            "ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_API_KEY": "synthetic-local-key",
            "CLAUDE_CONFIG_DIR": str(temporary / "config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
            "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        }):
            environment = dict(os.environ)
            for name in ("HYPERLOOM_SPECIALIST_EFFORT", "CLAUDE_CODE_EFFORT_LEVEL",
                         "GEAK_CLAUDE_EFFORT", "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"):
                environment.pop(name, None)
            with subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
                try:
                    output, _ = process.communicate(
                        input=b"Synthetic catalog capture. Return terminal text. Do not use any tool.\n",
                        timeout=options.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    raise RuntimeError("The synthetic Hyperloom capture exceeded its time limit.") from None
                if process.returncode:
                    raise RuntimeError("The synthetic Hyperloom CLI failed. No catalog is accepted.")
            _native_result(output, TERMINAL)
        if recording.errors or recording.catalog is None:
            raise RuntimeError("The synthetic Hyperloom capture did not complete exactly once.")
        if recording.request["model"] != options.model:
            raise RuntimeError("The synthetic Hyperloom request changed the selected model.")
        policy = recording.request.get("portable_policy", {})
        if not policy.get("applied") or not policy.get("only_marker_bytes_changed"):
            raise RuntimeError("The portable cache policy declined the synthetic Hyperloom request.")
    if bindings(options) != before:
        raise RuntimeError("A Hyperloom source changed during the synthetic capture.")
    return recording, profile
