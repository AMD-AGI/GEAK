# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep local helper state inside one native SDK client lifetime."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from dataclasses import is_dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from .cache_proxy import HTTPTransport, SharedToolCacheProxy
from .helper_driver import Unsupported
from .native_helpers import HelperMirror, HelperTransport, NativeHelperRegistry
from .sdk_cache import LEGACY_POLICY, CachedSDKClient, native_shared_tool_cache
from .shared_tool_cache import CachePolicyResult


class _UnchangedPolicy:
    enabled = True

    def apply(self, body, **_kwargs):
        return CachePolicyResult(body, False, "disabled")


def helper_support(options):
    """Return a fixed reason when final tool inputs cannot be bound safely."""
    required = ("env", "session_id", "hooks", "session_store", "session_store_flush", "cwd", "setting_sources")
    if not is_dataclass(options) or any(not hasattr(options, name) for name in required):
        return "unsupported_sdk_options"
    if any(getattr(options, name, None) for name in (
        "resume", "continue_conversation", "fork_session", "resume_session_at", "session_import",
    )):
        return "unsupported_session_options"
    if getattr(options, "can_use_tool", None) is not None:
        return "unsupported_permission_callback"
    if getattr(options, "system_prompt", None) is not None:
        return "unsupported_system_prompt"
    if any((options.hooks or {}).get(name) for name in ("PreToolUse", "PostToolUse", "PostToolUseFailure")):
        return "unsupported_tool_hooks"
    if options.session_store is not None and options.session_store_flush != "eager":
        return "unsupported_mirror_flush"
    # External settings may add input-changing hooks after this SDK callback.
    # Retain those configurations on the original native path.
    settings = getattr(options, "settings", None)
    if settings:
        try:
            parsed = json.loads(settings)
        except (ValueError, TypeError):
            return "unsupported_external_settings"
        if not isinstance(parsed, dict) or parsed.get("hooks"):
            return "unsupported_external_settings"
    if options.setting_sources != []:
        return "unsupported_settings_profile"
    if getattr(options, "plugins", None):
        return "unsupported_external_settings"
    if set(getattr(options, "extra_args", None) or {}) - {"effort"}:
        return "unsupported_cli_overrides"
    return None


class WorkflowSDKClient:
    """Keep original SDK behavior when local helper setup is unsupported.

    Existing permission callbacks and input-changing hooks use the original
    provider path. The helper bridge never returns an allow decision. Cache
    marking can remain enabled independently of local helper support.
    """

    def __init__(self, client_factory, options, *, workflow_request=None, source_root=None,
            helpers_enabled=False, cache_enabled=False, contract_factory=None,
            hook_matcher_factory=None, proxy_factory=SharedToolCacheProxy):
        if type(helpers_enabled) is not bool or type(cache_enabled) is not bool:
            raise ValueError("The cost control settings must be booleans.")
        self.client_factory = client_factory
        self.options = options
        self.workflow_request = workflow_request
        self.source_root = source_root
        self.helpers_enabled = helpers_enabled
        self.cache_enabled = cache_enabled
        self.contract_factory = contract_factory
        self.hook_matcher_factory = hook_matcher_factory
        self.proxy_factory = proxy_factory
        self.helper_status = "disabled"
        self.registry = None
        self.cache_session = None
        self._temporary = None
        self._cache = None
        self._client = None
        self._entered_client = None

    def _prepare(self):
        if not self.helpers_enabled:
            return None
        reason = helper_support(self.options)
        if reason is not None:
            self.helper_status = reason
            return None
        if self.workflow_request is None or self.source_root is None:
            self.helper_status = "unsupported_workflow_request"
            return None
        factory = self.contract_factory
        if factory is None:
            from .source_contract import KernelWorkflowContract
            factory = KernelWorkflowContract
        matcher = self.hook_matcher_factory
        if matcher is None:
            from claude_agent_sdk import HookMatcher
            matcher = HookMatcher
        contract = factory(self.workflow_request, self.source_root)
        session_id = self.options.session_id or str(uuid.uuid4())
        self._temporary = tempfile.TemporaryDirectory(prefix="geak-native-helpers-")
        self.registry = NativeHelperRegistry(Path(self._temporary.name) / "registry", session_id,
            self.options.cwd or Path.cwd(), contract,
            native_shell={**os.environ, **(self.options.env or {})}.get("SHELL") or "unknown")
        hooks = dict(self.options.hooks or {})
        for event, callback in (("PreToolUse", self.registry.pre), ("PostToolUse", self.registry.post),
                ("PostToolUseFailure", self.registry.post)):
            hooks[event] = [*hooks.get(event, []), matcher(matcher="Workflow|Bash|StructuredOutput", hooks=[callback])]
        return replace(self.options, session_id=session_id, hooks=hooks,
            session_store=HelperMirror(self.registry, self.options.session_store), session_store_flush="eager")

    async def __aenter__(self):
        try:
            try:
                options = self._prepare()
            except (OSError, ValueError, TypeError, ImportError, Unsupported):
                self.helper_status = "unsupported_helper_configuration"
                self._clear()
                options = None
            if options is None:
                self._client = CachedSDKClient(self.client_factory, self.options, enabled=self.cache_enabled)
                self._entered_client = await self._client.__aenter__()
                self.cache_session = self._client.cache_session
                return self

            def proxy(endpoint, policy, *, enabled):
                transport = HelperTransport(HTTPTransport(endpoint), self.registry, base_path=urlsplit(endpoint).path)
                return self.proxy_factory(endpoint, policy if self.cache_enabled else _UnchangedPolicy(),
                    enabled=enabled, transport=transport)

            # Helpers still need the proxy when caching is off. Ignore optional
            # registered-catalog settings in that unchanged transport-only path.
            self._cache = native_shared_tool_cache(options, enabled=True, proxy_factory=proxy,
                policy_mode=None if self.cache_enabled else LEGACY_POLICY)
            self.cache_session = self._cache.__enter__()
            if self.cache_session.status != "active":
                # Unsupported transport setup must not retain helper hooks.
                self.helper_status = self.cache_session.status
                self._cache.__exit__(None, None, None)
                self._cache = None
                self._clear()
                self._client = CachedSDKClient(self.client_factory, self.options, enabled=self.cache_enabled)
                self._entered_client = await self._client.__aenter__()
                self.cache_session = self._client.cache_session
                return self
            self.helper_status = "active"
            self._client = self.client_factory(options=self.cache_session.options)
            self._entered_client = await self._client.__aenter__()
            return self
        except BaseException:
            if self._cache is not None:
                self._cache.__exit__(*sys.exc_info())
                self._cache = None
            self._clear()
            raise

    def _clear(self):
        if self.registry is not None:
            self.registry.close()
            self.registry = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    async def __aexit__(self, kind, value, traceback):
        try:
            return await self._client.__aexit__(kind, value, traceback)
        finally:
            try:
                if self.registry is not None:
                    self.registry.refresh_native_completions()
            finally:
                try:
                    if self._cache is not None:
                        self._cache.__exit__(kind, value, traceback)
                        self._cache = None
                finally:
                    self._clear()

    def __getattr__(self, name):
        if self._entered_client is None:
            raise AttributeError(name)
        return getattr(self._entered_client, name)

    async def receive_messages(self):
        async for message in self._entered_client.receive_messages():
            if self.registry is not None:
                self.registry.observe(message)
            yield message

    async def receive_response(self):
        async for message in self._entered_client.receive_response():
            if self.registry is not None:
                self.registry.observe(message)
            yield message
