# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the mandatory stopping route within one native SDK lifetime."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from urllib.parse import urlsplit

from .cache_proxy import HTTPTransport, SharedToolCacheProxy
from .native_helpers import HelperMirror
from .quality_stop_controller import require
from .quality_stop_native import CheckpointTransport, NativeProducerCensus
from .sdk_cache import native_shared_tool_cache
from .sdk_helpers import _UnchangedPolicy, helper_support


class QualityStopSDKClient:
    """Reject unsupported stopping setup before opening any provider route."""

    def __init__(self, client_factory, options, *, workflow_request, source_root, controller,
                 hook_matcher_factory=None, proxy_factory=SharedToolCacheProxy):
        self.client_factory, self.options = client_factory, options
        self.workflow_request, self.source_root, self.controller = workflow_request, source_root, controller
        self.hook_matcher_factory, self.proxy_factory = hook_matcher_factory, proxy_factory
        self.registry = self._cache = self._client = self._entered = None

    async def __aenter__(self):
        require(helper_support(self.options) is None, "unsupported_stopping_sdk_options")
        declared_session = getattr(self.controller, "native_session_id", None)
        require(declared_session is None or self.options.session_id in (None, declared_session), "native_session_configuration_changed")
        session_id = declared_session or self.options.session_id or str(uuid.uuid4())
        session_options = replace(self.options, session_id=session_id)
        boundary = getattr(self.controller, "boundary", None)
        prepared = boundary.prepare_sdk_options(session_options) if boundary is not None else session_options
        require(helper_support(prepared) is None, "unsupported_stopping_sdk_options")
        require(prepared.session_id == session_id, "native_session_configuration_changed")
        factory = self.hook_matcher_factory
        if factory is None:
            from claude_agent_sdk import HookMatcher
            factory = HookMatcher
        session = session_id
        self.registry = NativeProducerCensus(session_id=session, root_request=self.workflow_request,
                                            source_root=self.source_root, workspace=prepared.cwd,
                                            controller=self.controller)
        self.controller.bind_native_closure(self.registry.confirm_return)
        hooks = dict(prepared.hooks or {})
        for event, callback in (("PreToolUse", self.registry.pre), ("PostToolUse", self.registry.post),
                                ("PostToolUseFailure", self.registry.post)):
            hooks[event] = [factory(matcher=".*", hooks=[callback], timeout=30)]
        options = replace(prepared, session_id=session, hooks=hooks,
                          session_store=HelperMirror(self.registry, prepared.session_store), session_store_flush="eager")
        def proxy(endpoint, _policy, *, enabled):
            transport = CheckpointTransport(HTTPTransport(endpoint), self.registry, base_path=urlsplit(endpoint).path)
            return self.proxy_factory(endpoint, _UnchangedPolicy(), enabled=enabled, transport=transport)
        try:
            # Reuse only the loopback lifecycle. The policy does not add cache
            # markers, change scientific bodies, or enable administrative helpers.
            self._cache = native_shared_tool_cache(options, enabled=True, proxy_factory=proxy)
            session = self._cache.__enter__()
            require(session.status == "active", "stopping_transport_unavailable")
            self._client = self.client_factory(options=session.options)
            self._entered = await self._client.__aenter__()
            if boundary is not None:
                boundary.native_started()
            return self
        except BaseException:
            failure = sys.exc_info()
            try:
                if self._entered is not None:
                    await self._client.__aexit__(*failure)
            finally:
                try:
                    if self._cache is not None:
                        self._cache.__exit__(*failure)
                finally:
                    self.registry.close()
            raise

    async def __aexit__(self, kind, value, traceback):
        try:
            return await self._client.__aexit__(kind, value, traceback)
        finally:
            try:
                self._cache.__exit__(kind, value, traceback)
            finally:
                self.registry.close()

    def __getattr__(self, name):
        if self._entered is None:
            raise AttributeError(name)
        return getattr(self._entered, name)

    async def query(self, prompt, session_id="default"):
        """Pin the single host root prompt before native inference can start."""
        require(self._entered is not None, "native_sdk_not_entered")
        self.registry.bind_root_prompt(prompt, session_id)
        return await self._entered.query(prompt, session_id=session_id)

    async def receive_messages(self):
        async for message in self._entered.receive_messages():
            self.registry.observe(message)
            yield message

    async def receive_response(self):
        async for message in self._entered.receive_response():
            self.registry.observe(message)
            yield message
