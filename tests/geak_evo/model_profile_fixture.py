"""Explicit synthetic host profile; never launches a model or changes user config."""
import json
from geak_evo.archive.refs import file_ref
from geak_evo.model_profile import ModelProfile


def codex_route():
    return {"schema": "geak-codex-responses-route-v2", "backend": "codex", "wire_api": "responses",
            "requested_model": "gpt-6-astra", "requested_reasoning_effort": "xhigh",
            "base_url": "https://example.invalid/v1", "api_key_env": "SYNTHETIC_API_KEY",
            "header_env": {"X-Synthetic": "SYNTHETIC_HEADER"}, "reported_deployment_model": "gpt-6-astra",
            "qualification_ref": {"path": "/synthetic/qualification.json", "sha256": "0" * 64},
            "retry_policy": "codex_defaults"}


def profile(root, *, route=None, backend="codex"):
    route = codex_route() if route is None else route
    path = root / (backend + "-model-route.json")
    path.write_text(json.dumps(route, sort_keys=True))
    return ModelProfile.from_route(route, file_ref(path), workflow_backend=backend)
