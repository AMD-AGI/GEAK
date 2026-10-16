"""Shared identity of an explicit host-selected model route.

The existing route file owns provider configuration. This value only carries
its immutable address and requested identity; it neither selects a model nor
authenticates provider weights or reported reasoning effort.
"""

from dataclasses import asdict, dataclass
import json
import re

from ._util import require
from .archive.refs import read_ref


# Read-only interpretation of receipts written before explicit routes existed.
# This identity is never selected for a new call.
LEGACY_UNROUTED_CLAUDE_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class ModelProfile:
    workflow_backend: str
    requested_model: str
    requested_reasoning_effort: str
    route_path: str
    route_sha256: str

    def __post_init__(self):
        require(self.workflow_backend in ("codex", "claude"), "unsupported model backend", ValueError)
        for value in (self.requested_model, self.requested_reasoning_effort):
            require(type(value) is str and 0 < len(value) <= 256 and value.strip() == value
                    and not any(ord(c) < 32 for c in value), "explicit model and reasoning effort required", ValueError)
        require(type(self.route_path) is str and self.route_path.startswith("/")
                and type(self.route_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", self.route_sha256),
                "exact model route reference required", ValueError)

    @property
    def model_route_ref(self):
        return {"path": self.route_path, "sha256": self.route_sha256}

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_route(cls, route, reference, *, workflow_backend):
        require(type(route) is dict, "model route must be an object", ValueError)
        require(type(reference) is dict and set(reference) == {"path", "sha256"},
                "exact model route reference required", ValueError)
        # The retained route schemas use different field names. Read their
        # explicit identities without adding fields to the pinned route.
        if route.get("schema") == "geak-codex-responses-route-v2":
            route_backend = route.get("backend")
            require(route_backend == "codex" and "workflow_backend" not in route,
                    "model profile backend differs from route", ValueError)
        elif route.get("schema") == "geak-native-model-route-v1":
            route_backend = route.get("workflow_backend")
            require(route_backend == "claude" and "backend" not in route,
                    "model profile backend differs from route", ValueError)
        else:
            raise ValueError("unsupported model route schema")
        require(route_backend == workflow_backend,
                "model profile backend differs from route", ValueError)
        return cls(workflow_backend, route.get("requested_model"), route.get("requested_reasoning_effort"),
                   reference["path"], reference["sha256"])

    def verify(self):
        """Recheck the pinned route bytes before dispatch, without altering them."""
        def unique(items):
            value = dict(items)
            require(len(value) == len(items), "duplicate model route field", ValueError)
            return value
        route = json.loads(read_ref(self.model_route_ref), object_pairs_hook=unique)
        require(self == self.from_route(route, self.model_route_ref, workflow_backend=self.workflow_backend),
                "model profile differs from pinned route", ValueError)
        return route
