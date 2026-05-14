"""Embedding factory for GEAK RAG.

By default this returns a local HuggingFaceEmbeddings instance (the
historical GEAK behavior). When ``GEAK_EMBEDDING_BASE_URL`` is set the
factory returns an OpenAI-compatible remote embedding client instead, so
operators can offload the heavy embedding inference to a LiteLLM /
OpenAI-compatible gateway and avoid the slow first-run CPU build.

Environment variables (all optional):

* ``GEAK_EMBEDDING_BASE_URL`` -- when set, use a remote OpenAI-compatible
  ``/v1/embeddings`` endpoint. The value is the base URL only.
* ``GEAK_EMBEDDING_API_KEY`` -- API key for the remote endpoint. Falls
  back to ``OPENAI_API_KEY`` and ``SAFE_API_KEY``.
* ``GEAK_EMBEDDING_MODEL`` -- remote model name. Defaults to the same
  HuggingFace model name the caller would have used locally so a gateway
  that proxies the BGE model needs no extra configuration.

Leave ``GEAK_EMBEDDING_BASE_URL`` unset to keep the original local
HuggingFace path, which avoids surprising existing GEAK users.
"""

from __future__ import annotations

import os
from typing import Any


_BASE_URL_ENV = "GEAK_EMBEDDING_BASE_URL"
_API_KEY_ENVS = ("GEAK_EMBEDDING_API_KEY", "OPENAI_API_KEY", "SAFE_API_KEY")
_MODEL_ENV = "GEAK_EMBEDDING_MODEL"


def _strip_env(name: str) -> str:
    value = os.environ.get(name, "")
    return value.strip() if isinstance(value, str) else ""


def remote_endpoint_configured() -> bool:
    """True when the caller explicitly opted into a remote embedding endpoint."""
    return bool(_strip_env(_BASE_URL_ENV))


def _resolve_api_key() -> str:
    for name in _API_KEY_ENVS:
        value = _strip_env(name)
        if value:
            return value
    return ""


def _resolve_model(default_model: str) -> str:
    return _strip_env(_MODEL_ENV) or default_model


def make_embeddings(
    *,
    huggingface_model_name: str,
    device: str = "cpu",
    normalize: bool = True,
) -> Any:
    """Return a LangChain embedding instance.

    Args:
        huggingface_model_name: Model name to use when falling back to the
            local HuggingFace path. Also used as the default remote model
            name when ``GEAK_EMBEDDING_MODEL`` is unset.
        device: HuggingFace ``model_kwargs.device``. Ignored on the remote
            path.
        normalize: Whether to L2-normalize embeddings on the HuggingFace
            path. Remote endpoints are expected to handle normalization
            on their side.
    """
    base_url = _strip_env(_BASE_URL_ENV)
    if base_url:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                f"{_BASE_URL_ENV} is set but langchain-openai is not installed. "
                "Install it (pip install langchain-openai) or unset "
                f"{_BASE_URL_ENV} to use the local HuggingFace embedding model."
            ) from exc

        api_key = _resolve_api_key() or "not-needed"
        model = _resolve_model(huggingface_model_name)
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=huggingface_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )
