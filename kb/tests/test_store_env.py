"""Credential resolution for the KB Store client (kb/store_client.py).

The knobs that turn remote warm start on are read from the environment in exactly one place per
language. This pins the contract that place implements:
  - the GEAK_-prefixed name wins, so a host that also runs Hyperloom (with its own KB_STORE_*)
    can point GEAK at a different store without a collision;
  - the un-prefixed name is the fallback, so every existing setup keeps working untouched;
  - surrounding whitespace is stripped, and an unset/empty value reads as absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb.store_client import (  # noqa: E402
    KBStoreClient,
    KBStoreError,
    kb_store_token,
    kb_store_url,
)

_VARS = ("GEAK_KB_STORE_URL", "KB_STORE_URL", "GEAK_KB_STORE_TOKEN", "KB_STORE_TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every case starts from none of the four names set."""
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)
    yield


def test_prefixed_name_wins(monkeypatch):
    monkeypatch.setenv("GEAK_KB_STORE_URL", "https://geak.example/kb")
    monkeypatch.setenv("KB_STORE_URL", "https://hyperloom.example/kb")
    monkeypatch.setenv("GEAK_KB_STORE_TOKEN", "geak-token")
    monkeypatch.setenv("KB_STORE_TOKEN", "hl-token")
    assert kb_store_url() == "https://geak.example/kb"
    assert kb_store_token() == "geak-token"


def test_bare_name_is_fallback(monkeypatch):
    monkeypatch.setenv("KB_STORE_URL", "https://hyperloom.example/kb")
    monkeypatch.setenv("KB_STORE_TOKEN", "hl-token")
    assert kb_store_url() == "https://hyperloom.example/kb"
    assert kb_store_token() == "hl-token"


def test_empty_prefixed_falls_back_to_bare(monkeypatch):
    # An exported-but-empty GEAK_ name must not shadow a real bare value.
    monkeypatch.setenv("GEAK_KB_STORE_URL", "")
    monkeypatch.setenv("KB_STORE_URL", "https://hyperloom.example/kb")
    monkeypatch.setenv("GEAK_KB_STORE_TOKEN", "")
    monkeypatch.setenv("KB_STORE_TOKEN", "hl-token")
    assert kb_store_url() == "https://hyperloom.example/kb"
    assert kb_store_token() == "hl-token"


def test_whitespace_stripped(monkeypatch):
    monkeypatch.setenv("GEAK_KB_STORE_URL", "  https://geak.example/kb  ")
    monkeypatch.setenv("GEAK_KB_STORE_TOKEN", "\tgeak-token\n")
    assert kb_store_url() == "https://geak.example/kb"
    assert kb_store_token() == "geak-token"


def test_none_set_reads_absent():
    assert kb_store_url() == ""
    assert kb_store_token() == ""


def test_from_env_uses_prefixed(monkeypatch):
    monkeypatch.setenv("GEAK_KB_STORE_URL", "https://geak.example/kb/")
    monkeypatch.setenv("GEAK_KB_STORE_TOKEN", "geak-token")
    client = KBStoreClient.from_env()
    # base_url is stored with the trailing slash trimmed.
    assert client._base == "https://geak.example/kb"


def test_from_env_falls_back_to_bare(monkeypatch):
    monkeypatch.setenv("KB_STORE_URL", "https://hyperloom.example/kb")
    monkeypatch.setenv("KB_STORE_TOKEN", "hl-token")
    client = KBStoreClient.from_env()
    assert client._base == "https://hyperloom.example/kb"


def test_from_env_raises_when_unset():
    with pytest.raises(KBStoreError):
        KBStoreClient.from_env()
