"""GEAK_KB_READ_ONLY (kb/store_client.py): reads reach the store, writes never leave the box.

The service has no delete, so a trial run against a shared store must be able to recall without
leaving anything behind. Every write is a POST through ``_request``; the guard sits there.
"""

import io
import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb.store_client import KBStoreClient, KBStoreError, kb_read_only  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.get_method())
        return _Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GEAK_KB_READ_ONLY", raising=False)
    return seen


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), (" YES ", True), ("on", True),
    ("", False), ("0", False), ("false", False), ("no", False),
])
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("GEAK_KB_READ_ONLY", value)
    assert kb_read_only() is expected


def test_read_only_refuses_writes_before_any_request(monkeypatch, calls):
    monkeypatch.setenv("GEAK_KB_READ_ONLY", "1")
    client = KBStoreClient("https://kb.example", "token")
    with pytest.raises(KBStoreError, match="GEAK_KB_READ_ONLY"):
        client.put_knowledge("geak:e2e:x", {"k": "v"})
    with pytest.raises(KBStoreError, match="GEAK_KB_READ_ONLY"):
        client.set_champion("geak:e2e:x", "s1", value=1.0)
    assert calls == []


def test_read_only_still_reads(monkeypatch, calls):
    monkeypatch.setenv("GEAK_KB_READ_ONLY", "1")
    client = KBStoreClient("https://kb.example", "token")
    assert client._request("GET", "/v1/kb/geak:e2e:x") == {"ok": True}
    assert calls == ["GET"]


def test_writes_go_through_when_unset(calls):
    client = KBStoreClient("https://kb.example", "token")
    client.put_knowledge("geak:e2e:x", {"k": "v"})
    assert calls == ["POST"]
