"""Tests for web search providers."""
from __future__ import annotations

import json

from nyx.web_search import search_web


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._payload


def test_searxng_search_parses_json(monkeypatch):
    """SearXNG provider should use structured JSON instead of scraping HTML."""
    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append((req.full_url, timeout))
        return _FakeResponse({
            "results": [
                {"title": "Nyx", "url": "https://example.com/nyx", "content": "structured"},
                {"title": "Extra", "url": "https://example.com/extra", "content": "ignored"},
            ]
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    results = search_web("nyx cli", provider="searxng", max_results=1, searxng_base_url="https://search.test/")

    assert len(results) == 1
    assert results[0].title == "Nyx"
    assert results[0].url == "https://example.com/nyx"
    assert results[0].snippet == "structured"
    assert "format=json" in calls[0][0]
