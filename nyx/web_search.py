"""
Web search capability.

Supports DuckDuckGo (built-in, no API key needed) and a pluggable
interface for other search providers.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# DuckDuckGo (free, no API key)
# ---------------------------------------------------------------------------


def _ddg_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo and parse results from HTML."""
    url = "https://html.duckduckgo.com/html/"
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )

    results: list[SearchResult] = []
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return results

    import re
    # Parse DuckDuckGo HTML results
    # Find all result blocks: look for <a class="result__a" ...> links
    # and their sibling result__snippet elements
    link_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    # Broader: find all result__body divs
    result_blocks = re.findall(
        r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html,
        re.DOTALL,
    )

    for block in result_blocks[:max_results]:
        # Extract link
        link_m = re.search(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        # Extract snippet
        snippet_m = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
            block,
            re.DOTALL,
        )
        if link_m:
            title = re.sub(r"<[^>]+>", "", link_m.group(2)).strip()
            link = link_m.group(1).strip()
            snippet = ""
            if snippet_m:
                snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip()
            results.append(SearchResult(title=title, url=link, snippet=snippet))

    # Fallback: simpler parsing if result_blocks didn't match
    if not results:
        all_links = link_pattern.findall(html)
        all_snippets = snippet_pattern.findall(html)
        for i, (link, raw_title) in enumerate(all_links[:max_results]):
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = ""
            if i < len(all_snippets):
                snippet = re.sub(r"<[^>]+>", "", all_snippets[i]).strip()
            if title and link:
                results.append(SearchResult(title=title, url=link.strip(), snippet=snippet))

    # Ensure full URLs
    for r in results:
        if r.url.startswith("//"):
            r.url = "https:" + r.url
        elif r.url.startswith("/"):
            r.url = "https://duckduckgo.com" + r.url

    return results


# ---------------------------------------------------------------------------
# Generic web page fetcher
# ---------------------------------------------------------------------------


def fetch_page(url: str, timeout: int = 15) -> str:
    """Fetch and extract text content from a URL."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # Strip HTML tags
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Limit to ~8000 chars
    return text[:8000]


# ---------------------------------------------------------------------------
# Search registry
# ---------------------------------------------------------------------------

_SEARCH_PROVIDERS: dict[str, Any] = {
    "duckduckgo": _ddg_search,
}


def search_web(query: str, provider: str = "duckduckgo", max_results: int = 5) -> list[SearchResult]:
    """Search the web using the configured provider."""
    func = _SEARCH_PROVIDERS.get(provider)
    if not func:
        available = ", ".join(_SEARCH_PROVIDERS)
        raise ValueError(f"Unknown search provider '{provider}'. Available: {available}")
    return func(query, max_results)


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results as a readable string for LLM context."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
    return "\n\n".join(parts) if parts else "No results found."