"""
Web search capability.

Supports DuckDuckGo (built-in, no API key needed) and a pluggable
interface for other search providers.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# DuckDuckGo (free, no API key)
# ---------------------------------------------------------------------------


def _clean_html(html_text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html_text).strip()


def _ddg_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo and parse results from HTML.

    Uses multiple parsing strategies for robustness against markup changes.
    """
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
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return results

    # Strategy 1: Parse result__body divs (current DDG markup)
    result_blocks = re.findall(
        r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html,
        re.DOTALL,
    )

    for block in result_blocks[:max_results]:
        link_m = re.search(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        snippet_m = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
            block,
            re.DOTALL,
        )
        if link_m:
            title = _clean_html(link_m.group(2))
            link = link_m.group(1).strip()
            snippet = _clean_html(snippet_m.group(1)) if snippet_m else ""
            results.append(SearchResult(title=title, url=link, snippet=snippet))

    # Strategy 2: Fallback — parse result__a links and result__snippet elements
    if not results:
        link_pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        all_links = link_pattern.findall(html)
        all_snippets = snippet_pattern.findall(html)
        for i, (link, raw_title) in enumerate(all_links[:max_results]):
            title = _clean_html(raw_title)
            snippet = _clean_html(all_snippets[i]) if i < len(all_snippets) else ""
            if title and link:
                results.append(SearchResult(title=title, url=link.strip(), snippet=snippet))

    # Strategy 3: Last resort — generic article/link extraction
    if not results:
        article_pattern = re.compile(
            r'<article[^>]*>(.*?)</article>',
            re.DOTALL,
        )
        for article in article_pattern.findall(html)[:max_results]:
            link_m = re.search(r'href="(https?://[^"]+)"', article)
            title_m = re.search(r'<h[23][^>]*>(.*?)</h[23]>', article)
            snippet_m = re.search(r'<p[^>]*>(.*?)</p>', article)
            if link_m:
                title = _clean_html(title_m.group(1)) if title_m else ""
                snippet = _clean_html(snippet_m.group(1)) if snippet_m else ""
                results.append(SearchResult(title=title, url=link_m.group(1), snippet=snippet))

    # Ensure full URLs
    for r in results:
        if r.url.startswith("//"):
            r.url = "https:" + r.url
        elif r.url.startswith("/"):
            r.url = "https://duckduckgo.com" + r.url

    logger.debug("DuckDuckGo returned %d results for '%s'", len(results), query)
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
        logger.warning("Failed to fetch %s: %s", url, e)
        return f"Error fetching {url}: {e}"

    # Strip HTML tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Limit to ~8000 chars
    return text[:8000]


# ---------------------------------------------------------------------------
# Search registry
# ---------------------------------------------------------------------------

_SEARCH_PROVIDERS: dict[str, Callable[[str, int], list[SearchResult]]] = {
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