"""
Web search capability.

Supports DuckDuckGo (built-in, no API key needed) and a pluggable
interface for other search providers.
"""
from __future__ import annotations

import logging
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_SEARXNG_URL = "https://searx.be/search"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# SearXNG (JSON API, no HTML scraping)
# ---------------------------------------------------------------------------


def _searxng_search(query: str, max_results: int = 5, base_url: str = DEFAULT_SEARXNG_URL) -> list[SearchResult]:
    """Search a SearXNG instance and parse structured JSON results."""
    params = urllib.parse.urlencode({"q": query, "format": "json", "language": "en"})
    separator = "&" if "?" in base_url else "?"
    url = f"{base_url}{separator}{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Nyx/1.0 (+https://github.com)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("SearXNG search failed: %s", e)
        return []

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.warning("SearXNG returned invalid JSON: %s", e)
        return []

    raw_results = data.get("results", []) if isinstance(data, dict) else []
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url_value = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if title and url_value:
            results.append(SearchResult(title=title, url=url_value, snippet=snippet))
        if len(results) >= max_results:
            break

    logger.debug("SearXNG returned %d results for '%s'", len(results), query)
    return results


# ---------------------------------------------------------------------------
# DuckDuckGo legacy fallback (free, no API key)
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


def fetch_page(url: str, timeout: int = 15, mode: str = "clean") -> str:
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

    if mode == "clean":
        # First remove comments
        html_cleaned = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        # Strip script and style
        html_cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html_cleaned, flags=re.DOTALL)
        html_cleaned = re.sub(r"<style[^>]*>.*?</style>", "", html_cleaned, flags=re.DOTALL)
        
        # Strip common non-article components
        html_cleaned = re.sub(r"<header[^>]*>.*?</header>", "", html_cleaned, flags=re.DOTALL)
        html_cleaned = re.sub(r"<footer[^>]*>.*?</footer>", "", html_cleaned, flags=re.DOTALL)
        html_cleaned = re.sub(r"<nav[^>]*>.*?</nav>", "", html_cleaned, flags=re.DOTALL)
        html_cleaned = re.sub(r"<aside[^>]*>.*?</aside>", "", html_cleaned, flags=re.DOTALL)
        html_cleaned = re.sub(r"<form[^>]*>.*?</form>", "", html_cleaned, flags=re.DOTALL)
        
        # Try to find main article content
        content_matches = []
        for tag in ["article", "main"]:
            content_matches.extend(re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", html_cleaned, flags=re.DOTALL))
            
        if not content_matches:
            # Look for div with class/id containing article, main-content, post-body, etc.
            div_patterns = [
                r'<div[^>]*class="[^"]*(?:article|main-content|post-body|entry-content|markdown-body)[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*id="[^"]*(?:article|main-content|post-body|entry-content|markdown-body)[^"]*"[^>]*>(.*?)</div>'
            ]
            for pat in div_patterns:
                content_matches.extend(re.findall(pat, html_cleaned, flags=re.DOTALL))
                
        if content_matches:
            html = "\n".join(content_matches)
        else:
            # Fallback to html_cleaned to at least have headers/navigation/footers stripped
            html = html_cleaned

    # Preserve block structure by inserting newlines before stripping tags
    text = re.sub(r"<(?:/p|/div|/h\d|br\s*/?|/li|/tr)>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Clean up horizontal spacing, but keep vertical spacing
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
            
    text = "\n\n".join(lines)
    return text[:10000]



# ---------------------------------------------------------------------------
# Search registry
# ---------------------------------------------------------------------------

_SEARCH_PROVIDERS: dict[str, Callable[[str, int], list[SearchResult]]] = {
    "searxng": _searxng_search,
    "duckduckgo": _ddg_search,
}


def search_web(
    query: str,
    provider: str = "searxng",
    max_results: int = 5,
    searxng_base_url: str = DEFAULT_SEARXNG_URL,
) -> list[SearchResult]:
    """Search the web using the configured provider."""
    if provider == "searxng":
        return _searxng_search(query, max_results, searxng_base_url)
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
