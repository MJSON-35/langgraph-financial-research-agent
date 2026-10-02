"""Tavily news-search fallback helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from stock_picker.env_utils import get_env_bool, get_env_str, load_project_env


def fetch_tavily_news_records(
    candidate: dict[str, Any],
    run_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Tavily news records for one candidate.

    Returns (records, status). Status is one of: disabled, missing_api_key,
    missing_package, error, fetched.
    """
    load_project_env()
    tavily_enabled = (
        bool(run_metadata["news_tavily_enabled"])
        if "news_tavily_enabled" in run_metadata
        else get_env_bool("NEWS_TAVILY_ENABLED", False)
    )
    if not tavily_enabled:
        return [], "disabled"
    api_key = get_env_str("TAVILY_API_KEY")
    if not api_key:
        return [], "missing_api_key"
    try:
        from tavily import TavilyClient
    except Exception:
        return [], "missing_package"

    ticker = str(candidate.get("ticker", "")).strip()
    company_name = str(candidate.get("name", candidate.get("company_name", ""))).strip()
    query = build_tavily_query(candidate, run_metadata)
    timeout_seconds = max(1.0, min(float(run_metadata.get("tavily_timeout_seconds", 20)), 120.0))
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=int(run_metadata.get("tavily_max_results", 5) or 5),
            search_depth=str(run_metadata.get("tavily_search_depth", "basic") or "basic"),
            topic=str(run_metadata.get("tavily_topic", "news") or "news"),
            days=int(run_metadata.get("tavily_days", 30) or 30),
            include_answer=bool(run_metadata.get("tavily_include_answer", False)),
            include_raw_content=bool(run_metadata.get("tavily_include_raw_content", False)),
            timeout=timeout_seconds,
        )
    except TypeError:
        try:
            response = client.search(
                query=query,
                max_results=int(run_metadata.get("tavily_max_results", 5) or 5),
                search_depth=str(run_metadata.get("tavily_search_depth", "basic") or "basic"),
                include_answer=bool(run_metadata.get("tavily_include_answer", False)),
                include_raw_content=bool(run_metadata.get("tavily_include_raw_content", False)),
                timeout=timeout_seconds,
            )
        except Exception:
            return [], "error"
    except Exception:
        return [], "error"

    records = []
    for item in response.get("results", []) if isinstance(response, dict) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", item.get("raw_content", "")) or "").strip()
        published_date = item.get("published_date") or item.get("publishedDate") or item.get("date")
        records.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "headline": title,
                "summary": content[:600],
                "content": content,
                "url": url,
                "source": "tavily",
                "source_domain": _domain_from_url(url),
                "published_date": str(published_date or "").strip() or None,
                "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "query": query,
            }
        )
    return records, "fetched"


def build_tavily_query(candidate: dict[str, Any], run_metadata: dict[str, Any]) -> str:
    ticker = str(candidate.get("ticker", "")).strip()
    company_name = str(candidate.get("name", candidate.get("company_name", ""))).strip()
    market = str(candidate.get("market", "")).upper()
    if "KOSPI" in market or ticker.endswith(".KS"):
        return f"{company_name} recent news earnings guidance demand risk"
    return f"{company_name} {ticker} latest earnings guidance demand news"


def _domain_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc
