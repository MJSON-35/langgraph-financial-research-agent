"""Deterministic evidence checks used by LangGraph conditional edges."""

from __future__ import annotations

from typing import Any

from stock_picker.env_utils import get_env_str, load_project_env
from stock_picker.state import StockPickerState


REQUIRED_MACRO_SIGNALS = (
    "growth_signal",
    "inflation_signal",
    "policy_signal",
    "volatility_signal",
)


def usable_news_count(candidate: dict[str, Any], run_metadata: dict[str, Any]) -> int:
    """Count recent, relevant documents that survived the existing quality filter."""
    minimum_quality = float(run_metadata.get("news_min_quality_score", 0.60))
    documents = candidate.get("news_documents_used", []) or []
    return sum(
        1
        for document in documents
        if float(document.get("quality_score") or 0.0) >= minimum_quality
    )


def candidates_needing_external_news(state: StockPickerState) -> list[str]:
    """Return eligible tickers whose usable local/vector evidence is below target."""
    metadata = state.get("run_metadata", {})
    required_count = max(
        1,
        int(
            metadata.get(
                "news_external_min_usable_articles",
                metadata.get("news_retrieval_top_k", 5),
            )
            or 1
        ),
    )
    candidate_limit = max(0, int(metadata.get("tavily_search_top_n_candidates", 20) or 20))
    return [
        str(candidate.get("ticker", "")).strip()
        for candidate in state.get("filtered_candidates", [])[:candidate_limit]
        if str(candidate.get("ticker", "")).strip()
        and usable_news_count(candidate, metadata) < required_count
    ]


def decide_news_route(state: StockPickerState) -> dict[str, Any]:
    """Choose whether the graph should invoke Tavily after local evidence checks."""
    missing_tickers = candidates_needing_external_news(state)
    metadata = state.get("run_metadata", {})
    if not missing_tickers:
        return {"route": "news_finalize", "reason": "existing_news_evidence_sufficient", "tickers": []}
    if not bool(metadata.get("news_tavily_enabled", False)):
        return {"route": "news_finalize", "reason": "tavily_disabled", "tickers": missing_tickers}
    return {
        "route": "tavily_news_tool",
        "reason": "insufficient_local_news_coverage",
        "tickers": missing_tickers,
    }


def route_news(state: StockPickerState) -> str:
    """LangGraph route selector after the news evidence-check node."""
    return str(state.get("routing_decisions", {}).get("news", {}).get("route", "news_finalize"))


def macro_evidence_is_sufficient(state: StockPickerState) -> bool:
    """Return whether configured structured macro evidence can support analysis."""
    metadata = state.get("run_metadata", {})
    if str(metadata.get("macro_mode", "")).lower() == "mock":
        return True
    if metadata.get("macro_context_source"):
        return True
    context = metadata.get("macro_context")
    return isinstance(context, dict) and all(context.get(key) not in (None, "") for key in REQUIRED_MACRO_SIGNALS)


def decide_macro_route(state: StockPickerState) -> dict[str, Any]:
    """Choose whether FRED is both needed and available for this run."""
    metadata = state.get("run_metadata", {})
    if macro_evidence_is_sufficient(state):
        return {"route": "macro_agent", "reason": "existing_macro_context_sufficient"}
    if str(metadata.get("macro_mode", "")).lower() != "fred":
        return {"route": "macro_agent", "reason": "fred_mode_disabled"}
    series_map = metadata.get("fred_series_map")
    if not isinstance(series_map, dict) or not series_map:
        return {"route": "macro_agent", "reason": "fred_series_map_missing"}
    load_project_env()
    if not get_env_str("FRED_API_KEY"):
        return {"route": "macro_agent", "reason": "fred_credential_unavailable"}
    return {"route": "fred_macro_tool", "reason": "insufficient_macro_evidence"}


def route_macro(state: StockPickerState) -> str:
    """LangGraph route selector after the macro evidence-check node."""
    return str(state.get("routing_decisions", {}).get("macro", {}).get("route", "macro_agent"))
