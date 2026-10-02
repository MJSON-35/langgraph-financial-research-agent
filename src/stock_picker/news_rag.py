"""Lightweight news attachment and summary helpers for candidate stocks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stock_picker.export_utils import PROJECT_ROOT, save_step_summary
from stock_picker.llm_utils import coerce_string_list, is_ollama_enabled, run_ollama_json_prompt
from stock_picker.news_vectorstore import (
    add_news_documents,
    build_news_document,
    deduplicate_news_items,
    get_news_vectorstore,
    is_news_recent,
    make_news_document_id,
    retrieve_recent_news,
    score_news_quality,
)
from stock_picker.research_tools import (
    TavilyNewsSearchTool,
    ToolResult,
    build_skipped_tool_audit,
    build_tool_audit,
)
from stock_picker.routing import decide_news_route
from stock_picker.state import StockPickerState
from stock_picker.tavily_news import fetch_tavily_news_records


POSITIVE_KEYWORDS = [
    "beat",
    "growth",
    "upgrade",
    "approval",
    "record",
    "strong demand",
    "raised guidance",
    "expansion",
    "earnings beat",
    "margin improvement",
]

NEGATIVE_KEYWORDS = [
    "lawsuit",
    "investigation",
    "downgrade",
    "miss",
    "guidance cut",
    "recall",
    "fraud",
    "regulatory",
    "strike",
    "weak demand",
    "margin pressure",
    "antitrust",
    "accounting issue",
]


def news_rag_node(state: StockPickerState) -> StockPickerState:
    """Attach robust news summaries to shortlisted candidates."""
    candidates = attach_news_to_candidates(
        state.get("filtered_candidates", []),
        state.get("run_metadata", {}),
    )
    state["filtered_candidates"] = candidates
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"News RAG attached qualitative news fields for {len(candidates)} candidates.",
    ]
    export_news_rag_step_summary(state)
    return state


def news_evidence_node(state: StockPickerState) -> StockPickerState:
    """Collect and score local, candidate, and vector-store news without web search."""
    state["filtered_candidates"] = attach_news_to_candidates(
        state.get("filtered_candidates", []),
        state.get("run_metadata", {}),
        allow_tavily=False,
    )
    return state


def news_evidence_check_node(state: StockPickerState) -> StockPickerState:
    """Record the deterministic news route before the conditional edge runs."""
    decision = decide_news_route(state)
    state["routing_decisions"] = {
        **state.get("routing_decisions", {}),
        "news": decision,
    }
    missing_tickers = set(decision.get("tickers", []))
    for candidate in state.get("filtered_candidates", []):
        summary = candidate.get("news_quality_summary", {})
        if not isinstance(summary, dict) or summary.get("tavily_status") != "deferred_to_router":
            continue
        ticker = str(candidate.get("ticker", "")).strip()
        if ticker not in missing_tickers:
            summary["tavily_status"] = "skipped_sufficient_news"
        elif decision["route"] == "news_finalize":
            summary["tavily_status"] = f"skipped_{decision['reason']}"
    if decision["route"] == "news_finalize":
        state["tool_audit_trail"] = [
            *state.get("tool_audit_trail", []),
            build_skipped_tool_audit("tavily_news_search", str(decision["reason"])),
        ]
    return state


def tavily_news_tool_node(state: StockPickerState) -> StockPickerState:
    """Invoke Tavily only for candidates selected by the news evidence router."""
    decision = state.get("routing_decisions", {}).get("news", decide_news_route(state))
    target_tickers = set(decision.get("tickers", []))
    candidates = state.get("filtered_candidates", [])
    target_candidates = [item for item in candidates if str(item.get("ticker", "")) in target_tickers]
    refreshed = attach_news_to_candidates(
        target_candidates,
        state.get("run_metadata", {}),
        allow_tavily=True,
    )
    refreshed_by_ticker = {str(item.get("ticker", "")): item for item in refreshed}
    state["filtered_candidates"] = [
        refreshed_by_ticker.get(str(item.get("ticker", "")), item)
        for item in candidates
    ]

    statuses = [
        str((item.get("news_quality_summary", {}) or {}).get("tavily_status", "error"))
        for item in refreshed
    ]
    error_types = sorted({status for status in statuses if status != "fetched"})
    result = ToolResult(
        tool_name="tavily_news_search",
        success=bool(statuses) and all(status == "fetched" for status in statuses),
        result=None,
        error_type=",".join(error_types) if error_types else None,
        metadata={
            "attempted_count": len(target_candidates),
            "successful_count": sum(status == "fetched" for status in statuses),
            "retrieved_count": sum(
                int((item.get("news_quality_summary", {}) or {}).get("retrieved_from_tavily", 0))
                for item in refreshed
            ),
        },
    )
    state["tool_audit_trail"] = [
        *state.get("tool_audit_trail", []),
        build_tool_audit(result, reason=str(decision.get("reason", "insufficient_local_news_coverage"))),
    ]
    return state


def news_finalize_node(state: StockPickerState) -> StockPickerState:
    """Finalize the news stage after either branch and export one coherent summary."""
    candidates = state.get("filtered_candidates", [])
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"News RAG attached qualitative news fields for {len(candidates)} candidates.",
    ]
    export_news_rag_step_summary(state)
    return state


def attach_news_to_candidates(
    candidates: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
    *,
    allow_tavily: bool = True,
) -> list[dict[str, Any]]:
    """Return candidates with headline, summary, sentiment, risk, and source fields."""
    metadata = run_metadata or {}
    local_news = load_local_news(metadata.get("news_source_path"))
    vectorstore = get_news_vectorstore(metadata)
    updated_candidates: list[dict[str, Any]] = []
    tavily_top_n = int(metadata.get("tavily_search_top_n_candidates", 20) or 20)
    retrieval_top_k = int(metadata.get("news_retrieval_top_k", 5) or 5)

    for idx, candidate in enumerate(candidates):
        updated = dict(candidate)
        ticker = str(updated.get("ticker", "")).strip()
        records = [dict(item) for item in local_news.get(ticker, [])]
        source_notes = coerce_string_list(updated.get("source_notes"))
        quality_summary = {
            "retrieved_from_vectorstore": 0,
            "retrieved_from_tavily": 0,
            "stored_new_documents": 0,
            "deduplicated_count": 0,
            "filtered_low_quality_count": 0,
            "old_news_filtered_count": 0,
            "tavily_status": "not_called",
            "tavily_error_type": None,
        }

        if records:
            source_notes.append("local_news_file")
        else:
            records = build_records_from_existing_candidate_fields(updated)
            if records:
                source_notes.append("candidate_existing_news")

        vector_records = retrieve_recent_news(vectorstore, updated, metadata)
        if vector_records:
            records.extend(vector_records)
            source_notes.append("news_vectorstore")
            quality_summary["retrieved_from_vectorstore"] = len(vector_records)

        enough_news = len(records) >= retrieval_top_k
        if not enough_news and idx < tavily_top_n and allow_tavily:
            tool_result = TavilyNewsSearchTool(search_fn=fetch_tavily_news_records).invoke(updated, metadata)
            tavily_records = tool_result.result if isinstance(tool_result.result, list) else []
            tavily_status = str(tool_result.metadata.get("provider_status", "error"))
            quality_summary["tavily_status"] = tavily_status
            quality_summary["tavily_error_type"] = tool_result.error_type
            if tavily_records:
                records.extend(tavily_records)
                source_notes.append("tavily_news")
                quality_summary["retrieved_from_tavily"] = len(tavily_records)
        elif enough_news:
            quality_summary["tavily_status"] = "skipped_sufficient_news"
        elif not allow_tavily:
            quality_summary["tavily_status"] = "deferred_to_router"
        else:
            quality_summary["tavily_status"] = "skipped_candidate_limit"

        if not records and bool(metadata.get("news_yfinance_enabled", False)):
            records = fetch_yfinance_news_records(ticker)
            if records:
                source_notes.append("yfinance_news")

        scored_records, scoring_summary = score_filter_and_store_records(records, updated, metadata, vectorstore)
        quality_summary.update(
            {
                "stored_new_documents": scoring_summary["stored_new_documents"],
                "deduplicated_count": scoring_summary["deduplicated_count"],
                "filtered_low_quality_count": scoring_summary["filtered_low_quality_count"],
                "old_news_filtered_count": scoring_summary["old_news_filtered_count"],
            }
        )
        records = scored_records

        headlines = [record["headline"] for record in records if record.get("headline")]
        summaries = [record["summary"] for record in records if record.get("summary")]
        sentiment_hint = infer_news_sentiment(headlines + summaries)
        risk_flags = infer_news_risk_flags(headlines + summaries)
        rag_summary = summarize_news(
            ticker=ticker,
            headlines=headlines,
            summaries=summaries,
            sentiment_hint=sentiment_hint,
            risk_flags=risk_flags,
            run_metadata=metadata,
        )

        if not headlines and not summaries:
            rag_summary = "news unavailable"
            sentiment_hint = "unknown"
            source_notes.append("news unavailable")

        updated["recent_news_headlines"] = headlines
        updated["recent_news_summaries"] = summaries
        updated["news_rag_summary"] = rag_summary
        updated["news_sentiment_hint"] = sentiment_hint
        updated["news_risk_flags"] = risk_flags
        updated["source_notes"] = dedupe(source_notes)
        updated["news_documents_used"] = build_news_documents_used(records)
        updated["news_quality_summary"] = quality_summary
        updated["qualitative_signals"] = dedupe(
            [
                *coerce_string_list(updated.get("qualitative_signals")),
                *(["news_positive_hint"] if sentiment_hint == "positive" else []),
                *(["news_negative_hint"] if sentiment_hint == "negative" else []),
                *[f"news_risk:{flag}" for flag in risk_flags],
            ]
        )
        updated_candidates.append(updated)

    return updated_candidates


def load_local_news(path_value: Any) -> dict[str, list[dict[str, str]]]:
    """Load local CSV or JSON news records keyed by ticker."""
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        return {}

    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return normalize_news_records(list(csv.DictReader(handle)))
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                records = []
                for ticker, items in raw.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                records.append({"ticker": ticker, **item})
                    elif isinstance(items, dict):
                        records.append({"ticker": ticker, **items})
                return normalize_news_records(records)
            if isinstance(raw, list):
                return normalize_news_records(raw)
    except (OSError, json.JSONDecodeError, csv.Error):
        return {}
    return {}


def normalize_news_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        ticker = str(record.get("ticker", record.get("symbol", ""))).strip()
        if not ticker:
            continue
        headline = str(record.get("headline", record.get("title", ""))).strip()
        summary = str(record.get("summary", record.get("description", ""))).strip()
        source = str(record.get("source", record.get("source_note", ""))).strip()
        url = str(record.get("url", "")).strip()
        source_domain = str(record.get("source_domain", "")).strip()
        published_date = str(record.get("published_date", record.get("date", ""))).strip()
        grouped.setdefault(ticker, []).append(
            {
                "ticker": ticker,
                "headline": headline,
                "summary": summary,
                "source": source,
                "source_domain": source_domain,
                "url": url,
                "published_date": published_date,
            }
        )
    return grouped


def build_records_from_existing_candidate_fields(candidate: dict[str, Any]) -> list[dict[str, str]]:
    headlines = coerce_string_list(candidate.get("recent_news_headlines"))
    summaries = coerce_string_list(candidate.get("recent_news_summaries"))
    records: list[dict[str, str]] = []
    for idx, headline in enumerate(headlines):
        records.append(
            {
                "headline": headline,
                "summary": summaries[idx] if idx < len(summaries) else "",
                "source": "candidate_existing_news",
                "ticker": str(candidate.get("ticker", "")).strip(),
                "company_name": str(candidate.get("name", "")).strip(),
            }
        )
    return records


def fetch_yfinance_news_records(ticker: str) -> list[dict[str, str]]:
    """Best-effort direct Yahoo Finance news fetch. Failures return an empty list."""
    if not ticker:
        return []
    try:
        import yfinance as yf

        raw_news = getattr(yf.Ticker(ticker), "news", None) or []
    except Exception:
        return []

    records: list[dict[str, str]] = []
    for item in raw_news[:5]:
        if not isinstance(item, dict):
            continue
        content = item.get("content", item)
        if isinstance(content, dict):
            headline = str(content.get("title", item.get("title", ""))).strip()
            summary = str(content.get("summary", item.get("summary", ""))).strip()
            provider = content.get("provider", {})
            source = provider.get("displayName", "") if isinstance(provider, dict) else ""
        else:
            headline = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip()
            source = str(item.get("publisher", "")).strip()
        if headline or summary:
            published_date = content.get("pubDate") if isinstance(content, dict) else item.get("providerPublishTime")
            records.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "summary": summary,
                    "source": "yfinance",
                    "source_domain": str(source or "finance.yahoo.com"),
                    "published_date": str(published_date or ""),
                }
            )
    return records


def score_filter_and_store_records(
    records: list[dict[str, Any]],
    candidate: dict[str, Any],
    run_metadata: dict[str, Any],
    vectorstore: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score, filter, deduplicate, and optionally persist qualitative news."""
    min_quality = float(run_metadata.get("news_min_quality_score", 0.60))
    low_quality = float(run_metadata.get("news_low_quality_threshold", 0.40))
    scored: list[dict[str, Any]] = []
    old_filtered = 0
    low_filtered = 0
    for record in records:
        current = {**record, "ticker": record.get("ticker") or candidate.get("ticker")}
        if not is_news_recent(current, run_metadata):
            old_filtered += 1
            continue
        quality = score_news_quality(current, candidate, run_metadata)
        current.update(quality)
        if quality["quality_score"] < low_quality:
            low_filtered += 1
            continue
        scored.append(current)

    before_dedup = len(scored)
    if bool(run_metadata.get("news_deduplicate", True)):
        scored = deduplicate_news_items(scored)
    deduped_count = before_dedup - len(scored)

    docs = []
    for record in scored:
        if float(record.get("quality_score") or 0.0) < min_quality:
            continue
        if record.get("source") == "vectorstore":
            continue
        doc = build_news_document(record, candidate, record)
        doc.metadata["document_id"] = make_news_document_id(record, candidate)
        docs.append(doc)
    stored_count = add_news_documents(vectorstore, docs)

    usable = sorted(scored, key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
    return usable, {
        "stored_new_documents": stored_count,
        "deduplicated_count": deduped_count,
        "filtered_low_quality_count": low_filtered,
        "old_news_filtered_count": old_filtered,
    }


def build_news_documents_used(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for record in records[:5]:
        documents.append(
            {
                "headline": record.get("headline", ""),
                "source_domain": record.get("source_domain", ""),
                "published_date": record.get("published_date", ""),
                "url": record.get("url", ""),
                "quality_score": float(record.get("quality_score") or 0.0),
            }
        )
    return documents


def infer_news_sentiment(texts: list[str]) -> str:
    joined = " ".join(texts).lower()
    if not joined.strip():
        return "unknown"
    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in joined)
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in joined)
    if positive_hits > negative_hits:
        return "positive"
    if negative_hits > positive_hits:
        return "negative"
    return "neutral"


def infer_news_risk_flags(texts: list[str]) -> list[str]:
    joined = " ".join(texts).lower()
    return [keyword.replace(" ", "_") for keyword in NEGATIVE_KEYWORDS if keyword in joined]


def summarize_news(
    *,
    ticker: str,
    headlines: list[str],
    summaries: list[str],
    sentiment_hint: str,
    risk_flags: list[str],
    run_metadata: dict[str, Any],
) -> str:
    """Use optional Ollama summary, then fall back to a deterministic sentence."""
    if is_ollama_enabled(run_metadata, "news_rag") and bool(run_metadata.get("news_ollama_enabled", False)):
        prompt = (
            "Summarize the following stock news as qualitative support only. "
            "Do not alter quantitative rankings. Return JSON with key news_rag_summary.\n"
            f"Ticker: {ticker}\nHeadlines: {json.dumps(headlines[:5], ensure_ascii=True)}\n"
            f"Summaries: {json.dumps(summaries[:5], ensure_ascii=True)}\n"
            f"Sentiment hint: {sentiment_hint}; risk flags: {risk_flags}"
        )
        llm_output, _ = run_ollama_json_prompt(prompt, run_metadata, "news_rag")
        if isinstance(llm_output, dict) and str(llm_output.get("news_rag_summary", "")).strip():
            return str(llm_output["news_rag_summary"]).strip()

    if not headlines and not summaries:
        return "news unavailable"
    lead = headlines[0] if headlines else summaries[0]
    risk_clause = f" Risk flags: {', '.join(risk_flags[:3])}." if risk_flags else ""
    return f"{ticker} recent news summary: {lead} Sentiment hint: {sentiment_hint}.{risk_clause}"


def export_news_rag_step_summary(state: StockPickerState) -> None:
    candidates = state.get("filtered_candidates", [])
    tavily_status_counts: dict[str, int] = {}
    for candidate in candidates:
        status = str((candidate.get("news_quality_summary", {}) or {}).get("tavily_status", "unknown"))
        tavily_status_counts[status] = tavily_status_counts.get(status, 0) + 1
    summary = {
        "status": "completed",
        "news_covered_count": sum(
            1 for candidate in candidates if candidate.get("news_rag_summary") != "news unavailable"
        ),
        "tavily_status_counts": tavily_status_counts,
        "tool_audit": [
            item
            for item in state.get("tool_audit_trail", [])
            if item.get("tool_name") == "tavily_news_search"
        ],
        "per_stock_news": {
            str(candidate.get("ticker", "UNKNOWN")): {
                "news_rag_summary": candidate.get("news_rag_summary", "news unavailable"),
                "news_sentiment_hint": candidate.get("news_sentiment_hint", "unknown"),
                "news_risk_flags": candidate.get("news_risk_flags", []),
                "source_notes": candidate.get("source_notes", []),
                "news_documents_used": candidate.get("news_documents_used", []),
                "news_quality_summary": candidate.get("news_quality_summary", {}),
            }
            for candidate in candidates
        },
        "note": "News is used as qualitative context and does not override deterministic quant scores.",
    }
    save_step_summary(
        stage_name="news_rag",
        summary_data=summary,
        step_number=2,
        output_dir=state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries"),
    )
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "news_rag",
        ],
    }


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
