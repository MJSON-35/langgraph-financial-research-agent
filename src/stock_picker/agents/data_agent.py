"""Data agent for loading candidate inputs and applying the quant prefilter."""

from __future__ import annotations

import pandas as pd

from stock_picker.export_utils import save_step_summary
from stock_picker.prefilter import build_prefilter_breakdown_map, run_quant_prefilter
from stock_picker.providers import load_candidate_frame
from stock_picker.state import StockPickerState


def build_candidate_debug_summary(ranked_df: pd.DataFrame, top_k: int = 3) -> str:
    """Compact one-line summary of the top candidates after prefiltering."""
    if ranked_df.empty:
        return "Selected candidates: none"

    top_rows = ranked_df.head(top_k).to_dict(orient="records")
    parts = [
        f"{row['ticker']}#{int(row['prefilter_rank'])}({row['quant_score']:.3f})"
        for row in top_rows
    ]
    return "Selected candidates: " + ", ".join(parts)


def build_data_agent_step_summary(
    universe_size: int,
    ranked_df: pd.DataFrame,
    validation_summary: dict[str, object] | None = None,
    data_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a small export payload for the data-agent stage."""
    top_rows = ranked_df.head(5).to_dict(orient="records")
    top_candidate_tickers = [str(row.get("ticker", "UNKNOWN")) for row in top_rows]
    quant_score_snapshot = {
        str(row.get("ticker", "UNKNOWN")): round(float(row.get("quant_score", 0.0)), 4)
        for row in top_rows
    }
    return {
        "input_universe_size": universe_size,
        "selected_candidate_count": len(ranked_df),
        "top_candidate_tickers": top_candidate_tickers,
        "quant_score_snapshot": quant_score_snapshot,
        "validation_summary": validation_summary or {},
        "data_period": (data_context or {}).get("history_period"),
        "data_composition": {
            "universe_name": (data_context or {}).get("universe_name"),
            "universe_markets": (data_context or {}).get("universe_markets", {}),
            "raw_fields": (data_context or {}).get("raw_fields", []),
            "feature_windows": (data_context or {}).get("feature_windows", {}),
            "feature_snapshot_path": (data_context or {}).get("feature_snapshot_path"),
            "price_history_path": (data_context or {}).get("price_history_path"),
            "real_data_provider_note": (data_context or {}).get("real_data_provider_note"),
        },
        "meaningful_run_check": (data_context or {}).get("meaningful_run_check", {}),
        "note": (
            "Loaded the input universe, enriched structured stock fields, and reduced the universe "
            "to a ranked shortlist with the quantitative pre-filter."
        ),
    }


def data_agent_node(state: StockPickerState) -> StockPickerState:
    """Load candidate data through providers, prefilter it, and attach stock data."""
    universe_df, provider_info = load_candidate_frame(state)
    validation_summary = provider_info.get("validation_summary", {})
    data_context = provider_info.get("data_context", {})
    top_n = int(state.get("run_metadata", {}).get("prefilter_top_n", 10))

    ranked_df, breakdown_df = run_quant_prefilter(universe_df, top_n=top_n)
    candidate_summary = build_candidate_debug_summary(ranked_df)

    enriched = []
    stock_data = dict(state.get("stock_data", {}))

    for candidate in ranked_df.to_dict(orient="records"):
        updated = dict(candidate)
        updated["data_note"] = "Loaded quantitative features plus any free qualitative fields available from the configured data provider."
        ticker = updated.get("ticker", "UNKNOWN")
        stock_data[ticker] = {
            "agent": "data_agent",
            "status": "completed",
            "summary": "Loaded price-based metrics, optional quote fundamentals, and free qualitative fields through the configured data provider.",
            "ticker": ticker,
            "price_source": provider_info["feature_provider"],
            "fundamental_source": provider_info["feature_provider"],
            "qualitative_sources": ["yfinance_quote", "yfinance_news", "local_mock_data"],
            "fields_available": [
                "price",
                "ret_1m",
                "ret_12m",
                "volatility_60d",
                "max_drawdown_1y",
                "avg_volume_20d",
                "market_cap",
                "trailing_pe",
                "price_to_book",
                "roe",
                "raw_roe",
                "cleaned_roe",
                "raw_trailing_pe",
                "cleaned_trailing_pe",
                "raw_price_to_book",
                "cleaned_price_to_book",
                "raw_ret_1m",
                "cleaned_ret_1m",
                "raw_ret_12m",
                "cleaned_ret_12m",
                "raw_avg_volume_20d",
                "cleaned_avg_volume_20d",
                "data_quality_flags",
                "data_quality_warning",
                "feature_confidence",
                "quality_score_cleaned",
                "quality_score_confidence",
                "available_price_days",
                "has_full_12m_history",
                "momentum_12m_confidence",
                "business_summary",
                "recent_news_headlines",
                "recent_news_summaries",
                "news_rag_summary",
                "news_sentiment_hint",
                "news_risk_flags",
                "qualitative_signals",
                "value_score",
                "quality_score",
                "growth_score",
                "momentum_score",
                "liquidity_score",
                "volatility_proxy",
                "drawdown_proxy",
                "sector_risk",
                "event_flags",
                "quant_score",
                "liquidity_penalty",
                "data_quality_penalty",
                "prefilter_rank",
            ],
        }
        enriched.append(updated)

    state["filtered_candidates"] = enriched
    state["universe"] = universe_df.to_dict(orient="records")
    state["prefilter_score_breakdown"] = build_prefilter_breakdown_map(breakdown_df)
    state["stock_data"] = stock_data
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"Data agent loaded universe via {provider_info['universe_provider']}.",
        f"Data agent enriched features via {provider_info['feature_provider']}.",
        (
            "Validation summary: "
            f"{validation_summary.get('row_count', len(universe_df))} rows loaded; "
            f"missing counts={validation_summary.get('missing_by_column', {})}"
        ),
        (
            "Data period and composition: "
            f"history_period={data_context.get('history_period', 'n/a')}, "
            f"markets={data_context.get('universe_markets', {})}"
        ),
        f"Quant prefilter ranked {len(universe_df)} names and kept {len(enriched)} candidates.",
        f"Data agent created placeholder stock_data records for {len(enriched)} candidates.",
        candidate_summary,
    ]
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "universe_provider": provider_info["universe_provider"],
        "feature_provider": provider_info["feature_provider"],
        "universe_count": len(universe_df),
        "prefilter_method": "percentile composite: 0.4 value + 0.3 quality + 0.3 momentum",
        "prefilter_liquidity_rule": "liquidity percentile floor with low-liquidity penalty",
        "prefilter_input_count": len(universe_df),
        "prefilter_output_count": len(ranked_df),
        "prefilter_top_n": top_n,
        "data_validation_summary": validation_summary,
        "data_context": data_context,
        "meaningful_run_check": data_context.get("meaningful_run_check", {}),
    }
    if validation_summary.get("warning"):
        state["debug_notes"] = [
            *state.get("debug_notes", []),
            f"Data warning: {validation_summary['warning']}",
        ]
    save_step_summary(
        stage_name="data_agent",
        summary_data=build_data_agent_step_summary(
            len(universe_df),
            ranked_df,
            validation_summary,
            data_context,
        ),
        step_number=1,
        output_dir=state["run_metadata"].get("output_dir", "outputs/step_summaries"),
    )
    state["run_metadata"]["exported_step_summaries"] = [
        *state["run_metadata"].get("exported_step_summaries", []),
        "data_agent",
    ]
    # TODO: Add API-backed providers behind this adapter layer once needed.
    # TODO: Add caching and retry logic inside providers, not inside agents.
    return state
