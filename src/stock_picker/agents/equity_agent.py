"""Equity agent with optional Ollama-backed structured interpretation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from stock_picker.agents.common import build_insufficient_output, collect_missing_fields
from stock_picker.export_utils import save_step_summary
from stock_picker.llm_utils import coerce_string_list, is_ollama_enabled, run_ollama_json_prompt
from stock_picker.prompts import render_equity_agent_prompt
from stock_picker.state import StockPickerState


def equity_agent_node(state: StockPickerState) -> StockPickerState:
    """Build per-stock equity views from structured inputs, optionally via local Ollama."""
    equity_analysis = dict(state.get("equity_analysis", {}))
    stock_data = state.get("stock_data", {})
    run_metadata = state.get("run_metadata", {})

    for candidate in state.get("filtered_candidates", []):
        ticker = candidate.get("ticker", "UNKNOWN")
        structured_view = None
        if is_ollama_enabled(run_metadata, "equity_agent"):
            prompt = render_equity_agent_prompt(candidate, stock_data.get(ticker, {}))
            llm_output, llm_error = run_ollama_json_prompt(prompt, run_metadata, "equity_agent")
            if llm_output:
                structured_view = coerce_equity_output(llm_output, candidate)
            else:
                state["debug_notes"] = [
                    *state.get("debug_notes", []),
                    f"Equity agent fell back to rule-based logic for {ticker} because Ollama was unavailable ({llm_error}).",
                ]

        if structured_view is None:
            structured_view = build_structured_equity_view(candidate, stock_data.get(ticker, {}))

        equity_analysis[ticker] = structured_view
        candidate["equity_note"] = structured_view["summary"]

    state["equity_analysis"] = equity_analysis
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"Equity agent added structured views for {len(equity_analysis)} names.",
    ]
    export_equity_agent_step_summary(state)
    return state


def build_equity_agent_step_summary(state: StockPickerState) -> dict[str, object]:
    """Build a compact export payload for the equity-analysis stage."""
    equity_analysis = state.get("equity_analysis", {})
    completed_views = [
        view for view in equity_analysis.values() if view.get("status") == "completed"
    ]
    strongest_names = [
        ticker
        for ticker, _ in sorted(
            equity_analysis.items(),
            key=lambda item: float(item[1].get("score", 0.0)),
            reverse=True,
        )[:3]
    ]

    strength_counts = Counter(
        strength
        for view in completed_views
        for strength in view.get("strengths", [])
    )
    weakness_counts = Counter(
        weakness
        for view in completed_views
        for weakness in view.get("weaknesses", [])
    )

    return {
        "number_of_stocks_analyzed": len(equity_analysis),
        "strongest_names": strongest_names,
        "common_strengths": [item for item, _ in strength_counts.most_common(3)],
        "common_weaknesses": [item for item, _ in weakness_counts.most_common(3)],
        "note": (
            "This stage converted structured stock signals into comparable equity views that help "
            "the final portfolio step reward stronger underlying business and factor profiles."
        ),
    }


def export_equity_agent_step_summary(state: StockPickerState) -> None:
    """Save the equity-agent step summary using the shared export utility."""
    output_dir = state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries")
    save_step_summary(
        stage_name="equity_agent",
        summary_data=build_equity_agent_step_summary(state),
        step_number=4,
        output_dir=output_dir,
    )
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "equity_agent",
        ],
    }


def build_structured_equity_view(candidate: dict[str, Any], stock_record: dict[str, Any]) -> dict[str, Any]:
    """Create a concise equity output from precomputed structured signals."""
    ticker = str(candidate.get("ticker", "UNKNOWN"))
    missing_inputs = get_missing_equity_inputs(candidate)
    if missing_inputs:
        return build_insufficient_output(
            agent="equity_agent",
            summary="Insufficient data for an equity view.",
            missing_inputs=missing_inputs,
            extra_fields={
                "ticker": ticker,
                "score": 0.0,
                "overall_view": "unknown",
                "investment_thesis": "Insufficient data.",
                "strengths": [],
                "weaknesses": ["missing required structured inputs"],
            },
        )

    valuation = float(candidate.get("value_score", 0.0))
    profitability = float(candidate.get("quality_score", 0.0))
    growth = infer_growth_signal(candidate)
    momentum = float(candidate.get("momentum_score", 0.0))
    qualitative_signals = coerce_string_list(candidate.get("qualitative_signals"))

    strengths = build_strengths(valuation, profitability, growth, momentum, qualitative_signals)
    weaknesses = build_weaknesses(valuation, profitability, growth, momentum, stock_record, candidate)
    overall_view = infer_overall_view(valuation, profitability, growth, momentum)
    structured_score = round((valuation + profitability + growth + momentum) / 4.0, 4)

    return {
        "agent": "equity_agent",
        "status": "completed",
        "ticker": ticker,
        "score": structured_score,
        "overall_view": overall_view,
        "summary": "Equity view based on precomputed valuation, profitability, growth, momentum, and free qualitative signals.",
        "investment_thesis": build_investment_thesis(
            overall_view,
            valuation,
            profitability,
            growth,
            momentum,
            qualitative_signals,
            str(candidate.get("data_quality_warning", "")),
        ),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "missing_inputs": [],
    }


def coerce_equity_output(llm_output: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ollama output into the project equity schema."""
    return {
        "agent": "equity_agent",
        "status": str(llm_output.get("status", "completed")),
        "ticker": str(llm_output.get("ticker", candidate.get("ticker", "UNKNOWN"))),
        "score": float(llm_output.get("score", candidate.get("quant_score", 0.0)) or 0.0),
        "overall_view": str(llm_output.get("overall_view", "unknown")),
        "summary": str(llm_output.get("summary", "Equity view from local Ollama model.")),
        "investment_thesis": str(llm_output.get("investment_thesis", "Insufficient data.")),
        "strengths": coerce_string_list(llm_output.get("strengths")),
        "weaknesses": coerce_string_list(llm_output.get("weaknesses")),
        "missing_inputs": coerce_string_list(llm_output.get("missing_inputs")),
    }


def get_missing_equity_inputs(candidate: dict[str, Any]) -> list[str]:
    """List structured candidate fields required for the MVP equity view."""
    return collect_missing_fields(
        candidate,
        ["ticker", "value_score", "quality_score", "momentum_score"],
    )


def infer_growth_signal(candidate: dict[str, Any]) -> float:
    """Use structured proxy inputs only; fall back to a conservative blended signal."""
    explicit_growth = candidate.get("growth_score")
    if explicit_growth is not None:
        return float(explicit_growth)

    quality = float(candidate.get("quality_score", 0.0) or 0.0)
    momentum = float(candidate.get("momentum_score", 0.0) or 0.0)
    return round((quality + momentum) / 2.0, 4)


def infer_overall_view(valuation: float, profitability: float, growth: float, momentum: float) -> str:
    positive_count = sum(score >= 0.70 for score in [valuation, profitability, growth, momentum])
    weak_count = sum(score < 0.45 for score in [valuation, profitability, growth, momentum])

    if positive_count >= 3 and weak_count == 0:
        return "positive"
    if weak_count >= 2:
        return "negative"
    return "neutral"


def build_strengths(
    valuation: float,
    profitability: float,
    growth: float,
    momentum: float,
    qualitative_signals: list[str],
) -> list[str]:
    strengths: list[str] = []
    if valuation >= 0.70:
        strengths.append("attractive valuation signal")
    if profitability >= 0.70:
        strengths.append("strong profitability signal")
    if growth >= 0.70:
        strengths.append("supportive growth signal")
    if momentum >= 0.70:
        strengths.append("positive momentum signal")
    if qualitative_signals:
        strengths.append(f"qualitative support: {qualitative_signals[0]}")
    if not strengths:
        strengths.append("no clear factor-based strength")
    return strengths


def build_weaknesses(
    valuation: float,
    profitability: float,
    growth: float,
    momentum: float,
    stock_record: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    weaknesses: list[str] = []
    if valuation < 0.45:
        weaknesses.append("weak valuation signal")
    if profitability < 0.45:
        weaknesses.append("weak profitability signal")
    if growth < 0.45:
        weaknesses.append("weak growth signal")
    if momentum < 0.45:
        weaknesses.append("weak momentum signal")
    if not stock_record:
        weaknesses.append("limited stock-level structured data record")
    if not stock_record.get("summary"):
        weaknesses.append("no filing-based or transcript-based review yet")
    if str(candidate.get("feature_confidence", "")).lower() == "low":
        weaknesses.append("low feature confidence")
    if coerce_string_list(candidate.get("data_quality_flags")):
        weaknesses.append("data quality caution")
    return weaknesses


def build_investment_thesis(
    overall_view: str,
    valuation: float,
    profitability: float,
    growth: float,
    momentum: float,
    qualitative_signals: list[str],
    data_quality_warning: str,
) -> str:
    qualitative_clause = f" Qualitative note: {qualitative_signals[0]}." if qualitative_signals else ""
    quality_clause = ""
    if data_quality_warning and data_quality_warning != "No material data-quality warnings.":
        quality_clause = f" Data quality caution: {data_quality_warning}"
    if overall_view == "positive":
        return (
            "The stock combines supportive valuation, profitability, growth, and momentum signals "
            "well enough to remain a strong MVP candidate."
            + qualitative_clause
            + quality_clause
        )
    if overall_view == "negative":
        return (
            "Multiple structured signals are weak, so the stock should be treated cautiously until "
            "better supporting data is available."
            + qualitative_clause
            + quality_clause
        )
    strongest_dimension = max(
        [
            ("valuation", valuation),
            ("profitability", profitability),
            ("growth", growth),
            ("momentum", momentum),
        ],
        key=lambda item: item[1],
    )[0]
    return (
        f"The stock is acceptable for the shortlist, with {strongest_dimension} as the clearest supporting signal."
        f"{qualitative_clause}{quality_clause}"
    )
