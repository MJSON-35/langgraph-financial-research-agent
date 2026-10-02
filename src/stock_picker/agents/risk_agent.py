"""Risk agent with optional Ollama-backed structured interpretation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from stock_picker.agents.common import (
    build_insufficient_output,
    collect_missing_fields,
    normalize_string_list,
)
from stock_picker.export_utils import save_step_summary
from stock_picker.llm_utils import coerce_string_list, is_ollama_enabled, run_ollama_json_prompt
from stock_picker.prompts import render_risk_agent_prompt
from stock_picker.state import StockPickerState


def risk_agent_node(state: StockPickerState) -> StockPickerState:
    """Build per-stock risk views from structured inputs, optionally via local Ollama."""
    risk_analysis = dict(state.get("risk_analysis", {}))
    stock_data = state.get("stock_data", {})
    run_metadata = state.get("run_metadata", {})

    for candidate in state.get("filtered_candidates", []):
        ticker = candidate.get("ticker", "UNKNOWN")
        structured_view = None
        if is_ollama_enabled(run_metadata, "risk_agent"):
            prompt = render_risk_agent_prompt(candidate, stock_data.get(ticker, {}))
            llm_output, llm_error = run_ollama_json_prompt(prompt, run_metadata, "risk_agent")
            if llm_output:
                structured_view = coerce_risk_output(llm_output, candidate)
            else:
                state["debug_notes"] = [
                    *state.get("debug_notes", []),
                    f"Risk agent fell back to rule-based logic for {ticker} because Ollama was unavailable ({llm_error}).",
                ]

        if structured_view is None:
            structured_view = build_structured_risk_view(candidate)

        risk_analysis[ticker] = structured_view
        candidate["risk_note"] = structured_view["risk_comment"]

    state["risk_analysis"] = risk_analysis
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"Risk agent applied structured risk checks to {len(risk_analysis)} names.",
    ]
    export_risk_agent_step_summary(state)
    return state


def build_risk_agent_step_summary(state: StockPickerState) -> dict[str, object]:
    """Build a compact export payload for the risk-analysis stage."""
    risk_analysis = state.get("risk_analysis", {})
    high_risk_names = [
        ticker
        for ticker, view in risk_analysis.items()
        if view.get("risk_level") == "high"
    ]
    theme_counts = Counter(
        risk
        for view in risk_analysis.values()
        for risk in view.get("main_risks", [])
        if risk != "no major structured risk flags"
    )
    return {
        "number_of_stocks_reviewed": len(risk_analysis),
        "high_risk_names": high_risk_names,
        "dominant_risk_themes": [item for item, _ in theme_counts.most_common(3)],
        "note": (
            "This stage reduced conviction in names with elevated proxy risks, helping the final "
            "selection favor candidates with more manageable downside flags."
        ),
    }


def export_risk_agent_step_summary(state: StockPickerState) -> None:
    """Save the risk-agent step summary using the shared export utility."""
    output_dir = state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries")
    save_step_summary(
        stage_name="risk_agent",
        summary_data=build_risk_agent_step_summary(state),
        step_number=5,
        output_dir=output_dir,
    )
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "risk_agent",
        ],
    }


def build_structured_risk_view(candidate: dict[str, Any]) -> dict[str, Any]:
    """Create an interpretable risk summary using only structured proxy fields."""
    ticker = str(candidate.get("ticker", "UNKNOWN"))
    missing_inputs = get_missing_risk_inputs(candidate)
    if missing_inputs:
        return build_insufficient_output(
            agent="risk_agent",
            summary="Insufficient data for a risk view.",
            missing_inputs=missing_inputs,
            extra_fields={
                "ticker": ticker,
                "risk_level": "unknown",
                "max_position_size": 0.0,
                "main_risks": ["missing required structured inputs"],
                "risk_comment": "Insufficient data.",
            },
        )

    volatility_proxy = float(candidate.get("volatility_proxy", 0.0) or 0.0)
    drawdown_proxy = float(candidate.get("drawdown_proxy", 0.0) or 0.0)
    sector_risk = str(candidate.get("sector_risk", "medium")).lower()
    event_flags = normalize_event_flags(candidate.get("event_flags", []))
    news_risk_flags = normalize_event_flags(candidate.get("news_risk_flags", []))
    liquidity_penalty = float(candidate.get("liquidity_penalty", 0.0) or 0.0)
    data_quality_penalty = float(candidate.get("data_quality_penalty", 0.0) or 0.0)
    news_headlines = coerce_string_list(candidate.get("recent_news_headlines"))

    risk_score = compute_risk_score(
        volatility_proxy=volatility_proxy,
        drawdown_proxy=drawdown_proxy,
        sector_risk=sector_risk,
        event_flags=[*event_flags, *news_risk_flags],
        liquidity_penalty=liquidity_penalty,
    )
    risk_level = infer_risk_level(risk_score)
    main_risks = build_main_risks(
        volatility_proxy=volatility_proxy,
        drawdown_proxy=drawdown_proxy,
        sector_risk=sector_risk,
        event_flags=[*event_flags, *news_risk_flags],
        liquidity_penalty=liquidity_penalty,
        data_quality_penalty=data_quality_penalty,
        news_headlines=news_headlines,
    )

    return {
        "agent": "risk_agent",
        "status": "completed",
        "ticker": ticker,
        "risk_level": risk_level,
        "max_position_size": infer_max_position_size(risk_level),
        "summary": "Risk view based on structured volatility, drawdown, sector, event, and free headline proxies.",
        "main_risks": main_risks,
        "risk_comment": build_risk_comment(risk_level, main_risks),
        "missing_inputs": [],
    }


def coerce_risk_output(llm_output: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ollama output into the project risk schema."""
    max_position_size = llm_output.get("max_position_size")
    return {
        "agent": "risk_agent",
        "status": str(llm_output.get("status", "completed")),
        "ticker": str(llm_output.get("ticker", candidate.get("ticker", "UNKNOWN"))),
        "risk_level": str(llm_output.get("risk_level", "unknown")),
        "max_position_size": float(max_position_size if max_position_size is not None else 0.0),
        "summary": str(llm_output.get("summary", "Risk view from local Ollama model.")),
        "main_risks": coerce_string_list(llm_output.get("main_risks")),
        "risk_comment": str(llm_output.get("risk_comment", "Insufficient data.")),
        "missing_inputs": coerce_string_list(llm_output.get("missing_inputs")),
    }


def get_missing_risk_inputs(candidate: dict[str, Any]) -> list[str]:
    """List structured candidate fields required for the MVP risk view."""
    return collect_missing_fields(candidate, ["ticker", "volatility_proxy", "drawdown_proxy"])


def normalize_event_flags(event_flags: Any) -> list[str]:
    """Normalize event flags into a compact list of labels."""
    return normalize_string_list(event_flags)


def compute_risk_score(
    volatility_proxy: float,
    drawdown_proxy: float,
    sector_risk: str,
    event_flags: list[str],
    liquidity_penalty: float,
) -> int:
    score = 0
    if volatility_proxy >= 0.55:
        score += 2
    elif volatility_proxy >= 0.40:
        score += 1

    if drawdown_proxy >= 0.45:
        score += 2
    elif drawdown_proxy >= 0.30:
        score += 1

    if sector_risk == "high":
        score += 1
    elif sector_risk == "low":
        score -= 1

    if liquidity_penalty > 0:
        score += 1

    if event_flags:
        score += 1

    return max(score, 0)


def infer_risk_level(risk_score: int) -> str:
    if risk_score >= 4:
        return "high"
    if risk_score >= 2:
        return "medium"
    return "low"


def infer_max_position_size(risk_level: str) -> float:
    if risk_level == "high":
        return 0.03
    if risk_level == "medium":
        return 0.05
    return 0.10


def build_main_risks(
    volatility_proxy: float,
    drawdown_proxy: float,
    sector_risk: str,
    event_flags: list[str],
    liquidity_penalty: float,
    data_quality_penalty: float,
    news_headlines: list[str],
) -> list[str]:
    risks: list[str] = []
    if volatility_proxy >= 0.40:
        risks.append("elevated volatility proxy")
    if drawdown_proxy >= 0.30:
        risks.append("elevated drawdown proxy")
    if sector_risk == "high":
        risks.append("high sector risk")
    if liquidity_penalty > 0:
        risks.append("below-preferred liquidity percentile")
    if data_quality_penalty > 0:
        risks.append("data_quality_caution")
    if event_flags:
        risks.extend([f"event:{flag}" for flag in event_flags[:2]])
    if any(keyword in " ".join(news_headlines).lower() for keyword in ["lawsuit", "probe", "investigation", "recall"]):
        risks.append("headline risk signal")
    if not risks:
        risks.append("no major structured risk flags")
    return risks


def build_risk_comment(risk_level: str, main_risks: list[str]) -> str:
    if risk_level == "high":
        return f"High risk profile. Main concerns: {', '.join(main_risks[:3])}."
    if risk_level == "medium":
        return f"Manageable but non-trivial risk. Watch: {', '.join(main_risks[:3])}."
    return "Low-to-moderate risk based on current structured proxy inputs."
