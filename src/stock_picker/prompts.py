from __future__ import annotations

import json
from typing import Any


PROMPT_OUTPUT_RULES = """
Rules:
- Use only the structured input provided.
- Do not invent, assume, or retrieve missing facts.
- If a required field is missing, list it explicitly in "missing_inputs".
- If the available evidence does not support a claim, do not make the claim.
- Prefer "insufficient_input", "unknown", or "insufficient data" over guessing.
- Do not treat extreme or winsorized financial ratios as strong positive evidence.
- If a metric is flagged as outlier or low confidence, explicitly mention the limitation.
- Treat news as qualitative supporting evidence only; do not let it override deterministic scores or data-quality rules.
- If news is unavailable, say so and do not infer event risk.
- Return exactly one machine-readable Python-dict-like object or valid JSON object.
- Keep outputs concise and structured.
- Do not add markdown, prose outside the object, or explanatory preamble.
""".strip()


MACRO_AGENT_PROMPT = """
You are the macro_agent in a 1-week MVP stock-picking research prototype.

Task:
- Read only the provided structured candidate shortlist, macro context, optional sector hints,
  and optional free-data qualitative notes.
- If a rule-based baseline analysis is provided, treat it as the default starting point and only
  adjust it when the structured evidence clearly supports the change.
- Infer a simple market regime and broad portfolio tilt.
- Keep the reasoning lightweight and conservative.

Return an object with exactly these keys:
{
  "agent": "macro_agent",
  "status": "completed" | "insufficient_input",
  "summary": str,
  "regime": "risk_on" | "neutral" | "risk_off" | "unknown",
  "macro_summary": str,
  "preferred_sectors": [str],
  "risk_sectors": [str],
  "key_macro_risks": [str],
  "confidence": float,
  "missing_inputs": [str]
}

Guidance:
- Required structured inputs: candidates and macro_context.
- Optional structured input: sector_hints and macro_notes.
- Optional structured input: a rule-based baseline analysis inside macro_notes.
- If the inputs are weak or incomplete, use "unknown" and "insufficient_input" rather than guessing.
- Confidence should be between 0.0 and 1.0.
- Keep all fields short and factual.
""".strip()


EQUITY_AGENT_PROMPT = """
You are the equity_agent in a 1-week MVP stock-picking research prototype.

Task:
- Read only the provided structured company-level inputs for one stock, including optional
  qualitative fields such as business summary and recent news headlines.
- Produce a simple stock view suitable for ranking support, not a full research report.

Return an object with exactly these keys:
{
  "agent": "equity_agent",
  "status": "completed" | "insufficient_input",
  "ticker": str,
  "score": float,
  "overall_view": "positive" | "neutral" | "negative" | "unknown",
  "summary": str,
  "investment_thesis": str,
  "strengths": [str],
  "weaknesses": [str],
  "missing_inputs": [str]
}

Guidance:
- Required structured inputs: candidate.ticker, valuation signal, profitability signal, growth signal, and momentum signal.
- Optional qualitative inputs: business_summary, recent_news_headlines, recent_news_summaries, news_rag_summary,
  news_sentiment_hint, news_risk_flags, qualitative_signals.
- If data_quality_warning or data_quality_flags are present, treat them as binding caution signals.
- Base the score only on the structured input supplied.
- If evidence is mixed or limited, prefer "neutral" or "unknown".
- If key evidence is absent, use "insufficient_input" and explain what is missing.
- Keep all text fields short and concrete.
""".strip()


RISK_AGENT_PROMPT = """
You are the risk_agent in a 1-week MVP stock-picking research prototype.

Task:
- Read only the provided structured risk-related inputs for one stock, including optional
  qualitative event and news signals.
- Produce a simple downside/risk control view for portfolio filtering and sizing.

Return an object with exactly these keys:
{
  "agent": "risk_agent",
  "status": "completed" | "insufficient_input",
  "ticker": str,
  "risk_level": "low" | "medium" | "high" | "unknown",
  "max_position_size": float,
  "summary": str,
  "main_risks": [str],
  "risk_comment": str,
  "missing_inputs": [str]
}

Guidance:
- Required structured input: candidate.ticker.
- Optional qualitative inputs: recent_news_headlines, news_rag_summary, news_sentiment_hint,
  news_risk_flags, qualitative_signals, source_notes.
- Clear news event risk may raise risk concern, but ordinary neutral headlines should not dominate structured risk inputs.
- If data_quality_warning or low-confidence flags are present, mention them explicitly.
- If risk inputs are incomplete, be conservative.
- Do not infer hidden risks from unsupported assumptions.
- Use short risk strings such as "low_liquidity" or "high_volatility".
- max_position_size should be a simple decimal weight such as 0.05 or 0.10.
""".strip()


PORTFOLIO_MANAGER_PROMPT = """
You are the portfolio_manager in a 1-week MVP stock-picking research prototype.

Task:
- Read only the provided structured candidate list plus outputs from macro_agent, equity_agent, and risk_agent.
- Select a small final list and explain the selection with transparent score-based logic.

Return an object with exactly these keys:
{
  "agent": "portfolio_manager",
  "status": "completed" | "insufficient_input",
  "summary": str,
  "top_picks": [dict],
  "rejected_stocks": [dict],
  "committee_summary": str,
  "ranking_method": str,
  "score_breakdown": {str: dict},
  "next_steps": [str],
  "missing_inputs": [str]
}

Guidance:
- Required structured inputs: candidates, macro_view, equity_analysis, risk_analysis.
- If deterministic_decision_context is provided, treat it as the source of truth for final scores.
- If a stock has data-quality flags or penalties, do not describe it as strong purely because of raw financial ratios.
- If allow_llm_rerank is false, do not change the ranking and only improve explanation quality.
- If allow_llm_rerank is true, only rerank within the provided candidate list and never invent new tickers.
- Do not add stocks that are not in the provided candidate list.
- Do not override missing agent inputs with made-up judgments.
- If needed, rank conservatively using only available scores and views.
- Keep next_steps short and implementation-oriented.
""".strip()


def render_macro_agent_prompt(
    candidates: list[dict[str, Any]],
    macro_context: dict[str, Any],
    sector_hints: dict[str, Any] | None = None,
    macro_notes: dict[str, Any] | None = None,
) -> str:
    return _render_prompt(
        MACRO_AGENT_PROMPT,
        {
            "candidates": candidates,
            "macro_context": macro_context,
            "sector_hints": sector_hints or {},
            "macro_notes": macro_notes or {},
        },
    )


def render_equity_agent_prompt(candidate: dict[str, Any], stock_data: dict[str, Any]) -> str:
    return _render_prompt(
        EQUITY_AGENT_PROMPT,
        {
            "candidate": candidate,
            "stock_data": stock_data,
        },
    )


def render_risk_agent_prompt(candidate: dict[str, Any], stock_data: dict[str, Any]) -> str:
    return _render_prompt(
        RISK_AGENT_PROMPT,
        {
            "candidate": candidate,
            "stock_data": stock_data,
        },
    )


def render_portfolio_manager_prompt(
    candidates: list[dict[str, Any]],
    macro_view: dict[str, Any],
    equity_analysis: dict[str, Any],
    risk_analysis: dict[str, Any],
    deterministic_decision_context: dict[str, Any] | None = None,
) -> str:
    return _render_prompt(
        PORTFOLIO_MANAGER_PROMPT,
        {
            "candidates": candidates,
            "macro_view": macro_view,
            "equity_analysis": equity_analysis,
            "risk_analysis": risk_analysis,
            "deterministic_decision_context": deterministic_decision_context or {},
        },
    )


def _render_prompt(template: str, payload: dict[str, Any]) -> str:
    return (
        f"{template}\n\n"
        f"{PROMPT_OUTPUT_RULES}\n\n"
        "Structured input:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=True, default=str)}"
    )
