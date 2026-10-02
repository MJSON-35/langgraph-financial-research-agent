from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class StockIdea(TypedDict, total=False):
    """Single stock record shared across the prototype."""

    ticker: str
    name: str
    sector: str
    market: str
    price: float
    ret_1m: float
    ret_12m: float
    volatility_60d: float
    max_drawdown_1y: float
    avg_volume_20d: float
    market_cap: float
    trailing_pe: float
    price_to_book: float
    roe: float
    raw_roe: float
    cleaned_roe: float
    raw_trailing_pe: float
    cleaned_trailing_pe: float
    raw_price_to_book: float
    cleaned_price_to_book: float
    raw_ret_12m: float
    cleaned_ret_12m: float
    raw_ret_1m: float
    cleaned_ret_1m: float
    raw_avg_volume_20d: float
    cleaned_avg_volume_20d: float
    raw_market_cap: float
    cleaned_market_cap: float
    roe_was_winsorized: bool
    roe_is_extreme_outlier: bool
    data_quality_flags: List[str]
    data_quality_warning: str
    data_quality_penalty: float
    feature_confidence: str
    valuation_confidence: str
    quality_score_cleaned: float
    quality_score_confidence: str
    available_price_days: int
    has_full_12m_history: bool
    momentum_12m_confidence: str
    business_summary: str
    recent_news_headlines: List[str]
    recent_news_summaries: List[str]
    news_rag_summary: str
    news_sentiment_hint: str
    news_risk_flags: List[str]
    qualitative_signals: List[str]
    source_notes: List[str]
    value_score: float
    quality_score: float
    growth_score: float
    momentum_score: float
    liquidity_score: float
    volatility_proxy: float
    drawdown_proxy: float
    sector_risk: str
    event_flags: List[str]
    quant_score: float
    liquidity_penalty: float
    prefilter_rank: int
    final_score: float
    confidence: str
    data_note: str
    equity_note: str
    risk_note: str
    rationale: str


class AgentSummary(TypedDict, total=False):
    """Small reusable payload shared by all agent outputs."""

    agent: str
    status: str
    summary: str
    missing_inputs: List[str]


class DataAgentOutput(AgentSummary, total=False):
    ticker: str
    price_source: str
    fundamental_source: str
    fields_available: List[str]
    qualitative_sources: List[str]


class MacroAgentOutput(AgentSummary, total=False):
    """Standardized portfolio-level macro output."""

    regime: str
    macro_summary: str
    preferred_sectors: List[str]
    risk_sectors: List[str]
    key_macro_risks: List[str]
    confidence: float


class EquityAgentOutput(AgentSummary, total=False):
    """Standardized per-stock equity view."""

    ticker: str
    score: float
    overall_view: str
    investment_thesis: str
    strengths: List[str]
    weaknesses: List[str]


class RiskAgentOutput(AgentSummary, total=False):
    """Standardized per-stock risk view."""

    ticker: str
    risk_level: str
    max_position_size: float
    main_risks: List[str]
    risk_comment: str


class PortfolioDecision(AgentSummary, total=False):
    """Standardized final portfolio output for downstream consumption."""

    top_picks: List[StockIdea]
    rejected_stocks: List[Dict[str, Any]]
    committee_summary: str
    confidence_level: str
    ranking_method: str
    score_breakdown: Dict[str, Dict[str, Any]]
    decision_traces: List[Dict[str, Any]]
    llm_rerank_applied: bool
    next_steps: List[str]


class StockPickerState(TypedDict, total=False):
    """Shared LangGraph state for the full workflow."""

    universe: List[StockIdea]
    filtered_candidates: List[StockIdea]
    prefilter_score_breakdown: Dict[str, Dict[str, Any]]
    stock_data: Dict[str, DataAgentOutput]
    macro_view: MacroAgentOutput
    equity_analysis: Dict[str, EquityAgentOutput]
    risk_analysis: Dict[str, RiskAgentOutput]
    final_decision: PortfolioDecision
    backtest_results: Dict[str, Any]
    report: str
    debug_notes: List[str]
    run_metadata: Dict[str, Any]
    error: Optional[str]
