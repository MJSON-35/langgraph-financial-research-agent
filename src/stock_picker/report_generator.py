"""Generate presentation-friendly final reports from saved step summaries."""

from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_picker.security import sanitize_for_export


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_STEP_SUMMARY_DIR = DEFAULT_OUTPUT_DIR / "step_summaries"
DEFAULT_FINAL_REPORT_MD = DEFAULT_OUTPUT_DIR / "final_report.md"
DEFAULT_FINAL_REPORT_JSON = DEFAULT_OUTPUT_DIR / "final_report.json"
DEFAULT_FULL_PIPELINE_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "full_pipeline_summary.json"
DEFAULT_PROJECT_TITLE = "Multi-Agent LLM Stock Picker Prototype"
DEFAULT_PIPELINE = "START -> data_agent -> news_rag -> macro_agent -> equity_agent -> risk_agent -> portfolio_manager -> backtesting -> END"
DEFAULT_LIMITATIONS = [
    "The current system uses simple rule-based logic and proxy inputs.",
    "The main workflow is designed for a research demo, not production trading.",
    "Optional real-data adapters exist, but the stable demo path is mock-first and lightweight.",
    "pykrx is mainly used for Korean price, volume, and market-cap style fields; some fundamentals need local supplementation.",
    "yfinance is convenient for U.S. data but may include missing, delayed, or revised fields.",
    "News-based scores require forward testing or a point-in-time news database for rigorous historical validation.",
    "Transaction costs and slippage remain limited in the current validation layer.",
    "News quality scoring is heuristic.",
]
STAGE_SEQUENCE = [
    "data_agent",
    "news_rag",
    "macro_agent",
    "equity_agent",
    "risk_agent",
    "portfolio_manager",
    "backtesting",
]


def generate_final_reports(
    output_dir: str | Path | None = None,
    run_metadata: dict[str, Any] | None = None,
    project_title: str = DEFAULT_PROJECT_TITLE,
    pipeline: str = DEFAULT_PIPELINE,
    limitations: list[str] | None = None,
) -> dict[str, Path]:
    """Read step summaries and write final markdown and JSON reports."""
    resolved_output_dir = resolve_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    step_summaries = load_step_summaries(resolved_output_dir / "step_summaries")
    if not step_summaries:
        step_summaries = load_step_summaries(resolved_output_dir)
    payload = build_final_report_payload(
        step_summaries=step_summaries,
        run_metadata=sanitize_for_export(run_metadata or {}),
        project_title=project_title,
        pipeline=pipeline,
        limitations=limitations or DEFAULT_LIMITATIONS,
    )

    md_path = resolved_output_dir / DEFAULT_FINAL_REPORT_MD.name
    json_path = resolved_output_dir / DEFAULT_FINAL_REPORT_JSON.name
    full_pipeline_summary_path = resolved_output_dir / DEFAULT_FULL_PIPELINE_SUMMARY_JSON.name
    score_breakdown_path = resolved_output_dir / "score_breakdown.csv"
    md_path.write_text(build_final_markdown_report(payload), encoding="utf-8")
    payload = sanitize_for_export(payload)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    full_pipeline_summary_path.write_text(
        json.dumps(build_full_pipeline_summary(payload), indent=2),
        encoding="utf-8",
    )
    write_score_breakdown_csv(payload, score_breakdown_path)
    return {
        "markdown": md_path,
        "json": json_path,
        "full_pipeline_summary": full_pipeline_summary_path,
        "score_breakdown": score_breakdown_path,
    }


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve the output directory for the final report artifacts."""
    if output_dir is None:
        return DEFAULT_OUTPUT_DIR
    path = Path(output_dir)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_step_summaries(step_summary_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Read and sort all saved step summary JSON files."""
    directory = DEFAULT_STEP_SUMMARY_DIR if step_summary_dir is None else Path(step_summary_dir)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=_step_file_sort_key)
    summaries: list[dict[str, Any]] = []
    for path in files:
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            summaries.append(
                {
                    "stage_name": _infer_stage_name(path),
                    "status": "unavailable",
                    "note": f"Summary file could not be read: {path.name}",
                    "source_file": path.name,
                }
            )
    return summaries


def build_final_report_payload(
    *,
    step_summaries: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    project_title: str,
    pipeline: str,
    limitations: list[str],
) -> dict[str, Any]:
    """Build one structured payload for final markdown and JSON export."""
    summaries_by_stage = {
        summary.get("stage_name", summary.get("agent", "unknown")): summary
        for summary in step_summaries
    }
    ordered_summaries = [
        summaries_by_stage.get(stage_name, _build_unavailable_summary(stage_name))
        for stage_name in STAGE_SEQUENCE
    ]
    portfolio_summary = summaries_by_stage.get(
        "portfolio_manager",
        _build_unavailable_summary("portfolio_manager"),
    )
    missing_stages = [
        stage_name for stage_name in STAGE_SEQUENCE if stage_name not in summaries_by_stage
    ]
    reporting_warnings = [
        f"Stage summary unavailable: {stage_name}" for stage_name in missing_stages
    ]

    return sanitize_for_export({
        "project_title": project_title,
        "run_metadata": run_metadata,
        "pipeline": pipeline,
        "step_summaries": ordered_summaries,
        "data_agent_summary": summaries_by_stage.get("data_agent", _build_unavailable_summary("data_agent")),
        "macro_agent_summary": summaries_by_stage.get("macro_agent", _build_unavailable_summary("macro_agent")),
        "news_rag_summary": summaries_by_stage.get("news_rag", _build_unavailable_summary("news_rag")),
        "equity_agent_summary": summaries_by_stage.get("equity_agent", _build_unavailable_summary("equity_agent")),
        "risk_agent_summary": summaries_by_stage.get("risk_agent", _build_unavailable_summary("risk_agent")),
        "portfolio_manager_summary": portfolio_summary,
        "backtesting_summary": summaries_by_stage.get("backtesting", _build_unavailable_summary("backtesting")),
        "final_top_3_picks": portfolio_summary.get("top_3_picks", []),
        "rejected_stocks": portfolio_summary.get("rejected_stocks", []),
        "overall_committee_conclusion": portfolio_summary.get(
            "committee_summary",
            "Unavailable because the portfolio manager summary is missing.",
        ),
        "limitations": limitations,
        "reporting_warnings": reporting_warnings,
    })


def build_full_pipeline_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact ordered JSON summary for the full pipeline."""
    run_metadata = sanitize_for_export(payload.get("run_metadata", {}))
    return sanitize_for_export({
        "project_title": payload.get("project_title", DEFAULT_PROJECT_TITLE),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_name": run_metadata.get("scenario_name", run_metadata.get("name", "unknown")),
        "pipeline": payload.get("pipeline", DEFAULT_PIPELINE),
        "run_metadata": run_metadata,
        "steps": payload.get("step_summaries", []),
        "final_top_3_picks": payload.get("final_top_3_picks", []),
        "rejected_stocks": payload.get("rejected_stocks", []),
        "overall_committee_conclusion": payload.get("overall_committee_conclusion", ""),
        "backtesting_summary": payload.get("backtesting_summary", {}),
    })


def build_final_markdown_report(payload: dict[str, Any]) -> str:
    """Render a concise investment-committee style markdown memo with stage detail."""
    data_summary = payload.get("data_agent_summary", {})
    macro_summary = payload.get("macro_agent_summary", {})
    news_summary = payload.get("news_rag_summary", {})
    equity_summary = payload.get("equity_agent_summary", {})
    risk_summary = payload.get("risk_agent_summary", {})
    portfolio_summary = payload.get("portfolio_manager_summary", {})
    backtesting_summary = payload.get("backtesting_summary", {})
    run_metadata = payload.get("run_metadata", {})
    top_picks = payload.get("final_top_3_picks", [])
    rejected_stocks = payload.get("rejected_stocks", [])
    scenario_name = run_metadata.get("scenario_name", "unknown")
    shortlist_size = data_summary.get("selected_candidate_count", data_summary.get("shortlist_count", "n/a"))
    preferred_sectors = _format_or_unavailable(macro_summary.get("preferred_sectors", [])[:3])
    key_macro_risks = _format_or_unavailable(macro_summary.get("key_macro_risks", [])[:3])
    dominant_risk_themes = _format_or_unavailable(risk_summary.get("dominant_risk_themes", [])[:3])
    common_strengths = _format_or_unavailable(equity_summary.get("common_strengths", [])[:3])
    reporting_warnings = payload.get("reporting_warnings", [])
    data_validation = data_summary.get("validation_summary", {})
    data_period = data_summary.get("data_period", run_metadata.get("data_context", {}).get("history_period"))
    data_composition = data_summary.get("data_composition", run_metadata.get("data_context", {}))
    universe_markets = data_composition.get("universe_markets", {})
    feature_windows = data_composition.get("feature_windows", {})
    raw_fields = data_composition.get("raw_fields", [])
    committee_conclusion = portfolio_summary.get("committee_summary", "Unavailable.")
    committee_confidence = portfolio_summary.get("confidence_level", "unknown")
    llm_rerank_applied = portfolio_summary.get("llm_rerank_applied", False)
    portfolio_data_quality_notes = portfolio_summary.get("portfolio_data_quality_notes", [])
    meaningful_run_check = run_metadata.get("meaningful_run_check", {})
    news_quality_audit = run_metadata.get("news_quality_audit_summary", {})
    forward_ledger = run_metadata.get("forward_testing_ledger", {})

    lines = [
        f"# {payload.get('project_title', DEFAULT_PROJECT_TITLE)}",
        "",
        "## Investment Committee Memo",
        f"- Scenario: {scenario_name}",
        f"- Pipeline: {payload.get('pipeline', DEFAULT_PIPELINE)}",
        f"- Shortlist size: {shortlist_size}",
    ]

    lines.extend(
        [
            "",
            "## Committee Chair View",
            f"- Final committee call: {committee_conclusion}",
            f"- Confidence level: {committee_confidence}",
            f"- Deterministic ranking rule: {portfolio_summary.get('ranking_method', 'Unavailable.')}",
            f"- LLM rerank applied: {'Yes' if llm_rerank_applied else 'No'}",
            f"- Top 3 decision focus: {_format_or_unavailable([pick.get('ticker', 'UNKNOWN') for pick in top_picks[:3]])}",
            "",
            "## Data Quality Review",
            f"- Portfolio-level cautions: {_format_or_unavailable(portfolio_data_quality_notes)}",
            "- Interpretation rule: raw financial ratios were cleaned before scoring, and flagged or low-confidence metrics were not treated as strong positive evidence.",
            "- Why this matters: the committee can now separate genuinely strong fundamentals from missing, capped, or potentially distorted ratios.",
            "",
            "### Top 3 Data Quality Snapshot",
        ]
    )

    if top_picks:
        for pick in top_picks[:3]:
            lines.append(
                f"- {pick.get('ticker', 'UNKNOWN')}: feature_confidence={pick.get('feature_confidence', 'unknown')}, "
                f"valuation_confidence={pick.get('valuation_confidence', 'unknown')}, "
                f"quality_confidence={pick.get('quality_score_confidence', 'unknown')}, "
                f"warning={pick.get('data_quality_warning', 'Unavailable')}"
            )
    else:
        lines.append("- Unavailable.")

    lines.extend(
        [
            "",
            "## Meaningful Run Check",
            f"- Price coverage: {_fmt_pct_from_decimal(meaningful_run_check.get('price_coverage_ratio'))}",
            f"- Return coverage: {_fmt_pct_from_decimal(meaningful_run_check.get('return_coverage_ratio'))}",
            f"- Fundamental coverage: {_fmt_pct_from_decimal(meaningful_run_check.get('fundamental_coverage_ratio'))}",
            f"- News coverage: {_fmt_pct_from_decimal(meaningful_run_check.get('news_coverage_ratio'))}",
            f"- Backtest status: {meaningful_run_check.get('backtest_status', backtesting_summary.get('status', 'unknown'))}",
            f"- Meaningful run: {'Yes' if meaningful_run_check.get('is_meaningful_run') else 'No'}",
            f"- Reasons: {_format_or_unavailable(meaningful_run_check.get('reasons', []))}",
            "",
            "## Data Coverage Summary",
            f"- KRX/pykrx provider: {'enabled' if run_metadata.get('krx_enabled') else 'disabled'}",
            f"- yfinance provider: {'enabled' if run_metadata.get('yfinance_enabled') else 'disabled'}",
            f"- Feature snapshot path: {run_metadata.get('feature_snapshot_path', data_composition.get('feature_snapshot_path', 'Unavailable'))}",
            f"- Price history path: {run_metadata.get('price_history_source_path', data_composition.get('price_history_path', 'Unavailable'))}",
            f"- Real data provider note: {data_composition.get('real_data_provider_note') or run_metadata.get('real_data_provider_note', 'Unavailable')}",
            f"- Missing field counts: {_format_validation_missing(data_validation.get('missing_by_column', {}))}",
            "",
            "## News Quality Audit Summary",
            f"- Stored news rows: {news_quality_audit.get('stored_count', 0)}",
            f"- Filtered rows: {news_quality_audit.get('filtered_count', 0)}",
            f"- Duplicate rows: {news_quality_audit.get('duplicate_count', 0)}",
            f"- Old-news filtered rows: {news_quality_audit.get('old_news_filtered_count', 0)}",
            f"- Low-confidence rows: {news_quality_audit.get('low_confidence_count', 0)}",
            f"- Average quality score: {_fmt_num(news_quality_audit.get('avg_quality_score'))}",
            f"- Audit file: {news_quality_audit.get('csv_path', 'Unavailable')}",
            "",
            "## Forward Testing Ledger",
            f"- Status: {forward_ledger.get('status', 'disabled')}",
            f"- Snapshot path: {forward_ledger.get('snapshot_path', 'Unavailable')}",
            f"- Pending realized-return file: {forward_ledger.get('pending_realized_returns_file', 'Unavailable')}",
            f"- Evaluation horizons: {_format_or_unavailable([str(item) for item in forward_ledger.get('horizons', [])])}",
        ]
    )

    lines.extend(["", "## Recommendation"])
    lines.extend(
        [
            f"- Committee conclusion: {committee_conclusion}",
            f"- Macro backdrop: {macro_summary.get('macro_summary', 'Unavailable.')}",
            f"- Preferred sectors: {preferred_sectors}",
            f"- Key macro risks: {key_macro_risks}",
            "",
            "## Final Picks",
        ]
    )
    if top_picks:
        for idx, pick in enumerate(top_picks, start=1):
            lines.append(
                f"- {idx}. **{pick.get('ticker', 'UNKNOWN')}** ({pick.get('name', '')})"
                f" | score={pick.get('final_score', 0.0):.2f}"
            )
            lines.append(f"  - Rationale: {pick.get('rationale', 'No rationale available.')}")
    else:
        lines.append("- Unavailable.")

    lines.extend(
        [
            "",
            "## News RAG Summary",
            f"- Tavily status counts: {_format_mapping(news_summary.get('tavily_status_counts', {}))}",
            "| Ticker | Sentiment | Risk Flags | Summary | Source Count | Best Source Domains | Avg Quality | Recent News | Vectorstore Hits | Tavily Hits | Low Quality Filtered | Deduplicated |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    per_stock_news = news_summary.get("per_stock_news", {})
    news_rows = per_stock_news or {
        pick.get("ticker", "UNKNOWN"): {
            "news_rag_summary": pick.get("news_rag_summary", "news unavailable"),
            "news_sentiment_hint": pick.get("news_sentiment_hint", "unknown"),
            "news_risk_flags": pick.get("news_risk_flags", []),
            "source_notes": pick.get("source_notes", []),
            "news_documents_used": pick.get("news_documents_used", []),
            "news_quality_summary": pick.get("news_quality_summary", {}),
        }
        for pick in top_picks
    }
    for ticker, record in list(news_rows.items())[:10]:
        documents = record.get("news_documents_used", []) or []
        quality = record.get("news_quality_summary", {}) or {}
        lines.append(
            f"| {ticker} | {record.get('news_sentiment_hint', 'unknown')} | "
            f"{_format_or_unavailable(record.get('news_risk_flags', []))} | "
            f"{_escape_table_text(record.get('news_rag_summary', 'news unavailable'))} | "
            f"{len(documents)} | "
            f"{_format_or_unavailable(_best_source_domains(documents))} | "
            f"{_average_quality(documents):.2f} | "
            f"{_recent_news_count(documents)} | "
            f"{int(quality.get('retrieved_from_vectorstore', 0) or 0)} | "
            f"{int(quality.get('retrieved_from_tavily', 0) or 0)} | "
            f"{int(quality.get('filtered_low_quality_count', 0) or 0)} | "
            f"{int(quality.get('deduplicated_count', 0) or 0)} |"
        )
    if not news_rows:
        lines.append("| Unavailable | unknown | Unavailable | news unavailable | 0 | Unavailable | 0.00 | 0 | 0 | 0 | 0 | 0 |")

    lines.extend(["", "## News Evidence Used"])
    evidence_rows = _top_news_evidence_rows(top_picks, news_rows)
    if evidence_rows:
        lines.extend(
            [
                "| Ticker | Headline | Source | Published | Quality | URL |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for ticker, document in evidence_rows:
            lines.append(
                f"| {ticker} | {_escape_table_text(document.get('headline', ''))} | "
                f"{_escape_table_text(document.get('source_domain', ''))} | "
                f"{_escape_table_text(document.get('published_date', ''))} | "
                f"{float(document.get('quality_score') or 0.0):.2f} | "
                f"{_escape_table_text(document.get('url', ''))} |"
            )
    else:
        lines.append("- No news evidence was used.")

    lines.extend(
        [
            "",
            "## News RAG Limitations",
            "- Search results may be incomplete or inaccurate.",
            "- Company name ambiguity can still produce false matches.",
            "- Source trust is a heuristic rule, not a verified source audit.",
            "- Older news can remain in the local vectorstore, but retrieval applies the 30-day window.",
            "- News is qualitative support only and does not replace deterministic quantitative ranking.",
        ]
    )

    lines.extend(
        [
            "",
            "## Ex-post Validation",
            f"- Status: {backtesting_summary.get('status', 'unknown')}",
            f"- As-of date: {backtesting_summary.get('as_of_date') or 'Unavailable'}",
            f"- Holding horizons: {_format_or_unavailable([str(item) for item in backtesting_summary.get('horizons', [])])}",
            f"- Top 3 equal-weight return: {_format_return_map(backtesting_summary.get('portfolio_equal_weight_returns', {}))}",
            f"- Benchmark return: {_format_return_map(backtesting_summary.get('benchmark_forward_returns', {}))}",
            f"- Excess return: {_format_return_map(backtesting_summary.get('excess_returns', {}))}",
            "",
            "### Validation Limitations",
        ]
    )
    for limitation in backtesting_summary.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Key Risks",
            f"- Macro risks: {key_macro_risks}",
            f"- Dominant risk themes: {dominant_risk_themes}",
            f"- High-risk names: {_format_or_unavailable(risk_summary.get('high_risk_names', [])[:5])}",
            "",
            "## Stage-by-Stage Process",
            "### 1. Data Agent",
            "- Role: load the target universe, fetch market and fundamental fields, and run the quantitative pre-filter.",
            f"- What it consumed: {run_metadata.get('universe_name', run_metadata.get('universe_source', 'configured universe input'))}.",
            f"- Data period: {data_period or 'Unavailable'}.",
            f"- Data composition: universe={data_composition.get('universe_name', 'Unavailable')}, markets={_format_market_breakdown(universe_markets)}.",
            f"- Feature windows: {_format_feature_windows(feature_windows)}.",
            f"- What it produced: shortlist size {shortlist_size} with top candidates {_format_or_unavailable(data_summary.get('top_candidate_tickers', [])[:5])}.",
            f"- Validation: missing counts {_format_validation_missing(data_validation.get('missing_by_column', {}))}.",
            f"- Note: {data_summary.get('note', 'Unavailable.')}",
            "",
            "### 2. Macro Agent",
            "- Role: convert structured macro context into a light top-down sector overlay.",
            f"- Preferred sectors: {preferred_sectors}.",
            f"- Risk sectors: {_format_or_unavailable(macro_summary.get('risk_sectors', [])[:3])}.",
            f"- Note: {macro_summary.get('note', 'Unavailable.')}",
            "",
            "### 3. Equity Agent",
            "- Role: translate precomputed valuation, profitability, growth, and momentum signals into per-stock equity views.",
            f"- Strongest names: {_format_or_unavailable(equity_summary.get('strongest_names', [])[:5])}.",
            f"- Common strengths: {common_strengths}.",
            f"- Common weaknesses: {_format_or_unavailable(equity_summary.get('common_weaknesses', [])[:3])}.",
            f"- Note: {equity_summary.get('note', 'Unavailable.')}",
            "",
            "### 4. Risk Agent",
            "- Role: assign interpretable per-stock risk levels from structured risk proxies.",
            f"- High-risk names: {_format_or_unavailable(risk_summary.get('high_risk_names', [])[:5])}.",
            f"- Dominant risk themes: {dominant_risk_themes}.",
            f"- Note: {risk_summary.get('note', 'Unavailable.')}",
            "",
            "### 5. Portfolio Manager",
            "- Role: combine quant, macro, equity, and risk views into the final ranking.",
            f"- Decision method: {portfolio_summary.get('ranking_method', 'Transparent additive scoring over quant, equity, macro, sector, and risk inputs.')}",
            f"- Committee summary: {committee_conclusion}",
            f"- Confidence level: {committee_confidence}",
            f"- LLM rerank applied: {'Yes' if llm_rerank_applied else 'No'}",
            f"- Note: {portfolio_summary.get('note', 'Unavailable.')}",
            "",
            "## Rejected Names",
        ]
    )

    if rejected_stocks:
        for item in rejected_stocks:
            lines.append(
                f"- {item.get('ticker', 'UNKNOWN')} ({item.get('name', '')})"
                f" | score={item.get('final_score', 0.0):.2f}"
                f" | reason={item.get('reason', 'No reason available.')}"
            )
    else:
        lines.append("- Unavailable.")

    lines.extend(
        [
            "",
            "## Top 3 Data Used By Each Agent",
        ]
    )
    if top_picks:
        for idx, pick in enumerate(top_picks[:3], start=1):
            lines.extend(
                [
                    f"### {idx}. {pick.get('ticker', 'UNKNOWN')} ({pick.get('name', '')})",
                    f"- Data agent inputs: sector={pick.get('sector', 'n/a')}, market={pick.get('market', 'n/a')}, "
                    f"price={_fmt_num(pick.get('price'))}, ret_1m={_fmt_pct_from_decimal(pick.get('ret_1m'))}, "
                    f"ret_12m={_fmt_pct_from_decimal(pick.get('ret_12m'))}, volatility_60d={_fmt_num(pick.get('volatility_60d'))}, "
                    f"max_drawdown_1y={_fmt_pct_from_decimal(pick.get('max_drawdown_1y'))}, avg_volume_20d={_fmt_num(pick.get('avg_volume_20d'))}, "
                    f"trailing_pe={_fmt_num(pick.get('trailing_pe'))}, price_to_book={_fmt_num(pick.get('price_to_book'))}, roe={_fmt_pct_ratio(pick.get('roe'))}.",
                    f"- Raw vs cleaned core fields: raw_roe={_fmt_pct_ratio(pick.get('raw_roe'))}, cleaned_roe={_fmt_pct_ratio(pick.get('cleaned_roe'))}, "
                    f"raw_pe={_fmt_num(pick.get('raw_trailing_pe'))}, cleaned_pe={_fmt_num(pick.get('cleaned_trailing_pe'))}, "
                    f"raw_pb={_fmt_num(pick.get('raw_price_to_book'))}, cleaned_pb={_fmt_num(pick.get('cleaned_price_to_book'))}, "
                    f"raw_ret_12m={_fmt_pct_from_decimal(pick.get('raw_ret_12m'))}, cleaned_ret_12m={_fmt_pct_from_decimal(pick.get('cleaned_ret_12m'))}, "
                    f"raw_ret_1m={_fmt_pct_from_decimal(pick.get('raw_ret_1m'))}, cleaned_ret_1m={_fmt_pct_from_decimal(pick.get('cleaned_ret_1m'))}, "
                    f"raw_avg_volume_20d={_fmt_num(pick.get('raw_avg_volume_20d'))}, cleaned_avg_volume_20d={_fmt_num(pick.get('cleaned_avg_volume_20d'))}.",
                    f"- Data agent score view: quant_score={_fmt_num(pick.get('quant_score'))}, prefilter_rank={pick.get('prefilter_rank', 'n/a')}, "
                    f"value_score={_fmt_num(pick.get('value_score'))}, quality_score={_fmt_num(pick.get('quality_score'))}, adjusted_quality_score={_fmt_num(pick.get('quality_score_cleaned'))}, momentum_score={_fmt_num(pick.get('momentum_score'))}, liquidity_score={_fmt_num(pick.get('liquidity_score'))}.",
                    f"- Qualitative inputs used: business_summary={'Available' if pick.get('business_summary') else 'Unavailable'}, "
                    f"recent_news_headlines={len(pick.get('recent_news_headlines', []))}, qualitative_signals={_format_or_unavailable(pick.get('qualitative_signals', []))}.",
                    f"- Data quality: flags={_format_or_unavailable(pick.get('data_quality_flags', []))}, feature_confidence={pick.get('feature_confidence', 'unknown')}, "
                    f"quality_confidence={pick.get('quality_score_confidence', 'unknown')}, valuation_confidence={pick.get('valuation_confidence', 'unknown')}, "
                    f"has_full_12m_history={pick.get('has_full_12m_history', 'unknown')}, momentum_12m_confidence={pick.get('momentum_12m_confidence', 'unknown')}, "
                    f"warning={pick.get('data_quality_warning', 'Unavailable')}.",
                    f"- Equity agent interpretation: {_format_pick_note(pick.get('equity_note'))}.",
                    f"- Risk agent interpretation: risk_level={_infer_risk_level_from_note_or_penalty(pick, portfolio_summary)}, note={_format_pick_note(pick.get('risk_note'))}.",
                    f"- Macro agent impact: preferred_sector_fit={_yes_no_sector_fit(pick.get('sector'), macro_summary.get('preferred_sectors', []))}, "
                    f"risk_sector_flag={_yes_no_sector_fit(pick.get('sector'), macro_summary.get('risk_sectors', []))}, macro_summary={macro_summary.get('macro_summary', 'Unavailable')}.",
                    f"- Portfolio manager decision: final_score={_fmt_num(pick.get('final_score'))}, confidence={pick.get('confidence', 'n/a')}, "
                    f"data_quality_penalty={_fmt_num(pick.get('data_quality_penalty'))}, "
                    f"rationale={pick.get('rationale', 'Unavailable')}.",
                ]
            )
    else:
        lines.append("- Unavailable.")

    lines.extend(
        [
            "",
            "## Detailed Tables",
            "### Quant Shortlist Snapshot",
            "| Ticker | Quant Score |",
            "| --- | ---: |",
        ]
    )
    for ticker, score in data_summary.get("quant_score_snapshot", {}).items():
        lines.append(f"| {ticker} | {float(score):.4f} |")
    if not data_summary.get("quant_score_snapshot"):
        lines.append("| Unavailable | Unavailable |")

    lines.extend(
        [
            "",
            "### Final Score Breakdown",
            "| Ticker | Name | Sector | Quant | Equity Bonus | Macro Bonus | Sector Bonus | Sector Penalty | Risk Penalty | Data Quality Penalty | Final | Rank | Recommendation | Rationale |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    score_breakdown = portfolio_summary.get("final_score_breakdown", {})
    for ticker, record in score_breakdown.items():
        pick = _find_pick_or_rejected(ticker, top_picks, rejected_stocks)
        lines.append(
            f"| {ticker} | {_escape_table_text(pick.get('name', record.get('name', '')))} | "
            f"{record.get('sector', pick.get('sector', ''))} | "
            f"{float(record.get('quant_score', 0.0)):.4f} | "
            f"{float(record.get('equity_bonus', 0.0)):+.2f} | "
            f"{float(record.get('macro_bonus', 0.0)):+.2f} | "
            f"{float(record.get('sector_bonus', 0.0)):+.2f} | "
            f"{float(record.get('sector_penalty', 0.0)):.2f} | "
            f"{float(record.get('risk_penalty', 0.0)):.2f} | "
            f"{float(record.get('data_quality_penalty', 0.0)):.2f} | "
            f"{float(record.get('final_score', 0.0)):.4f} | "
            f"{int(record.get('final_rank', record.get('rank', 0)) or 0)} | "
            f"{record.get('recommendation', 'n/a')} | "
            f"{_escape_table_text(record.get('rationale', record.get('decision_summary', '')))} |"
        )
    if not score_breakdown:
        lines.append("| Unavailable |  |  |  |  |  |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "### Risk Review Snapshot",
            "| Theme | Detail |",
            "| --- | --- |",
            f"| Dominant risk themes | {dominant_risk_themes} |",
            f"| High-risk names | {_format_or_unavailable(risk_summary.get('high_risk_names', [])[:5])} |",
            f"| Macro risks | {key_macro_risks} |",
            f"| Data period | {data_period or 'Unavailable'} |",
            f"| Data composition | {_format_market_breakdown(universe_markets)} |",
        ]
    )

    lines.extend(
        [
            "",
            "## Memo Notes",
            f"- Data coverage warning: {data_validation.get('warning', 'None')}",
            f"- Raw input fields: {_format_or_unavailable(raw_fields)}",
            f"- Effective process: shortlist -> macro overlay -> equity view -> risk review -> final ranking",
            f"- Decision traces stored: {'Yes' if bool(portfolio_summary.get('decision_traces')) else 'No'}",
            "",
            "## Committee Conclusion",
            f"- {payload.get('overall_committee_conclusion', 'Unavailable.')}",
        ]
    )

    if reporting_warnings:
        lines.extend(
            [
                "",
                "## Reporting Notes",
            ]
        )
        for warning in reporting_warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Limitations",
        ]
    )
    for limitation in payload.get("limitations", DEFAULT_LIMITATIONS):
        lines.append(f"- {limitation}")

    return "\n".join(lines)


def _step_file_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    try:
        return int(prefix), path.name
    except ValueError:
        return 999, path.name


def _build_unavailable_summary(stage_name: str) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "status": "unavailable",
        "note": f"Summary unavailable for {stage_name}.",
    }


def _infer_stage_name(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def _format_or_unavailable(items: list[str]) -> str:
    if items is None:
        return "Unavailable"
    if isinstance(items, str):
        return items if items.strip() else "Unavailable"
    if not isinstance(items, list):
        try:
            if items != items:
                return "Unavailable"
        except Exception:
            pass
        return str(items) if str(items).strip() else "Unavailable"
    return ", ".join(items) if items else "Unavailable"


def _format_validation_missing(missing_by_column: dict[str, Any]) -> str:
    if not missing_by_column:
        return "Unavailable"
    ordered_pairs = list(missing_by_column.items())
    preview = [f"{column}={count}" for column, count in ordered_pairs[:8]]
    if len(ordered_pairs) > 8:
        preview.append("...")
    return ", ".join(preview)


def _format_market_breakdown(market_counts: dict[str, Any]) -> str:
    if not market_counts:
        return "Unavailable"
    return ", ".join(f"{market}={count}" for market, count in market_counts.items())


def _format_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "Unavailable"
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _format_feature_windows(feature_windows: dict[str, Any]) -> str:
    if not feature_windows:
        return "Unavailable"
    return ", ".join(f"{field}={window}" for field, window in feature_windows.items())


def _fmt_num(value: Any) -> str:
    try:
        if value is None:
            return "Unavailable"
        numeric = float(value)
        if numeric != numeric:
            return "Unavailable"
        if abs(numeric) >= 1000:
            return f"{numeric:,.2f}"
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return "Unavailable"


def _fmt_pct_from_decimal(value: Any) -> str:
    try:
        if value is None:
            return "Unavailable"
        numeric = float(value)
        if numeric != numeric:
            return "Unavailable"
        return f"{numeric * 100:.2f}%"
    except Exception:
        return "Unavailable"


def _fmt_pct_ratio(value: Any) -> str:
    try:
        if value is None:
            return "Unavailable"
        numeric = float(value)
        if numeric != numeric:
            return "Unavailable"
        if abs(numeric) <= 2:
            return f"{numeric * 100:.2f}%"
        return f"{numeric:.2f}%"
    except Exception:
        return "Unavailable"


def _format_pick_note(note: Any) -> str:
    text = str(note or "").strip()
    return text if text else "Unavailable"


def _yes_no_sector_fit(sector: Any, sectors: list[str]) -> str:
    sector_text = str(sector or "").strip().lower()
    normalized = {str(item).strip().lower() for item in sectors if str(item).strip()}
    return "Yes" if sector_text in normalized else "No"


def _infer_risk_level_from_note_or_penalty(pick: dict[str, Any], portfolio_summary: dict[str, Any]) -> str:
    traces = portfolio_summary.get("decision_traces", []) or []
    ticker = pick.get("ticker")
    for trace in traces:
        if trace.get("ticker") == ticker:
            return str(trace.get("risk_level", "n/a"))
    return "n/a"


def write_score_breakdown_csv(payload: dict[str, Any], output_path: Path) -> None:
    """Write the final score breakdown table used by the markdown report."""
    portfolio_summary = payload.get("portfolio_manager_summary", {})
    score_breakdown = portfolio_summary.get("final_score_breakdown", {})
    top_picks = payload.get("final_top_3_picks", [])
    rejected_stocks = payload.get("rejected_stocks", [])
    columns = [
        "ticker",
        "name",
        "sector",
        "quant_score",
        "equity_bonus",
        "macro_bonus",
        "sector_bonus",
        "sector_penalty",
        "risk_penalty",
        "data_quality_penalty",
        "final_score",
        "final_rank",
        "recommendation",
        "rationale",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for ticker, record in score_breakdown.items():
            pick = _find_pick_or_rejected(ticker, top_picks, rejected_stocks)
            writer.writerow(
                {
                    "ticker": ticker,
                    "name": record.get("name", pick.get("name", "")),
                    "sector": record.get("sector", pick.get("sector", "")),
                    "quant_score": record.get("quant_score", ""),
                    "equity_bonus": record.get("equity_bonus", ""),
                    "macro_bonus": record.get("macro_bonus", ""),
                    "sector_bonus": record.get("sector_bonus", ""),
                    "sector_penalty": record.get("sector_penalty", ""),
                    "risk_penalty": record.get("risk_penalty", ""),
                    "data_quality_penalty": record.get("data_quality_penalty", ""),
                    "final_score": record.get("final_score", ""),
                    "final_rank": record.get("final_rank", record.get("rank", "")),
                    "recommendation": record.get("recommendation", ""),
                    "rationale": record.get("rationale", record.get("decision_summary", "")),
                }
            )


def _find_pick_or_rejected(ticker: str, top_picks: list[dict[str, Any]], rejected_stocks: list[dict[str, Any]]) -> dict[str, Any]:
    for item in [*top_picks, *rejected_stocks]:
        if item.get("ticker") == ticker:
            return item
    return {}


def _escape_table_text(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _format_return_map(values: dict[str, Any]) -> str:
    if not values:
        return "Unavailable"
    rendered = []
    for key, value in values.items():
        if not str(key).endswith("_return") and not str(key).endswith("d"):
            continue
        rendered.append(f"{key}={_fmt_pct_from_decimal(value)}")
    return ", ".join(rendered) if rendered else "Unavailable"


def _best_source_domains(documents: list[dict[str, Any]]) -> list[str]:
    domains = []
    for document in documents:
        domain = str(document.get("source_domain", "")).strip()
        if domain and domain not in domains:
            domains.append(domain)
    return domains[:3]


def _average_quality(documents: list[dict[str, Any]]) -> float:
    scores = []
    for document in documents:
        try:
            scores.append(float(document.get("quality_score") or 0.0))
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else 0.0


def _recent_news_count(documents: list[dict[str, Any]]) -> int:
    return sum(1 for document in documents if document.get("published_date"))


def _top_news_evidence_rows(
    top_picks: list[dict[str, Any]],
    news_rows: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    tickers = [str(pick.get("ticker", "UNKNOWN")) for pick in top_picks[:3]]
    if not tickers:
        tickers = list(news_rows.keys())[:3]
    rows: list[tuple[str, dict[str, Any]]] = []
    for ticker in tickers:
        record = news_rows.get(ticker, {})
        documents = record.get("news_documents_used", []) or []
        for document in documents[:3]:
            rows.append((ticker, document))
    return rows
