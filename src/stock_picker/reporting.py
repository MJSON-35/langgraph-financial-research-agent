from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List

from stock_picker.export_utils import save_step_summary
from stock_picker.report_generator import generate_final_reports
from stock_picker.state import StockIdea, StockPickerState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_STEP_SUMMARY_DIR = DEFAULT_OUTPUT_DIR / "step_summaries"
DEFAULT_FINAL_REPORT_MD = DEFAULT_OUTPUT_DIR / "final_report.md"
DEFAULT_FINAL_REPORT_JSON = DEFAULT_OUTPUT_DIR / "final_report.json"
STEP_ORDER = [
    "data_agent",
    "news_rag",
    "macro_agent",
    "equity_agent",
    "risk_agent",
    "portfolio_manager",
    "backtesting",
]


def build_short_report(state: StockPickerState) -> str:
    """Create a compact human-readable summary of the final picks."""
    final_decision = state.get("final_decision", {})
    picks: List[StockIdea] = final_decision.get("top_picks", [])
    macro_view = state.get("macro_view", {})
    score_components = final_decision.get("score_breakdown", {})

    if not picks:
        return "No picks were generated."

    lines = [
        "Multi-agent stock picker prototype report",
        f"Macro view: {macro_view.get('summary', 'No macro summary available.')}",
        f"Decision method: {final_decision.get('ranking_method', 'No ranking method available.')}",
        f"Selection summary: {final_decision.get('summary', 'No summary available.')}",
        "",
        "Top 3 picks:",
    ]

    for idx, pick in enumerate(picks, start=1):
        ticker = pick.get("ticker", "UNKNOWN")
        component = score_components.get(ticker, {})
        lines.append(
            f"{idx}. {pick.get('ticker')} ({pick.get('name')}): "
            f"score={pick.get('final_score', 0.0):.2f} | "
            f"recommendation={component.get('recommendation', 'n/a')} | "
            f"rationale={pick.get('rationale', 'Placeholder rationale')}"
        )

    # TODO: Expand this into a richer report with factor exposures and risk flags.
    # TODO: Save reports to disk so prototype runs can be compared over time.
    return "\n".join(lines)


def finalize_report(state: StockPickerState) -> StockPickerState:
    state["report"] = build_short_report(state)
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        "Reporting step generated the short prototype report.",
    ]
    return state


def prepare_reporting_outputs(state: StockPickerState) -> None:
    """Create fresh output folders for step summaries and final reports."""
    step_dir = resolve_step_summary_dir(state)
    step_dir.mkdir(parents=True, exist_ok=True)
    for path in step_dir.glob("*.json"):
        try:
            path.unlink()
        except PermissionError:
            # Keep demo reruns resilient on Windows if a JSON file is still open elsewhere.
            continue

    resolve_output_dir(state).mkdir(parents=True, exist_ok=True)


def instrument_node(agent_name: str, node_fn: Callable[[StockPickerState], StockPickerState]) -> Callable[[StockPickerState], StockPickerState]:
    """Wrap an existing node so it writes a lightweight step summary after running."""

    def wrapped(state: StockPickerState) -> StockPickerState:
        updated_state = node_fn(state)
        exported_steps = updated_state.get("run_metadata", {}).get("exported_step_summaries", [])
        if agent_name in exported_steps:
            return updated_state
        write_step_summary(agent_name, updated_state)
        return updated_state

    return wrapped


def write_step_summary(agent_name: str, state: StockPickerState) -> None:
    """Persist a small structured snapshot for one pipeline step."""
    step_payload = build_step_summary(agent_name, state)
    step_index = STEP_ORDER.index(agent_name) + 1 if agent_name in STEP_ORDER else 99
    save_step_summary(
        stage_name=agent_name,
        summary_data=step_payload,
        step_number=step_index,
        output_dir=resolve_step_summary_dir(state),
    )


def build_step_summary(agent_name: str, state: StockPickerState) -> dict[str, Any]:
    """Convert the current shared state into a compact step-level summary."""
    builders: dict[str, Callable[[StockPickerState], dict[str, Any]]] = {
        "data_agent": summarize_data_step,
        "news_rag": summarize_news_rag_step,
        "macro_agent": summarize_macro_step,
        "equity_agent": summarize_equity_step,
        "risk_agent": summarize_risk_step,
        "portfolio_manager": summarize_portfolio_step,
        "backtesting": summarize_backtesting_step,
    }
    builder = builders.get(agent_name, lambda current_state: {"agent": agent_name, "status": "unknown"})
    summary = builder(state)
    summary["agent"] = agent_name
    return summary


def summarize_data_step(state: StockPickerState) -> dict[str, Any]:
    candidates = state.get("filtered_candidates", [])
    return {
        "status": "completed",
        "universe_count": len(state.get("universe", [])),
        "shortlist_count": len(candidates),
        "top_candidates": [
            {
                "ticker": item.get("ticker", "UNKNOWN"),
                "name": item.get("name", ""),
                "sector": item.get("sector", ""),
                "quant_score": item.get("quant_score", 0.0),
                "prefilter_rank": item.get("prefilter_rank", "n/a"),
            }
            for item in candidates[:5]
        ],
    }


def summarize_macro_step(state: StockPickerState) -> dict[str, Any]:
    macro_view = state.get("macro_view", {})
    return {
        "status": macro_view.get("status", "unknown"),
        "macro_summary": macro_view.get("macro_summary", ""),
        "preferred_sectors": macro_view.get("preferred_sectors", []),
        "risk_sectors": macro_view.get("risk_sectors", []),
        "key_macro_risks": macro_view.get("key_macro_risks", []),
        "tool_audit": [
            item for item in state.get("tool_audit_trail", []) if item.get("tool_name") == "fred_macro"
        ],
    }


def summarize_news_rag_step(state: StockPickerState) -> dict[str, Any]:
    candidates = state.get("filtered_candidates", [])
    return {
        "status": "completed",
        "news_covered_count": sum(
            1 for item in candidates if item.get("news_rag_summary") != "news unavailable"
        ),
        "tool_audit": [
            item
            for item in state.get("tool_audit_trail", [])
            if item.get("tool_name") == "tavily_news_search"
        ],
        "per_stock_news": {
            str(item.get("ticker", "UNKNOWN")): {
                "news_rag_summary": item.get("news_rag_summary", "news unavailable"),
                "news_sentiment_hint": item.get("news_sentiment_hint", "unknown"),
                "news_risk_flags": item.get("news_risk_flags", []),
                "source_notes": item.get("source_notes", []),
                "news_documents_used": item.get("news_documents_used", []),
                "news_quality_summary": item.get("news_quality_summary", {}),
            }
            for item in candidates
        },
    }


def summarize_equity_step(state: StockPickerState) -> dict[str, Any]:
    equity_analysis = state.get("equity_analysis", {})
    return {
        "status": "completed" if equity_analysis else "empty",
        "per_stock": {
            ticker: {
                "overall_view": view.get("overall_view", "n/a"),
                "strengths": view.get("strengths", [])[:2],
                "weaknesses": view.get("weaknesses", [])[:2],
                "investment_thesis": view.get("investment_thesis", ""),
            }
            for ticker, view in equity_analysis.items()
        },
    }


def summarize_risk_step(state: StockPickerState) -> dict[str, Any]:
    risk_analysis = state.get("risk_analysis", {})
    return {
        "status": "completed" if risk_analysis else "empty",
        "per_stock": {
            ticker: {
                "risk_level": view.get("risk_level", "n/a"),
                "main_risks": view.get("main_risks", [])[:3],
                "risk_comment": view.get("risk_comment", ""),
            }
            for ticker, view in risk_analysis.items()
        },
    }


def summarize_portfolio_step(state: StockPickerState) -> dict[str, Any]:
    final_decision = state.get("final_decision", {})
    return {
        "status": final_decision.get("status", "unknown"),
        "committee_summary": final_decision.get("committee_summary", ""),
        "ranking_method": final_decision.get("ranking_method", ""),
        "top_picks": final_decision.get("top_picks", []),
        "rejected_stocks": final_decision.get("rejected_stocks", []),
    }


def summarize_backtesting_step(state: StockPickerState) -> dict[str, Any]:
    result = state.get("backtest_results", {})
    return {
        "status": result.get("status", "unknown"),
        "as_of_date": result.get("as_of_date", ""),
        "horizons": result.get("horizons", []),
        "portfolio_equal_weight_returns": result.get("portfolio_equal_weight_returns", {}),
        "benchmark_forward_returns": result.get("benchmark_forward_returns", {}),
        "excess_returns": result.get("excess_returns", {}),
    }


def build_final_report_payload(state: StockPickerState) -> dict[str, Any]:
    """Combine saved step summaries and final state into one structured report."""
    step_summaries = []
    step_dir = resolve_step_summary_dir(state)
    for step_file in sorted(step_dir.glob("*.json")):
        step_summaries.append(json.loads(step_file.read_text(encoding="utf-8")))

    return {
        "project": "multi-agent stock picker prototype",
        "pipeline": "START -> data_agent -> macro_agent -> equity_agent -> risk_agent -> portfolio_manager -> END",
        "step_summaries": step_summaries,
        "final_decision": state.get("final_decision", {}),
        "report": state.get("report", build_short_report(state)),
    }


def build_final_markdown_report(payload: dict[str, Any]) -> str:
    """Render the combined step summaries into a concise markdown report."""
    final_decision = payload.get("final_decision", {})
    lines = [
        "# Final Workflow Report",
        "",
        f"Pipeline: `{payload.get('pipeline', '')}`",
        "",
        "## Step Summaries",
    ]

    for step in payload.get("step_summaries", []):
        lines.append(f"### {step.get('agent', 'unknown')}")
        lines.append(f"- Status: {step.get('status', 'unknown')}")
        if step.get("macro_summary"):
            lines.append(f"- Summary: {step['macro_summary']}")
        if step.get("committee_summary"):
            lines.append(f"- Committee summary: {step['committee_summary']}")
        if step.get("top_candidates"):
            top_names = ", ".join(item.get("ticker", "UNKNOWN") for item in step["top_candidates"][:3]) or "None"
            lines.append(f"- Top shortlist: {top_names}")
        if step.get("per_stock"):
            stock_names = ", ".join(list(step["per_stock"].keys())[:5]) or "None"
            lines.append(f"- Covered stocks: {stock_names}")
        if step.get("preferred_sectors"):
            lines.append(f"- Preferred sectors: {', '.join(step['preferred_sectors'])}")
        if step.get("key_macro_risks"):
            lines.append(f"- Key macro risks: {', '.join(step['key_macro_risks'])}")

    lines.extend(["", "## Final Picks"])
    top_picks = final_decision.get("top_picks", [])
    if top_picks:
        for idx, pick in enumerate(top_picks, start=1):
            lines.append(
                f"{idx}. **{pick.get('ticker', 'UNKNOWN')}** ({pick.get('name', '')})"
                f" | score={pick.get('final_score', 0.0):.2f}"
            )
            lines.append(f"   - Rationale: {pick.get('rationale', 'No rationale available.')}")
    else:
        lines.append("No picks were generated.")

    lines.extend(["", "## Rejected Stocks"])
    rejected_stocks = final_decision.get("rejected_stocks", [])
    if rejected_stocks:
        for item in rejected_stocks:
            lines.append(
                f"- **{item.get('ticker', 'UNKNOWN')}** ({item.get('name', '')})"
                f" | score={item.get('final_score', 0.0):.2f}"
                f" | reason={item.get('reason', 'No reason available.')}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines)


def write_final_reports(state: StockPickerState) -> None:
    """Write the combined final markdown and JSON reports."""
    generate_final_reports(
        output_dir=resolve_output_dir(state),
        run_metadata=state.get("run_metadata", {}),
    )


def resolve_output_dir(state: StockPickerState) -> Path:
    """Return the configured output directory for reports."""
    configured = state.get("run_metadata", {}).get("output_dir")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return DEFAULT_OUTPUT_DIR


def resolve_step_summary_dir(state: StockPickerState) -> Path:
    """Return the output directory for step summaries."""
    return resolve_output_dir(state) / "step_summaries"


def resolve_final_report_md_path(state: StockPickerState) -> Path:
    """Return the markdown report output path."""
    return resolve_output_dir(state) / DEFAULT_FINAL_REPORT_MD.name


def resolve_final_report_json_path(state: StockPickerState) -> Path:
    """Return the JSON report output path."""
    return resolve_output_dir(state) / DEFAULT_FINAL_REPORT_JSON.name
