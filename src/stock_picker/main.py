"""Small runnable entrypoint for the stock picker MVP."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '2\.10\.2' or newer of 'numexpr'.*",
    category=UserWarning,
)

from stock_picker.env_utils import apply_env_defaults_to_run_metadata, load_project_env
from stock_picker.data_providers.feature_builder import build_meaningful_run_check
from stock_picker.forward_testing import write_forward_test_snapshot
from stock_picker.graph import build_graph
from stock_picker.news_quality_audit import write_news_quality_audit
from stock_picker.news_vectorstore import cleanup_old_news_documents
from stock_picker.report_generator import generate_final_reports, resolve_output_dir
from stock_picker.reporting import (
    prepare_reporting_outputs,
)
from stock_picker.security import reject_sensitive_config, sanitize_for_export
from stock_picker.state import StockPickerState

SAMPLE_UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_universe.csv"
DEMO_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "demo_run.json"
FULL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "full_run.example.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "demo_report.md"


def build_sample_state() -> StockPickerState:
    """Build a small sample run configuration for the prototype."""
    return {
        "universe": [],
        "filtered_candidates": [],
        "prefilter_score_breakdown": {},
        "stock_data": {},
        "macro_view": {},
        "equity_analysis": {},
        "risk_analysis": {},
        "final_decision": {},
        "debug_notes": [],
        "run_metadata": {
            "prototype_stage": "minimal_skeleton",
            "data_mode": "mock",
            "macro_mode": "state",
            "use_ollama": False,
            "ollama_base_url": "http://127.0.0.1:11434",
            "ollama_model": "gemma:7b",
            "ollama_enabled_agents": ["macro_agent", "equity_agent", "risk_agent", "portfolio_manager"],
            "universe_source": str(SAMPLE_UNIVERSE_PATH),
            "prefilter_top_n": 10,
            "macro_context": {
                "growth_signal": "stable",
                "inflation_signal": "cooling",
                "policy_signal": "unchanged",
                "volatility_signal": "moderate",
                "growth": {
                    "pmi_level": 51.8,
                    "pmi_trend": "stable",
                    "employment_trend": "stable",
                },
                "inflation": {
                    "headline_cpi_trend": "cooling",
                    "core_cpi_trend": "sticky",
                },
                "policy": {
                    "rate_direction": "unchanged",
                    "yield_curve": "slightly_inverted",
                },
                "market": {
                    "volatility_regime": "moderate",
                    "usd_trend": "firm",
                    "oil_trend": "stable",
                },
            },
            "macro_notes": {
                "source_mode": "local_structured_demo",
                "notes": [
                    "Use structured macro fields conservatively.",
                    "Do not infer unsupported macro shocks.",
                ],
            },
            "sector_hints": {
                "overweight": ["Technology"],
                "underweight": ["Energy"],
                "key_macro_risks": ["policy uncertainty"],
            },
        },
    }


def load_demo_config(config_path: Path = DEMO_CONFIG_PATH) -> dict[str, Any]:
    """Load a run configuration from a local JSON file."""
    load_project_env()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Run config must contain a top-level JSON object.")
    reject_sensitive_config(config)
    return config


def build_demo_state(config: dict[str, Any]) -> StockPickerState:
    """Build a fresh workflow state from the demo configuration."""
    reject_sensitive_config(config)
    run_metadata = {
        **dict(config.get("run_metadata", {})),
        "scenario_name": config.get("name", "demo"),
    }
    run_metadata = apply_env_defaults_to_run_metadata(run_metadata)
    return {
        "universe": [],
        "filtered_candidates": [],
        "prefilter_score_breakdown": {},
        "stock_data": {},
        "macro_view": {},
        "equity_analysis": {},
        "risk_analysis": {},
        "final_decision": {},
        "debug_notes": [],
        "run_metadata": run_metadata,
    }


def parse_args() -> argparse.Namespace:
    """Parse a minimal CLI so demo and full-mode runs can coexist."""
    parser = argparse.ArgumentParser(description="Run the multi-agent stock picker workflow.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEMO_CONFIG_PATH),
        help="Path to a JSON config file. Defaults to the credential-free offline demo.",
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Use data/full_run.example.json after generating public universe CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a configured workflow end to end and save report artifacts."""
    args = parse_args()
    config_path = FULL_CONFIG_PATH if args.full_mode else Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[2] / config_path).resolve()

    config = load_demo_config(config_path)
    app = build_graph()
    initial_state = build_demo_state(config)
    prepare_reporting_outputs(initial_state)
    result = app.invoke(initial_state)
    output_dir = resolve_output_dir(result.get("run_metadata", {}).get("output_dir", "outputs"))
    step_summary_dir = output_dir / "step_summaries"

    print_demo_header(config, config_path)
    print_section("1) Shortlisted Candidates")
    print_shortlisted_candidates(result)

    print_section("2) Macro Summary")
    print_macro_summary(result)

    print_section("3) Equity Analysis Summary")
    print_equity_summary(result)

    print_section("4) Risk Analysis Summary")
    print_risk_summary(result)

    print_section("5) Final Top 3 Picks")
    print_top_picks(result)

    print_section("6) Rejected Stocks With Reasons")
    print_rejected_stocks(result)

    write_demo_report(result, config, resolve_report_path(config))
    print_section("Report Output")
    print(f"Saved step summaries to {_display_path(step_summary_dir)}")
    print(f"Saved demo markdown report to {_display_path(resolve_report_path(config))}")
    try:
        meaningful = build_meaningful_run_check(
            pd.DataFrame(result.get("filtered_candidates", [])),
            news_covered_count=sum(1 for item in result.get("filtered_candidates", []) if item.get("news_documents_used")),
            backtest_status=result.get("backtest_results", {}).get("status", "unknown"),
        )
        result["run_metadata"] = {**result.get("run_metadata", {}), "meaningful_run_check": meaningful}
    except Exception:
        pass
    try:
        audit_summary = write_news_quality_audit(result, output_dir)
        result["run_metadata"] = {
            **result.get("run_metadata", {}),
            "news_quality_audit_summary": audit_summary,
        }
        print(f"Saved news quality audit to {_display_path(audit_summary.get('csv_path'))}")
    except Exception as exc:
        print(f"Reporting export warning: news quality audit failed safely ({sanitize_for_export(str(exc))}).")
    try:
        cleanup_summary = cleanup_old_news_documents(result.get("run_metadata", {}))
        result["run_metadata"] = {
            **result.get("run_metadata", {}),
            "vectorstore_cleanup_summary": cleanup_summary,
        }
        print(f"Saved vectorstore cleanup report to {_display_path(cleanup_summary.get('report_path'))}")
    except Exception as exc:
        print(f"Reporting export warning: vectorstore cleanup failed safely ({sanitize_for_export(str(exc))}).")
    try:
        forward_summary = write_forward_test_snapshot(result, output_dir)
        result["run_metadata"] = {
            **result.get("run_metadata", {}),
            "forward_testing_ledger": forward_summary,
        }
        if forward_summary.get("enabled"):
            print(f"Saved forward testing snapshot to {_display_path(forward_summary.get('snapshot_path'))}")
    except Exception as exc:
        print(f"Reporting export warning: forward testing snapshot failed safely ({sanitize_for_export(str(exc))}).")
    try:
        final_report_paths = generate_final_reports(
            output_dir=result.get("run_metadata", {}).get("output_dir", "outputs"),
            run_metadata=result.get("run_metadata", {}),
        )
        print(f"Saved final markdown report to {_display_path(final_report_paths['markdown'])}")
        print(f"Saved combined JSON report to {_display_path(final_report_paths['json'])}")
        print(f"Saved full pipeline summary to {_display_path(final_report_paths['full_pipeline_summary'])}")
    except OSError as exc:
        print(f"Reporting export warning: final report generation was skipped ({sanitize_for_export(str(exc))}).")
    except Exception as exc:
        print(f"Reporting export warning: final report generation failed safely ({sanitize_for_export(str(exc))}).")


def format_list(items: list[str], limit: int = 0) -> str:
    """Format a short list for console output."""
    visible_items = items[:limit] if limit else items
    return ", ".join(visible_items) or "None"


def print_demo_header(config: dict[str, Any], config_path: Path) -> None:
    """Print a small stable header for presentation-friendly runs."""
    print("Multi-Agent Stock Picker MVP Demo")
    print(f"Scenario: {config.get('name', 'demo')}")
    print(f"Config: {_display_path(config_path)}")
    print(config.get("description", ""))


def _display_path(path_value: Any) -> str:
    """Render project paths without exposing a local account directory."""
    if path_value in (None, ""):
        return "Unavailable"
    return str(sanitize_for_export(str(path_value)))


def print_section(title: str) -> None:
    """Print a clean console section heading."""
    print(f"\n[{title}]")


def print_shortlisted_candidates(result: dict[str, Any]) -> None:
    candidates = result.get("filtered_candidates", [])
    if not candidates:
        print("No shortlisted candidates.")
        return

    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"{idx}. {candidate.get('ticker', 'UNKNOWN'):<4} "
            f"{candidate.get('name', '')} | "
            f"sector={candidate.get('sector', 'n/a')} | "
            f"score={candidate.get('quant_score', 0.0):.2f} | "
            f"rank={candidate.get('prefilter_rank', 'n/a')}"
        )


def print_macro_summary(result: dict[str, Any]) -> None:
    macro_view = result.get("macro_view", {})
    print(f"Status: {macro_view.get('status', 'n/a')}")
    print(f"Summary: {macro_view.get('macro_summary', 'No macro summary available.')}")
    print(f"Preferred sectors: {format_list(macro_view.get('preferred_sectors', []))}")
    print(f"Risk sectors: {format_list(macro_view.get('risk_sectors', []))}")
    print(f"Key macro risks: {format_list(macro_view.get('key_macro_risks', []))}")


def print_equity_summary(result: dict[str, Any]) -> None:
    equity_analysis = result.get("equity_analysis", {})
    if not equity_analysis:
        print("No equity analysis available.")
        return

    for ticker, view in equity_analysis.items():
        print(
            f"{ticker}: view={view.get('overall_view', 'n/a')} | "
            f"strengths={format_list(view.get('strengths', []), limit=2)} | "
            f"weaknesses={format_list(view.get('weaknesses', []), limit=2)}"
        )


def print_risk_summary(result: dict[str, Any]) -> None:
    risk_analysis = result.get("risk_analysis", {})
    if not risk_analysis:
        print("No risk analysis available.")
        return

    for ticker, view in risk_analysis.items():
        print(
            f"{ticker}: risk_level={view.get('risk_level', 'n/a')} | "
            f"main_risks={format_list(view.get('main_risks', []), limit=3)} | "
            f"comment={view.get('risk_comment', 'No risk comment available.')}"
        )


def print_top_picks(result: dict[str, Any]) -> None:
    final_decision = result.get("final_decision", {})
    top_picks = final_decision.get("top_picks", [])
    if not top_picks:
        print("No final picks.")
        return

    print(f"Committee summary: {final_decision.get('committee_summary', 'No committee summary available.')}")
    for idx, pick in enumerate(top_picks, start=1):
        print(
            f"{idx}. {pick.get('ticker', 'UNKNOWN'):<4} {pick.get('name', '')} | "
            f"score={pick.get('final_score', 0.0):.2f} | "
            f"rationale={pick.get('rationale', 'No rationale available.')}"
        )


def print_rejected_stocks(result: dict[str, Any]) -> None:
    rejected_stocks = result.get("final_decision", {}).get("rejected_stocks", [])
    if not rejected_stocks:
        print("No rejected stocks.")
        return

    for item in rejected_stocks:
        print(
            f"- {item.get('ticker', 'UNKNOWN'):<4} {item.get('name', '')} | "
            f"score={item.get('final_score', 0.0):.2f} | "
            f"reason={item.get('reason', 'No reason available.')}"
        )


def resolve_report_path(config: dict[str, Any]) -> Path:
    """Resolve the markdown report path from the demo config."""
    configured_path = config.get("report_output")
    if not configured_path:
        return DEFAULT_REPORT_PATH

    report_path = Path(configured_path)
    if report_path.is_absolute():
        return report_path
    return (DEMO_CONFIG_PATH.parent.parent / report_path).resolve()


def build_markdown_report(result: dict[str, Any], config: dict[str, Any]) -> str:
    """Build a concise markdown report for demo sharing."""
    macro_view = result.get("macro_view", {})
    final_decision = result.get("final_decision", {})
    top_picks = final_decision.get("top_picks", [])
    rejected_stocks = final_decision.get("rejected_stocks", [])

    lines = [
        "# Multi-Agent Stock Picker Demo Report",
        "",
        f"Scenario: `{config.get('name', 'demo')}`",
        "",
        "## Macro Summary",
        f"- Status: {macro_view.get('status', 'n/a')}",
        f"- Summary: {macro_view.get('macro_summary', 'No macro summary available.')}",
        f"- Preferred sectors: {format_list(macro_view.get('preferred_sectors', []))}",
        f"- Risk sectors: {format_list(macro_view.get('risk_sectors', []))}",
        f"- Key macro risks: {format_list(macro_view.get('key_macro_risks', []))}",
        "",
        "## Top 3 Picks",
    ]

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
    if rejected_stocks:
        for item in rejected_stocks:
            lines.append(
                f"- **{item.get('ticker', 'UNKNOWN')}** ({item.get('name', '')})"
                f" | score={item.get('final_score', 0.0):.2f}"
                f" | reason={item.get('reason', 'No reason available.')}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Workflow Notes",
            f"- Ranking method: {final_decision.get('ranking_method', 'No ranking method available.')}",
            f"- Committee summary: {final_decision.get('committee_summary', 'No committee summary available.')}",
        ]
    )
    return "\n".join(lines)


def write_demo_report(result: dict[str, Any], config: dict[str, Any], report_path: Path) -> None:
    """Write the markdown report artifact for the stable demo run."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(result, config), encoding="utf-8")


if __name__ == "__main__":
    main()
