"""Single as-of-date ex-post validation for final top picks."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_picker.export_utils import PROJECT_ROOT, save_step_summary
from stock_picker.state import StockPickerState


DEFAULT_HORIZONS = [21, 63]
BACKTEST_LIMITATIONS = [
    "Survivorship bias may be present.",
    "Transaction costs are not included.",
    "Slippage is not included.",
    "This is a single as-of-date validation, not a rolling backtest.",
    "yfinance data may contain missing values or adjusted-price convention differences.",
    "Forward returns must use only price observations after the as_of_date to avoid lookahead bias.",
]


def backtesting_node(state: StockPickerState) -> StockPickerState:
    """Run optional ex-post validation after the portfolio decision is available."""
    result = run_backtest(
        final_decision=state.get("final_decision", {}),
        run_metadata=state.get("run_metadata", {}),
    )
    final_decision = dict(state.get("final_decision", {}))
    final_decision["backtest"] = result
    state["final_decision"] = final_decision
    state["backtest_results"] = result
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        f"Backtesting completed with status={result.get('status', 'unknown')}.",
    ]
    export_backtest_step_summary(state)
    return state


def run_backtest(final_decision: dict[str, Any], run_metadata: dict[str, Any]) -> dict[str, Any]:
    """Calculate forward returns for final top picks when sufficient data exists."""
    output_dir = resolve_output_dir(run_metadata.get("backtest_output_dir") or run_metadata.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = bool(run_metadata.get("backtest_enabled", True))
    top_picks = final_decision.get("top_picks", []) or []
    tickers = [str(pick.get("ticker", "")).strip() for pick in top_picks[:3] if str(pick.get("ticker", "")).strip()]
    benchmark_ticker = str(run_metadata.get("benchmark_ticker", "")).strip()
    as_of_date = str(run_metadata.get("as_of_date", "")).strip()
    horizons = normalize_horizons(run_metadata.get("backtest_horizons", DEFAULT_HORIZONS))

    if not enabled:
        result = build_skipped_result("backtest_enabled is false", tickers, as_of_date, horizons, benchmark_ticker)
        write_backtest_outputs(result, output_dir)
        return result
    if not tickers:
        result = build_skipped_result("no top_picks available", tickers, as_of_date, horizons, benchmark_ticker)
        write_backtest_outputs(result, output_dir)
        return result
    if not as_of_date:
        result = build_skipped_result("as_of_date not provided", tickers, as_of_date, horizons, benchmark_ticker)
        write_backtest_outputs(result, output_dir)
        return result

    all_tickers = tickers + ([benchmark_ticker] if benchmark_ticker else [])
    prices, source_note = load_price_history(
        tickers=all_tickers,
        run_metadata=run_metadata,
        as_of_date=as_of_date,
        horizons=horizons,
    )
    if prices.empty:
        result = build_skipped_result("price history unavailable", tickers, as_of_date, horizons, benchmark_ticker)
        result["price_source_note"] = source_note
        write_backtest_outputs(result, output_dir)
        return result

    individual = {
        ticker: compute_forward_returns(prices, ticker, as_of_date, horizons)
        for ticker in tickers
    }
    benchmark = compute_forward_returns(prices, benchmark_ticker, as_of_date, horizons) if benchmark_ticker else {}
    portfolio = compute_equal_weight_portfolio_returns(individual, horizons)
    excess = {
        f"{horizon}d": safe_round(portfolio.get(f"{horizon}d_return") - benchmark.get(f"{horizon}d_return"))
        if portfolio.get(f"{horizon}d_return") is not None and benchmark.get(f"{horizon}d_return") is not None
        else None
        for horizon in horizons
    }

    status = "completed" if any(value is not None for value in portfolio.values()) else "insufficient_data"
    result = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of_date,
        "horizons": horizons,
        "tickers": tickers,
        "benchmark_ticker": benchmark_ticker,
        "price_source_note": source_note,
        "individual_forward_returns": individual,
        "portfolio_equal_weight_returns": portfolio,
        "benchmark_forward_returns": benchmark,
        "excess_returns": excess,
        "limitations": BACKTEST_LIMITATIONS,
    }
    write_backtest_outputs(result, output_dir)
    return result


def load_price_history(
    *,
    tickers: list[str],
    run_metadata: dict[str, Any],
    as_of_date: str,
    horizons: list[int],
) -> tuple[pd.DataFrame, str]:
    local_path = run_metadata.get("price_history_source_path")
    if local_path:
        frame = load_local_price_history(local_path)
        if not frame.empty:
            return frame, "local_price_history"

    if not bool(run_metadata.get("backtest_yfinance_enabled", False)):
        return pd.DataFrame(), "price history unavailable; yfinance backtest fetch disabled"

    try:
        import yfinance as yf

        start = pd.Timestamp(as_of_date)
        end = start + pd.Timedelta(days=max(horizons) * 3 + 14)
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            threads=False,
            timeout=15,
        )
        if raw.empty:
            return pd.DataFrame(), "yfinance returned empty price history"
        if isinstance(raw.columns, pd.MultiIndex):
            price = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
        else:
            price = raw[["Adj Close"]] if "Adj Close" in raw.columns else raw[["Close"]]
            price.columns = tickers[:1]
        return normalize_price_frame(price), "yfinance"
    except Exception as exc:
        return pd.DataFrame(), f"yfinance price fetch failed: {exc}"


def load_local_price_history(path_value: Any) -> pd.DataFrame:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "date" not in frame.columns:
        return pd.DataFrame()

    if {"ticker", "close"} <= set(frame.columns):
        pivot = frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    else:
        pivot = frame.set_index("date")
    return normalize_price_frame(pivot)


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    normalized = normalized[normalized.index.notna()]
    normalized = normalized.sort_index()
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def compute_forward_returns(
    prices: pd.DataFrame,
    ticker: str,
    as_of_date: str,
    horizons: list[int],
) -> dict[str, Any]:
    if not ticker or ticker not in prices.columns:
        return {"status": "insufficient_data", "reason": "ticker price column missing"}
    series = prices[ticker].dropna()
    start_date = pd.Timestamp(as_of_date)
    future = series[series.index >= start_date]
    if len(future) <= min(horizons):
        return {"status": "insufficient_data", "reason": "not enough observations after as_of_date"}

    base_price = float(future.iloc[0])
    result: dict[str, Any] = {
        "status": "completed",
        "base_date": future.index[0].strftime("%Y-%m-%d"),
        "base_price": safe_round(base_price),
    }
    for horizon in horizons:
        key = f"{horizon}d_return"
        if len(future) <= horizon or base_price == 0:
            result[key] = None
            result[f"{horizon}d_status"] = "insufficient_data"
            continue
        result[key] = safe_round(float(future.iloc[horizon]) / base_price - 1.0)
        result[f"{horizon}d_end_date"] = future.index[horizon].strftime("%Y-%m-%d")
    return result


def compute_equal_weight_portfolio_returns(individual: dict[str, dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        key = f"{horizon}d_return"
        values = [
            returns.get(key)
            for returns in individual.values()
            if isinstance(returns.get(key), (int, float))
        ]
        result[key] = safe_round(sum(values) / len(values)) if values else None
    return result


def write_backtest_outputs(result: dict[str, Any], output_dir: Path) -> None:
    summary_path = output_dir / "backtest_summary.csv"
    report_path = output_dir / "backtest_report.md"
    rows = build_backtest_summary_rows(result)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=backtest_summary_columns())
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(build_backtest_markdown(result), encoding="utf-8")


def build_backtest_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    horizons = result.get("horizons", DEFAULT_HORIZONS)
    for ticker, returns in result.get("individual_forward_returns", {}).items():
        row = {"type": "individual", "ticker": ticker, "status": returns.get("status", result.get("status"))}
        for horizon in horizons:
            row[f"{horizon}d_return"] = returns.get(f"{horizon}d_return")
        rows.append(row)
    portfolio_row = {"type": "portfolio_equal_weight", "ticker": "TOP3", "status": result.get("status")}
    benchmark_row = {"type": "benchmark", "ticker": result.get("benchmark_ticker", ""), "status": result.get("status")}
    excess_row = {"type": "excess", "ticker": "TOP3_MINUS_BENCHMARK", "status": result.get("status")}
    for horizon in horizons:
        portfolio_row[f"{horizon}d_return"] = result.get("portfolio_equal_weight_returns", {}).get(f"{horizon}d_return")
        benchmark_row[f"{horizon}d_return"] = result.get("benchmark_forward_returns", {}).get(f"{horizon}d_return")
        excess_row[f"{horizon}d_return"] = result.get("excess_returns", {}).get(f"{horizon}d")
    rows.extend([portfolio_row, benchmark_row, excess_row])
    return rows


def backtest_summary_columns() -> list[str]:
    return ["type", "ticker", "status", "21d_return", "63d_return"]


def build_backtest_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Ex-post Validation",
        "",
        f"- Status: {result.get('status', 'unknown')}",
        f"- As-of date: {result.get('as_of_date') or 'Unavailable'}",
        f"- Horizons: {', '.join(str(item) for item in result.get('horizons', DEFAULT_HORIZONS))} trading days",
        f"- Top 3 tickers: {', '.join(result.get('tickers', [])) or 'Unavailable'}",
        f"- Benchmark: {result.get('benchmark_ticker') or 'Unavailable'}",
        f"- Price source: {result.get('price_source_note', 'Unavailable')}",
        "",
        "## Equal-weight Portfolio",
    ]
    for key, value in result.get("portfolio_equal_weight_returns", {}).items():
        lines.append(f"- {key}: {format_return(value)}")
    lines.extend(["", "## Benchmark And Excess"])
    for key, value in result.get("benchmark_forward_returns", {}).items():
        if key.endswith("_return"):
            lines.append(f"- benchmark {key}: {format_return(value)}")
    for key, value in result.get("excess_returns", {}).items():
        lines.append(f"- excess {key}: {format_return(value)}")
    lines.extend(["", "## Individual Forward Returns"])
    for ticker, returns in result.get("individual_forward_returns", {}).items():
        rendered = ", ".join(
            f"{key}={format_return(value)}" for key, value in returns.items() if key.endswith("_return")
        )
        lines.append(f"- {ticker}: {rendered or returns.get('reason', 'Unavailable')}")
    lines.extend(["", "## Limitations"])
    for limitation in result.get("limitations", BACKTEST_LIMITATIONS):
        lines.append(f"- {limitation}")
    return "\n".join(lines)


def export_backtest_step_summary(state: StockPickerState) -> None:
    result = state.get("backtest_results", {})
    save_step_summary(
        stage_name="backtesting",
        summary_data={
            "status": result.get("status", "unknown"),
            "as_of_date": result.get("as_of_date", ""),
            "horizons": result.get("horizons", DEFAULT_HORIZONS),
            "portfolio_equal_weight_returns": result.get("portfolio_equal_weight_returns", {}),
            "benchmark_forward_returns": result.get("benchmark_forward_returns", {}),
            "excess_returns": result.get("excess_returns", {}),
            "limitations": result.get("limitations", BACKTEST_LIMITATIONS),
            "note": "Single as-of-date validation; not a production backtest.",
        },
        step_number=7,
        output_dir=state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries"),
    )
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "backtesting",
        ],
    }


def build_skipped_result(
    reason: str,
    tickers: list[str],
    as_of_date: str,
    horizons: list[int],
    benchmark_ticker: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of_date,
        "horizons": horizons,
        "tickers": tickers,
        "benchmark_ticker": benchmark_ticker,
        "individual_forward_returns": {},
        "portfolio_equal_weight_returns": {f"{horizon}d_return": None for horizon in horizons},
        "benchmark_forward_returns": {},
        "excess_returns": {f"{horizon}d": None for horizon in horizons},
        "limitations": BACKTEST_LIMITATIONS,
    }


def normalize_horizons(value: Any) -> list[int]:
    if not isinstance(value, list):
        return DEFAULT_HORIZONS
    horizons = []
    for item in value:
        try:
            horizon = int(item)
        except (TypeError, ValueError):
            continue
        if horizon > 0:
            horizons.append(horizon)
    return horizons or DEFAULT_HORIZONS


def resolve_output_dir(output_dir: Any) -> Path:
    if not output_dir:
        return PROJECT_ROOT / "outputs"
    path = Path(str(output_dir))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_round(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return round(numeric, 6)


def format_return(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Unavailable"
