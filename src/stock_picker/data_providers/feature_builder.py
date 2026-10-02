"""Build standardized stock features from price history snapshots."""

from __future__ import annotations

from typing import Any

import pandas as pd


PRICE_COLUMNS = ["date", "ticker", "close", "volume", "market", "source"]
FEATURE_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "market",
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
    "feature_source",
    "price_source",
    "fundamental_source",
    "as_of_date",
    "available_price_days",
    "has_full_12m_history",
    "data_quality_flags",
    "data_quality_warning",
]


def build_features_from_price_history(
    price_history: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    as_of_date: str | None = None,
    feature_source: str = "price_history_snapshot",
) -> pd.DataFrame:
    """Calculate standard quantitative features without touching vectorstore data."""
    if price_history.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    prices = normalize_price_history(price_history)
    if as_of_date:
        prices = prices[prices["date"] <= pd.Timestamp(as_of_date)]
    rows: list[dict[str, Any]] = []
    universe_map = _records_by_ticker(universe)
    fundamental_map = _records_by_ticker(fundamentals)
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        close = pd.to_numeric(group["close"], errors="coerce").dropna()
        volume = pd.to_numeric(group.get("volume", pd.Series(dtype=float)), errors="coerce").dropna()
        meta = universe_map.get(str(ticker), {})
        fmeta = fundamental_map.get(str(ticker), {})
        flags: list[str] = []
        if close.empty:
            flags.append("missing_price_history")
        available_days = int(len(close))
        if available_days < 252:
            flags.append("insufficient_12m_history")
        for field in ["trailing_pe", "price_to_book", "roe"]:
            if pd.isna(fmeta.get(field, pd.NA)):
                flags.append(f"missing_{field}")
        returns = close.pct_change().dropna()
        running_max = close.tail(252).cummax()
        drawdown = close.tail(252) / running_max - 1.0 if not running_max.empty else pd.Series(dtype=float)
        rows.append(
            {
                "ticker": str(ticker),
                "name": meta.get("name", fmeta.get("name", "")),
                "sector": meta.get("sector", fmeta.get("sector", "")),
                "market": meta.get("market", group["market"].dropna().iloc[-1] if "market" in group and group["market"].notna().any() else ""),
                "price": _last(close),
                "ret_1m": _period_return(close, 21),
                "ret_12m": _period_return(close, 252),
                "volatility_60d": _volatility(returns.tail(60)),
                "max_drawdown_1y": _safe_float(drawdown.min()) if not drawdown.empty else None,
                "avg_volume_20d": _safe_float(volume.tail(20).mean()) if not volume.empty else None,
                "market_cap": _safe_float(fmeta.get("market_cap")),
                "trailing_pe": _safe_float(fmeta.get("trailing_pe")),
                "price_to_book": _safe_float(fmeta.get("price_to_book")),
                "roe": _safe_float(fmeta.get("roe")),
                "feature_source": feature_source,
                "price_source": group["source"].dropna().iloc[-1] if "source" in group and group["source"].notna().any() else feature_source,
                "fundamental_source": fmeta.get("fundamental_source", "local_csv_or_missing"),
                "as_of_date": as_of_date or group["date"].max().date().isoformat(),
                "available_price_days": available_days,
                "has_full_12m_history": available_days >= 252,
                "data_quality_flags": ",".join(sorted(set(flags))),
                "data_quality_warning": "no material quality warnings" if not flags else "; ".join(sorted(set(flags))),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def normalize_price_history(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in PRICE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["ticker"] = normalized["ticker"].astype(str)
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce")
    return normalized.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])


def build_meaningful_run_check(
    features: pd.DataFrame,
    news_covered_count: int = 0,
    backtest_status: str = "unknown",
) -> dict[str, Any]:
    total = max(len(features), 1)
    price_coverage = _coverage(features, ["price"])
    return_coverage = _coverage(features, ["ret_1m", "ret_12m"])
    fundamental_coverage = _coverage(features, ["trailing_pe", "price_to_book", "roe"])
    news_coverage = round(news_covered_count / total, 4) if total else 0.0
    reasons = []
    if price_coverage < 0.90:
        reasons.append("price coverage below 90%")
    if return_coverage < 0.80:
        reasons.append("return coverage below 80%")
    if fundamental_coverage < 0.50:
        reasons.append("fundamental coverage below 50%")
    if backtest_status != "completed":
        reasons.append("backtest not completed")
    return {
        "price_coverage_ratio": price_coverage,
        "return_coverage_ratio": return_coverage,
        "fundamental_coverage_ratio": fundamental_coverage,
        "news_coverage_ratio": news_coverage,
        "backtest_status": backtest_status,
        "is_meaningful_run": not reasons,
        "reasons": reasons,
    }


def _coverage(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame.empty:
        return 0.0
    present = frame[columns].notna().any(axis=1) if all(col in frame for col in columns) else pd.Series(False, index=frame.index)
    return round(float(present.mean()), 4)


def _records_by_ticker(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return {}
    return {str(row["ticker"]): row for row in frame.to_dict(orient="records")}


def _period_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    start = float(close.iloc[-days - 1])
    end = float(close.iloc[-1])
    return round(end / start - 1.0, 6) if start else None


def _volatility(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    return round(float(returns.std() * (252 ** 0.5)), 6)


def _last(series: pd.Series) -> float | None:
    return _safe_float(series.iloc[-1]) if not series.empty else None


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return None if pd.isna(numeric) else round(numeric, 6)
