"""Fail-soft U.S. equity data provider backed by yfinance."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def default_history_window(end_date: str | None = None, days: int = 430) -> tuple[str, str]:
    """Return a simple calendar-day window long enough for 12-month features."""
    end = pd.Timestamp(end_date).date() if end_date else date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def fetch_us_price_history_and_fundamentals(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch U.S. daily close/volume and basic fundamentals with graceful fallback."""
    normalized_tickers = [str(ticker).strip() for ticker in tickers if str(ticker).strip()]
    if not normalized_tickers:
        return pd.DataFrame(), pd.DataFrame(), {"status": "skipped", "reason": "no tickers"}

    try:
        import yfinance as yf
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "skipped", "reason": f"yfinance import failed: {exc}"},
        )
    _configure_yfinance_cache(yf)

    price_history = _download_price_history(yf, normalized_tickers, start_date, end_date)
    fundamentals = _download_fundamentals(yf, normalized_tickers)
    status = "completed" if not price_history.empty or not fundamentals.empty else "skipped"
    reason = "" if status == "completed" else "yfinance returned no usable data"
    return price_history, fundamentals, {"status": status, "reason": reason, "ticker_count": len(normalized_tickers)}


def _download_price_history(
    yf: Any,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False,
            auto_adjust=True,
            timeout=15,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if isinstance(raw.columns, pd.MultiIndex):
        close_frame = _select_multiindex_field(raw, "Close")
        volume_frame = _select_multiindex_field(raw, "Volume")
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        close_frame = raw[[close_col]].rename(columns={close_col: tickers[0]}) if close_col else pd.DataFrame()
        volume_frame = raw[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in raw.columns else pd.DataFrame()

    for ticker in tickers:
        if ticker not in close_frame.columns:
            continue
        close_series = pd.to_numeric(close_frame[ticker], errors="coerce")
        volume_series = (
            pd.to_numeric(volume_frame[ticker], errors="coerce")
            if ticker in volume_frame.columns
            else pd.Series(index=close_series.index, dtype=float)
        )
        for dt, close_value in close_series.dropna().items():
            rows.append(
                {
                    "date": pd.Timestamp(dt).date().isoformat(),
                    "ticker": ticker,
                    "close": float(close_value),
                    "volume": float(volume_series.get(dt, 0.0) or 0.0),
                    "market": "US",
                    "source": "yfinance",
                }
            )
    return pd.DataFrame(rows)


def _configure_yfinance_cache(yf: Any) -> None:
    """Move yfinance sqlite caches out of OneDrive/project paths when possible."""
    try:
        cache_dir = Path.home() / "AppData" / "Local" / "Temp" / "stock_picker_yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.cache.set_cache_location(str(cache_dir))
    except Exception:
        return


def _select_multiindex_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    level_zero = raw.columns.get_level_values(0)
    if field in level_zero:
        return raw[field]
    if field == "Close" and "Adj Close" in level_zero:
        return raw["Adj Close"]
    return pd.DataFrame(index=raw.index)


def _download_fundamentals(yf: Any, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_info() or {}
        except Exception:
            info = {}
        rows.append(
            {
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or "",
                "sector": info.get("sector") or "",
                "market_cap": _safe_float(info.get("marketCap")),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "price_to_book": _safe_float(info.get("priceToBook")),
                "roe": _safe_float(info.get("returnOnEquity")),
                "fundamental_source": "yfinance" if info else "missing",
            }
        )
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if pd.isna(numeric):
            return None
        return numeric
    except Exception:
        return None
