"""Fail-soft pykrx provider for Korean equity price snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd


def fetch_krx_price_history(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch KRX OHLCV data through pykrx. Failures return an empty frame."""
    try:
        from pykrx import stock
    except Exception as exc:
        return pd.DataFrame(), {"status": "skipped", "reason": f"pykrx import failed: {exc}"}
    rows = []
    failed = []
    start = _compact_date(start_date)
    end = _compact_date(end_date)
    for ticker in tickers:
        krx_ticker = str(ticker).replace(".KS", "").zfill(6)
        try:
            raw = stock.get_market_ohlcv_by_date(start, end, krx_ticker)
        except Exception as exc:
            failed.append({"ticker": ticker, "reason": str(exc)})
            continue
        if raw.empty:
            failed.append({"ticker": ticker, "reason": "empty pykrx response"})
            continue
        for date_value, row in raw.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(date_value).date().isoformat(),
                    "ticker": str(ticker),
                    "close": row.get("종가"),
                    "volume": row.get("거래량"),
                    "market": "KR",
                    "source": "pykrx",
                }
            )
    return pd.DataFrame(rows), {
        "status": "completed" if rows else "skipped",
        "provider": "pykrx",
        "requested_tickers": len(tickers),
        "failed": failed,
    }


def default_history_window(end_date: str | None = None, days: int = 430) -> tuple[str, str]:
    end = pd.Timestamp(end_date or datetime.now().date())
    start = end - timedelta(days=days)
    return start.date().isoformat(), end.date().isoformat()


def _compact_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")
