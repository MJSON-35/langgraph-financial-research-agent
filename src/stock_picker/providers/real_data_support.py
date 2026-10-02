"""Helpers for minimal yfinance-backed MVP data loading."""

from __future__ import annotations

import math
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
import io
import os
from pathlib import Path
from typing import Any

import pandas as pd

from stock_picker.config import get_financial_data_quality_config
from stock_picker.financial_data_quality import validate_financial_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache" / "yfinance"
PREDEFINED_UNIVERSE_FILES = {
    "sp500": [DATA_DIR / "universes" / "sp500.csv"],
    "kospi200": [DATA_DIR / "universes" / "kospi200.csv"],
    "sp500_kospi200": [
        DATA_DIR / "universes" / "sp500.csv",
        DATA_DIR / "universes" / "kospi200.csv",
    ],
    "sp500_full": [DATA_DIR / "universes" / "sp500_full.csv"],
    "kospi200_full": [DATA_DIR / "universes" / "kospi200_full.csv"],
    "sp500_kospi200_full": [
        DATA_DIR / "universes" / "sp500_full.csv",
        DATA_DIR / "universes" / "kospi200_full.csv",
    ],
}
REAL_DATA_REQUIRED_COLUMNS = [
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
]
FUNDAMENTAL_COLUMNS = ["market_cap", "trailing_pe", "price_to_book", "roe"]


def get_predefined_universe_files(universe_name: str) -> list[Path] | None:
    """Return local CSV snapshots for a supported universe name."""
    return PREDEFINED_UNIVERSE_FILES.get(str(universe_name).lower())


def build_data_validation_summary(candidate_df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact validation summary for the loaded real-data frame."""
    if candidate_df.empty:
        return {
            "row_count": 0,
            "missing_by_column": {column: 0 for column in REAL_DATA_REQUIRED_COLUMNS},
            "fundamental_missing_ratio": 1.0,
            "warning": "No rows were loaded.",
        }

    missing_by_column = {
        column: int(candidate_df[column].isna().sum()) if column in candidate_df.columns else len(candidate_df)
        for column in REAL_DATA_REQUIRED_COLUMNS
    }
    available_fundamentals = candidate_df.reindex(columns=FUNDAMENTAL_COLUMNS)
    total_fundamental_cells = max(len(candidate_df) * len(FUNDAMENTAL_COLUMNS), 1)
    missing_fundamental_cells = int(available_fundamentals.isna().sum().sum())
    missing_ratio = round(missing_fundamental_cells / total_fundamental_cells, 4)
    warning = None
    if missing_ratio > 0.65:
        warning = (
            "Fundamental coverage is low. The pipeline will keep running, but value and "
            "quality signals may be weaker than price-based signals."
        )

    return {
        "row_count": len(candidate_df),
        "missing_by_column": missing_by_column,
        "fundamental_missing_ratio": missing_ratio,
        "warning": warning,
    }


def infer_market_from_ticker(ticker: str) -> str:
    """Map common ticker suffixes to a simple market label."""
    normalized = str(ticker).upper()
    if normalized.endswith(".KS") or normalized.endswith(".KQ"):
        return "KOSPI200"
    return "SP500"


def import_yfinance():
    """Import yfinance lazily so mock mode remains lightweight."""
    _clear_broken_proxy_env()
    try:
        import yfinance as yf
        import yfinance.cache as yf_cache
        import multitasking
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Install it or switch back to mock mode.") from exc
    try:
        multitasking.set_max_threads(0)
    except Exception:
        pass
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_cache_location(str(CACHE_DIR))
    except Exception:
        pass
    try:
        yf.set_tz_cache_location(str(CACHE_DIR))
    except Exception:
        pass
    _disable_yfinance_sqlite_cache(yf_cache)
    return yf


def _disable_yfinance_sqlite_cache(yf_cache: Any) -> None:
    """Disable sqlite-backed yfinance caches when local file IO is unreliable."""
    tz_dummy = None
    cookie_dummy = None
    isin_dummy = None
    try:
        tz_dummy = yf_cache._TzCacheDummy()
        yf_cache._TzCacheManager._tz_cache = tz_dummy
    except Exception:
        pass
    try:
        cookie_dummy = yf_cache._CookieCacheDummy()
        yf_cache._CookieCacheManager._Cookie_cache = cookie_dummy
    except Exception:
        pass
    try:
        isin_dummy = yf_cache._ISINCacheDummy()
        yf_cache._ISINCacheManager._isin_cache = isin_dummy
    except Exception:
        pass
    if tz_dummy is not None:
        try:
            yf_cache.get_tz_cache = lambda: tz_dummy
        except Exception:
            pass
    if cookie_dummy is not None:
        try:
            yf_cache.get_cookie_cache = lambda: cookie_dummy
        except Exception:
            pass
    if isin_dummy is not None:
        try:
            yf_cache.get_isin_cache = lambda: isin_dummy
        except Exception:
            pass


@contextmanager
def _temporary_proxy_bypass():
    """Ignore broken loopback proxy env vars during yfinance requests."""
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    removed: dict[str, str] = {}
    try:
        for key in proxy_keys:
            value = os.environ.get(key)
            if value and "127.0.0.1:9" in value:
                removed[key] = value
                os.environ.pop(key, None)
        yield
    finally:
        os.environ.update(removed)


def _clear_broken_proxy_env() -> None:
    """Remove loopback proxy env vars that break outbound yfinance requests."""
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in proxy_keys:
        value = os.environ.get(key)
        if value and "127.0.0.1:9" in value:
            os.environ.pop(key, None)


def safe_float(value: object) -> float | None:
    """Convert arbitrary values to float while preserving missing values."""
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    except Exception:
        return None


def scaled_score(value: float | None, low: float, high: float) -> float:
    """Map a numeric value into a 0-1 score."""
    if value is None or high <= low:
        return 0.50
    clipped = min(max(value, low), high)
    return round((clipped - low) / (high - low), 4)


def scaled_inverse_score(value: float | None, low: float, high: float) -> float:
    """Map lower-is-better values into a 0-1 score."""
    if value is None or high <= low:
        return 0.50
    clipped = min(max(value, low), high)
    return round(1.0 - ((clipped - low) / (high - low)), 4)


def bounded_or_default(value: float | None, default: float, lower: float, upper: float) -> float:
    """Bound values into a safe interval or return a fallback."""
    if value is None:
        return default
    return round(min(max(value, lower), upper), 4)


def average_available(values: list[float | None], default: float = 0.50) -> float:
    """Average the valid numeric values only."""
    valid = [value for value in values if value is not None]
    if not valid:
        return default
    return round(sum(valid) / len(valid), 4)


def infer_sector_risk(sector: object) -> str:
    """Use a simple interpretable sector-to-risk mapping."""
    normalized = str(sector or "").strip().lower()
    if normalized in {"energy", "materials", "real estate"}:
        return "high"
    if normalized in {"healthcare", "utilities", "consumer staples"}:
        return "low"
    return "medium"


def download_price_metrics(tickers: list[str], history_period: str = "2y") -> dict[str, dict[str, float | None]]:
    """Download daily price history once and compute the MVP price-based fields."""
    yf = import_yfinance()
    try:
        with _temporary_proxy_bypass(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            history = yf.download(
                tickers=tickers,
                period=history_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker",
                timeout=15,
            )
    except Exception:
        history = pd.DataFrame()

    return {ticker: extract_price_metrics(history, ticker) for ticker in tickers}


def extract_price_metrics(history: pd.DataFrame, ticker: str) -> dict[str, float | None]:
    """Compute price, returns, volatility, drawdown, and average volume."""
    if history.empty:
        return empty_price_metrics()

    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(0):
            return empty_price_metrics()
        ticker_frame = history[ticker].copy()
    else:
        ticker_frame = history.copy()

    close_series = pd.to_numeric(ticker_frame.get("Close"), errors="coerce").dropna()
    volume_series = pd.to_numeric(ticker_frame.get("Volume"), errors="coerce").dropna()
    if close_series.empty:
        return empty_price_metrics()

    trailing_close_21 = get_trailing_reference(close_series, periods_back=21)
    trailing_close_252 = get_trailing_reference(close_series, periods_back=252)
    close_60d = close_series.tail(60)
    close_1y = close_series.tail(252)

    ret_1m = ((float(close_series.iloc[-1]) / trailing_close_21) - 1.0) if trailing_close_21 else None
    ret_12m = ((float(close_series.iloc[-1]) / trailing_close_252) - 1.0) if trailing_close_252 else None

    volatility_60d = None
    if len(close_60d) >= 20:
        vol_returns = close_60d.pct_change().dropna()
        if not vol_returns.empty:
            volatility_60d = float(vol_returns.std() * math.sqrt(252))

    max_drawdown_1y = None
    if len(close_1y) >= 20:
        running_max = close_1y.cummax()
        drawdowns = close_1y / running_max - 1.0
        if not drawdowns.empty:
            max_drawdown_1y = abs(float(drawdowns.min()))

    avg_volume_20d = float(volume_series.tail(20).mean()) if not volume_series.empty else None

    return {
        "price": float(close_series.iloc[-1]),
        "ret_1m": ret_1m,
        "ret_12m": ret_12m,
        "volatility_60d": volatility_60d,
        "max_drawdown_1y": max_drawdown_1y,
        "avg_volume_20d": avg_volume_20d,
        "available_price_days": int(len(close_series)),
        "has_full_12m_history": bool(len(close_series) >= 252),
    }


def empty_price_metrics() -> dict[str, float | None]:
    """Return an all-missing metrics payload for failed downloads."""
    return {
        "price": None,
        "ret_1m": None,
        "ret_12m": None,
        "volatility_60d": None,
        "max_drawdown_1y": None,
        "avg_volume_20d": None,
        "available_price_days": 0,
        "has_full_12m_history": False,
    }


def get_trailing_reference(series: pd.Series, periods_back: int) -> float | None:
    """Safely access a trailing price reference for return calculations."""
    if len(series) <= periods_back:
        return None
    reference = safe_float(series.iloc[-(periods_back + 1)])
    return reference if reference and reference > 0 else None


@lru_cache(maxsize=512)
def fetch_yfinance_fundamentals_cached(ticker: str) -> dict[str, object]:
    """Fetch a minimal set of quote and fundamental fields for one ticker."""
    yf = import_yfinance()
    ticker_obj = yf.Ticker(ticker)
    try:
        with _temporary_proxy_bypass(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            info = ticker_obj.get_info() or {}
    except Exception:
        info = {}

    news_headlines = fetch_yfinance_news_headlines(ticker_obj)
    business_summary = str(info.get("longBusinessSummary") or "").strip()
    source_notes = ["yfinance_quote"]
    if news_headlines:
        source_notes.append("yfinance_news")

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "market": infer_market_from_ticker(ticker),
        "market_cap": safe_float(info.get("marketCap")),
        "trailing_pe": safe_float(info.get("trailingPE")),
        "price_to_book": safe_float(info.get("priceToBook")),
        "roe": safe_float(info.get("returnOnEquity")),
        "business_summary": business_summary,
        "recent_news_headlines": news_headlines,
        "recent_news_summaries": [],
        "qualitative_signals": infer_qualitative_signals(business_summary, news_headlines),
        "source_notes": source_notes,
        "event_flags": [],
    }


def fetch_yfinance_news_headlines(ticker_obj: Any) -> list[str]:
    """Fetch a small list of recent Yahoo Finance news headlines when available."""
    try:
        with _temporary_proxy_bypass(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw_news = getattr(ticker_obj, "news", None)
    except Exception:
        raw_news = None

    headlines: list[str] = []
    if isinstance(raw_news, list):
        for item in raw_news[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if title:
                headlines.append(title)
    return headlines


def infer_qualitative_signals(business_summary: str, news_headlines: list[str]) -> list[str]:
    """Create a few lightweight qualitative tags from free text fields."""
    tags: list[str] = []
    summary_lower = business_summary.lower()
    joined_news = " ".join(news_headlines).lower()

    if any(keyword in summary_lower for keyword in ["platform", "software", "subscription", "cloud"]):
        tags.append("recurring_or_platform_business")
    if any(keyword in summary_lower for keyword in ["semiconductor", "chip", "memory"]):
        tags.append("cyclical_industry_exposure")
    if any(keyword in joined_news for keyword in ["guidance", "outlook", "forecast"]):
        tags.append("management_outlook_in_news")
    if any(keyword in joined_news for keyword in ["lawsuit", "probe", "investigation", "recall"]):
        tags.append("headline_risk_flag")
    return tags


def build_yfinance_feature_frame(universe_df: pd.DataFrame, history_period: str = "2y") -> pd.DataFrame:
    """Build one clean DataFrame with raw real-data fields plus MVP proxy scores."""
    frame = universe_df.copy()
    tickers = frame.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.strip().tolist()
    if not tickers:
        return frame

    price_metrics_by_ticker = download_price_metrics(tickers, history_period=history_period)
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            fundamentals = fetch_yfinance_fundamentals_cached(ticker)
        except Exception:
            fundamentals = {"ticker": ticker, "event_flags": ["yfinance_fundamentals_unavailable"]}
        price_metrics = price_metrics_by_ticker.get(ticker, empty_price_metrics())
        rows.append(build_real_data_row(ticker, fundamentals, price_metrics))

    feature_df = pd.DataFrame(rows)
    merged = frame.merge(feature_df, on="ticker", how="left", suffixes=("", "_yf"))
    for column in list(merged.columns):
        if column.endswith("_yf"):
            base_column = column.removesuffix("_yf")
            if base_column not in merged.columns:
                merged[base_column] = merged[column]
            else:
                merged[base_column] = merged[base_column].fillna(merged[column])

    merged = merged[[column for column in merged.columns if not column.endswith("_yf")]]
    for column in REAL_DATA_REQUIRED_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged


def build_real_data_row(
    ticker: str,
    fundamentals: dict[str, object],
    price_metrics: dict[str, float | None],
) -> dict[str, object]:
    """Convert raw yfinance fields into both required outputs and MVP factor proxies."""
    sector = fundamentals.get("sector")
    trailing_pe = safe_float(fundamentals.get("trailing_pe"))
    price_to_book = safe_float(fundamentals.get("price_to_book"))
    roe = safe_float(fundamentals.get("roe"))
    ret_1m = safe_float(price_metrics.get("ret_1m"))
    ret_12m = safe_float(price_metrics.get("ret_12m"))
    avg_volume_20d = safe_float(price_metrics.get("avg_volume_20d"))
    volatility_60d = safe_float(price_metrics.get("volatility_60d"))
    max_drawdown_1y = safe_float(price_metrics.get("max_drawdown_1y"))
    available_price_days = int(price_metrics.get("available_price_days") or 0)
    has_full_12m_history = bool(price_metrics.get("has_full_12m_history", False))

    event_flags = list(fundamentals.get("event_flags", []))
    if price_metrics.get("price") is None:
        event_flags.append("missing_price_history")
    if sector in (None, "", "Unknown"):
        event_flags.append("missing_sector")

    row = {
        "ticker": ticker,
        "name": fundamentals.get("name"),
        "sector": sector,
        "market": fundamentals.get("market") or infer_market_from_ticker(ticker),
        "price": safe_float(price_metrics.get("price")),
        "ret_1m": ret_1m,
        "ret_12m": ret_12m,
        "volatility_60d": volatility_60d,
        "max_drawdown_1y": max_drawdown_1y,
        "avg_volume_20d": avg_volume_20d,
        "market_cap": safe_float(fundamentals.get("market_cap")),
        "trailing_pe": trailing_pe,
        "price_to_book": price_to_book,
        "roe": roe,
        "raw_roe": roe,
        "business_summary": fundamentals.get("business_summary", ""),
        "recent_news_headlines": list(fundamentals.get("recent_news_headlines", [])),
        "recent_news_summaries": list(fundamentals.get("recent_news_summaries", [])),
        "qualitative_signals": list(fundamentals.get("qualitative_signals", [])),
        "source_notes": list(fundamentals.get("source_notes", [])),
        "available_price_days": available_price_days,
        "has_full_12m_history": has_full_12m_history,
        "sector_risk": infer_sector_risk(sector),
        "event_flags": event_flags,
    }
    cleaned = validate_financial_features(row, get_financial_data_quality_config())

    cleaned_trailing_pe = cleaned.get("cleaned_trailing_pe")
    cleaned_price_to_book = cleaned.get("cleaned_price_to_book")
    cleaned_roe = cleaned.get("cleaned_roe")
    cleaned_ret_1m = cleaned.get("cleaned_ret_1m")
    cleaned_ret_12m = cleaned.get("cleaned_ret_12m")
    cleaned_volume = cleaned.get("cleaned_avg_volume_20d")
    cleaned_market_cap = cleaned.get("cleaned_market_cap")

    value_score = average_available(
        [
            scaled_inverse_score(cleaned_trailing_pe, low=8.0, high=35.0),
            scaled_inverse_score(cleaned_price_to_book, low=0.8, high=6.0),
        ],
        default=0.50,
    )
    quality_score = cleaned.get("quality_score_cleaned")
    if quality_score is None:
        quality_score = scaled_score(cleaned_roe, low=0.03, high=0.25)
    growth_score = scaled_score(cleaned_ret_12m, low=-0.20, high=0.40)
    momentum_score = average_available(
        [
            scaled_score(cleaned_ret_1m, low=-0.15, high=0.15),
            scaled_score(cleaned_ret_12m, low=-0.30, high=0.60),
        ],
        default=0.50,
    )
    liquidity_score = average_available(
        [
            scaled_score(cleaned_volume, low=100_000, high=50_000_000),
            scaled_score(cleaned_market_cap, low=1_000_000_000, high=500_000_000_000),
        ],
        default=0.50,
    )

    cleaned.update(
        {
            "trailing_pe": cleaned_trailing_pe,
            "price_to_book": cleaned_price_to_book,
            "roe": cleaned_roe,
            "ret_1m": cleaned_ret_1m,
            "ret_12m": cleaned_ret_12m,
            "avg_volume_20d": cleaned_volume,
            "market_cap": cleaned_market_cap,
            "value_score": value_score,
            "quality_score": quality_score,
            "growth_score": growth_score,
            "momentum_score": momentum_score,
            "liquidity_score": liquidity_score,
            "volatility_proxy": scaled_score(volatility_60d, low=0.10, high=0.60),
            "drawdown_proxy": bounded_or_default(max_drawdown_1y, default=0.50, lower=0.0, upper=1.0),
        }
    )
    return cleaned
