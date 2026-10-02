"""Provider adapters for universe loading and stock-level feature enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Protocol
import math

import pandas as pd

from stock_picker.providers.real_data_support import (
    build_data_validation_summary,
    build_yfinance_feature_frame,
    get_predefined_universe_files,
    REAL_DATA_REQUIRED_COLUMNS,
)
from stock_picker.data_providers.feature_builder import (
    build_features_from_price_history,
    build_meaningful_run_check,
)
from stock_picker.data_providers.krx_provider import (
    default_history_window as default_krx_history_window,
    fetch_krx_price_history,
)
from stock_picker.data_providers.snapshot_io import (
    load_feature_snapshot,
    save_feature_snapshot,
    save_price_history_snapshot,
)
from stock_picker.data_providers.us_yfinance_provider import (
    fetch_us_price_history_and_fundamentals,
)
from stock_picker.export_utils import PROJECT_ROOT
from stock_picker.state import StockPickerState
from stock_picker.universe_loader import load_universe_csv


class UniverseProvider(Protocol):
    """Load a stock universe into a DataFrame."""

    def get_universe(self) -> pd.DataFrame: ...


class FeatureProvider(Protocol):
    """Enrich a universe DataFrame with structured stock-level fields."""

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame: ...


def build_mock_universe_df() -> pd.DataFrame:
    """Small built-in dataset so the prototype runs without external inputs."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "name": "Alpha Systems",
                "sector": "Technology",
                "market": "KOSPI",
                "business_summary": "Enterprise software and automation platform provider.",
                "recent_news_headlines": ["Alpha Systems expands enterprise automation partnerships."],
                "qualitative_signals": ["platform_business"],
                "value_score": 0.72,
                "quality_score": 0.91,
                "momentum_score": 0.84,
                "liquidity_score": 0.88,
                "volatility_proxy": 0.32,
                "drawdown_proxy": 0.24,
                "sector_risk": "medium",
                "event_flags": [],
            },
            {
                "ticker": "BBB",
                "name": "Beta Holdings",
                "sector": "Industrials",
                "market": "KOSPI",
                "business_summary": "Capital goods maker with cyclical industrial demand exposure.",
                "recent_news_headlines": ["Beta Holdings prepares for upcoming earnings update."],
                "qualitative_signals": ["cyclical_exposure"],
                "value_score": 0.81,
                "quality_score": 0.67,
                "momentum_score": 0.61,
                "liquidity_score": 0.79,
                "volatility_proxy": 0.41,
                "drawdown_proxy": 0.34,
                "sector_risk": "medium",
                "event_flags": ["earnings_soon"],
            },
            {
                "ticker": "CCC",
                "name": "Core Retail",
                "sector": "Consumer",
                "market": "KOSDAQ",
                "business_summary": "Retail operator with consumer demand sensitivity.",
                "recent_news_headlines": ["Core Retail reports resilient same-store sales momentum."],
                "qualitative_signals": ["consumer_demand_sensitive"],
                "value_score": 0.77,
                "quality_score": 0.74,
                "momentum_score": 0.83,
                "liquidity_score": 0.71,
                "volatility_proxy": 0.46,
                "drawdown_proxy": 0.38,
                "sector_risk": "medium",
                "event_flags": [],
            },
            {
                "ticker": "DDD",
                "name": "Delta Energy",
                "sector": "Energy",
                "market": "KOSPI",
                "business_summary": "Energy producer exposed to commodity price swings.",
                "recent_news_headlines": ["Delta Energy faces volatile commodity price backdrop."],
                "qualitative_signals": ["commodity_exposure"],
                "value_score": 0.86,
                "quality_score": 0.58,
                "momentum_score": 0.55,
                "liquidity_score": 0.63,
                "volatility_proxy": 0.58,
                "drawdown_proxy": 0.49,
                "sector_risk": "high",
                "event_flags": ["commodity_exposure"],
            },
            {
                "ticker": "EEE",
                "name": "Ever Health",
                "sector": "Healthcare",
                "market": "KOSDAQ",
                "business_summary": "Healthcare company with relatively defensive demand profile.",
                "recent_news_headlines": ["Ever Health launches a new specialty care initiative."],
                "qualitative_signals": ["defensive_demand_profile"],
                "value_score": 0.64,
                "quality_score": 0.87,
                "momentum_score": 0.73,
                "liquidity_score": 0.76,
                "volatility_proxy": 0.29,
                "drawdown_proxy": 0.20,
                "sector_risk": "low",
                "event_flags": [],
            },
            {
                "ticker": "FFF",
                "name": "Future Chips",
                "sector": "Semiconductors",
                "market": "KOSPI",
                "business_summary": "Semiconductor supplier exposed to memory and product cycles.",
                "recent_news_headlines": ["Future Chips benefits from stronger AI memory demand."],
                "qualitative_signals": ["semiconductor_cycle"],
                "value_score": 0.69,
                "quality_score": 0.93,
                "momentum_score": 0.92,
                "liquidity_score": 0.90,
                "volatility_proxy": 0.35,
                "drawdown_proxy": 0.27,
                "sector_risk": "medium",
                "event_flags": ["product_cycle"],
            },
        ]
    )


@dataclass
class MockUniverseProvider:
    def get_universe(self) -> pd.DataFrame:
        return build_mock_universe_df()


@dataclass
class StateUniverseProvider:
    records: list[dict]

    def get_universe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


@dataclass
class CsvUniverseProvider:
    path: str | Path

    def get_universe(self) -> pd.DataFrame:
        return load_universe_csv(_resolve_project_path(self.path))


@dataclass
class PredefinedUniverseProvider:
    """Load a local snapshot for a named S&P500/KOSPI200 research universe."""

    universe_name: str

    def get_universe(self) -> pd.DataFrame:
        file_paths = get_predefined_universe_files(self.universe_name)
        if not file_paths:
            raise ValueError(f"Unsupported predefined universe '{self.universe_name}'.")

        frames = [load_universe_csv(path) for path in file_paths]
        combined = pd.concat(frames, ignore_index=True)
        return combined.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)


@dataclass
class MockFeatureProvider:
    """Ensure the mock dataset has the full MVP field set."""

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        enriched = universe_df.copy()
        defaults = {
            "business_summary": "",
            "recent_news_headlines": "",
            "recent_news_summaries": "",
            "qualitative_signals": "",
            "source_notes": "",
            "value_score": 0.50,
            "quality_score": 0.50,
            "momentum_score": 0.50,
            "liquidity_score": 0.50,
            "volatility_proxy": 0.35,
            "drawdown_proxy": 0.25,
            "sector_risk": "medium",
            "event_flags": "",
        }
        for column, default in defaults.items():
            if column not in enriched.columns:
                enriched[column] = default

        for list_column in ["recent_news_headlines", "recent_news_summaries", "qualitative_signals", "source_notes"]:
            enriched[list_column] = enriched[list_column].apply(_normalize_event_flags_cell)
        enriched["event_flags"] = enriched["event_flags"].apply(_normalize_event_flags_cell)
        return enriched


@dataclass
class PassThroughFeatureProvider:
    """Use existing structured fields as-is."""

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        enriched = universe_df.copy()
        defaults = {
            "business_summary": "",
            "recent_news_headlines": "",
            "recent_news_summaries": "",
            "qualitative_signals": "",
            "source_notes": "",
            "value_score": 0.50,
            "quality_score": 0.50,
            "growth_score": 0.50,
            "momentum_score": 0.50,
            "liquidity_score": 0.50,
            "volatility_proxy": 0.50,
            "drawdown_proxy": 0.50,
            "sector_risk": "medium",
            "event_flags": "",
        }
        for column, default in defaults.items():
            if column not in enriched.columns:
                enriched[column] = default
        for list_column in ["recent_news_headlines", "recent_news_summaries", "qualitative_signals", "source_notes"]:
            if list_column in enriched.columns:
                enriched[list_column] = enriched[list_column].apply(_normalize_event_flags_cell)
        if "event_flags" in enriched.columns:
            enriched["event_flags"] = enriched["event_flags"].apply(_normalize_event_flags_cell)
        return enriched


@dataclass
class CsvFeatureProvider:
    """Optional real-data adapter: merge structured fields from a local CSV."""

    path: str | Path
    join_key: str = "ticker"

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = load_universe_csv(self.path)
        if self.join_key not in universe_df.columns:
            raise ValueError(f"Universe is missing join key '{self.join_key}' for feature merge.")
        if self.join_key not in feature_df.columns:
            raise ValueError(f"Feature CSV is missing join key '{self.join_key}'.")

        merged = universe_df.merge(feature_df, on=self.join_key, how="left", suffixes=("", "_feature"))
        for column in merged.columns:
            if column.endswith("_feature"):
                base_column = column.removesuffix("_feature")
                merged[base_column] = merged[base_column].fillna(merged[column])
        merged = merged[[column for column in merged.columns if not column.endswith("_feature")]]

        for list_column in ["recent_news_headlines", "recent_news_summaries", "qualitative_signals", "source_notes"]:
            if list_column in merged.columns:
                merged[list_column] = merged[list_column].apply(_normalize_event_flags_cell)
        if "event_flags" in merged.columns:
            merged["event_flags"] = merged["event_flags"].apply(_normalize_event_flags_cell)
        return merged


@dataclass
class SnapshotFeatureProvider:
    """Load standardized quantitative features from a local CSV/parquet snapshot."""

    path: str | Path
    join_key: str = "ticker"

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = load_feature_snapshot(self.path)
        if feature_df.empty:
            return PassThroughFeatureProvider().enrich(universe_df)
        merged = _merge_feature_frame(universe_df, feature_df, self.join_key)
        return PassThroughFeatureProvider().enrich(merged)


@dataclass
class RealDataFeatureProvider:
    """Build feature snapshots from pykrx/yfinance when explicitly enabled."""

    run_metadata: dict[str, object]

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        if universe_df.empty:
            return PassThroughFeatureProvider().enrich(universe_df)

        start_date, end_date = _resolve_history_window(self.run_metadata)
        as_of_date = str(self.run_metadata.get("as_of_date") or end_date)
        kr_tickers, us_tickers = _split_market_tickers(universe_df)
        price_frames: list[pd.DataFrame] = []
        fundamental_frames: list[pd.DataFrame] = []
        provider_notes: list[str] = []

        if bool(self.run_metadata.get("krx_enabled", False)) and kr_tickers:
            kr_prices, kr_status = fetch_krx_price_history(
                kr_tickers,
                start_date=start_date,
                end_date=end_date,
            )
            price_frames.append(kr_prices)
            provider_notes.append(f"krx={kr_status.get('status', 'unknown')}:{kr_status.get('reason', '')}")

        if bool(self.run_metadata.get("yfinance_enabled", False)) and us_tickers:
            us_prices, us_fundamentals, us_status = fetch_us_price_history_and_fundamentals(
                us_tickers,
                start_date=start_date,
                end_date=end_date,
            )
            price_frames.append(us_prices)
            fundamental_frames.append(us_fundamentals)
            provider_notes.append(f"yfinance={us_status.get('status', 'unknown')}:{us_status.get('reason', '')}")

        price_history = _concat_non_empty(price_frames)
        fundamentals = _concat_non_empty(fundamental_frames)
        if price_history.empty:
            snapshot_path = self.run_metadata.get("feature_snapshot_path")
            snapshot_frame = load_feature_snapshot(snapshot_path) if snapshot_path else pd.DataFrame()
            provider_note = "; ".join(provider_notes) or "real data providers skipped"
            if not snapshot_frame.empty:
                self.run_metadata["real_data_provider_note"] = f"{provider_note}; fallback=snapshot_feature_provider"
                merged = _merge_feature_frame(universe_df, snapshot_frame, "ticker")
                return PassThroughFeatureProvider().enrich(merged)
            self.run_metadata["real_data_provider_note"] = f"{provider_note}; fallback=pass_through"
            return PassThroughFeatureProvider().enrich(universe_df)

        features = build_features_from_price_history(
            price_history,
            universe=universe_df,
            fundamentals=fundamentals,
            as_of_date=as_of_date,
            feature_source="pykrx_yfinance_snapshot",
        )
        if bool(self.run_metadata.get("save_feature_snapshot", True)):
            save_feature_snapshot(features, self.run_metadata.get("feature_snapshot_path", "data/features/latest_features.csv"))
        if bool(self.run_metadata.get("save_price_history", True)):
            save_price_history_snapshot(
                price_history,
                self.run_metadata.get("price_history_source_path", "data/price_history/price_history.csv"),
            )
        self.run_metadata["real_data_provider_note"] = "; ".join(provider_notes) or "real data providers completed"
        merged = _merge_feature_frame(universe_df, features, "ticker")
        return PassThroughFeatureProvider().enrich(merged)


@dataclass
class YFinanceFeatureProvider:
    """Minimal real-data adapter using yfinance for the MVP field set."""

    history_period: str = "2y"
    risk_free_defaults: dict[str, object] = field(
        default_factory=lambda: {
            "value_score": 0.50,
            "quality_score": 0.50,
            "growth_score": 0.50,
            "momentum_score": 0.50,
            "liquidity_score": 0.50,
            "volatility_proxy": 0.50,
            "drawdown_proxy": 0.50,
            "sector_risk": "medium",
            "event_flags": [],
        }
    )

    def enrich(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        try:
            merged = build_yfinance_feature_frame(universe_df, history_period=self.history_period)
        except Exception:
            merged = universe_df.copy()

        if merged.empty:
            return MockFeatureProvider().enrich(merged)

        for column in REAL_DATA_REQUIRED_COLUMNS:
            if column not in merged.columns:
                merged[column] = pd.NA

        for column, default in self.risk_free_defaults.items():
            if column not in merged.columns:
                merged[column] = default
            if isinstance(default, list):
                merged[column] = merged[column].apply(_normalize_event_flags_cell)
            elif isinstance(default, str):
                merged[column] = merged[column].fillna(default)
            else:
                merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(float(default))

        for list_column in ["recent_news_headlines", "recent_news_summaries", "qualitative_signals", "source_notes"]:
            if list_column in merged.columns:
                merged[list_column] = merged[list_column].apply(_normalize_event_flags_cell)
        if "event_flags" in merged.columns:
            merged["event_flags"] = merged["event_flags"].apply(_normalize_event_flags_cell)
        return merged

    def _safe_fetch_features(self, ticker: str) -> dict[str, object]:
        try:
            return _fetch_yfinance_features_cached(ticker, self.history_period)
        except Exception:
            return {"ticker": ticker, "event_flags": ["yfinance_data_unavailable"]}


def resolve_universe_provider(state: StockPickerState) -> tuple[UniverseProvider, str]:
    run_metadata = state.get("run_metadata", {})
    data_mode = str(run_metadata.get("data_mode", "")).lower()
    universe_records = state.get("universe", [])
    universe_source = run_metadata.get("universe_source")
    universe_name = str(run_metadata.get("universe_name", "")).lower()

    if data_mode == "mock":
        return MockUniverseProvider(), "mock_provider"
    if get_predefined_universe_files(universe_name):
        return PredefinedUniverseProvider(universe_name), f"predefined_provider:{universe_name}"
    if universe_records:
        return StateUniverseProvider(universe_records), "state_provider"
    if universe_source:
        return CsvUniverseProvider(universe_source), f"csv_provider:{universe_source}"
    return MockUniverseProvider(), "mock_provider"


def resolve_feature_provider(state: StockPickerState) -> tuple[FeatureProvider, str]:
    run_metadata = state.get("run_metadata", {})
    data_mode = str(run_metadata.get("data_mode", "")).lower()
    features_source = run_metadata.get("features_source")
    snapshot_path = run_metadata.get("feature_snapshot_path")

    if data_mode == "mock":
        return MockFeatureProvider(), "mock_feature_provider"
    if bool(run_metadata.get("real_data_enabled", False)):
        return RealDataFeatureProvider(run_metadata), "real_data_feature_provider:pykrx_yfinance"
    if snapshot_path and _resolve_project_path(snapshot_path).exists():
        return SnapshotFeatureProvider(_resolve_project_path(snapshot_path)), f"snapshot_feature_provider:{snapshot_path}"
    if data_mode == "yfinance":
        return YFinanceFeatureProvider(), "yfinance_feature_provider"
    if str(features_source).lower() == "yfinance":
        return YFinanceFeatureProvider(), "yfinance_feature_provider"
    if features_source:
        return CsvFeatureProvider(_resolve_project_path(features_source)), f"csv_feature_provider:{features_source}"
    return PassThroughFeatureProvider(), "pass_through_feature_provider"


def load_candidate_frame(state: StockPickerState) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and enrich candidate data through provider adapters."""
    universe_provider, universe_provider_label = resolve_universe_provider(state)
    feature_provider, feature_provider_label = resolve_feature_provider(state)
    run_metadata = state.get("run_metadata", {})

    universe_df = universe_provider.get_universe()
    candidate_df = feature_provider.enrich(universe_df)
    validation_summary = build_data_validation_summary(candidate_df)
    meaningful_run_check = build_meaningful_run_check(
        candidate_df,
        news_covered_count=_count_news_covered(candidate_df),
        backtest_status="pending",
    )
    data_context = {
        "universe_name": run_metadata.get("universe_name", run_metadata.get("universe_source", "configured input")),
        "universe_markets": _summarize_market_counts(universe_df),
        "history_period": getattr(feature_provider, "history_period", None),
        "feature_windows": {
            "price": "latest close",
            "ret_1m": "approximately 21 trading days",
            "ret_12m": "approximately 252 trading days",
            "volatility_60d": "approximately 60 trading days",
            "max_drawdown_1y": "approximately 252 trading days",
            "avg_volume_20d": "approximately 20 trading days",
        },
        "raw_fields": [
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
            "business_summary",
            "recent_news_headlines",
            "qualitative_signals",
        ],
        "feature_snapshot_path": run_metadata.get("feature_snapshot_path"),
        "price_history_path": run_metadata.get("price_history_source_path"),
        "real_data_provider_note": run_metadata.get("real_data_provider_note", ""),
        "meaningful_run_check": meaningful_run_check,
    }
    return candidate_df, {
        "universe_provider": universe_provider_label,
        "feature_provider": feature_provider_label,
        "validation_summary": validation_summary,
        "data_context": data_context,
    }


def _merge_feature_frame(universe_df: pd.DataFrame, feature_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    if join_key not in universe_df.columns or join_key not in feature_df.columns:
        return universe_df.copy()
    merged = universe_df.merge(feature_df, on=join_key, how="left", suffixes=("", "_feature"))
    for column in list(merged.columns):
        if not column.endswith("_feature"):
            continue
        base_column = column.removesuffix("_feature")
        if base_column in merged.columns:
            merged[base_column] = merged[column].combine_first(merged[base_column])
        else:
            merged[base_column] = merged[column]
    return merged[[column for column in merged.columns if not column.endswith("_feature")]]


def _resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_history_window(run_metadata: dict[str, object]) -> tuple[str, str]:
    end_date = str(run_metadata.get("history_end_date") or run_metadata.get("as_of_date") or "").strip() or None
    start_date = str(run_metadata.get("history_start_date") or "").strip() or None
    default_start, default_end = default_krx_history_window(end_date=end_date)
    return start_date or default_start, end_date or default_end


def _split_market_tickers(universe_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    kr_tickers: list[str] = []
    us_tickers: list[str] = []
    for row in universe_df.to_dict(orient="records"):
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        market = str(row.get("market", "")).upper()
        if market.startswith("KOS") or market in {"KR", "KOREA"} or ticker.endswith((".KS", ".KQ")):
            kr_tickers.append(ticker)
        else:
            us_tickers.append(ticker)
    return kr_tickers, us_tickers


def _concat_non_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _count_news_covered(candidate_df: pd.DataFrame) -> int:
    if "recent_news_headlines" not in candidate_df.columns:
        return 0
    return int(candidate_df["recent_news_headlines"].apply(lambda value: bool(_normalize_event_flags_cell(value))).sum())


def _summarize_market_counts(universe_df: pd.DataFrame) -> dict[str, int]:
    if "market" not in universe_df.columns:
        return {}
    counts = (
        universe_df["market"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .sort_index()
    )
    return {str(index): int(value) for index, value in counts.items()}


def _normalize_event_flags_cell(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, "", float("nan")):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


@lru_cache(maxsize=256)
def _fetch_yfinance_features_cached(ticker: str, history_period: str) -> dict[str, object]:
    yf = _import_yfinance()
    ticker_obj = yf.Ticker(ticker)

    info = {}
    try:
        info = ticker_obj.get_info() or {}
    except Exception:
        info = {}

    history = pd.DataFrame()
    try:
        history = ticker_obj.history(period=history_period, auto_adjust=True)
    except Exception:
        history = pd.DataFrame()

    close_series = pd.to_numeric(history.get("Close"), errors="coerce").dropna() if "Close" in history else pd.Series(dtype=float)
    volume_series = pd.to_numeric(history.get("Volume"), errors="coerce").dropna() if "Volume" in history else pd.Series(dtype=float)

    trailing_pe = _safe_float(info.get("trailingPE"))
    roe = _safe_float(info.get("returnOnEquity"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    average_volume = _safe_float(info.get("averageVolume"))
    sector = info.get("sector") or None

    value_score = _scaled_inverse_score(trailing_pe, low=8.0, high=35.0)
    quality_score = _scaled_score(roe, low=0.05, high=0.25)
    growth_proxy = max(revenue_growth if revenue_growth is not None else float("nan"), earnings_growth if earnings_growth is not None else float("nan"))
    growth_score = _scaled_score(growth_proxy, low=0.00, high=0.20)

    momentum_return = None
    volatility_proxy = None
    drawdown_proxy = None
    if len(close_series) >= 20:
        first_close = float(close_series.iloc[0])
        last_close = float(close_series.iloc[-1])
        if first_close > 0:
            momentum_return = (last_close / first_close) - 1.0
        returns = close_series.pct_change().dropna()
        if not returns.empty:
            volatility_proxy = float(returns.std() * math.sqrt(252))
        running_max = close_series.cummax()
        drawdowns = close_series / running_max - 1.0
        if not drawdowns.empty:
            drawdown_proxy = abs(float(drawdowns.min()))

    momentum_score = _scaled_score(momentum_return, low=-0.20, high=0.30)
    liquidity_score = _scaled_score(
        average_volume if average_volume is not None else volume_series.mean() if not volume_series.empty else None,
        low=100_000,
        high=5_000_000,
    )

    event_flags = []
    if _safe_float(info.get("earningsTimestamp")) is not None or _safe_float(info.get("earningsTimestampStart")) is not None:
        event_flags.append("earnings_window")
    if sector is None:
        event_flags.append("missing_sector")
    if close_series.empty:
        event_flags.append("missing_price_history")

    return {
        "ticker": ticker,
        "sector": sector,
        "value_score": value_score,
        "quality_score": quality_score,
        "growth_score": growth_score,
        "momentum_score": momentum_score,
        "liquidity_score": liquidity_score,
        "volatility_proxy": _bounded_or_default(volatility_proxy, default=0.50, lower=0.0, upper=1.0),
        "drawdown_proxy": _bounded_or_default(drawdown_proxy, default=0.50, lower=0.0, upper=1.0),
        "sector_risk": _infer_sector_risk(sector),
        "event_flags": event_flags,
    }


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Install it or switch back to mock mode.") from exc
    return yf


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    except Exception:
        return None


def _scaled_score(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.50
    if high <= low:
        return 0.50
    clipped = min(max(value, low), high)
    return round((clipped - low) / (high - low), 4)


def _scaled_inverse_score(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.50
    if high <= low:
        return 0.50
    clipped = min(max(value, low), high)
    return round(1.0 - ((clipped - low) / (high - low)), 4)


def _bounded_or_default(value: float | None, default: float, lower: float, upper: float) -> float:
    if value is None:
        return default
    return round(min(max(value, lower), upper), 4)


def _infer_sector_risk(sector: object) -> str:
    normalized = str(sector or "").strip().lower()
    if normalized in {"energy", "materials", "real estate"}:
        return "high"
    if normalized in {"healthcare", "utilities", "consumer staples"}:
        return "low"
    return "medium"


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False
