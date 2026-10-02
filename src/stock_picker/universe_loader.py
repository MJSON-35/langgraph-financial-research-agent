from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("ticker",)
OPTIONAL_COLUMNS = ("name", "sector", "market")


def load_universe_csv(path: str | Path) -> pd.DataFrame:
    """Load a local universe CSV and return a clean pandas DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    _validate_universe_frame(df, csv_path)

    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column in df.columns:
            df[column] = _clean_string_series(df[column])

    df = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def _validate_universe_frame(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Universe CSV is missing required columns {missing}: {path}")

    if df.empty:
        raise ValueError(f"Universe CSV is empty: {path}")

    ticker_values = _clean_string_series(df["ticker"])
    if ticker_values.eq("").any():
        raise ValueError(f"Universe CSV contains blank ticker values: {path}")


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned
