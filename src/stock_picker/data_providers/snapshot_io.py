"""Snapshot IO for quantitative feature and price-history data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stock_picker.export_utils import PROJECT_ROOT
from stock_picker.data_providers.feature_builder import FEATURE_COLUMNS, PRICE_COLUMNS


def save_feature_snapshot(features: pd.DataFrame, path_value: Any) -> dict[str, Any]:
    return _save_snapshot(features, path_value, FEATURE_COLUMNS)


def save_price_history_snapshot(price_history: pd.DataFrame, path_value: Any) -> dict[str, Any]:
    return _save_snapshot(price_history, path_value, PRICE_COLUMNS)


def load_feature_snapshot(path_value: Any) -> pd.DataFrame:
    return _load_snapshot(path_value, FEATURE_COLUMNS)


def load_price_history_snapshot(path_value: Any) -> pd.DataFrame:
    return _load_snapshot(path_value, PRICE_COLUMNS)


def build_missing_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in frame.columns}


def _save_snapshot(frame: pd.DataFrame, path_value: Any, columns: list[str]) -> dict[str, Any]:
    path = _resolve_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output.to_csv(path, index=False, encoding="utf-8")
    parquet_path = None
    if path.suffix.lower() == ".csv":
        parquet_path = path.with_suffix(".parquet")
        try:
            output.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None
    return {
        "path": str(path),
        "parquet_path": str(parquet_path) if parquet_path else "",
        "rows": int(len(output)),
        "missing_by_column": build_missing_summary(output),
    }


def _load_snapshot(path_value: Any, columns: list[str]) -> pd.DataFrame:
    path = _resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _resolve_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
