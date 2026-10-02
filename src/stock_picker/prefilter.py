"""Simple transparent quant prefilter used by the MVP data agent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_picker.config import FINANCIAL_DATA_QUALITY_CONFIG


DEFAULT_FACTOR_WEIGHTS = {
    "value_score": 0.40,
    "quality_score": 0.30,
    "momentum_score": 0.30,
}

LIQUIDITY_COLUMN = "liquidity_score"
DEFAULT_TOP_N = 10
DEFAULT_LIQUIDITY_FLOOR = 0.20
DEFAULT_LIQUIDITY_PENALTY_THRESHOLD = 0.40
DEFAULT_LIQUIDITY_PENALTY = 0.05
DEFAULT_DATA_QUALITY_OUTLIER_PENALTY = float(FINANCIAL_DATA_QUALITY_CONFIG["outlier_penalty"])


def run_quant_prefilter(
    universe_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    factor_weights: dict[str, float] | None = None,
    liquidity_floor: float = DEFAULT_LIQUIDITY_FLOOR,
    liquidity_penalty_threshold: float = DEFAULT_LIQUIDITY_PENALTY_THRESHOLD,
    liquidity_penalty: float = DEFAULT_LIQUIDITY_PENALTY,
    data_quality_outlier_penalty: float = DEFAULT_DATA_QUALITY_OUTLIER_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the universe with a transparent percentile-based composite score.

    Returns:
        ranked_candidates_df: top-N candidate rows sorted by quant_score descending
        breakdown_df: full-universe score breakdown for inspection and debugging
    """
    if universe_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    weights = factor_weights or DEFAULT_FACTOR_WEIGHTS
    _validate_prefilter_inputs(universe_df)

    breakdown_df = _build_prefilter_factor_frame(universe_df)
    for factor in weights:
        breakdown_df[f"{factor}_pct"] = pd.to_numeric(breakdown_df[factor], errors="coerce")

    breakdown_df[f"{LIQUIDITY_COLUMN}_pct"] = pd.to_numeric(
        breakdown_df[LIQUIDITY_COLUMN],
        errors="coerce",
    ).fillna(0.0)
    breakdown_df["liquidity_pass"] = breakdown_df[f"{LIQUIDITY_COLUMN}_pct"] >= liquidity_floor
    breakdown_df["liquidity_penalty"] = (
        breakdown_df[f"{LIQUIDITY_COLUMN}_pct"] < liquidity_penalty_threshold
    ).astype(float) * liquidity_penalty
    breakdown_df["data_quality_outlier_penalty"] = (
        breakdown_df.get("roe_is_extreme_outlier", False).astype(float)
        if "roe_is_extreme_outlier" in breakdown_df.columns
        else 0.0
    ) * data_quality_outlier_penalty
    if "has_full_12m_history" in breakdown_df.columns:
        breakdown_df["data_quality_outlier_penalty"] = (
            breakdown_df["data_quality_outlier_penalty"]
            + (~breakdown_df["has_full_12m_history"].fillna(False)).astype(float) * data_quality_outlier_penalty
        )
    if "data_quality_flags" in breakdown_df.columns:
        momentum_outlier_mask = breakdown_df["data_quality_flags"].apply(
            lambda flags: isinstance(flags, list) and any(
                flag in {"momentum_12m_outlier", "momentum_1m_outlier"} for flag in flags
            )
        )
        breakdown_df["data_quality_outlier_penalty"] = (
            breakdown_df["data_quality_outlier_penalty"] + momentum_outlier_mask.astype(float) * data_quality_outlier_penalty
        )

    weighted_component_columns = []
    for factor, weight in weights.items():
        component_col = f"{factor}_component"
        available_col = f"{factor}_available_weight"
        breakdown_df[available_col] = breakdown_df[f"{factor}_pct"].notna().astype(float) * weight
        breakdown_df[component_col] = breakdown_df[f"{factor}_pct"].fillna(0.0) * weight
        weighted_component_columns.append(component_col)

    available_weight_columns = [f"{factor}_available_weight" for factor in weights]
    breakdown_df["effective_factor_weight"] = breakdown_df[available_weight_columns].sum(axis=1)
    breakdown_df["base_composite_score"] = np.where(
        breakdown_df["effective_factor_weight"] > 0,
        breakdown_df[weighted_component_columns].sum(axis=1) / breakdown_df["effective_factor_weight"],
        0.0,
    ).round(4)
    breakdown_df["quant_score"] = (
        breakdown_df["base_composite_score"] - breakdown_df["liquidity_penalty"] - breakdown_df["data_quality_outlier_penalty"]
    ).round(4)

    eligible_df = breakdown_df[breakdown_df["liquidity_pass"]].copy()
    ranked_df = eligible_df.sort_values(
        by=["quant_score", f"{LIQUIDITY_COLUMN}_pct", "ticker"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked_df["prefilter_rank"] = ranked_df.index + 1

    top_candidates_df = ranked_df.head(top_n).copy()
    top_candidates_df["liquidity_penalty"] = top_candidates_df["liquidity_penalty"].round(4)

    breakdown_columns = [
        "ticker",
        "quant_score",
        "base_composite_score",
        "effective_factor_weight",
        "liquidity_penalty",
        "data_quality_outlier_penalty",
        "liquidity_pass",
        "prefilter_rank",
    ]
    for factor in weights:
        breakdown_columns.extend(
            [
                factor,
                f"{factor}_pct",
                f"{factor}_component",
            ]
        )
    breakdown_columns.extend([LIQUIDITY_COLUMN, f"{LIQUIDITY_COLUMN}_pct"])

    full_breakdown_df = ranked_df[breakdown_columns].copy()
    return top_candidates_df, full_breakdown_df


def _validate_prefilter_inputs(universe_df: pd.DataFrame) -> None:
    required_columns = {"ticker"}
    missing_columns = sorted(required_columns - set(universe_df.columns))
    if missing_columns:
        raise ValueError(f"Prefilter input is missing required columns: {missing_columns}")


def _build_prefilter_factor_frame(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build factor scores from real-data columns first, then fall back to mock factor columns."""
    breakdown_df = universe_df.copy()
    existing_value_score = breakdown_df.get("value_score")
    existing_quality_score = breakdown_df.get("quality_score")
    existing_momentum_score = breakdown_df.get("momentum_score")
    existing_liquidity_score = breakdown_df.get(LIQUIDITY_COLUMN)

    trailing_pe_pct = _percentile_rank(breakdown_df.get("cleaned_trailing_pe", breakdown_df.get("trailing_pe")), ascending=False)
    price_to_book_pct = _percentile_rank(breakdown_df.get("cleaned_price_to_book", breakdown_df.get("price_to_book")), ascending=False)
    quality_score_cleaned = _series_or_nan(
        breakdown_df,
        "quality_score_cleaned",
    )
    cleaned_roe_pct = _percentile_rank(breakdown_df.get("cleaned_roe", breakdown_df.get("roe")), ascending=True)
    ret_12m_pct = _percentile_rank(breakdown_df.get("cleaned_ret_12m", breakdown_df.get("ret_12m")), ascending=True)
    ret_1m_pct = _percentile_rank(breakdown_df.get("cleaned_ret_1m", breakdown_df.get("ret_1m")), ascending=True)
    avg_volume_pct = _percentile_rank(breakdown_df.get("cleaned_avg_volume_20d", breakdown_df.get("avg_volume_20d")), ascending=True)
    market_cap_pct = _percentile_rank(breakdown_df.get("cleaned_market_cap", breakdown_df.get("market_cap")), ascending=True)

    value_from_raw = _weighted_average_scores(
        [
            (trailing_pe_pct, 0.6),
            (price_to_book_pct, 0.4),
        ]
    )
    quality_from_raw = quality_score_cleaned.combine_first(cleaned_roe_pct)
    momentum_from_raw = _weighted_average_scores(
        [
            (ret_12m_pct, 0.85),
            (ret_1m_pct, 0.15),
        ]
    )
    liquidity_from_raw = _weighted_average_scores(
        [
            (avg_volume_pct, 0.8),
            (market_cap_pct, 0.2),
        ]
    )

    breakdown_df["value_score"] = _coalesce_scores(value_from_raw, existing_value_score)
    breakdown_df["quality_score"] = _coalesce_scores(quality_from_raw, existing_quality_score)
    breakdown_df["momentum_score"] = _coalesce_scores(momentum_from_raw, existing_momentum_score)
    breakdown_df[LIQUIDITY_COLUMN] = _coalesce_scores(liquidity_from_raw, existing_liquidity_score)

    breakdown_df["value_score_source"] = _build_score_source_label(
        preferred=value_from_raw,
        fallback=existing_value_score,
        preferred_label="raw_valuation",
        fallback_label="existing_value_score",
    )
    breakdown_df["quality_score_source"] = _build_score_source_label(
        preferred=quality_from_raw,
        fallback=existing_quality_score,
        preferred_label="raw_roe",
        fallback_label="existing_quality_score",
    )
    breakdown_df["momentum_score_source"] = _build_score_source_label(
        preferred=momentum_from_raw,
        fallback=existing_momentum_score,
        preferred_label="raw_returns",
        fallback_label="existing_momentum_score",
    )
    breakdown_df["liquidity_score_source"] = _build_score_source_label(
        preferred=liquidity_from_raw,
        fallback=existing_liquidity_score,
        preferred_label="raw_liquidity",
        fallback_label="existing_liquidity_score",
    )
    breakdown_df["data_quality_penalty"] = _series_or_nan(
        breakdown_df,
        "data_quality_penalty",
    ).fillna(0.0)
    return breakdown_df


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series for a column or a NaN-filled aligned fallback series."""
    if column not in df.columns:
        return pd.Series(index=df.index, data=np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _percentile_rank(series: pd.Series | None, ascending: bool = True) -> pd.Series:
    """Compute percentile ranks while keeping missing values as missing."""
    if series is None:
        return pd.Series(dtype=float)

    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    if valid.empty:
        return pd.Series(index=numeric_series.index, data=np.nan, dtype=float)
    if len(valid) == 1:
        result = pd.Series(index=numeric_series.index, data=np.nan, dtype=float)
        result.loc[valid.index] = 1.0
        return result

    return numeric_series.rank(method="average", pct=True, ascending=ascending).astype(float)


def _weighted_average_scores(weighted_series: list[tuple[pd.Series, float]]) -> pd.Series:
    """Average only the available score components, renormalizing their weights."""
    if not weighted_series:
        return pd.Series(dtype=float)

    base_index = weighted_series[0][0].index
    weighted_sum = pd.Series(index=base_index, data=0.0, dtype=float)
    available_weight = pd.Series(index=base_index, data=0.0, dtype=float)

    for series, weight in weighted_series:
        aligned = pd.to_numeric(series, errors="coerce").reindex(base_index)
        valid_mask = aligned.notna()
        weighted_sum = weighted_sum.add(aligned.fillna(0.0) * weight, fill_value=0.0)
        available_weight = available_weight.add(valid_mask.astype(float) * weight, fill_value=0.0)

    return weighted_sum.div(available_weight.replace(0.0, np.nan)).round(4)


def _coalesce_scores(preferred: pd.Series, fallback: pd.Series | None) -> pd.Series:
    """Use derived raw-data scores first, then existing precomputed scores if available."""
    if fallback is None:
        return preferred
    fallback_numeric = pd.to_numeric(fallback, errors="coerce")
    return preferred.combine_first(fallback_numeric)


def _build_score_source_label(
    *,
    preferred: pd.Series,
    fallback: pd.Series | None,
    preferred_label: str,
    fallback_label: str,
) -> pd.Series:
    """Track whether each factor came from raw real-data fields or a fallback score."""
    source = pd.Series(index=preferred.index, data="missing", dtype=object)
    source.loc[preferred.notna()] = preferred_label
    if fallback is not None:
        fallback_numeric = pd.to_numeric(fallback, errors="coerce")
        source.loc[preferred.isna() & fallback_numeric.notna()] = fallback_label
    return source


def build_prefilter_breakdown_map(breakdown_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert the breakdown DataFrame into a state-friendly dict keyed by ticker."""
    if breakdown_df.empty:
        return {}

    records = breakdown_df.to_dict(orient="records")
    return {str(record["ticker"]): record for record in records}
