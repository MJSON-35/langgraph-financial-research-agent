"""Robust cleaning helpers for free financial and market features."""

from __future__ import annotations

import math
from typing import Any


def is_missing(value: Any) -> bool:
    """Return True when a scalar should be treated as missing or invalid."""
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return isinstance(value, str) and not value.strip()
    return math.isnan(numeric) or math.isinf(numeric)


def cap_value(value: Any, min_value: float, max_value: float) -> tuple[float | None, bool]:
    """Cap a scalar to a closed interval and report whether capping happened."""
    if is_missing(value):
        return None, False
    numeric = float(value)
    capped = min(max(numeric, min_value), max_value)
    return capped, capped != numeric


def clean_roe(raw_roe: Any, config: dict[str, Any]) -> tuple[float | None, list[str]]:
    """Clean ROE in decimal form and flag missing/extreme/winsorized cases."""
    flags: list[str] = []
    if is_missing(raw_roe):
        return None, ["roe_missing"]

    numeric = float(raw_roe)
    if abs(numeric) > float(config["roe_extreme_abs_threshold"]):
        flags.append("roe_extreme_outlier")

    cleaned_roe, was_capped = cap_value(
        numeric,
        float(config["roe_min"]),
        float(config["roe_max"]),
    )
    if was_capped:
        flags.append("roe_winsorized")
    return cleaned_roe, flags


def build_quality_score(features: dict[str, Any], config: dict[str, Any]) -> tuple[float | None, str]:
    """Build a quality score from cleaned profitability fields, with confidence."""
    components: list[float] = []
    confidence = "high"

    cleaned_roe = features.get("cleaned_roe")
    cleaned_roa = features.get("cleaned_roa")
    cleaned_profit_margin = features.get("cleaned_profit_margin")
    cleaned_operating_margin = features.get("cleaned_operating_margin")
    flags = set(features.get("data_quality_flags", []))

    if not is_missing(cleaned_roe):
        components.append(_scaled_score(float(cleaned_roe), low=0.03, high=0.25))
    if not is_missing(cleaned_roa):
        components.append(_scaled_score(float(cleaned_roa), low=0.01, high=0.15))
    if not is_missing(cleaned_profit_margin):
        components.append(_scaled_score(float(cleaned_profit_margin), low=0.02, high=0.30))
    if not is_missing(cleaned_operating_margin):
        components.append(_scaled_score(float(cleaned_operating_margin), low=0.03, high=0.30))

    if not components:
        return None, "low"

    score = sum(components) / len(components)
    if "roe_extreme_outlier" in flags:
        score -= float(config["outlier_penalty"])
        confidence = "medium"
    if len(components) == 1:
        confidence = "medium" if confidence == "high" else confidence
    if "roe_missing" in flags:
        confidence = "low"

    return round(max(score, 0.0), 4), confidence


def validate_financial_features(stock_record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return cleaned feature fields, flags, warnings, and confidence metadata."""
    cleaned = dict(stock_record)
    flags: list[str] = []

    raw_roe = stock_record.get("raw_roe", stock_record.get("roe"))
    cleaned["raw_roe"] = None if is_missing(raw_roe) else float(raw_roe)
    cleaned_roe, roe_flags = clean_roe(raw_roe, config)
    cleaned["cleaned_roe"] = cleaned_roe
    cleaned["roe_was_winsorized"] = "roe_winsorized" in roe_flags
    cleaned["roe_is_extreme_outlier"] = "roe_extreme_outlier" in roe_flags
    flags.extend(roe_flags)

    raw_pe = stock_record.get("trailing_pe")
    cleaned["raw_trailing_pe"] = None if is_missing(raw_pe) else float(raw_pe)
    if is_missing(raw_pe):
        cleaned["cleaned_trailing_pe"] = None
        flags.append("trailing_pe_missing")
    else:
        pe_capped, pe_was_capped = cap_value(raw_pe, float(config["pe_min"]), float(config["pe_max"]))
        cleaned["cleaned_trailing_pe"] = pe_capped
        if pe_was_capped:
            flags.append("trailing_pe_capped")

    raw_pb = stock_record.get("price_to_book")
    cleaned["raw_price_to_book"] = None if is_missing(raw_pb) else float(raw_pb)
    if is_missing(raw_pb):
        cleaned["cleaned_price_to_book"] = None
        flags.append("price_to_book_missing")
    else:
        pb_capped, pb_was_capped = cap_value(raw_pb, float(config["pb_min"]), float(config["pb_max"]))
        cleaned["cleaned_price_to_book"] = pb_capped
        if pb_was_capped:
            flags.append("price_to_book_capped")

    raw_ret_12m = stock_record.get("ret_12m")
    cleaned["raw_ret_12m"] = None if is_missing(raw_ret_12m) else float(raw_ret_12m)
    if is_missing(raw_ret_12m):
        cleaned["cleaned_ret_12m"] = None
        flags.append("ret_12m_missing")
    else:
        ret12_capped, ret12_was_capped = cap_value(
            raw_ret_12m,
            float(config["momentum_12m_min"]),
            float(config["momentum_12m_max"]),
        )
        cleaned["cleaned_ret_12m"] = ret12_capped
        if ret12_was_capped:
            flags.append("momentum_12m_outlier")

    raw_ret_1m = stock_record.get("ret_1m")
    cleaned["raw_ret_1m"] = None if is_missing(raw_ret_1m) else float(raw_ret_1m)
    if is_missing(raw_ret_1m):
        cleaned["cleaned_ret_1m"] = None
        flags.append("ret_1m_missing")
    else:
        ret1_capped, ret1_was_capped = cap_value(
            raw_ret_1m,
            float(config["momentum_1m_min"]),
            float(config["momentum_1m_max"]),
        )
        cleaned["cleaned_ret_1m"] = ret1_capped
        if ret1_was_capped:
            flags.append("momentum_1m_outlier")

    raw_volume = stock_record.get("avg_volume_20d")
    cleaned["raw_avg_volume_20d"] = None if is_missing(raw_volume) else float(raw_volume)
    if is_missing(raw_volume) or float(raw_volume) <= float(config["volume_min"]):
        cleaned["cleaned_avg_volume_20d"] = None
        flags.append("avg_volume_20d_missing")
    else:
        cleaned["cleaned_avg_volume_20d"] = float(raw_volume)

    raw_market_cap = stock_record.get("market_cap")
    cleaned["raw_market_cap"] = None if is_missing(raw_market_cap) else float(raw_market_cap)
    if is_missing(raw_market_cap) or float(raw_market_cap) <= 0:
        cleaned["cleaned_market_cap"] = None
        flags.append("market_cap_missing")
    else:
        cleaned["cleaned_market_cap"] = float(raw_market_cap)

    available_price_days = int(stock_record.get("available_price_days", 0) or 0)
    cleaned["available_price_days"] = available_price_days
    cleaned["has_full_12m_history"] = bool(stock_record.get("has_full_12m_history", available_price_days >= 252))
    cleaned["momentum_12m_confidence"] = "high" if cleaned["has_full_12m_history"] else "low"
    if not cleaned["has_full_12m_history"]:
        flags.append("insufficient_12m_price_history")

    cleaned["valuation_confidence"] = infer_valuation_confidence(cleaned)
    cleaned_quality_score, quality_confidence = build_quality_score(cleaned, config)
    cleaned["quality_score_cleaned"] = cleaned_quality_score
    cleaned["quality_score_confidence"] = quality_confidence

    flags = _dedupe(flags)
    cleaned["data_quality_flags"] = flags
    cleaned["feature_confidence"] = infer_feature_confidence(cleaned, flags)
    cleaned["data_quality_warning"] = build_data_quality_summary(cleaned)
    cleaned["data_quality_penalty"] = infer_data_quality_penalty(cleaned, config)
    return cleaned


def build_data_quality_summary(stock_record: dict[str, Any]) -> str:
    """Build a concise natural-language data-quality summary for downstream agents."""
    flags = set(stock_record.get("data_quality_flags", []))
    messages: list[str] = []

    if "roe_winsorized" in flags or "roe_extreme_outlier" in flags:
        raw = _fmt_pct(stock_record.get("raw_roe"))
        cleaned = _fmt_pct(stock_record.get("cleaned_roe"))
        messages.append(
            f"ROE was winsorized from {raw} to {cleaned}; quality score should be interpreted with caution."
        )
    elif "roe_missing" in flags:
        messages.append("ROE is missing; profitability confidence is low.")

    valuation_missing = [flag for flag in ["trailing_pe_missing", "price_to_book_missing"] if flag in flags]
    if valuation_missing:
        messages.append("Trailing P/E and/or P/B are missing; valuation confidence is low.")

    if "momentum_12m_outlier" in flags:
        messages.append("12-month momentum was capped due to an extreme value.")
    if "insufficient_12m_price_history" in flags:
        messages.append("12-month return is based on limited price history and should be interpreted cautiously.")
    if "avg_volume_20d_missing" in flags:
        messages.append("Average 20-day volume is missing or invalid; liquidity confidence is low.")

    return " ".join(messages) if messages else "No material data-quality warnings."


def infer_valuation_confidence(features: dict[str, Any]) -> str:
    """Infer confidence in valuation metrics from missing/capped inputs."""
    pe = features.get("cleaned_trailing_pe")
    pb = features.get("cleaned_price_to_book")
    available = int(not is_missing(pe)) + int(not is_missing(pb))
    if available == 2:
        return "high"
    if available == 1:
        return "medium"
    return "low"


def infer_feature_confidence(features: dict[str, Any], flags: list[str]) -> str:
    """Infer one feature confidence label for the candidate."""
    if "insufficient_12m_price_history" in flags:
        return "low"
    if features.get("valuation_confidence") == "low" and features.get("quality_score_confidence") == "low":
        return "low"
    if any(flag in flags for flag in ["roe_extreme_outlier", "momentum_12m_outlier", "momentum_1m_outlier"]):
        return "medium"
    return "high"


def infer_data_quality_penalty(features: dict[str, Any], config: dict[str, Any]) -> float:
    """Convert data-quality issues into a compact penalty used downstream."""
    penalty = 0.0
    flags = set(features.get("data_quality_flags", []))
    if "roe_extreme_outlier" in flags:
        penalty += float(config["outlier_penalty"])
    if {"trailing_pe_missing", "price_to_book_missing"} <= flags:
        penalty += float(config["missing_penalty"])
    if "insufficient_12m_price_history" in flags or "momentum_12m_outlier" in flags:
        penalty += float(config["outlier_penalty"])
    if str(features.get("feature_confidence", "high")).lower() == "low":
        penalty += float(config["missing_penalty"])
    return round(penalty, 4)


def _scaled_score(value: float | None, low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.50
    clipped = min(max(value, low), high)
    return round((clipped - low) / (high - low), 4)


def _fmt_pct(value: Any) -> str:
    if is_missing(value):
        return "Unavailable"
    numeric = float(value)
    return f"{numeric * 100:.2f}%"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
