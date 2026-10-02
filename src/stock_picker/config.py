"""Central configuration for lightweight prototype components."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PORTFOLIO_MANAGER_CONFIG: dict[str, Any] = {
    "quant_weight": 0.50,
    "equity_weight": 0.20,
    "macro_adjustment_weight": 1.00,
    "risk_penalty_weight": 1.00,
    "sector_concentration_penalty": 0.03,
    "missing_data_penalty": 0.02,
    "top_n": 3,
    "allow_llm_rerank": False,
    "max_same_sector_in_top3": 2,
    "macro_sector_preference_bonus": 0.02,
    "macro_sector_risk_penalty": 0.02,
    "risk_level_penalties": {
        "low": 0.00,
        "medium": 0.03,
        "high": 0.08,
        "unknown": 0.04,
    },
}

FINANCIAL_DATA_QUALITY_CONFIG: dict[str, Any] = {
    "roe_min": -0.50,
    "roe_max": 0.50,
    "roe_extreme_abs_threshold": 1.00,
    "pe_min": 0,
    "pe_max": 80,
    "pb_min": 0,
    "pb_max": 30,
    "momentum_12m_min": -0.90,
    "momentum_12m_max": 3.00,
    "momentum_1m_min": -0.70,
    "momentum_1m_max": 1.00,
    "volume_min": 0,
    "outlier_penalty": 0.03,
    "missing_penalty": 0.02,
    "use_winsorized_roe_for_quality": True,
    "use_quality_composite": True,
}


def get_portfolio_manager_config(run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return portfolio-manager config with optional run-level overrides."""
    resolved = deepcopy(PORTFOLIO_MANAGER_CONFIG)
    overrides = (run_metadata or {}).get("portfolio_manager_config", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key == "risk_level_penalties" and isinstance(value, dict):
                resolved["risk_level_penalties"] = {
                    **resolved.get("risk_level_penalties", {}),
                    **value,
                }
            else:
                resolved[key] = value
    return resolved


def get_financial_data_quality_config(run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return financial-data quality config with optional run-level overrides."""
    resolved = deepcopy(FINANCIAL_DATA_QUALITY_CONFIG)
    overrides = (run_metadata or {}).get("financial_data_quality_config", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            resolved[key] = value
    return resolved
