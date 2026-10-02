"""Provider adapters for loading mock, local, or optional external macro context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

from stock_picker.env_utils import get_env_str, load_project_env
from stock_picker.export_utils import PROJECT_ROOT
from stock_picker.state import StockPickerState


class MacroProvider(Protocol):
    """Load a structured macro context dictionary."""

    def get_macro_context(self) -> dict[str, object]: ...


def build_mock_macro_context() -> dict[str, object]:
    """Small built-in macro context for MVP runs."""
    return normalize_macro_context(
        {
            "growth_signal": "stable",
            "inflation_signal": "cooling",
            "policy_signal": "unchanged",
            "volatility_signal": "moderate",
            "growth": {
                "pmi_level": 51.8,
                "pmi_trend": "stable",
                "employment_trend": "stable",
            },
            "inflation": {
                "headline_cpi_trend": "cooling",
                "core_cpi_trend": "sticky",
            },
            "policy": {
                "rate_direction": "unchanged",
                "yield_curve": "slightly_inverted",
            },
            "market": {
                "volatility_regime": "moderate",
                "usd_trend": "firm",
                "oil_trend": "stable",
            },
        }
    )


def normalize_macro_context(payload: dict[str, object]) -> dict[str, object]:
    """Normalize richer macro inputs into a compact LLM-friendly schema."""
    growth = payload.get("growth", {}) if isinstance(payload.get("growth"), dict) else {}
    inflation = payload.get("inflation", {}) if isinstance(payload.get("inflation"), dict) else {}
    policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
    market = payload.get("market", {}) if isinstance(payload.get("market"), dict) else {}

    normalized = {
        "growth_signal": "stable",
        "inflation_signal": "cooling",
        "policy_signal": "unchanged",
        "volatility_signal": "moderate",
    }
    normalized.update(payload)

    normalized["growth_signal"] = str(
        payload.get("growth_signal")
        or growth.get("pmi_trend")
        or growth.get("growth_signal")
        or "stable"
    ).lower()
    normalized["inflation_signal"] = str(
        payload.get("inflation_signal")
        or inflation.get("headline_cpi_trend")
        or inflation.get("inflation_signal")
        or "cooling"
    ).lower()
    normalized["policy_signal"] = str(
        payload.get("policy_signal")
        or policy.get("rate_direction")
        or policy.get("policy_signal")
        or "unchanged"
    ).lower()
    normalized["volatility_signal"] = str(
        payload.get("volatility_signal")
        or market.get("volatility_regime")
        or market.get("volatility_signal")
        or "moderate"
    ).lower()
    normalized["growth"] = growth
    normalized["inflation"] = inflation
    normalized["policy"] = policy
    normalized["market"] = market
    normalized["macro_feature_summary"] = build_macro_feature_summary(normalized)
    return normalized


def build_macro_feature_summary(payload: dict[str, object]) -> dict[str, object]:
    """Create a small summary block for prompts and reports."""
    growth = payload.get("growth", {}) if isinstance(payload.get("growth"), dict) else {}
    inflation = payload.get("inflation", {}) if isinstance(payload.get("inflation"), dict) else {}
    policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
    market = payload.get("market", {}) if isinstance(payload.get("market"), dict) else {}
    return {
        "growth_signal": payload.get("growth_signal", "stable"),
        "inflation_signal": payload.get("inflation_signal", "cooling"),
        "policy_signal": payload.get("policy_signal", "unchanged"),
        "volatility_signal": payload.get("volatility_signal", "moderate"),
        "pmi_level": growth.get("pmi_level"),
        "employment_trend": growth.get("employment_trend"),
        "core_cpi_trend": inflation.get("core_cpi_trend"),
        "yield_curve": policy.get("yield_curve"),
        "usd_trend": market.get("usd_trend"),
        "oil_trend": market.get("oil_trend"),
    }


@dataclass
class MockMacroProvider:
    def get_macro_context(self) -> dict[str, object]:
        return build_mock_macro_context()


@dataclass
class StateMacroProvider:
    macro_context: dict[str, object]

    def get_macro_context(self) -> dict[str, object]:
        return normalize_macro_context(dict(self.macro_context))


@dataclass
class JsonMacroProvider:
    path: str | Path

    def get_macro_context(self) -> dict[str, object]:
        json_path = Path(self.path)
        if not json_path.is_absolute():
            json_path = (PROJECT_ROOT / json_path).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"Macro context file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Macro context JSON must contain a top-level object.")
        return normalize_macro_context(payload)


@dataclass
class FredMacroProvider:
    """Optional macro adapter using the public FRED observations endpoint."""

    series_map: dict[str, str]

    def get_macro_context(self) -> dict[str, object]:
        load_project_env()
        api_key = get_env_str("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY is not configured in the environment.")
        latest_values = {}
        for signal_name, series_id in self.series_map.items():
            latest_values[signal_name] = _fetch_latest_fred_value(series_id, api_key)
        return _map_fred_values_to_macro_context(latest_values)


def resolve_macro_provider(state: StockPickerState) -> tuple[MacroProvider, str]:
    run_metadata = state.get("run_metadata", {})
    macro_mode = str(run_metadata.get("macro_mode", "")).lower()
    macro_context = run_metadata.get("macro_context")
    macro_context_source = run_metadata.get("macro_context_source")
    fred_series_map = run_metadata.get("fred_series_map")

    if macro_mode == "mock":
        return MockMacroProvider(), "mock_macro_provider"
    if isinstance(macro_context, dict) and macro_context:
        return StateMacroProvider(macro_context), "state_macro_provider"
    if macro_context_source:
        return JsonMacroProvider(macro_context_source), f"json_macro_provider:{macro_context_source}"
    if macro_mode == "fred" and isinstance(fred_series_map, dict) and fred_series_map:
        return FredMacroProvider(fred_series_map), "fred_macro_provider"
    return StateMacroProvider({}), "empty_macro_provider"


def load_macro_context(state: StockPickerState) -> tuple[dict[str, object], dict[str, str]]:
    provider, provider_label = resolve_macro_provider(state)
    try:
        macro_context = normalize_macro_context(provider.get_macro_context())
        return macro_context, {"macro_provider": provider_label, "macro_provider_status": "ok"}
    except Exception as exc:
        return {}, {"macro_provider": provider_label, "macro_provider_status": f"error:{type(exc).__name__}"}


def _fetch_latest_fred_value(series_id: str, api_key: str) -> float | None:
    normalized_series_id = str(series_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized_series_id):
        raise ValueError("FRED series identifiers may contain only letters, numbers, dot, dash, and underscore.")
    query = urlencode(
        {
            "series_id": normalized_series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{query}"
    with urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    observations = payload.get("observations", [])
    if not observations:
        return None
    value = observations[0].get("value")
    if value in (None, ".", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _map_fred_values_to_macro_context(values: dict[str, float | None]) -> dict[str, object]:
    growth_value = values.get("growth")
    inflation_value = values.get("inflation")
    policy_value = values.get("policy_rate")
    volatility_value = values.get("volatility")

    return normalize_macro_context(
        {
        "growth_signal": "weakening" if growth_value is not None and growth_value < 0 else "stable",
        "inflation_signal": "rising" if inflation_value is not None and inflation_value > 3.0 else "cooling",
        "policy_signal": "tightening" if policy_value is not None and policy_value > 4.0 else "unchanged",
        "volatility_signal": "high" if volatility_value is not None and volatility_value > 25.0 else "moderate",
        "growth": {
            "gdp_like_signal": growth_value,
        },
        "inflation": {
            "headline_inflation": inflation_value,
        },
        "policy": {
            "policy_rate": policy_value,
        },
        "market": {
            "volatility_index": volatility_value,
        },
    }
    )
