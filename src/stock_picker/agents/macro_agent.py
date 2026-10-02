"""Macro agent with optional Ollama-backed structured reasoning."""

from __future__ import annotations

from collections import Counter
from typing import Any

from stock_picker.agents.common import build_insufficient_output, dedupe_preserve_order
from stock_picker.export_utils import save_step_summary
from stock_picker.llm_utils import coerce_string_list, is_ollama_enabled, run_ollama_json_prompt
from stock_picker.prompts import render_macro_agent_prompt
from stock_picker.providers import load_macro_context
from stock_picker.state import StockPickerState


RISK_ON_SECTORS = {"Technology", "Semiconductors", "Consumer", "Industrials"}
DEFENSIVE_SECTORS = {"Healthcare", "Utilities", "Consumer Staples"}
STRUCTURAL_RISK_SECTORS = {"Energy", "Materials", "Real Estate"}


def build_macro_agent_step_summary(state: StockPickerState) -> dict[str, object]:
    """Build a small export payload for the macro-agent stage."""
    macro_view = state.get("macro_view", {})
    preferred_sectors = macro_view.get("preferred_sectors", [])
    risk_sectors = macro_view.get("risk_sectors", [])
    downstream_note = (
        "This stage added a light macro overlay that can favor preferred sectors and reduce conviction "
        "in sectors flagged as macro risks during final selection."
    )
    return {
        "macro_summary": macro_view.get("macro_summary", "No macro summary available."),
        "preferred_sectors": preferred_sectors,
        "risk_sectors": risk_sectors,
        "key_macro_risks": macro_view.get("key_macro_risks", []),
        "note": downstream_note,
    }


def export_macro_agent_step_summary(state: StockPickerState) -> None:
    """Save the macro-agent step summary using the shared export utility."""
    output_dir = state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries")
    save_step_summary(
        stage_name="macro_agent",
        summary_data=build_macro_agent_step_summary(state),
        step_number=3,
        output_dir=output_dir,
    )
    run_metadata = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "macro_agent",
        ],
    }
    state["run_metadata"] = run_metadata


def macro_agent_node(state: StockPickerState) -> StockPickerState:
    """Build a macro overlay with a stable rule-based baseline and optional Ollama refinement."""
    candidates = state.get("filtered_candidates", [])
    macro_context, provider_info = load_macro_context(state)
    sector_hints = state.get("run_metadata", {}).get("sector_hints", {})
    macro_notes = dict(state.get("run_metadata", {}).get("macro_notes", {}))

    missing_inputs = []
    if not candidates:
        missing_inputs.append("filtered_candidates")
    if not macro_context:
        missing_inputs.append("macro_context")

    if missing_inputs:
        state["macro_view"] = build_insufficient_output(
            agent="macro_agent",
            summary="Insufficient data for a macro assessment.",
            missing_inputs=missing_inputs,
            extra_fields={
                "regime": "unknown",
                "macro_summary": "Insufficient data. Required structured macro inputs were not provided.",
                "preferred_sectors": [],
                "risk_sectors": [],
                "key_macro_risks": [],
                "confidence": 0.0,
            },
        )
        state["debug_notes"] = [
            *state.get("debug_notes", []),
            (
                "Macro agent status: insufficient_input "
                f"({provider_info['macro_provider']} / {provider_info['macro_provider_status']})."
            ),
        ]
        export_macro_agent_step_summary(state)
        return state

    baseline_macro_view = build_rule_based_macro_view(candidates, macro_context, sector_hints)
    macro_notes = {
        **macro_notes,
        "macro_feature_summary": macro_context.get("macro_feature_summary", {}),
        "baseline_analysis": baseline_macro_view,
    }

    macro_view = None
    run_metadata = state.get("run_metadata", {})
    if is_ollama_enabled(run_metadata, "macro_agent"):
        prompt = render_macro_agent_prompt(
            candidates=candidates,
            macro_context=macro_context,
            sector_hints=sector_hints,
            macro_notes=macro_notes,
        )
        llm_output, llm_error = run_ollama_json_prompt(prompt, run_metadata, "macro_agent")
        if llm_output:
            macro_view = merge_macro_output_with_baseline(llm_output, baseline_macro_view)
            state["debug_notes"] = [
                *state.get("debug_notes", []),
                "Macro agent used hybrid mode: rule-based baseline plus local Ollama refinement.",
            ]
        else:
            state["debug_notes"] = [
                *state.get("debug_notes", []),
                f"Macro agent kept the rule-based baseline because Ollama was unavailable ({llm_error}).",
            ]

    if macro_view is None:
        macro_view = baseline_macro_view
        state["debug_notes"] = [
            *state.get("debug_notes", []),
            f"Macro agent used the rule-based baseline ({macro_view['regime']}) via {provider_info['macro_provider']}.",
        ]

    state["macro_view"] = macro_view
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "macro_provider": provider_info["macro_provider"],
        "macro_provider_status": provider_info["macro_provider_status"],
    }
    export_macro_agent_step_summary(state)
    return state


def build_rule_based_macro_view(
    candidates: list[dict[str, Any]],
    macro_context: dict[str, Any],
    sector_hints: dict[str, Any],
) -> dict[str, Any]:
    """Return the original rule-based macro output."""
    candidate_sectors = [str(item.get("sector", "")).strip() for item in candidates if item.get("sector")]
    sector_counts = Counter(candidate_sectors)
    top_candidate_sectors = [sector for sector, _ in sector_counts.most_common(3)]

    growth_signal = str(macro_context.get("growth_signal", "stable")).lower()
    inflation_signal = str(macro_context.get("inflation_signal", "stable")).lower()
    policy_signal = str(macro_context.get("policy_signal", "unchanged")).lower()
    volatility_signal = str(macro_context.get("volatility_signal", "moderate")).lower()

    regime_score = 0
    if growth_signal in {"improving", "strong"}:
        regime_score += 1
    elif growth_signal in {"weakening", "weak"}:
        regime_score -= 1
    if inflation_signal in {"cooling", "contained"}:
        regime_score += 1
    elif inflation_signal in {"rising", "hot"}:
        regime_score -= 1
    if policy_signal in {"easing", "supportive"}:
        regime_score += 1
    elif policy_signal in {"tightening", "restrictive"}:
        regime_score -= 1
    if volatility_signal in {"high", "stressed"}:
        regime_score -= 1
    elif volatility_signal in {"low", "calm"}:
        regime_score += 1

    if regime_score >= 2:
        regime = "risk_on"
    elif regime_score <= -2:
        regime = "risk_off"
    else:
        regime = "neutral"

    hinted_overweight = list(sector_hints.get("overweight", []))
    hinted_underweight = list(sector_hints.get("underweight", []))
    hinted_risks = list(sector_hints.get("key_macro_risks", sector_hints.get("macro_risks", [])))

    if regime == "risk_on":
        base_preferred = [sector for sector in top_candidate_sectors if sector in RISK_ON_SECTORS]
        if not base_preferred:
            base_preferred = [sector for sector in top_candidate_sectors if sector not in STRUCTURAL_RISK_SECTORS]
        base_risk = [sector for sector in top_candidate_sectors if sector in STRUCTURAL_RISK_SECTORS]
    elif regime == "risk_off":
        base_preferred = [sector for sector in top_candidate_sectors if sector in DEFENSIVE_SECTORS]
        base_risk = [sector for sector in top_candidate_sectors if sector in RISK_ON_SECTORS]
    else:
        base_preferred = [sector for sector in top_candidate_sectors if sector not in STRUCTURAL_RISK_SECTORS]
        base_risk = [sector for sector in top_candidate_sectors if sector in STRUCTURAL_RISK_SECTORS]

    preferred_sectors = dedupe_preserve_order(hinted_overweight + base_preferred)[:4]
    risk_sectors = dedupe_preserve_order(hinted_underweight + base_risk)[:4]
    key_macro_risks = dedupe_preserve_order(
        hinted_risks
        + _derive_macro_risks(growth_signal, inflation_signal, policy_signal, volatility_signal)
    )[:4]

    macro_summary = f"{regime} macro overlay from structured growth, inflation, policy, and volatility signals."
    confidence = round(min(0.75, 0.35 + 0.10 * len(macro_context)), 2)
    return {
        "agent": "macro_agent",
        "status": "completed",
        "summary": macro_summary,
        "regime": regime,
        "macro_summary": macro_summary,
        "preferred_sectors": preferred_sectors,
        "risk_sectors": risk_sectors,
        "key_macro_risks": key_macro_risks,
        "confidence": confidence,
        "missing_inputs": [],
    }


def coerce_macro_output(llm_output: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ollama output into the project macro schema."""
    return {
        "agent": "macro_agent",
        "status": str(llm_output.get("status", "completed")),
        "summary": str(llm_output.get("summary", llm_output.get("macro_summary", ""))),
        "regime": str(llm_output.get("regime", "unknown")),
        "macro_summary": str(llm_output.get("macro_summary", llm_output.get("summary", "Insufficient data."))),
        "preferred_sectors": coerce_string_list(llm_output.get("preferred_sectors")),
        "risk_sectors": coerce_string_list(llm_output.get("risk_sectors")),
        "key_macro_risks": coerce_string_list(llm_output.get("key_macro_risks")),
        "confidence": float(llm_output.get("confidence", 0.0) or 0.0),
        "missing_inputs": coerce_string_list(llm_output.get("missing_inputs")),
    }


def merge_macro_output_with_baseline(
    llm_output: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Use rule-based macro results as the anchor and let the LLM refine only supported fields."""
    normalized = coerce_macro_output(llm_output)
    llm_status = normalized.get("status", "completed")
    if llm_status != "completed":
        return {
            **baseline,
            "summary": baseline.get("summary", ""),
            "macro_summary": baseline.get("macro_summary", ""),
        }

    preferred_sectors = normalized.get("preferred_sectors") or baseline.get("preferred_sectors", [])
    risk_sectors = normalized.get("risk_sectors") or baseline.get("risk_sectors", [])
    key_macro_risks = normalized.get("key_macro_risks") or baseline.get("key_macro_risks", [])

    llm_summary = str(normalized.get("summary") or "").strip()
    llm_macro_summary = str(normalized.get("macro_summary") or "").strip()
    llm_regime = str(normalized.get("regime") or "").strip().lower()

    baseline_summary = str(baseline.get("summary", "")).strip()
    baseline_macro_summary = str(baseline.get("macro_summary", "")).strip()
    baseline_regime = str(baseline.get("regime", "unknown")).strip().lower()

    summary = _prefer_supported_macro_text(llm_summary, baseline_summary)
    macro_summary = _prefer_supported_macro_text(llm_macro_summary, baseline_macro_summary)
    regime = llm_regime if _is_supported_regime(llm_regime) else baseline_regime

    return {
        **baseline,
        "status": "completed",
        "summary": summary,
        "regime": regime,
        "macro_summary": macro_summary,
        "preferred_sectors": preferred_sectors,
        "risk_sectors": risk_sectors,
        "key_macro_risks": key_macro_risks,
        "confidence": max(
            float(baseline.get("confidence", 0.0) or 0.0),
            float(normalized.get("confidence", 0.0) or 0.0),
        ),
        "missing_inputs": [],
    }


def _prefer_supported_macro_text(candidate_text: str, baseline_text: str) -> str:
    """Keep the stable baseline summary when the LLM output is empty or overly generic."""
    normalized = candidate_text.strip().lower()
    if not normalized:
        return baseline_text
    unsupported_markers = {
        "insufficient data.",
        "insufficient data",
        "unknown",
        "n/a",
    }
    if normalized in unsupported_markers:
        return baseline_text
    return candidate_text


def _is_supported_regime(regime: str) -> bool:
    """Return whether an LLM regime label is one of the allowed portfolio states."""
    return regime in {"risk_on", "neutral", "risk_off", "unknown"}


def _derive_macro_risks(
    growth_signal: str,
    inflation_signal: str,
    policy_signal: str,
    volatility_signal: str,
) -> list[str]:
    risks: list[str] = []
    if growth_signal in {"weakening", "weak"}:
        risks.append("growth slowdown")
    if inflation_signal in {"rising", "hot"}:
        risks.append("inflation pressure")
    if policy_signal in {"tightening", "restrictive"}:
        risks.append("tight policy")
    if volatility_signal in {"high", "stressed"}:
        risks.append("market volatility")
    if not risks:
        risks.append("policy uncertainty")
    return risks
