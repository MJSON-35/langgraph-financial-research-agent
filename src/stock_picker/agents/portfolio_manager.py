"""Portfolio manager that behaves like a deterministic investment committee chair."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List

from stock_picker.agents.common import build_insufficient_output
from stock_picker.config import get_portfolio_manager_config
from stock_picker.export_utils import save_step_summary
from stock_picker.llm_utils import coerce_string_list, is_ollama_enabled, run_ollama_json_prompt
from stock_picker.prompts import render_portfolio_manager_prompt
from stock_picker.state import StockIdea, StockPickerState


EQUITY_STANCE_FALLBACK_SCORE = {
    "positive": 0.80,
    "neutral": 0.55,
    "negative": 0.25,
    "unknown": 0.40,
}

RISK_CONFIDENCE_FLOOR = {
    "low": "high",
    "medium": "medium",
    "high": "low",
    "unknown": "low",
}


def build_portfolio_manager_step_summary(state: StockPickerState) -> dict[str, object]:
    """Build a compact export payload for the final decision stage."""
    final_decision = state.get("final_decision", {})
    return {
        "top_3_picks": final_decision.get("top_picks", []),
        "rejected_stocks": final_decision.get("rejected_stocks", []),
        "final_score_breakdown": final_decision.get("score_breakdown", {}),
        "decision_traces": final_decision.get("decision_traces", []),
        "committee_summary": final_decision.get("committee_summary", ""),
        "confidence_level": final_decision.get("confidence_level", "unknown"),
        "llm_rerank_applied": final_decision.get("llm_rerank_applied", False),
        "portfolio_data_quality_notes": final_decision.get("portfolio_data_quality_notes", []),
        "note": (
            "This stage combined the earlier quant, macro, equity, and risk outputs into one final "
            "investment-committee style decision with deterministic ranking, decision traces, and "
            "clear explanations for both selected and rejected names."
        ),
    }


def export_portfolio_manager_step_summary(state: StockPickerState) -> None:
    """Save the portfolio-manager step summary using the shared export utility."""
    output_dir = state.get("run_metadata", {}).get("output_dir", "outputs/step_summaries")
    save_step_summary(
        stage_name="portfolio_manager",
        summary_data=build_portfolio_manager_step_summary(state),
        step_number=6,
        output_dir=output_dir,
    )
    state["run_metadata"] = {
        **state.get("run_metadata", {}),
        "exported_step_summaries": [
            *state.get("run_metadata", {}).get("exported_step_summaries", []),
            "portfolio_manager",
        ],
    }


def portfolio_manager_node(state: StockPickerState) -> StockPickerState:
    """Combine agent outputs into a deterministic investment committee decision."""
    macro_view = state.get("macro_view", {})
    candidates: List[StockIdea] = state.get("filtered_candidates", [])
    equity_analysis = state.get("equity_analysis", {})
    risk_analysis = state.get("risk_analysis", {})
    missing_inputs = []

    if not candidates:
        missing_inputs.append("filtered_candidates")
    if not state.get("macro_view"):
        missing_inputs.append("macro_view")
    if not equity_analysis:
        missing_inputs.append("equity_analysis")
    if not risk_analysis:
        missing_inputs.append("risk_analysis")

    if missing_inputs:
        state["final_decision"] = build_insufficient_output(
            agent="portfolio_manager",
            summary="Insufficient data for a portfolio decision.",
            missing_inputs=missing_inputs,
            extra_fields={
                "top_picks": [],
                "rejected_stocks": [],
                "committee_summary": "Insufficient data.",
                "confidence_level": "low",
                "ranking_method": "unavailable",
                "score_breakdown": {},
                "decision_traces": [],
                "llm_rerank_applied": False,
                "data_quality_penalty": 0.0,
                "next_steps": ["Provide the missing structured inputs and rerun the workflow."],
            },
        )
        state["report"] = build_short_report(state["final_decision"], state.get("macro_view", {}))
        state["debug_notes"] = [
            *state.get("debug_notes", []),
            "Portfolio manager returned insufficient_input due to missing structured inputs.",
        ]
        export_portfolio_manager_step_summary(state)
        return state

    config = get_portfolio_manager_config(state.get("run_metadata", {}))
    deterministic_decision = build_deterministic_portfolio_decision(
        candidates=candidates,
        macro_view=macro_view,
        equity_analysis=equity_analysis,
        risk_analysis=risk_analysis,
        config=config,
    )

    final_decision = deepcopy(deterministic_decision)
    llm_rerank_applied = False
    if is_ollama_enabled(state.get("run_metadata", {}), "portfolio_manager"):
        prompt = render_portfolio_manager_prompt(
            candidates=deterministic_decision["ranked_candidates"],
            macro_view=macro_view,
            equity_analysis=equity_analysis,
            risk_analysis=risk_analysis,
            deterministic_decision_context={
                "config": config,
                "ranking_method": deterministic_decision["ranking_method"],
                "decision_traces": deterministic_decision["decision_traces"],
                "top_pick_tickers": [pick.get("ticker", "UNKNOWN") for pick in deterministic_decision["top_picks"]],
                "allow_llm_rerank": bool(config.get("allow_llm_rerank", False)),
            },
        )
        llm_output, llm_error = run_ollama_json_prompt(
            prompt,
            state.get("run_metadata", {}),
            "portfolio_manager",
        )
        if llm_output:
            final_decision, llm_rerank_applied = merge_llm_portfolio_output(
                llm_output=llm_output,
                deterministic_decision=deterministic_decision,
                allow_llm_rerank=bool(config.get("allow_llm_rerank", False)),
            )
            state["debug_notes"] = [
                *state.get("debug_notes", []),
                (
                    "Portfolio manager used local Ollama for committee narrative generation."
                    if not llm_rerank_applied
                    else "Portfolio manager applied Ollama-assisted reranking on top of deterministic scores."
                ),
            ]
        else:
            state["debug_notes"] = [
                *state.get("debug_notes", []),
                f"Portfolio manager fell back to deterministic committee logic because Ollama was unavailable ({llm_error}).",
            ]

    final_decision.pop("ranked_candidates", None)
    state["final_decision"] = final_decision
    state["report"] = build_short_report(state["final_decision"], macro_view)
    state["debug_notes"] = [
        *state.get("debug_notes", []),
        (
            f"Portfolio manager ranked {len(deterministic_decision['decision_traces'])} candidates and selected "
            f"{len(state['final_decision'].get('top_picks', []))} picks."
        ),
    ]
    export_portfolio_manager_step_summary(state)
    return state


def build_deterministic_portfolio_decision(
    *,
    candidates: List[StockIdea],
    macro_view: Dict[str, Any],
    equity_analysis: Dict[str, Dict[str, Any]],
    risk_analysis: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the baseline deterministic investment-committee decision."""
    traces = [
        build_decision_trace(
            candidate=candidate,
            macro_view=macro_view,
            equity_view=equity_analysis.get(candidate.get("ticker", "UNKNOWN"), {}),
            risk_view=risk_analysis.get(candidate.get("ticker", "UNKNOWN"), {}),
            config=config,
        )
        for candidate in candidates
    ]

    ranked_traces = stabilize_sector_penalties(traces, config)
    annotate_rank_and_decisions(ranked_traces, config)
    top_picks = build_top_picks(ranked_traces, config)
    rejected_stocks = build_rejected_stocks(ranked_traces)
    committee_summary = build_committee_summary(
        macro_view=macro_view,
        top_picks=top_picks,
        rejected_stocks=rejected_stocks,
        decision_traces=ranked_traces,
    )
    portfolio_data_quality_notes = summarize_portfolio_data_quality(ranked_traces)
    ranking_method = (
        "final_score = "
        f"{config['quant_weight']:.2f} * quant_score + "
        f"{config['equity_weight']:.2f} * equity_score + "
        "macro_adjustment_weight * macro_adjustment - "
        "risk_penalty_weight * risk_penalty - "
        "sector_concentration_penalty - missing_data_penalty - data_quality_penalty"
    )
    confidence_level = infer_portfolio_confidence(top_picks)
    score_breakdown = {trace["ticker"]: trace_to_score_breakdown(trace) for trace in ranked_traces}

    return {
        "agent": "portfolio_manager",
        "status": "completed",
        "summary": (
            "Selected the top names with deterministic committee-chair logic that combines quant, "
            "equity conviction, macro sector fit, risk penalties, diversification controls, and data-quality penalties."
        ),
        "top_picks": top_picks,
        "rejected_stocks": rejected_stocks,
        "committee_summary": committee_summary,
        "confidence_level": confidence_level,
        "ranking_method": ranking_method,
        "score_breakdown": score_breakdown,
        "decision_traces": ranked_traces,
        "llm_rerank_applied": False,
        "portfolio_data_quality_notes": portfolio_data_quality_notes,
        "next_steps": [
            "Add portfolio-level covariance and risk budgeting.",
            "Add transcript or revision data to raise committee confidence.",
            "Test sensitivity to sector and missing-data penalties.",
        ],
        "missing_inputs": [],
        "ranked_candidates": [build_ranked_candidate_record(trace) for trace in ranked_traces],
    }


def build_decision_trace(
    *,
    candidate: StockIdea,
    macro_view: Dict[str, Any],
    equity_view: Dict[str, Any],
    risk_view: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct one deterministic committee decision trace for a candidate."""
    ticker = str(candidate.get("ticker", "UNKNOWN"))
    company_name = str(candidate.get("name", ""))
    sector = str(candidate.get("sector", "UNKNOWN"))
    quant_score = clamp_score(candidate.get("quant_score", 0.0))
    equity_score = infer_equity_score(equity_view)
    macro_adjustment = round(
        config["macro_adjustment_weight"] * infer_macro_adjustment(candidate, macro_view, config),
        4,
    )
    risk_penalty = round(
        config["risk_penalty_weight"] * infer_risk_penalty(risk_view, config),
        4,
    )
    missing_data_penalty, missing_reasons = infer_missing_data_penalty(candidate, equity_view, risk_view, config)
    data_quality_penalty, data_quality_reasons = infer_data_quality_penalty(candidate, config)

    weighted_quant = round(config["quant_weight"] * quant_score, 4)
    weighted_equity = round(config["equity_weight"] * equity_score, 4)
    preliminary_score = round(
        weighted_quant + weighted_equity + macro_adjustment - risk_penalty - missing_data_penalty - data_quality_penalty,
        4,
    )

    positive_reasons = build_positive_reasons(candidate, equity_view, macro_adjustment, quant_score)
    negative_reasons = build_negative_reasons(risk_view, missing_reasons + data_quality_reasons, macro_adjustment)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "quant_score": quant_score,
        "equity_score": equity_score,
        "weighted_quant_component": weighted_quant,
        "weighted_equity_component": weighted_equity,
        "macro_adjustment": macro_adjustment,
        "risk_penalty": risk_penalty,
        "sector_concentration_penalty": 0.0,
        "missing_data_penalty": missing_data_penalty,
        "data_quality_penalty": data_quality_penalty,
        "preliminary_score": preliminary_score,
        "final_score": preliminary_score,
        "rank": 0,
        "deterministic_rank": 0,
        "decision": "rejected",
        "main_positive_reasons": positive_reasons,
        "main_negative_reasons": negative_reasons,
        "confidence": infer_trace_confidence(
            equity_score=equity_score,
            risk_level=str(risk_view.get("risk_level", "unknown")),
            missing_data_penalty=missing_data_penalty,
            data_quality_penalty=data_quality_penalty,
        ),
        "equity_conviction": str(equity_view.get("overall_view", "unknown")),
        "adjusted_quality_score": candidate.get("quality_score_cleaned", candidate.get("quality_score")),
        "macro_regime": str(macro_view.get("regime", "unknown")),
        "risk_level": str(risk_view.get("risk_level", "unknown")),
        "investment_thesis": str(equity_view.get("investment_thesis", "")),
        "risk_comment": str(risk_view.get("risk_comment", "")),
        "data_quality_flags": coerce_string_list(candidate.get("data_quality_flags")),
        "data_quality_warning": str(candidate.get("data_quality_warning", "")),
        "feature_confidence": str(candidate.get("feature_confidence", "unknown")),
        "candidate_record": deepcopy(candidate),
    }


def stabilize_sector_penalties(traces: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Iteratively apply sector concentration penalties until ranks stabilize."""
    order = sorted(traces, key=trace_sort_key, reverse=True)
    for _ in range(4):
        updated_order: List[Dict[str, Any]] = []
        sector_counts: Dict[str, int] = {}
        for trace in order:
            updated = dict(trace)
            sector = updated.get("sector", "UNKNOWN")
            prior_count = sector_counts.get(sector, 0)
            penalty = config["sector_concentration_penalty"] if prior_count >= config["max_same_sector_in_top3"] else 0.0
            updated["sector_concentration_penalty"] = round(penalty, 4)
            updated["final_score"] = round(updated["preliminary_score"] - penalty, 4)
            if penalty > 0 and "sector concentration issue" not in updated["main_negative_reasons"]:
                updated["main_negative_reasons"].append("sector concentration issue")
            sector_counts[sector] = prior_count + 1
            updated_order.append(updated)

        new_order = sorted(updated_order, key=trace_sort_key, reverse=True)
        if [item["ticker"] for item in new_order] == [item["ticker"] for item in order]:
            return new_order
        order = new_order
    return order


def annotate_rank_and_decisions(traces: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """Add final ranks and selected/rejected labels in place."""
    top_n = int(config["top_n"])
    selected = traces[:top_n]
    selected_sectors = [trace.get("sector", "UNKNOWN") for trace in selected]
    sector_counts = {sector: selected_sectors.count(sector) for sector in set(selected_sectors)}
    strongest_selected_by_sector = {
        sector: max(
            (trace for trace in selected if trace.get("sector") == sector),
            key=lambda trace: float(trace.get("final_score", 0.0)),
        )["ticker"]
        for sector in sector_counts
    }

    for idx, trace in enumerate(traces, start=1):
        trace["rank"] = idx
        trace["deterministic_rank"] = idx
        trace["decision"] = "selected" if idx <= top_n else "rejected"
        if trace["decision"] == "rejected":
            sector = trace.get("sector", "UNKNOWN")
            if sector_counts.get(sector, 0) >= config["max_same_sector_in_top3"]:
                leader = strongest_selected_by_sector.get(sector)
                if leader and leader != trace["ticker"]:
                    trace["main_negative_reasons"].append("stronger alternative exists in same sector")
        trace["main_negative_reasons"] = dedupe_reasons(trace["main_negative_reasons"])
        trace["main_positive_reasons"] = dedupe_reasons(trace["main_positive_reasons"])


def build_top_picks(traces: List[Dict[str, Any]], config: Dict[str, Any]) -> List[StockIdea]:
    """Convert selected traces back into stock records for downstream compatibility."""
    top_n = int(config["top_n"])
    top_picks: List[StockIdea] = []
    for trace in traces[:top_n]:
        record = deepcopy(trace["candidate_record"])
        record["final_score"] = trace["final_score"]
        record["rationale"] = build_trace_rationale(trace)
        record["confidence"] = trace["confidence"]
        top_picks.append(record)
    return top_picks


def build_rejected_stocks(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build specific rejected-stock explanations from committee traces."""
    rejected: List[Dict[str, Any]] = []
    for trace in traces:
        if trace["decision"] != "rejected":
            continue
        rejected.append(
            {
                "ticker": trace["ticker"],
                "name": trace["company_name"],
                "sector": trace["sector"],
                "final_score": trace["final_score"],
                "confidence": trace["confidence"],
                "reason": "; ".join(trace["main_negative_reasons"][:3]) or "lower relative committee conviction",
            }
        )
    return rejected


def merge_llm_portfolio_output(
    *,
    llm_output: Dict[str, Any],
    deterministic_decision: Dict[str, Any],
    allow_llm_rerank: bool,
) -> tuple[Dict[str, Any], bool]:
    """Merge Ollama narrative output while preserving deterministic scoring as the source of truth."""
    merged = deepcopy(deterministic_decision)
    llm_summary = str(llm_output.get("summary", "")).strip()
    llm_committee_summary = str(llm_output.get("committee_summary", "")).strip()
    llm_next_steps = coerce_string_list(llm_output.get("next_steps"))

    if llm_summary and llm_summary.lower() not in {"insufficient data.", "insufficient_input"}:
        merged["summary"] = llm_summary
    if llm_committee_summary and llm_committee_summary.lower() not in {"insufficient data.", "insufficient_input"}:
        merged["committee_summary"] = llm_committee_summary
    if llm_next_steps:
        merged["next_steps"] = llm_next_steps

    if not allow_llm_rerank:
        merged["llm_rerank_applied"] = False
        return merged, False

    candidate_map = {
        record.get("ticker", "UNKNOWN"): record for record in deterministic_decision.get("ranked_candidates", [])
    }
    requested_order = [
        str(item.get("ticker", "")).strip()
        for item in llm_output.get("top_picks", [])
        if isinstance(item, dict) and str(item.get("ticker", "")).strip() in candidate_map
    ]
    top_n = len(deterministic_decision.get("top_picks", []))
    if len(requested_order) < top_n:
        merged["llm_rerank_applied"] = False
        return merged, False

    selected_set = requested_order[:top_n]
    trace_by_ticker = {trace["ticker"]: dict(trace) for trace in deterministic_decision["decision_traces"]}
    new_traces: List[Dict[str, Any]] = []
    for idx, ticker in enumerate(selected_set, start=1):
        trace = trace_by_ticker[ticker]
        trace["rank"] = idx
        trace["decision"] = "selected"
        new_traces.append(trace)
    rejected_rank = top_n + 1
    for trace in deterministic_decision["decision_traces"]:
        if trace["ticker"] in selected_set:
            continue
        updated = dict(trace)
        updated["rank"] = rejected_rank
        updated["decision"] = "rejected"
        if "llm reranking preferred other candidates" not in updated["main_negative_reasons"]:
            updated["main_negative_reasons"].append("llm reranking preferred other candidates")
        updated["main_negative_reasons"] = dedupe_reasons(updated["main_negative_reasons"])
        new_traces.append(updated)
        rejected_rank += 1

    merged["decision_traces"] = new_traces
    merged["top_picks"] = build_top_picks(new_traces, {"top_n": top_n})
    merged["rejected_stocks"] = build_rejected_stocks(new_traces)
    merged["llm_rerank_applied"] = True
    return merged, True


def infer_equity_score(equity_view: Dict[str, Any]) -> float:
    """Return a normalized deterministic equity score."""
    raw_score = equity_view.get("score")
    if raw_score is None:
        return EQUITY_STANCE_FALLBACK_SCORE.get(str(equity_view.get("overall_view", "unknown")), 0.40)
    try:
        return clamp_score(float(raw_score))
    except (TypeError, ValueError):
        return EQUITY_STANCE_FALLBACK_SCORE.get(str(equity_view.get("overall_view", "unknown")), 0.40)


def infer_macro_adjustment(candidate: StockIdea, macro_view: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Return a deterministic macro sector adjustment."""
    sector = str(candidate.get("sector", "UNKNOWN"))
    preferred = set(coerce_string_list(macro_view.get("preferred_sectors")))
    risk_sectors = set(coerce_string_list(macro_view.get("risk_sectors")))
    adjustment = 0.0
    if sector in preferred:
        adjustment += float(config.get("macro_sector_preference_bonus", 0.02))
    if sector in risk_sectors:
        adjustment -= float(config.get("macro_sector_risk_penalty", 0.02))
    return round(adjustment, 4)


def infer_risk_penalty(risk_view: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Return a deterministic penalty from the structured risk level."""
    risk_level = str(risk_view.get("risk_level", "unknown")).lower()
    return float(config.get("risk_level_penalties", {}).get(risk_level, 0.04))


def infer_missing_data_penalty(
    candidate: StockIdea,
    equity_view: Dict[str, Any],
    risk_view: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[float, list[str]]:
    """Return a penalty and explicit reasons for missing structured evidence."""
    unit_penalty = float(config["missing_data_penalty"])
    reasons: list[str] = []

    if is_missing(candidate.get("trailing_pe")) or is_missing(candidate.get("price_to_book")) or is_missing(candidate.get("roe")):
        reasons.append("missing fundamental evidence")
    if not str(candidate.get("business_summary", "")).strip() and not coerce_string_list(candidate.get("recent_news_headlines")):
        reasons.append("insufficient qualitative evidence")
    if equity_view.get("status") != "completed":
        reasons.append("equity analysis incomplete")
    if risk_view.get("status") != "completed":
        reasons.append("risk analysis incomplete")

    reasons = dedupe_reasons(reasons)
    penalty = round(unit_penalty * len(reasons), 4)
    return penalty, reasons


def infer_data_quality_penalty(candidate: StockIdea, config: Dict[str, Any]) -> tuple[float, list[str]]:
    """Return a deterministic penalty and reasons for low-confidence or outlier-driven inputs."""
    unit_penalty = float(config["missing_data_penalty"])
    reasons: list[str] = []
    flags = set(coerce_string_list(candidate.get("data_quality_flags")))

    if "roe_extreme_outlier" in flags:
        reasons.append("extreme profitability outlier")
    if "insufficient_12m_price_history" in flags:
        reasons.append("insufficient 12m price history")
    if "momentum_12m_outlier" in flags or "momentum_1m_outlier" in flags:
        reasons.append("abnormal momentum outlier")
    if {"trailing_pe_missing", "price_to_book_missing"} <= flags:
        reasons.append("multiple missing valuation fields")
    if str(candidate.get("feature_confidence", "")).lower() == "low":
        reasons.append("low feature confidence")

    reasons = dedupe_reasons(reasons)
    penalty = round(float(candidate.get("data_quality_penalty", 0.0) or 0.0) + unit_penalty * len(reasons), 4)
    return penalty, reasons


def build_positive_reasons(
    candidate: StockIdea,
    equity_view: Dict[str, Any],
    macro_adjustment: float,
    quant_score: float,
) -> list[str]:
    """Build positive committee reasons for one candidate."""
    reasons: list[str] = []
    if quant_score >= 0.70:
        reasons.append("strong quant score")
    elif quant_score >= 0.55:
        reasons.append("solid quant score")

    if infer_equity_score(equity_view) >= 0.70:
        reasons.append("strong equity conviction")
    elif str(equity_view.get("overall_view", "unknown")) == "positive":
        reasons.append("positive equity view")

    if macro_adjustment > 0:
        reasons.append("good macro sector fit")

    if float(candidate.get("liquidity_penalty", 0.0) or 0.0) <= 0:
        reasons.append("acceptable liquidity profile")

    return dedupe_reasons(reasons) or ["meets the minimum shortlist standard"]


def build_negative_reasons(
    risk_view: Dict[str, Any],
    missing_reasons: list[str],
    macro_adjustment: float,
) -> list[str]:
    """Build negative committee reasons for one candidate."""
    reasons: list[str] = []
    risk_level = str(risk_view.get("risk_level", "unknown")).lower()
    if risk_level == "high":
        reasons.append("high risk penalty")
    elif risk_level == "medium":
        reasons.append("moderate risk penalty")
    if macro_adjustment < 0:
        reasons.append("poor macro sector fit")
    reasons.extend(missing_reasons)
    return dedupe_reasons(reasons)


def infer_trace_confidence(
    *,
    equity_score: float,
    risk_level: str,
    missing_data_penalty: float,
    data_quality_penalty: float,
) -> str:
    """Infer one confidence label for a candidate decision trace."""
    if (
        missing_data_penalty <= 0
        and data_quality_penalty <= 0
        and equity_score >= 0.70
        and RISK_CONFIDENCE_FLOOR.get(risk_level, "low") != "low"
    ):
        return "high"
    if (
        missing_data_penalty <= 0.02
        and data_quality_penalty <= 0.03
        and RISK_CONFIDENCE_FLOOR.get(risk_level, "low") in {"high", "medium"}
    ):
        return "medium"
    return "low"


def infer_portfolio_confidence(top_picks: List[StockIdea]) -> str:
    """Infer one overall committee confidence label from selected names."""
    confidences = [str(pick.get("confidence", "low")).lower() for pick in top_picks]
    if confidences and all(level == "high" for level in confidences):
        return "high"
    if confidences and any(level == "low" for level in confidences):
        return "medium" if any(level == "high" for level in confidences) else "low"
    return "medium"


def build_committee_summary(
    *,
    macro_view: Dict[str, Any],
    top_picks: List[StockIdea],
    rejected_stocks: List[Dict[str, Any]],
    decision_traces: List[Dict[str, Any]],
) -> str:
    """Build a chair-style committee memo summary with required contents."""
    regime = str(macro_view.get("regime", "unknown"))
    top_pick_summary = ", ".join(
        f"{pick.get('ticker', 'UNKNOWN')} ({pick.get('sector', 'UNKNOWN')})" for pick in top_picks
    ) or "no selections"
    selected_reasons = summarize_trace_reasons(
        [trace for trace in decision_traces if trace.get("decision") == "selected"],
        field="main_positive_reasons",
    )
    rejected_reasons = summarize_trace_reasons(
        [trace for trace in decision_traces if trace.get("decision") == "rejected"],
        field="main_negative_reasons",
    )
    portfolio_risks = summarize_trace_reasons(decision_traces, field="main_negative_reasons")
    data_quality_notes = summarize_portfolio_data_quality(decision_traces)
    limitations = [
        "the committee still relies on proxy risk inputs and limited free qualitative data",
        "the current ranking does not yet use portfolio covariance or optimization",
        "some financial ratios required capping or low-confidence handling before scoring",
    ]
    additional_data = [
        "earnings call transcripts",
        "estimate revision data",
        "portfolio-level covariance and benchmark risk",
        "cleaner point-in-time fundamental datasets and audited quality ratios",
    ]

    data_quality_summary = ", ".join(data_quality_notes) if data_quality_notes else "no material quality warnings"

    return (
        f"Market regime: {regime}. "
        f"Top 3 selected: {top_pick_summary}, mainly because of {selected_reasons}. "
        f"Rejected names were excluded due to {rejected_reasons}. "
        f"Portfolio-level key risks: {portfolio_risks}. "
        f"Data-quality cautions: {data_quality_summary}. "
        f"Current decision limitations: {', '.join(limitations)}. "
        f"To raise confidence further, the committee would want {', '.join(additional_data)}."
    )


def build_short_report(final_decision: Dict[str, Any], macro_view: Dict[str, Any]) -> str:
    """Create a compact report without needing a separate reporting node."""
    picks = final_decision.get("top_picks", [])
    if not picks:
        return "No picks were generated."

    lines = [
        "Multi-agent stock picker prototype report",
        f"Macro view: {macro_view.get('macro_summary', 'No macro summary available.')}",
        f"Preferred sectors: {', '.join(macro_view.get('preferred_sectors', [])) or 'None'}",
        f"Risk sectors: {', '.join(macro_view.get('risk_sectors', [])) or 'None'}",
        f"Key macro risks: {', '.join(macro_view.get('key_macro_risks', [])) or 'None'}",
        f"Decision method: {final_decision.get('ranking_method', 'No ranking method available.')}",
        f"Committee summary: {final_decision.get('committee_summary', 'No committee summary available.')}",
        f"Confidence level: {final_decision.get('confidence_level', 'unknown')}",
        f"Data quality notes: {', '.join(final_decision.get('portfolio_data_quality_notes', [])) or 'None'}",
        "",
        "Top picks:",
    ]

    for idx, pick in enumerate(picks, start=1):
        lines.append(
            f"{idx}. {pick.get('ticker')} ({pick.get('name')}): "
            f"score={pick.get('final_score', 0.0):.2f} | "
            f"confidence={pick.get('confidence', 'unknown')} | "
            f"rationale={pick.get('rationale', 'No rationale available.')} | "
            f"data_quality={pick.get('data_quality_warning', 'No material data-quality warnings.')}"
        )

    return "\n".join(lines)


def trace_to_score_breakdown(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a decision trace to the legacy score-breakdown map used by reports."""
    return {
        "ticker": trace["ticker"],
        "name": trace["company_name"],
        "final_rank": trace["rank"],
        "quant_score": trace["quant_score"],
        "equity_score": trace["equity_score"],
        "adjusted_quality_score": trace.get("adjusted_quality_score"),
        "equity_bonus": trace["weighted_equity_component"],
        "equity_stance": trace.get("equity_conviction", "unknown"),
        "equity_summary": trace.get("investment_thesis", ""),
        "macro_regime": trace.get("macro_regime", "unknown"),
        "macro_bonus": 0.0,
        "macro_adjustment": trace["macro_adjustment"],
        "sector": trace["sector"],
        "sector_bonus": max(trace["macro_adjustment"], 0.0),
        "sector_penalty": abs(min(trace["macro_adjustment"], 0.0)),
        "risk_level": trace.get("risk_level", "unknown"),
        "risk_summary": trace.get("risk_comment", ""),
        "risk_penalty": trace["risk_penalty"],
        "missing_data_penalty": trace["missing_data_penalty"],
        "data_quality_penalty": trace.get("data_quality_penalty", 0.0),
        "sector_concentration_penalty": trace["sector_concentration_penalty"],
        "final_score": trace["final_score"],
        "recommendation": "selected" if trace["decision"] == "selected" else "rejected",
        "decision_summary": build_trace_rationale(trace),
        "rationale": build_trace_rationale(trace),
        "confidence": trace["confidence"],
        "data_quality_flags": trace.get("data_quality_flags", []),
        "data_quality_warning": trace.get("data_quality_warning", ""),
        "feature_confidence": trace.get("feature_confidence", "unknown"),
    }


def build_ranked_candidate_record(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lightweight candidate record for prompt rendering and downstream display."""
    return {
        "ticker": trace["ticker"],
        "name": trace["company_name"],
        "sector": trace["sector"],
        "quant_score": trace["quant_score"],
        "equity_score": trace["equity_score"],
        "macro_adjustment": trace["macro_adjustment"],
        "risk_penalty": trace["risk_penalty"],
        "missing_data_penalty": trace["missing_data_penalty"],
        "data_quality_penalty": trace.get("data_quality_penalty", 0.0),
        "sector_concentration_penalty": trace["sector_concentration_penalty"],
        "final_score": trace["final_score"],
        "confidence": trace["confidence"],
        "decision_summary": build_trace_rationale(trace),
        "data_quality_flags": trace.get("data_quality_flags", []),
        "data_quality_warning": trace.get("data_quality_warning", ""),
        "feature_confidence": trace.get("feature_confidence", "unknown"),
    }


def build_trace_rationale(trace: Dict[str, Any]) -> str:
    """Render a concise deterministic rationale from one decision trace."""
    return (
        f"Started from weighted quant ({trace['weighted_quant_component']:.2f}) and weighted equity ({trace['weighted_equity_component']:.2f}), "
        f"applied macro adjustment {trace['macro_adjustment']:+.2f}, "
        f"risk penalty {trace['risk_penalty']:.2f}, "
        f"sector concentration penalty {trace['sector_concentration_penalty']:.2f}, "
        f"missing-data penalty {trace['missing_data_penalty']:.2f}, "
        f"and data-quality penalty {trace.get('data_quality_penalty', 0.0):.2f}."
    )


def summarize_trace_reasons(traces: List[Dict[str, Any]], *, field: str) -> str:
    """Return a short comma-joined summary of common reasons."""
    reasons: list[str] = []
    for trace in traces:
        reasons.extend(coerce_string_list(trace.get(field)))
    reasons = dedupe_reasons(reasons)
    return ", ".join(reasons[:4]) if reasons else "limited structured evidence"


def summarize_portfolio_data_quality(traces: List[Dict[str, Any]]) -> list[str]:
    """Return a short portfolio-level summary of the main data-quality cautions."""
    notes: list[str] = []
    for trace in traces:
        notes.extend(coerce_string_list(trace.get("data_quality_flags")))
    normalized = []
    mapping = {
        "roe_extreme_outlier": "extreme ROE outlier present",
        "roe_winsorized": "winsorized profitability ratios present",
        "trailing_pe_missing": "missing P/E fields present",
        "price_to_book_missing": "missing P/B fields present",
        "insufficient_12m_price_history": "limited 12m price history present",
        "momentum_12m_outlier": "capped 12m momentum present",
        "momentum_1m_outlier": "capped 1m momentum present",
    }
    for note in dedupe_reasons(notes):
        normalized.append(mapping.get(note, note.replace("_", " ")))
    return normalized[:5]


def trace_sort_key(trace: Dict[str, Any]) -> tuple[float, float, float, str]:
    """Provide a stable descending sort key for final ranking."""
    return (
        float(trace.get("final_score", 0.0)),
        float(trace.get("equity_score", 0.0)),
        float(trace.get("quant_score", 0.0)),
        str(trace.get("ticker", "")),
    )


def dedupe_reasons(reasons: List[str]) -> List[str]:
    """Deduplicate reasons while preserving order."""
    seen: set[str] = set()
    cleaned: List[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def clamp_score(value: Any) -> float:
    """Clamp a score-like value to the 0-1 interval."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def is_missing(value: Any) -> bool:
    """Return True when a scalar should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False
