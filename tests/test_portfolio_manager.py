from __future__ import annotations

import unittest

from stock_picker.agents.portfolio_manager import (
    build_decision_trace,
    build_deterministic_portfolio_decision,
    portfolio_manager_node,
)
from stock_picker.config import get_portfolio_manager_config


def make_candidate(
    ticker: str,
    *,
    sector: str,
    quant_score: float,
    trailing_pe: float | None = 20.0,
    price_to_book: float | None = 3.0,
    roe: float | None = 0.20,
    business_summary: str = "Company summary",
    recent_news_headlines: list[str] | None = None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": f"{ticker} Corp",
        "sector": sector,
        "market": "TEST",
        "quant_score": quant_score,
        "trailing_pe": trailing_pe,
        "price_to_book": price_to_book,
        "roe": roe,
        "business_summary": business_summary,
        "recent_news_headlines": ["steady operations"] if recent_news_headlines is None else recent_news_headlines,
        "liquidity_penalty": 0.0,
        "data_quality_penalty": 0.0,
        "data_quality_flags": [],
        "data_quality_warning": "",
        "feature_confidence": "high",
        "quality_score_cleaned": roe,
        "has_full_12m_history": True,
    }


def make_equity_view(score: float, overall_view: str = "positive") -> dict[str, object]:
    return {
        "agent": "equity_agent",
        "status": "completed",
        "ticker": "unused",
        "score": score,
        "overall_view": overall_view,
        "investment_thesis": f"{overall_view} thesis",
        "strengths": ["strong equity conviction"],
        "weaknesses": [],
    }


def make_risk_view(level: str, comment: str = "risk note") -> dict[str, object]:
    return {
        "agent": "risk_agent",
        "status": "completed",
        "ticker": "unused",
        "risk_level": level,
        "main_risks": [f"{level}_risk"],
        "risk_comment": comment,
        "max_position_size": 0.05,
    }


class PortfolioManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.macro_view = {
            "regime": "neutral",
            "macro_summary": "neutral macro backdrop",
            "preferred_sectors": ["Technology", "Semiconductors"],
            "risk_sectors": ["Energy"],
            "key_macro_risks": ["policy uncertainty"],
        }

    def build_state(
        self,
        candidates: list[dict[str, object]],
        equity_scores: dict[str, tuple[float, str]],
        risk_levels: dict[str, str],
        run_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        equity_analysis = {
            ticker: make_equity_view(score=score, overall_view=view)
            for ticker, (score, view) in equity_scores.items()
        }
        risk_analysis = {
            ticker: make_risk_view(level=level, comment=f"{ticker} {level} risk")
            for ticker, level in risk_levels.items()
        }
        return {
            "filtered_candidates": candidates,
            "macro_view": self.macro_view,
            "equity_analysis": equity_analysis,
            "risk_analysis": risk_analysis,
            "final_decision": {},
            "debug_notes": [],
            "run_metadata": {
                "use_ollama": False,
                **(run_metadata or {}),
            },
        }

    def test_final_score_is_deterministic(self) -> None:
        config = get_portfolio_manager_config({})
        candidate = make_candidate("AAA", sector="Technology", quant_score=0.80)
        trace = build_decision_trace(
            candidate=candidate,
            macro_view=self.macro_view,
            equity_view=make_equity_view(0.90, "positive"),
            risk_view=make_risk_view("medium"),
            config=config,
        )
        expected = round(0.50 * 0.80 + 0.20 * 0.90 + 0.02 - 0.03 - 0.0, 4)
        self.assertEqual(trace["final_score"], expected)

    def test_allow_llm_rerank_false_keeps_final_score_order(self) -> None:
        candidates = [
            make_candidate("AAA", sector="Technology", quant_score=0.82),
            make_candidate("BBB", sector="Financials", quant_score=0.78),
            make_candidate("CCC", sector="Industrials", quant_score=0.75),
            make_candidate("DDD", sector="Energy", quant_score=0.60),
        ]
        state = self.build_state(
            candidates=candidates,
            equity_scores={
                "AAA": (0.80, "positive"),
                "BBB": (0.75, "positive"),
                "CCC": (0.70, "neutral"),
                "DDD": (0.40, "negative"),
            },
            risk_levels={"AAA": "medium", "BBB": "medium", "CCC": "medium", "DDD": "high"},
            run_metadata={"portfolio_manager_config": {"allow_llm_rerank": False}},
        )
        result = portfolio_manager_node(state)
        traces = result["final_decision"]["decision_traces"]
        top_pick_tickers = [pick["ticker"] for pick in result["final_decision"]["top_picks"]]
        expected_order = [trace["ticker"] for trace in traces[:3]]
        self.assertEqual(top_pick_tickers, expected_order)

    def test_high_risk_penalty_lowers_ranking(self) -> None:
        config = get_portfolio_manager_config({})
        candidate = make_candidate("AAA", sector="Technology", quant_score=0.75)
        low_risk_trace = build_decision_trace(
            candidate=candidate,
            macro_view=self.macro_view,
            equity_view=make_equity_view(0.75, "positive"),
            risk_view=make_risk_view("low"),
            config=config,
        )
        high_risk_trace = build_decision_trace(
            candidate=candidate,
            macro_view=self.macro_view,
            equity_view=make_equity_view(0.75, "positive"),
            risk_view=make_risk_view("high"),
            config=config,
        )
        self.assertLess(high_risk_trace["final_score"], low_risk_trace["final_score"])

    def test_missing_data_penalty_is_applied(self) -> None:
        config = get_portfolio_manager_config({})
        candidate = make_candidate(
            "AAA",
            sector="Technology",
            quant_score=0.72,
            trailing_pe=None,
            price_to_book=None,
            roe=None,
            business_summary="",
            recent_news_headlines=[],
        )
        trace = build_decision_trace(
            candidate=candidate,
            macro_view=self.macro_view,
            equity_view=make_equity_view(0.70, "neutral"),
            risk_view=make_risk_view("medium"),
            config=config,
        )
        self.assertGreater(trace["missing_data_penalty"], 0.0)
        self.assertIn("insufficient qualitative evidence", trace["main_negative_reasons"])

    def test_data_quality_penalty_is_applied(self) -> None:
        config = get_portfolio_manager_config({})
        candidate = make_candidate("AAA", sector="Technology", quant_score=0.72)
        candidate["data_quality_penalty"] = 0.03
        candidate["data_quality_flags"] = ["roe_extreme_outlier", "roe_winsorized"]
        candidate["data_quality_warning"] = "ROE was winsorized from 127.66% to 50.00%."
        candidate["feature_confidence"] = "low"
        trace = build_decision_trace(
            candidate=candidate,
            macro_view=self.macro_view,
            equity_view=make_equity_view(0.70, "neutral"),
            risk_view=make_risk_view("medium"),
            config=config,
        )
        self.assertGreater(trace["data_quality_penalty"], 0.0)
        self.assertIn("extreme profitability outlier", trace["main_negative_reasons"])
        self.assertEqual(trace["confidence"], "low")

    def test_rejected_stocks_have_specific_reasons(self) -> None:
        candidates = [
            make_candidate("AAA", sector="Technology", quant_score=0.82),
            make_candidate("BBB", sector="Technology", quant_score=0.80),
            make_candidate("CCC", sector="Technology", quant_score=0.78),
            make_candidate("DDD", sector="Health Care", quant_score=0.77),
        ]
        state = self.build_state(
            candidates=candidates,
            equity_scores={
                "AAA": (0.80, "positive"),
                "BBB": (0.78, "positive"),
                "CCC": (0.76, "positive"),
                "DDD": (0.72, "positive"),
            },
            risk_levels={"AAA": "low", "BBB": "medium", "CCC": "medium", "DDD": "medium"},
        )
        result = portfolio_manager_node(state)
        rejected = result["final_decision"]["rejected_stocks"]
        self.assertTrue(rejected)
        for item in rejected:
            self.assertTrue(item["reason"].strip())
            self.assertNotEqual(item["reason"].strip().lower(), "lower score")

    def test_committee_summary_is_generated(self) -> None:
        candidates = [
            make_candidate("AAA", sector="Technology", quant_score=0.82),
            make_candidate("BBB", sector="Financials", quant_score=0.79),
            make_candidate("CCC", sector="Health Care", quant_score=0.76),
        ]
        decision = build_deterministic_portfolio_decision(
            candidates=candidates,
            macro_view=self.macro_view,
            equity_analysis={
                "AAA": make_equity_view(0.8, "positive"),
                "BBB": make_equity_view(0.75, "positive"),
                "CCC": make_equity_view(0.7, "neutral"),
            },
            risk_analysis={
                "AAA": make_risk_view("medium"),
                "BBB": make_risk_view("medium"),
                "CCC": make_risk_view("low"),
            },
            config=get_portfolio_manager_config({}),
        )
        summary = decision["committee_summary"]
        self.assertTrue(summary.strip())
        self.assertIn("Market regime:", summary)
        self.assertIn("Top 3 selected:", summary)

    def test_sector_concentration_penalty_is_applied(self) -> None:
        candidates = [
            make_candidate("AAA", sector="Technology", quant_score=0.82),
            make_candidate("BBB", sector="Technology", quant_score=0.81),
            make_candidate("CCC", sector="Technology", quant_score=0.80),
            make_candidate("DDD", sector="Financials", quant_score=0.79),
        ]
        decision = build_deterministic_portfolio_decision(
            candidates=candidates,
            macro_view=self.macro_view,
            equity_analysis={
                "AAA": make_equity_view(0.82, "positive"),
                "BBB": make_equity_view(0.81, "positive"),
                "CCC": make_equity_view(0.80, "positive"),
                "DDD": make_equity_view(0.79, "positive"),
            },
            risk_analysis={
                "AAA": make_risk_view("low"),
                "BBB": make_risk_view("low"),
                "CCC": make_risk_view("low"),
                "DDD": make_risk_view("low"),
            },
            config=get_portfolio_manager_config({}),
        )
        trace_map = {trace["ticker"]: trace for trace in decision["decision_traces"]}
        self.assertGreaterEqual(trace_map["CCC"]["sector_concentration_penalty"], 0.03)


if __name__ == "__main__":
    unittest.main()
