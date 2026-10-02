from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from stock_picker.graph import build_graph
from stock_picker.main import build_sample_state
from stock_picker.config import get_financial_data_quality_config
from stock_picker.financial_data_quality import (
    build_quality_score,
    clean_roe,
    validate_financial_features,
)
from stock_picker.report_generator import generate_final_reports


class FinancialDataQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_financial_data_quality_config({})
        self.test_root = Path(__file__).resolve().parents[1] / "test_output"
        self.test_root.mkdir(parents=True, exist_ok=True)

    def test_clean_roe_caps_extreme_outlier(self) -> None:
        cleaned, flags = clean_roe(1.2766, self.config)
        self.assertEqual(cleaned, 0.50)
        self.assertIn("roe_extreme_outlier", flags)
        self.assertIn("roe_winsorized", flags)

    def test_clean_roe_caps_high_value(self) -> None:
        cleaned, flags = clean_roe(0.8457, self.config)
        self.assertEqual(cleaned, 0.50)
        self.assertNotIn("roe_extreme_outlier", flags)
        self.assertIn("roe_winsorized", flags)

    def test_clean_roe_missing_flag(self) -> None:
        cleaned, flags = clean_roe(None, self.config)
        self.assertIsNone(cleaned)
        self.assertIn("roe_missing", flags)

    def test_missing_pe_lowers_valuation_confidence(self) -> None:
        record = validate_financial_features(
            {
                "ticker": "AAA",
                "trailing_pe": None,
                "price_to_book": 2.0,
                "roe": 0.15,
                "ret_12m": 0.20,
                "ret_1m": 0.03,
                "avg_volume_20d": 100000,
                "market_cap": 1_000_000_000,
                "available_price_days": 260,
                "has_full_12m_history": True,
            },
            self.config,
        )
        self.assertEqual(record["valuation_confidence"], "medium")

    def test_momentum_outlier_flag_is_created(self) -> None:
        record = validate_financial_features(
            {
                "ticker": "AAA",
                "trailing_pe": 12,
                "price_to_book": 2.0,
                "roe": 0.15,
                "ret_12m": 5.6,
                "ret_1m": 0.03,
                "avg_volume_20d": 100000,
                "market_cap": 1_000_000_000,
                "available_price_days": 260,
                "has_full_12m_history": True,
            },
            self.config,
        )
        self.assertIn("momentum_12m_outlier", record["data_quality_flags"])

    def test_insufficient_price_history_flag_is_created(self) -> None:
        record = validate_financial_features(
            {
                "ticker": "AAA",
                "trailing_pe": 12,
                "price_to_book": 2.0,
                "roe": 0.15,
                "ret_12m": 0.40,
                "ret_1m": 0.03,
                "avg_volume_20d": 100000,
                "market_cap": 1_000_000_000,
                "available_price_days": 120,
                "has_full_12m_history": False,
            },
            self.config,
        )
        self.assertFalse(record["has_full_12m_history"])
        self.assertIn("insufficient_12m_price_history", record["data_quality_flags"])

    def test_quality_score_uses_cleaned_roe(self) -> None:
        features = {
            "cleaned_roe": 0.50,
            "raw_roe": 1.2766,
            "data_quality_flags": ["roe_extreme_outlier", "roe_winsorized"],
        }
        score, confidence = build_quality_score(features, self.config)
        self.assertIsNotNone(score)
        self.assertLess(score, 1.0)
        self.assertEqual(confidence, "medium")

    def test_final_report_includes_data_quality_warning(self) -> None:
        root = self.test_root / f"report_{uuid.uuid4().hex}"
        step_dir = root / "step_summaries"
        step_dir.mkdir(parents=True, exist_ok=True)

        (step_dir / "01_data_agent.json").write_text(
            json.dumps(
                {
                    "stage_name": "data_agent",
                    "selected_candidate_count": 1,
                    "top_candidate_tickers": ["AAA"],
                    "validation_summary": {"missing_by_column": {"trailing_pe": 1}},
                }
            ),
            encoding="utf-8",
        )
        (step_dir / "05_portfolio_manager.json").write_text(
            json.dumps(
                {
                    "stage_name": "portfolio_manager",
                    "top_3_picks": [
                        {
                            "ticker": "AAA",
                            "name": "AAA Corp",
                            "sector": "Technology",
                            "market": "TEST",
                            "quant_score": 0.7,
                            "quality_score": 0.8,
                            "quality_score_cleaned": 0.5,
                            "ret_1m": 0.1,
                            "ret_12m": 0.2,
                            "raw_ret_1m": 0.1,
                            "raw_ret_12m": 0.2,
                            "volatility_60d": 0.3,
                            "max_drawdown_1y": 0.2,
                            "avg_volume_20d": 100000,
                            "raw_avg_volume_20d": 100000,
                            "cleaned_avg_volume_20d": 100000,
                            "trailing_pe": 10,
                            "raw_trailing_pe": 10,
                            "cleaned_trailing_pe": 10,
                            "price_to_book": 2,
                            "raw_price_to_book": 2,
                            "cleaned_price_to_book": 2,
                            "roe": 0.5,
                            "raw_roe": 1.2766,
                            "cleaned_roe": 0.5,
                            "data_quality_flags": ["roe_extreme_outlier", "roe_winsorized"],
                            "feature_confidence": "medium",
                            "quality_score_confidence": "medium",
                            "valuation_confidence": "high",
                            "has_full_12m_history": True,
                            "momentum_12m_confidence": "high",
                            "data_quality_warning": "ROE was winsorized from 127.66% to 50.00%; quality score should be interpreted with caution.",
                            "data_quality_penalty": 0.03,
                            "final_score": 0.55,
                            "confidence": "medium",
                            "rationale": "Test rationale",
                        }
                    ],
                    "rejected_stocks": [],
                    "committee_summary": "Test committee summary",
                    "confidence_level": "medium",
                    "ranking_method": "test method",
                    "final_score_breakdown": {},
                    "decision_traces": [],
                }
            ),
            encoding="utf-8",
        )

        paths = generate_final_reports(output_dir=root, run_metadata={"scenario_name": "unit-test"})
        markdown = paths["markdown"].read_text(encoding="utf-8")
        self.assertIn("ROE was winsorized", markdown)

    def test_end_to_end_pipeline_still_runs(self) -> None:
        app = build_graph()
        state = build_sample_state()
        state["run_metadata"]["use_ollama"] = False
        run_root = self.test_root / f"e2e_{uuid.uuid4().hex}"
        run_root.mkdir(parents=True, exist_ok=True)
        state["run_metadata"]["output_dir"] = str(run_root)
        result = app.invoke(state)
        self.assertTrue(result.get("final_decision"))
        self.assertIn("top_picks", result["final_decision"])
        self.assertIn("report", result)


if __name__ == "__main__":
    unittest.main()
