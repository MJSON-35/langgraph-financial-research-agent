from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_picker.data_providers.feature_builder import (
    build_features_from_price_history,
    build_meaningful_run_check,
)
from stock_picker.data_providers.snapshot_io import (
    load_feature_snapshot,
    load_price_history_snapshot,
    save_feature_snapshot,
    save_price_history_snapshot,
)


class FeatureBuilderTests(unittest.TestCase):
    def test_price_history_builds_standard_features(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="D")
        price_history = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAA"] * len(dates),
                "close": [100 + idx for idx in range(len(dates))],
                "volume": [1000 + idx for idx in range(len(dates))],
                "market": ["US"] * len(dates),
                "source": ["unit_test"] * len(dates),
            }
        )
        fundamentals = pd.DataFrame(
            [{"ticker": "AAA", "market_cap": 1000, "trailing_pe": 12.0, "price_to_book": 2.0, "roe": 0.15}]
        )

        features = build_features_from_price_history(
            price_history,
            fundamentals=fundamentals,
            as_of_date="2024-09-16",
        )

        self.assertEqual(len(features), 1)
        row = features.iloc[0]
        self.assertGreater(row["ret_1m"], 0)
        self.assertGreater(row["ret_12m"], 0)
        self.assertGreater(row["volatility_60d"], 0)
        self.assertLessEqual(row["max_drawdown_1y"], 0)
        self.assertTrue(row["has_full_12m_history"])

    def test_snapshot_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "features.csv"
            price_path = Path(tmp) / "prices.csv"
            features = pd.DataFrame([{"ticker": "AAA", "price": 100.0}])
            prices = pd.DataFrame([{"date": "2025-01-01", "ticker": "AAA", "close": 100.0}])

            save_feature_snapshot(features, feature_path)
            save_price_history_snapshot(prices, price_path)

            self.assertEqual(load_feature_snapshot(feature_path).iloc[0]["ticker"], "AAA")
            self.assertEqual(load_price_history_snapshot(price_path).iloc[0]["ticker"], "AAA")

    def test_meaningful_run_check_false_when_coverage_missing(self) -> None:
        features = pd.DataFrame([{"ticker": "AAA", "price": None, "ret_1m": None, "trailing_pe": None}])
        check = build_meaningful_run_check(features, backtest_status="skipped")
        self.assertFalse(check["is_meaningful_run"])
        self.assertIn("price coverage below 90%", check["reasons"])


if __name__ == "__main__":
    unittest.main()
