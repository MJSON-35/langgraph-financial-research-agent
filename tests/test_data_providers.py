from __future__ import annotations

import builtins
import unittest
from unittest.mock import Mock, patch

from stock_picker.data_providers.krx_provider import fetch_krx_price_history
from stock_picker.data_providers.us_yfinance_provider import fetch_us_price_history_and_fundamentals
from stock_picker.providers.data_providers import RealDataFeatureProvider, resolve_feature_provider


class DataProviderTests(unittest.TestCase):
    def test_krx_provider_skips_when_pykrx_import_fails(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pykrx" or name.startswith("pykrx."):
                raise ImportError("pykrx unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            frame, status = fetch_krx_price_history(["005930"], start_date="2025-01-01", end_date="2025-02-01")

        self.assertTrue(frame.empty)
        self.assertEqual(status["status"], "skipped")

    def test_yfinance_provider_skips_when_download_fails(self) -> None:
        fake_yf = Mock()
        fake_yf.download.side_effect = RuntimeError("network unavailable")
        fake_yf.Ticker.return_value.get_info.side_effect = RuntimeError("network unavailable")

        with patch.dict("sys.modules", {"yfinance": fake_yf}):
            prices, fundamentals, status = fetch_us_price_history_and_fundamentals(
                ["AAPL"],
                start_date="2025-01-01",
                end_date="2025-02-01",
            )

        self.assertTrue(prices.empty)
        self.assertEqual(fundamentals.iloc[0]["ticker"], "AAPL")
        self.assertIn(status["status"], {"completed", "skipped"})

    def test_real_data_enabled_prioritizes_live_provider_over_snapshot(self) -> None:
        provider, label = resolve_feature_provider(
            {
                "run_metadata": {
                    "real_data_enabled": True,
                    "feature_snapshot_path": "data/features/meaningful_latest_features_v3.csv",
                }
            }
        )
        self.assertIsInstance(provider, RealDataFeatureProvider)
        self.assertEqual(label, "real_data_feature_provider:pykrx_yfinance")


if __name__ == "__main__":
    unittest.main()
