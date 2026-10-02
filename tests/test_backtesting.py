from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from stock_picker.backtesting import run_backtest


class BacktestingTests(unittest.TestCase):
    def test_missing_price_data_skips_safely(self) -> None:
        result = run_backtest(
            final_decision={"top_picks": [{"ticker": "AAA"}]},
            run_metadata={
                "as_of_date": "2025-01-02",
                "backtest_enabled": True,
                "backtest_yfinance_enabled": False,
            },
        )
        self.assertIn(result["status"], {"skipped", "insufficient_data"})

    def test_local_price_csv_computes_21_day_forward_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            price_path = Path(tmp) / "prices.csv"
            output_dir = Path(tmp) / "outputs"
            with price_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", "ticker", "close"])
                writer.writeheader()
                for idx in range(25):
                    writer.writerow(
                        {
                            "date": f"2025-01-{idx + 1:02d}",
                            "ticker": "AAA",
                            "close": 100 + idx,
                        }
                    )
            result = run_backtest(
                final_decision={"top_picks": [{"ticker": "AAA"}]},
                run_metadata={
                    "as_of_date": "2025-01-01",
                    "backtest_enabled": True,
                    "price_history_source_path": str(price_path),
                    "backtest_output_dir": str(output_dir),
                    "backtest_horizons": [21],
                },
            )
            self.assertEqual(result["status"], "completed")
            self.assertAlmostEqual(
                result["individual_forward_returns"]["AAA"]["21d_return"],
                0.21,
            )
            self.assertTrue((output_dir / "backtest_summary.csv").exists())
            self.assertTrue((output_dir / "backtest_report.md").exists())


if __name__ == "__main__":
    unittest.main()
