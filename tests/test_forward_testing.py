from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stock_picker.forward_testing import write_forward_test_snapshot


class ForwardTestingTests(unittest.TestCase):
    def test_forward_test_enabled_writes_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "run_metadata": {
                    "forward_test_enabled": True,
                    "forward_test_dir": str(Path(tmp) / "forward_tests"),
                    "as_of_date": "2025-01-02",
                    "backtest_horizons": [21, 63],
                },
                "filtered_candidates": [{"ticker": "AAA", "final_score": 0.8}],
                "final_decision": {"top_picks": [{"ticker": "AAA", "final_score": 0.8}]},
            }
            summary = write_forward_test_snapshot(state)
            snapshot_dir = Path(summary["snapshot_path"])

            self.assertTrue((snapshot_dir / "run_metadata.json").exists())
            self.assertTrue((snapshot_dir / "candidate_snapshot.csv").exists())
            self.assertTrue((snapshot_dir / "score_breakdown.csv").exists())
            self.assertTrue((snapshot_dir / "news_documents_used.json").exists())
            self.assertTrue((snapshot_dir / "top_picks.csv").exists())
            self.assertTrue((snapshot_dir / "final_report.md").exists())
            pending = json.loads((snapshot_dir / "realized_returns_pending.json").read_text(encoding="utf-8"))
            self.assertEqual(pending["status"], "pending")

    def test_forward_test_disabled_is_safe(self) -> None:
        summary = write_forward_test_snapshot({"run_metadata": {"forward_test_enabled": False}})
        self.assertEqual(summary["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
