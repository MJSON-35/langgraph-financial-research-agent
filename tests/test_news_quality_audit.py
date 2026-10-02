from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_picker.news_quality_audit import write_news_quality_audit


class NewsQualityAuditTests(unittest.TestCase):
    def test_empty_audit_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_news_quality_audit({"run_metadata": {}}, tmp)
            csv_path = Path(summary["csv_path"])
            md_path = Path(summary["markdown_path"])

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(len(pd.read_csv(csv_path)), 0)

    def test_audit_rows_from_news_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "filtered_candidates": [
                    {
                        "ticker": "AAA",
                        "name": "Alpha",
                        "news_documents_used": [
                            {
                                "headline": "Alpha reports new guidance",
                                "source_domain": "reuters.com",
                                "published_date": "2025-01-01",
                                "quality_score": 0.82,
                                "url": "https://example.com/a",
                            }
                        ],
                    }
                ],
                "run_metadata": {},
            }
            summary = write_news_quality_audit(state, tmp)
            frame = pd.read_csv(summary["csv_path"])

            self.assertEqual(frame.iloc[0]["ticker"], "AAA")
            self.assertEqual(summary["stored_count"], 1)


if __name__ == "__main__":
    unittest.main()
