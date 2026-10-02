from __future__ import annotations

import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from stock_picker.news_rag import attach_news_to_candidates
from stock_picker.news_vectorstore import (
    build_news_document,
    build_local_embedding_model,
    deduplicate_news_items,
    is_news_recent,
    score_news_quality,
)
from stock_picker.tavily_news import fetch_tavily_news_records


def _days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


class FakeVectorStore:
    def __init__(self, records=None) -> None:
        self.records = records or []
        self.upserts = []
        self.queries = 0

    def upsert(self, ids, documents, metadatas) -> None:
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})

    def query(self, query_texts, n_results, where, include):
        self.queries += 1
        ticker = where.get("ticker")
        filtered = [item for item in self.records if item["metadata"].get("ticker") == ticker]
        return {
            "documents": [[item["document"] for item in filtered[:n_results]]],
            "metadatas": [[item["metadata"] for item in filtered[:n_results]]],
            "distances": [[0.1 for _ in filtered[:n_results]]],
        }


class NewsRagTests(unittest.TestCase):
    def test_news_unavailable_fallback_keeps_candidates(self) -> None:
        candidates = [{"ticker": "AAA", "name": "AAA Corp"}]
        result = attach_news_to_candidates(
            candidates,
            {"use_ollama": False, "news_vectorstore_enabled": False},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["news_rag_summary"], "news unavailable")
        self.assertEqual(result[0]["news_sentiment_hint"], "unknown")

    def test_local_csv_attaches_summary_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ticker", "headline", "summary", "source"])
                writer.writeheader()
                writer.writerow(
                    {
                        "ticker": "AAA",
                        "headline": "AAA posts earnings beat with strong demand",
                        "summary": "Management raised guidance.",
                        "source": "unit-test",
                    }
                )
            result = attach_news_to_candidates(
                [{"ticker": "AAA", "name": "AAA Corp"}],
                {
                    "news_source_path": str(path),
                    "use_ollama": False,
                    "news_vectorstore_enabled": False,
                },
            )
        self.assertIn("earnings beat", result[0]["recent_news_headlines"][0])
        self.assertEqual(result[0]["news_sentiment_hint"], "positive")
        self.assertIn("local_news_file", result[0]["source_notes"])

    def test_vectorstore_does_not_store_quant_fields(self) -> None:
        candidate = {
            "ticker": "AAA",
            "name": "AAA Corp",
            "price": 100,
            "roe": 0.2,
            "quant_score": 0.9,
        }
        news = {
            "ticker": "AAA",
            "headline": "AAA Corp wins new contract",
            "summary": "AAA Corp announced expansion.",
            "url": "https://reuters.com/markets/aaa",
            "source_domain": "reuters.com",
            "published_date": _days_ago(2),
            "price": 100,
            "roe": 0.2,
            "quant_score": 0.9,
        }
        quality = score_news_quality(news, candidate, {"use_ollama": False})
        doc = build_news_document(news, candidate, quality)
        self.assertNotIn("price", doc.metadata)
        self.assertNotIn("roe", doc.metadata)
        self.assertNotIn("quant_score", doc.metadata)
        self.assertIn("quality_score", doc.metadata)

    def test_deduplicate_news_by_url_and_headline(self) -> None:
        items = [
            {
                "ticker": "AAA",
                "headline": "AAA Corp posts earnings beat",
                "url": "https://example.com/a?utm_source=x",
                "published_date": _days_ago(1),
                "quality_score": 0.7,
            },
            {
                "ticker": "AAA",
                "headline": "AAA Corp posts earnings beat",
                "url": "https://example.com/a",
                "published_date": _days_ago(1),
                "quality_score": 0.8,
            },
            {
                "ticker": "AAA",
                "headline": "AAA Corp posts earnings beat today",
                "url": "https://example.com/b",
                "published_date": _days_ago(2),
                "quality_score": 0.6,
            },
        ]
        deduped = deduplicate_news_items(items)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["quality_score"], 0.8)

    def test_quality_score_filters_bad_company_match(self) -> None:
        quality = score_news_quality(
            {
                "headline": "Unrelated retailer reports store traffic",
                "summary": "No matching company mention.",
                "source_domain": "reuters.com",
                "published_date": _days_ago(1),
                "query": "AAA Corp AAA latest news",
            },
            {"ticker": "AAA", "name": "AAA Corp"},
            {"news_require_company_match": True, "news_low_quality_threshold": 0.40},
        )
        self.assertLess(quality["quality_score"], 0.40)

    def test_recency_filters_old_news(self) -> None:
        self.assertFalse(is_news_recent({"published_date": _days_ago(45)}, {"tavily_days": 30}))
        self.assertTrue(is_news_recent({"published_date": _days_ago(7)}, {"tavily_days": 30}))

    def test_tavily_skips_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "stock_picker.tavily_news.load_project_env", return_value=False
        ):
            records, status = fetch_tavily_news_records(
                {"ticker": "AAA", "name": "AAA Corp"},
                {"news_tavily_enabled": True},
            )
        self.assertEqual(records, [])
        self.assertEqual(status, "missing_api_key")

    def test_news_rag_uses_vectorstore_before_tavily(self) -> None:
        fake_store = FakeVectorStore(
            [
                {
                    "document": "AAA Corp reports strong demand\nManagement raised guidance",
                    "metadata": {
                        "ticker": "AAA",
                        "company_name": "AAA Corp",
                        "source_domain": "reuters.com",
                        "published_date": _days_ago(1),
                        "quality_score": 0.9,
                    },
                }
            ]
            * 5
        )
        with patch("stock_picker.news_rag.get_news_vectorstore", return_value=fake_store), patch(
            "stock_picker.news_rag.fetch_tavily_news_records"
        ) as tavily:
            result = attach_news_to_candidates(
                [{"ticker": "AAA", "name": "AAA Corp"}],
                {"news_vectorstore_enabled": True, "news_retrieval_top_k": 5, "news_tavily_enabled": True},
            )
        tavily.assert_not_called()
        self.assertEqual(result[0]["news_quality_summary"]["retrieved_from_vectorstore"], 5)
        self.assertEqual(result[0]["news_quality_summary"]["tavily_status"], "skipped_sufficient_news")

    def test_local_embedding_fallback(self) -> None:
        class FakeEmbeddings:
            def __init__(self, model_name, fallback_model_name=None) -> None:
                raise RuntimeError("model load failed")

        with patch("stock_picker.news_vectorstore.LocalSentenceTransformerEmbeddings", FakeEmbeddings):
            self.assertIsNone(
                build_local_embedding_model(
                    {
                        "news_embedding_model": "intfloat/multilingual-e5-small",
                        "news_embedding_fallback_model": "sentence-transformers/all-MiniLM-L6-v2",
                    }
                )
            )

    def test_news_quality_summary_added_to_candidate(self) -> None:
        result = attach_news_to_candidates(
            [
                {
                    "ticker": "AAA",
                    "name": "AAA Corp",
                    "recent_news_headlines": ["AAA Corp posts earnings beat"],
                    "recent_news_summaries": ["AAA Corp raised guidance."],
                }
            ],
            {"use_ollama": False, "news_vectorstore_enabled": False},
        )
        self.assertIn("news_quality_summary", result[0])
        self.assertIn("news_documents_used", result[0])
        self.assertGreaterEqual(result[0]["news_quality_summary"]["stored_new_documents"], 0)


if __name__ == "__main__":
    unittest.main()
