from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from stock_picker.agents.macro_agent import (
    fred_macro_tool_node,
    macro_agent_node,
    macro_evidence_check_node,
)
from stock_picker.graph import build_graph
from stock_picker.main import DEMO_CONFIG_PATH, build_demo_state, load_demo_config
from stock_picker.news_rag import (
    news_evidence_check_node,
    news_evidence_node,
    tavily_news_tool_node,
)
from stock_picker.research_tools import ToolResult, build_tool_audit
from stock_picker.routing import route_macro, route_news


def _candidate_with_documents(count: int) -> dict[str, object]:
    return {
        "ticker": "AAA",
        "name": "AAA Corp",
        "sector": "Technology",
        "news_documents_used": [
            {
                "headline": f"AAA evidence {index}",
                "quality_score": 0.9,
                "published_date": datetime.now(timezone.utc).date().isoformat(),
            }
            for index in range(count)
        ],
    }


class NewsRoutingTests(unittest.TestCase):
    def test_tool_audit_masks_sensitive_metadata_defensively(self) -> None:
        result = ToolResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error_type="provider_error",
            metadata={"access_token": "test-only-sensitive-value", "result_count": 0},
        )
        audit = build_tool_audit(result, reason="test_failure")

        self.assertEqual(audit["metadata"]["access_token"], "***")
        self.assertNotIn("test-only-sensitive-value", str(audit))

    def test_sufficient_news_skips_tavily_and_records_reason(self) -> None:
        state = {
            "filtered_candidates": [_candidate_with_documents(2)],
            "run_metadata": {
                "news_tavily_enabled": True,
                "news_external_min_usable_articles": 2,
            },
        }
        result = news_evidence_check_node(state)

        self.assertEqual(route_news(result), "news_finalize")
        audit = result["tool_audit_trail"][-1]
        self.assertFalse(audit["invoked"])
        self.assertEqual(audit["reason"], "existing_news_evidence_sufficient")

    def test_insufficient_news_invokes_tavily_and_merges_evidence(self) -> None:
        state = {
            "filtered_candidates": [{"ticker": "AAA", "name": "AAA Corp", "sector": "Technology"}],
            "run_metadata": {
                "news_tavily_enabled": True,
                "news_vectorstore_enabled": False,
                "news_external_min_usable_articles": 1,
                "news_min_quality_score": 0.40,
                "news_low_quality_threshold": 0.20,
            },
        }
        state = news_evidence_node(state)
        state = news_evidence_check_node(state)
        self.assertEqual(route_news(state), "tavily_news_tool")

        record = {
            "ticker": "AAA",
            "company_name": "AAA Corp",
            "headline": "AAA Corp reports earnings growth",
            "summary": "AAA Corp raised guidance after strong demand.",
            "url": "https://example.com/aaa",
            "source": "tavily",
            "source_domain": "example.com",
            "published_date": datetime.now(timezone.utc).date().isoformat(),
            "query": "AAA Corp latest earnings news",
        }
        with patch("stock_picker.news_rag.fetch_tavily_news_records", return_value=([record], "fetched")) as search:
            state = tavily_news_tool_node(state)

        search.assert_called_once()
        candidate = state["filtered_candidates"][0]
        self.assertEqual(candidate["news_quality_summary"]["retrieved_from_tavily"], 1)
        self.assertIn("AAA Corp reports earnings growth", candidate["recent_news_headlines"])
        self.assertTrue(state["tool_audit_trail"][-1]["invoked"])

    def test_tavily_failure_continues_and_records_safe_error(self) -> None:
        state = {
            "filtered_candidates": [{"ticker": "AAA", "name": "AAA Corp"}],
            "run_metadata": {
                "news_tavily_enabled": True,
                "news_vectorstore_enabled": False,
                "news_external_min_usable_articles": 1,
            },
        }
        state = news_evidence_node(state)
        state = news_evidence_check_node(state)
        with patch("stock_picker.news_rag.fetch_tavily_news_records", return_value=([], "error")):
            state = tavily_news_tool_node(state)

        self.assertEqual(len(state["filtered_candidates"]), 1)
        audit = state["tool_audit_trail"][-1]
        self.assertFalse(audit["success"])
        self.assertEqual(audit["error_type"], "error")
        self.assertNotIn("api_key", str(audit).lower())


class MacroRoutingTests(unittest.TestCase):
    def test_structured_macro_evidence_skips_fred(self) -> None:
        state = {
            "run_metadata": {
                "macro_mode": "fred",
                "macro_context": {
                    "growth_signal": "stable",
                    "inflation_signal": "cooling",
                    "policy_signal": "unchanged",
                    "volatility_signal": "moderate",
                },
            }
        }
        result = macro_evidence_check_node(state)

        self.assertEqual(route_macro(result), "macro_agent")
        self.assertEqual(result["tool_audit_trail"][-1]["reason"], "existing_macro_context_sufficient")

    def test_insufficient_macro_evidence_invokes_fred_tool(self) -> None:
        state = {
            "filtered_candidates": [{"ticker": "AAA", "sector": "Technology"}],
            "run_metadata": {
                "macro_mode": "fred",
                "fred_series_map": {"growth": "GDPC1"},
            },
        }
        tool_result = ToolResult(
            tool_name="fred_macro",
            success=True,
            result={
                "growth_signal": "stable",
                "inflation_signal": "cooling",
                "policy_signal": "unchanged",
                "volatility_signal": "moderate",
            },
            metadata={"series_count": 1, "requested_signals": ["growth"]},
        )
        with patch.dict(os.environ, {"FRED_API_KEY": "test-only-placeholder"}, clear=True):
            state = macro_evidence_check_node(state)
            self.assertEqual(route_macro(state), "fred_macro_tool")
            with patch("stock_picker.agents.macro_agent.FredMacroTool.invoke", return_value=tool_result) as invoke:
                state = fred_macro_tool_node(state)
        invoke.assert_called_once_with({"growth": "GDPC1"})
        state = macro_agent_node(state)
        self.assertEqual(state["run_metadata"]["macro_provider"], "fred_macro_tool")
        self.assertTrue(state["tool_audit_trail"][-1]["success"])

    def test_fred_failure_continues_with_insufficient_fallback(self) -> None:
        state = {
            "filtered_candidates": [{"ticker": "AAA", "sector": "Technology"}],
            "run_metadata": {
                "macro_mode": "fred",
                "fred_series_map": {"growth": "GDPC1"},
            },
        }
        failed = ToolResult(
            tool_name="fred_macro",
            success=False,
            result={},
            error_type="network_error",
            metadata={"series_count": 1},
        )
        with patch.dict(os.environ, {"FRED_API_KEY": "test-only-placeholder"}, clear=True):
            state = macro_evidence_check_node(state)
            with patch("stock_picker.agents.macro_agent.FredMacroTool.invoke", return_value=failed):
                state = fred_macro_tool_node(state)
        state = macro_agent_node(state)

        self.assertEqual(state["macro_view"]["status"], "insufficient_input")
        self.assertEqual(state["tool_audit_trail"][-1]["error_type"], "network_error")
        self.assertNotIn("test-only-placeholder", str(state))


class ConditionalGraphTests(unittest.TestCase):
    def test_compiled_graph_contains_both_conditional_tool_branches(self) -> None:
        mermaid = build_graph().get_graph().draw_mermaid()
        for node_name in (
            "news_evidence_check",
            "tavily_news_tool",
            "macro_evidence_check",
            "fred_macro_tool",
        ):
            self.assertIn(node_name, mermaid)

    def test_offline_demo_does_not_invoke_external_tools(self) -> None:
        local_tmp = Path.cwd() / ".tmp"
        local_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_tmp) as tmp, patch.dict(os.environ, {}, clear=True):
            config = load_demo_config(DEMO_CONFIG_PATH)
            state = build_demo_state(config)
            state["run_metadata"]["output_dir"] = str(Path(tmp) / "outputs")
            with patch(
                "stock_picker.news_rag.TavilyNewsSearchTool.invoke",
                side_effect=AssertionError("Tavily must stay offline"),
            ), patch(
                "stock_picker.agents.macro_agent.FredMacroTool.invoke",
                side_effect=AssertionError("FRED must stay offline"),
            ):
                result = build_graph().invoke(state)

        self.assertTrue(result.get("final_decision", {}).get("top_picks"))
        self.assertTrue(all(not item["invoked"] for item in result.get("tool_audit_trail", [])))


if __name__ == "__main__":
    unittest.main()
