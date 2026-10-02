"""LangGraph workflow definition for the MVP stock picker pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_picker.backtesting import backtesting_node
from stock_picker.agents.data_agent import data_agent_node
from stock_picker.agents.equity_agent import equity_agent_node
from stock_picker.agents.macro_agent import (
    fred_macro_tool_node,
    macro_agent_node,
    macro_evidence_check_node,
)
from stock_picker.agents.portfolio_manager import portfolio_manager_node
from stock_picker.agents.risk_agent import risk_agent_node
from stock_picker.news_rag import (
    news_evidence_check_node,
    news_evidence_node,
    news_finalize_node,
    tavily_news_tool_node,
)
from stock_picker.reporting import instrument_node
from stock_picker.routing import route_macro, route_news
from stock_picker.state import StockPickerState


def build_graph():
    """Compile the workflow with deterministic evidence-based tool routing."""
    workflow = StateGraph(StockPickerState)

    workflow.add_node("data_agent", instrument_node("data_agent", data_agent_node))
    workflow.add_node("news_evidence", news_evidence_node)
    workflow.add_node("news_evidence_check", news_evidence_check_node)
    workflow.add_node("tavily_news_tool", tavily_news_tool_node)
    workflow.add_node("news_finalize", instrument_node("news_rag", news_finalize_node))
    workflow.add_node("macro_evidence_check", macro_evidence_check_node)
    workflow.add_node("fred_macro_tool", fred_macro_tool_node)
    workflow.add_node("macro_agent", instrument_node("macro_agent", macro_agent_node))
    workflow.add_node("equity_agent", instrument_node("equity_agent", equity_agent_node))
    workflow.add_node("risk_agent", instrument_node("risk_agent", risk_agent_node))
    workflow.add_node("portfolio_manager", instrument_node("portfolio_manager", portfolio_manager_node))
    workflow.add_node("backtesting", instrument_node("backtesting", backtesting_node))

    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "news_evidence")
    workflow.add_edge("news_evidence", "news_evidence_check")
    workflow.add_conditional_edges(
        "news_evidence_check",
        route_news,
        {
            "news_finalize": "news_finalize",
            "tavily_news_tool": "tavily_news_tool",
        },
    )
    workflow.add_edge("tavily_news_tool", "news_finalize")
    workflow.add_edge("news_finalize", "macro_evidence_check")
    workflow.add_conditional_edges(
        "macro_evidence_check",
        route_macro,
        {
            "macro_agent": "macro_agent",
            "fred_macro_tool": "fred_macro_tool",
        },
    )
    workflow.add_edge("fred_macro_tool", "macro_agent")
    workflow.add_edge("macro_agent", "equity_agent")
    workflow.add_edge("equity_agent", "risk_agent")
    workflow.add_edge("risk_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", "backtesting")
    workflow.add_edge("backtesting", END)

    return workflow.compile()
