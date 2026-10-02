"""LangGraph workflow definition for the MVP stock picker pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_picker.backtesting import backtesting_node
from stock_picker.agents.data_agent import data_agent_node
from stock_picker.agents.equity_agent import equity_agent_node
from stock_picker.agents.macro_agent import macro_agent_node
from stock_picker.agents.portfolio_manager import portfolio_manager_node
from stock_picker.agents.risk_agent import risk_agent_node
from stock_picker.news_rag import news_rag_node
from stock_picker.reporting import instrument_node
from stock_picker.state import StockPickerState


def build_graph():
    """Compile the minimal sequential workflow used by the prototype."""
    workflow = StateGraph(StockPickerState)

    workflow.add_node("data_agent", instrument_node("data_agent", data_agent_node))
    workflow.add_node("news_rag", instrument_node("news_rag", news_rag_node))
    workflow.add_node("macro_agent", instrument_node("macro_agent", macro_agent_node))
    workflow.add_node("equity_agent", instrument_node("equity_agent", equity_agent_node))
    workflow.add_node("risk_agent", instrument_node("risk_agent", risk_agent_node))
    workflow.add_node("portfolio_manager", instrument_node("portfolio_manager", portfolio_manager_node))
    workflow.add_node("backtesting", instrument_node("backtesting", backtesting_node))

    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "news_rag")
    workflow.add_edge("news_rag", "macro_agent")
    workflow.add_edge("macro_agent", "equity_agent")
    workflow.add_edge("equity_agent", "risk_agent")
    workflow.add_edge("risk_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", "backtesting")
    workflow.add_edge("backtesting", END)

    # TODO: If the prototype grows, consider fan-out/fan-in patterns so some
    # agents can work in parallel instead of strictly sequentially.
    return workflow.compile()
