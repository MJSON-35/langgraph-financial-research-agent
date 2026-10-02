"""Small typed wrappers around optional external research providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError

from stock_picker.providers.macro_providers import FredMacroProvider
from stock_picker.security import sanitize_for_export
from stock_picker.tavily_news import fetch_tavily_news_records


@dataclass(frozen=True)
class ToolResult:
    """Uniform, secret-safe result returned by every research tool."""

    tool_name: str
    success: bool
    result: Any
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchTool(Protocol):
    """Minimal interface for traceable provider invocation."""

    name: str

    def invoke(self, *args: Any, **kwargs: Any) -> ToolResult: ...


class TavilyNewsSearchTool:
    """Typed adapter over the existing Tavily news provider."""

    name = "tavily_news_search"

    def __init__(
        self,
        search_fn: Callable[
            [dict[str, Any], dict[str, Any]],
            tuple[list[dict[str, Any]], str],
        ] = fetch_tavily_news_records,
    ) -> None:
        self._search_fn = search_fn

    def invoke(self, candidate: dict[str, Any], run_metadata: dict[str, Any]) -> ToolResult:
        ticker = str(candidate.get("ticker", "")).strip()
        try:
            records, provider_status = self._search_fn(candidate, run_metadata)
        except Exception as exc:  # Defensive boundary around an optional provider.
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=[],
                error_type=_classify_provider_error(exc),
                metadata={"ticker": ticker, "result_count": 0, "provider_status": "error"},
            )
        success = provider_status == "fetched"
        return ToolResult(
            tool_name=self.name,
            success=success,
            result=records,
            error_type=None if success else provider_status,
            metadata={
                "ticker": ticker,
                "result_count": len(records),
                "provider_status": provider_status,
            },
        )


class FredMacroTool:
    """Typed adapter over the existing FRED macro provider."""

    name = "fred_macro"

    def invoke(self, series_map: dict[str, str]) -> ToolResult:
        safe_metadata = {
            "series_count": len(series_map),
            "requested_signals": sorted(str(key) for key in series_map),
        }
        try:
            context = FredMacroProvider(series_map).get_macro_context()
        except Exception as exc:  # Provider errors are reduced to a category only.
            return ToolResult(
                tool_name=self.name,
                success=False,
                result={},
                error_type=_classify_provider_error(exc),
                metadata=safe_metadata,
            )
        return ToolResult(
            tool_name=self.name,
            success=True,
            result=context,
            metadata=safe_metadata,
        )


def build_tool_audit(
    result: ToolResult,
    *,
    reason: str,
    invoked: bool = True,
) -> dict[str, Any]:
    """Convert a tool result to export-safe audit metadata."""
    return sanitize_for_export(
        {
            "tool_name": result.tool_name,
            "invoked": invoked,
            "reason": reason,
            "success": result.success if invoked else None,
            "error_type": result.error_type if invoked else None,
            "metadata": result.metadata,
        }
    )


def build_skipped_tool_audit(tool_name: str, reason: str) -> dict[str, Any]:
    """Build an audit entry for a tool that routing deliberately skipped."""
    return {
        "tool_name": tool_name,
        "invoked": False,
        "reason": reason,
        "success": None,
        "error_type": None,
        "metadata": {},
    }


def _classify_provider_error(exc: Exception) -> str:
    if isinstance(exc, (HTTPError, URLError, TimeoutError, ConnectionError)):
        return "network_error"
    if isinstance(exc, ValueError):
        return "invalid_request"
    if isinstance(exc, RuntimeError):
        return "provider_unavailable"
    return "provider_error"
