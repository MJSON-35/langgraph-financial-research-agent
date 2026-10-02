"""Forward-testing ledger artifacts for point-in-time research runs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_picker.export_utils import PROJECT_ROOT
from stock_picker.security import sanitize_for_export


def write_forward_test_snapshot(state: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Persist the run's point-in-time inputs and decisions when enabled."""
    run_metadata = state.get("run_metadata", {})
    if not bool(run_metadata.get("forward_test_enabled", False)):
        return {"enabled": False, "status": "disabled"}

    base_dir = _resolve_dir(run_metadata.get("forward_test_dir", "outputs/forward_tests"))
    run_date = str(run_metadata.get("as_of_date") or datetime.now(timezone.utc).date().isoformat())
    snapshot_dir = base_dir / run_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    _write_json(snapshot_dir / "run_metadata.json", _redact_run_metadata(run_metadata))
    _write_csv(snapshot_dir / "candidate_snapshot.csv", state.get("filtered_candidates", []))
    _write_csv(snapshot_dir / "top_picks.csv", state.get("final_decision", {}).get("top_picks", []))
    _write_score_breakdown(snapshot_dir / "score_breakdown.csv", state.get("final_decision", {}))
    _write_json(snapshot_dir / "news_documents_used.json", _collect_news_documents(state.get("filtered_candidates", [])))
    (snapshot_dir / "final_report.md").write_text(_build_snapshot_report(state), encoding="utf-8")

    horizons = _normalize_horizons(run_metadata.get("backtest_horizons", [21, 63]))
    pending = {
        "horizons": horizons,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_evaluation_dates": _expected_evaluation_dates(run_date, horizons),
        "note": "realized returns should be filled after horizon has passed",
    }
    _write_json(snapshot_dir / "realized_returns_pending.json", pending)
    return {
        "enabled": True,
        "status": "written",
        "snapshot_path": str(snapshot_dir),
        "pending_realized_returns_file": str(snapshot_dir / "realized_returns_pending.json"),
        "horizons": horizons,
    }


def _resolve_dir(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(sanitize_for_export(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(sanitize_for_export(records))
    frame.to_csv(path, index=False, encoding="utf-8")


def _write_score_breakdown(path: Path, final_decision: dict[str, Any]) -> None:
    breakdown = final_decision.get("final_score_breakdown", {}) or {}
    rows = []
    for ticker, record in breakdown.items():
        row = {"ticker": ticker}
        row.update(record)
        rows.append(row)
    _write_csv(path, rows)


def _collect_news_documents(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        ticker = str(candidate.get("ticker", "UNKNOWN"))
        result[ticker] = candidate.get("news_documents_used", []) or []
    return result


def _redact_run_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    return sanitize_for_export(run_metadata)


def _build_snapshot_report(state: dict[str, Any]) -> str:
    final_decision = state.get("final_decision", {})
    lines = [
        "# Forward Test Snapshot",
        "",
        f"- Created at: {datetime.now(timezone.utc).isoformat()}",
        f"- Committee summary: {final_decision.get('committee_summary', 'Unavailable')}",
        "- Purpose: freeze the inputs, news evidence, agent outputs, and final ranking seen at run time.",
        "- Realized returns are intentionally pending until future horizons have passed.",
        "",
        "## Top Picks",
    ]
    for pick in final_decision.get("top_picks", [])[:3]:
        lines.append(f"- {pick.get('ticker', 'UNKNOWN')}: final_score={pick.get('final_score', 'n/a')}")
    if not final_decision.get("top_picks"):
        lines.append("- None")
    return "\n".join(lines)


def _normalize_horizons(value: Any) -> list[int]:
    if not isinstance(value, list):
        return [21, 63]
    horizons = []
    for item in value:
        try:
            horizon = int(item)
        except Exception:
            continue
        if horizon > 0:
            horizons.append(horizon)
    return horizons or [21, 63]


def _expected_evaluation_dates(run_date: str, horizons: list[int]) -> dict[str, str]:
    try:
        start = datetime.fromisoformat(run_date).date()
    except ValueError:
        start = datetime.now(timezone.utc).date()
    return {f"{horizon}d": (start + timedelta(days=int(horizon * 1.5))).isoformat() for horizon in horizons}
