"""Export an auditable table of news quality decisions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from stock_picker.export_utils import PROJECT_ROOT


AUDIT_COLUMNS = [
    "ticker",
    "company_name",
    "headline",
    "source_domain",
    "published_date",
    "retrieved_at",
    "company_match_score",
    "source_trust_score",
    "recency_score",
    "query_relevance_score",
    "quality_score",
    "decision",
    "quality_reason",
    "url",
]


def write_news_quality_audit(state: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Write CSV/Markdown news audit artifacts, including the empty case."""
    resolved_output_dir = _resolve_output_dir(output_dir or state.get("run_metadata", {}).get("output_dir", "outputs"))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_news_quality_rows(state.get("filtered_candidates", []))
    csv_path = resolved_output_dir / "news_quality_audit.csv"
    md_path = resolved_output_dir / "news_quality_audit.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_news_quality_rows(rows)
    md_path.write_text(build_news_quality_markdown(summary, rows), encoding="utf-8")
    return {**summary, "csv_path": str(csv_path), "markdown_path": str(md_path)}


def build_news_quality_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        ticker = str(candidate.get("ticker", "UNKNOWN"))
        company_name = str(candidate.get("name", candidate.get("company_name", "")))
        for document in candidate.get("news_documents_used", []) or []:
            quality_score = _safe_float(document.get("quality_score"))
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "headline": document.get("headline", ""),
                    "source_domain": document.get("source_domain", ""),
                    "published_date": document.get("published_date", ""),
                    "retrieved_at": document.get("retrieved_at", ""),
                    "company_match_score": _safe_float(document.get("company_match_score")),
                    "source_trust_score": _safe_float(document.get("source_trust_score")),
                    "recency_score": _safe_float(document.get("recency_score")),
                    "query_relevance_score": _safe_float(document.get("query_relevance_score")),
                    "quality_score": quality_score,
                    "decision": "stored" if quality_score >= 0.60 else "used_low_confidence",
                    "quality_reason": document.get("quality_reason", ""),
                    "url": document.get("url", ""),
                }
            )
    return rows


def summarize_news_quality_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [str(row.get("decision", "")) for row in rows]
    scores = [_safe_float(row.get("quality_score")) for row in rows if row.get("quality_score") not in ("", None)]
    domains = []
    for row in rows:
        domain = str(row.get("source_domain", "")).strip()
        if domain and domain not in domains:
            domains.append(domain)
    return {
        "stored_count": decisions.count("stored"),
        "filtered_count": sum(1 for decision in decisions if decision.startswith("filtered")),
        "duplicate_count": decisions.count("filtered_duplicate"),
        "old_news_filtered_count": decisions.count("filtered_old_news"),
        "low_confidence_count": decisions.count("used_low_confidence"),
        "avg_quality_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "source_domains": domains[:10],
        "row_count": len(rows),
    }


def build_news_quality_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# News Quality Audit",
        "",
        f"- Rows: {summary.get('row_count', 0)}",
        f"- Stored: {summary.get('stored_count', 0)}",
        f"- Low confidence used: {summary.get('low_confidence_count', 0)}",
        f"- Filtered: {summary.get('filtered_count', 0)}",
        f"- Average quality score: {summary.get('avg_quality_score', 0.0):.4f}",
        f"- Source domains: {', '.join(summary.get('source_domains', [])) or 'Unavailable'}",
        "",
        "This audit records qualitative news evidence only. Quantitative data remains in CSV/parquet snapshots.",
    ]
    if rows:
        lines.extend(["", "## Sample Rows"])
        for row in rows[:10]:
            lines.append(
                f"- {row.get('ticker')}: {row.get('headline')} "
                f"({row.get('source_domain')}, quality={row.get('quality_score')})"
            )
    return "\n".join(lines)


def _resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(str(output_dir))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except Exception:
        return 0.0
