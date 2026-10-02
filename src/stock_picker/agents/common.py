"""Small shared helpers for keeping agent modules consistent and compact."""

from __future__ import annotations

from typing import Any


def collect_missing_fields(record: dict[str, Any], required_fields: list[str]) -> list[str]:
    """Return required field names that are missing or blank."""
    missing_fields: list[str] = []
    for field in required_fields:
        value = record.get(field)
        if value is None or value == "":
            missing_fields.append(field)
    return missing_fields


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a scalar or list-like input into a clean list of strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Drop blanks and duplicates while keeping the original order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def build_insufficient_output(
    *,
    agent: str,
    summary: str,
    missing_inputs: list[str],
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistent insufficient-input payload for agent outputs."""
    return {
        "agent": agent,
        "status": "insufficient_input",
        "summary": summary,
        "missing_inputs": missing_inputs,
        **extra_fields,
    }
