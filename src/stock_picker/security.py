"""Small security helpers for public configuration and exported artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "private_key",
    "authorization",
}
MASK = "***"


def is_sensitive_key(key: object) -> bool:
    """Return whether a metadata key may contain authentication material."""
    normalized = str(key).strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return any(keyword in normalized or keyword.replace("_", "") in compact for keyword in SENSITIVE_KEYWORDS)


def find_sensitive_key_paths(value: Any, prefix: str = "") -> list[str]:
    """Find sensitive mapping keys without reading or returning their values."""
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if is_sensitive_key(key):
                paths.append(path)
            else:
                paths.extend(find_sensitive_key_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(find_sensitive_key_paths(nested, f"{prefix}[{index}]"))
    return paths


def reject_sensitive_config(value: Any) -> None:
    """Reject secrets embedded in JSON configuration before state is created."""
    sensitive_paths = find_sensitive_key_paths(value)
    if sensitive_paths:
        joined = ", ".join(sensitive_paths)
        raise ValueError(
            "Secret-like fields are not allowed in config files. "
            f"Use environment variables instead. Fields: {joined}"
        )


def sanitize_for_export(value: Any) -> Any:
    """Recursively mask secret fields and remove local path details from exports."""
    if isinstance(value, Mapping):
        return {
            str(key): MASK if is_sensitive_key(key) else sanitize_for_export(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_export(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_export(item) for item in value]
    if isinstance(value, Path):
        return _sanitize_text(str(value))
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_text(value: str) -> str:
    text = value
    local_roots = [(str(PROJECT_ROOT), ".")]
    try:
        home_root = str(Path.home())
        if home_root != str(PROJECT_ROOT):
            local_roots.append((home_root, "~"))
    except RuntimeError:
        pass
    for local_root, replacement in local_roots:
        if local_root:
            text = text.replace(local_root, replacement)
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password|credential)=([^&\s]+)",
        lambda match: f"{match.group(1)}={MASK}",
        text,
    )
    text = re.sub(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@", rf"\1{MASK}@", text)
    return text
