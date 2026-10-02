"""Environment variable helpers for local runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}

ENV_METADATA_MAP: dict[str, tuple[str, str]] = {
    "OLLAMA_BASE_URL": ("ollama_base_url", "str"),
    "OLLAMA_MODEL": ("ollama_model", "str"),
    "NEWS_TAVILY_ENABLED": ("news_tavily_enabled", "bool"),
    "NEWS_VECTORSTORE_ENABLED": ("news_vectorstore_enabled", "bool"),
    "NEWS_VECTORSTORE_PERSIST_DIR": ("news_vectorstore_persist_dir", "str"),
    "NEWS_EMBEDDING_MODEL": ("news_embedding_model", "str"),
    "NEWS_EMBEDDING_FALLBACK_MODEL": ("news_embedding_fallback_model", "str"),
    "NEWS_EMBEDDING_CACHE_DIR": ("news_embedding_cache_dir", "str"),
    "NEWS_EMBEDDING_LOCAL_FILES_ONLY": ("news_embedding_local_files_only", "bool"),
    "TAVILY_MAX_RESULTS": ("tavily_max_results", "int"),
    "TAVILY_SEARCH_DEPTH": ("tavily_search_depth", "str"),
    "TAVILY_TOPIC": ("tavily_topic", "str"),
    "TAVILY_DAYS": ("tavily_days", "int"),
    "TAVILY_RECENT_PRIORITY_DAYS": ("tavily_recent_priority_days", "int"),
    "TAVILY_INCLUDE_ANSWER": ("tavily_include_answer", "bool"),
    "TAVILY_INCLUDE_RAW_CONTENT": ("tavily_include_raw_content", "bool"),
    "NEWS_RETRIEVAL_TOP_K": ("news_retrieval_top_k", "int"),
    "NEWS_MIN_QUALITY_SCORE": ("news_min_quality_score", "float"),
    "NEWS_LOW_QUALITY_THRESHOLD": ("news_low_quality_threshold", "float"),
    "NEWS_REQUIRE_COMPANY_MATCH": ("news_require_company_match", "bool"),
    "NEWS_DEDUPLICATE": ("news_deduplicate", "bool"),
    "BACKTEST_ENABLED": ("backtest_enabled", "bool"),
    "BACKTEST_YFINANCE_ENABLED": ("backtest_yfinance_enabled", "bool"),
}


def load_project_env(env_path: str | Path | None = None) -> bool:
    """Load project .env if python-dotenv is available.

    Returns True only when python-dotenv loaded an existing file. Missing
    python-dotenv or missing .env is treated as a normal no-op.
    """
    path = Path(env_path) if env_path is not None else ENV_PATH
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except Exception:
        return False
    try:
        return bool(load_dotenv(path, override=False))
    except Exception:
        return False


def get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in BOOL_TRUE:
        return True
    if normalized in BOOL_FALSE:
        return False
    return default


def get_env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def get_env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def get_env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def mask_secret(value: str | None) -> str:
    """Mask a secret so logs/reports never expose the original value."""
    if not value:
        return ""
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * max(4, len(text) - 4)}{text[-2:]}"


def apply_env_defaults_to_run_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    """Apply env defaults without overriding explicit JSON/run_metadata values.

    Priority is: explicit non-secret metadata > environment variable > code defaults.

    Secret environment variables are deliberately absent from ``ENV_METADATA_MAP``
    so their values never enter workflow state.
    """
    updated = dict(run_metadata)
    for env_name, (metadata_key, value_type) in ENV_METADATA_MAP.items():
        if metadata_key in updated:
            continue
        if env_name not in os.environ:
            continue
        if value_type == "bool":
            updated[metadata_key] = get_env_bool(env_name)
        elif value_type == "int":
            value = get_env_int(env_name)
            if value is not None:
                updated[metadata_key] = value
        elif value_type == "float":
            value = get_env_float(env_name)
            if value is not None:
                updated[metadata_key] = value
        else:
            value = get_env_str(env_name)
            if value is not None:
                updated[metadata_key] = value
    return updated
