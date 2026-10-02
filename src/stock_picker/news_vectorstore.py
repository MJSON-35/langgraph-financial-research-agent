"""Local qualitative-news vectorstore utilities.

This module is intentionally strict about what can enter the news cache:
only unstructured news text and a small allowlist of descriptive metadata are
persisted. Quantitative market, factor, and backtest fields are removed before
storage.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from stock_picker.env_utils import get_env_bool, get_env_str, load_project_env
from stock_picker.export_utils import PROJECT_ROOT


DEFAULT_PERSIST_DIR = "data/vectorstore/chroma_news"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FORBIDDEN_QUANT_FIELDS = {
    "price",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "avg_volume",
    "avg_volume_20d",
    "market_cap",
    "ret_1m",
    "ret_12m",
    "return",
    "forward_return",
    "volatility",
    "volatility_60d",
    "max_drawdown",
    "max_drawdown_1y",
    "trailing_pe",
    "forward_pe",
    "pe",
    "price_to_book",
    "pb",
    "roe",
    "roa",
    "profit_margin",
    "operating_margin",
    "value_score",
    "momentum_score",
    "liquidity_score",
    "quant_score",
    "final_score",
    "backtest_return",
    "benchmark_return",
    "excess_return",
}

ALLOWED_METADATA_FIELDS = {
    "ticker",
    "company_name",
    "market",
    "sector",
    "source",
    "source_domain",
    "url",
    "published_date",
    "retrieved_at",
    "query",
    "sentiment_hint",
    "risk_flags",
    "quality_score",
    "company_match_score",
    "source_trust_score",
    "recency_score",
    "query_relevance_score",
}

HIGH_TRUST_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "investing.com",
    "yna.co.kr",
    "hankyung.com",
    "mk.co.kr",
    "thelec.kr",
}

LOW_TRUST_DOMAINS = {
    "penny-stocks.com",
    "stocktwits.com",
    "reddit.com",
    "x.com",
    "twitter.com",
}


class SimpleDocument:
    """Small stand-in compatible with the fields LangChain Document exposes."""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class LocalSentenceTransformerEmbeddings:
    """Chroma embedding adapter backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        fallback_model_name: str | None = None,
        cache_folder: str | None = None,
        token: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model_name = model_name
            self.model = _load_sentence_transformer(
                SentenceTransformer,
                model_name,
                cache_folder=cache_folder,
                token=token,
                local_files_only=local_files_only,
            )
        except Exception:
            if not fallback_model_name or fallback_model_name == model_name:
                raise
            from sentence_transformers import SentenceTransformer

            self.model_name = fallback_model_name
            self.model = _load_sentence_transformer(
                SentenceTransformer,
                fallback_model_name,
                cache_folder=cache_folder,
                token=token,
                local_files_only=local_files_only,
            )

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def name(self) -> str:
        return f"local_sentence_transformer_{self.model_name.replace('/', '_')}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_news_vectorstore(run_metadata: dict[str, Any]) -> Any | None:
    """Create a persistent Chroma collection, or return None when unavailable."""
    load_project_env()
    vectorstore_enabled = (
        bool(run_metadata["news_vectorstore_enabled"])
        if "news_vectorstore_enabled" in run_metadata
        else get_env_bool("NEWS_VECTORSTORE_ENABLED", False)
    )
    if not vectorstore_enabled:
        return None
    try:
        import chromadb
    except Exception:
        return None

    embedding_model = build_local_embedding_model(run_metadata)
    if embedding_model is None:
        return None

    persist_dir = _resolve_persist_dir(run_metadata)
    persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        return client.get_or_create_collection(
            name="qualitative_news",
            embedding_function=embedding_model,
            metadata={"description": "Qualitative news text only; quantitative fields excluded."},
        )
    except Exception:
        return None


def build_local_embedding_model(run_metadata: dict[str, Any]) -> Any | None:
    """Load the configured local embedding model with a local fallback model."""
    if not _embedding_runtime_compatible():
        return None
    model_name = str(run_metadata.get("news_embedding_model") or DEFAULT_EMBEDDING_MODEL)
    fallback_name = str(
        run_metadata.get("news_embedding_fallback_model") or DEFAULT_EMBEDDING_FALLBACK_MODEL
    )
    cache_folder = _resolve_embedding_cache_dir(run_metadata)
    token = get_env_str("HF_TOKEN")
    local_files_only = bool(run_metadata.get("news_embedding_local_files_only", True))
    if cache_folder is not None:
        cache_folder.mkdir(parents=True, exist_ok=True)
    try:
        return LocalSentenceTransformerEmbeddings(
            model_name,
            fallback_name,
            str(cache_folder) if cache_folder is not None else None,
            token,
            local_files_only,
        )
    except Exception:
        return None


def _embedding_runtime_compatible() -> bool:
    """Avoid noisy native-extension crashes in known incompatible NumPy/SciPy stacks."""
    try:
        import numpy as np

        numpy_major = int(str(np.__version__).split(".", 1)[0])
    except Exception:
        return True
    if numpy_major < 2:
        return True
    return False


def build_news_document(
    news_item: dict[str, Any],
    candidate: dict[str, Any],
    quality_info: dict[str, Any] | None = None,
) -> SimpleDocument:
    """Build a sanitized qualitative-news document for vectorstore storage."""
    quality = quality_info or {}
    sanitized = sanitize_news_for_vectorstore(news_item)
    headline = sanitized.get("headline", "")
    summary = sanitized.get("summary", "")
    source_domain = sanitized.get("source_domain", "")
    source_note = f"Source: {source_domain}" if source_domain else "Source: unavailable"
    page_content = "\n".join(part for part in [headline, summary, source_note] if part).strip()

    metadata = {
        "ticker": str(candidate.get("ticker", news_item.get("ticker", ""))).strip(),
        "company_name": str(candidate.get("name", news_item.get("company_name", ""))).strip(),
        "market": str(candidate.get("market", news_item.get("market", ""))).strip(),
        "sector": str(candidate.get("sector", news_item.get("sector", ""))).strip(),
        "source": str(sanitized.get("source", "unknown")).strip(),
        "source_domain": source_domain,
        "url": str(sanitized.get("url", "")).strip(),
        "published_date": str(sanitized.get("published_date", "")).strip(),
        "retrieved_at": str(sanitized.get("retrieved_at", _now_iso())).strip(),
        "query": str(sanitized.get("query", "")).strip(),
        "sentiment_hint": str(quality.get("sentiment_hint", "")).strip(),
        "risk_flags": _metadata_list(quality.get("risk_flags", sanitized.get("risk_flags", []))),
        "quality_score": float(quality.get("quality_score", sanitized.get("quality_score", 0.0)) or 0.0),
        "company_match_score": float(quality.get("company_match_score", 0.0) or 0.0),
        "source_trust_score": float(quality.get("source_trust_score", 0.0) or 0.0),
        "recency_score": float(quality.get("recency_score", 0.0) or 0.0),
        "query_relevance_score": float(quality.get("query_relevance_score", 0.0) or 0.0),
    }
    metadata = validate_no_quant_fields(metadata)
    return SimpleDocument(page_content=page_content, metadata=metadata)


def sanitize_news_for_vectorstore(news_item: dict[str, Any]) -> dict[str, Any]:
    """Return only qualitative news fields used for page_content and metadata."""
    source_domain = str(news_item.get("source_domain") or _domain_from_url(news_item.get("url", ""))).strip()
    return {
        "ticker": str(news_item.get("ticker", "")).strip(),
        "company_name": str(news_item.get("company_name", "")).strip(),
        "headline": str(news_item.get("headline", news_item.get("title", ""))).strip(),
        "summary": str(news_item.get("summary", news_item.get("snippet", ""))).strip(),
        "content": str(news_item.get("content", "")).strip(),
        "source": str(news_item.get("source", "unknown")).strip(),
        "source_domain": source_domain,
        "url": str(news_item.get("url", "")).strip(),
        "published_date": _date_to_iso(news_item.get("published_date")),
        "retrieved_at": str(news_item.get("retrieved_at") or _now_iso()).strip(),
        "query": str(news_item.get("query", "")).strip(),
    }


def validate_no_quant_fields(document_or_metadata: Any) -> dict[str, Any]:
    """Remove forbidden quantitative fields and keep only allowed metadata keys."""
    metadata = (
        dict(getattr(document_or_metadata, "metadata", {}))
        if hasattr(document_or_metadata, "metadata")
        else dict(document_or_metadata or {})
    )
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = str(key).strip()
        if normalized_key in FORBIDDEN_QUANT_FIELDS:
            continue
        if normalized_key not in ALLOWED_METADATA_FIELDS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[normalized_key] = "" if value is None else value
        elif isinstance(value, list):
            cleaned[normalized_key] = _metadata_list(value)
        else:
            cleaned[normalized_key] = str(value)
    if hasattr(document_or_metadata, "metadata"):
        document_or_metadata.metadata = cleaned
    return cleaned


def make_news_document_id(news_item: dict[str, Any], candidate: dict[str, Any]) -> str:
    ticker = str(candidate.get("ticker", news_item.get("ticker", ""))).strip().upper()
    url_key = normalize_url(str(news_item.get("url", "")))
    headline_key = normalize_headline(str(news_item.get("headline", news_item.get("title", ""))))
    date_key = _date_to_iso(news_item.get("published_date"))[:10]
    raw = "|".join([ticker, url_key or headline_key, date_key])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def add_news_documents(vectorstore: Any, docs: list[SimpleDocument]) -> int:
    """Persist documents into Chroma-like collection; return stored count."""
    if vectorstore is None or not docs:
        return 0
    ids = [doc.metadata.get("document_id") or _doc_id_from_doc(doc) for doc in docs]
    documents = [doc.page_content for doc in docs]
    metadatas = [validate_no_quant_fields(doc.metadata) for doc in docs]
    try:
        vectorstore.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(docs)
    except Exception:
        return 0


def retrieve_recent_news(
    vectorstore: Any,
    candidate: dict[str, Any],
    run_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retrieve recent high-quality news for one ticker from a Chroma collection."""
    if vectorstore is None:
        return []
    ticker = str(candidate.get("ticker", "")).strip()
    if not ticker:
        return []
    top_k = int(run_metadata.get("news_retrieval_top_k", 5) or 5)
    query = f"{candidate.get('name', '')} {ticker} recent company news risk earnings guidance"
    try:
        result = vectorstore.query(
            query_texts=[query],
            n_results=max(top_k * 3, top_k),
            where={"ticker": ticker},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    items: list[dict[str, Any]] = []
    for content, metadata in zip(docs, metadatas):
        metadata = metadata or {}
        if not _within_window(metadata.get("published_date"), int(run_metadata.get("tavily_days", 30) or 30)):
            continue
        if float(metadata.get("quality_score") or 0.0) < float(run_metadata.get("news_min_quality_score", 0.60)):
            continue
        lines = [line.strip() for line in str(content or "").splitlines() if line.strip()]
        items.append(
            {
                "ticker": ticker,
                "company_name": metadata.get("company_name", candidate.get("name", "")),
                "headline": lines[0] if lines else "",
                "summary": lines[1] if len(lines) > 1 else "",
                "source": "vectorstore",
                "source_domain": metadata.get("source_domain", ""),
                "url": metadata.get("url", ""),
                "published_date": metadata.get("published_date", ""),
                "retrieved_at": metadata.get("retrieved_at", ""),
                "query": metadata.get("query", ""),
                "quality_score": float(metadata.get("quality_score") or 0.0),
                "company_match_score": float(metadata.get("company_match_score") or 0.0),
                "source_trust_score": float(metadata.get("source_trust_score") or 0.0),
                "recency_score": float(metadata.get("recency_score") or 0.0),
                "query_relevance_score": float(metadata.get("query_relevance_score") or 0.0),
            }
        )
        if len(items) >= top_k:
            break
    return items


def cleanup_old_news_documents(run_metadata: dict[str, Any], cutoff_days: int = 90) -> dict[str, Any]:
    """Create a dry-run cleanup report for old vectorstore news documents.

    Actual deletion is disabled unless news_vectorstore_cleanup_delete=true.
    """
    output_dir = Path(str(run_metadata.get("output_dir", "outputs")))
    output_dir = output_dir if output_dir.is_absolute() else (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vectorstore_cleanup_report.md"

    if not bool(run_metadata.get("news_vectorstore_enabled", False)):
        summary = {"status": "skipped", "reason": "news vectorstore disabled", "old_document_count": 0}
        report_path.write_text(_build_cleanup_report(summary), encoding="utf-8")
        return {**summary, "report_path": str(report_path)}

    vectorstore = get_news_vectorstore(run_metadata)
    if vectorstore is None:
        summary = {"status": "skipped", "reason": "vectorstore unavailable", "old_document_count": 0}
        report_path.write_text(_build_cleanup_report(summary), encoding="utf-8")
        return {**summary, "report_path": str(report_path)}

    try:
        raw = vectorstore.get(include=["metadatas"])
    except Exception as exc:
        summary = {"status": "skipped", "reason": f"vectorstore get failed: {exc}", "old_document_count": 0}
        report_path.write_text(_build_cleanup_report(summary), encoding="utf-8")
        return {**summary, "report_path": str(report_path)}

    ids = raw.get("ids", []) or []
    metadatas = raw.get("metadatas", []) or []
    old_ids = [
        doc_id
        for doc_id, metadata in zip(ids, metadatas)
        if not _within_window((metadata or {}).get("published_date"), cutoff_days)
    ]
    delete_enabled = bool(run_metadata.get("news_vectorstore_cleanup_delete", False))
    deleted_count = 0
    if delete_enabled and old_ids:
        try:
            vectorstore.delete(ids=old_ids)
            deleted_count = len(old_ids)
        except Exception:
            deleted_count = 0
    summary = {
        "status": "completed",
        "mode": "delete" if delete_enabled else "dry_run",
        "cutoff_days": cutoff_days,
        "total_document_count": len(ids),
        "old_document_count": len(old_ids),
        "deleted_count": deleted_count,
    }
    report_path.write_text(_build_cleanup_report(summary), encoding="utf-8")
    return {**summary, "report_path": str(report_path)}


def deduplicate_news_items(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by normalized URL, headline, then close-date fuzzy headline."""
    kept: list[dict[str, Any]] = []
    url_index: dict[str, int] = {}
    headline_index: dict[str, int] = {}
    for item in news_items:
        normalized_url = normalize_url(str(item.get("url", "")))
        normalized_headline = normalize_headline(str(item.get("headline", "")))
        duplicate_idx: int | None = None
        if normalized_url and normalized_url in url_index:
            duplicate_idx = url_index[normalized_url]
        elif normalized_headline and normalized_headline in headline_index:
            duplicate_idx = headline_index[normalized_headline]
        else:
            duplicate_idx = _find_fuzzy_duplicate(item, kept)

        if duplicate_idx is None:
            kept.append(item)
            idx = len(kept) - 1
            if normalized_url:
                url_index[normalized_url] = idx
            if normalized_headline:
                headline_index[normalized_headline] = idx
            continue

        if _quality_tuple(item) > _quality_tuple(kept[duplicate_idx]):
            kept[duplicate_idx] = item
    return kept


def score_news_quality(
    news_item: dict[str, Any],
    candidate: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score company match, source trust, recency, and query relevance."""
    text = " ".join(str(news_item.get(key, "")) for key in ("headline", "summary", "content")).lower()
    aliases = _candidate_aliases(candidate, run_metadata)
    company_match_score = 1.0 if any(alias and alias in text for alias in aliases) else 0.0
    if company_match_score == 0.0:
        weak_aliases = [alias.split()[0] for alias in aliases if len(alias.split()) > 1]
        company_match_score = 0.5 if any(alias and alias in text for alias in weak_aliases) else 0.0

    source_domain = str(news_item.get("source_domain") or _domain_from_url(news_item.get("url", ""))).lower()
    source_trust_score = _source_trust_score(source_domain)
    recency_score = _recency_score(news_item.get("published_date"), int(run_metadata.get("tavily_days", 30) or 30))
    query_relevance_score = _query_relevance_score(news_item, candidate, aliases)
    quality_score = (
        0.40 * company_match_score
        + 0.20 * source_trust_score
        + 0.20 * recency_score
        + 0.20 * query_relevance_score
    )
    if bool(run_metadata.get("news_require_company_match", True)) and company_match_score == 0.0:
        quality_score = min(quality_score, float(run_metadata.get("news_low_quality_threshold", 0.40)) - 0.01)

    date_confidence = "high" if _parse_date(news_item.get("published_date")) else "low"
    return {
        "quality_score": round(max(0.0, min(1.0, quality_score)), 4),
        "company_match_score": company_match_score,
        "source_trust_score": source_trust_score,
        "recency_score": recency_score,
        "query_relevance_score": query_relevance_score,
        "date_confidence": date_confidence,
        "quality_reason": (
            f"company_match={company_match_score:.2f}, source={source_trust_score:.2f}, "
            f"recency={recency_score:.2f}, relevance={query_relevance_score:.2f}"
        ),
    }


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip().lower())
    tracking_prefixes = ("utm_",)
    tracking_names = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if not key.startswith(tracking_prefixes) and key not in tracking_names
    ]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", urlencode(query), ""))


def normalize_headline(headline: str) -> str:
    text = headline.lower().strip()
    text = re.sub(r"[^a-z0-9가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_news_recent(news_item: dict[str, Any], run_metadata: dict[str, Any]) -> bool:
    return _within_window(news_item.get("published_date"), int(run_metadata.get("tavily_days", 30) or 30))


def _resolve_persist_dir(run_metadata: dict[str, Any]) -> Path:
    configured = run_metadata.get("news_vectorstore_persist_dir") or DEFAULT_PERSIST_DIR
    path = Path(str(configured))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_embedding_cache_dir(run_metadata: dict[str, Any]) -> Path | None:
    configured = run_metadata.get("news_embedding_cache_dir")
    if not configured:
        return None
    path = Path(str(configured))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_sentence_transformer(
    sentence_transformer_cls: Any,
    model_name: str,
    *,
    cache_folder: str | None,
    token: str | None,
    local_files_only: bool,
) -> Any:
    kwargs = {"cache_folder": cache_folder, "local_files_only": local_files_only}
    if token:
        try:
            return sentence_transformer_cls(model_name, token=token, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            return sentence_transformer_cls(model_name, use_auth_token=token, **kwargs)
    try:
        return sentence_transformer_cls(model_name, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        return sentence_transformer_cls(model_name, **kwargs)


def _metadata_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _doc_id_from_doc(doc: SimpleDocument) -> str:
    raw = f"{doc.metadata.get('ticker', '')}|{doc.metadata.get('url', '')}|{doc.page_content[:120]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _quality_tuple(item: dict[str, Any]) -> tuple[float, float]:
    return (
        float(item.get("quality_score") or 0.0),
        float(item.get("source_trust_score") or 0.0),
    )


def _find_fuzzy_duplicate(item: dict[str, Any], kept: list[dict[str, Any]]) -> int | None:
    ticker = str(item.get("ticker", "")).upper()
    headline = normalize_headline(str(item.get("headline", "")))
    date_value = _parse_date(item.get("published_date"))
    if not headline or date_value is None:
        return None
    for idx, existing in enumerate(kept):
        if str(existing.get("ticker", "")).upper() != ticker:
            continue
        existing_date = _parse_date(existing.get("published_date"))
        if existing_date is None or abs((date_value - existing_date).days) > 2:
            continue
        existing_headline = normalize_headline(str(existing.get("headline", "")))
        if SequenceMatcher(None, headline, existing_headline).ratio() >= 0.85:
            return idx
    return None


def _candidate_aliases(candidate: dict[str, Any], run_metadata: dict[str, Any]) -> list[str]:
    ticker = str(candidate.get("ticker", "")).strip()
    bare_ticker = ticker.split(".")[0]
    name = str(candidate.get("name", candidate.get("company_name", ""))).strip()
    aliases = [ticker, bare_ticker, name]
    alias_map = run_metadata.get("company_alias_map", {})
    mapped = alias_map.get(ticker, []) if isinstance(alias_map, dict) else []
    if isinstance(mapped, str):
        mapped = [mapped]
    aliases.extend(mapped if isinstance(mapped, list) else [])
    return [alias.lower() for alias in aliases if str(alias).strip()]


def _source_trust_score(domain: str) -> float:
    if not domain:
        return 0.4
    domain = domain.lower().replace("www.", "")
    if any(domain == item or domain.endswith("." + item) for item in HIGH_TRUST_DOMAINS):
        return 0.9
    if any(domain == item or domain.endswith("." + item) for item in LOW_TRUST_DOMAINS):
        return 0.2
    if "ir." in domain or "investor" in domain:
        return 0.7
    return 0.5


def _recency_score(value: Any, window_days: int) -> float:
    parsed = _parse_date(value)
    if parsed is None:
        return 0.4
    age = (datetime.now(timezone.utc).date() - parsed.date()).days
    if age < 0:
        return 0.7
    if age <= 7:
        return 1.0
    if age <= window_days:
        return 0.7
    return 0.0


def _within_window(value: Any, window_days: int) -> bool:
    parsed = _parse_date(value)
    if parsed is None:
        return True
    age = (datetime.now(timezone.utc).date() - parsed.date()).days
    return age <= window_days


def _query_relevance_score(news_item: dict[str, Any], candidate: dict[str, Any], aliases: list[str]) -> float:
    headline_content = " ".join(
        str(news_item.get(key, "")) for key in ("headline", "summary", "content")
    ).lower()
    query = str(news_item.get("query", "")).lower()
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9가-힣]+", query)
        if len(term) >= 3 and term not in {"stock", "news", "latest", "recent"}
    }
    overlap = sum(1 for term in query_terms if term in headline_content)
    company_hit = any(alias and alias in headline_content for alias in aliases)
    if company_hit and overlap >= 1:
        return 0.8
    if company_hit:
        return 0.6
    if overlap >= 2:
        return 0.5
    return 0.2


def _date_to_iso(value: Any) -> str:
    parsed = _parse_date(value)
    return parsed.date().isoformat() if parsed else ""


def _parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            numeric = int(text)
            if numeric > 10_000_000_000:
                numeric = numeric // 1000
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except Exception:
            return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _domain_from_url(url: Any) -> str:
    netloc = urlparse(str(url or "")).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_cleanup_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Vectorstore Cleanup Report",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Mode: {summary.get('mode', 'dry_run')}",
        f"- Cutoff days: {summary.get('cutoff_days', 'n/a')}",
        f"- Total documents: {summary.get('total_document_count', 0)}",
        f"- Old documents: {summary.get('old_document_count', 0)}",
        f"- Deleted documents: {summary.get('deleted_count', 0)}",
    ]
    if summary.get("reason"):
        lines.append(f"- Reason: {summary.get('reason')}")
    lines.append("- Retrieval still applies a separate recent-news window, normally 30 days.")
    return "\n".join(lines)
