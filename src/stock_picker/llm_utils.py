"""Small helpers for optional Ollama-backed local LLM execution."""

from __future__ import annotations

import ast
import json
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}


def resolve_ollama_base_url(run_metadata: dict[str, Any]) -> str:
    """Validate the prototype's Ollama endpoint as a loopback HTTP URL."""
    raw_url = str(run_metadata.get("ollama_base_url", "http://127.0.0.1:11434")).strip().rstrip("/")
    parsed = urlparse(raw_url)
    if parsed.scheme != "http" or parsed.hostname not in ALLOWED_OLLAMA_HOSTS:
        raise ValueError("Ollama base URL must use http on localhost or a loopback IP address.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Ollama base URL must not contain credentials, a path, query, or fragment.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama base URL contains an invalid port.") from exc
    return raw_url


def is_ollama_enabled(run_metadata: dict[str, Any], agent_name: str) -> bool:
    """Return whether a given agent should try local Ollama execution."""
    if run_metadata.get("use_ollama") is False:
        return False
    enabled_agents = {
        str(name).strip()
        for name in run_metadata.get("ollama_enabled_agents", [])
        if str(name).strip()
    }
    if enabled_agents:
        return agent_name in enabled_agents
    return bool(run_metadata.get("use_ollama"))


def resolve_ollama_model(run_metadata: dict[str, Any], agent_name: str) -> str:
    """Resolve the configured Ollama model for one agent."""
    model_map = run_metadata.get("ollama_models", {})
    if isinstance(model_map, dict):
        model = model_map.get(agent_name)
        if model:
            return str(model)
    return str(run_metadata.get("ollama_model", "qwen2.5:7b"))


def run_ollama_json_prompt(
    prompt: str,
    run_metadata: dict[str, Any],
    agent_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call a local Ollama model and parse the first JSON-like object returned."""
    try:
        base_url = resolve_ollama_base_url(run_metadata)
    except ValueError as exc:
        return None, str(exc)
    model = resolve_ollama_model(run_metadata, agent_name)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(run_metadata.get("ollama_temperature", 0.0)),
        },
    }
    request = Request(
        url=f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=float(run_metadata.get("ollama_timeout_seconds", 90))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return None, f"Ollama request failed: {exc}"
    except OSError as exc:
        return None, f"Ollama connection failed: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"Ollama response JSON decode failed: {exc}"

    raw_text = str(body.get("response", "")).strip()
    if not raw_text:
        return None, "Ollama returned an empty response."

    parsed = parse_json_like_object(raw_text)
    if not isinstance(parsed, dict):
        return None, "Ollama response did not contain a valid object."
    return parsed, None


def probe_ollama(run_metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return a small health payload from a local Ollama server when available."""
    try:
        base_url = resolve_ollama_base_url(run_metadata)
    except ValueError as exc:
        return None, str(exc)
    request = Request(url=f"{base_url}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=float(run_metadata.get("ollama_timeout_seconds", 15))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return None, f"Ollama probe failed: {exc}"
    except OSError as exc:
        return None, f"Ollama connection failed: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"Ollama tags response decode failed: {exc}"
    return body if isinstance(body, dict) else None, None


def parse_json_like_object(text: str) -> dict[str, Any] | None:
    """Parse JSON or a Python-dict-like string into a dictionary."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        pass
    return None


def coerce_string_list(value: Any) -> list[str]:
    """Normalize arbitrary model outputs into a compact string list."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]
