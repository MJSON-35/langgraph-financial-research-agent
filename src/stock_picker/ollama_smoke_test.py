"""Small local smoke test for the optional Ollama-backed agent path."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_picker.llm_utils import probe_ollama, resolve_ollama_base_url, resolve_ollama_model
from stock_picker.main import DEMO_CONFIG_PATH, load_demo_config


AGENT_NAMES = ["macro_agent", "equity_agent", "risk_agent", "portfolio_manager"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local Ollama connectivity and configured models.")
    parser.add_argument(
        "--config",
        default=str(DEMO_CONFIG_PATH),
        help="Run configuration to inspect (for example, data/ollama_run.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Print a small connectivity and configuration summary for local Ollama usage."""
    config_path = Path(parse_args().config)
    if not config_path.is_absolute():
        config_path = (Path(DEMO_CONFIG_PATH).parent.parent / config_path).resolve()
    config = load_demo_config(config_path)
    run_metadata = dict(config.get("run_metadata", {}))

    print("Ollama Smoke Test")
    print(f"Scenario: {config.get('name', 'demo')}")
    try:
        display_base_url = resolve_ollama_base_url(run_metadata)
    except ValueError:
        display_base_url = "invalid local endpoint"
    print(f"Base URL: {display_base_url}")
    print(f"Use Ollama: {bool(run_metadata.get('use_ollama'))}")
    print("Agent model mapping:")
    for agent_name in AGENT_NAMES:
        print(f"- {agent_name}: {resolve_ollama_model(run_metadata, agent_name)}")

    tags_payload, error = probe_ollama(run_metadata)
    if error:
        print(f"\nHealth check: unavailable ({error})")
        print("Next steps:")
        print("- Start the local server with `ollama serve`.")
        print(f"- Pull the configured model with `ollama pull {resolve_ollama_model(run_metadata, 'equity_agent')}`.")
        print("- Re-run this smoke test before using LLM-backed agents.")
        return

    models = tags_payload.get("models", []) if isinstance(tags_payload, dict) else []
    model_names = [
        str(model.get("model", "")).strip()
        for model in models
        if isinstance(model, dict) and str(model.get("model", "")).strip()
    ]

    print("\nHealth check: available")
    print(f"Installed models: {', '.join(model_names) or 'none reported'}")

    missing_models = [
        resolve_ollama_model(run_metadata, agent_name)
        for agent_name in AGENT_NAMES
        if resolve_ollama_model(run_metadata, agent_name) not in model_names
    ]
    if missing_models:
        unique_missing = []
        for model_name in missing_models:
            if model_name not in unique_missing:
                unique_missing.append(model_name)
        print(f"Missing configured models: {', '.join(unique_missing)}")
    else:
        print("Configured models: all available")

    print("\nSuggested rollout:")
    print("- 1. Start with macro_agent and equity_agent only.")
    print("- 2. Keep risk_agent and portfolio_manager on fallback mode until JSON stability looks good.")
    print("- 3. Compare Ollama outputs against the current rule-based baseline.")


if __name__ == "__main__":
    main()
