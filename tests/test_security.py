from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_picker.env_utils import apply_env_defaults_to_run_metadata
from stock_picker.export_utils import save_step_summary
from stock_picker.forward_testing import write_forward_test_snapshot
from stock_picker.llm_utils import resolve_ollama_base_url
from stock_picker.providers.macro_providers import FredMacroProvider
from stock_picker.report_generator import generate_final_reports
from stock_picker.security import PROJECT_ROOT, find_sensitive_key_paths, reject_sensitive_config, sanitize_for_export


class SecurityTests(unittest.TestCase):
    def test_sensitive_metadata_is_recursively_masked(self) -> None:
        payload = {
            "api_key": "example-api-value",
            "nested": {
                "refresh_token": "example-refresh-value",
                "safe": "visible",
            },
        }
        sanitized = sanitize_for_export(payload)
        serialized = json.dumps(sanitized)

        self.assertNotIn("example-api-value", serialized)
        self.assertNotIn("example-refresh-value", serialized)
        self.assertEqual(sanitized["api_key"], "***")
        self.assertEqual(sanitized["nested"]["refresh_token"], "***")
        self.assertEqual(sanitized["nested"]["safe"], "visible")

    def test_project_path_is_exported_as_relative_path(self) -> None:
        local_path = PROJECT_ROOT / "outputs" / "report.json"
        sanitized = sanitize_for_export(str(local_path))
        self.assertNotIn(str(PROJECT_ROOT), sanitized)
        self.assertTrue(sanitized.startswith("."))

    def test_sensitive_config_is_rejected_without_echoing_value(self) -> None:
        secret_value = "do-not-print-this-value"
        config = {"run_metadata": {"fred_api_key": secret_value}}
        with self.assertRaises(ValueError) as context:
            reject_sensitive_config(config)
        self.assertNotIn(secret_value, str(context.exception))

    def test_final_report_json_masks_sensitive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret_value = "report-secret-value"
            paths = generate_final_reports(
                output_dir=root,
                run_metadata={
                    "scenario_name": "security-test",
                    "api_key": secret_value,
                    "nested": {"password": secret_value},
                },
            )
            final_json = paths["json"].read_text(encoding="utf-8")
            summary_json = paths["full_pipeline_summary"].read_text(encoding="utf-8")

            self.assertNotIn(secret_value, final_json)
            self.assertNotIn(secret_value, summary_json)
            self.assertIn('"api_key": "***"', final_json)
            self.assertIn('"password": "***"', final_json)

    def test_step_summary_masks_nested_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret_value = "step-secret-value"
            path = save_step_summary(
                "security_test",
                {"nested": {"credential": secret_value}},
                output_dir=tmp,
            )
            content = path.read_text(encoding="utf-8")
            self.assertNotIn(secret_value, content)
            self.assertEqual(json.loads(content)["nested"]["credential"], "***")

    def test_forward_snapshot_masks_sensitive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret_value = "forward-secret-value"
            state = {
                "run_metadata": {
                    "forward_test_enabled": True,
                    "forward_test_dir": str(Path(tmp) / "forward"),
                    "as_of_date": "2025-01-02",
                    "client_secret": secret_value,
                    "nested": {"access_token": secret_value},
                },
                "filtered_candidates": [],
                "final_decision": {},
            }
            result = write_forward_test_snapshot(state)
            metadata_text = (Path(result["snapshot_path"]) / "run_metadata.json").read_text(encoding="utf-8")

            self.assertNotIn(secret_value, metadata_text)
            self.assertEqual(json.loads(metadata_text)["client_secret"], "***")
            self.assertEqual(json.loads(metadata_text)["nested"]["access_token"], "***")

    def test_ollama_url_is_limited_to_loopback(self) -> None:
        self.assertEqual(
            resolve_ollama_base_url({"ollama_base_url": "http://127.0.0.1:11434"}),
            "http://127.0.0.1:11434",
        )
        with self.assertRaises(ValueError):
            resolve_ollama_base_url({"ollama_base_url": "http://169.254.169.254/latest/meta-data"})
        with self.assertRaises(ValueError):
            resolve_ollama_base_url({"ollama_base_url": "http://user:pass@localhost:11434"})

    def test_public_configs_have_no_sensitive_fields(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for config_name in ("demo_run.json", "ollama_run.json", "full_run.example.json"):
            config = json.loads((project_root / "data" / config_name).read_text(encoding="utf-8"))
            self.assertEqual(find_sensitive_key_paths(config), [], config_name)

    def test_fred_provider_reads_key_from_environment_helper_only(self) -> None:
        provider = FredMacroProvider({"growth": "GDP"})
        self.assertNotIn("api_key", vars(provider))
        with patch("stock_picker.providers.macro_providers.load_project_env"), patch(
            "stock_picker.providers.macro_providers.get_env_str", return_value="environment-only-value"
        ), patch(
            "stock_picker.providers.macro_providers._fetch_latest_fred_value", return_value=1.0
        ) as fetch:
            provider.get_macro_context()
        self.assertEqual(fetch.call_args.args, ("GDP", "environment-only-value"))

    def test_fred_key_is_not_copied_into_run_metadata(self) -> None:
        with patch.dict(os.environ, {"FRED_API_KEY": "environment-only-value"}, clear=True):
            metadata = apply_env_defaults_to_run_metadata({"macro_mode": "fred"})
        self.assertNotIn("fred_api_key", metadata)
        self.assertNotIn("FRED_API_KEY", metadata)


if __name__ == "__main__":
    unittest.main()
