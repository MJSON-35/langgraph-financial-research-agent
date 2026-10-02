from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_picker.graph import build_graph
from stock_picker.main import DEMO_CONFIG_PATH, FULL_CONFIG_PATH, build_demo_state, load_demo_config


class PublicConfigTests(unittest.TestCase):
    def test_demo_and_full_configs_exist_and_load(self) -> None:
        for config_path in (DEMO_CONFIG_PATH, FULL_CONFIG_PATH):
            self.assertTrue(config_path.exists(), str(config_path))
            config = load_demo_config(config_path)
            self.assertIsInstance(config.get("run_metadata"), dict)

    def test_config_file_rejects_embedded_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret_value = "config-secret-value"
            path = Path(tmp) / "unsafe.json"
            path.write_text(
                json.dumps({"name": "unsafe", "run_metadata": {"FRED_API_KEY": secret_value}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as context:
                load_demo_config(path)
            self.assertNotIn(secret_value, str(context.exception))

    def test_offline_public_config_runs_without_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config = load_demo_config(DEMO_CONFIG_PATH)
            state = build_demo_state(config)
            state["run_metadata"]["output_dir"] = str(Path(tmp) / "outputs")
            result = build_graph().invoke(state)

            self.assertTrue(result.get("final_decision", {}).get("top_picks"))
            self.assertEqual(result.get("run_metadata", {}).get("data_mode"), "csv")


if __name__ == "__main__":
    unittest.main()
