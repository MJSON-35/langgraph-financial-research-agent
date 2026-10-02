from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_picker.env_utils import get_env_bool, load_project_env, mask_secret
from stock_picker.tavily_news import fetch_tavily_news_records


class EnvUtilsTests(unittest.TestCase):
    def test_load_project_env_missing_file_does_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".env"
            self.assertFalse(load_project_env(missing))

    def test_get_env_bool_parses_true_false_strings(self) -> None:
        with patch.dict(os.environ, {"YES_FLAG": "true", "NO_FLAG": "False"}, clear=False):
            self.assertTrue(get_env_bool("YES_FLAG"))
            self.assertFalse(get_env_bool("NO_FLAG", default=True))

    def test_mask_secret_does_not_expose_full_key(self) -> None:
        secret = "tvly_example_secret_key"
        masked = mask_secret(secret)
        self.assertNotEqual(masked, secret)
        self.assertNotIn("example_secret", masked)
        self.assertTrue(masked.startswith("tv"))

    def test_tavily_skips_without_api_key(self) -> None:
        with patch.dict(os.environ, {"NEWS_TAVILY_ENABLED": "true"}, clear=True), patch(
            "stock_picker.tavily_news.load_project_env", return_value=False
        ):
            records, status = fetch_tavily_news_records(
                {"ticker": "AAA", "name": "AAA Corp"},
                {},
            )
        self.assertEqual(records, [])
        self.assertEqual(status, "missing_api_key")


if __name__ == "__main__":
    unittest.main()
