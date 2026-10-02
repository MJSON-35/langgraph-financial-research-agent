"""Small reusable helpers for exporting step summaries to disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_picker.security import sanitize_for_export


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP_SUMMARY_DIR = PROJECT_ROOT / "outputs" / "step_summaries"


def save_step_summary(
    stage_name: str,
    summary_data: dict[str, Any],
    step_number: int | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Save one step summary as JSON under ``outputs/step_summaries``."""
    step_summary_dir = _resolve_step_summary_dir(output_dir)
    step_summary_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage_name": stage_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **summary_data,
    }
    file_name = _build_step_summary_filename(stage_name, step_number)
    output_path = step_summary_dir / file_name
    output_path.write_text(json.dumps(sanitize_for_export(payload), indent=2), encoding="utf-8")
    return output_path


def _resolve_step_summary_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return DEFAULT_STEP_SUMMARY_DIR
    path = Path(output_dir)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _build_step_summary_filename(stage_name: str, step_number: int | None) -> str:
    if step_number is None:
        return f"{stage_name}.json"
    return f"{step_number:02d}_{stage_name}.json"
