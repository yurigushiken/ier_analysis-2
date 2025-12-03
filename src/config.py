"""Configuration constants for the ier_analysis-2 module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExtensionConfig:
    """Simple container for project paths and thresholds."""

    repo_root: Path
    raw_child_dir: Path
    raw_adult_dir: Path
    output_root: Path
    thresholds: List[int]
    required_columns: List[str]


def _default_config() -> ExtensionConfig:
    repo_root = Path(__file__).resolve().parents[1]
    raw_base = repo_root / "data" / "csvs_human_verified_vv"
    return ExtensionConfig(
        repo_root=repo_root,
        raw_child_dir=raw_base / "child",
        raw_adult_dir=raw_base / "adult",
        output_root=repo_root / "outputs",
        thresholds=[3, 4, 5],
        required_columns=[
            "Participant",
            "Frame Number",
            "What",
            "Where",
            "Onset",
            "Offset",
            "trial_number",
            "participant_type",
            "participant_age_months",
            "event_verified",
            "segment",
            "frame_count_trial_number",
        ],
    )


EXTENSION_CONFIG = _default_config()

__all__ = ["ExtensionConfig", "EXTENSION_CONFIG"]

