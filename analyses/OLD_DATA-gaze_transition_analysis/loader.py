"""Data loading helpers for the gaze transition analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT = Path("outputs/min4-70_percent/gaze_fixations_combined_min4.csv")


def load_fixations(path: Path | None = None, *, condition_codes: List[str]) -> pd.DataFrame:
    """Load fixation rows and filter to the requested condition codes."""
    source = Path(path or DEFAULT_INPUT).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Fixation file not found: {source}")
    df = pd.read_csv(source)
    if "condition" not in df.columns:
        raise ValueError("Fixation CSV must contain a 'condition' column.")
    filtered = df[df["condition"].isin(condition_codes)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for conditions: {condition_codes}")
    required_cols = [
        "participant_id",
        "trial_number",
        "aoi_category",
        "gaze_start_frame",
        "gaze_end_frame",
        "participant_age_months",
    ]
    missing = [col for col in required_cols if col not in filtered.columns]
    if missing:
        raise ValueError(f"Fixation CSV missing required columns: {missing}")
    return filtered

