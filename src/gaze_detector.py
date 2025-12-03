"""Standalone gaze fixation detection for ier_analysis-2."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .aoi_mapper import map_what_where


OUTPUT_COLUMNS = [
    "participant_id",
    "participant_type",
    "participant_age_months",
    "trial_number",
    "condition",
    "segment",
    "aoi_category",
    "gaze_start_frame",
    "gaze_end_frame",
    "gaze_duration_frames",
    "gaze_duration_ms",
    "gaze_onset_time",
    "gaze_offset_time",
    "min_frames",
]


def detect_fixations(dataframe: pd.DataFrame, *, min_frames: int) -> pd.DataFrame:
    """Detect gaze fixations using the provided minimum frame threshold."""
    if min_frames < 1:
        raise ValueError("min_frames must be at least 1.")

    if dataframe.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    ordered = dataframe.sort_values(["Participant", "trial_number", "Frame Number"]).copy()
    fixations: List[Dict[str, object]] = []

    for (_, _), group in ordered.groupby(["Participant", "trial_number"], sort=False):
        _extract_fixations_from_group(group, fixations, min_frames=min_frames)

    if not fixations:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.DataFrame(fixations, columns=OUTPUT_COLUMNS)


def _extract_fixations_from_group(group: pd.DataFrame, fixations: List[Dict[str, object]], *, min_frames: int) -> None:
    current_aoi = None
    buffer: List[pd.Series] = []

    for _, row in group.iterrows():
        try:
            aoi = map_what_where(row["What"], row["Where"])
        except ValueError:
            current_aoi = None
            buffer = []
            continue

        if buffer and _has_event_boundary(buffer[-1], row):
            _finalize_fixation(buffer, fixations, min_frames)
            current_aoi = None
            buffer = []

        if aoi == current_aoi:
            buffer.append(row)
        else:
            _finalize_fixation(buffer, fixations, min_frames)
            current_aoi = aoi
            buffer = [row]

    _finalize_fixation(buffer, fixations, min_frames)


def _has_event_boundary(previous_row: pd.Series, current_row: pd.Series) -> bool:
    prev_count = int(previous_row.get("frame_count_trial_number", 0))
    curr_count = int(current_row.get("frame_count_trial_number", 0))
    if curr_count < prev_count:
        return True
    prev_condition = previous_row.get("event_verified")
    curr_condition = current_row.get("event_verified")
    return str(curr_condition) != str(prev_condition)


def _finalize_fixation(buffer: List[pd.Series], fixations: List[Dict[str, object]], min_frames: int) -> None:
    if len(buffer) < min_frames:
        return

    first = buffer[0]
    last = buffer[-1]
    duration_ms = (float(last["Offset"]) - float(first["Onset"])) * 1000.0

    fixations.append(
        {
            "participant_id": str(first["Participant"]),
            "participant_type": str(first["participant_type"]),
            "participant_age_months": int(first["participant_age_months"]),
            "trial_number": int(first["trial_number"]),
            "condition": str(first["event_verified"]),
            "segment": str(first["segment"]),
            "aoi_category": map_what_where(first["What"], first["Where"]),
            "gaze_start_frame": int(first["frame_count_trial_number"]),
            "gaze_end_frame": int(last["frame_count_trial_number"]),
            "gaze_duration_frames": len(buffer),
            "gaze_duration_ms": duration_ms,
            "gaze_onset_time": float(first["Onset"]),
            "gaze_offset_time": float(last["Offset"]),
            "min_frames": int(min_frames),
        }
    )


__all__ = ["detect_fixations"]

