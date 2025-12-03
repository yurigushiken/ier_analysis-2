"""Latency computation helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

FRAME_RATE = 30.0  # frames per second


def compute_latencies(
    fixations_df: pd.DataFrame,
    *,
    window_start: int,
    window_end: int,
    toy_aois: List[str],
) -> pd.DataFrame:
    """Return per-trial latency records."""
    if fixations_df.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "trial_number",
                "condition",
                "participant_age_months",
                "latency_frames",
                "latency_ms",
                "latency_seconds",
            ]
        )

    filtered = fixations_df[fixations_df["aoi_category"].isin(toy_aois)].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values(
        ["participant_id", "trial_number", "condition", "gaze_start_frame"]
    )

    records = []
    for (participant_id, trial_number, condition), trial_df in filtered.groupby(
        ["participant_id", "trial_number", "condition"], sort=False
    ):
        latency = _first_latency_for_trial(trial_df, window_start, window_end)
        if latency is None:
            continue
        rows = trial_df.iloc[0]
        frames = latency
        seconds = frames / FRAME_RATE
        ms = seconds * 1000.0
        records.append(
            {
                "participant_id": participant_id,
                "trial_number": trial_number,
                "condition": condition,
                "participant_age_months": float(rows["participant_age_months"]),
                "latency_frames": float(frames),
                "latency_ms": float(ms),
                "latency_seconds": float(seconds),
            }
        )
    return pd.DataFrame(records)


def summarize_by_cohort(
    latency_df: pd.DataFrame,
    *,
    cohorts: List[Dict[str, int]],
) -> pd.DataFrame:
    """Aggregate latency metrics per cohort."""
    columns = [
        "cohort",
        "mean_latency_frames",
        "sem_latency_frames",
        "mean_latency_ms",
        "sem_latency_ms",
        "mean_latency_seconds",
        "sem_latency_seconds",
        "trials",
    ]
    if latency_df.empty:
        return pd.DataFrame(columns=columns)

    working = latency_df.copy()
    if "latency_ms" not in working.columns:
        working["latency_ms"] = working["latency_frames"] / FRAME_RATE * 1000.0
    if "latency_seconds" not in working.columns:
        working["latency_seconds"] = working["latency_frames"] / FRAME_RATE
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        return pd.DataFrame(columns=columns)

    def _sem(values: pd.Series) -> float:
        if len(values) <= 1:
            return 0.0
        return float(values.std(ddof=1) / np.sqrt(len(values)))

    summary = (
        working.groupby("cohort")
        .agg(
            mean_latency_frames=("latency_frames", "mean"),
            sem_latency_frames=("latency_frames", _sem),
            mean_latency_ms=("latency_ms", "mean"),
            sem_latency_ms=("latency_ms", _sem),
            mean_latency_seconds=("latency_seconds", "mean"),
            sem_latency_seconds=("latency_seconds", _sem),
            trials=("latency_frames", "count"),
        )
        .reset_index()
    )

    cohort_order = [c["label"] for c in cohorts]
    summary["cohort"] = pd.Categorical(summary["cohort"], categories=cohort_order, ordered=True)
    summary = summary.sort_values("cohort").reset_index(drop=True)
    return summary


def _first_latency_for_trial(
    trial_df: pd.DataFrame,
    window_start: int,
    window_end: int,
) -> float | None:
    """Return latency frames for first qualifying fixation or None."""
    for row in trial_df.itertuples():
        start = int(row.gaze_start_frame)
        end = int(row.gaze_end_frame)
        if end < window_start:
            continue
        if start > window_end:
            break
        if start < window_start <= end:
            return 0.0
        if window_start <= start <= window_end:
            return float(start - window_start)
    return None


def _assign_cohort(age: float, cohorts: List[Dict[str, int]]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None

