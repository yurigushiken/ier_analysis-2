"""Transition computation utilities."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import pandas as pd


def compute_transitions(df: pd.DataFrame, *, aoi_nodes: Sequence[str]) -> pd.DataFrame:
    """Return per-trial transition counts (long form)."""
    filtered = df[df["aoi_category"].isin(aoi_nodes)].copy()
    filtered = filtered.sort_values(
        ["participant_id", "trial_number", "gaze_start_frame", "gaze_end_frame"]
    )
    rows = []
    for (participant_id, trial_number), trial_df in filtered.groupby(
        ["participant_id", "trial_number"], sort=False
    ):
        categories = trial_df["aoi_category"].tolist()
        if len(categories) < 2:
            continue
        age = float(trial_df["participant_age_months"].iloc[0])
        condition = trial_df["condition"].iloc[0]
        transitions = []
        for current, nxt in zip(categories, categories[1:]):
            if current == nxt:
                continue
            if current not in aoi_nodes or nxt not in aoi_nodes:
                continue
            transitions.append((current, nxt))
        counts = Counter(transitions)
        for (from_aoi, to_aoi), count in counts.items():
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_number": trial_number,
                    "condition": condition,
                    "participant_age_months": age,
                    "from_aoi": from_aoi,
                    "to_aoi": to_aoi,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def to_wide_counts(transitions_df: pd.DataFrame, *, aoi_nodes: Sequence[str]) -> pd.DataFrame:
    """Pivot transition counts so each column is a from_to pair."""
    if transitions_df.empty:
        return transitions_df.copy()
    transitions_df = transitions_df.copy()
    transitions_df["transition_key"] = transitions_df.apply(
        lambda row: f"{row['from_aoi']}_to_{row['to_aoi']}", axis=1
    )
    keys = []
    for from_aoi in aoi_nodes:
        for to_aoi in aoi_nodes:
            if from_aoi == to_aoi:
                continue
            keys.append(f"{from_aoi}_to_{to_aoi}")
    pivot = transitions_df.pivot_table(
        index=["participant_id", "trial_number", "condition", "participant_age_months"],
        columns="transition_key",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    # Ensure all expected columns exist
    for key in keys:
        if key not in pivot.columns:
            pivot[key] = 0
    ordered = [
        "participant_id",
        "trial_number",
        "condition",
        "participant_age_months",
        *keys,
    ]
    return pivot[ordered]

