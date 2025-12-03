"""Aggregation utilities for transition matrices."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd


def assign_cohort(age: float, cohorts: List[Dict]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None


def build_transition_matrix(
    transitions_df: pd.DataFrame,
    *,
    cohorts: List[Dict],
    aoi_nodes: Sequence[str],
) -> pd.DataFrame:
    """Aggregate transitions per cohort returning long-form matrix data."""
    if transitions_df.empty:
        return pd.DataFrame(
            columns=["cohort", "from_aoi", "to_aoi", "mean_count"]
        )
    df = transitions_df.copy()
    df["cohort"] = df["participant_age_months"].apply(lambda age: assign_cohort(age, cohorts))
    df = df.dropna(subset=["cohort"])
    if df.empty:
        raise ValueError("All transitions were dropped after cohort assignment.")

    trial_rows = (
        df.groupby(["cohort", "participant_id", "trial_number"])
        .size()
        .reset_index(name="transition_events")
    )
    trials_per_cohort = trial_rows.groupby("cohort").size().to_dict()

    grouped = (
        df.groupby(["cohort", "from_aoi", "to_aoi"])["count"]
        .sum()
        .reset_index()
    )

    records = []
    for cohort in [c["label"] for c in cohorts]:
        cohort_trials = float(trials_per_cohort.get(cohort, 0))
        for from_aoi in aoi_nodes:
            for to_aoi in aoi_nodes:
                if from_aoi == to_aoi:
                    continue
                match = grouped[
                    (grouped["cohort"] == cohort)
                    & (grouped["from_aoi"] == from_aoi)
                    & (grouped["to_aoi"] == to_aoi)
                ]
                total_count = match["count"].iloc[0] if not match.empty else 0.0
                mean_count = 0.0 if cohort_trials == 0 else total_count / cohort_trials
                records.append(
                    {
                        "cohort": cohort,
                        "from_aoi": from_aoi,
                        "to_aoi": to_aoi,
                        "mean_count": mean_count,
                    }
                )
    return pd.DataFrame(records)

