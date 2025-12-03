"""Overlap detection helpers for reaction look analysis."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd


def compute_reaction_flags(
    fixations_df: pd.DataFrame,
    *,
    target_aoi: str | Sequence[str],
    window_start: int,
    window_end: int,
    condition_codes: List[str],
) -> pd.DataFrame:
    """Return per-trial looked_at_target flags."""
    if fixations_df.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "trial_number",
                "condition",
                "participant_age_months",
                "looked_at_target",
            ]
        )
    df = fixations_df[fixations_df["condition"].isin(condition_codes)].copy()
    if df.empty:
        return df
    df = df.sort_values(
        ["participant_id", "trial_number", "condition", "gaze_start_frame"]
    )
    target_set = _normalize_target_aoi(target_aoi)
    rows = []
    for key, trial_df in df.groupby(["participant_id", "trial_number", "condition"], sort=False):
        subset = trial_df[trial_df["aoi_category"].isin(target_set)]
        looked = 0
        for fix in subset.itertuples():
            start = int(fix.gaze_start_frame)
            end = int(fix.gaze_end_frame)
            if end < window_start:
                continue
            if start > window_end:
                break
            if start <= window_end and end >= window_start:
                looked = 1
                break
        meta = trial_df.iloc[0]
        rows.append(
            {
                "participant_id": key[0],
                "trial_number": key[1],
                "condition": key[2],
                "participant_age_months": float(meta["participant_age_months"]),
                "looked_at_target": looked,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_cohort(
    reaction_df: pd.DataFrame,
    *,
    cohorts: List[Dict],
) -> pd.DataFrame:
    """Return mean looked proportion per cohort."""
    columns = ["cohort", "mean_looked", "trials"]
    if reaction_df.empty:
        return pd.DataFrame(columns=columns)
    working = reaction_df.copy()
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        working.groupby("cohort")
        .agg(
            mean_looked=("looked_at_target", "mean"),
            trials=("trial_number", "count"),
        )
        .reset_index()
    )
    order = [c["label"] for c in cohorts]
    summary["cohort"] = pd.Categorical(summary["cohort"], categories=order, ordered=True)
    return summary.sort_values("cohort").reset_index(drop=True)


def _normalize_target_aoi(target: str | Sequence[str]) -> Set[str]:
    if isinstance(target, str):
        return {target}
    if isinstance(target, Sequence):
        normalized: Set[str] = set()
        for value in target:
            if not isinstance(value, str):
                raise TypeError("target_aoi entries must be strings.")
            normalized.add(value)
        if not normalized:
            raise ValueError("target_aoi list must contain at least one string value.")
        return normalized
    raise TypeError("target_aoi must be a string or a sequence of strings.")


def _assign_cohort(age: float, cohorts: List[Dict]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None

