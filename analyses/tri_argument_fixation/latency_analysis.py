"""Processing-efficiency (latency) helpers for the tri-argument analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FRAME_RATE = 30.0  # frames per second


def compute_latency_metrics(
    fixations_df: pd.DataFrame,
    trial_results: pd.DataFrame,
    *,
    aoi_groups: Dict[str, List[str]],
    condition_codes: List[str],
    frame_window: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Return per-trial latency (ms/frames) for successful trifecta trials."""
    successful = trial_results[
        (trial_results["tri_argument_success"] == 1)
        & (trial_results["condition"].isin(condition_codes))
    ].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "trial_number",
                "condition",
                "cohort",
                "participant_age_months",
                "latency_ms",
                "latency_frames",
            ]
        )

    filtered = fixations_df[fixations_df["condition"].isin(condition_codes)].copy()
    window_start = 0
    if frame_window:
        start = frame_window["start"]
        end = frame_window["end"]
        window_start = frame_window.get("event_onset", start)
        filtered = filtered[
            (filtered["gaze_end_frame"] >= start) & (filtered["gaze_start_frame"] <= end)
        ].copy()

    filtered = filtered.sort_values(
        ["participant_id", "trial_number", "condition", "gaze_start_frame"]
    )
    arg_lookup = _build_argument_lookup(aoi_groups)
    required_count = len(aoi_groups)
    success_index = successful.set_index(["participant_id", "trial_number", "condition"])
    rows: List[Dict[str, object]] = []

    for key, trial_df in filtered.groupby(
        ["participant_id", "trial_number", "condition"], sort=False
    ):
        if key not in success_index.index:
            continue
        seen = set()
        latency_ms = None
        latency_frames = None
        for fix in trial_df.itertuples():
            argument = arg_lookup.get(fix.aoi_category)
            if not argument:
                continue
            seen.add(argument)
            if len(seen) == required_count:
                absolute_frame = int(
                    getattr(
                        fix,
                        "gaze_start_frame",
                        getattr(fix, "frame_count_trial_number", 0),
                    )
                )
                latency_frames = max(absolute_frame - window_start, 0)
                latency_seconds = latency_frames / FRAME_RATE
                latency_ms = latency_seconds * 1000.0
                break
        if latency_ms is None:
            continue
        meta = success_index.loc[key]
        rows.append(
            {
                "participant_id": key[0],
                "trial_number": key[1],
                "condition": key[2],
                "cohort": meta["cohort"],
                "participant_age_months": meta["participant_age_months"],
                "latency_ms": latency_ms,
                "latency_frames": latency_frames,
                "latency_seconds": latency_frames / FRAME_RATE,
            }
        )
    return pd.DataFrame(rows)


def summarize_latency_by_cohort(
    latency_df: pd.DataFrame,
    cohorts: List[Dict[str, object]],
) -> pd.DataFrame:
    """Aggregate mean/SEM latency per cohort."""
    if latency_df.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                "mean_latency_ms",
                "sem_latency_ms",
                "mean_latency_frames",
                "sem_latency_frames",
                "mean_latency_seconds",
                "sem_latency_seconds",
                "trials",
            ]
        )

    def _sem(values: pd.Series) -> float:
        if len(values) <= 1:
            return 0.0
        return float(values.std(ddof=1) / np.sqrt(len(values)))

    grouped = (
        latency_df.groupby("cohort")
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
    grouped["cohort"] = pd.Categorical(grouped["cohort"], categories=cohort_order, ordered=True)
    grouped = grouped.sort_values("cohort").reset_index(drop=True)
    return grouped


def run_latency_trend(
    latency_df: pd.DataFrame,
    *,
    infant_cohorts: List[Dict[str, object]],
) -> Tuple[Dict[str, float], str]:
    """Run GEE linear trend on latency vs. age (infant cohorts only)."""
    if latency_df.empty:
        return {}, "Latency trend: no successful trials available."
    if not infant_cohorts:
        return {}, "Latency trend: no infant cohorts defined."

    min_age = infant_cohorts[0]["min_months"]
    max_age = infant_cohorts[-1]["max_months"]
    working = latency_df[
        (latency_df["participant_age_months"] >= min_age)
        & (latency_df["participant_age_months"] <= max_age)
    ].copy()
    if working.empty:
        return {}, "Latency trend: no infant trials in the specified age range."
    working["age_numeric"] = working["participant_age_months"]
    model = smf.gee(
        "latency_ms ~ age_numeric",
        groups="participant_id",
        data=working,
        family=sm.families.Gaussian(),
    )
    try:
        result = model.fit()
    except ValueError as exc:
        return {}, f"Latency trend: model failed to converge ({exc})."

    coef = float(result.params.get("age_numeric", float("nan")))
    pvalue = float(result.pvalues.get("age_numeric", float("nan")))
    summary_text = "\n".join(
        [
            "Processing-efficiency (latency) linear trend",
            f"Infant range: {min_age}–{max_age} months",
            f"Coefficient (ms/month): {coef:.3f}",
            f"p-value: {pvalue:.4f}",
            "",
            result.summary().as_text(),
        ]
    )
    return {"coef": coef, "pvalue": pvalue}, summary_text


def _build_argument_lookup(aoi_groups: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for argument, categories in aoi_groups.items():
        for category in categories:
            lookup[category] = argument.lower()
    return lookup

