"""Data wrangling helpers for the tri-argument fixation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

MONAD_CATEGORIES: Sequence[str] = ("Toy_Only", "Man_Only", "Woman_Only", "Other")
DYAD_CATEGORIES: Sequence[str] = ("Man_Toy", "Woman_Toy", "Woman_Man")
FULL_CATEGORY: Sequence[str] = ("Full_Trifecta",)
EVENT_CATEGORY_ORDER: Sequence[str] = (*MONAD_CATEGORIES, *DYAD_CATEGORIES, *FULL_CATEGORY)


def load_fixations(config: Dict) -> pd.DataFrame:
    """Return the gaze fixation dataframe specified by the analysis config."""
    threshold_dir = Path(config["input_threshold_dir"]).expanduser().resolve()
    filename = config.get("input_filename", "gaze_fixations_combined_min4.csv")
    input_path = threshold_dir / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Gaze fixation file not found: {input_path}")
    return pd.read_csv(input_path)


def compute_trial_metrics(
    df: pd.DataFrame,
    *,
    aoi_groups: Dict[str, List[str]],
    condition_codes: List[str],
    frame_window: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Compute per-trial coverage for all AOI groups."""
    filtered = df[df["condition"].isin(condition_codes)].copy()
    if frame_window:
        start = frame_window["start"]
        end = frame_window["end"]
        filtered = filtered[
            (filtered["gaze_end_frame"] >= start) & (filtered["gaze_start_frame"] <= end)
        ].copy()

    grouped = filtered.groupby(["participant_id", "trial_number", "condition"], sort=False)
    rows: List[Dict[str, object]] = []
    for (_, _, _), trial_df in grouped:
        row = trial_df.iloc[0]
        rows.append(
            {
                "participant_id": row["participant_id"],
                "trial_number": row["trial_number"],
                "condition": row["condition"],
                "participant_type": row["participant_type"],
                "participant_age_months": row["participant_age_months"],
                "tri_argument_success": _has_all_aoi(trial_df, aoi_groups),
            }
        )
    return pd.DataFrame(rows)


def filter_by_min_trials(trial_df: pd.DataFrame, *, min_trials: int) -> pd.DataFrame:
    """Drop participants who do not contribute the configured number of trials."""
    if min_trials <= 1:
        return trial_df
    counts = trial_df.groupby("participant_id")["trial_number"].nunique()
    eligible = counts[counts >= min_trials].index
    return trial_df[trial_df["participant_id"].isin(eligible)].copy()


def assign_cohort(age_months: float, cohorts: List[Dict[str, object]]) -> Optional[str]:
    """Return the cohort label for the given age."""
    for cohort in cohorts:
        if cohort["min_months"] <= age_months <= cohort["max_months"]:
            return cohort["label"]
    return None


def attach_cohorts(trial_df: pd.DataFrame, cohorts: List[Dict[str, object]]) -> pd.DataFrame:
    """Annotate each trial with a cohort label and drop rows outside the definitions."""
    trial_df = trial_df.copy()
    trial_df["cohort"] = trial_df["participant_age_months"].apply(
        lambda age: assign_cohort(age, cohorts)
    )
    trial_df = trial_df.dropna(subset=["cohort"])
    return trial_df


def summarize_by_cohort(trial_df: pd.DataFrame, cohorts: List[Dict[str, object]]) -> pd.DataFrame:
    """Aggregate success rates per cohort."""
    summary = (
        trial_df.groupby("cohort")
        .agg(
            participants=("participant_id", "nunique"),
            total_trials=("trial_number", "count"),
            successful_trials=("tri_argument_success", "sum"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["successful_trials"] / summary["total_trials"]
    cohort_order = [c["label"] for c in cohorts]
    summary["cohort"] = pd.Categorical(summary["cohort"], categories=cohort_order, ordered=True)
    summary = summary.sort_values("cohort").reset_index(drop=True)
    return summary


def determine_output_root(config: Dict, config_path: Path) -> Path:
    """Resolve the output directory for a config."""
    configured = config.get("output_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    config_stem = config_path.stem
    parent = config_path.parent
    if parent.name == "configs":
        analysis_root = parent.parent
    else:
        analysis_root = parent
    return (analysis_root / config_stem).resolve()


def classify_event_structure(
    fixations_df: pd.DataFrame,
    trial_results: pd.DataFrame,
    *,
    aoi_groups: Dict[str, List[str]],
    condition_codes: List[str],
    frame_window: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Return per-trial event structure categories (monads/dyads/trifecta)."""
    relevant_trials = trial_results[trial_results["condition"].isin(condition_codes)].copy()
    if relevant_trials.empty:
        return relevant_trials[
            ["participant_id", "trial_number", "condition", "cohort", "tri_argument_success"]
        ].assign(event_category=pd.Series(dtype="object"))

    filtered = fixations_df[fixations_df["condition"].isin(condition_codes)].copy()
    if frame_window:
        start = frame_window["start"]
        end = frame_window["end"]
        filtered = filtered[
            (filtered["gaze_end_frame"] >= start) & (filtered["gaze_start_frame"] <= end)
        ].copy()

    arg_lookup = _invert_aoi_groups(aoi_groups)
    filtered["argument_label"] = filtered["aoi_category"].map(arg_lookup)
    arg_sets = (
        filtered.groupby(["participant_id", "trial_number", "condition"], sort=False)["argument_label"]
        .agg(lambda values: {value for value in values if pd.notna(value)})
        .to_dict()
    )

    def determine_category(row: pd.Series) -> str:
        if bool(row["tri_argument_success"]):
            return "Full_Trifecta"
        key = (row["participant_id"], row["trial_number"], row["condition"])
        args_seen = arg_sets.get(key, set())
        return _categorize_arguments(args_seen)

    relevant_trials = relevant_trials.copy()
    relevant_trials["event_category"] = relevant_trials.apply(determine_category, axis=1)
    return relevant_trials[
        [
            "participant_id",
            "trial_number",
            "condition",
            "cohort",
            "tri_argument_success",
            "event_category",
        ]
    ].copy()


def summarize_event_structure(
    events: pd.DataFrame, cohorts: List[Dict[str, object]]
) -> pd.DataFrame:
    """Aggregate event categories by cohort with counts + percentages."""
    counts: Dict[Tuple[str, str], int] = {}
    if not events.empty:
        grouped = events.groupby(["cohort", "event_category"]).size()
        counts = grouped.to_dict()

    cohort_order = [c["label"] for c in cohorts]
    records: List[Dict[str, object]] = []
    for cohort in cohort_order:
        cohort_total = sum(counts.get((cohort, category), 0) for category in EVENT_CATEGORY_ORDER)
        for category in EVENT_CATEGORY_ORDER:
            count = counts.get((cohort, category), 0)
            percentage = 0.0 if cohort_total == 0 else (count / cohort_total) * 100
            records.append(
                {
                    "cohort": cohort,
                    "event_category": category,
                    "count": int(count),
                    "percentage": percentage,
                }
            )
    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df["cohort"] = pd.Categorical(
            summary_df["cohort"], categories=cohort_order, ordered=True
        )
        summary_df["event_category"] = pd.Categorical(
            summary_df["event_category"], categories=EVENT_CATEGORY_ORDER, ordered=True
        )
        summary_df = summary_df.sort_values(["cohort", "event_category"]).reset_index(drop=True)
    return summary_df


def _has_all_aoi(trial_df: pd.DataFrame, aoi_groups: Dict[str, List[str]]) -> bool:
    for aois in aoi_groups.values():
        if not trial_df["aoi_category"].isin(aois).any():
            return False
    return True


def _invert_aoi_groups(aoi_groups: Dict[str, List[str]]) -> Dict[str, str]:
    mapping = {}
    for argument, categories in aoi_groups.items():
        for category in categories:
            mapping[category] = argument
    return mapping


def _categorize_arguments(args_seen: Set[str]) -> str:
    normalized = {arg.lower() for arg in args_seen}
    if normalized == {"man", "toy"}:
        return "Man_Toy"
    if normalized == {"woman", "toy"}:
        return "Woman_Toy"
    if normalized == {"woman", "man"}:
        return "Woman_Man"
    if normalized == {"toy"}:
        return "Toy_Only"
    if normalized == {"man"}:
        return "Man_Only"
    if normalized == {"woman"}:
        return "Woman_Only"
    return "Other"

