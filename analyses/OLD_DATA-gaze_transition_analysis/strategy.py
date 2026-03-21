"""Strategy aggregation and statistics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FACE_AOIS = {"man_face", "woman_face"}
BODY_AOIS = {"man_body", "woman_body"}
TOY_AOIS = {"toy_present", "toy_location"}

AGENT_AGENT_ATTENTION_KEY = "agent_agent_attention"
AGENT_OBJECT_BINDING_KEY = "agent_object_binding"
MOTION_TRACKING_KEY = "motion_tracking"

AGENT_AGENT_ATTENTION_PCT = f"{AGENT_AGENT_ATTENTION_KEY}_pct"
AGENT_OBJECT_BINDING_PCT = f"{AGENT_OBJECT_BINDING_KEY}_pct"
MOTION_TRACKING_PCT = f"{MOTION_TRACKING_KEY}_pct"

AGENT_AGENT_ATTENTION_MEAN = f"{AGENT_AGENT_ATTENTION_KEY}_mean"
AGENT_OBJECT_BINDING_MEAN = f"{AGENT_OBJECT_BINDING_KEY}_mean"
MOTION_TRACKING_MEAN = f"{MOTION_TRACKING_KEY}_mean"

STRATEGY_COLUMNS = [
    (AGENT_AGENT_ATTENTION_PCT, "Agent-Agent Attention"),
    (AGENT_OBJECT_BINDING_PCT, "Agent-Object Binding"),
    (MOTION_TRACKING_PCT, "Motion Tracking"),
]


def compute_strategy_proportions(transitions_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-trial normalized strategy proportions."""
    if transitions_df.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "trial_number",
                "condition",
                "participant_age_months",
                "total_transitions",
                AGENT_AGENT_ATTENTION_PCT,
                AGENT_OBJECT_BINDING_PCT,
                MOTION_TRACKING_PCT,
            ]
        )
    grouped = []
    for (participant_id, trial_number), trial_df in transitions_df.groupby(
        ["participant_id", "trial_number"], sort=False
    ):
        total = trial_df["count"].sum()
        if total == 0:
            continue
        counts = {
            AGENT_AGENT_ATTENTION_KEY: 0.0,
            AGENT_OBJECT_BINDING_KEY: 0.0,
            MOTION_TRACKING_KEY: 0.0,
        }
        for row in trial_df.itertuples():
            strategy_key = _categorize_transition(row.from_aoi, row.to_aoi)
            if strategy_key:
                counts[strategy_key] += row.count
        grouped.append(
            {
                "participant_id": participant_id,
                "trial_number": trial_number,
                "condition": trial_df["condition"].iloc[0],
                "participant_age_months": float(trial_df["participant_age_months"].iloc[0]),
                "total_transitions": total,
                AGENT_AGENT_ATTENTION_PCT: counts[AGENT_AGENT_ATTENTION_KEY] / total,
                AGENT_OBJECT_BINDING_PCT: counts[AGENT_OBJECT_BINDING_KEY] / total,
                MOTION_TRACKING_PCT: counts[MOTION_TRACKING_KEY] / total,
            }
        )
    return pd.DataFrame(grouped)


def _categorize_transition(from_aoi: str, to_aoi: str) -> str | None:
    if {from_aoi, to_aoi} == {"man_face", "woman_face"}:
        return AGENT_AGENT_ATTENTION_KEY
    if (
        (from_aoi in FACE_AOIS and to_aoi in TOY_AOIS)
        or (to_aoi in FACE_AOIS and from_aoi in TOY_AOIS)
    ):
        return AGENT_OBJECT_BINDING_KEY
    if (
        (from_aoi in BODY_AOIS and to_aoi in TOY_AOIS)
        or (to_aoi in BODY_AOIS and from_aoi in TOY_AOIS)
    ):
        return MOTION_TRACKING_KEY
    return None


def build_strategy_summary(
    proportions_df: pd.DataFrame,
    *,
    cohorts: List[Dict],
) -> pd.DataFrame:
    """Average strategy proportions per cohort."""
    if proportions_df.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                AGENT_AGENT_ATTENTION_MEAN,
                AGENT_OBJECT_BINDING_MEAN,
                MOTION_TRACKING_MEAN,
            ]
        )
    working = proportions_df.copy()
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    grouped = (
        working.groupby("cohort")
        .agg(
            **{
                AGENT_AGENT_ATTENTION_MEAN: (AGENT_AGENT_ATTENTION_PCT, "mean"),
                AGENT_OBJECT_BINDING_MEAN: (AGENT_OBJECT_BINDING_PCT, "mean"),
                MOTION_TRACKING_MEAN: (MOTION_TRACKING_PCT, "mean"),
            }
        )
        .reset_index()
    )
    return grouped


def build_strategy_descriptive_stats(
    proportions_df: pd.DataFrame,
    *,
    cohorts: List[Dict],
) -> pd.DataFrame:
    """Return mean/SEM per strategy/cohort."""
    if proportions_df.empty:
        return pd.DataFrame(
            columns=["cohort", "strategy", "mean", "sem", "n_trials"]
        )
    working = proportions_df.copy()
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        return pd.DataFrame(
            columns=["cohort", "strategy", "mean", "sem", "n_trials"]
        )
    rows = []
    for cohort, cohort_df in working.groupby("cohort"):
        n_trials = len(cohort_df)
        for col, label in STRATEGY_COLUMNS:
            mean = float(cohort_df[col].mean())
            if n_trials > 1:
                std = float(cohort_df[col].std(ddof=1))
                sem = std / math.sqrt(n_trials)
            else:
                sem = 0.0
            rows.append(
                {
                    "cohort": cohort,
                    "strategy": label,
                    "mean": mean,
                    "sem": sem,
                    "n_trials": n_trials,
                }
            )
    return pd.DataFrame(rows)


def _assign_cohort(age: float, cohorts: List[Dict]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None


def run_strategy_gee(
    proportions_df: pd.DataFrame,
    *,
    cohorts: List[Dict],
    value_column: str,
    metric_label: str,
) -> Tuple[pd.DataFrame, str]:
    """Run GEE on the requested strategy proportion."""
    if proportions_df.empty:
        return pd.DataFrame(), f"{metric_label}: no strategy proportions available."
    working = proportions_df.copy()
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        return pd.DataFrame(), f"{metric_label}: no rows after cohort assignment."
    reference = cohorts[0]["label"]
    working["cohort"] = pd.Categorical(
        working["cohort"],
        categories=[c["label"] for c in cohorts],
        ordered=True,
    )
    formula = f"{value_column} ~ C(cohort, Treatment(reference='{reference}'))"
    if "total_transitions" in working:
        weights = working["total_transitions"].fillna(0)
    else:
        weights = pd.Series(1.0, index=working.index)

    model = smf.gee(
        formula=formula,
        groups="participant_id",
        data=working,
        family=sm.families.Gaussian(),
        weights=weights,
    )
    report_body = ""
    try:
        result = model.fit()
        report_body = result.summary().as_text()
    except ValueError as exc:
        report_body = f"GEE failed to converge: {exc}"
        return (
            pd.DataFrame([{"cohort": reference, "coef": 0.0, "pvalue": None}]),
            _format_gee_report(metric_label, reference, working, report_body),
        )

    report_lines = [
        f"GEE results for {metric_label}",
        f"Reference cohort: {reference}",
        f"Participants: {working['participant_id'].nunique()}",
        f"Trials: {len(working)}",
        "",
        report_body,
    ]

    stats_rows = [{"cohort": reference, "coef": 0.0, "pvalue": None}]
    for cohort in working["cohort"].cat.categories[1:]:
        term = f"C(cohort, Treatment(reference='{reference}'))[T.{cohort}]"
        if term in result.params:
            stats_rows.append(
                {
                    "cohort": cohort,
                    "coef": result.params[term],
                    "pvalue": result.pvalues[term],
                }
            )
    return pd.DataFrame(stats_rows), "\n".join(report_lines)


def run_linear_trend_test(
    proportions_df: pd.DataFrame,
    *,
    infant_cohorts: List[Dict],
    value_column: str,
    metric_label: str,
) -> Tuple[Dict[str, float], str]:
    """Run linear trend (age numeric) GEE on infant cohorts (7-11 months)."""
    working = proportions_df.copy()
    if not infant_cohorts:
        return {}, f"{metric_label}: no infant cohorts available for linear trend."
    min_age = infant_cohorts[0]["min_months"]
    max_age = infant_cohorts[-1]["max_months"]
    working = working[
        (working["participant_age_months"] >= min_age)
        & (working["participant_age_months"] <= max_age)
    ].copy()
    if working.empty:
        return {}, f"{metric_label}: no rows within infant cohort range."
    working["age_numeric"] = working["participant_age_months"]
    if "total_transitions" in working:
        weights = working["total_transitions"].fillna(0)
    else:
        weights = pd.Series(1.0, index=working.index)
    model = smf.gee(
        f"{value_column} ~ age_numeric",
        groups="participant_id",
        data=working,
        family=sm.families.Gaussian(),
        weights=weights,
    )
    try:
        result = model.fit()
    except ValueError:
        return {}, f"{metric_label}: linear trend failed to converge."
    coef = float(result.params.get("age_numeric", 0.0))
    pvalue = float(result.pvalues.get("age_numeric", float("nan")))
    lines = [
        f"Linear Trend Test ({metric_label}, infants {min_age}-{max_age} mo)",
        result.summary().as_text(),
    ]
    return {"coef": coef, "pvalue": pvalue}, "\n\n".join(lines)
def build_significance_annotations(
    gee_results: pd.DataFrame,
    *,
    reference: str,
    cohort_order: Sequence[str],
) -> List[Dict[str, object]]:
    """Return significance annotations for plotting."""
    annotations = []
    if gee_results is None or gee_results.empty:
        return annotations
    order = {label: idx for idx, label in enumerate(cohort_order)}
    ref_idx = order.get(reference, 0)
    for row in gee_results.itertuples():
        if row.cohort == reference:
            continue
        to_idx = order.get(row.cohort)
        if to_idx is None or row.pvalue is None:
            continue
        label = _format_pvalue(row.pvalue)
        if not label:
            continue
        annotations.append(
            {
                "from_idx": ref_idx,
                "to_idx": to_idx,
                "label": label,
                "pvalue": row.pvalue,
            }
        )
    return annotations


def _format_pvalue(pvalue: float) -> str | None:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    if pvalue < 0.1:
        return f"p={pvalue:.2f}"
    return None


def _format_gee_report(metric_label: str, reference: str, working: pd.DataFrame, body: str) -> str:
    lines = [
        f"GEE results for {metric_label}",
        f"Reference cohort: {reference}",
        f"Participants: {working['participant_id'].nunique() if 'participant_id' in working else 0}",
        f"Trials: {len(working)}",
        "",
        body,
    ]
    return "\n".join(lines)

