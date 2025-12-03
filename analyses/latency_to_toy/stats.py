"""Statistical helpers for latency-to-toy analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_adult_reference_gee(
    latency_df: pd.DataFrame,
    cohorts: List[Dict[str, int]],
) -> Tuple[pd.DataFrame, str]:
    """Run latency_frames ~ C(cohort, Treatment(reference='Adults'))."""
    if latency_df.empty:
        raise ValueError("No latency data available for GEE.")
    working = latency_df.copy()
    cohort_labels = [c["label"] for c in cohorts]
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        raise ValueError("No rows remaining after cohort assignment.")
    reference = None
    for label in cohort_labels:
        if "adult" in label.lower():
            reference = label
            break
    if reference is None:
        reference = cohort_labels[-1]
    working["cohort"] = pd.Categorical(
        working["cohort"],
        categories=cohort_labels,
        ordered=True,
    )
    formula = f"latency_frames ~ C(cohort, Treatment(reference='{reference}'))"
    model = smf.gee(
        formula=formula,
        groups="participant_id",
        data=working,
        family=sm.families.Gaussian(),
    )
    result = model.fit()
    stats_rows = []
    conf_int = result.conf_int()
    summary_lines = [
        "Latency GEE with adult reference",
        f"Reference cohort: {reference}",
        f"Participants: {working['participant_id'].nunique()}",
        f"Trials: {len(working)}",
        "",
        result.summary().as_text(),
    ]
    for cohort in working["cohort"].cat.categories:
        if cohort == reference:
            stats_rows.append(
                {
                    "cohort": cohort,
                    "coef": 0.0,
                    "pvalue": np.nan,
                    "std_err": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        term = f"C(cohort, Treatment(reference='{reference}'))[T.{cohort}]"
        coef = float(result.params.get(term, np.nan))
        pvalue = float(result.pvalues.get(term, np.nan))
        std_err = float(result.bse.get(term, np.nan))
        ci_low, ci_high = conf_int.loc[term]
        stats_rows.append(
            {
                "cohort": cohort,
                "coef": coef,
                "pvalue": pvalue,
                "std_err": std_err,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(stats_rows), "\n".join(summary_lines)
def _assign_cohort(age: float, cohorts: List[Dict[str, int]]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None


def summarize_adult_vs_infant(
    latency_df: pd.DataFrame,
    *,
    infant_cohorts: List[Dict[str, int]],
    cohorts: List[Dict[str, int]],
) -> Dict[str, float]:
    """Return adult vs infant mean latency comparison."""
    if latency_df.empty:
        return {}

    if infant_cohorts:
        infant_min = infant_cohorts[0]["min_months"]
        infant_max = infant_cohorts[-1]["max_months"]
    else:
        infant_min = cohorts[0]["min_months"]
        infant_max = cohorts[0]["max_months"]

    adult_min = None
    for cohort in cohorts:
        if "adult" in cohort["label"].lower():
            adult_min = cohort["min_months"]
            break
    if adult_min is None:
        adult_min = cohorts[-1]["min_months"]

    infants = latency_df[
        (latency_df["participant_age_months"] >= infant_min)
        & (latency_df["participant_age_months"] <= infant_max)
    ]["latency_frames"]
    adults = latency_df[latency_df["participant_age_months"] >= adult_min]["latency_frames"]

    return {
        "infant_mean_frames": float(infants.mean()) if not infants.empty else float("nan"),
        "infant_trials": int(infants.count()),
        "adult_mean_frames": float(adults.mean()) if not adults.empty else float("nan"),
        "adult_trials": int(adults.count()),
    }


def run_infant_linear_trend(
    latency_df: pd.DataFrame,
    *,
    infant_cohorts: List[Dict[str, int]],
) -> Tuple[Dict[str, float], str]:
    """Run latency_frames ~ age_numeric for infant cohorts only."""
    if latency_df.empty:
        return {}, "Latency linear trend: no latency data available."
    if not infant_cohorts:
        return {}, "Latency linear trend: no infant cohorts configured."
    infant_min = infant_cohorts[0]["min_months"]
    infant_max = infant_cohorts[-1]["max_months"]
    working = latency_df[
        (latency_df["participant_age_months"] >= infant_min)
        & (latency_df["participant_age_months"] <= infant_max)
    ].copy()
    if working.empty:
        return {}, "Latency linear trend: no infant trials within the specified range."
    working["age_numeric"] = working["participant_age_months"]
    model = smf.gee(
        "latency_frames ~ age_numeric",
        groups="participant_id",
        data=working,
        family=sm.families.Gaussian(),
    )
    try:
        result = model.fit()
    except ValueError as exc:
        return {}, f"Latency linear trend failed to converge: {exc}"
    coef = float(result.params.get("age_numeric", 0.0))
    intercept = float(result.params.get("Intercept", 0.0))
    pvalue = float(result.pvalues.get("age_numeric", float("nan")))
    stats = {
        "coef": coef,
        "intercept": intercept,
        "pvalue": pvalue,
        "age_min": infant_min,
        "age_max": infant_max,
        "n_participants": int(working["participant_id"].nunique()),
        "n_trials": int(len(working)),
    }
    summary_lines = [
        f"Latency linear trend for infants ({infant_min}-{infant_max} months)",
        f"Participants: {stats['n_participants']}",
        f"Trials: {stats['n_trials']}",
        "",
        result.summary().as_text(),
    ]
    return stats, "\n".join(summary_lines)

