"""Stat helpers for reaction look analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_adult_reference_gee(
    reaction_df: pd.DataFrame,
    cohorts: List[Dict],
) -> Tuple[pd.DataFrame, str]:
    """Adult-reference binomial GEE."""
    if reaction_df.empty:
        raise ValueError("No reaction data for GEE.")
    working = reaction_df.copy()
    working["cohort"] = working["participant_age_months"].apply(
        lambda age: _assign_cohort(age, cohorts)
    )
    working = working.dropna(subset=["cohort"])
    if working.empty:
        raise ValueError("Reaction data empty after cohort assignment.")
    labels = [c["label"] for c in cohorts]
    reference = _find_adult_label(cohorts)
    working["cohort"] = pd.Categorical(working["cohort"], categories=labels, ordered=True)
    formula = f"looked_at_target ~ C(cohort, Treatment(reference='{reference}'))"
    model = smf.gee(
        formula=formula,
        groups="participant_id",
        data=working,
        family=sm.families.Binomial(),
    )
    result = model.fit()
    stats_rows = []
    conf_int = result.conf_int()
    report_lines = [
        "Adult-reference GEE (looked_at_target)",
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
                    "odds_ratio": 1.0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_low_or": 1.0,
                    "ci_high_or": 1.0,
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
                "odds_ratio": float(np.exp(coef)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_low_or": float(np.exp(ci_low)),
                "ci_high_or": float(np.exp(ci_high)),
            }
        )
    return pd.DataFrame(stats_rows), "\n".join(report_lines)


def run_linear_trend(
    reaction_df: pd.DataFrame,
    infant_cohorts: List[Dict],
) -> Tuple[Dict[str, float], str]:
    """Infant-only linear trend looked_at_target ~ age_numeric."""
    if reaction_df.empty or not infant_cohorts:
        return {}, "No infant data for linear trend."
    min_age = infant_cohorts[0]["min_months"]
    max_age = infant_cohorts[-1]["max_months"]
    working = reaction_df[
        (reaction_df["participant_age_months"] >= min_age)
        & (reaction_df["participant_age_months"] <= max_age)
    ].copy()
    if working.empty:
        return {}, "No infant trials in specified range."
    working["age_numeric"] = working["participant_age_months"]
    model = smf.gee(
        "looked_at_target ~ age_numeric",
        groups="participant_id",
        data=working,
        family=sm.families.Binomial(),
    )
    try:
        result = model.fit()
    except ValueError as exc:
        return {}, f"Linear trend failed: {exc}"
    coef = float(result.params.get("age_numeric", np.nan))
    intercept = float(result.params.get("Intercept", np.nan))
    pvalue = float(result.pvalues.get("age_numeric", np.nan))
    report = "\n".join(
        [
            "Infant linear trend (looked_at_target)",
            f"Infant age range: {min_age}-{max_age} months",
            f"Coefficient (per month): {coef:.3f}",
            f"p-value: {pvalue:.4f}",
            "",
            result.summary().as_text(),
        ]
    )
    return {"coef": coef, "pvalue": pvalue, "intercept": intercept}, report


def _assign_cohort(age: float, cohorts: List[Dict]) -> str | None:
    for cohort in cohorts:
        if cohort["min_months"] <= age <= cohort["max_months"]:
            return cohort["label"]
    return None


def _find_adult_label(cohorts: List[Dict]) -> str:
    for cohort in cohorts:
        if "adult" in cohort["label"].lower():
            return cohort["label"]
    return cohorts[-1]["label"]

