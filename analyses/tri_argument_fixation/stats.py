"""Statistical helpers (GEE + significance formatting)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_gee_analysis(
    trial_results: pd.DataFrame,
    reports_dir: Path,
    config: Dict,
    *,
    filename_prefix: str,
) -> Optional[pd.DataFrame]:
    """Fit the configured GEE model and persist textual diagnostics."""
    gee_cfg = config.get("gee", {})
    if not gee_cfg.get("enabled"):
        return None

    cohorts: List[str] = [c["label"] for c in config.get("cohorts", [])]
    if not cohorts:
        raise ValueError("Cohorts must be defined to run GEE analysis.")

    reference = gee_cfg.get("reference_cohort", cohorts[0])
    cohorts = [reference] + [label for label in cohorts if label != reference]

    data = trial_results[trial_results["cohort"].isin(cohorts)].copy()
    if data.empty:
        raise ValueError("No data available for GEE after cohort filtering.")

    data["tri_argument_success"] = data["tri_argument_success"].astype(int)
    data["cohort"] = pd.Categorical(data["cohort"], categories=cohorts, ordered=True)

    formula = f"tri_argument_success ~ C(cohort, Treatment(reference='{reference}'))"
    model = smf.gee(formula, groups="participant_id", data=data, family=sm.families.Binomial())
    result = model.fit()

    stats_summary = _build_stats_summary(result, cohorts, reference)
    _write_gee_report(result, data, reports_dir, reference, filename_prefix=filename_prefix)

    return stats_summary


def format_significance(pvalue: Optional[float]) -> Optional[str]:
    """Return the asterisk label for the supplied p-value."""
    if pvalue is None:
        return None
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return None


def _build_stats_summary(result, cohorts: List[str], reference: str) -> pd.DataFrame:
    coef = result.params
    bse = result.bse
    z = coef / bse
    pvalues = result.pvalues
    ci = result.conf_int()

    stats_rows = [{"cohort": reference, "coef": 0.0, "pvalue": None, "ci_low": 0.0, "ci_high": 0.0}]
    for cohort in cohorts[1:]:
        term = f"C(cohort, Treatment(reference='{reference}'))[T.{cohort}]"
        if term in coef.index:
            stats_rows.append(
                {
                    "cohort": cohort,
                    "coef": coef[term],
                    "pvalue": pvalues[term],
                    "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1],
                }
            )
    return pd.DataFrame(stats_rows)


def _write_gee_report(result, data: pd.DataFrame, reports_dir: Path, reference: str, *, filename_prefix: str) -> None:
    coef = result.params
    bse = result.bse
    z = coef / bse
    pvalues = result.pvalues
    ci = result.conf_int()

    trials_per_participant = data.groupby("participant_id").size()
    success_count = int(data["tri_argument_success"].sum())
    failure_count = int(len(data) - success_count)

    qic_value = _extract_qic(result)
    if qic_value is not None:
        try:
            qic_display = f"{float(qic_value):.3f}"
        except Exception:
            qic_display = str(qic_value)
    else:
        qic_display = "unavailable"

    rows = [
        "GEE (Binomial, logit link) results",
        "Descriptive statistics:",
        f"  Trials per participant - min: {trials_per_participant.min()}, max: {trials_per_participant.max()}, mean: {trials_per_participant.mean():.2f}",
        f"  Class balance - successes: {success_count}, failures: {failure_count}",
        "",
        f"Observations: {len(data)}",
        f"Participants: {data['participant_id'].nunique()}",
        "",
        "Model diagnostics:",
        f"  QIC: {qic_display}",
        f"  Scale parameter: {result.scale:.3f}",
        f"  Covariance type: {getattr(result, 'cov_type', 'robust')}",
    ]
    cov_struct = getattr(result.model, "cov_struct", None)
    if cov_struct is not None and hasattr(cov_struct, "dep_params") and cov_struct.dep_params is not None:
        rows.append(f"  Working correlation parameter: {cov_struct.dep_params}")
    else:
        rows.append("  Working correlation parameter: not estimated (independence)")
    rows.append("")
    rows.append(f"Coefficient Summary (reference cohort: {reference})")
    header = f"{'Term':<30}{'Coef':>10}{'Std Err':>12}{'z':>10}{'P>|z|':>10}{'[0.025':>12}{'0.975]':>12}"
    rows.append(header)
    for term in coef.index:
        rows.append(
            f"{term:<30}{coef[term]:>10.4f}{bse[term]:>12.4f}{z[term]:>10.3f}{pvalues[term]:>10.4f}"
            f"{ci.loc[term, 0]:>12.4f}{ci.loc[term, 1]:>12.4f}"
        )

    report_path = reports_dir / f"{filename_prefix}_gee_results.txt"
    report_path.write_text("\n".join(rows), encoding="utf-8")


def _extract_qic(result) -> Optional[float]:
    qic_attr = getattr(result, "qic", None)
    qic_value = None
    if callable(qic_attr):
        try:
            qic_value = qic_attr()
        except Exception:
            qic_value = None
    elif isinstance(qic_attr, (int, float)):
        qic_value = qic_attr
    elif isinstance(qic_attr, (tuple, list, np.ndarray)) and len(qic_attr) > 0:
        qic_value = qic_attr[0]
        while isinstance(qic_value, (tuple, list, np.ndarray)) and len(qic_value) > 0:
            qic_value = qic_value[0]
    return qic_value


def run_success_linear_trend(
    trial_results: pd.DataFrame,
    *,
    infant_cohorts: List[Dict[str, int]],
) -> Tuple[Dict[str, float], str]:
    """Run Binomial GEE on infant cohorts only (success ~ age_numeric)."""
    if not infant_cohorts:
        return {}, "Tri-argument linear trend: no infant cohorts configured."

    cohort_mapping = {
        cohort["label"]: (cohort["min_months"] + cohort["max_months"]) / 2 for cohort in infant_cohorts
    }
    cohort_labels = list(cohort_mapping.keys())
    data = trial_results[trial_results["cohort"].isin(cohort_labels)].copy()
    if data.empty:
        return {}, "Tri-argument linear trend: no infant cohort trials available."

    data["age_numeric"] = data["cohort"].map(cohort_mapping)
    data = data.dropna(subset=["age_numeric"])
    if data.empty:
        return {}, "Tri-argument linear trend: missing numeric ages for infant cohorts."

    data["tri_argument_success"] = data["tri_argument_success"].astype(int)
    model = smf.gee(
        "tri_argument_success ~ age_numeric",
        groups="participant_id",
        data=data,
        family=sm.families.Binomial(),
    )
    try:
        result = model.fit()
    except ValueError as exc:
        return {}, f"Tri-argument linear trend failed to converge: {exc}"

    coef = float(result.params.get("age_numeric", 0.0))
    intercept = float(result.params.get("Intercept", 0.0))
    pvalue = float(result.pvalues.get("age_numeric", float("nan")))
    stats_summary = {
        "coef": coef,
        "intercept": intercept,
        "pvalue": pvalue,
        "age_min": float(min(cohort_mapping.values())),
        "age_max": float(max(cohort_mapping.values())),
        "n_participants": int(data["participant_id"].nunique()),
        "n_trials": int(len(data)),
    }
    summary_lines = [
        f"Tri-argument success linear trend (infants {int(stats_summary['age_min'])}-{int(stats_summary['age_max'])} months)",
        f"Participants: {stats_summary['n_participants']}",
        f"Trials: {stats_summary['n_trials']}",
        "",
        result.summary().as_text(),
    ]
    return stats_summary, "\n".join(summary_lines)

