"""Tri-argument fixation analysis runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

CONDITION_LABELS = {
    "gw": "Give with toy",
    "gwo": "Give (no toy)",
    "sw": "Show",
    "swo": "Show (no toy)",
    "hw": "Hug",
    "hwo": "Hug (no toy)",
    "ugw": "Upside-down give",
    "ugwo": "Upside-down give (no toy)",
    "uhw": "Upside-down hug",
    "uhwo": "Upside-down hug (no toy)",
    "f": "Floating",
}


def _condition_label(config: Dict) -> str:
    codes = config.get("condition_codes") or []
    first = (codes[0] if codes else "").lower()
    return CONDITION_LABELS.get(first, first.upper() or "Condition")
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from analyses.tri_argument_fixation import latency_analysis, pipeline, reports, stats, visuals
else:
    from . import latency_analysis, pipeline, reports, stats, visuals


def run_analysis(config_path: Path) -> None:
    """Entry point for the tri-argument fixation analysis."""
    config = _load_config(config_path)
    output_root = pipeline.determine_output_root(config, config_path)
    config_name = config_path.stem
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    for directory in (reports_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    df = pipeline.load_fixations(config)
    if df.empty:
        raise ValueError("No gaze fixation data available for the configured conditions.")

    trial_results = pipeline.compute_trial_metrics(
        df,
        aoi_groups=config["aoi_groups"],
        condition_codes=config["condition_codes"],
        frame_window=config.get("frame_window"),
    )
    trial_results = pipeline.filter_by_min_trials(
        trial_results, min_trials=config.get("min_trials_per_participant", 1)
    )
    trial_results = pipeline.attach_cohorts(trial_results, config["cohorts"])
    if trial_results.empty:
        raise ValueError("All trials were filtered out after cohort assignment.")

    summary = pipeline.summarize_by_cohort(trial_results, config["cohorts"])
    summary_path = tables_dir / f"{config_name}_tri_argument_summary.csv"
    summary.to_csv(summary_path, index=False)

    figure_path = figures_dir / f"{config_name}_tri_argument_success.png"
    condition_label = _condition_label(config)
    display_label = f"\"{condition_label}\""
    base_title = f"{display_label} - Tri-argument fixation coverage by cohort"

    stats_summary = stats.run_gee_analysis(trial_results, reports_dir, config, filename_prefix=config_name)
    visuals.plot_success(
        summary,
        figure_path,
        title=base_title,
        stats_summary=stats_summary,
        reference_label=(config.get("gee", {}) or {}).get("reference_cohort"),
    )

    if stats_summary is not None:
        forest_path = figures_dir / f"{config_name}_forest_plot_odds_ratios.png"
        visuals.plot_forest(
            stats_summary,
            forest_path,
            title=f"{display_label} - Odds ratios vs {(config.get('gee', {}) or {}).get('reference_cohort', 'reference cohort')}",
            reference_label=(config.get("gee", {}) or {}).get("reference_cohort"),
        )

    trials_fig_path = figures_dir / f"{config_name}_trials_per_participant.png"
    visuals.plot_trials_per_participant(
        trial_results,
        trials_fig_path,
        cohorts=config["cohorts"],
    )

    report_config = config.get("report", {})
    reports.write_text_report(summary, report_config, reports_dir, filename_prefix=config_name)
    reports.write_html_report(
        summary,
        report_config,
        reports_dir,
        figure_path.relative_to(output_root),
        filename_prefix=config_name,
    )
    reports.write_pdf_report(summary, report_config, reports_dir, figure_path, filename_prefix=config_name)

    events = pipeline.classify_event_structure(
        df,
        trial_results,
        aoi_groups=config["aoi_groups"],
        condition_codes=config["condition_codes"],
        frame_window=config.get("frame_window"),
    )
    event_summary = pipeline.summarize_event_structure(events, config["cohorts"])
    reports.write_event_structure_csv(event_summary, tables_dir, filename_prefix=config_name)
    visuals.plot_event_structure_breakdown(
        event_summary,
        figures_dir / f"{config_name}_event_structure_breakdown.png",
        title=f"{display_label} - Event structure coverage",
    )

    infant_cohorts = [cohort for cohort in config["cohorts"] if cohort["max_months"] <= 11]
    infant_labels = [c["label"] for c in infant_cohorts]
    adult_label = next((c["label"] for c in config["cohorts"] if "adult" in c["label"].lower()), None)

    trend_stats_success, trend_report_success = stats.run_success_linear_trend(
        trial_results,
        infant_cohorts=infant_cohorts,
    )
    reports_dir.joinpath(f"{config_name}_tri_argument_linear_trend.txt").write_text(
        trend_report_success,
        encoding="utf-8",
    )
    trend_columns = ["coef", "intercept", "pvalue", "age_min", "age_max", "n_participants", "n_trials"]
    trend_summary_path = tables_dir / f"{config_name}_tri_argument_linear_trend_summary.csv"
    if trend_stats_success:
        pd.DataFrame([trend_stats_success])[trend_columns].to_csv(trend_summary_path, index=False)
    else:
        pd.DataFrame(columns=trend_columns).to_csv(trend_summary_path, index=False)
    visuals.plot_trifecta_linear_trend(
        summary,
        figures_dir / f"{config_name}_tri_argument_linear_trend.png",
        title=f"{display_label} - Trifecta success linear trend",
        infant_labels=infant_labels,
        trend_stats=trend_stats_success if trend_stats_success else None,
        adult_label=adult_label,
    )

    latency_metrics = latency_analysis.compute_latency_metrics(
        df,
        trial_results,
        aoi_groups=config["aoi_groups"],
        condition_codes=config["condition_codes"],
        frame_window=config.get("frame_window"),
    )
    latency_summary = latency_analysis.summarize_latency_by_cohort(
        latency_metrics, config["cohorts"]
    )
    latency_summary.to_csv(tables_dir / f"{config_name}_latency_summary.csv", index=False)

    trend_stats, trend_report = latency_analysis.run_latency_trend(
        latency_metrics,
        infant_cohorts=infant_cohorts,
    )
    (reports_dir / f"{config_name}_latency_trend.txt").write_text(trend_report, encoding="utf-8")
    visuals.plot_latency_to_trifecta(
        latency_summary,
        figures_dir / f"{config_name}_latency_to_trifecta.png",
        title=f"{display_label} - Processing efficiency (latency to trifecta)",
        cohort_order=[c["label"] for c in config["cohorts"]],
        trend_stats=trend_stats,
    )


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config




def main() -> None:
    parser = argparse.ArgumentParser(description="Run tri-argument fixation analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/tri_argument_fixation/config.yaml"),
        help="Path to analysis configuration YAML file.",
    )
    args = parser.parse_args()
    run_analysis(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()
