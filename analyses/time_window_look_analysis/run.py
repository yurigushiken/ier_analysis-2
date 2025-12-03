"""CLI runner for the reaction-look analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

if __package__ in (None, ""):
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from analyses.time_window_look_analysis import calculator, stats, visuals
else:
    from . import calculator, stats, visuals


def run_analysis(config_path: Path) -> None:
    """Execute time-window look analysis."""
    config = _load_config(config_path)
    config_name = config_path.stem
    output_dir = _determine_output_root(config, config_path)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    for folder in (tables_dir, figures_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)

    fixations = _load_fixations(config)
    target_aoi = config["target_aoi"]
    reaction_df = calculator.compute_reaction_flags(
        fixations,
        target_aoi=target_aoi,
        window_start=int(config["window_start"]),
        window_end=int(config["window_end"]),
        condition_codes=config.get("condition_codes") or [],
    )
    summary = calculator.summarize_by_cohort(reaction_df, cohorts=config["cohorts"])
    summary.to_csv(tables_dir / f"{config_name}_time_window_summary.csv", index=False)

    infant_cohorts = [c for c in config["cohorts"] if c["max_months"] <= 11]
    gee_results, gee_report = stats.run_adult_reference_gee(reaction_df, cohorts=config["cohorts"])
    trend_stats, trend_report = stats.run_linear_trend(
        reaction_df,
        infant_cohorts=infant_cohorts,
    )
    report_path = reports_dir / f"{config_name}_time_window_stats.txt"
    report_path.write_text("\n\n".join([gee_report, trend_report]), encoding="utf-8")

    condition_code = config.get("condition_codes", [""])[0].lower()
    condition_label = _friendly_condition_name(condition_code)
    window_desc = f"frame {config['window_start']} to frame {config['window_end']}"
    target_label = _format_target_label(target_aoi)
    bar_title = f'"{condition_label}" – Probability of fixation on {target_label}\n' f"during the period {window_desc}"
    visuals.plot_time_window_bar(
        summary,
        figure_path=figures_dir / f"{config_name}_time_window_bar_plot.png",
        title=bar_title,
        cohort_order=[c["label"] for c in config["cohorts"]],
        gee_results=gee_results,
        reference=_find_adult_label(config["cohorts"]),
    )
    forest_title = (
        f'"{condition_label}" – Probability of fixation on {target_label}\n'
        f"during the period {window_desc} (odds ratios vs Adults)"
    )
    visuals.plot_time_window_forest(
        gee_results,
        cohort_order=[c["label"] for c in config["cohorts"]],
        figure_path=figures_dir / f"{config_name}_time_window_forest_plot.png",
        title=forest_title,
    )
    # Linear trend output removed per instructions.
    if trend_stats:
        trend_report_path = reports_dir / f"{config_name}_time_window_linear_trend.txt"
        trend_report_path.write_text(trend_report, encoding="utf-8")
        linear_title = (
            f'"{condition_label}" – Probability of fixation on {target_label}\n'
            "Linear trend across infant cohorts"
        )
        visuals.plot_time_window_linear_trend(
            summary,
            figure_path=figures_dir / f"{config_name}_time_window_linear_trend.png",
            trend_stats=trend_stats,
            infant_labels=[c["label"] for c in infant_cohorts],
            adult_label=_find_adult_label(config["cohorts"]),
            title=linear_title,
        )


def _load_fixations(config: Dict) -> pd.DataFrame:
    path = Path(config["input_fixations"]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fixation file not found: {path}")
    return pd.read_csv(path)


def _determine_output_root(config: Dict, config_path: Path) -> Path:
    configured = config.get("output_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    return (config_path.parent / config_path.stem).resolve()


def _find_adult_label(cohorts: List[Dict]) -> str:
    for cohort in cohorts:
        if "adult" in cohort["label"].lower():
            return cohort["label"]
    return cohorts[-1]["label"]


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _friendly_condition_name(code: str) -> str:
    mapping = {
        "sw": "Show with toy",
        "gwo": "Give without toy",
        "gw": "Give with toy",
    }
    return mapping.get(code.lower(), code.upper())


def _format_target_label(target: str | List[str]) -> str:
    if isinstance(target, list):
        return ", ".join(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-window look analysis.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    args = parser.parse_args()
    run_analysis(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()

