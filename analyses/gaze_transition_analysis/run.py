"""CLI entry point for the gaze transition analysis."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

if __package__ in (None, ""):
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from analyses.gaze_transition_analysis import loader, matrix, transitions, visuals, strategy
else:
    from . import loader, matrix, transitions, visuals, strategy


def run_analysis(config_path: Path) -> None:
    config = _load_config(config_path)
    config_name = config_path.stem
    output_root = _determine_output_root(config, config_path)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    for folder in (tables_dir, figures_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)

    condition_codes = config.get("condition_codes") or ["gw"]
    aoi_nodes = config["aoi_nodes"]
    cohorts = config["cohorts"]

    fixations = loader.load_fixations(
        Path(config.get("input_fixations")) if config.get("input_fixations") else None,
        condition_codes=condition_codes,
    )
    transitions_df = transitions.compute_transitions(fixations, aoi_nodes=aoi_nodes)
    wide_counts = transitions.to_wide_counts(transitions_df, aoi_nodes=aoi_nodes)
    wide_counts.to_csv(tables_dir / f"{config_path.stem}_transition_counts.csv", index=False)

    matrix_df = matrix.build_transition_matrix(
        transitions_df,
        cohorts=cohorts,
        aoi_nodes=aoi_nodes,
    )
    matrix_df.to_csv(tables_dir / f"{config_path.stem}_transition_matrix.csv", index=False)

    visuals.plot_heatmap(
        matrix_df,
        aoi_nodes=aoi_nodes,
        cohorts=[c["label"] for c in cohorts],
        figure_path=figures_dir / f"{config_name}_transition_heatmap.png",
        title=f"\"{_condition_label(condition_codes)}\" - AOI transition frequencies",
        subtitle="Columns represent From→To AOI steps (excluding off-screen)",
    )
    visuals.plot_transition_networks(
        matrix_df,
        cohorts=[c["label"] for c in cohorts],
        aoi_nodes=aoi_nodes,
        figures_dir=figures_dir,
        filename_prefix=config_name,
    )

    strategy_df = strategy.compute_strategy_proportions(transitions_df)
    strategy_df.to_csv(tables_dir / f"{config_path.stem}_strategy_proportions.csv", index=False)
    strategy_summary = strategy.build_strategy_summary(strategy_df, cohorts=cohorts)
    strategy_summary.to_csv(tables_dir / f"{config_name}_strategy_summary.csv", index=False)
    descriptive_stats = strategy.build_strategy_descriptive_stats(strategy_df, cohorts=cohorts)
    descriptive_stats.to_csv(
        tables_dir / f"{config_name}_strategy_descriptive_stats.csv", index=False
    )
    _write_descriptive_summary(
        strategy_summary,
        reports_dir / f"{config_name}_strategy_descriptive_stats.txt",
    )

    cohort_labels = [c["label"] for c in cohorts]
    infant_cohorts = [c for c in cohorts if c["max_months"] <= 11]
    reference_label = cohort_labels[0]
    adult_label = next((label for label in cohort_labels if "adult" in label.lower()), None)
    display_order = [reference_label] + [label for label in cohort_labels if label != reference_label]
    condition_label = _condition_label(condition_codes)
    arrow = "↔"
    social_caption = f"Fixations man_face {arrow} woman_face ('agent-agent attention')"
    mechanical_caption = f"Fixations toy {arrow} body ('motion tracking')"
    object_caption = f"Fixations face {arrow} toy ('agent-object binding')"

    social_gee, social_gee_report = strategy.run_strategy_gee(
        strategy_df,
        cohorts=cohorts,
        value_column=strategy.AGENT_AGENT_ATTENTION_PCT,
        metric_label="Agent-Agent Attention",
    )
    social_annotations = strategy.build_significance_annotations(
        social_gee,
        reference=reference_label,
        cohort_order=display_order,
    )
    social_trend, social_trend_report = strategy.run_linear_trend_test(
        strategy_df,
        infant_cohorts=infant_cohorts,
        value_column=strategy.AGENT_AGENT_ATTENTION_PCT,
        metric_label="Agent-Agent Attention",
    )
    _write_stats_report(
        reports_dir / f"{config_name}_stats_agent_agent_attention.txt",
        gee_text=social_gee_report,
        trend_text=social_trend_report,
    )
    visuals.plot_single_strategy_bars(
        strategy_summary,
        value_column=strategy.AGENT_AGENT_ATTENTION_MEAN,
        label="Agent-Agent Attention",
        figure_path=figures_dir / f"{config_name}_agent_agent_attention_strategy.png",
        title="\n".join(
            [
                condition_label,
                social_caption,
                "Strategy prevalence by cohort",
            ]
        ),
        cohorts_order=display_order,
        color="#1f77b4",
        annotations=social_annotations,
        reference_label=reference_label,
    )
    visuals.plot_linear_trend(
        strategy_summary,
        social_trend if social_trend else None,
        figure_path=figures_dir / f"{config_name}_linear_trend_agent_agent_attention.png",
        value_column=strategy.AGENT_AGENT_ATTENTION_MEAN,
        label="Agent-Agent Attention",
        title="\n".join([condition_label, social_caption, "Linear trend across infant cohorts"]),
        y_axis_label=f"man_face {arrow} woman_face",
        adult_label=adult_label,
    )

    mechanical_gee, mechanical_gee_report = strategy.run_strategy_gee(
        strategy_df,
        cohorts=cohorts,
        value_column=strategy.MOTION_TRACKING_PCT,
        metric_label="Motion Tracking",
    )
    mechanical_annotations = strategy.build_significance_annotations(
        mechanical_gee,
        reference=reference_label,
        cohort_order=display_order,
    )
    mechanical_trend, mechanical_trend_report = strategy.run_linear_trend_test(
        strategy_df,
        infant_cohorts=infant_cohorts,
        value_column=strategy.MOTION_TRACKING_PCT,
        metric_label="Motion Tracking",
    )
    _write_stats_report(
        reports_dir / f"{config_name}_stats_motion_tracking.txt",
        gee_text=mechanical_gee_report,
        trend_text=mechanical_trend_report,
    )
    visuals.plot_single_strategy_bars(
        strategy_summary,
        value_column=strategy.MOTION_TRACKING_MEAN,
        label="Motion Tracking",
        figure_path=figures_dir / f"{config_name}_motion_tracking_strategy.png",
        title="\n".join(
            [
                condition_label,
                mechanical_caption,
                "Strategy prevalence by cohort",
            ]
        ),
        cohorts_order=display_order,
        color="#2ca02c",
        annotations=mechanical_annotations,
        reference_label=reference_label,
    )
    visuals.plot_linear_trend(
        strategy_summary,
        mechanical_trend if mechanical_trend else None,
        figure_path=figures_dir / f"{config_name}_linear_trend_motion_tracking.png",
        value_column=strategy.MOTION_TRACKING_MEAN,
        label="Motion Tracking",
        title="\n".join([condition_label, mechanical_caption, "Linear trend across infant cohorts"]),
        y_axis_label=f"toy {arrow} body",
        adult_label=adult_label,
    )

    object_gee, object_gee_report = strategy.run_strategy_gee(
        strategy_df,
        cohorts=cohorts,
        value_column=strategy.AGENT_OBJECT_BINDING_PCT,
        metric_label="Agent-Object Binding",
    )
    object_annotations = strategy.build_significance_annotations(
        object_gee,
        reference=reference_label,
        cohort_order=display_order,
    )
    object_trend, object_trend_report = strategy.run_linear_trend_test(
        strategy_df,
        infant_cohorts=infant_cohorts,
        value_column=strategy.AGENT_OBJECT_BINDING_PCT,
        metric_label="Agent-Object Binding",
    )
    _write_stats_report(
        reports_dir / f"{config_name}_stats_agent_object_binding.txt",
        gee_text=object_gee_report,
        trend_text=object_trend_report,
    )
    visuals.plot_single_strategy_bars(
        strategy_summary,
        value_column=strategy.AGENT_OBJECT_BINDING_MEAN,
        label="Agent-Object Binding",
        figure_path=figures_dir / f"{config_name}_agent_object_binding_strategy.png",
        title="\n".join(
            [
                condition_label,
                object_caption,
                "Strategy prevalence by cohort",
            ]
        ),
        cohorts_order=display_order,
        color="#9467bd",
        annotations=object_annotations,
        reference_label=reference_label,
    )
    visuals.plot_linear_trend(
        strategy_summary,
        object_trend if object_trend else None,
        figure_path=figures_dir / f"{config_name}_linear_trend_agent_object_binding.png",
        value_column=strategy.AGENT_OBJECT_BINDING_MEAN,
        label="Agent-Object Binding",
        title="\n".join([condition_label, object_caption, "Linear trend across infant cohorts"]),
        y_axis_label=f"face {arrow} toy",
        adult_label=adult_label,
    )
    _write_text_report(
        transitions_df,
        matrix_df,
        reports_dir / f"{config_name}_transition_summary.txt",
        condition_codes,
    )


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


def _determine_output_root(config: Dict, config_path: Path) -> Path:
    configured = config.get("output_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    return (config_path.parent / config_path.stem).resolve()


def _condition_label(codes: List[str]) -> str:
    mapping = {
        "gw": "Give with Toy",
        "sw": "Show",
        "gwo": "Give (no toy)",
        "swo": "Show (no toy)",
        "ugw": "Upside-down Give with Toy",
        "ugwo": "Upside-down Give (no toy)",
    }
    first = codes[0].lower() if codes else ""
    return mapping.get(first, first.upper() or "Condition")


def _write_text_report(
    transitions_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    output_path: Path,
    condition_codes: List[str],
) -> None:
    total_transitions = int(transitions_df["count"].sum()) if not transitions_df.empty else 0
    lines = [
        f"Condition: {_condition_label(condition_codes)}",
        f"Total transitions counted: {total_transitions}",
        "",
        "Top cohort transition pairs:",
    ]
    if not matrix_df.empty:
        for cohort, cohort_df in matrix_df.groupby("cohort"):
            lines.append(f"{cohort}:")
            top = (
                cohort_df.sort_values("mean_count", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            for row in top.itertuples():
                lines.append(
                    f"  {row.from_aoi} -> {row.to_aoi}: mean {row.mean_count:.2f}"
                )
            lines.append("")
    else:
        lines.append("No transitions available.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_stats_report(output_path: Path, *, gee_text: str, trend_text: str) -> None:
    sections = []
    if gee_text:
        sections.append(gee_text.strip())
    if trend_text:
        sections.append(trend_text.strip())
    content = "\n\n".join(filter(None, sections)) or "No statistics available."
    output_path.write_text(content, encoding="utf-8")


def _write_descriptive_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    header = "Cohort,Agent_Agent_Attention,Agent_Object_Binding,Motion_Tracking"
    if summary_df.empty:
        output_path.write_text(f"{header}\nN/A,N/A,N/A,N/A\n", encoding="utf-8")
        return
    lines = [header]
    for row in summary_df.itertuples():
        lines.append(
            ",".join(
                [
                    str(row.cohort),
                    f"{getattr(row, strategy.AGENT_AGENT_ATTENTION_MEAN):.3f}",
                    f"{getattr(row, strategy.AGENT_OBJECT_BINDING_MEAN):.3f}",
                    f"{getattr(row, strategy.MOTION_TRACKING_MEAN):.3f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gaze transition analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analyses/gaze_transition_analysis/config.yaml"),
        help="YAML config path.",
    )
    args = parser.parse_args()
    run_analysis(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()

