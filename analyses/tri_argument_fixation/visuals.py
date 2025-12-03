"""Plotting utilities for the tri-argument analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas.api.types import CategoricalDtype

from .pipeline import EVENT_CATEGORY_ORDER, DYAD_CATEGORIES, MONAD_CATEGORIES, FULL_CATEGORY
from .stats import format_significance

DEFAULT_DPI = 300

CATEGORY_COLORS = {
    "Toy_Only": "#fee8c8",
    "Man_Only": "#fdd49e",
    "Woman_Only": "#fdbb84",
    "Other": "#e34a33",
    "Man_Toy": "#91bfdb",
    "Woman_Toy": "#74add1",
    "Woman_Man": "#4575b4",
    "Full_Trifecta": "#1a9850",
}


def plot_success(
    summary: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    stats_summary: Optional[pd.DataFrame] = None,
    reference_label: Optional[str] = None,
) -> None:
    """Render the cohort-level success chart with optional significance bars."""
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    x_pos = range(len(summary))
    ax.bar(x_pos, summary["success_rate"] * 100, color="#4C72B0")
    ax.set_ylabel("Tri-argument coverage (%)")
    ax.set_xlabel("Cohort")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(summary["cohort"], rotation=30, ha="right")
    for idx, rate in enumerate(summary["success_rate"]):
        ax.text(idx, rate * 100 + 1, f"{rate * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    if stats_summary is not None and reference_label is not None:
        _annotate_significance(ax, summary, stats_summary, reference_label)

    # Footnote removed per latest requirements (legend space no longer needed)
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close()


def plot_forest(
    stats_summary: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    reference_label: Optional[str] = None,
) -> None:
    """Render the odds-ratio forest plot."""
    plot_df = stats_summary.copy()
    plot_df["odds_ratio"] = np.exp(plot_df["coef"])
    plot_df["ci_low_or"] = np.exp(plot_df["ci_low"])
    plot_df["ci_high_or"] = np.exp(plot_df["ci_high"])

    y_pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)

    or_values = plot_df["odds_ratio"]
    err_low = or_values - plot_df["ci_low_or"]
    err_high = plot_df["ci_high_or"] - or_values

    ax.errorbar(or_values, y_pos, xerr=[err_low, err_high], fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["cohort"])
    ax.set_xlabel("Odds ratio vs reference")
    ax.set_title("\n".join(_wrap_title(title, width=45)))
    finite_high = plot_df["ci_high_or"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_low = plot_df["ci_low_or"].replace([np.inf, -np.inf], np.nan).dropna()
    max_high = finite_high.max() if not finite_high.empty else 3
    min_low = finite_low.min() if not finite_low.empty else 0.3
    span = max_high - min_low
    left = max(0, min_low - 0.2 * span)
    right = max(max_high + 0.2 * span, left + 1)
    ax.set_xlim(left=left, right=right)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    if reference_label:
        ax.text(0.02, 0.98, f"Reference: {reference_label}", transform=ax.transAxes, va="top", fontsize=9)
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.22, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close()


def plot_trials_per_participant(
    trial_results: pd.DataFrame,
    figure_path: Path,
    *,
    cohorts: List[Dict[str, object]],
) -> None:
    """Render bar chart summarizing valid trial counts."""
    if trial_results.empty:
        return

    cohort_order = [c["label"] for c in cohorts]
    counts = (
        trial_results.groupby("participant_id")
        .agg(trial_count=("trial_number", "nunique"), cohort=("cohort", lambda x: x.iloc[0]))
        .reset_index()
    )
    counts["cohort"] = pd.Categorical(counts["cohort"], categories=cohort_order, ordered=True)
    counts = counts.sort_values(["cohort", "participant_id"])

    plt.figure(figsize=(max(8, len(counts) * 0.35), 4))
    colors = plt.cm.tab20.colors
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(cohort_order)}
    bar_colors = [color_map.get(cohort, "#4C72B0") for cohort in counts["cohort"]]

    ax = plt.gca()
    ax.bar(range(len(counts)), counts["trial_count"], color=bar_colors)
    ax.set_ylabel("Valid trials")
    ax.set_xlabel("Participant")
    ax.set_title("Trials contributed per participant")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts["participant_id"], rotation=60, ha="right", fontsize=8)
    legend_handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=6, label=label)
        for label in cohort_order
        if label in counts["cohort"].values
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close()


def plot_event_structure_breakdown(summary: pd.DataFrame, figure_path: Path, *, title: str) -> None:
    """Render a 100% stacked bar showing event-structure categories by cohort."""
    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No trials to summarize", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    pivot = summary.pivot(index="cohort", columns="event_category", values="percentage").fillna(0.0)
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot = pivot.droplevel(0, axis=1)
    pivot = pivot.reindex(columns=EVENT_CATEGORY_ORDER, fill_value=0.0)
    if isinstance(summary["cohort"].dtype, CategoricalDtype):
        ordered_cohorts = list(summary["cohort"].cat.categories)
    else:
        ordered_cohorts = list(dict.fromkeys(summary["cohort"]))
    pivot = pivot.reindex(ordered_cohorts, fill_value=0.0)
    cohorts = pivot.index.tolist()
    positions = np.arange(len(cohorts))
    bottom = np.zeros(len(cohorts))

    fig, ax = plt.subplots(figsize=(max(8, len(cohorts) * 1.2), 5))
    for category in EVENT_CATEGORY_ORDER:
        values = pivot[category].to_numpy() if category in pivot else np.zeros(len(cohorts))
        color = CATEGORY_COLORS.get(category, "#bbbbbb")
        ax.bar(positions, values, bottom=bottom, color=color, label=category.replace("_", " "))
        bottom += values

    ax.set_xticks(positions)
    ax.set_xticklabels(cohorts, rotation=30, ha="right")
    ax.set_ylabel("Proportion of trials (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    legend_categories = list(reversed(EVENT_CATEGORY_ORDER))
    legend_handles = []
    for category in legend_categories:
        if category not in pivot.columns:
            continue
        label = _friendly_event_label(category)
        legend_handles.append(
            Patch(facecolor=CATEGORY_COLORS.get(category, "#bbbbbb"), label=label)
        )
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close()


def plot_latency_to_trifecta(
    summary: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    cohort_order: Sequence[str],
    trend_stats: Optional[Dict[str, float]] = None,
) -> None:
    """Plot processing-efficiency (latency) means per cohort."""
    if summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No successful trifecta trials to summarize", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    working = summary.set_index("cohort").reindex(cohort_order).reset_index()
    working = working.dropna(subset=["mean_latency_frames"])
    x = np.arange(len(working))
    y = working["mean_latency_frames"]

    fig, ax = plt.subplots(figsize=(max(7, len(working) * 1.1), 4.5))
    ax.plot(x, y, marker="o", color="#C44E52", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(working["cohort"], rotation=30, ha="right")
    ax.set_ylabel("Latency to trifecta (frames)")
    ax.set_xlabel("Cohort")
    ax.set_ylim(0, max(10, y.max() * 1.2))
    ax.set_title(title)
    for idx, value in enumerate(y):
        ax.text(x[idx], value + 1, f"{value:.1f}f", ha="center", va="bottom", fontsize=9)

    if trend_stats and trend_stats.get("pvalue") is not None:
        pvalue = trend_stats["pvalue"]
        ax.text(
            0.02,
            0.92,
            f"Linear trend p={pvalue:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_trifecta_linear_trend(
    summary: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    infant_labels: Sequence[str],
    trend_stats: Optional[Dict[str, float]],
    adult_label: Optional[str],
) -> None:
    """Render developmental trend plot for trifecta success."""
    if summary.empty or not infant_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No infant cohorts to plot", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    working = summary[summary["cohort"].isin(infant_labels)].copy()
    if working.empty or not trend_stats:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Linear trend unavailable", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    x = working["cohort"].str.extract(r"(\d+)").astype(float)[0].to_numpy()
    y = (working["success_rate"].to_numpy()) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#1f77b4", label="Infant cohorts")

    slope = float(trend_stats.get("coef", 0.0))
    intercept = float(trend_stats.get("intercept", 0.0))
    x_line = np.linspace(x.min(), x.max(), 200) if len(x) > 1 else np.array([x.min(), x.min() + 1])
    y_pred = (1.0 / (1.0 + np.exp(-(intercept + slope * x_line)))) * 100
    ax.plot(x_line, y_pred, color="#ff7f0e", label="Logistic fit")

    ticks = list(x)
    tick_labels = [str(int(val)) for val in x]
    adult_values = []
    if adult_label:
        adult_value = summary.loc[summary["cohort"] == adult_label, "success_rate"]
        if not adult_value.empty:
            adult_x = (max(x) if len(x) else 11) + 1
            adult_y = adult_value.to_numpy() * 100
            ax.scatter([adult_x], adult_y, color="#f4a261", marker="s", label=adult_label)
            ticks.append(adult_x)
            tick_labels.append(adult_label)
            adult_values.extend(adult_y.tolist())

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Trifecta success (%)")
    ax.set_xlabel("Cohort")
    combined_y = list(y) + adult_values
    y_max = max(combined_y) if combined_y else max(y)
    y_max = max(y_max, np.max(y_pred))
    ax.set_ylim(0, min(100, max(40, y_max + 10)))
    ax.set_title(title)
    pvalue = trend_stats.get("pvalue")
    if pvalue is not None and not np.isnan(pvalue):
        ax.text(
            0.02,
            0.92,
            f"slope={slope:.3f}, p={pvalue:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2))
    plt.subplots_adjust(top=0.84, bottom=0.32, left=0.12, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _annotate_significance(
    ax, summary: pd.DataFrame, stats_summary: pd.DataFrame, reference_label: str
) -> None:
    bars = summary["success_rate"] * 100
    idx_map = {cohort: i for i, cohort in enumerate(summary["cohort"])}
    ref_idx = idx_map.get(reference_label)
    if ref_idx is None:
        return
    bump_tracker: Dict[tuple, float] = {}
    for _, row in stats_summary.iterrows():
        cohort = row["cohort"]
        if cohort == reference_label:
            continue
        label = format_significance(row["pvalue"])
        other_idx = idx_map.get(cohort)
        if not label or other_idx is None:
            continue
        base_height = max(bars.iloc[ref_idx], bars.iloc[other_idx])
        key = (min(ref_idx, other_idx), max(ref_idx, other_idx))
        bump = bump_tracker.get(key, 0.0)
        y = base_height + 5 + bump
        ax.plot([ref_idx, ref_idx, other_idx, other_idx], [y, y + 2, y + 2, y], color="black", linewidth=1)
        ax.text((ref_idx + other_idx) / 2, y + 2.5, label, ha="center", va="bottom", fontsize=10)
        bump_tracker[key] = bump + 6


def _wrap_title(title: str, width: int = 50) -> List[str]:
    words = title.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if next_len > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def _friendly_event_label(category: str) -> str:
    if category == "Full_Trifecta":
        return "Trifecta"
    return category.replace("_", " ")

