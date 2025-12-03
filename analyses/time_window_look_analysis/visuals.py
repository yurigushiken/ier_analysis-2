"""Plotting helpers for reaction look analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def plot_time_window_bar(
    summary_df: pd.DataFrame,
    *,
    figure_path: Path,
    title: str,
    cohort_order: Sequence[str],
    gee_results: pd.DataFrame | None = None,
    reference: str | None = None,
) -> None:
    """Render bar chart."""
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No reaction data available", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        return

    ordered = sorted(cohort_order, key=lambda label: (0 if "adult" in label.lower() else 1, cohort_order.index(label)))
    working = summary_df.set_index("cohort").reindex(ordered).reset_index()
    positions = np.arange(len(working))
    values = working["mean_looked"] * 100

    fig, ax = plt.subplots(figsize=(max(7, len(working) * 1.05), 6))
    colors = ["#f4a261" if label == reference else "#4C72B0" for label in working["cohort"]]
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(working["cohort"], rotation=30, ha="right")
    ax.set_ylabel("Looked at target (%)")
    ax.set_xlabel("Cohort")
    ax.set_ylim(0, max(10, values.max() * 1.2))
    ax.set_title(title, pad=20)
    for idx, value in enumerate(values):
        ax.text(positions[idx], value + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    if gee_results is not None:
        idx_map = {label: idx for idx, label in enumerate(working["cohort"])}
        for row in gee_results.itertuples():
            if np.isnan(row.pvalue) or row.cohort == reference or row.pvalue >= 0.1:
                continue
            idx = idx_map.get(row.cohort)
            if idx is None:
                continue
            y = values[idx] + 0.05 * max(10, values.max())
            label = "***" if row.pvalue < 0.001 else "**" if row.pvalue < 0.01 else "*" if row.pvalue < 0.05 else f"p={row.pvalue:.2f}"
            ax.text(positions[idx], y, label, ha="center", va="bottom", fontsize=11)

    plt.subplots_adjust(top=0.78, bottom=0.22, left=0.12, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_time_window_forest(
    gee_results: pd.DataFrame,
    *,
    cohort_order: Sequence[str],
    figure_path: Path,
    title: str,
) -> None:
    """Forest plot for odds ratios vs adults."""
    if gee_results is None or gee_results.empty:
        return
    working = gee_results.set_index("cohort").reindex(cohort_order).reset_index()
    or_values = working["odds_ratio"].fillna(1.0).to_numpy()
    ci_low_series = working["ci_low_or"].replace([np.inf, -np.inf], np.nan).fillna(working["odds_ratio"])
    ci_high_series = working["ci_high_or"].replace([np.inf, -np.inf], np.nan).fillna(working["odds_ratio"])
    ci_low = ci_low_series.to_numpy()
    ci_high = ci_high_series.to_numpy()
    y_pos = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(6, max(4, len(working) * 0.6)))
    ax.axvline(1.0, color="gray", linestyle="--")
    ax.errorbar(
        or_values,
        y_pos,
        xerr=[or_values - ci_low, ci_high - or_values],
        fmt="o",
        color="#C44E52",
        ecolor="#C44E52",
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(working["cohort"])
    ax.set_xlabel("Odds ratio vs Adults")
    finite_high = ci_high[np.isfinite(ci_high)]
    finite_low = ci_low[np.isfinite(ci_low)]
    max_high = finite_high.max() if finite_high.size else 3
    min_low = finite_low.min() if finite_low.size else 0.3
    span = max_high - min_low
    left_bound = max(0, min_low - 0.2 * span)
    right_bound = max(max_high + 0.2 * span, left_bound + 1)
    ax.set_xlim(left=left_bound, right=right_bound)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_title(title)
    if "pvalue" in working.columns:
        for idx, row in enumerate(gee_results.itertuples()):
            if np.isnan(row.pvalue) or row.pvalue >= 0.05:
                continue
            label = "***" if row.pvalue < 0.001 else "**" if row.pvalue < 0.01 else "*"
            ax.text(or_values[idx], y_pos[idx] + 0.1, label, ha="center", va="bottom", fontsize=12)
    plt.subplots_adjust(top=0.78, bottom=0.12, left=0.2, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

def plot_time_window_linear_trend(
    summary_df: pd.DataFrame,
    *,
    figure_path: Path,
    trend_stats: Dict[str, float] | None,
    infant_labels: Sequence[str],
    adult_label: str | None,
    title: str,
) -> None:
    """Render developmental trend plot with infant scatter, logistic fit, and adult marker."""
    if summary_df.empty or not infant_labels or not trend_stats:
        return
    working = summary_df[summary_df["cohort"].isin(infant_labels)].copy()
    if working.empty:
        return
    working["age_numeric"] = (
        working["cohort"].str.extract(r"(\d+(?:\.\d+)?)").astype(float)[0]
    )
    working = working.dropna(subset=["age_numeric"])
    if working.empty:
        return
    x = working["age_numeric"].to_numpy()
    y = working["mean_looked"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#1f77b4", label="Infant cohorts", zorder=3)

    slope = float(trend_stats.get("coef", 0.0))
    intercept = float(trend_stats.get("intercept", 0.0))
    x_line = np.linspace(x.min(), x.max(), 200)
    logits = intercept + slope * x_line
    y_pred = (1.0 / (1.0 + np.exp(-logits))) * 100
    ax.plot(x_line, y_pred, color="#ff7f0e", label="Logistic fit", linewidth=2)

    ticks = list(x)
    tick_labels = [str(int(val)) for val in x]
    adult_values: list[float] = []
    if adult_label:
        adult_row = summary_df[summary_df["cohort"] == adult_label]
        if not adult_row.empty:
            adult_y = adult_row["mean_looked"].iloc[0] * 100
            adult_x = (max(x) if len(x) else 11) + 1
            ax.scatter([adult_x], [adult_y], color="#f4a261", marker="s", s=90, label=adult_label)
            ticks.append(adult_x)
            tick_labels.append(adult_label)
            adult_values.append(adult_y)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Probability of fixation (%)")
    ax.set_xlabel("Cohort")
    combined_values = list(y_pred) + list(y) + adult_values
    y_max = max(combined_values) if combined_values else 0
    ax.set_ylim(0, min(100, max(40, y_max + 10)))
    ax.set_title(title, pad=20)
    pvalue = trend_stats.get("pvalue")
    if pvalue is not None and not np.isnan(pvalue):
        ax.text(
            0.02,
            0.92,
            f"Trend p={_format_trend_pvalue(pvalue)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1)
    plt.subplots_adjust(top=0.84, bottom=0.32, left=0.12, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def _format_trend_pvalue(pvalue: float) -> str:
    if pvalue < 1e-4:
        return "<0.0001"
    if pvalue < 1e-3:
        return "<0.001"
    return f"={pvalue:.3f}"

