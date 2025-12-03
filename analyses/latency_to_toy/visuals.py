"""Plotting utilities for latency-to-toy analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_latency_bar(
    summary_df: pd.DataFrame,
    *,
    figure_path: Path,
    title: str,
    cohort_order: Sequence[str],
    gee_results: pd.DataFrame,
    reference_label: str,
) -> None:
    """Render latency bar chart with adult reference annotations."""
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No latency data available", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        return

    ordered = sorted(cohort_order, key=lambda label: (0 if "adult" in label.lower() else 1, cohort_order.index(label)))
    working = summary_df.set_index("cohort").reindex(ordered).reset_index()
    working = working.dropna(subset=["mean_latency_frames"])
    positions = np.arange(len(working))
    values = working["mean_latency_frames"].to_numpy()

    colors = ["#f4a261" if cohort == reference_label else "#4C72B0" for cohort in working["cohort"]]
    fig, ax = plt.subplots(figsize=(max(7, len(working) * 1.1), 6))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(working["cohort"], rotation=30, ha="right")
    ax.set_ylabel("Mean latency to toy (frames)")
    ax.set_xlabel("Cohort")
    ax.set_ylim(0, max(10, values.max() * 1.4))
    ax.set_title(title, pad=20)
    for idx, value in enumerate(values):
        ax.text(positions[idx], value + 0.02 * max(10, values.max()), f"{value:.1f}f", ha="center", va="bottom", fontsize=10)

    idx_map = {cohort: idx for idx, cohort in enumerate(working["cohort"])}
    ref_idx = idx_map.get(reference_label)
    if gee_results is not None and ref_idx is not None:
        text_height = max(2, 0.05 * max(10, values.max()))
        for row in gee_results.itertuples():
            if np.isnan(row.pvalue) or row.cohort == reference_label or row.pvalue >= 0.05:
                continue
            other_idx = idx_map.get(row.cohort)
            if other_idx is None:
                continue
            y = values[other_idx] + text_height
            label = "***" if row.pvalue < 0.001 else "**" if row.pvalue < 0.01 else "*"
            ax.text(
                positions[other_idx],
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )
    plt.subplots_adjust(top=0.84, bottom=0.2, left=0.12, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_latency_forest(
    gee_results: pd.DataFrame,
    *,
    cohort_order: Sequence[str],
    reference_label: str,
    figure_path: Path,
    title: str,
) -> None:
    """Forest plot showing latency differences vs adults."""
    if gee_results is None or gee_results.empty:
        return
    working = gee_results.set_index("cohort").reindex(cohort_order).reset_index()
    y_pos = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(6, max(4, len(working) * 0.6)))
    ax.axvline(0.0, color="#555555", linestyle="--")
    effects = working["coef"].fillna(0.0).to_numpy()
    ci_low = working["ci_low"].fillna(0.0).to_numpy()
    ci_high = working["ci_high"].fillna(0.0).to_numpy()
    line_color = "#1f77b4"
    ax.errorbar(
        effects,
        y_pos,
        xerr=[effects - ci_low, ci_high - effects],
        fmt="o",
        color=line_color,
        ecolor=line_color,
        capsize=4,
        linewidth=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(working["cohort"])
    ax.set_xlabel("Latency difference vs Adults (frames)")
    ax.set_title(title)
    finite_high = ci_high[np.isfinite(ci_high)]
    finite_low = ci_low[np.isfinite(ci_low)]
    max_range = max(abs(finite_low.min() if finite_low.size else -1), abs(finite_high.max() if finite_high.size else 1), 1)
    ax.set_xlim(left=-max_range * 1.3, right=max_range * 1.3)
    for idx, row in enumerate(gee_results.itertuples()):
        if np.isnan(row.pvalue) or row.pvalue >= 0.05:
            continue
        label = "***" if row.pvalue < 0.001 else "**" if row.pvalue < 0.01 else "*"
        ax.text(effects[idx], y_pos[idx] + 0.1, label, ha="center", va="bottom", fontsize=12)
    plt.subplots_adjust(top=0.78, bottom=0.12, left=0.2, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_latency_linear_trend(
    summary_df: pd.DataFrame,
    *,
    figure_path: Path,
    title: str,
    trend_stats: Dict[str, float] | None,
    infant_labels: Sequence[str],
    adult_label: str | None,
) -> None:
    """Scatter + linear fit for infant cohorts with adult marker."""
    if summary_df.empty or not infant_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No latency data available for trend", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        return

    infant_df = summary_df[summary_df["cohort"].isin(infant_labels)].copy()
    infant_df = infant_df.dropna(subset=["mean_latency_frames"])
    if infant_df.empty or not trend_stats:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Latency linear trend unavailable", ha="center", va="center", fontsize=12)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
        return

    x = infant_df["cohort"].str.extract(r"(\d+)").astype(float)[0].to_numpy()
    y = infant_df["mean_latency_frames"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#1f77b4", label="Infant cohorts")

    slope = float(trend_stats.get("coef", 0.0))
    intercept = float(trend_stats.get("intercept", np.mean(y) if len(y) else 0.0))
    if len(x) >= 2:
        x_line = np.linspace(x.min(), x.max(), 100)
    else:
        x_line = np.array([x.min(), x.min() + 1])
    ax.plot(x_line, intercept + slope * x_line, color="#ff7f0e", label="Linear fit")

    ticks = list(x)
    tick_labels = [str(int(val)) for val in x]
    adult_series = summary_df[summary_df["cohort"] == adult_label]["mean_latency_frames"] if adult_label else pd.Series()
    if not adult_series.empty:
        adult_x = x.max() + 1 if len(x) else 12
        ax.scatter([adult_x], adult_series.to_numpy(), color="#f4a261", label=adult_label or "Adults")
        ticks.append(adult_x)
        tick_labels.append(adult_label or "Adults")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Mean latency to toy (frames)")
    ax.set_title(title)

    pvalue = trend_stats.get("pvalue")
    if pvalue is not None and not np.isnan(pvalue):
        ax.text(
            0.02,
            0.92,
            f"slope={slope:.2f}, p={pvalue:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    ax.legend()
    plt.subplots_adjust(top=0.84, bottom=0.25, left=0.15, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

