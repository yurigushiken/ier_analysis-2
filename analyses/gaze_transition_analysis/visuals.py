"""Visualization helpers for transition matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_DPI = 300


def _format_trend_pvalue(pvalue: float | None) -> str:
    if pvalue is None or np.isnan(pvalue):
        return "p = NA"
    if pvalue < 1e-4:
        return "p < 0.0001"
    if pvalue < 1e-3:
        return "p < 0.001"
    return f"p = {pvalue:.3f}"


def _format_linear_trend_ylabel(label: str | None) -> str | None:
    """Format long y-axis labels so they render cleanly when rotated."""
    if not label:
        return label
    prefix = "Mean share of gaze transitions:"
    if label.startswith(prefix) and "\n" not in label:
        rest = label[len(prefix) :].strip()
        if rest:
            return f"{prefix}\n{rest}"
        return prefix
    return label


def _split_title_first_line(title: str | None) -> tuple[str | None, str | None]:
    """Split a multi-line title into (first_line, remaining_lines)."""
    if not title:
        return None, None
    parts = [p for p in str(title).split("\n")]
    if not parts:
        return None, None
    first = parts[0].strip() or None
    rest = "\n".join([p for p in parts[1:] if p is not None]).strip() or None
    return first, rest


def _build_linear_trend_title_layout(ax: plt.Axes, title: str | None) -> Dict[str, object]:
    """Return figure-centered title layout aligned to the axes center.

    We align to the axes center (not the full figure center) so the title visually centers
    over the plotting region even when large y-labels/legends shift the axes box.
    """
    lines = [line.strip() for line in (title or "").split("\n") if line.strip()]
    pos = ax.get_position()
    x_center = float(pos.x0 + pos.width / 2.0)
    # Compact vertical spacing between title lines.
    y_top = 0.958
    dy = 0.040
    ys = [y_top - i * dy for i in range(len(lines))]
    sizes = [16] + [12] * max(0, len(lines) - 1)
    weights = ["normal"] * len(lines)
    return {"x": x_center, "lines": lines, "ys": ys, "sizes": sizes, "weights": weights}


def _linear_trend_axes_top_for_title_lines(n_lines: int) -> float:
    """Return a tuned subplots_adjust(top=...) for title line count."""
    if n_lines <= 2:
        return 0.88
    return 0.83


def _apply_linear_trend_yaxis_format(ax: plt.Axes) -> None:
    """Use consistent 0.1 ticks with one-decimal labels."""
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def _add_transition_icon(
    ax: plt.Axes,
    image_path: Path,
    *,
    xy: tuple[float, float] = (0.5, 0.995),
    zoom: float = 0.108675,
) -> AnnotationBbox:
    """Add a small transition icon image inside the axes (axes-fraction coords)."""
    arr = mpimg.imread(image_path)
    image = OffsetImage(arr, zoom=zoom)
    ab = AnnotationBbox(
        image,
        xy=xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 1.0),
        pad=0.0,
        zorder=0.0,
    )
    ax.add_artist(ab)
    return ab


def _apply_linear_trend_title(fig: plt.Figure, ax: plt.Axes, title: str | None) -> Dict[str, object]:
    layout = _build_linear_trend_title_layout(ax, title)
    x = layout["x"]
    for line, y, size, weight in zip(
        layout["lines"], layout["ys"], layout["sizes"], layout["weights"], strict=False
    ):
        fig.text(
            x,
            y,
            line,
            ha="center",
            va="top",
            fontsize=size,
            fontweight=weight,
        )
    return layout


def plot_heatmap(
    matrix_df: pd.DataFrame,
    *,
    aoi_nodes: Sequence[str],
    cohorts: Sequence[str],
    figure_path: Path,
    title: str,
    subtitle: str = "",
) -> None:
    """Plot a cohort-stacked heatmap."""
    if matrix_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No transitions available", ha="center", va="center", fontsize=12)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return

    from_to = [(frm, to) for frm in aoi_nodes for to in aoi_nodes if frm != to]
    num_rows = len(cohorts)
    fig, axes = plt.subplots(num_rows, 1, figsize=(max(6, len(from_to) * 0.5), 3 * num_rows))
    if num_rows == 1:
        axes = [axes]

    for ax, cohort in zip(axes, cohorts):
        cohort_df = matrix_df[matrix_df["cohort"] == cohort]
        data = []
        labels = []
        for frm, to in from_to:
            row = cohort_df[
                (cohort_df["from_aoi"] == frm) & (cohort_df["to_aoi"] == to)
            ]
            value = row["mean_count"].iloc[0] if not row.empty else 0.0
            data.append(value)
            labels.append(f"{frm}\n→ {to}")
        ax.imshow(np.array([data]), aspect="auto", cmap="viridis")
        positions = np.arange(len(labels)) + 0.5
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
        ax.set_xlim(0, len(labels))
        ax.set_yticks([])
        ax.set_ylabel(cohort, rotation=0, labelpad=40, fontsize=11)
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(full_title, fontsize=14)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.12, right=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close(fig)


NODE_LAYOUT = {
    "man_face": (-0.6, 0.8),
    "man_body": (-0.6, 0.4),
    "woman_face": (0.6, 0.8),
    "woman_body": (0.6, 0.4),
    "toy_present": (0.0, 0.5),
    "toy_location": (0.0, 0.6),
}


def plot_transition_networks(
    matrix_df: pd.DataFrame,
    *,
    cohorts: Sequence[str],
    aoi_nodes: Sequence[str],
    figures_dir: Path,
    filename_prefix: str,
    threshold: float = 0.02,
) -> None:
    """Draw network graphs per cohort showing transition probabilities."""
    if matrix_df.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        cohort_df = matrix_df[matrix_df["cohort"] == cohort]
        if cohort_df.empty:
            continue
        total = cohort_df["mean_count"].sum()
        if total <= 0:
            continue
        G = nx.DiGraph()
        for node in aoi_nodes:
            if node == "off_screen":
                continue
            G.add_node(node)
        for row in cohort_df.itertuples():
            if row.from_aoi == row.to_aoi:
                continue
            probability = row.mean_count / total
            if probability < threshold:
                continue
            G.add_edge(row.from_aoi, row.to_aoi, weight=probability)
        if not G.edges:
            continue
        positions = {node: NODE_LAYOUT.get(node, (0.0, 0.0)) for node in G.nodes}
        weights = [max(0.5, edge_data["weight"] * 10) for _, _, edge_data in G.edges(data=True)]
        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw_networkx_nodes(G, positions, node_color="#f4f1de", edgecolors="#333333", node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, positions, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            G,
            positions,
            arrowstyle="-|>",
            arrowsize=15,
            edge_color="#1d3557",
            width=weights,
            ax=ax,
        )
        edge_labels = {(u, v): f"{data['weight']*100:.1f}%" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=8, ax=ax)
        ax.set_title(f"{filename_prefix} - {cohort}")
        ax.axis("off")
        safe_label = cohort.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(figures_dir / f"{filename_prefix}_network_{safe_label}.png", dpi=DEFAULT_DPI)
        plt.close(fig)


def plot_single_strategy_bars(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    label: str,
    figure_path: Path,
    title: str,
    cohorts_order: Sequence[str],
    color: str,
    annotations: Optional[Sequence[Dict[str, object]]] = None,
    reference_label: str | None = None,
) -> None:
    """Render a single-strategy bar chart with optional annotations."""
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No strategy data", ha="center", va="center", fontsize=12)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return
    cohorts = list(cohorts_order)
    if reference_label and reference_label in cohorts:
        cohorts = [reference_label] + [c for c in cohorts if c != reference_label]
    reference_label = cohorts_order[0] if cohorts_order else None
    working = (
        summary_df.set_index("cohort")
        .reindex(cohorts)
        .reset_index()
        .rename(columns={"index": "cohort"})
    )
    x = np.arange(len(cohorts))
    values = working[value_column].fillna(0.0).to_numpy()

    reference_idx = next((i for i, cohort in enumerate(cohorts) if cohort == reference_label), None)
    fig, ax = plt.subplots(figsize=(max(6, len(cohorts) * 0.9), 4.8))
    bar_colors = [color] * len(cohorts)
    if reference_idx is not None:
        bar_colors[reference_idx] = "#f4a261"
    ax.bar(x, values, color=bar_colors, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=30, ha="right")
    ax.set_ylabel("Mean proportion of transitions")
    ax.set_ylim(0, min(1.0, max(0.35, values.max() + 0.15)))
    ax.set_title(title)

    max_bar = values.max() if len(values) else 0
    if annotations:
        for ann in annotations:
            idx = ann["to_idx"]
            symbol = ann.get("label", "*")
            y = values[idx] + 0.03 * max(1.0, max_bar)
            ax.text(x[idx], y, symbol, ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_linear_trend(
    summary_df: pd.DataFrame,
    trend_metrics: Dict[str, float] | None,
    *,
    figure_path: Path,
    value_column: str,
    label: str,
    title: str,
    y_axis_label: str | None = None,
    adult_label: str | None = None,
    legend_loc: str = "best",
    transition_icon_path: Path | None = None,
) -> None:
    if summary_df.empty or trend_metrics is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for linear trend", ha="center", va="center", fontsize=12)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return
    infants = summary_df[summary_df["cohort"].str.contains("month")]
    if infants.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No infant cohorts", ha="center", va="center", fontsize=12)
        fig.savefig(figure_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        return
    x = infants["cohort"].str.extract(r"(\d+)").astype(float)[0].to_numpy()
    y = infants[value_column].to_numpy()
    coef = trend_metrics.get("coef", 0.0)
    intercept = trend_metrics.get("intercept")
    if intercept is None or np.isnan(intercept):
        intercept = float(np.mean(y) - coef * np.mean(x))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#1f77b4", label="Infant cohort means")
    x_line = np.linspace(x.min(), x.max(), 100) if len(x) > 1 else np.array([x.min(), x.min() + 1])
    ax.plot(x_line, intercept + coef * x_line, color="#ff7f0e", label="Linear fit (infants)")
    ax.set_xlabel("Age (months)", fontsize=13)
    xticks = list(sorted(set(x)))
    tick_labels = [str(int(val)) for val in xticks]
    adult_series = (
        summary_df.loc[summary_df["cohort"] == adult_label, value_column]
        if adult_label
        else pd.Series(dtype=float)
    )
    if not adult_series.empty:
        adult_x = (max(xticks) if xticks else 11) + 1
        adult_legend_label = (
            f"{adult_label} (reference)" if adult_label else "Adults (reference)"
        )
        ax.scatter(
            [adult_x],
            adult_series.to_numpy(),
            color="#f4a261",
            marker="s",
            label=adult_legend_label,
        )
        xticks.append(adult_x)
        tick_labels.append(adult_label or "Adults")
    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    axis_label = y_axis_label or f"{label} proportion"
    axis_label = _format_linear_trend_ylabel(axis_label)
    ax.set_ylabel(axis_label, fontsize=11)
    combined_values = list(y)
    if not adult_series.empty:
        combined_values.extend(adult_series.to_list())
    max_value = max(combined_values) if combined_values else 0.0
    upper = min(1.0, max(0.35, max_value + 0.1))
    ax.set_ylim(0, upper)
    _apply_linear_trend_yaxis_format(ax)
    ax.set_title("")
    title_lines = [line.strip() for line in (title or "").split("\n") if line.strip()]
    pvalue = trend_metrics.get("pvalue")
    pvalue_text = _format_trend_pvalue(pvalue)
    ax.text(
        0.05,
        0.92,
        f"coef = {coef:.3f}, {pvalue_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.legend(
        loc=legend_loc,
        fontsize=9,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.4,
        handletextpad=0.4,
        markerscale=0.9,
    )
    plt.subplots_adjust(
        top=_linear_trend_axes_top_for_title_lines(len(title_lines)),
        bottom=0.18,
        left=0.12,
        right=0.95,
    )
    layout = _apply_linear_trend_title(fig, ax, title)
    if transition_icon_path is not None:
        try:
            _add_transition_icon(ax, transition_icon_path)
        except Exception:
            # Purely presentational; don't fail the analysis if it can't be loaded.
            pass
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=DEFAULT_DPI)
    plt.close(fig)

