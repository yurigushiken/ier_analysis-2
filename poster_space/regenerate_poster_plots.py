"""Regenerate the 4 linear trend plots for the poster without titles."""

import sys
sys.path.insert(0, r"D:\ier_analysis-2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(r"D:\ier_analysis-2\poster_space\poster\figures")

# ---------- Data ----------

gw_summary = pd.DataFrame({
    "cohort": ["7-month-olds", "8-month-olds", "9-month-olds", "10-month-olds", "11-month-olds", "Adults"],
    "agent_agent_attention_mean": [0.03846, 0.05873, 0.13977, 0.22152, 0.20323, 0.24439],
    "motion_tracking_mean": [0.54567, 0.44206, 0.13339, 0.23633, 0.09099, 0.29765],
})

sw_summary = pd.DataFrame({
    "cohort": ["7-month-olds", "8-month-olds", "9-month-olds", "10-month-olds", "11-month-olds", "Adults"],
    "agent_agent_attention_mean": [0.05625, 0.12474, 0.07487, 0.17148, 0.15773, 0.34273],
    "motion_tracking_mean": [0.45625, 0.27762, 0.30212, 0.16822, 0.09157, 0.06000],
})

# GEE linear trend coefficients from reports
gw_social_trend = {"coef": 0.0529, "intercept": -0.3113, "pvalue": 0.001}
gw_motion_trend = {"coef": -0.0777, "intercept": 0.9512, "pvalue": 0.006}
sw_social_trend = {"coef": 0.0328, "intercept": -0.1683, "pvalue": 0.007}
sw_motion_trend = {"coef": -0.0751, "intercept": 0.9054, "pvalue": 0.000}


def format_pvalue(pvalue):
    if pvalue < 1e-4:
        return "p < 0.0001"
    if pvalue < 1e-3:
        return "p < 0.001"
    return f"p = {pvalue:.3f}"


def plot_trend_notitle(summary_df, trend_metrics, value_column, y_axis_label,
                       adult_label, legend_loc, figure_path):
    infants = summary_df[summary_df["cohort"].str.contains("month")]
    x = infants["cohort"].str.extract(r"(\d+)").astype(float)[0].to_numpy()
    y = infants[value_column].to_numpy()

    coef = trend_metrics["coef"]
    intercept = trend_metrics["intercept"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#1f77b4", label="Infant cohort means")

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + coef * x_line, color="#ff7f0e", label="Linear fit (infants)")

    ax.set_xlabel("Age (months)", fontsize=13)
    xticks = list(sorted(set(x)))
    tick_labels = [str(int(val)) for val in xticks]

    adult_series = summary_df.loc[summary_df["cohort"] == adult_label, value_column]
    if not adult_series.empty:
        adult_x = max(xticks) + 1
        ax.scatter([adult_x], adult_series.to_numpy(), color="#f4a261", marker="s",
                   label=f"{adult_label} (reference)")
        xticks.append(adult_x)
        tick_labels.append(adult_label)

    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    # Format y-axis label with line break if long
    if y_axis_label and ":" in y_axis_label and "\n" not in y_axis_label:
        prefix, rest = y_axis_label.split(":", 1)
        y_axis_label = f"{prefix}:\n{rest.strip()}"
    ax.set_ylabel(y_axis_label, fontsize=11)

    combined = list(y)
    if not adult_series.empty:
        combined.extend(adult_series.to_list())
    upper = min(1.0, max(0.35, max(combined) + 0.1))
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    pvalue_text = format_pvalue(trend_metrics["pvalue"])
    ax.text(0.05, 0.92, f"coef = {coef:.3f}, {pvalue_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ax.legend(loc=legend_loc, fontsize=9, frameon=True, borderpad=0.3,
              labelspacing=0.3, handlelength=1.4, handletextpad=0.4, markerscale=0.9)

    # No title - reclaim space
    plt.subplots_adjust(top=0.95, bottom=0.18, left=0.12, right=0.95)

    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {figure_path}")


# ---------- Generate ----------

plot_trend_notitle(
    gw_summary, gw_social_trend,
    value_column="agent_agent_attention_mean",
    y_axis_label="Mean share of gaze transitions: man's face \u2194 woman's face",
    adult_label="Adults", legend_loc="lower right",
    figure_path=OUTPUT_DIR / "give_social_attention.png",
)

plot_trend_notitle(
    gw_summary, gw_motion_trend,
    value_column="motion_tracking_mean",
    y_axis_label="Mean share of gaze transitions: toy \u2194 body",
    adult_label="Adults", legend_loc="upper right",
    figure_path=OUTPUT_DIR / "give_motion_tracking.png",
)

plot_trend_notitle(
    sw_summary, sw_social_trend,
    value_column="agent_agent_attention_mean",
    y_axis_label="Mean share of gaze transitions: man's face \u2194 woman's face",
    adult_label="Adults", legend_loc="lower right",
    figure_path=OUTPUT_DIR / "show_social_attention.png",
)

plot_trend_notitle(
    sw_summary, sw_motion_trend,
    value_column="motion_tracking_mean",
    y_axis_label="Mean share of gaze transitions: toy \u2194 body",
    adult_label="Adults", legend_loc="upper right",
    figure_path=OUTPUT_DIR / "show_motion_tracking.png",
)

print("All 4 poster plots regenerated without titles.")
