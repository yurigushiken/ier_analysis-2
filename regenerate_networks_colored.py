"""
Regenerate all gaze transition network diagrams with colored nodes,
directed arrows, and a 2% threshold note.

Produces 12 PNG files (6 GIVE + 6 SHOW conditions).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "man_face":     "#E74C3C",   # coral / red
    "woman_face":   "#3498DB",   # blue
    "toy_present":  "#2ECC71",   # green
    "toy_location": "#82E0AA",   # lighter green
    "man_body":     "#E67E22",   # orange
    "woman_body":   "#9B59B6",   # purple
}

LABEL_MAP = {
    "man_face":     "Man's\nFace",
    "woman_face":   "Woman's\nFace",
    "toy_present":  "Toy",
    "toy_location": "Toy\nLocation",
    "man_body":     "Man's\nBody",
    "woman_body":   "Woman's\nBody",
}

# Fixed positions  (x, y) -- spread out for clarity
NODE_POS = {
    "man_face":     (0.10, 0.88),
    "woman_face":   (0.90, 0.88),
    "toy_location": (0.50, 0.70),
    "toy_present":  (0.50, 0.42),
    "man_body":     (0.10, 0.08),
    "woman_body":   (0.90, 0.08),
}

COHORTS = [
    "7-month-olds",
    "8-month-olds",
    "9-month-olds",
    "10-month-olds",
    "11-month-olds",
    "Adults",
]

THRESHOLD = 0.02  # 2 %

# Datasets
BASE = r"D:\ier_analysis-2\analyses\gaze_transition_analysis"
DATASETS = [
    (
        "GIVE",
        os.path.join(BASE, "gw_transitions_min3_50_percent - Copy (7)",
                     "tables", "gw_transitions_min3_50_percent_transition_matrix.csv"),
        os.path.join(BASE, "gw_transitions_min3_50_percent - Copy (7)", "figures"),
        "gw_transitions_min3_50_percent_network",
    ),
    (
        "SHOW",
        os.path.join(BASE, "sw_transitions_min3_50_percent - Copy (3)",
                     "tables", "sw_transitions_min3_50_percent_transition_matrix.csv"),
        os.path.join(BASE, "sw_transitions_min3_50_percent - Copy (3)", "figures"),
        "sw_transitions_min3_50_percent_network",
    ),
]


def _bezier_point(x1, y1, x2, y2, rad, t):
    """
    Point on the quadratic Bezier that matplotlib's arc3,rad=r produces.
    """
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy) or 1e-9
    px, py = -dy / L, dx / L
    cx = mx + px * rad * L * 0.5
    cy = my + py * rad * L * 0.5
    bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
    by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
    return bx, by


def _repel_labels(positions, min_dist=0.045, iterations=30):
    """
    Simple iterative repulsion to push overlapping label positions apart.
    *positions* is a list of [x, y] pairs (mutable).
    """
    pts = [list(p) for p in positions]
    for _ in range(iterations):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                d = np.hypot(dx, dy)
                if d < min_dist and d > 0:
                    # push apart
                    overlap = (min_dist - d) / 2.0
                    ux, uy = dx / d, dy / d
                    pts[i][0] -= ux * overlap
                    pts[i][1] -= uy * overlap
                    pts[j][0] += ux * overlap
                    pts[j][1] += uy * overlap
                    moved = True
                elif d == 0:
                    # exactly overlapping -- nudge arbitrarily
                    pts[j][0] += min_dist * 0.5
                    pts[j][1] += min_dist * 0.5
                    moved = True
        if not moved:
            break
    return pts


def draw_network(cohort_df, cohort, condition_label, out_path):
    """Draw a single network diagram for one cohort and save to *out_path*."""

    # --- Compute proportions ------------------------------------------------
    total = cohort_df["mean_count"].sum()
    if total == 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, f"No transitions recorded\n{condition_label} - {cohort}",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    cohort_df = cohort_df.copy()
    cohort_df["prop"] = cohort_df["mean_count"] / total

    # --- Setup figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.12, 1.08)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # --- Draw nodes ---------------------------------------------------------
    for aoi, (x, y) in NODE_POS.items():
        color = NODE_COLORS[aoi]
        ax.scatter(x, y, s=2200, c=color, zorder=5,
                   edgecolors="white", linewidths=2.5)
        ax.text(x, y, LABEL_MAP[aoi], ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=6)

    # --- Collect edges that exceed threshold --------------------------------
    edges = []
    for _, row in cohort_df.iterrows():
        if row["prop"] >= THRESHOLD:
            edges.append((row["from_aoi"], row["to_aoi"], row["prop"]))

    if not edges:
        ax.set_title(f"{condition_label} \u2013 {cohort}", fontsize=14,
                     fontweight="bold", pad=14)
        ax.text(0.5, -0.08, "Note: Only transitions \u2265 2% shown",
                ha="center", va="top", fontsize=7, color="gray",
                transform=ax.transAxes)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    max_prop = max(e[2] for e in edges)
    min_lw, max_lw = 1.0, 5.0

    edge_set = {(e[0], e[1]) for e in edges}

    # --- Draw edges ---------------------------------------------------------
    edge_info = []  # store (from, to, prop, rad) for label placement
    for from_aoi, to_aoi, prop in edges:
        x1, y1 = NODE_POS[from_aoi]
        x2, y2 = NODE_POS[to_aoi]

        lw = min_lw + (prop / max_prop) * (max_lw - min_lw) if max_prop > 0 else min_lw

        reverse_exists = (to_aoi, from_aoi) in edge_set
        rad = 0.20 if reverse_exists else 0.08

        arrow = FancyArrowPatch(
            posA=(x1, y1), posB=(x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=lw,
            color="#3C3C3C",
            alpha=0.68,
            zorder=3,
            shrinkA=25, shrinkB=25,
        )
        ax.add_patch(arrow)
        edge_info.append((from_aoi, to_aoi, prop, rad))

    # --- Compute initial label positions ------------------------------------
    raw_positions = []
    for from_aoi, to_aoi, prop, rad in edge_info:
        x1, y1 = NODE_POS[from_aoi]
        x2, y2 = NODE_POS[to_aoi]
        lx, ly = _bezier_point(x1, y1, x2, y2, rad, t=0.30)
        raw_positions.append([lx, ly])

    # --- Repel overlapping labels -------------------------------------------
    adjusted = _repel_labels(raw_positions, min_dist=0.055, iterations=50)

    # --- Place edge labels --------------------------------------------------
    for idx, (from_aoi, to_aoi, prop, rad) in enumerate(edge_info):
        pct_str = f"{prop * 100:.1f}%"
        lx, ly = adjusted[idx]

        ax.text(lx, ly, pct_str, ha="center", va="center",
                fontsize=6, color="#2C2C2C", zorder=7,
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec="none", alpha=0.90))

    # --- Title & threshold note ---------------------------------------------
    ax.set_title(f"{condition_label} \u2013 {cohort}", fontsize=14,
                 fontweight="bold", pad=14)
    ax.text(0.5, -0.08, "Note: Only transitions \u2265 2% shown",
            ha="center", va="top", fontsize=7, color="gray",
            transform=ax.transAxes)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for condition_label, csv_path, fig_dir, prefix in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {condition_label} condition")
        print(f"  CSV : {csv_path}")
        print(f"  Figs: {fig_dir}")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)

        for cohort in COHORTS:
            cdf = df[df["cohort"] == cohort].copy()
            fname = f"{prefix}_{cohort}.png"
            out_path = os.path.join(fig_dir, fname)
            draw_network(cdf, cohort, condition_label, out_path)
            print(f"  [OK] {fname}")

    print("\nAll 12 network diagrams regenerated successfully.")


if __name__ == "__main__":
    main()
