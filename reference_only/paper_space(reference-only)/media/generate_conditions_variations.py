"""Six variations of the 24 prime-target condition display."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def get_valid():
    """Return valid (prime, target) pairs and distances."""
    pairs = {}
    for p in range(1, 7):
        for t in range(1, 7):
            if p != t and abs(p - t) <= 3:
                pairs[(p, t)] = abs(p - t)
    return pairs


INK = "#1f2328"
MUTED = "#6b7280"
LIGHT = "#cbd5e1"
DIST_FILL = {1: "#c7dafa", 2: "#6aa1ee", 3: "#2563eb"}
DIST_TXT = {1: INK, 2: "white", 3: "white"}


def draw_legend(ax, x0, y, cs=0.4):
    """Draw distance legend at (x0, y)."""
    for i, (d, lbl) in enumerate([(1, "Dist 1"), (2, "Dist 2"), (3, "Dist 3")]):
        lx = x0 + i * 1.6
        ax.add_patch(patches.FancyBboxPatch(
            (lx, y), cs, cs * 0.8,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=DIST_FILL[d], edgecolor="white", linewidth=0.6))
        ax.text(lx + cs + 0.12, y + cs * 0.4, lbl,
                fontsize=7, ha="left", va="center", color=INK)


# ------------------------------------------------------------------
# Variation A: 6x6 matrix, colored by distance, no diagonal fill
# ------------------------------------------------------------------
def var_a(ax):
    ax.set_title("A: Color matrix (distance shading)", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()
    cs, gap = 0.75, 0.06
    x0, y0 = 0.8, 0.9

    for t in range(1, 7):
        ax.text(x0 + (t - 1) * (cs + gap) + cs / 2, y0 + 6 * (cs + gap) + 0.12,
                str(t), fontsize=8, ha="center", va="bottom", color=INK, fontweight="bold")
    ax.text(x0 + 2.5 * (cs + gap), y0 + 6 * (cs + gap) + 0.42,
            "Target", fontsize=7.5, ha="center", color=MUTED)

    for p in range(1, 7):
        ax.text(x0 - 0.15, y0 + (6 - p) * (cs + gap) + cs / 2,
                str(p), fontsize=8, ha="right", va="center", color=INK, fontweight="bold")
    ax.text(x0 - 0.48, y0 + 2.5 * (cs + gap) + cs / 2,
            "Prime", fontsize=7.5, ha="center", va="center", color=MUTED, rotation=90)

    for p in range(1, 7):
        for t in range(1, 7):
            cx = x0 + (t - 1) * (cs + gap)
            cy = y0 + (6 - p) * (cs + gap)
            if p == t:
                ax.plot([cx + 0.1, cx + cs - 0.1], [cy + 0.1, cy + cs - 0.1],
                        color="#e2e8f0", lw=0.8)
                ax.plot([cx + 0.1, cx + cs - 0.1], [cy + cs - 0.1, cy + 0.1],
                        color="#e2e8f0", lw=0.8)
            elif (p, t) in valid:
                d = valid[(p, t)]
                rect = patches.FancyBboxPatch(
                    (cx, cy), cs, cs,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor=DIST_FILL[d], edgecolor="white", linewidth=0.8)
                ax.add_patch(rect)
                ax.text(cx + cs / 2, cy + cs / 2, f"{p}{t}",
                        fontsize=8, ha="center", va="center",
                        color=DIST_TXT[d], fontweight="bold")

    draw_legend(ax, x0, 0.25)


# ------------------------------------------------------------------
# Variation B: 6x6 matrix, outline only (no color fill)
# ------------------------------------------------------------------
def var_b(ax):
    ax.set_title("B: Outline matrix (no fill)", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()
    cs, gap = 0.75, 0.06
    x0, y0 = 0.8, 0.9

    for t in range(1, 7):
        ax.text(x0 + (t - 1) * (cs + gap) + cs / 2, y0 + 6 * (cs + gap) + 0.12,
                str(t), fontsize=8, ha="center", va="bottom", color=INK, fontweight="bold")
    ax.text(x0 + 2.5 * (cs + gap), y0 + 6 * (cs + gap) + 0.42,
            "Target", fontsize=7.5, ha="center", color=MUTED)

    for p in range(1, 7):
        ax.text(x0 - 0.15, y0 + (6 - p) * (cs + gap) + cs / 2,
                str(p), fontsize=8, ha="right", va="center", color=INK, fontweight="bold")
    ax.text(x0 - 0.48, y0 + 2.5 * (cs + gap) + cs / 2,
            "Prime", fontsize=7.5, ha="center", va="center", color=MUTED, rotation=90)

    for p in range(1, 7):
        for t in range(1, 7):
            cx = x0 + (t - 1) * (cs + gap)
            cy = y0 + (6 - p) * (cs + gap)
            if p == t:
                rect = patches.FancyBboxPatch(
                    (cx, cy), cs, cs,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=0.6)
                ax.add_patch(rect)
            elif (p, t) in valid:
                rect = patches.FancyBboxPatch(
                    (cx, cy), cs, cs,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor="white", edgecolor="#94a3b8", linewidth=1.0)
                ax.add_patch(rect)
                ax.text(cx + cs / 2, cy + cs / 2, f"{p}{t}",
                        fontsize=8, ha="center", va="center",
                        color=INK, fontweight="bold")


# ------------------------------------------------------------------
# Variation C: Grouped rows by prime (compact horizontal chips)
# ------------------------------------------------------------------
def var_c(ax):
    ax.set_title("C: Rows grouped by prime", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()

    cw, ch = 0.72, 0.55
    gap_x, gap_y = 0.08, 0.18
    x0, y0 = 1.4, 0.6

    by_prime = {}
    for (p, t), d in sorted(valid.items()):
        by_prime.setdefault(p, []).append((t, d))

    for ridx, prime in enumerate(range(1, 7)):
        ry = y0 + (5 - ridx) * (ch + gap_y)
        ax.text(x0 - 0.15, ry + ch / 2, f"P{prime}",
                fontsize=8, ha="right", va="center", color=INK, fontweight="bold")
        for cidx, (target, dist) in enumerate(by_prime[prime]):
            cx = x0 + cidx * (cw + gap_x)
            rect = patches.FancyBboxPatch(
                (cx, ry), cw, ch,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor=DIST_FILL[dist], edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(cx + cw / 2, ry + ch / 2, f"{prime}{target}",
                    fontsize=8, ha="center", va="center",
                    color=DIST_TXT[dist], fontweight="bold")

    draw_legend(ax, x0, 0.05)


# ------------------------------------------------------------------
# Variation D: Flat chip cloud (all 24 in reading order)
# ------------------------------------------------------------------
def var_d(ax):
    ax.set_title("D: Flat chip layout (reading order)", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()
    codes = sorted(valid.keys())

    cw, ch = 0.72, 0.6
    gap = 0.1
    cols = 6
    x0, y0 = 0.5, 1.2

    for i, (p, t) in enumerate(codes):
        row, col = divmod(i, cols)
        cx = x0 + col * (cw + gap)
        cy = y0 + (3 - row) * (ch + gap)
        d = valid[(p, t)]
        rect = patches.FancyBboxPatch(
            (cx, cy), cw, ch,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=DIST_FILL[d], edgecolor="white", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(cx + cw / 2, cy + ch / 2, f"{p}{t}",
                fontsize=9, ha="center", va="center",
                color=DIST_TXT[d], fontweight="bold")

    draw_legend(ax, x0, 0.55)


# ------------------------------------------------------------------
# Variation E: Diagonal bands (grouped by distance)
# ------------------------------------------------------------------
def var_e(ax):
    ax.set_title("E: Grouped by distance", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()

    by_dist = {1: [], 2: [], 3: []}
    for (p, t), d in sorted(valid.items()):
        by_dist[d].append((p, t))

    cw, ch = 0.65, 0.55
    gap = 0.08
    x0, y0 = 0.3, 0.6

    for didx, dist in enumerate([1, 2, 3]):
        ry = y0 + (2 - didx) * (ch + 0.65)
        ax.text(x0, ry + ch / 2, f"Distance {dist}",
                fontsize=8, ha="left", va="center", color=INK, fontweight="bold")
        pairs = by_dist[dist]
        cx0 = x0 + 1.55
        for cidx, (p, t) in enumerate(pairs):
            cx = cx0 + cidx * (cw + gap)
            rect = patches.FancyBboxPatch(
                (cx, ry), cw, ch,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor=DIST_FILL[dist], edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(cx + cw / 2, ry + ch / 2, f"{p}{t}",
                    fontsize=8, ha="center", va="center",
                    color=DIST_TXT[dist], fontweight="bold")


# ------------------------------------------------------------------
# Variation F: Minimal dot-grid (dots for valid, empty for invalid)
# ------------------------------------------------------------------
def var_f(ax):
    ax.set_title("F: Minimal dot grid", fontsize=9,
                 fontweight="bold", color=INK, pad=8)
    valid = get_valid()
    cs, gap = 0.75, 0.06
    x0, y0 = 0.8, 0.9

    for t in range(1, 7):
        ax.text(x0 + (t - 1) * (cs + gap) + cs / 2, y0 + 6 * (cs + gap) + 0.12,
                str(t), fontsize=8, ha="center", va="bottom", color=INK, fontweight="bold")
    ax.text(x0 + 2.5 * (cs + gap), y0 + 6 * (cs + gap) + 0.42,
            "Target", fontsize=7.5, ha="center", color=MUTED)

    for p in range(1, 7):
        ax.text(x0 - 0.15, y0 + (6 - p) * (cs + gap) + cs / 2,
                str(p), fontsize=8, ha="right", va="center", color=INK, fontweight="bold")
    ax.text(x0 - 0.48, y0 + 2.5 * (cs + gap) + cs / 2,
            "Prime", fontsize=7.5, ha="center", va="center", color=MUTED, rotation=90)

    # Light grid lines
    for p in range(1, 7):
        for t in range(1, 7):
            cx = x0 + (t - 1) * (cs + gap) + cs / 2
            cy = y0 + (6 - p) * (cs + gap) + cs / 2
            if p == t:
                ax.scatter([cx], [cy], s=8, c="#e2e8f0", marker="x", linewidths=0.6, zorder=2)
            elif (p, t) in valid:
                d = valid[(p, t)]
                ax.scatter([cx], [cy], s=180, c=DIST_FILL[d],
                           edgecolors="white", linewidths=0.5, zorder=3)
                ax.text(cx, cy, f"{p}{t}", fontsize=6.5, ha="center", va="center",
                        color=DIST_TXT[d], fontweight="bold", zorder=4)

    draw_legend(ax, x0, 0.25)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), facecolor="white")
    fig.suptitle("Condition display variations (choose one)",
                 fontsize=14, fontweight="bold", color=INK, y=0.98)

    for ax in axes.flat:
        ax.set_xlim(-0.2, 6.2)
        ax.set_ylim(-0.1, 6.6)
        ax.set_aspect("equal")
        ax.axis("off")

    var_a(axes[0, 0])
    var_b(axes[0, 1])
    var_c(axes[0, 2])
    var_d(axes[1, 0])
    var_e(axes[1, 1])
    var_f(axes[1, 2])

    fig.subplots_adjust(hspace=0.18, wspace=0.12)
    output = Path(__file__).with_name("conditions_variations.png")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
