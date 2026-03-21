"""Clean, abstract trial schematic + compact condition matrix for CCN paper."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main() -> None:
    bg = "white"
    ink = "#1f2328"
    muted = "#6b7280"
    prime_fill = "#e8eef5"
    prime_edge = "#94a3b8"
    target_fill = "#2563eb"
    target_edge = "#1e40af"
    dist_colors = {1: "#c7dafa", 2: "#6aa1ee", 3: "#2563eb"}
    dist_text = {1: ink, 2: "white", 3: "white"}

    fig = plt.figure(figsize=(7.0, 7.8), facecolor=bg)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0.12,
                          left=0.06, right=0.96, top=0.96, bottom=0.04)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # (a) Trial timeline — abstract, no dot arrays
    # ------------------------------------------------------------------
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(-0.1, 2.8)
    ax_top.axis("off")

    ax_top.text(0.05, 2.55, "(a)  Trial structure",
                fontsize=14, fontweight="bold", color=ink)
    ax_top.text(0.05, 2.20, "Example: condition 42 (prime = 4, target = 2)",
                fontsize=9.5, color=muted)

    bw, bh = 1.1, 1.0
    y_box = 0.55

    # First prime
    x1 = 0.2
    rect = patches.FancyBboxPatch(
        (x1, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=prime_fill, edgecolor=prime_edge, linewidth=1.1)
    ax_top.add_patch(rect)
    ax_top.text(x1 + bw / 2, y_box + bh / 2, "4",
                fontsize=24, ha="center", va="center",
                color=prime_edge, fontweight="bold")

    # Ellipsis
    ax_top.text(1.7, y_box + bh / 2, "\u2026",
                fontsize=22, ha="center", va="center", color=muted)

    # Last prime
    x2 = 2.2
    rect = patches.FancyBboxPatch(
        (x2, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=prime_fill, edgecolor=prime_edge, linewidth=1.1)
    ax_top.add_patch(rect)
    ax_top.text(x2 + bw / 2, y_box + bh / 2, "4",
                fontsize=24, ha="center", va="center",
                color=prime_edge, fontweight="bold")

    # Brace / label above primes
    mid_prime = (x1 + x2 + bw) / 2
    ax_top.text(mid_prime, y_box + bh + 0.18,
                "Prime (3\u20135 repeats, 250 ms each)",
                fontsize=9.5, ha="center", va="bottom", color=muted)

    # Arrow
    ax_top.annotate(
        "", xy=(4.35, y_box + bh / 2),
        xytext=(3.65, y_box + bh / 2),
        arrowprops=dict(arrowstyle="->,head_width=0.25", lw=1.8, color=muted))

    # Target
    tx = 4.55
    rect = patches.FancyBboxPatch(
        (tx, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=target_fill, edgecolor=target_edge, linewidth=1.6)
    ax_top.add_patch(rect)
    ax_top.text(tx + bw / 2, y_box + bh / 2, "2",
                fontsize=24, ha="center", va="center",
                color="white", fontweight="bold")
    ax_top.text(tx + bw / 2, y_box + bh + 0.18, "Target (250 ms)",
                fontsize=9.5, ha="center", va="bottom",
                color=target_edge, fontweight="bold")

    # Condition label to the right
    ax_top.annotate(
        "Condition 42",
        xy=(tx + bw + 0.12, y_box + bh / 2),
        xytext=(tx + bw + 0.55, y_box + bh / 2),
        fontsize=12, fontweight="bold", color=ink, va="center",
        arrowprops=dict(arrowstyle="-", lw=1.2, color=muted))

    # ------------------------------------------------------------------
    # (b) Compact 6x6 condition matrix
    # ------------------------------------------------------------------
    ax_bot.set_xlim(-1.0, 7.8)
    ax_bot.set_ylim(-0.8, 8.6)
    ax_bot.axis("off")

    ax_bot.text(-0.85, 8.35, "(b)  24 prime-target conditions",
                fontsize=14, fontweight="bold", color=ink)

    valid = {(p, t) for p in range(1, 7) for t in range(1, 7)
             if p != t and abs(p - t) <= 3}

    cs = 0.95          # cell size
    gap = 0.08
    x0 = 0.6           # left edge of grid
    y0 = 1.0           # bottom edge of grid

    # Column headers (Target)
    col_top = y0 + 6 * (cs + gap)
    for t in range(1, 7):
        cx = x0 + (t - 1) * (cs + gap) + cs / 2
        ax_bot.text(cx, col_top + 0.15, str(t),
                    fontsize=11, ha="center", va="bottom",
                    color=ink, fontweight="bold")
    ax_bot.text(x0 + 2.5 * (cs + gap) + cs / 2,
                col_top + 0.62,
                "Target numerosity", fontsize=10, ha="center", color=muted)

    # Row headers (Prime)
    for p in range(1, 7):
        ry = y0 + (6 - p) * (cs + gap) + cs / 2
        ax_bot.text(x0 - 0.22, ry, str(p),
                    fontsize=11, ha="right", va="center",
                    color=ink, fontweight="bold")
    ax_bot.text(x0 - 0.65,
                y0 + 2.5 * (cs + gap) + cs / 2,
                "Prime\nnumerosity", fontsize=10, ha="center",
                va="center", color=muted, rotation=90, linespacing=1.3)

    for p in range(1, 7):
        for t in range(1, 7):
            cx = x0 + (t - 1) * (cs + gap)
            cy = y0 + (6 - p) * (cs + gap)
            if p == t:
                # Diagonal — no-change (not analyzed)
                rect = patches.FancyBboxPatch(
                    (cx, cy), cs, cs,
                    boxstyle="round,pad=0.01,rounding_size=0.06",
                    facecolor="#f1f5f9", edgecolor="#e2e8f0", linewidth=0.7)
                ax_bot.add_patch(rect)
                ax_bot.text(cx + cs / 2, cy + cs / 2, "\u2014",
                            fontsize=10, ha="center", va="center", color="#cbd5e1")
            elif (p, t) in valid:
                dist = abs(p - t)
                rect = patches.FancyBboxPatch(
                    (cx, cy), cs, cs,
                    boxstyle="round,pad=0.01,rounding_size=0.06",
                    facecolor=dist_colors[dist],
                    edgecolor="white", linewidth=1.0)
                ax_bot.add_patch(rect)
                ax_bot.text(cx + cs / 2, cy + cs / 2, f"{p}{t}",
                            fontsize=11, ha="center", va="center",
                            color=dist_text[dist], fontweight="bold")
            # else: out-of-range, leave blank

    # Legend — compact, below grid
    ly = 0.15
    for i, (d, lbl) in enumerate([(1, "Distance 1"),
                                   (2, "Distance 2"),
                                   (3, "Distance 3")]):
        lx = x0 + i * 2.3
        ax_bot.add_patch(patches.FancyBboxPatch(
            (lx, ly), 0.55, 0.38,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=dist_colors[d], edgecolor="white", linewidth=0.8))
        ax_bot.text(lx + 0.72, ly + 0.19, lbl,
                    fontsize=9.5, ha="left", va="center", color=ink)

    # Gray = no-change annotation
    ax_bot.add_patch(patches.FancyBboxPatch(
        (x0 + 3 * 2.3, ly), 0.55, 0.38,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor="#f1f5f9", edgecolor="#e2e8f0", linewidth=0.8))
    ax_bot.text(x0 + 3 * 2.3 + 0.72, ly + 0.19, "No change",
                fontsize=9.5, ha="left", va="center", color=muted)

    output = Path(__file__).with_name("option1_trial_schematic.png")
    fig.savefig(output, dpi=600, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
