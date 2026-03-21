"""Generate flat chip layouts for 24-condition and 6-condition panels."""
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def _draw_chips(labels, cols, output_name, *, cw=0.78, ch=0.62, gap=0.10):
    fill = "#e0ecf9"
    text_color = "#1e3a5f"
    x0, y0 = 0.0, 0.0
    rows = (len(labels) + cols - 1) // cols

    total_w = cols * cw + (cols - 1) * gap
    total_h = rows * ch + (rows - 1) * gap

    fig, ax = plt.subplots(
        figsize=(total_w * 1.15, total_h * 1.15),
        facecolor="white",
    )
    ax.set_xlim(-0.15, total_w + 0.15)
    ax.set_ylim(-0.15, total_h + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, label in enumerate(labels):
        row, col = divmod(i, cols)
        cx = x0 + col * (cw + gap)
        cy = y0 + (rows - 1 - row) * (ch + gap)
        rect = patches.FancyBboxPatch(
            (cx, cy),
            cw,
            ch,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=fill,
            edgecolor=fill,
            linewidth=0,
        )
        ax.add_patch(rect)
        ax.text(
            cx + cw / 2,
            cy + ch / 2,
            str(label),
            fontsize=11,
            ha="center",
            va="center",
            color=text_color,
            fontweight="bold",
        )

    output = Path(__file__).with_name(output_name)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    # 24 prime-target change conditions for temporal analysis.
    cond24 = [f"{p}{t}" for p in range(1, 7) for t in range(1, 7) if p != t and abs(p - t) <= 3]
    _draw_chips(cond24, cols=6, output_name="conditions-24.png")

    # Six target numerosities for whole-epoch static analysis.
    cond6 = [str(x) for x in range(1, 7)]
    _draw_chips(cond6, cols=6, output_name="conditions-6.png")


if __name__ == "__main__":
    main()
