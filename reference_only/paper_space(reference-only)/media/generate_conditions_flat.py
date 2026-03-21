"""Variations of flat 4x6 chip layout for 24 conditions (no distance coloring)."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


INK = "#1f2328"
MUTED = "#6b7280"

CODES = sorted(
    (p, t) for p in range(1, 7) for t in range(1, 7)
    if p != t and abs(p - t) <= 3
)


def draw_grid(ax, title, fill, edge, text_color, lw, radius, fontsize=9,
              shadow=False, bold=True):
    """Draw 4x6 chip grid with given style."""
    ax.set_title(title, fontsize=9, fontweight="bold", color=INK, pad=6)

    cw, ch = 0.78, 0.62
    gap = 0.10
    cols = 6
    x0, y0 = 0.15, 0.4

    for i, (p, t) in enumerate(CODES):
        row, col = divmod(i, cols)
        cx = x0 + col * (cw + gap)
        cy = y0 + (3 - row) * (ch + gap)

        if shadow:
            ax.add_patch(patches.FancyBboxPatch(
                (cx + 0.03, cy - 0.03), cw, ch,
                boxstyle=f"round,pad=0.01,rounding_size={radius}",
                facecolor="#e2e8f0", edgecolor="none"))

        rect = patches.FancyBboxPatch(
            (cx, cy), cw, ch,
            boxstyle=f"round,pad=0.01,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        ax.text(cx + cw / 2, cy + ch / 2, f"{p}{t}",
                fontsize=fontsize, ha="center", va="center",
                color=text_color, fontweight="bold" if bold else "normal")


def main():
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), facecolor="white")
    fig.suptitle("Flat 4x6 condition chip variations",
                 fontsize=14, fontweight="bold", color=INK, y=0.98)

    for ax in axes.flat:
        ax.set_xlim(-0.2, 5.8)
        ax.set_ylim(-0.1, 3.6)
        ax.set_aspect("equal")
        ax.axis("off")

    # 1: White fill, gray border
    draw_grid(axes[0, 0], "1: White / gray border",
              fill="white", edge="#94a3b8", text_color=INK, lw=1.0, radius=0.06)

    # 2: Light blue fill, no border
    draw_grid(axes[0, 1], "2: Light blue / no border",
              fill="#e0ecf9", edge="#e0ecf9", text_color="#1e3a5f", lw=0, radius=0.06)

    # 3: Medium blue fill, white text
    draw_grid(axes[0, 2], "3: Blue fill / white text",
              fill="#3b82f6", edge="#2563eb", text_color="white", lw=0.8, radius=0.06)

    # 4: Light gray fill, dark text, rounded
    draw_grid(axes[1, 0], "4: Light gray / rounded",
              fill="#f1f5f9", edge="#cbd5e1", text_color=INK, lw=0.8, radius=0.10)

    # 5: White with shadow
    draw_grid(axes[1, 1], "5: White / drop shadow",
              fill="white", edge="#e2e8f0", text_color=INK, lw=0.6, radius=0.06,
              shadow=True)

    # 6: Outline only, no fill
    draw_grid(axes[1, 2], "6: Outline only (no fill)",
              fill="none", edge="#64748b", text_color=INK, lw=1.2, radius=0.06)

    # 7: Dark fill, light text
    draw_grid(axes[2, 0], "7: Dark slate / light text",
              fill="#334155", edge="#1e293b", text_color="#e2e8f0", lw=0.8, radius=0.06)

    # 8: Warm beige / earthy
    draw_grid(axes[2, 1], "8: Warm tone",
              fill="#fef3c7", edge="#d97706", text_color="#78350f", lw=0.8, radius=0.06)

    # 9: Minimal, no border, muted text
    draw_grid(axes[2, 2], "9: Ultra-minimal (no border, light weight)",
              fill="#f8fafc", edge="#f8fafc", text_color="#475569", lw=0, radius=0.04,
              bold=False, fontsize=9)

    fig.subplots_adjust(hspace=0.22, wspace=0.08)
    output = Path(__file__).with_name("conditions_flat_variations.png")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
