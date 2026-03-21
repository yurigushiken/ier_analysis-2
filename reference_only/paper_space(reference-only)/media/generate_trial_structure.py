"""Minimal trial-structure figure: prime sequence -> target -> condition label."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _dot_layout(n: int):
    """Return normalized dot centers in [-0.5, 0.5] box coordinates."""
    layouts = {
        1: [(0.0, 0.0)],
        2: [(-0.18, 0.0), (0.18, 0.0)],
        3: [(-0.20, 0.18), (0.20, 0.18), (0.0, -0.18)],
        4: [(-0.18, 0.18), (0.18, 0.18), (-0.18, -0.18), (0.18, -0.18)],
        5: [(-0.20, 0.20), (0.20, 0.20), (0.0, 0.0), (-0.20, -0.20), (0.20, -0.20)],
        6: [(-0.20, 0.22), (0.20, 0.22), (-0.20, 0.0), (0.20, 0.0), (-0.20, -0.22), (0.20, -0.22)],
    }
    return layouts[n]


def _draw_dot_array(ax, x: float, y: float, w: float, h: float, n: int, color: str) -> None:
    """Draw a simple dot-array stimulus centered inside a rounded box."""
    # Dot radius scales with box size; chosen to keep arrays legible in compact figure.
    r = min(w, h) * 0.095
    cx = x + w / 2
    cy = y + h / 2
    for dx, dy in _dot_layout(n):
        ax.add_patch(
            patches.Circle(
                (cx + dx * w, cy + dy * h),
                r,
                facecolor=color,
                edgecolor="none",
            )
        )


def main() -> None:
    bg = "white"
    ink = "#1f2328"
    muted = "#6b7280"
    prime_fill = "#e8eef5"
    prime_edge = "#94a3b8"
    target_fill = "#2563eb"
    target_edge = "#1e40af"

    fig, ax = plt.subplots(figsize=(7.0, 1.8), facecolor=bg)
    ax.set_xlim(-0.05, 7.7)
    ax.set_ylim(-0.15, 2.05)
    ax.axis("off")

    bw, bh = 1.1, 1.0
    y_box = 0.45

    # First prime
    x1 = 0.2
    rect = patches.FancyBboxPatch(
        (x1, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=prime_fill, edgecolor=prime_edge, linewidth=1.1)
    ax.add_patch(rect)
    _draw_dot_array(ax, x1, y_box, bw, bh, n=4, color="white")

    # Ellipsis
    ax.text(1.7, y_box + bh / 2, "\u2026",
            fontsize=22, ha="center", va="center", color=muted)

    # Last prime
    x2 = 2.2
    rect = patches.FancyBboxPatch(
        (x2, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=prime_fill, edgecolor=prime_edge, linewidth=1.1)
    ax.add_patch(rect)
    _draw_dot_array(ax, x2, y_box, bw, bh, n=4, color="white")

    # Label above primes
    mid_prime = (x1 + x2 + bw) / 2
    ax.text(mid_prime, y_box + bh + 0.12,
            "Prime (3\u20135 repeats, 250 ms each)",
            fontsize=9.5, ha="center", va="bottom", color=muted)

    # Arrow
    ax.annotate(
        "", xy=(4.35, y_box + bh / 2),
        xytext=(3.65, y_box + bh / 2),
        arrowprops=dict(arrowstyle="->,head_width=0.25", lw=1.8, color=muted))

    # Target
    tx = 4.55
    rect = patches.FancyBboxPatch(
        (tx, y_box), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=target_fill, edgecolor=target_edge, linewidth=1.6)
    ax.add_patch(rect)
    _draw_dot_array(ax, tx, y_box, bw, bh, n=2, color="white")
    ax.text(tx + bw / 2, y_box + bh + 0.12, "Target (250 ms)",
            fontsize=9.5, ha="center", va="bottom",
            color=target_edge, fontweight="bold")

    # Condition label
    ax.annotate(
        "Condition 42",
        xy=(tx + bw + 0.12, y_box + bh / 2),
        xytext=(tx + bw + 0.55, y_box + bh / 2),
        fontsize=12, fontweight="bold", color=ink, va="center",
        arrowprops=dict(arrowstyle="-", lw=1.2, color=muted))

    output = Path(__file__).with_name("trial_structure.png")
    fig.savefig(output, dpi=600, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
