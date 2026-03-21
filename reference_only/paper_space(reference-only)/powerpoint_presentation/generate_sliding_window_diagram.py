"""Generate a publication-ready sliding window visualization for the PPT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "media"
OUTPUT.mkdir(exist_ok=True)

# -- Colors --
BG = "#F5F0E8"
BLUE_DARK = "#3E648C"
BLUE_MED = "#9CC4E4"
BLUE_LIGHT = "#D2E4F3"
TEXT_DARK = "#263242"
TEXT_MUTED = "#5A6E82"
ACCENT_WARM = "#E8C9A0"
WHITE = "#FFFFFF"
GREEN = "#7CB98B"
ORANGE = "#E8A96A"


def main():
    fig = plt.figure(figsize=(14, 7.5))
    fig.patch.set_facecolor(BG)

    # Two rows: top = EEG signal + windows, bottom = RDM at each window
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.35,
                          left=0.06, right=0.96, top=0.88, bottom=0.08)

    # ===== TOP: EEG epoch with sliding windows =====
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_facecolor(BG)

    # Simulated EEG-like waveform
    np.random.seed(42)
    t = np.linspace(-100, 700, 800)
    # Create a plausible ERP-like waveform
    signal = (
        -3.0 * np.exp(-((t - 100) ** 2) / (2 * 30 ** 2))   # N1
        + 4.0 * np.exp(-((t - 200) ** 2) / (2 * 50 ** 2))   # P2
        - 2.5 * np.exp(-((t - 300) ** 2) / (2 * 60 ** 2))   # N2
        + 3.0 * np.exp(-((t - 450) ** 2) / (2 * 100 ** 2))  # P3
        + 0.4 * np.random.randn(len(t))
    )
    ax_top.plot(t, signal, color=BLUE_DARK, lw=1.8, zorder=3)
    ax_top.fill_between(t, signal - 0.8, signal + 0.8,
                        alpha=0.15, color=BLUE_MED, zorder=2)

    # Stimulus onset line
    ax_top.axvline(0, color="#C44040", lw=1.5, ls="--", zorder=4, alpha=0.7)
    ax_top.text(5, ax_top.get_ylim()[1] * 0.6, "Stimulus\nonset",
                fontsize=9, color="#C44040", va="bottom", ha="left")

    # Baseline region
    ax_top.axvspan(-100, 0, alpha=0.08, color=TEXT_MUTED, zorder=1)
    ax_top.text(-50, -5.5, "Baseline", ha="center", fontsize=8,
                color=TEXT_MUTED, style="italic")

    # Draw sliding windows
    window_size = 100  # ms
    step = 10  # ms
    highlight_windows = [
        (-100, ACCENT_WARM),
        (100, GREEN),
        (300, BLUE_MED),
        (500, ORANGE),
    ]

    y_bottom = -6.5
    y_height = 1.0

    # Show all windows as faint bars
    for w_start in range(-100, 700 - window_size + 1, step):
        rect = Rectangle((w_start, y_bottom), window_size, y_height,
                          facecolor=BLUE_LIGHT, edgecolor="none",
                          alpha=0.15, zorder=1)
        ax_top.add_patch(rect)

    # Highlight specific windows
    for w_start, color in highlight_windows:
        rect = Rectangle((w_start, y_bottom), window_size, y_height,
                          facecolor=color, edgecolor=BLUE_DARK,
                          alpha=0.6, lw=1.5, zorder=5)
        ax_top.add_patch(rect)
        # Bracket to signal
        ax_top.plot([w_start + window_size / 2, w_start + window_size / 2],
                    [y_bottom + y_height, signal[np.argmin(np.abs(t - (w_start + window_size / 2)))] - 0.5],
                    color=color, lw=1.0, ls=":", alpha=0.7, zorder=4)

    # Window labels
    ax_top.text(400, y_bottom - 0.8,
                f"Sliding window: {window_size} ms width, {step} ms step",
                ha="center", fontsize=10, color=TEXT_DARK, fontweight="bold")

    ax_top.set_xlim(-120, 720)
    ax_top.set_ylim(y_bottom - 1.2, 7)
    ax_top.set_xlabel("Time (ms)", fontsize=11, color=TEXT_DARK)
    ax_top.set_ylabel("Amplitude (a.u.)", fontsize=11, color=TEXT_DARK)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_color(TEXT_MUTED)
    ax_top.spines["left"].set_color(TEXT_MUTED)
    ax_top.tick_params(colors=TEXT_MUTED, labelsize=9)

    # ===== BOTTOM: RDM at each window =====
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.set_facecolor(BG)
    ax_bot.axis("off")

    ax_bot.text(0.5, 0.98, "At each window: pairwise LDA decoding produces a neural RDM",
                transform=ax_bot.transAxes, ha="center", va="top",
                fontsize=12, fontweight="bold", color=TEXT_DARK)

    # Draw 4 mini RDMs corresponding to highlighted windows
    rdm_size = 0.17
    rdm_gap = 0.07
    total_w = 4 * rdm_size + 3 * rdm_gap
    start_x = (1 - total_w) / 2

    window_labels = ["-100 to 0 ms", "100 to 200 ms", "300 to 400 ms", "500 to 600 ms"]
    window_colors = [ACCENT_WARM, GREEN, BLUE_MED, ORANGE]

    for i, (label, wc) in enumerate(zip(window_labels, window_colors)):
        x = start_x + i * (rdm_size + rdm_gap)
        y = 0.15

        # Generate a plausible RDM pattern (gets more structured over time)
        np.random.seed(i + 10)
        n = 6
        if i == 0:
            rdm = 50.0 + 2.0 * np.random.randn(n, n)
        elif i == 1:
            rdm = 50.0 + np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float) * 3
            rdm = rdm + 2.0 * np.random.randn(n, n)
        elif i == 2:
            rdm = 50.0 + np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float) * 2.5
            rdm[:4, :4] = rdm[:4, :4] + 1
            rdm[4:, 4:] = rdm[4:, 4:] + 1
            rdm[:4, 4:] = rdm[:4, 4:] + 4
            rdm[4:, :4] = rdm[4:, :4] + 4
            rdm = rdm + 1.0 * np.random.randn(n, n)
        else:
            rdm = 50.0 + np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float) * 2
            rdm[:4, :4] = rdm[:4, :4] + 0.5
            rdm[4:, 4:] = rdm[4:, 4:] + 0.5
            rdm[:4, 4:] = rdm[:4, 4:] + 3
            rdm[4:, :4] = rdm[4:, :4] + 3
            rdm = rdm + 1.5 * np.random.randn(n, n)

        np.fill_diagonal(rdm, 50)
        rdm = (rdm + rdm.T) / 2

        # Create inset axes for RDM
        inset = ax_bot.inset_axes([x, y, rdm_size, 0.6])
        inset.imshow(rdm, cmap="Blues", aspect="equal",
                     vmin=48, vmax=62, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(wc)
            spine.set_linewidth(2.5)

        # Label
        ax_bot.text(x + rdm_size / 2, y - 0.05, label,
                    transform=ax_bot.transAxes, ha="center", va="top",
                    fontsize=9, color=TEXT_DARK, fontweight="bold")

        # Arrow between RDMs
        if i < 3:
            arrow_x = x + rdm_size + rdm_gap * 0.15
            ax_bot.annotate(
                "", xy=(arrow_x + rdm_gap * 0.5, y + 0.3),
                xytext=(arrow_x, y + 0.3),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK, lw=1.5),
            )

    # Bottom annotation
    ax_bot.text(0.5, -0.08,
                "Each window yields one 24x24 or 6x6 neural RDM, "
                "then correlated with model RDMs (Spearman rho)",
                transform=ax_bot.transAxes, ha="center", va="bottom",
                fontsize=10, color=TEXT_MUTED, style="italic")

    # Title
    fig.suptitle("Time-Resolved RSA: Sliding Window Approach",
                 fontsize=18, fontweight="bold", color=TEXT_DARK, y=0.96)

    out = OUTPUT / "sliding_window_diagram.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
