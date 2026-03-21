"""Generate a publication-ready analysis pipeline workflow diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "media"
OUTPUT.mkdir(exist_ok=True)

# -- Colors (matching PPT beige/blue theme) ----------------------------------
BG = "#F5F0E8"
BLUE_DARK = "#3E648C"
BLUE_MED = "#9CC4E4"
BLUE_LIGHT = "#D2E4F3"
TEXT_DARK = "#263242"
TEXT_MUTED = "#5A6E82"
ACCENT_WARM = "#E8C9A0"
WHITE = "#FFFFFF"


def rounded_box(ax, x, y, w, h, text, *, color=BLUE_LIGHT, text_color=TEXT_DARK,
                fontsize=11, bold=False, edge_color=BLUE_MED, lw=1.5,
                subtext=None, subsize=8.5):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edge_color, linewidth=lw,
        transform=ax.transData, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.02 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=3)
    if subtext:
        ax.text(x, y - 0.045, subtext,
                ha="center", va="center", fontsize=subsize,
                color=TEXT_MUTED, zorder=3, style="italic")


def arrow_down(ax, x, y1, y2, **kw):
    color = kw.pop("color", BLUE_DARK)
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, shrinkA=2, shrinkB=2),
                zorder=1)


def arrow_right(ax, x1, y, x2, **kw):
    color = kw.pop("color", BLUE_DARK)
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, shrinkA=2, shrinkB=2),
                zorder=1)


def bracket_label(ax, x, y1, y2, text, side="right", offset=0.08):
    """Draw a bracket with label."""
    xo = x + offset if side == "right" else x - offset
    ax.plot([x, xo, xo, x], [y1, y1, y2, y2],
            color=TEXT_MUTED, lw=1.2, zorder=1)
    ax.text(xo + 0.03, (y1 + y2) / 2, text,
            ha="left", va="center", fontsize=8, color=TEXT_MUTED)


def main():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # -- Title --
    ax.text(0.5, 1.00, "Analysis Pipeline", ha="center", va="top",
            fontsize=20, fontweight="bold", color=TEXT_DARK)
    ax.text(0.5, 0.965, "From EEG recording to representational similarity analysis",
            ha="center", va="top", fontsize=11, color=TEXT_MUTED)

    # ---- COLUMN 1: Data ----
    bw, bh = 0.18, 0.065

    # Row 1: Data collection
    rounded_box(ax, 0.13, 0.85, bw, bh, "EEG Recording",
                color=ACCENT_WARM, edge_color="#C4A070", bold=True,
                subtext="128-ch, 24 participants")

    arrow_down(ax, 0.13, 0.85 - bh / 2, 0.73 + bh / 2)

    # Row 2: Preprocessing
    rounded_box(ax, 0.13, 0.73, bw, bh, "HAPPE Preprocessing",
                subtext="1.5-35 Hz, wavelet artifact corr.")

    arrow_down(ax, 0.13, 0.73 - bh / 2, 0.61 + bh / 2)

    # Row 3: Epoching
    rounded_box(ax, 0.13, 0.61, bw, bh, "Epoch Extraction",
                subtext="-100 to 700 ms, baseline corr.")

    arrow_down(ax, 0.13, 0.61 - bh / 2, 0.49 + bh / 2)

    # Row 4: Trial structure
    rounded_box(ax, 0.13, 0.49, bw, bh, "Trial Labeling",
                subtext="24 prime-target conditions")

    # ---- COLUMN 2: Decoding ----
    arrow_right(ax, 0.13 + bw / 2, 0.49, 0.42 - bw / 2)

    rounded_box(ax, 0.42, 0.49, bw + 0.02, bh, "Pairwise LDA Decoding",
                color=BLUE_MED, edge_color=BLUE_DARK, bold=True,
                subtext="GroupKFold CV, Ledoit-Wolf")

    arrow_down(ax, 0.42, 0.49 - bh / 2, 0.37 + bh / 2)

    rounded_box(ax, 0.42, 0.37, bw + 0.02, bh, "Subject-Level Accuracy",
                subtext="Balanced acc. per pair per subject")

    arrow_down(ax, 0.42, 0.37 - bh / 2, 0.25 + bh / 2)

    rounded_box(ax, 0.42, 0.25, bw + 0.02, bh, "Neural RDM Construction",
                color=BLUE_MED, edge_color=BLUE_DARK,
                subtext="Accuracy -> dissimilarity matrix")

    # ---- COLUMN 3: RSA split ----
    # Branch left: Whole-epoch
    arrow_down(ax, 0.42, 0.25 - bh / 2, 0.11 + bh / 2 + 0.015)

    # Main RSA box
    rounded_box(ax, 0.42, 0.12, bw + 0.02, bh + 0.01,
                "RSA: Model-Brain Comparison",
                color=BLUE_DARK, edge_color="#263242", text_color=WHITE,
                bold=True, subtext="Spearman corr. per subject",
                subsize=8.5)

    # ---- Right column: Two branches ----
    # Branch right arrows
    bx_left = 0.22
    bx_right = 0.62

    # Whole-epoch branch
    ax.annotate("", xy=(bx_left + 0.02, 0.12),
                xytext=(0.42 - (bw + 0.02) / 2, 0.12),
                arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK,
                                lw=1.5, shrinkA=0, shrinkB=2),
                zorder=1)

    rounded_box(ax, 0.12, 0.12, bw - 0.02, bh - 0.005,
                "Whole-Epoch", color=BLUE_LIGHT, bold=True, fontsize=10,
                subtext="0-700 ms collapsed")

    # Temporal branch
    ax.annotate("", xy=(bx_right + 0.08, 0.12),
                xytext=(0.42 + (bw + 0.02) / 2, 0.12),
                arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK,
                                lw=1.5, shrinkA=0, shrinkB=2),
                zorder=1)

    rounded_box(ax, 0.78, 0.12, bw, bh - 0.005,
                "Time-Resolved", color=BLUE_LIGHT, bold=True, fontsize=10,
                subtext="Sliding windows, -100 to 700 ms")

    # ---- Far right: Model families ----
    model_x = 0.88
    model_bw = 0.20
    model_bh = 0.055
    models = [
        ("Target PI (boundary 4-5 or 3-4)", BLUE_LIGHT),
        ("Prime x Target PI", BLUE_LIGHT),
        ("ANS (log-ratio)", BLUE_LIGHT),
        ("Pixel (visual control)", "#F0EDE6"),
        ("RT (task-difficulty control)", "#F0EDE6"),
    ]
    my = 0.85
    ax.text(model_x, 0.92, "Model RDMs", ha="center", fontsize=13,
            fontweight="bold", color=TEXT_DARK)
    for label, col in models:
        rounded_box(ax, model_x, my, model_bw, model_bh, label,
                    color=col, fontsize=9, edge_color=BLUE_MED, lw=1.0)
        my -= 0.075

    # Arrow from model family to RSA box
    ax.annotate("", xy=(0.42 + (bw + 0.02) / 2 + 0.02, 0.12 + (bh + 0.01) / 2),
                xytext=(model_x - model_bw / 2, my + 0.075 - model_bh / 2),
                arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK,
                                lw=1.5, connectionstyle="arc3,rad=-0.3"),
                zorder=1)

    # ---- Bottom: Inference ----
    inf_y = 0.02
    inf_items = [
        (0.12, "t-tests + Holm\n(whole-epoch)"),
        (0.42, "Cluster permutation\n(temporal fits)"),
        (0.72, "Fisher-z model\ndifferences"),
    ]
    for ix, label in inf_items:
        rounded_box(ax, ix, inf_y, bw, bh - 0.005, label,
                    color="#F0EDE6", edge_color="#C4A070", fontsize=9,
                    lw=1.0)

    ax.text(0.42, inf_y - 0.047, "Statistical Inference",
            ha="center", fontsize=11, fontweight="bold", color=TEXT_MUTED)

    plt.tight_layout(pad=0.3)
    out = OUTPUT / "workflow_diagram.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
