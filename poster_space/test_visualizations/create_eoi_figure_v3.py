"""
Create a publication-quality EOI (Elements of Interest) paradigm figure - V3.
Corrected overlay positions: v2 had all overlays shifted too far LEFT.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import numpy as np

# Load the source image
img_path = r"D:\ier_analysis-2\presentation\media\vlcsnap-2026-03-21-14h23m36s754.png"
img = Image.open(img_path)
img_array = np.array(img)
img_w, img_h = img.size
print(f"Image dimensions: {img_w} x {img_h}")

# Create figure - 8x6 inches at 300 DPI
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)

# Display the image
ax.imshow(img_array, aspect='auto', extent=[0, img_w, img_h, 0])
ax.set_xlim(0, img_w)
ax.set_ylim(img_h, 0)
ax.set_axis_off()

# --- Define EOI regions (x, y, width, height) --- CORRECTED for v3
# All shifted RIGHT relative to v2 to match actual element positions
regions = {
    "Man's Face":   {"rect": (118, 20, 100, 115),    "color": "#E03030"},
    "Woman's Face": {"rect": (425, 40, 105, 115),    "color": "#3070E0"},
    "Toy":          {"rect": (248, 145, 115, 155),    "color": "#20A040"},
    "Man's Body":   {"rect": (90, 125, 195, 315),    "color": "#E08020"},
    "Woman's Body": {"rect": (375, 140, 195, 300),    "color": "#9040C0"},
}

# Draw region outlines with rounded corners
for name, info in regions.items():
    x, y, w, h = info["rect"]
    color = info["color"]

    # Draw a rounded rectangle border (no fill)
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=3",
        linewidth=2.5,
        edgecolor=color,
        facecolor='none',
        zorder=3,
    )
    ax.add_patch(rect)

    # Also draw a subtle semi-transparent fill
    rect_fill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=3",
        linewidth=0,
        edgecolor='none',
        facecolor=color,
        alpha=0.08,
        zorder=2,
    )
    ax.add_patch(rect_fill)

# --- Draw labels with colored backgrounds and connector arrows ---
def add_label(ax, text, label_x, label_y, target_x, target_y, color, ha='left', rad=0.15):
    """Add a label with colored background and an arrow to the target region."""
    ax.annotate(
        text,
        xy=(target_x, target_y),
        xytext=(label_x, label_y),
        fontsize=9,
        fontweight='bold',
        fontfamily='sans-serif',
        color='white',
        ha=ha,
        va='center',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor=color,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=1.8,
            connectionstyle=f'arc3,rad={rad}',
            mutation_scale=12,
        ),
        zorder=10,
    )

# Man's Face label - top left corner, arrow to face region center
rx, ry, rw, rh = regions["Man's Face"]["rect"]
add_label(ax, "Man's Face", 20, 20, rx + rw/2, ry + rh/2, "#E03030", ha='left')

# Woman's Face label - top right corner, arrow to face region center
rx, ry, rw, rh = regions["Woman's Face"]["rect"]
add_label(ax, "Woman's Face", img_w - 20, 30, rx + rw/2, ry + rh/2, "#3070E0", ha='right')

# Toy label - left of the toy rectangle, arrow to toy center
rx, ry, rw, rh = regions["Toy"]["rect"]
add_label(ax, "Toy", 155, 170, rx + rw*0.3, ry + rh*0.35, "#20A040", ha='right')

# Man's Body label - bottom left, arrow to body region
rx, ry, rw, rh = regions["Man's Body"]["rect"]
add_label(ax, "Man's Body", 20, img_h - 35, rx + rw/2, ry + rh*0.6, "#E08020", ha='left')

# Woman's Body label - bottom right, arrow to body region
rx, ry, rw, rh = regions["Woman's Body"]["rect"]
add_label(ax, "Woman's Body", img_w - 20, img_h - 35, rx + rw/2, ry + rh*0.6, "#9040C0", ha='right')

# --- Gaze Point annotation ---
gaze_x, gaze_y = 310, 275
# Draw a small dashed circle around the gaze point
gaze_circle = plt.Circle((gaze_x, gaze_y), 12, fill=False, edgecolor='#00BFFF', linewidth=2, linestyle='--', zorder=5)
ax.add_patch(gaze_circle)

# Gaze point label - to the right, arrow pointing to gaze circle
add_label(ax, "Gaze Point", img_w - 20, 210, gaze_x + 12, gaze_y, '#00BFFF', ha='right', rad=0.0)

# --- Title ---
fig.suptitle(
    "Elements of Interest (EOI) Coding",
    fontsize=14,
    fontweight='bold',
    fontfamily='sans-serif',
    y=0.96,
    color='#222222',
)

# --- Subtitle at bottom ---
fig.text(
    0.5, 0.02,
    "Computer-vision segmented regions tracked frame-by-frame as objects",
    ha='center',
    va='bottom',
    fontsize=10,
    fontstyle='italic',
    fontfamily='sans-serif',
    color='#555555',
)

# Save
output_path = r"D:\ier_analysis-2\poster_space\test_visualizations\eoi_paradigm_figure_v3.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Figure saved to: {output_path}")
