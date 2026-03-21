"""Option 3: Compact visual table of conditions grouped by distance. Grayscale, 600 DPI."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(4.2, 2.4))

conditions_by_dist = {
    1: ['12','21','23','32','34','43','45','54','56','65'],
    2: ['13','31','24','42','35','53','46','64'],
    3: ['14','41','25','52','36','63'],
}

# Grayscale colors by distance (lighter = closer)
dist_colors = {1: '#b8b8b8', 2: '#787878', 3: '#404040'}
text_colors = {1: '#222222', 2: 'white', 3: 'white'}

y_positions = {1: 1.55, 2: 0.90, 3: 0.25}
cell_w = 0.28
cell_h = 0.38
gap = 0.045
label_w = 0.95  # space for labels on left

for dist, conds in conditions_by_dist.items():
    y = y_positions[dist]
    color = dist_colors[dist]
    txt_color = text_colors[dist]

    # Distance label - clearly to the left of cells
    ax.text(label_w - 0.08, y + cell_h/2, f'Distance {dist}',
            ha='right', va='center', fontsize=8, fontweight='bold',
            color='#333333')

    # Draw condition cells
    x_start = label_w + 0.02
    for i, cond in enumerate(conds):
        x = x_start + i * (cell_w + gap)
        rect = patches.FancyBboxPatch((x, y), cell_w, cell_h,
                                       boxstyle="round,pad=0.03",
                                       facecolor=color, edgecolor='white',
                                       linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + cell_w/2, y + cell_h/2, cond,
                ha='center', va='center', fontsize=7.5,
                color=txt_color, fontweight='bold')

    # Count after last cell
    x_end = x_start + len(conds) * (cell_w + gap)
    ax.text(x_end + 0.03, y + cell_h/2, f'({len(conds)})',
            ha='left', va='center', fontsize=7.5, color='#999999')

# Total at bottom
ax.text(label_w + 0.02, 0.0, 'Total: 10 + 8 + 6 = 24 conditions',
        fontsize=7.5, color='#666666', style='italic')

ax.set_xlim(-0.05, 4.6)
ax.set_ylim(-0.12, 2.1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('option3_compact_table.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved option3_compact_table.png")
