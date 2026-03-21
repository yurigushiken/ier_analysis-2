"""Option 2: Network diagram showing valid prime-target transitions."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(4.0, 2.5))

# Place numerosities 1-6 in a horizontal line
node_y = 1.25
node_xs = {i: 0.5 + (i-1) * 0.7 for i in range(1, 7)}
node_r = 0.22

# Colors for PI/ANS categories
colors = {
    1: '#e8913a',  # Orange - "one"
    2: '#4a90d9',  # Blue - PI range
    3: '#4a90d9',
    4: '#4a90d9',
    5: '#5cb85c',  # Green - ANS range
    6: '#5cb85c',
}

# Draw edges first (behind nodes)
# Valid pairs: |prime - target| <= 3, prime != target
edge_colors_by_dist = {1: '#cccccc', 2: '#bbbbbb', 3: '#aaaaaa'}
for p in range(1, 7):
    for t in range(p+1, 7):  # Only draw each edge once
        if abs(p - t) <= 3:
            dist = abs(p - t)
            # Draw curved arc above or below
            x1, x2 = node_xs[p], node_xs[t]
            mid_x = (x1 + x2) / 2
            # Height of arc depends on distance
            arc_h = 0.15 + dist * 0.18
            # Alternate above/below for visual clarity
            if dist % 2 == 1:
                arc_y = node_y + arc_h
            else:
                arc_y = node_y - arc_h

            # Draw with bezier-like path
            verts = [(x1, node_y), (mid_x, arc_y), (x2, node_y)]
            from matplotlib.path import Path
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none',
                                       edgecolor=edge_colors_by_dist[dist],
                                       lw=0.8, alpha=0.7)
            ax.add_patch(patch)

# Draw nodes
for num in range(1, 7):
    circle = plt.Circle((node_xs[num], node_y), node_r,
                        facecolor=colors[num], edgecolor='white',
                        linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(node_xs[num], node_y, str(num), ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', zorder=6)

# Legend for categories
legend_y = 0.25
legend_items = [
    ('#e8913a', '"One"'),
    ('#4a90d9', 'PI range (2-4)'),
    ('#5cb85c', 'ANS range (5-6)'),
]
for i, (color, label) in enumerate(legend_items):
    x = 0.5 + i * 1.4
    circle = plt.Circle((x, legend_y), 0.1, facecolor=color,
                        edgecolor='white', linewidth=1, zorder=5)
    ax.add_patch(circle)
    ax.text(x + 0.18, legend_y, label, va='center', fontsize=7.5,
            color='#333333')

# Edge distance legend
ax.text(0.5, 2.35, 'Arcs = valid prime-target pairs (distance 1-3)',
        fontsize=7, color='#888888', style='italic')

# Count label
ax.text(4.0, 2.35, '24 conditions total',
        fontsize=7.5, fontweight='bold', color='#333333', ha='right')

ax.set_xlim(0, 4.2)
ax.set_ylim(-0.05, 2.55)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('option2_network_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved option2_network_diagram.png")
