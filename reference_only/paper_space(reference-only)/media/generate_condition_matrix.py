"""Generate a condition matrix figure showing all 24 prime-target conditions."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(3.2, 3.2))

primes = range(1, 7)
targets = range(1, 7)

# Colors
valid_color = '#4a90d9'      # Blue for valid conditions
diagonal_color = '#e0e0e0'   # Light gray for no-change (diagonal)
invalid_color = '#f5f5f5'    # Very light gray for non-existent

for p_idx, prime in enumerate(primes):
    for t_idx, target in enumerate(targets):
        x = t_idx
        y = 5 - p_idx  # Flip so prime 1 is at top

        if prime == target:
            # Diagonal: no-change trials (not analyzed)
            color = diagonal_color
            ax.add_patch(patches.Rectangle((x, y), 1, 1,
                         facecolor=color, edgecolor='white', linewidth=1.5))
        elif abs(prime - target) <= 3:
            # Valid condition
            ax.add_patch(patches.Rectangle((x, y), 1, 1,
                         facecolor=valid_color, edgecolor='white', linewidth=1.5))
            ax.text(x + 0.5, y + 0.5, f'{prime}{target}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white')
        else:
            # Not in design
            ax.add_patch(patches.Rectangle((x, y), 1, 1,
                         facecolor=invalid_color, edgecolor='white', linewidth=1.5))

# Axis labels
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_xticks([i + 0.5 for i in range(6)])
ax.set_xticklabels([str(i) for i in range(1, 7)], fontsize=10)
ax.set_yticks([i + 0.5 for i in range(6)])
ax.set_yticklabels([str(i) for i in range(6, 0, -1)], fontsize=10)
ax.set_xlabel('Target numerosity', fontsize=11, labelpad=6)
ax.set_ylabel('Prime numerosity', fontsize=11, labelpad=6)
ax.set_aspect('equal')
ax.tick_params(length=0)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('condition_matrix.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved condition_matrix.png")
