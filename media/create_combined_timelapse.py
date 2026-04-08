"""Create a 6-row timelapse: original Gordon 4 rows + SHOW w/ + SHOW w/o."""

from PIL import Image, ImageDraw
from pathlib import Path

MEDIA = Path(r"D:\ier_analysis-2\media")
OUTPUT = MEDIA / "gordon2003_timelapse_with_show.png"

# Load original Gordon timelapse (4 rows: GIVE w/, GIVE w/o, HUG w/, HUG w/o)
original = Image.open(r"D:\ier_analysis-2\poster_space\poster\figures\gordon2003_timelapse.png")
orig_w, orig_h = original.size
print(f"Original timelapse: {orig_w}x{orig_h}")

# The original has 4 rows of 5 frames each
# Each row height is roughly orig_h / 4 = 96px
row_h = orig_h // 4
frame_w = orig_w // 5

# Load SHOW frames
show_frames = sorted((MEDIA / "frames-show").glob("*.png"))
show_without_frames = sorted((MEDIA / "frames-show-without").glob("*.png"))

print(f"SHOW frames: {len(show_frames)}, SHOW w/o frames: {len(show_without_frames)}")

# We need to match the style: blue border frames in a grid
# Each cell in the original is roughly frame_w x row_h with blue borders
border_color = (0, 0, 200)  # Blue similar to original
border_width = 3
padding = 2

# Target cell size (matching original grid)
cell_w = frame_w
cell_h = row_h

# Inner image area
inner_w = cell_w - 2 * (border_width + padding)
inner_h = cell_h - 2 * (border_width + padding)

def create_row(frame_paths, num_cells=5):
    """Create a single row of frames matching the original style."""
    row = Image.new("RGB", (orig_w, row_h), (220, 220, 230))  # Light gray background

    for i, path in enumerate(frame_paths[:num_cells]):
        img = Image.open(path).convert("RGB")
        # Resize to fit inner area while maintaining aspect
        img_ratio = img.width / img.height
        inner_ratio = inner_w / inner_h
        if img_ratio > inner_ratio:
            new_w = inner_w
            new_h = int(inner_w / img_ratio)
        else:
            new_h = inner_h
            new_w = int(inner_h * img_ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Create cell with blue border
        cell = Image.new("RGB", (cell_w, cell_h), (220, 220, 230))
        draw = ImageDraw.Draw(cell)

        # Blue border rectangle
        bx = padding
        by = padding
        bw = cell_w - 2 * padding
        bh = cell_h - 2 * padding
        draw.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=border_color, width=border_width)

        # Paste image centered within the border
        img_x = bx + border_width + (inner_w - new_w) // 2
        img_y = by + border_width + (inner_h - new_h) // 2
        cell.paste(img, (img_x, img_y))

        # Paste cell into row
        row.paste(cell, (i * cell_w, 0))

    return row

# Create two new rows
show_row = create_row(show_frames)
show_without_row = create_row(show_without_frames)

# Combine: original 4 rows + 2 new rows
combined_h = orig_h + 2 * row_h
combined = Image.new("RGB", (orig_w, combined_h), (220, 220, 230))
combined.paste(original, (0, 0))
combined.paste(show_row, (0, orig_h))
combined.paste(show_without_row, (0, orig_h + row_h))

combined.save(OUTPUT)
print(f"Saved combined timelapse: {OUTPUT} ({combined.size})")
