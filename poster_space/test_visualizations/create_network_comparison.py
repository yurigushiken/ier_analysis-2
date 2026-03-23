"""
Regenerate the network comparison triptych for GIVE condition.

Combines 7-month-olds, 11-month-olds, and Adults network diagrams
into a single side-by-side figure with title and 2% threshold note.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

FIGURES_DIR = Path(r"D:\ier_analysis-2\analyses\gaze_transition_analysis"
                   r"\gw_transitions_min3_50_percent - Copy (7)\figures")

PANELS = [
    ("7-month-olds", FIGURES_DIR / "gw_transitions_min3_50_percent_network_7-month-olds.png"),
    ("11-month-olds", FIGURES_DIR / "gw_transitions_min3_50_percent_network_11-month-olds.png"),
    ("Adults", FIGURES_DIR / "gw_transitions_min3_50_percent_network_adults.png"),
]

OUTPUT = Path(r"D:\ier_analysis-2\poster_space\test_visualizations\network_comparison_give.png")


def main():
    # Load panel images
    images = []
    for label, path in PANELS:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        images.append(Image.open(path))

    # Resize all to same height
    target_h = min(img.height for img in images)
    resized = []
    for img in images:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    # Layout parameters
    gap = 30
    title_h = 80
    footer_h = 40
    total_w = sum(img.width for img in resized) + gap * (len(resized) - 1) + 60  # 30px margin each side
    total_h = target_h + title_h + footer_h

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to get a reasonable font (fallback to default)
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        note_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
            note_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            title_font = ImageFont.load_default()
            note_font = ImageFont.load_default()

    # Title
    title = "GIVE: Developmental Shift in Gaze Networks"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 20), title, fill="black", font=title_font)

    # Paste panels
    x_offset = 30  # left margin
    for img in resized:
        canvas.paste(img, (x_offset, title_h))
        x_offset += img.width + gap

    # Footer / threshold note
    note = "Note: Only transitions \u2265 2% shown"
    bbox = draw.textbbox((0, 0), note, font=note_font)
    nw = bbox[2] - bbox[0]
    draw.text(((total_w - nw) // 2, title_h + target_h + 8), note, fill="gray", font=note_font)

    canvas.save(OUTPUT, dpi=(300, 300))
    print(f"Saved: {OUTPUT}")
    print(f"  Size: {total_w} x {total_h} px")


if __name__ == "__main__":
    main()
