"""Export each slide of the PPTX as a JPG using PowerPoint COM automation."""
import os
import sys
from pathlib import Path

def main():
    pptx_path = Path(__file__).parent / "pi_boundary_manuscript_presentation.pptx"
    out_dir = Path(__file__).parent / "slide_previews"
    out_dir.mkdir(exist_ok=True)

    if not pptx_path.exists():
        print(f"PPTX not found: {pptx_path}")
        sys.exit(1)

    import comtypes.client
    import time
    # Start PowerPoint
    ppt = comtypes.client.CreateObject("PowerPoint.Application")
    time.sleep(2)

    abs_path = str(pptx_path.resolve())
    presentation = ppt.Presentations.Open(abs_path, WithWindow=False)

    for i, slide in enumerate(presentation.Slides, 1):
        out_path = str((out_dir / f"slide_{i:02d}.jpg").resolve())
        slide.Export(out_path, "JPG", 1920, 1080)
        print(f"  Exported slide {i} -> {out_path}")

    presentation.Close()
    try:
        ppt.Quit()
    except Exception:
        pass
    print(f"Done. {i} slides exported.")

if __name__ == "__main__":
    main()
