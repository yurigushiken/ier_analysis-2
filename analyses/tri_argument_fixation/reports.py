"""Report writers for the tri-argument analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from matplotlib.backends.backend_pdf import PdfPages


def write_text_report(summary: pd.DataFrame, report_cfg: Dict, reports_dir: Path, *, filename_prefix: str) -> None:
    """Emit the plain-text report."""
    lines = [
        f"Research Question: {report_cfg.get('research_question', '')}",
        f"Hypothesis: {report_cfg.get('hypothesis', '')}",
        f"Prediction: {report_cfg.get('prediction', '')}",
        "",
        "Results:",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['cohort']}: {row['successful_trials']}/{row['total_trials']} "
            f"trials ({row['success_rate']*100:.1f}% coverage, "
            f"{int(row['participants'])} participants)"
        )
    report_path = reports_dir / f"{filename_prefix}_tri_argument_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(summary: pd.DataFrame, report_cfg: Dict, reports_dir: Path, figure_rel_path: Path, *, filename_prefix: str) -> None:
    """Emit the HTML report referencing the tri-argument chart."""
    table_html = summary.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tri-Argument Fixation Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ margin-bottom: 0; }}
    p {{ max-width: 800px; }}
    table {{ border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 1rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>Tri-Argument Fixation Analysis</h1>
  <p><strong>Research Question:</strong> {report_cfg.get("research_question", "")}</p>
  <p><strong>Hypothesis:</strong> {report_cfg.get("hypothesis", "")}</p>
  <p><strong>Prediction:</strong> {report_cfg.get("prediction", "")}</p>
  <h2>Results</h2>
  {table_html}
  <h2>Visualization</h2>
  <img src="../{figure_rel_path.as_posix()}" alt="Tri-argument coverage chart" width="600"/>
</body>
</html>
"""
    html_path = reports_dir / f"{filename_prefix}_tri_argument_report.html"
    html_path.write_text(html, encoding="utf-8")


def write_pdf_report(summary: pd.DataFrame, report_cfg: Dict, reports_dir: Path, figure_path: Path, *, filename_prefix: str) -> None:
    """Emit the PDF summary."""
    pdf_path = reports_dir / f"{filename_prefix}_tri_argument_report.pdf"
    text_lines = [
        "Tri-Argument Fixation Analysis",
        "",
        f"Research Question: {report_cfg.get('research_question', '')}",
        f"Hypothesis: {report_cfg.get('hypothesis', '')}",
        f"Prediction: {report_cfg.get('prediction', '')}",
        "",
        "Results:",
    ]
    for _, row in summary.iterrows():
        text_lines.append(
            f"- {row['cohort']}: {row['successful_trials']}/{row['total_trials']} trials "
            f"({row['success_rate']*100:.1f}%)"
        )

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        y = 0.95
        for line in text_lines:
            wrapped = textwrap.wrap(line, 80) or [""]
            for sub_line in wrapped:
                ax.text(0.05, y, sub_line, ha="left", va="top", fontsize=11)
                y -= 0.04
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
        img = plt.imread(figure_path)
        ax2.imshow(img)
        ax2.axis("off")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def write_event_structure_csv(summary: pd.DataFrame, tables_dir: Path, *, filename_prefix: str) -> Path:
    """Persist the event-structure breakdown summary."""
    output_path = tables_dir / f"{filename_prefix}_event_structure_breakdown.csv"
    summary.to_csv(output_path, index=False)
    return output_path

