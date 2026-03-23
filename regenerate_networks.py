"""Regenerate network diagram figures for GIVE and SHOW conditions.

Uses the already-computed transition matrix CSVs and the updated visuals.py
to re-render only the network PNGs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from analyses.gaze_transition_analysis import visuals

COHORTS = [
    "7-month-olds",
    "8-month-olds",
    "9-month-olds",
    "10-month-olds",
    "11-month-olds",
    "Adults",
]
AOI_NODES = [
    "man_face",
    "woman_face",
    "toy_present",
    "toy_location",
    "man_body",
    "woman_body",
]

CONDITIONS = [
    {
        "prefix": "gw_transitions_min3_50_percent",
        "dir": PROJECT_ROOT
        / "analyses"
        / "gaze_transition_analysis"
        / "gw_transitions_min3_50_percent - Copy (7)",
    },
    {
        "prefix": "sw_transitions_min3_50_percent",
        "dir": PROJECT_ROOT
        / "analyses"
        / "gaze_transition_analysis"
        / "sw_transitions_min3_50_percent - Copy (3)",
    },
]

for cond in CONDITIONS:
    csv_path = cond["dir"] / "tables" / f"{cond['prefix']}_transition_matrix.csv"
    figures_dir = cond["dir"] / "figures"
    print(f"Loading: {csv_path}")
    matrix_df = pd.read_csv(csv_path)
    print(f"  Rows: {len(matrix_df)}, Cohorts: {matrix_df['cohort'].unique().tolist()}")
    print(f"  Output: {figures_dir}")
    visuals.plot_transition_networks(
        matrix_df,
        cohorts=COHORTS,
        aoi_nodes=AOI_NODES,
        figures_dir=figures_dir,
        filename_prefix=cond["prefix"],
    )
    print(f"  Done: {cond['prefix']}")

print("\nAll network diagrams regenerated.")
