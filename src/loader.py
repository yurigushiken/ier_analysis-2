"""Local CSV loading helpers for ier_analysis-2."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def load_frame_csvs(
    directories: Iterable[Path],
    *,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate frame-level CSV files from the provided directories."""
    paths = []
    for directory in directories:
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Data directory not found: {dir_path}")
        paths.extend(sorted(dir_path.glob("*.csv")))

    required = list(required_columns or [])
    if not paths:
        return pd.DataFrame(columns=required)

    frames = []
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        df["source_file"] = str(csv_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    missing = [column for column in required if column not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {paths[0].parent}")

    return combined


__all__ = ["load_frame_csvs"]

