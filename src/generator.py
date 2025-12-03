"""CLI entry point for the independent gaze fixation generator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .config import EXTENSION_CONFIG
from .gaze_detector import detect_fixations
from .loader import load_frame_csvs

MIN_ONSCREEN_FRAMES = 30


def generate_for_thresholds(
    thresholds: Iterable[int] | None = None,
    *,
    child_dirs: Optional[Sequence[Path]] = None,
    adult_dirs: Optional[Sequence[Path]] = None,
    output_root: Path | None = None,
    min_onscreen_frames: int = MIN_ONSCREEN_FRAMES,
    dir_suffix: str = "",
    exclude_screen_nonroi: bool = False,
) -> None:
    """Generate gaze-fixation CSVs for the requested thresholds."""
    resolved_thresholds = list(thresholds or EXTENSION_CONFIG.thresholds)
    if not resolved_thresholds:
        raise ValueError("At least one threshold must be provided.")

    child_sources = list(child_dirs) if child_dirs else [EXTENSION_CONFIG.raw_child_dir]
    adult_sources = list(adult_dirs) if adult_dirs else [EXTENSION_CONFIG.raw_adult_dir]
    output_root = (output_root or EXTENSION_CONFIG.output_root).resolve()

    child_frames = _filter_trials_with_screen_time(
        load_frame_csvs(child_sources, required_columns=EXTENSION_CONFIG.required_columns),
        min_onscreen_frames,
    )
    adult_frames = _filter_trials_with_screen_time(
        load_frame_csvs(adult_sources, required_columns=EXTENSION_CONFIG.required_columns),
        min_onscreen_frames,
    )

    for threshold in resolved_thresholds:
        _generate_single_threshold(
            threshold=threshold,
            child_frames=child_frames,
            adult_frames=adult_frames,
            output_root=output_root,
            dir_suffix=dir_suffix,
            exclude_screen_nonroi=exclude_screen_nonroi,
        )


def _generate_single_threshold(
    *,
    threshold: int,
    child_frames: pd.DataFrame,
    adult_frames: pd.DataFrame,
    output_root: Path,
    dir_suffix: str = "",
    exclude_screen_nonroi: bool = False,
) -> None:
    suffix = dir_suffix or ""
    min_dir = output_root / f"min{threshold}{suffix}"
    min_dir.mkdir(parents=True, exist_ok=True)

    child_fix = _assign_cohort(
        detect_fixations(child_frames, min_frames=threshold),
        cohort="child",
    )
    adult_fix = _assign_cohort(
        detect_fixations(adult_frames, min_frames=threshold),
        cohort="adult",
    )

    if exclude_screen_nonroi:
        child_fix = _remove_screen_nonroi(child_fix)
        adult_fix = _remove_screen_nonroi(adult_fix)

    combined = pd.concat([child_fix, adult_fix], ignore_index=True)

    _write_output(child_fix, min_dir / f"gaze_fixations_child_min{threshold}.csv")
    _write_output(adult_fix, min_dir / f"gaze_fixations_adult_min{threshold}.csv")
    _write_output(combined, min_dir / f"gaze_fixations_combined_min{threshold}.csv")


def _assign_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    working = df.copy()
    working["cohort"] = cohort
    return working


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _remove_screen_nonroi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "aoi_category" not in df.columns:
        return df
    mask = df["aoi_category"] != "screen_nonAOI"
    return df[mask].copy()


def _filter_trials_with_screen_time(df: pd.DataFrame, min_frames: int) -> pd.DataFrame:
    if df.empty or "What" not in df.columns or "Where" not in df.columns:
        return df

    what = df["What"].astype(str).str.lower()
    where = df["Where"].astype(str).str.lower()
    offscreen_mask = (what == "no") & (where == "signal")
    on_screen = ~offscreen_mask
    key_cols = []
    for candidate in ("Participant", "trial_number_global", "trial_number"):
        if candidate in df.columns:
            key_cols.append(candidate)
            if candidate == "trial_number":
                break
        else:
            return df

    if "event_verified" in df.columns:
        key_cols.append("event_verified")
    elif "condition" in df.columns:
        key_cols.append("condition")

    if not key_cols:
        return df

    counts = (
        pd.DataFrame({col: df[col] for col in key_cols} | {"on_screen": on_screen.astype(int)})
        .groupby(key_cols)["on_screen"]
        .sum()
    )
    valid_keys = counts[counts >= min_frames].index
    mask = df.set_index(key_cols).index.isin(valid_keys)
    return df[mask].copy()


def run_cli() -> None:
    """Entry point for `python -m src.generator` style usage."""
    parser = argparse.ArgumentParser(description="Generate gaze-fixation CSVs for multiple thresholds.")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=EXTENSION_CONFIG.thresholds,
        help="Minimum frame thresholds for fixation detection (default: %(default)s).",
    )
    parser.add_argument(
        "--child-dir",
        dest="child_dirs",
        action="append",
        help="Path to a child cohort CSV directory. Can be provided multiple times.",
    )
    parser.add_argument(
        "--adult-dir",
        dest="adult_dirs",
        action="append",
        help="Path to an adult cohort CSV directory. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXTENSION_CONFIG.output_root,
        help="Directory where threshold-specific outputs will be written.",
    )
    parser.add_argument(
        "--min-onscreen-frames",
        type=int,
        default=MIN_ONSCREEN_FRAMES,
        help="Minimum on-screen frames required to keep a trial (default: %(default)s).",
    )
    parser.add_argument(
        "--dir-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the min-directory names (e.g., '-50_percent').",
    )
    parser.add_argument(
        "--exclude-screen-nonroi",
        action="store_true",
        help="Exclude 'screen_nonAOI' fixations from CSV outputs while still counting them toward on-screen time.",
    )
    args = parser.parse_args()

    generate_for_thresholds(
        args.thresholds,
        child_dirs=[Path(p) for p in args.child_dirs] if args.child_dirs else None,
        adult_dirs=[Path(p) for p in args.adult_dirs] if args.adult_dirs else None,
        output_root=args.output_root,
        min_onscreen_frames=args.min_onscreen_frames,
        dir_suffix=args.dir_suffix,
        exclude_screen_nonroi=args.exclude_screen_nonroi,
    )


if __name__ == "__main__":  # pragma: no cover
    run_cli()


__all__ = ["generate_for_thresholds", "run_cli"]

