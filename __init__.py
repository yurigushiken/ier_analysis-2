"""Convenience exports for the ier_analysis-2 tooling."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src import EXTENSION_CONFIG, run_cli  # pragma: no cover
else:
    from .src import EXTENSION_CONFIG, run_cli  # pragma: no cover

__all__ = ["EXTENSION_CONFIG", "run_cli"]

