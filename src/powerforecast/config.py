"""Project-wide paths and settings.

Every path in the project is derived from here. Nothing else in the codebase
should ever hardcode a directory, because hardcoded paths are the number one
reason a repository runs on the author's laptop and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Anchor: this file is at <root>/src/powerforecast/config.py, so the project root
# is three levels up. Resolving from __file__ keeps it correct no matter which
# directory the process was started from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical data layout.

    The raw -> interim -> processed split is a convention worth keeping:
    `raw` is immutable (never written to after download), `interim` holds
    intermediate artifacts, `processed` holds model-ready tables. When a
    result looks wrong, this layout lets you re-run any single stage instead
    of the whole pipeline.
    """

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    interim: Path = PROJECT_ROOT / "data" / "interim"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    reports: Path = PROJECT_ROOT / "reports"

    def create(self) -> None:
        """Create every directory. Safe to call repeatedly."""
        for field_name in ("data", "raw", "interim", "processed", "models", "reports"):
            path: Path = getattr(self, field_name)
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# Single source of truth for the random seed. Read from the environment so a
# CI job or a sweep can override it without editing code.
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))
