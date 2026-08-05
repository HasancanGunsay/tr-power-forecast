"""Project-wide paths and settings.

Every path in the project is derived from here. Nothing else in the codebase
should ever hardcode a directory, because hardcoded paths are the number one
reason a repository runs on the author's laptop and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class Settings(BaseSettings):
    """Credentials and tunables, read from the environment or `.env`.

    Secrets are typed as `SecretStr`, which keeps them out of logs and
    tracebacks: printing the object shows `**********` rather than the value.
    Retrieving the real string requires an explicit `.get_secret_value()`, so
    leaking one has to be a deliberate act rather than an accident.

    Every credential defaults to `None` instead of being required, because the
    test suite and CI must import this module without any secrets present.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # EPİAŞ Şeffaflık (Türkiye)
    epias_username: str | None = None
    epias_password: SecretStr | None = None

    # ENTSO-E Transparency (Europe)
    entsoe_api_key: SecretStr | None = None
    entsoe_bidding_zone: str = "10Y1001A1001A83F"  # Germany/Luxembourg

    random_seed: int = 42

    def require_epias(self) -> tuple[str, str]:
        """Return EPİAŞ credentials, failing loudly when they are absent.

        Callers get a clear message naming the missing variables instead of an
        opaque 401 from the API an hour into a pipeline run.
        """
        if self.epias_username is None or self.epias_password is None:
            raise MissingCredentialsError(
                "EPIAS_USERNAME and EPIAS_PASSWORD must be set in the environment "
                "or in .env (copy .env.example to get started)."
            )
        return self.epias_username, self.epias_password.get_secret_value()


class MissingCredentialsError(RuntimeError):
    """Raised when an operation needs credentials that were never configured."""


def get_settings() -> Settings:
    """Build settings from the current environment.

    Deliberately not cached: tests override environment variables between
    cases, and a module-level singleton would freeze the first value read.
    """
    return Settings()
