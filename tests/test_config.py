"""Tests for the path layout.

These look trivial, and that is the point: they fail loudly the day someone
moves a file and silently breaks every path in the project.
"""

from __future__ import annotations

from powerforecast.config import PATHS, RANDOM_SEED


def test_project_root_contains_pyproject() -> None:
    # If this fails, PROJECT_ROOT is pointing at the wrong directory.
    assert (PATHS.root / "pyproject.toml").is_file()


def test_data_paths_are_nested_under_data() -> None:
    for path in (PATHS.raw, PATHS.interim, PATHS.processed):
        assert PATHS.data in path.parents


def test_create_is_idempotent(tmp_path) -> None:
    from powerforecast.config import Paths

    paths = Paths(
        root=tmp_path,
        data=tmp_path / "data",
        raw=tmp_path / "data" / "raw",
        interim=tmp_path / "data" / "interim",
        processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )
    paths.create()
    paths.create()  # second call must not raise

    assert paths.raw.is_dir()
    assert paths.models.is_dir()


def test_random_seed_is_an_int() -> None:
    assert isinstance(RANDOM_SEED, int)
