"""Tests for reproducibility helpers."""

from __future__ import annotations

import random

from powerforecast.seed import set_seed


def test_same_seed_gives_same_sequence() -> None:
    set_seed(123)
    first = [random.random() for _ in range(5)]

    set_seed(123)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_different_seed_gives_different_sequence() -> None:
    set_seed(1)
    first = [random.random() for _ in range(5)]

    set_seed(2)
    second = [random.random() for _ in range(5)]

    assert first != second


def test_returns_the_applied_seed() -> None:
    assert set_seed(7) == 7


def test_works_without_optional_dependencies() -> None:
    # NumPy/PyTorch are optional here; the call must not raise either way.
    assert set_seed() > 0
