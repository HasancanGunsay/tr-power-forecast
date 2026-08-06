"""Ask where the best model still fails.

    uv run python -m powerforecast.analysis.diagnose

Reads the predictions saved by `analysis.experiment`, so it runs in seconds and
can be re-asked freely. If the file is missing, run the experiment first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from powerforecast.analysis.errors import (
    compare_on_worst,
    error_by,
    error_concentration,
    error_frame,
    worst_days,
)
from powerforecast.config import PATHS

BEST = "lightgbm + weather + hourly bias"
REFERENCE = "plan + hourly bias"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m powerforecast.analysis.experiment` first"
        )
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PATHS.reports / "predictions.parquet")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    actual = predictions["actual"]

    models = [c for c in predictions.columns if c != "actual"]
    frames = {name: error_frame(actual, predictions[name]) for name in models}
    best = frames[BEST]

    print("=" * 76)
    print(f"WHERE {BEST!r} STILL FAILS   ({len(best):,} hours)")
    print("=" * 76)

    print("\n[1] Is the error concentrated, or spread evenly?")
    print(error_concentration(best).to_string(float_format=lambda v: f"{v:,.1f}"))
    print("  A tie on RMSE beside a win on MAE means the tail is doing the damage.")

    print(f"\n[2] Worst {args.top} delivery days")
    print(
        worst_days(best, top=args.top).to_string(
            float_format=lambda v: f"{v:,.0f}",
            columns=["mae", "mape_%", "bias", "max_abs", "weekday"],
        )
    )

    print("\n[3] Error by month")
    print(error_by(best, "month").to_string(float_format=lambda v: f"{v:,.0f}"))

    print("\n[4] Error by weekday  (0 = Monday)")
    print(error_by(best, "dayofweek").to_string(float_format=lambda v: f"{v:,.0f}"))

    # Both directions, because each model's own worst hours are a different set.
    # Asking only one way answers only half the question — and it happens to be
    # the half that flatters whoever picked the reference.
    for reference in (REFERENCE, BEST):
        print(f"\n[5] On the 1% of hours {reference!r} finds hardest:")
        print(
            compare_on_worst(frames, reference=reference, quantile=0.99).to_string(
                float_format=lambda v: f"{v:,.0f}"
            )
        )


if __name__ == "__main__":
    main()
