from __future__ import annotations

import pandas as pd


def calculate_momentum(
    prices: pd.DataFrame,
    periods: tuple[int, int, int] = (20, 60, 120),
    weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> pd.DataFrame:
    frame = prices.sort_values(["instrument_id", "timestamp"]).copy()
    price_column = "feature_close" if "feature_close" in frame else "close"
    grouped = frame.groupby("instrument_id", sort=False)[price_column]
    names = ("momentum_20", "momentum_60", "momentum_120")
    for period, name in zip(periods, names, strict=True):
        frame[name] = grouped.transform(lambda values: values.div(values.shift(period)).sub(1))
    frame["momentum_score"] = sum(
        frame[name] * weight for name, weight in zip(names, weights, strict=True)
    )
    return frame[["instrument_id", "timestamp", *names, "momentum_score"]]
