from __future__ import annotations

import pandas as pd


def apply_allocation_method(signals: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = signals.copy()
    if method not in {"volatility_adjusted", "risk_budget"}:
        return frame
    for _, indices in frame.loc[frame["selected"]].groupby("timestamp", sort=False).groups.items():
        day = frame.loc[indices]
        if method == "volatility_adjusted":
            volatility = day["volatility_20"].where(day["volatility_20"] > 0)
            inverse = 1 / volatility
        else:
            relative_stop_distance = (day["atr"] / day["feature_close"]).where(
                (day["atr"] > 0) & (day["feature_close"] > 0)
            )
            inverse = 1 / relative_stop_distance
        if inverse.notna().any():
            total_exposure = day["target_weight"].sum()
            normalized = inverse.fillna(0) / inverse.sum()
            frame.loc[indices, "target_weight"] = normalized * total_exposure
    return frame
