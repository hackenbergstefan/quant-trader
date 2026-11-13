from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from vectorbt.portfolio.base import Portfolio as VectorbtPortfolio


def stats(portfolio: VectorbtPortfolio) -> pd.Series:
    """Return the five headline performance metrics for a Vectorbt portfolio."""
    values = {
        "Sharpe Ratio": portfolio.sharpe_ratio(group_by=True),
        "Calmar Ratio": portfolio.calmar_ratio(group_by=True),
        "Profit Factor": portfolio.trades.closed.profit_factor(group_by=True),
        "CAGR": portfolio.annualized_return(group_by=True),
        "Max Drawdown": portfolio.max_drawdown(group_by=True),
    }
    scalar_values = {}
    for name, value in values.items():
        if isinstance(value, pd.Series):
            if len(value) != 1:
                raise ValueError(f"Expected one portfolio group for {name}.")
            value = value.iloc[0]
        scalar_values[name] = value
    return pd.Series(scalar_values, name="Value")
