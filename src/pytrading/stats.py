from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from vectorbt.portfolio.base import Portfolio as VectorbtPortfolio


def calendar_cagr(values: pd.Series, initial_capital: float) -> float:
    """Calculate CAGR using elapsed calendar days, matching experiment reports."""
    elapsed_years = max(
        (values.index[-1] - values.index[0]).days / 365.25,
        1 / 252,
    )
    return float((float(values.iloc[-1]) / initial_capital) ** (1 / elapsed_years) - 1)


def stats(portfolio: VectorbtPortfolio) -> pd.Series:
    """Return the five headline performance metrics for a Vectorbt portfolio."""
    portfolio_values = portfolio.value(group_by=True)
    if isinstance(portfolio_values, pd.DataFrame):
        if portfolio_values.shape[1] != 1:
            raise ValueError("Expected one portfolio group for CAGR.")
        portfolio_values = portfolio_values.iloc[:, 0]
    initial_cash = portfolio.init_cash
    if isinstance(initial_cash, pd.Series):
        if len(initial_cash) != 1:
            raise ValueError("Expected one initial cash value for CAGR.")
        initial_cash = initial_cash.iloc[0]
    daily_returns = portfolio_values.pct_change(fill_method=None).dropna()
    daily_volatility = daily_returns.std(ddof=1)
    sharpe = (
        252**0.5 * daily_returns.mean() / daily_volatility
        if daily_volatility and pd.notna(daily_volatility)
        else float("nan")
    )
    max_drawdown = portfolio_values.div(portfolio_values.cummax()).sub(1).min()
    cagr = calendar_cagr(portfolio_values, float(initial_cash))
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    metric_values = {
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Profit Factor": portfolio.trades.closed.profit_factor(group_by=True),
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
    }
    scalar_values = {}
    for name, value in metric_values.items():
        if isinstance(value, pd.Series):
            if len(value) != 1:
                raise ValueError(f"Expected one portfolio group for {name}.")
            value = value.iloc[0]
        scalar_values[name] = value
    return pd.Series(scalar_values, name="Value")
