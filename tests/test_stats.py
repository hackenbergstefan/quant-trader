import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from pytrading.stats import stats


def test_stats_cagr_uses_elapsed_calendar_time_like_experiment_reports():
    dates = pd.bdate_range("2024-01-02", periods=253)
    close = pd.Series(
        100
        * np.exp(
            0.0005 * np.arange(len(dates)) + 0.01 * np.sin(np.arange(len(dates)) / 8)
        ),
        index=dates,
        name="A",
    )
    portfolio = vbt.Portfolio.from_holding(close, init_cash=100.0, freq="1D")

    elapsed_years = (dates[-1] - dates[0]).days / 365.25
    values = portfolio.value()
    expected_cagr = (float(values.iloc[-1]) / 100.0) ** (1 / elapsed_years) - 1
    daily_returns = values.pct_change(fill_method=None).dropna()
    expected_sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1)
    expected_drawdown = values.div(values.cummax()).sub(1).min()

    metrics = stats(portfolio)

    assert metrics["CAGR"] == pytest.approx(expected_cagr)
    assert metrics["CAGR"] != pytest.approx(
        float(portfolio.annualized_return(group_by=True))
    )
    assert metrics["Sharpe Ratio"] == pytest.approx(expected_sharpe)
    assert metrics["Sharpe Ratio"] != pytest.approx(
        float(portfolio.sharpe_ratio(group_by=True))
    )
    assert metrics["Calmar Ratio"] == pytest.approx(
        expected_cagr / abs(expected_drawdown)
    )
    assert metrics["Max Drawdown"] == pytest.approx(expected_drawdown)
