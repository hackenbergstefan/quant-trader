from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from functools import partial
from itertools import product
from math import ceil
from multiprocessing import Pool, cpu_count
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import vectorbt as vbt

if TYPE_CHECKING:
    from pytrading import BrokerConfig, Portfolio


@dataclass(frozen=True)
class MomentumConfig:
    topn: int = 10
    min_avg_daily_turnover: float = 50_000.0
    liquidity_lookback: int = 20
    momentum_20_weight: float = 1 / 3
    momentum_60_weight: float = 1 / 3
    momentum_120_weight: float = 1 / 3

    def __post_init__(self) -> None:
        weights = np.array(
            [
                self.momentum_20_weight,
                self.momentum_60_weight,
                self.momentum_120_weight,
            ]
        )
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "Momentum-Gewichte müssen endlich, nicht-negativ und > 0 sein."
            )


def lookback_prices(
    close: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    prices = close[(close.index >= start_date) & (close.index <= end_date)]
    prices = prices.dropna(axis=0, how="all")
    return prices.dropna(axis=1, how="all")


def month_end(prices: pd.DataFrame) -> pd.Index:
    return prices.groupby(prices.index.to_period("M")).tail(1).index


def _trend_filter(close: pd.DataFrame) -> pd.DataFrame:
    sma_50 = close.rolling(window=50, min_periods=50).mean()
    sma_200 = close.rolling(window=200, min_periods=200).mean()
    return (close > sma_50) & (sma_50 > sma_200)


def _momentum_scores(close: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    returns = [close.pct_change(period) for period in (20, 60, 120)]
    ranks = [period_returns.rank(axis=1, pct=True) for period_returns in returns]
    weights = (
        config.momentum_20_weight,
        config.momentum_60_weight,
        config.momentum_120_weight,
    )
    total_weight = sum(weights)
    weighted_score = (
        sum(rank.mul(weight) for rank, weight in zip(ranks, weights, strict=True))
        / total_weight
    )
    complete_history = returns[0].notna() & returns[1].notna() & returns[2].notna()
    return weighted_score.where(complete_history)


def _execution_events(close: pd.DataFrame, prices: pd.DataFrame):
    first_date = prices.index[0]
    previous_dates = close.index[close.index < first_date]
    events = []
    if len(previous_dates) > 0:
        events.append((first_date, previous_dates[-1]))

    for signal_date in month_end(prices):
        position = prices.index.get_loc(signal_date)
        if position + 1 >= len(prices.index):
            continue
        execution_date = prices.index[position + 1]
        if execution_date >= first_date:
            events.append((execution_date, signal_date))
    return events


def _portfolio_weights(
    close: pd.DataFrame,
    prices: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    turnover: pd.DataFrame | None,
    config: MomentumConfig,
) -> pd.DataFrame:
    weights = pd.DataFrame(
        np.nan,
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )
    for execution_date, signal_date in _execution_events(close, prices):
        scores = momentum_scores.loc[signal_date].dropna()
        scores = scores.sort_values(ascending=False)
        execution_prices = close.loc[execution_date].reindex(weights.columns)
        valid_prices = execution_prices[
            np.isfinite(execution_prices) & (execution_prices > 0)
        ]
        candidates = valid_prices.reindex(scores.index).dropna()
        if turnover is not None:
            history = close.index[close.index <= signal_date][
                -config.liquidity_lookback :
            ]
            recent_turnover = turnover.reindex(index=history, columns=weights.columns)
            enough_observations = recent_turnover.count() >= max(
                1, int(config.liquidity_lookback * 0.75)
            )
            liquid = recent_turnover.mean() >= config.min_avg_daily_turnover
            eligible = enough_observations & liquid
            candidates = candidates[eligible.reindex(candidates.index).fillna(False)]
        top = candidates.head(config.topn).index.tolist()
        if len(top) < config.topn:
            continue
        weights.loc[execution_date, valid_prices.index] = 0.0
        weights.loc[execution_date, top] = 1.0 / len(top)
    return weights


def _build_vectorbt_portfolio(
    prices: pd.DataFrame, weights: pd.DataFrame, broker: BrokerConfig
):
    return vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        direction="longonly",
        init_cash=broker.initial_cash,
        fees=broker.fees,
        fixed_fees=broker.fixed_fees,
        slippage=broker.slippage,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        allow_partial=True,
        freq="1D",
        log=True,
    )


def backtest(portfolio: Portfolio, momentum: MomentumConfig):
    close = portfolio.universe.close()
    turnover = portfolio.universe.turnover()
    return _backtest(
        close,
        turnover,
        portfolio.broker,
        portfolio.start_date,
        portfolio.end_date,
        momentum,
    )


def _backtest(
    close: pd.DataFrame,
    turnover: pd.DataFrame | None,
    broker: BrokerConfig,
    start_date: str,
    end_date: str,
    momentum: MomentumConfig,
):
    prices = lookback_prices(close, start_date, end_date)
    if turnover is not None and turnover.empty:
        raise ValueError("Keine historischen Umsatzdaten gefunden; Daten neu laden.")

    momentum_scores = _momentum_scores(close, momentum).where(_trend_filter(close))
    weights = _portfolio_weights(close, prices, momentum_scores, turnover, momentum)
    return _build_vectorbt_portfolio(prices, weights, broker)


def _run_sweep_case(
    close: pd.DataFrame,
    turnover: pd.DataFrame | None,
    broker: BrokerConfig,
    task: tuple[int, dict[str, object]],
) -> dict[str, object]:
    start_year, parameters = task
    config = MomentumConfig(**parameters)
    end_year = start_year + 3
    result = _backtest(
        close,
        turnover,
        broker,
        f"{start_year}-01-01",
        f"{end_year}-01-01",
        config,
    )
    from pytrading.stats import stats as calculate_stats

    return {
        "period": f"{start_year}-{end_year}",
        **asdict(config),
        **calculate_stats(result).to_dict(),
    }


def parameter_sweep(
    portfolio: Portfolio, parameter_ranges: Mapping[str, Iterable[object]]
) -> pd.DataFrame:
    """Return per-config min, max, and average stats across 3-year periods from 2015."""
    allowed_parameters = {field.name for field in fields(MomentumConfig)}
    unknown_parameters = set(parameter_ranges) - allowed_parameters
    if unknown_parameters:
        raise ValueError(
            "Unbekannte Momentum-Parameter: " + ", ".join(sorted(unknown_parameters))
        )

    parameter_names = tuple(parameter_ranges)
    parameter_values = [tuple(parameter_ranges[name]) for name in parameter_names]
    empty_ranges = [
        name for name, values in zip(parameter_names, parameter_values) if not values
    ]
    if empty_ranges:
        raise ValueError(
            "Parameterranges dürfen nicht leer sein: " + ", ".join(empty_ranges)
        )

    close = portfolio.universe.close()
    if close.empty:
        raise ValueError("Das Universe enthält keine historischen Kurse.")
    last_date = min(close.index.max(), pd.Timestamp(portfolio.end_date))
    turnover = portfolio.universe.turnover()
    periods = []
    start_year = 2015
    while pd.Timestamp(start_year + 3, 1, 1) <= last_date:
        periods.append(start_year)
        start_year += 1

    tasks = [
        (period_start, dict(zip(parameter_names, values, strict=True)))
        for period_start in periods
        for values in product(*parameter_values)
    ]
    if not tasks:
        return pd.DataFrame()

    processes = min(len(tasks), cpu_count() or 1)
    chunksize = ceil(len(tasks) / processes)
    worker = partial(_run_sweep_case, close, turnover, portfolio.broker)
    with Pool(processes=processes) as pool:
        rows = pool.map(worker, tasks, chunksize=chunksize)

    results = pd.DataFrame(rows)
    parameter_columns = [field.name for field in fields(MomentumConfig)]
    metric_columns = [
        column
        for column in results.columns
        if column not in {*parameter_columns, "period"}
    ]
    summary = results.groupby(parameter_columns, dropna=False)[metric_columns].agg(
        ["min", "max", "mean"]
    )
    summary.columns = [
        f"{metric}_{'avg' if aggregation == 'mean' else aggregation}"
        for metric, aggregation in summary.columns
    ]
    return summary.reset_index()
