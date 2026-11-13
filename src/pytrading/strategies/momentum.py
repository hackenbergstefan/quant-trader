from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import lru_cache, partial
from hashlib import sha256
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
    rebalance_frequency: str = "monthly"
    max_position_weight: float = 0.2
    position_sizing: str = "inverse_volatility"
    bullish_exposure: float = 1.0
    neutral_exposure: float = 0.5
    bearish_exposure: float = 0.0

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
        if self.topn < 1:
            raise ValueError("Top-N muss mindestens 1 sein.")
        if self.rebalance_frequency not in {"weekly", "monthly"}:
            raise ValueError("Rebalance-Frequenz muss 'weekly' oder 'monthly' sein.")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("Maximales Positionsgewicht muss in (0, 1] liegen.")
        if self.position_sizing not in {"equal", "inverse_volatility"}:
            raise ValueError(
                "Positionsgewichtung muss 'equal' oder 'inverse_volatility' sein."
            )
        exposures = (
            self.bullish_exposure,
            self.neutral_exposure,
            self.bearish_exposure,
        )
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in exposures):
            raise ValueError("Regionale Exposure-Grenzen müssen in [0, 1] liegen.")


class StrategyVariant(StrEnum):
    A = "A"
    AB = "AB"
    ABC = "ABC"
    ABD = "ABD"
    ABCD = "ABCD"


VARIANT_SIGNALS = {
    StrategyVariant.A: frozenset("A"),
    StrategyVariant.AB: frozenset("AB"),
    StrategyVariant.ABC: frozenset("ABC"),
    StrategyVariant.ABD: frozenset("ABD"),
    StrategyVariant.ABCD: frozenset("ABCD"),
}


def lookback_prices(
    close: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    prices = close[(close.index >= start_date) & (close.index <= end_date)]
    prices = prices.dropna(axis=0, how="all")
    return prices.dropna(axis=1, how="all")


def month_end(prices: pd.DataFrame) -> pd.Index:
    return prices.groupby(prices.index.to_period("M")).tail(1).index


def rebalance_dates(prices: pd.DataFrame, frequency: str) -> pd.Index:
    if frequency == "monthly":
        periods = prices.index.to_period("M")
    elif frequency == "weekly":
        periods = prices.index.to_period("W-FRI")
    else:
        raise ValueError("Rebalance-Frequenz muss 'weekly' oder 'monthly' sein.")
    return prices.groupby(periods).tail(1).index


def _instrument_close_series(
    close: pd.DataFrame,
    column: str,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
) -> pd.Series:
    exchange = (metadata or {}).get(column, {}).get("exchange")
    calendar_name = _EXCHANGE_CALENDARS.get(exchange or "")
    if not calendar_name:
        return close[column]

    calendar = _get_calendar(calendar_name)
    first_date = close.index.min().normalize()
    last_date = close.index.max().normalize()
    sessions = calendar.sessions_in_range(first_date, last_date)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    return close[column].reindex(sessions)


def _align_signal_to_dates(
    values: pd.Series,
    close: pd.DataFrame,
    column: str,
    metadata: Mapping[str, Mapping[str, str | None]] | None,
) -> pd.Series:
    exchange = (metadata or {}).get(column, {}).get("exchange")
    if _EXCHANGE_CALENDARS.get(exchange or ""):
        return values.reindex(
            close.index, method="ffill", tolerance=pd.Timedelta(days=5)
        )
    return values.reindex(close.index)


def _observed_returns(
    close: pd.DataFrame,
    period: int,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
) -> pd.DataFrame:
    columns = {}
    for column in close.columns:
        series = _instrument_close_series(close, column, metadata)
        returns = series.pct_change(period, fill_method=None)
        complete_window = (
            series.notna()
            .rolling(period + 1, min_periods=period + 1)
            .sum()
            .eq(period + 1)
        )
        eligible_returns = returns.where(complete_window)
        columns[column] = _align_signal_to_dates(
            eligible_returns, close, column, metadata
        )
    return pd.DataFrame(columns, index=close.index)


def _trend_filter(
    close: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
) -> pd.DataFrame:
    eligible = {}
    for column in close.columns:
        series = _instrument_close_series(close, column, metadata)
        sma_50 = series.rolling(window=50, min_periods=50).mean()
        sma_200 = series.rolling(window=200, min_periods=200).mean()
        observed_eligibility = (series > sma_50) & (sma_50 > sma_200)
        eligible[column] = (
            _align_signal_to_dates(observed_eligibility, close, column, metadata)
            .fillna(False)
            .astype(bool)
        )
    return pd.DataFrame(eligible, index=close.index, dtype=bool)


def _momentum_scores(
    close: pd.DataFrame,
    config: MomentumConfig,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    returns = [_observed_returns(close, period, metadata) for period in (20, 60, 120)]
    complete_history = returns[0].notna() & returns[1].notna() & returns[2].notna()
    if eligibility is not None:
        complete_history &= eligibility.reindex_like(close).fillna(False)
    eligible_returns = [
        period_returns.where(complete_history) for period_returns in returns
    ]
    ranks = [
        period_returns.rank(axis=1, pct=True, method="average")
        for period_returns in eligible_returns
    ]
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
    return weighted_score.where(complete_history)


def _realized_volatility(
    close: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
) -> pd.DataFrame:
    volatility = {}
    for column in close.columns:
        series = _instrument_close_series(close, column, metadata)
        daily_returns = series.pct_change(fill_method=None)
        rv20 = daily_returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
        rv60 = daily_returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252)
        volatility[column] = _align_signal_to_dates(
            0.5 * rv20 + 0.5 * rv60, close, column, metadata
        )
    return pd.DataFrame(volatility, index=close.index)


def _atr_14(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in close.columns:
        observed_close = close[column].dropna()
        observed_high = high[column].reindex(observed_close.index)
        observed_low = low[column].reindex(observed_close.index)
        previous_close = observed_close.shift(1)
        true_range = pd.concat(
            [
                observed_high - observed_low,
                (observed_high - previous_close).abs(),
                (observed_low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result[column] = true_range.rolling(14, min_periods=14).mean()
    return pd.DataFrame(result).reindex_like(close)


def _capped_weights(
    raw_weights: pd.Series, max_position_weight: float, max_exposure: float = 1.0
) -> pd.Series:
    weights = pd.Series(0.0, index=raw_weights.index, dtype=float)
    valid = raw_weights[np.isfinite(raw_weights) & (raw_weights > 0)]
    if valid.empty or max_exposure <= 0:
        return weights

    target_exposure = min(max_exposure, max_position_weight * len(valid))
    remaining = target_exposure
    available = valid.copy()
    while not available.empty and remaining > 0:
        proposed = available / available.sum() * remaining
        capped = proposed[proposed > max_position_weight]
        if capped.empty:
            weights.loc[proposed.index] = proposed
            break
        weights.loc[capped.index] = max_position_weight
        remaining -= max_position_weight * len(capped)
        available = available.drop(index=capped.index)
    return weights


def _inverse_volatility_weights(
    volatility: pd.Series, max_position_weight: float = 0.2
) -> pd.Series:
    inverse_volatility = 1 / volatility.replace(0, np.nan)
    inverse_volatility = inverse_volatility.replace([np.inf, -np.inf], np.nan).dropna()
    return _capped_weights(inverse_volatility, max_position_weight)


def _liquidity_eligibility(
    close: pd.DataFrame,
    turnover: pd.DataFrame | None,
    config: MomentumConfig,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
) -> pd.DataFrame | None:
    if turnover is None:
        return None
    minimum_observations = max(1, int(config.liquidity_lookback * 0.75))
    eligible = pd.DataFrame(False, index=close.index, columns=close.columns)
    for column in close.columns:
        if column not in turnover.columns:
            continue
        observed_dates = close.index[close[column].notna()]
        recent_turnover = turnover[column].reindex(observed_dates)
        window = recent_turnover.rolling(config.liquidity_lookback, min_periods=1)
        observed_eligibility = window.count().ge(
            minimum_observations
        ) & window.mean().ge(config.min_avg_daily_turnover)
        eligible[column] = _align_signal_to_dates(
            pd.Series(observed_eligibility.to_numpy(), index=observed_dates),
            close,
            column,
            metadata,
        )
        eligible[column] = eligible[column].fillna(False).astype(bool)
    return eligible


def _market_regimes(
    benchmarks: Mapping[str, pd.Series], dates: pd.Index
) -> pd.DataFrame:
    regimes = {}
    for region in ("EUROPE", "USA"):
        benchmark = benchmarks.get(region)
        if benchmark is None:
            regimes[region] = pd.Series("unknown", index=dates, dtype="object")
            continue
        close = benchmark.sort_index()
        if close.empty:
            regimes[region] = pd.Series("unknown", index=dates, dtype="object")
            continue
        sma_50 = close.rolling(50, min_periods=50).mean()
        sma_200 = close.rolling(200, min_periods=200).mean()
        values = pd.Series("unknown", index=close.index, dtype="object")
        values.loc[close.le(sma_200)] = "bearish"
        above_200 = close.gt(sma_200)
        values.loc[above_200 & sma_50.gt(sma_200)] = "bullish"
        values.loc[above_200 & sma_50.le(sma_200)] = "neutral"
        aligned = values.reindex(dates, method="ffill").fillna("unknown")
        aligned.loc[aligned.index > close.index[-1]] = "unknown"
        regimes[region] = aligned
    return pd.DataFrame(regimes, index=dates)


_EXCHANGE_CALENDARS = {
    "NYQ": "XNYS",
    "NMS": "XNAS",
    "PNK": "XNYS",
    "LSE": "XLON",
    "ISE": "XLON",
    "PAR": "XPAR",
    "GER": "XETR",
    "FRA": "XFRA",
    "EBS": "XSWX",
    "STO": "XSTO",
    "MIL": "XMIL",
    "AMS": "XAMS",
    "MCE": "XMAD",
    "CPH": "XCSE",
    "OSL": "XOSL",
    "WSE": "XWAR",
    "BRU": "XBRU",
    "HEL": "XHEL",
    "VIE": "XWBO",
    "LIS": "XLIS",
}
_EUROPEAN_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom",
}


def _region_for_country(country: str | None) -> str:
    if country == "United States":
        return "USA"
    if country in _EUROPEAN_COUNTRIES:
        return "EUROPE"
    return "OTHER"


def _next_execution_date(
    close: pd.DataFrame,
    column: str,
    signal_date: pd.Timestamp,
    calendar_name: str | None,
    observations: pd.Index | None = None,
) -> pd.Timestamp | None:
    observations = (
        observations if observations is not None else close[column].dropna().index
    )
    next_position = observations.searchsorted(signal_date, side="right")
    future_observations = observations[next_position:]
    if future_observations.empty:
        return None
    if calendar_name is None:
        return future_observations[0]

    calendar = _get_calendar(calendar_name)
    signal_day = pd.Timestamp(signal_date).normalize()
    session_day = _strict_next_session(calendar_name, signal_day)
    eligible_observations = future_observations[future_observations >= session_day]
    if eligible_observations.empty:
        return None
    execution_date = pd.Timestamp(eligible_observations[0]).normalize()
    if not calendar.is_session(execution_date):
        raise ValueError(
            f"Execution date {execution_date.date()} is not a {calendar_name} session."
        )
    return execution_date


@lru_cache(maxsize=32)
def _get_calendar(calendar_name: str):
    import exchange_calendars

    return exchange_calendars.get_calendar(calendar_name)


@lru_cache(maxsize=4096)
def _strict_next_session(calendar_name: str, signal_day: pd.Timestamp) -> pd.Timestamp:
    calendar = _get_calendar(calendar_name)
    session = calendar.date_to_session(signal_day, direction="next")
    session_day = pd.Timestamp(session).tz_localize(None).normalize()
    if session_day <= signal_day:
        session = calendar.next_session(session)
        session_day = pd.Timestamp(session).tz_localize(None).normalize()
    return session_day


def _execution_events(
    close: pd.DataFrame, prices: pd.DataFrame, frequency: str = "monthly"
):
    first_date = prices.index[0]
    previous_dates = close.index[close.index < first_date]
    events = []
    if len(previous_dates) > 0:
        events.append((first_date, previous_dates[-1]))

    for signal_date in rebalance_dates(prices, frequency):
        position = prices.index.get_loc(signal_date)
        if position + 1 >= len(prices.index):
            continue
        execution_date = prices.index[position + 1]
        if execution_date >= first_date:
            events.append((execution_date, signal_date))
    return events


def _asset_execution_events(
    close: pd.DataFrame,
    prices: pd.DataFrame,
    config: MomentumConfig,
    metadata: Mapping[str, Mapping[str, str | None]],
    execution_prices: pd.DataFrame | None = None,
) -> dict[tuple[pd.Timestamp, pd.Timestamp], list[str]]:
    first_date = prices.index[0]
    last_date = prices.index[-1]
    available_history = close.loc[close.index <= last_date]
    signal_dates = list(rebalance_dates(available_history, config.rebalance_frequency))
    previous_dates = close.index[close.index < first_date]
    if len(previous_dates):
        signal_dates.insert(0, previous_dates[-1])

    scheduled: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
    execution_prices = close if execution_prices is None else execution_prices
    observation_dates = {
        column: execution_prices.index[
            np.isfinite(execution_prices[column]) & execution_prices[column].gt(0)
        ]
        for column in close.columns
    }
    for signal_date in dict.fromkeys(signal_dates):
        for column in close.columns:
            exchange = (metadata.get(column) or {}).get("exchange")
            calendar_name = _EXCHANGE_CALENDARS.get(exchange or "")
            execution_date = _next_execution_date(
                close,
                column,
                pd.Timestamp(signal_date),
                calendar_name,
                observation_dates[column],
            )
            if (
                execution_date is None
                or execution_date not in prices.index
                or execution_date < first_date
            ):
                continue
            key = (pd.Timestamp(execution_date), pd.Timestamp(signal_date))
            scheduled.setdefault(key, []).append(column)
    return scheduled


def _portfolio_weights(
    close: pd.DataFrame,
    prices: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    turnover: pd.DataFrame | None,
    config: MomentumConfig,
    *,
    signals: frozenset[str] = frozenset("AB"),
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    benchmarks: Mapping[str, pd.Series] | None = None,
    volatility: pd.DataFrame | None = None,
    execution_prices: pd.DataFrame | None = None,
    initial_weights: pd.Series | None = None,
) -> pd.DataFrame:
    weights = pd.DataFrame(
        np.nan,
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )
    metadata = metadata or {}
    execution_prices = close if execution_prices is None else execution_prices
    regions = {
        column: _region_for_country((metadata.get(column) or {}).get("country"))
        for column in close.columns
    }
    benchmarks = benchmarks or {}
    regimes = _market_regimes(benchmarks, close.index)
    events = _asset_execution_events(close, prices, config, metadata, execution_prices)
    liquid = _liquidity_eligibility(close, turnover, config, metadata)
    execution_by_signal: dict[pd.Timestamp, dict[str, pd.Timestamp]] = {}
    for (execution_date, signal_date), columns in events.items():
        execution_by_signal.setdefault(signal_date, {}).update(
            {column: execution_date for column in columns}
        )

    held_targets = pd.Series(0.0, index=close.columns)
    if initial_weights is not None:
        held_targets = initial_weights.reindex(close.columns).fillna(0.0).astype(float)
    pending_orders: dict[pd.Timestamp, dict[str, float]] = {}
    execution_observations = {
        column: execution_prices.index[
            np.isfinite(execution_prices[column]) & execution_prices[column].gt(0)
        ]
        for column in close.columns
    }

    for signal_date in sorted(execution_by_signal):
        for pending_date in sorted(
            date for date in pending_orders if date <= signal_date
        ):
            for column, target in pending_orders.pop(pending_date).items():
                if pending_date in weights.index:
                    weights.loc[pending_date, column] = target
                    held_targets.loc[column] = target

        scores = momentum_scores.loc[signal_date].dropna()
        candidates = scores.sort_values(ascending=False, kind="mergesort")
        if liquid is not None:
            candidates = candidates[
                liquid.loc[signal_date].reindex(candidates.index).fillna(False)
            ]

        if "C" in signals and volatility is not None:
            vol_at_signal = volatility.loc[signal_date]
            valid_volatility = vol_at_signal.reindex(candidates.index).notna()
            if config.position_sizing == "inverse_volatility":
                valid_volatility &= vol_at_signal.reindex(candidates.index).gt(0)
            candidates = candidates[valid_volatility]
        selected = candidates.head(config.topn).index
        target_weights = pd.Series(0.0, index=close.columns)
        if not selected.empty:
            if "C" in signals and config.position_sizing == "inverse_volatility":
                selected_volatility = volatility.loc[signal_date].reindex(selected)
                selected_weights = _inverse_volatility_weights(
                    selected_volatility, config.max_position_weight
                )
            elif "C" in signals:
                selected_weights = _capped_weights(
                    pd.Series(1.0, index=selected), config.max_position_weight
                )
            else:
                selected_weights = pd.Series(
                    1 / len(selected), index=selected, dtype=float
                )
            target_weights.loc[selected_weights.index] = selected_weights

        if "D" in signals:
            regime_row = regimes.loc[signal_date]
            exposure_limits = {
                "bullish": config.bullish_exposure,
                "neutral": config.neutral_exposure,
                "bearish": config.bearish_exposure,
            }
            for region in {regions[column] for column in target_weights.index}:
                region_columns = [
                    column
                    for column in target_weights.index
                    if regions[column] == region
                ]
                regime = regime_row.get(region, "unknown")
                regional_limit = exposure_limits.get(regime, 0.0)
                regional_weight = target_weights.reindex(region_columns).sum()
                if regional_weight > regional_limit and regional_weight > 0:
                    target_weights.loc[region_columns] *= (
                        regional_limit / regional_weight
                    )

        retained_orders: dict[str, tuple[pd.Timestamp, float]] = {}
        for pending_date in sorted(pending_orders):
            orders = pending_orders[pending_date]
            for column, target in list(orders.items()):
                desired = float(target_weights.loc[column])
                already_held = np.isclose(desired, held_targets.loc[column])
                if not np.isclose(target, desired) or already_held:
                    del orders[column]
                    continue
                if column not in retained_orders:
                    retained_orders[column] = (pending_date, target)
                else:
                    del orders[column]
            if not orders:
                del pending_orders[pending_date]

        execution_dates = execution_by_signal[signal_date]
        reductions = [
            column
            for column in close.columns
            if target_weights.loc[column] < held_targets.loc[column] - 1e-12
        ]
        blocked_by_unscheduled_reduction = any(
            column not in execution_dates and column not in retained_orders
            for column in reductions
        )
        reduction_dates = [
            retained_orders[column][0]
            if column in retained_orders
            else execution_dates[column]
            for column in reductions
            if column in execution_dates or column in retained_orders
        ]
        latest_reduction_date = max(reduction_dates, default=None)
        for column in reductions:
            if column in retained_orders:
                continue
            execution_date = execution_dates.get(column)
            if execution_date is not None:
                pending_orders.setdefault(execution_date, {})[column] = float(
                    target_weights.loc[column]
                )

        for column in close.columns:
            if target_weights.loc[column] <= held_targets.loc[column] + 1e-12:
                continue
            if blocked_by_unscheduled_reduction:
                if column in retained_orders:
                    pending_date, _ = retained_orders[column]
                    del pending_orders[pending_date][column]
                    if not pending_orders[pending_date]:
                        del pending_orders[pending_date]
                continue
            execution_date = (
                retained_orders[column][0]
                if column in retained_orders
                else execution_dates.get(column)
            )
            if execution_date is None:
                continue
            if (
                latest_reduction_date is not None
                and execution_date <= latest_reduction_date
            ):
                if column in retained_orders:
                    del pending_orders[execution_date][column]
                    if not pending_orders[execution_date]:
                        del pending_orders[execution_date]
                exchange = (metadata.get(column) or {}).get("exchange")
                calendar_name = _EXCHANGE_CALENDARS.get(exchange or "")
                execution_date = _next_execution_date(
                    close,
                    column,
                    latest_reduction_date,
                    calendar_name,
                    execution_observations[column],
                )
            if execution_date in prices.index:
                pending_orders.setdefault(execution_date, {})[column] = float(
                    target_weights.loc[column]
                )

    for execution_date in sorted(pending_orders):
        for column, target in pending_orders[execution_date].items():
            if execution_date in weights.index:
                weights.loc[execution_date, column] = target
    return weights


def _build_vectorbt_portfolio(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    weights: pd.DataFrame,
    broker: BrokerConfig,
):
    return vbt.Portfolio.from_orders(
        close=close_prices,
        price=open_prices,
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


def backtest(
    portfolio: Portfolio,
    momentum: MomentumConfig,
    variant: StrategyVariant | str | None = None,
):
    close = portfolio.universe.close()
    open_prices = (
        portfolio.universe.open() if hasattr(portfolio.universe, "open") else close
    )
    turnover = portfolio.universe.turnover()
    metadata = (
        portfolio.universe.market_metadata()
        if hasattr(portfolio.universe, "market_metadata")
        else {}
    )
    return _backtest(
        close,
        turnover,
        portfolio.broker,
        portfolio.start_date,
        portfolio.end_date,
        momentum,
        open_prices=open_prices,
        variant=variant,
        metadata=metadata,
        benchmarks=portfolio.benchmarks,
    )


def run_variant(
    portfolio: Portfolio,
    variant: StrategyVariant | str,
    config: MomentumConfig | None = None,
):
    variant = StrategyVariant(variant)
    if config is None:
        config = _default_variant_config()
    return backtest(portfolio, config, variant=variant)


def _backtest(
    close: pd.DataFrame,
    turnover: pd.DataFrame | None,
    broker: BrokerConfig,
    start_date: str,
    end_date: str,
    momentum: MomentumConfig,
    *,
    open_prices: pd.DataFrame | None = None,
    variant: StrategyVariant | str | None = None,
    metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    benchmarks: Mapping[str, pd.Series] | None = None,
):
    close = close.loc[:, ~close.columns.duplicated(keep="first")]
    open_prices = close if open_prices is None else open_prices
    open_prices = open_prices.loc[:, ~open_prices.columns.duplicated(keep="first")]
    if turnover is not None:
        turnover = turnover.loc[:, ~turnover.columns.duplicated(keep="first")]
    prices = lookback_prices(close, start_date, end_date)
    execution_prices = open_prices.reindex(index=close.index, columns=close.columns)
    if turnover is not None and turnover.empty:
        raise ValueError("Keine historischen Umsatzdaten gefunden; Daten neu laden.")

    variant = StrategyVariant(variant or StrategyVariant.AB)
    signals = VARIANT_SIGNALS[variant]
    trend_eligibility = _trend_filter(close, metadata) if "B" in signals else None
    momentum_scores = _momentum_scores(
        close, momentum, metadata, eligibility=trend_eligibility
    )
    volatility = _realized_volatility(close, metadata) if "C" in signals else None
    weights = _portfolio_weights(
        close,
        prices,
        momentum_scores,
        turnover,
        momentum,
        signals=signals,
        metadata=metadata,
        benchmarks=benchmarks,
        volatility=volatility,
        execution_prices=execution_prices,
    )
    execution_prices = execution_prices.reindex(
        index=prices.index, columns=prices.columns
    )
    return _build_vectorbt_portfolio(prices, execution_prices, weights, broker)


def _default_variant_config() -> MomentumConfig:
    return MomentumConfig(
        topn=5,
        momentum_20_weight=0.2,
        momentum_60_weight=0.3,
        momentum_120_weight=0.5,
        rebalance_frequency="weekly",
    )


def signal_diagnostics(
    portfolio: Portfolio,
    variant: StrategyVariant | str,
    config: MomentumConfig | None = None,
) -> pd.DataFrame:
    variant = StrategyVariant(variant)
    config = config or _default_variant_config()
    signals = VARIANT_SIGNALS[variant]
    close = portfolio.universe.close().loc[:, lambda frame: ~frame.columns.duplicated()]
    turnover = portfolio.universe.turnover()
    open_prices = (
        portfolio.universe.open()
        if hasattr(portfolio.universe, "open")
        else portfolio.universe.close()
    )
    metadata = portfolio.universe.market_metadata()
    prices = lookback_prices(close, portfolio.start_date, portfolio.end_date)
    trend = _trend_filter(close, metadata)
    trend_eligibility = trend if "B" in signals else None
    scores = _momentum_scores(close, config, metadata, eligibility=trend_eligibility)
    volatility = (
        _realized_volatility(close, metadata)
        if "C" in signals
        else pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    )
    benchmarks = portfolio.benchmarks
    regimes = _market_regimes(benchmarks, close.index)
    position_weights = _portfolio_weights(
        close,
        prices,
        scores,
        turnover,
        config,
        signals=signals,
        metadata=metadata,
        benchmarks=benchmarks,
        volatility=volatility if "C" in signals else None,
        execution_prices=open_prices,
    )
    liquid = _liquidity_eligibility(close, turnover, config, metadata)
    event_lookup = {
        (signal_date, column): execution_date
        for (execution_date, signal_date), columns in _asset_execution_events(
            close, prices, config, metadata, open_prices
        ).items()
        for column in columns
    }
    period_returns = [
        _observed_returns(close, period, metadata) for period in (20, 60, 120)
    ]
    complete = (
        period_returns[0].notna()
        & period_returns[1].notna()
        & period_returns[2].notna()
    )
    if "B" in signals:
        complete &= trend
    raw_returns = [values.where(complete) for values in period_returns]
    period_ranks = [
        values.rank(axis=1, pct=True, method="average") for values in raw_returns
    ]
    combined_ranks = scores.rank(axis=1, ascending=False, method="min")
    signal_dates = rebalance_dates(prices, config.rebalance_frequency)
    rows = []
    exposure_limits = {
        "bullish": config.bullish_exposure,
        "neutral": config.neutral_exposure,
        "bearish": config.bearish_exposure,
    }
    for signal_date in signal_dates:
        signal_candidates = scores.loc[signal_date].dropna()
        if "C" in signals:
            signal_volatility = volatility.loc[signal_date]
            valid_volatility = signal_volatility.reindex(
                signal_candidates.index
            ).notna()
            if config.position_sizing == "inverse_volatility":
                valid_volatility &= signal_volatility.reindex(
                    signal_candidates.index
                ).gt(0)
            signal_candidates = signal_candidates[valid_volatility]
        if liquid is not None:
            signal_candidates = signal_candidates[
                liquid.loc[signal_date].reindex(signal_candidates.index).fillna(False)
            ]
        selected_names = set(
            signal_candidates.sort_values(ascending=False, kind="mergesort")
            .head(config.topn)
            .index
        )
        for column in close.columns:
            region = _region_for_country((metadata.get(column) or {}).get("country"))
            regime = regimes.at[signal_date, region] if region in regimes else "unknown"
            exclusion_reasons = []
            if not complete.at[signal_date, column]:
                exclusion_reasons.append("incomplete_momentum_history")
            elif "B" in signals and not trend.at[signal_date, column]:
                exclusion_reasons.append("trend_filter")
            elif "C" in signals and not np.isfinite(volatility.at[signal_date, column]):
                exclusion_reasons.append("insufficient_volatility_history")
            if liquid is not None and not liquid.at[signal_date, column]:
                exclusion_reasons.append("liquidity_filter")
            if "D" in signals:
                limit = (
                    exposure_limits.get(regime, 0.0)
                    if region in {"EUROPE", "USA"}
                    else 0.0
                )
                if regime == "unknown":
                    exclusion_reasons.append("unknown_regional_regime")
                elif limit == 0:
                    exclusion_reasons.append("regional_exposure_limit")
            if not exclusion_reasons and column not in selected_names:
                exclusion_reasons.append("outside_top_n")

            execution_date = event_lookup.get((signal_date, column))
            weight = np.nan
            if execution_date is not None:
                value = position_weights.at[execution_date, column]
                weight = 0.0 if pd.isna(value) else float(value)
            rows.append(
                {
                    "signal_date": signal_date,
                    "instrument": column,
                    "momentum_score": scores.at[signal_date, column],
                    "momentum_rank": combined_ranks.at[signal_date, column],
                    "momentum_20_rank": period_ranks[0].at[signal_date, column],
                    "momentum_60_rank": period_ranks[1].at[signal_date, column],
                    "momentum_120_rank": period_ranks[2].at[signal_date, column],
                    "trend_eligible": bool(trend.at[signal_date, column]),
                    "realized_volatility": volatility.at[signal_date, column],
                    "position_sizing_weight": weight,
                    "region": region,
                    "regional_regime": regime,
                    "regional_exposure_limit": exposure_limits.get(regime, 0.0),
                    "exclusion_reason": ";".join(exclusion_reasons),
                }
            )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    if hasattr(portfolio.universe, "high") and hasattr(portfolio.universe, "low"):
        atr = _atr_14(portfolio.universe.high(), portfolio.universe.low(), close)
        diagnostics["atr_14"] = diagnostics.apply(
            lambda row: atr.at[row["signal_date"], row["instrument"]], axis=1
        )
    return diagnostics


def _single_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected one shared portfolio group.")
        return values.iloc[:, 0]
    return values


def _variant_metrics(
    result,
    portfolio: Portfolio,
    config: MomentumConfig,
) -> dict[str, float | int]:
    values = _single_series(result.value())
    daily_returns = values.pct_change(fill_method=None).dropna()
    close = portfolio.universe.close().reindex(index=values.index)
    assets = (
        result.assets().reindex(index=values.index, columns=close.columns).fillna(0)
    )
    market_values = assets.mul(close).fillna(0)
    invested = market_values.sum(axis=1)
    exposure = invested.div(values.replace(0, np.nan)).fillna(0)
    drawdown = values.div(values.cummax()).sub(1)
    total_return = values.iloc[-1] / portfolio.broker.initial_cash - 1
    elapsed_years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 252)
    cagr = (values.iloc[-1] / portfolio.broker.initial_cash) ** (1 / elapsed_years) - 1
    volatility = daily_returns.std(ddof=1)
    sharpe = (
        np.sqrt(252) * daily_returns.mean() / volatility
        if volatility and np.isfinite(volatility)
        else np.nan
    )
    downside = daily_returns.clip(upper=0).pow(2).mean() ** 0.5
    sortino = (
        np.sqrt(252) * daily_returns.mean() / downside
        if downside and np.isfinite(downside)
        else np.nan
    )
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    closed_trades = result.trades.closed.records_readable
    pnls = closed_trades["PnL"] if "PnL" in closed_trades else pd.Series(dtype=float)
    profits = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    profit_factor = (
        profits / losses if losses > 0 else (np.inf if profits > 0 else np.nan)
    )

    durations = []
    for _, trade in closed_trades.iterrows():
        entry = trade["Entry Timestamp"]
        exit_ = trade["Exit Timestamp"]
        instrument = trade["Column"]
        instrument_dates = close.index[close[instrument].notna()]
        durations.append(
            max(
                0,
                instrument_dates.searchsorted(exit_)
                - instrument_dates.searchsorted(entry),
            )
        )
    orders = result.orders.records_readable
    traded_value = (
        (orders["Size"].abs() * orders["Price"]).sum() if not orders.empty else 0.0
    )
    mean_value = values.mean()
    metrics: dict[str, float | int] = {
        "CAGR": float(cagr),
        "Total Return": float(total_return),
        "Maximum Drawdown": max_drawdown,
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Profit Factor": float(profit_factor),
        "Number of Trades": len(closed_trades),
        "Average Holding Period (sessions)": float(np.mean(durations))
        if durations
        else 0.0,
        "Average Exposure": float(exposure.mean()),
        "Maximum Exposure": float(exposure.max()),
        "Cash Allocation": float((1 - exposure).mean()),
        "Turnover": float(traded_value / mean_value) if mean_value else 0.0,
        "Transaction Costs": float(orders["Fees"].sum()) if not orders.empty else 0.0,
    }
    metadata = portfolio.universe.market_metadata()
    for region in ("EUROPE", "USA", "OTHER"):
        columns = [
            column
            for column in market_values.columns
            if _region_for_country((metadata.get(column) or {}).get("country"))
            == region
        ]
        regional = (
            market_values.reindex(columns=columns)
            .sum(axis=1)
            .div(values.replace(0, np.nan))
            .fillna(0)
        )
        metrics[f"Average {region} Exposure"] = float(regional.mean())
        metrics[f"Maximum {region} Exposure"] = float(regional.max())
    return metrics


def variant_experiment(
    portfolio: Portfolio, config: MomentumConfig | None = None
) -> pd.DataFrame:
    config = config or _default_variant_config()
    close = portfolio.universe.close()
    turnover = portfolio.universe.turnover()
    fingerprint = sha256()
    for label, frame in (("close", close), ("turnover", turnover)):
        normalized = frame.sort_index().sort_index(axis=1)
        fingerprint.update(label.encode())
        fingerprint.update(repr((normalized.shape, tuple(normalized.columns))).encode())
        fingerprint.update(
            pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes()
        )
    for region, benchmark in sorted(portfolio.benchmarks.items()):
        normalized = benchmark.sort_index()
        fingerprint.update(region.encode())
        fingerprint.update(
            pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes()
        )
    metadata = portfolio.universe.market_metadata()
    fingerprint.update(
        repr(
            sorted(
                (
                    instrument,
                    values.get("country"),
                    values.get("exchange"),
                )
                for instrument, values in metadata.items()
            )
        ).encode()
    )
    rows = []
    for variant in StrategyVariant:
        result = run_variant(portfolio, variant, config)
        rows.append(
            {
                "Variant": variant.value,
                "Signals": ",".join(sorted(VARIANT_SIGNALS[variant])),
                "Start Date": portfolio.start_date,
                "End Date": portfolio.end_date,
                "Universe Size": close.shape[1],
                "Universe Fingerprint": fingerprint.hexdigest(),
                "Initial Cash": portfolio.broker.initial_cash,
                "Fees": portfolio.broker.fees,
                "Fixed Fees": portfolio.broker.fixed_fees,
                "Slippage": portfolio.broker.slippage,
                "Benchmarks Available": ",".join(sorted(portfolio.benchmarks)),
                **asdict(config),
                **_variant_metrics(result, portfolio, config),
            }
        )
    return pd.DataFrame(rows)


def _run_sweep_case(
    close: pd.DataFrame,
    open_prices: pd.DataFrame,
    turnover: pd.DataFrame | None,
    broker: BrokerConfig,
    metadata: Mapping[str, Mapping[str, str | None]],
    benchmarks: Mapping[str, pd.Series],
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
        open_prices=open_prices,
        metadata=metadata,
        benchmarks=benchmarks,
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
    open_prices = (
        portfolio.universe.open() if hasattr(portfolio.universe, "open") else close
    )
    if close.empty:
        raise ValueError("Das Universe enthält keine historischen Kurse.")
    last_date = min(close.index.max(), pd.Timestamp(portfolio.end_date))
    turnover = portfolio.universe.turnover()
    metadata = portfolio.universe.market_metadata()
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
    worker = partial(
        _run_sweep_case,
        close,
        open_prices,
        turnover,
        portfolio.broker,
        metadata,
        portfolio.benchmarks,
    )
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
