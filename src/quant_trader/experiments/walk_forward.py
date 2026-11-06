from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trader.backtesting.engine import BacktestEngine
from quant_trader.config import AppConfig
from quant_trader.features.liquidity import build_features
from quant_trader.portfolio.allocation import apply_allocation_method
from quant_trader.strategies.adaptive_momentum import AdaptiveMomentumStrategy


def run_walk_forward(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    config: AppConfig,
    *,
    actions=None,
    training_years: int = 5,
    test_years: int = 2,
    step_years: int = 1,
) -> pd.DataFrame:
    if min(training_years, test_years, step_years) < 1:
        raise ValueError("Walk-forward windows must all be positive whole years")
    all_bars = bars.copy()
    all_bars["timestamp"] = pd.to_datetime(all_bars["timestamp"])
    features = build_features(all_bars, config.strategy)
    strategy = AdaptiveMomentumStrategy(config.strategy, name="E_adaptive_momentum")
    signals = apply_allocation_method(
        strategy.generate_signals(features, reference), config.risk.allocation_method
    )
    first_date = all_bars["timestamp"].min().date()
    last_date = all_bars["timestamp"].max().date()
    cursor = pd.Timestamp(first_date)
    rows: list[dict[str, object]] = []
    while True:
        train_end = cursor + pd.DateOffset(years=training_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_end.date() > last_date:
            break
        period_bars = all_bars.loc[all_bars["timestamp"].between(test_start, test_end)]
        period_signals = signals.loc[
            pd.to_datetime(signals["timestamp"]).between(test_start, test_end)
        ]
        if not period_bars.empty and not period_signals.empty:
            engine = BacktestEngine(
                config.risk,
                config.execution,
                stop_atr_multiple=config.strategy.stop_atr_multiple,
                strategy_name="E_adaptive_momentum",
                max_positions=config.strategy.max_positions,
            )
            result = engine.run(
                period_bars,
                period_signals,
                [
                    action
                    for action in (actions or [])
                    if test_start.date() <= action.ex_date <= test_end.date()
                ],
            )
            rows.append(
                {
                    "training_start": cursor.date().isoformat(),
                    "training_end": train_end.date().isoformat(),
                    "test_start": result.equity_curve["timestamp"].min().date().isoformat(),
                    "test_end": result.equity_curve["timestamp"].max().date().isoformat(),
                    **result.metrics,
                }
            )
        cursor += pd.DateOffset(years=step_years)
    return pd.DataFrame(rows)


def run_holdout(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    config: AppConfig,
    in_sample_end: date = date(2022, 12, 31),
    out_sample_start: date = date(2023, 1, 1),
    out_sample_end: date = date(2024, 12, 31),
    actions=None,
) -> pd.DataFrame:
    from quant_trader.experiments.runner import run_variant

    rows = []
    for label, start, end in (
        ("in_sample", date(2010, 1, 1), in_sample_end),
        ("out_of_sample", out_sample_start, out_sample_end),
    ):
        subset_dates = pd.to_datetime(bars["timestamp"]).dt.date
        subset = bars.loc[subset_dates.between(start, end)]
        benchmark_dates = pd.to_datetime(reference["timestamp"]).dt.date
        benchmark = reference.loc[benchmark_dates.between(start, end)]
        if subset.empty or benchmark.empty:
            rows.append(
                {
                    "period": label,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "status": "unavailable",
                }
            )
            continue
        result = run_variant(
            "E",
            bars,
            reference,
            config.strategy,
            config.risk,
            config.execution,
            actions=actions,
            start_date=start,
            end_date=end,
        )
        rows.append(
            {
                "period": label,
                "start": result.metadata["start_date"],
                "end": result.metadata["end_date"],
                "status": "available",
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)
