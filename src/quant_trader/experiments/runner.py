from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_trader.backtesting.benchmark import buy_and_hold_benchmark, cash_benchmark
from quant_trader.backtesting.engine import BacktestEngine, BacktestResult
from quant_trader.config import AppConfig, ExecutionConfig, RiskConfig, StrategyConfig
from quant_trader.domain.models import CorporateAction
from quant_trader.features.liquidity import build_features
from quant_trader.portfolio.allocation import apply_allocation_method
from quant_trader.strategies.adaptive_momentum import AdaptiveMomentumStrategy


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    trend_filter: bool
    regime_filter: bool
    allocation_method: str = "equal_weight"


VARIANTS = (
    Variant("B_simple_momentum", False, False),
    Variant("C_momentum_trend", True, False),
    Variant("D_volatility_adjusted", True, False, "volatility_adjusted"),
    Variant("E_adaptive_momentum", True, True),
)


def run_variant(
    name: str,
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    strategy_config: StrategyConfig,
    risk: RiskConfig,
    execution: ExecutionConfig,
    actions: list[CorporateAction] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    aliases = {
        "simple_momentum": VARIANTS[0],
        "B": VARIANTS[0],
        "momentum_trend": VARIANTS[1],
        "C": VARIANTS[1],
        "volatility_adjusted": VARIANTS[2],
        "D": VARIANTS[2],
        "adaptive_momentum": VARIANTS[3],
        "E": VARIANTS[3],
    }
    variant = aliases.get(name, next((item for item in VARIANTS if item.name == name), None))
    if variant is None:
        raise ValueError(f"Unknown strategy {name!r}; choose B, C, D, E, or their full names")
    features = build_features(bars, strategy_config)
    strategy = AdaptiveMomentumStrategy(
        strategy_config,
        trend_filter=variant.trend_filter,
        regime_filter=variant.regime_filter,
        name=variant.name,
    )
    signals = strategy.generate_signals(features, reference)
    allocation_method = variant.allocation_method
    if allocation_method == "equal_weight" and risk.allocation_method != "equal_weight":
        allocation_method = risk.allocation_method
    signals = apply_allocation_method(signals, allocation_method)
    simulation_bars = bars.copy()
    simulation_signals = signals.copy()
    if start_date is not None:
        start = pd.Timestamp(start_date)
        simulation_bars = simulation_bars.loc[pd.to_datetime(simulation_bars["timestamp"]) >= start]
        simulation_signals = simulation_signals.loc[
            pd.to_datetime(simulation_signals["timestamp"]) >= start
        ]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        simulation_bars = simulation_bars.loc[pd.to_datetime(simulation_bars["timestamp"]) <= end]
        simulation_signals = simulation_signals.loc[
            pd.to_datetime(simulation_signals["timestamp"]) <= end
        ]
    if simulation_bars.empty:
        raise ValueError(f"No bars in requested simulation period for {name}")
    if actions is not None and (start_date is not None or end_date is not None):
        actions = [
            action
            for action in actions
            if (start_date is None or action.ex_date >= start_date)
            and (end_date is None or action.ex_date <= end_date)
        ]
    engine = BacktestEngine(
        risk,
        execution,
        stop_atr_multiple=strategy_config.stop_atr_multiple,
        strategy_name=variant.name,
        max_positions=strategy_config.max_positions,
    )
    return engine.run(simulation_bars, simulation_signals, actions)


def run_comparison(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    config: AppConfig,
    actions: list[CorporateAction] | None = None,
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    results: dict[str, BacktestResult] = {
        variant.name: run_variant(
            variant.name, bars, reference, config.strategy, config.risk, config.execution, actions
        )
        for variant in VARIANTS
    }
    results["A_euro_stoxx_50_buy_and_hold"] = buy_and_hold_benchmark(
        reference, config.risk.starting_cash_eur, config.execution.total_cost_rate
    )
    results["cash"] = cash_benchmark(config.risk.starting_cash_eur, bars["timestamp"])
    table = (
        pd.DataFrame([{"variant": name, **result.metrics} for name, result in results.items()])
        .sort_values("variant")
        .reset_index(drop=True)
    )
    return results, table


def run_cost_sensitivity(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    config: AppConfig,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    actions: list[CorporateAction] | None = None,
) -> pd.DataFrame:
    rows = []
    for multiplier in multipliers:
        adjusted_execution = config.execution.model_copy(
            update={key: value * multiplier for key, value in config.execution.model_dump().items()}
        )
        result = run_variant(
            "E", bars, reference, config.strategy, config.risk, adjusted_execution, actions
        )
        rows.append({"cost_multiplier": multiplier, **result.metrics})
    return pd.DataFrame(rows)


def run_parameter_sensitivity(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    config: AppConfig,
    neutral_zones: tuple[float, ...] = (0.01, 0.02, 0.05),
    stop_multiples: tuple[float, ...] = (1.5, 2.0, 2.5),
    actions: list[CorporateAction] | None = None,
) -> pd.DataFrame:
    rows = []
    for neutral_zone in neutral_zones:
        for stop_multiple in stop_multiples:
            strategy = config.strategy.model_copy(
                update={
                    "neutral_zone": neutral_zone,
                    "stop_atr_multiple": stop_multiple,
                }
            )
            result = run_variant(
                "E", bars, reference, strategy, config.risk, config.execution, actions
            )
            rows.append(
                {"neutral_zone": neutral_zone, "stop_atr_multiple": stop_multiple, **result.metrics}
            )
    return pd.DataFrame(rows)
