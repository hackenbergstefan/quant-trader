from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum_periods: tuple[int, int, int] = (20, 60, 120)
    momentum_weights: tuple[float, float, float] = (0.2, 0.3, 0.5)
    sma_fast: int = 50
    sma_slow: int = 200
    atr_period: int = 14
    volatility_periods: tuple[int, int] = (20, 60)
    liquidity_window: int = 20
    min_avg_daily_value_eur: float = 1_000_000
    min_history: int = 200
    max_positions: int = 5
    neutral_zone: float = 0.02
    stop_atr_multiple: float = 2.0
    regimes: dict[str, float] = Field(
        default_factory=lambda: {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0, "unknown": 0.0}
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> StrategyConfig:
        if not abs(sum(self.momentum_weights) - 1.0) < 1e-9:
            raise ValueError("momentum_weights must sum to 1")
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be shorter than sma_slow")
        if self.max_positions < 1 or self.min_history < self.sma_slow:
            raise ValueError("max_positions must be positive and min_history >= sma_slow")
        if not 0 <= self.neutral_zone < 1:
            raise ValueError("neutral_zone must be in [0, 1)")
        return self


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    starting_cash_eur: float = 5_000
    max_position_value_eur: float = 1_000
    max_portfolio_exposure: float = 1.0
    risk_budget_per_position: float = 0.01
    allocation_method: Literal["equal_weight", "volatility_adjusted", "risk_budget"] = (
        "equal_weight"
    )
    rebalance_threshold: float = 0.10

    @model_validator(mode="after")
    def validate_risk(self) -> RiskConfig:
        if self.starting_cash_eur <= 0 or self.max_position_value_eur <= 0:
            raise ValueError("starting cash and maximum position value must be positive")
        if not 0 < self.risk_budget_per_position <= 1:
            raise ValueError("risk_budget_per_position must be in (0, 1]")
        if not 0 <= self.max_portfolio_exposure <= 1:
            raise ValueError("max_portfolio_exposure must be in [0, 1]")
        if not 0 <= self.rebalance_threshold < 1:
            raise ValueError("rebalance_threshold must be in [0, 1)")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commission_bps: float = 1.0
    exchange_fee_bps: float = 0.5
    spread_bps: float = 5.0
    slippage_bps: float = 2.0

    @property
    def total_cost_rate(self) -> float:
        return (
            sum((self.commission_bps, self.exchange_fee_bps, self.spread_bps, self.slippage_bps))
            / 10_000
        )


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: str = "data/market.duckdb"
    provider: Literal["yahoo"] = "yahoo"
    benchmark_symbol: str = "^STOXX50E"
    benchmark_currency: str = "EUR"
    benchmark_exchange: str = "EUREX"
    benchmark_calendar: str = "XETR"


class AppConfig(BaseModel):
    strategy: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig
    data: DataConfig


def load_config(config_dir: str | Path = "config") -> AppConfig:
    root = Path(config_dir)

    def read(name: str) -> dict:
        path = root / name
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration file must contain a mapping: {path}")
        return payload

    return AppConfig(
        strategy=StrategyConfig.model_validate(read("strategy.yaml")),
        risk=RiskConfig.model_validate(read("risk.yaml")),
        execution=ExecutionConfig.model_validate(read("execution.yaml")),
        data=DataConfig.model_validate(read("data.yaml")),
    )
