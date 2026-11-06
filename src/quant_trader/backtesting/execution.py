from __future__ import annotations

from dataclasses import dataclass

from quant_trader.config import ExecutionConfig


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    cost_rate: float
    components_bps: dict[str, float]

    @classmethod
    def from_config(cls, config: ExecutionConfig) -> ExecutionModel:
        return cls(
            cost_rate=config.total_cost_rate,
            components_bps={
                "commission": config.commission_bps,
                "exchange_fee": config.exchange_fee_bps,
                "spread": config.spread_bps,
                "slippage": config.slippage_bps,
            },
        )

    def transaction_cost(self, notional_eur: float) -> float:
        return abs(notional_eur) * self.cost_rate
