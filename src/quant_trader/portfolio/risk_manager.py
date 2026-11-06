from __future__ import annotations

import math

from quant_trader.config import RiskConfig


class RiskManager:
    def __init__(self, config: RiskConfig, stop_atr_multiple: float, cost_rate: float) -> None:
        self.config = config
        self.stop_atr_multiple = stop_atr_multiple
        self.cost_rate = cost_rate

    def shares_for_entry(
        self,
        *,
        equity_eur: float,
        cash_eur: float,
        entry_price: float,
        fx_to_eur: float,
        atr: float,
        target_weight: float,
        invested_eur: float,
    ) -> int:
        if min(entry_price, fx_to_eur, atr) <= 0 or target_weight <= 0:
            return 0
        stop_distance_local = atr * self.stop_atr_multiple
        risk_budget = equity_eur * self.config.risk_budget_per_position
        risk_shares = math.floor(risk_budget / (stop_distance_local * fx_to_eur))
        exposure_room = max(0.0, equity_eur * self.config.max_portfolio_exposure - invested_eur)
        target_value = min(
            target_weight * equity_eur,
            self.config.max_position_value_eur,
            exposure_room,
            cash_eur / (1 + self.cost_rate),
        )
        notional_shares = math.floor(target_value / (entry_price * fx_to_eur))
        return max(0, min(risk_shares, notional_shares))
