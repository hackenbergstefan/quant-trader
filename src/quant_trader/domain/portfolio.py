from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class Position:
    instrument_id: str
    quantity: int
    average_entry_price: float
    entry_date: date
    stop_price: float
    entry_quantity: float
    entry_costs_eur: float
    dividends_eur: float
    realized_pnl_eur: float
    risk_budget_eur: float
    fx_to_eur: float


@dataclass(slots=True)
class Portfolio:
    cash_eur: float
    positions: dict[str, Position]
    transaction_costs_eur: float = 0.0
