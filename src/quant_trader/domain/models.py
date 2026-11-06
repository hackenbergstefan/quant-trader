from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Instrument:
    instrument_id: str
    symbol: str
    exchange: str
    currency: str
    calendar: str
    fx_to_eur: float = 1.0
    active: bool = True


class ActionType(StrEnum):
    DIVIDEND = "dividend"
    SPLIT = "split"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    instrument_id: str
    ex_date: date
    action_type: ActionType
    value: float
    currency: str


@dataclass(frozen=True, slots=True)
class Order:
    instrument_id: str
    decision_date: date
    target_weight: float
    atr: float | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    instrument_id: str
    timestamp: date
    side: str
    quantity: int
    price: float
    costs: float
    reason: str
