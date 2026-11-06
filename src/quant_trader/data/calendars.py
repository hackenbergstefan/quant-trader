from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class TradingCalendar(Protocol):
    def sessions(self, calendar: str, start: date, end: date) -> pd.DatetimeIndex: ...


class ExchangeTradingCalendar:
    def __init__(self) -> None:
        self._calendars: dict[str, object] = {}

    def sessions(self, calendar: str, start: date, end: date) -> pd.DatetimeIndex:
        try:
            import exchange_calendars as xcals
        except ImportError as exc:
            raise RuntimeError("Install exchange-calendars to validate exchange sessions") from exc
        if calendar not in self._calendars:
            try:
                self._calendars[calendar] = xcals.get_calendar(calendar)
            except Exception as exc:
                raise ValueError(f"Unsupported exchange calendar {calendar!r}: {exc}") from exc
        calendar_obj = self._calendars[calendar]
        return pd.DatetimeIndex(
            calendar_obj.sessions_in_range(start.isoformat(), end.isoformat())
        ).tz_localize(None)
