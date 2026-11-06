from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from quant_trader.domain.models import Instrument


class InstrumentRepository(Protocol):
    def list_instruments(self, active_only: bool = True) -> list[Instrument]: ...

    def get(self, instrument_id: str) -> Instrument: ...


class YamlInstrumentRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._instruments = {
            item["instrument_id"]: Instrument(**item) for item in payload.get("instruments", [])
        }
        if len(self._instruments) != len(payload.get("instruments", [])):
            raise ValueError(f"Duplicate instrument_id in {self.path}")

    def list_instruments(self, active_only: bool = True) -> list[Instrument]:
        result = list(self._instruments.values())
        return [item for item in result if item.active] if active_only else result

    def get(self, instrument_id: str) -> Instrument:
        try:
            return self._instruments[instrument_id]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument_id: {instrument_id}") from exc
