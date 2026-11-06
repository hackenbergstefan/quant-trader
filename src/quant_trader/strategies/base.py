from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def generate_signals(
        self, features: pd.DataFrame, reference: pd.DataFrame | None = None
    ) -> pd.DataFrame: ...
