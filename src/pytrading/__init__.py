from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from pytrading.data_helper import Universe


@dataclass(frozen=True)
class BrokerConfig:
    initial_cash: float = 1000.0
    fees: float = 0.001
    fixed_fees: float = 1.0
    slippage: float = 0.0005


@dataclass(frozen=True)
class Portfolio:
    universe: Universe
    start_date: str
    end_date: str
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    benchmarks: Mapping[str, pd.Series] = field(default_factory=dict)
