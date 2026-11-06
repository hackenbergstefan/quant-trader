from __future__ import annotations

from enum import StrEnum

import pandas as pd

from quant_trader.config import StrategyConfig


class MarketRegime(StrEnum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    UNKNOWN = "unknown"


def classify_market_regime(
    reference: pd.DataFrame, period: int = 200, neutral_zone: float = 0.02
) -> pd.DataFrame:
    frame = reference.sort_values("timestamp").copy()
    price_column = "feature_close" if "feature_close" in frame else "adjusted_close"
    if price_column not in frame or frame[price_column].isna().all():
        price_column = "close"
    if f"sma_{period}" not in frame:
        frame[f"sma_{period}"] = frame[price_column].rolling(period, min_periods=period).mean()
    close, average = frame[price_column], frame[f"sma_{period}"]
    regime = pd.Series(MarketRegime.UNKNOWN.value, index=frame.index, dtype="object")
    valid = close.notna() & average.notna() & (average > 0)
    regime.loc[valid & (close > average * (1 + neutral_zone))] = MarketRegime.BULLISH.value
    regime.loc[valid & (close < average * (1 - neutral_zone))] = MarketRegime.BEARISH.value
    regime.loc[valid & regime.eq(MarketRegime.UNKNOWN.value)] = MarketRegime.NEUTRAL.value
    frame["market_regime"] = regime
    return frame[["timestamp", "market_regime"]]


class AdaptiveMomentumStrategy:
    name = "adaptive_momentum"

    def __init__(
        self,
        config: StrategyConfig,
        *,
        trend_filter: bool = True,
        regime_filter: bool = True,
        liquidity_filter: bool = True,
        name: str | None = None,
    ) -> None:
        self.config = config
        self.trend_filter = trend_filter
        self.regime_filter = regime_filter
        self.liquidity_filter = liquidity_filter
        self.name = name or self.name

    def generate_signals(
        self, features: pd.DataFrame, reference: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        frame = features.sort_values(["timestamp", "instrument_id"]).copy()
        frame["history_count"] = frame.groupby("instrument_id", sort=False).cumcount() + 1
        regime_by_date: dict[pd.Timestamp, str] = {}
        if self.regime_filter and reference is not None and not reference.empty:
            classified = classify_market_regime(
                reference, self.config.sma_slow, self.config.neutral_zone
            )
            regime_by_date = {
                pd.Timestamp(row.timestamp): str(row.market_regime)
                for row in classified.itertuples(index=False)
            }

        frame["market_regime"] = [
            regime_by_date.get(pd.Timestamp(timestamp), MarketRegime.UNKNOWN.value)
            if self.regime_filter
            else MarketRegime.BULLISH.value
            for timestamp in frame["timestamp"]
        ]
        eligible = frame["history_count"].ge(self.config.min_history) & frame["momentum_score"].gt(
            0
        )
        if self.trend_filter:
            eligible &= frame["feature_close"].gt(frame[f"sma_{self.config.sma_fast}"])
            eligible &= frame[f"sma_{self.config.sma_fast}"].gt(
                frame[f"sma_{self.config.sma_slow}"]
            )
        if self.liquidity_filter:
            eligible &= frame["average_daily_traded_value_eur"].gt(
                self.config.min_avg_daily_value_eur
            )
        frame["eligible"] = eligible.fillna(False)
        frame["selected"] = False
        frame["target_weight"] = 0.0

        for timestamp, indices in frame.groupby("timestamp", sort=True).groups.items():
            day = frame.loc[indices]
            if self.regime_filter:
                regime = day["market_regime"].iloc[0]
                exposure = self.config.regimes.get(regime, 0.0)
            else:
                exposure = self.config.regimes.get("bullish", 1.0)
            if exposure <= 0:
                continue
            selected = (
                day.loc[day["eligible"]]
                .sort_values(
                    ["momentum_score", "instrument_id"], ascending=[False, True], kind="stable"
                )
                .head(self.config.max_positions)
            )
            if selected.empty:
                continue
            weight = exposure / len(selected)
            frame.loc[selected.index, "selected"] = True
            frame.loc[selected.index, "target_weight"] = weight

        frame["market_regime"] = frame["market_regime"].astype(str)
        return frame[
            [
                "instrument_id",
                "timestamp",
                "momentum_score",
                "feature_close",
                f"atr_{self.config.atr_period}",
                "volatility_20",
                "eligible",
                "selected",
                "target_weight",
                "market_regime",
            ]
        ].rename(columns={f"atr_{self.config.atr_period}": "atr"})
