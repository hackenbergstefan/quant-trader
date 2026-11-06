import pandas as pd

from quant_trader.features.momentum import calculate_momentum
from quant_trader.features.trend import calculate_trend
from quant_trader.strategies.adaptive_momentum import classify_market_regime


def make_prices(count: int = 205) -> pd.DataFrame:
    close = pd.Series(range(100, 100 + count), dtype=float)
    return pd.DataFrame(
        {
            "instrument_id": "TEST",
            "timestamp": pd.date_range("2020-01-01", periods=count, freq="B"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "adjusted_close": float("nan"),
            "volume": 100_000,
            "fx_to_eur": 1.0,
        }
    )


def test_momentum_indexing_and_warmup() -> None:
    prices = make_prices()

    features = calculate_momentum(prices)

    assert pd.isna(features.loc[19, "momentum_20"])
    assert features.loc[20, "momentum_20"] == (120 / 100) - 1
    assert pd.isna(features.loc[119, "momentum_120"])
    assert features.loc[120, "momentum_120"] == (220 / 100) - 1
    assert features.loc[200, "momentum_score"] == (
        0.2 * (300 / 280 - 1) + 0.3 * (300 / 240 - 1) + 0.5 * (300 / 180 - 1)
    )


def test_sma_and_atr_warmup() -> None:
    features = calculate_trend(make_prices(), (50, 200), 14)

    assert pd.isna(features.loc[48, "sma_50"])
    assert features.loc[49, "sma_50"] == sum(range(100, 150)) / 50
    assert pd.isna(features.loc[198, "sma_200"])
    assert features.loc[199, "sma_200"] == sum(range(100, 300)) / 200
    assert pd.isna(features.loc[12, "atr_14"])
    assert features.loc[13, "atr_14"] == 4


def test_market_regime_keeps_missing_reference_unknown() -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4),
            "close": [103.0, 100.0, 96.0, 100.0],
            "sma_200": [100.0, 100.0, 100.0, float("nan")],
        }
    )

    regimes = classify_market_regime(reference)

    assert regimes["market_regime"].tolist() == ["bullish", "neutral", "bearish", "unknown"]
