import pandas as pd
import pytest

from quant_trader.backtesting.benchmark import buy_and_hold_benchmark


def test_buy_and_hold_return_charges_entry_and_terminal_costs_against_initial_cash() -> None:
    reference = pd.DataFrame(
        {
            "instrument_id": "EUROSTOXX50",
            "timestamp": pd.date_range("2024-01-02", periods=2, freq="B"),
            "close": [100.0, 110.0],
            "adjusted_close": [100.0, 110.0],
        }
    )

    result = buy_and_hold_benchmark(reference, 5_000.0, 0.001)

    assert result.metrics["total_return_net"] == pytest.approx(1.1 * (1 - 0.001) ** 2 - 1)
    assert result.metrics["total_return_gross_estimate"] == pytest.approx(0.1)
    assert result.metrics["transaction_costs_eur"] == pytest.approx(5 + 5.4945)
