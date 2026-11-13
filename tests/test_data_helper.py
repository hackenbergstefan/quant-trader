from types import SimpleNamespace

import pandas as pd
import pytest

from pytrading.data_helper import Universe, _convert_to_eur


def test_convert_to_eur_handles_usd_and_gbpence():
    dates = pd.date_range("2024-01-02", periods=2)
    fx_rates = pd.DataFrame(
        {"USD": [2.0, 2.5], "GBP": [1.25, 1.2]},
        index=dates,
    )

    usd = _convert_to_eur(pd.Series([20.0, 25.0], index=dates), "USD", fx_rates)
    gbpence = _convert_to_eur(pd.Series([250.0, 240.0], index=dates), "GBp", fx_rates)

    pd.testing.assert_series_equal(usd, pd.Series([10.0, 10.0], index=dates))
    pd.testing.assert_series_equal(gbpence, pd.Series([2.0, 2.0], index=dates))


def test_universe_converts_close_and_turnover_but_not_volume():
    dates = pd.date_range("2024-01-02", periods=2)
    dataset = SimpleNamespace(
        isin="USD-STOCK",
        currency="USD",
        prices=pd.DataFrame(
            {
                "Close": [20.0, 25.0],
                "Raw Close": [20.0, 25.0],
                "Open": [19.0, 24.0],
                "Turnover": [200.0, 250.0],
                "Volume": [10, 10],
            },
            index=dates,
        ),
    )
    universe = Universe.__new__(Universe)
    universe.datasets = [dataset]
    universe._fx_rates = pd.DataFrame({"USD": [2.0, 2.5]}, index=dates)

    pd.testing.assert_series_equal(
        universe.close()["USD-STOCK"],
        pd.Series([10.0, 10.0], index=dates, name="USD-STOCK"),
    )
    pd.testing.assert_series_equal(
        universe.turnover()["USD-STOCK"],
        pd.Series([100.0, 100.0], index=dates, name="USD-STOCK"),
    )
    pd.testing.assert_series_equal(
        universe.open()["USD-STOCK"],
        pd.Series([9.5, 9.6], index=dates, name="USD-STOCK"),
    )
    pd.testing.assert_series_equal(
        universe.volume()["USD-STOCK"],
        pd.Series([10, 10], index=dates, name="USD-STOCK"),
    )


def test_universe_adjusts_ohl_to_adjusted_close_before_fx_conversion():
    dates = pd.date_range("2024-01-02", periods=2)
    dataset = SimpleNamespace(
        isin="USD-STOCK",
        currency="USD",
        prices=pd.DataFrame(
            {
                "Close": [10.0, 20.0],
                "Raw Close": [20.0, 40.0],
                "Open": [18.0, 38.0],
                "High": [22.0, 42.0],
                "Low": [16.0, 36.0],
                "Turnover": [200.0, 400.0],
            },
            index=dates,
        ),
    )
    universe = Universe.__new__(Universe)
    universe.datasets = [dataset]
    universe._fx_rates = pd.DataFrame({"USD": [2.0, 2.0]}, index=dates)

    assert universe.open()["USD-STOCK"].tolist() == pytest.approx([4.5, 9.5])
    assert universe.high()["USD-STOCK"].tolist() == pytest.approx([5.5, 10.5])
    assert universe.low()["USD-STOCK"].tolist() == pytest.approx([4.0, 9.0])
    assert universe.close()["USD-STOCK"].tolist() == pytest.approx([5.0, 10.0])
    assert universe.turnover()["USD-STOCK"].tolist() == pytest.approx([100, 200])


def test_universe_rejects_raw_ohl_without_adjustment_reference():
    dates = pd.date_range("2024-01-02", periods=2)
    dataset = SimpleNamespace(
        isin="BROKEN",
        currency="EUR",
        prices=pd.DataFrame({"Close": [10.0, 11.0], "Open": [9.0, 10.0]}, index=dates),
    )
    universe = Universe.__new__(Universe)
    universe.datasets = [dataset]
    universe._fx_rates = None

    with pytest.raises(ValueError, match="cannot align Open to adjusted Close"):
        universe.open()


def test_convert_to_eur_rejects_missing_fx_coverage():
    dates = pd.date_range("2024-01-02", periods=2)
    fx_rates = pd.DataFrame({"USD": [2.0]}, index=dates[1:])

    with pytest.raises(ValueError, match="Kein EUR/USD-Wechselkurs"):
        _convert_to_eur(pd.Series([20.0, 25.0], index=dates), "USD", fx_rates)


def test_fx_rate_loader_reads_downloaded_csv(tmp_path):
    dates = pd.date_range("2024-01-02", periods=2)
    expected = pd.DataFrame({"USD": [2.0, 2.5]}, index=dates)
    expected.index = pd.DatetimeIndex(expected.index.to_numpy(), name="Date")
    expected.to_csv(tmp_path / "fx_rates.csv", index_label="Date")
    universe = Universe.__new__(Universe)
    universe.data_dir = tmp_path

    rates = universe._load_fx_rates({"USD"}, dates[0], dates[-1])

    pd.testing.assert_frame_equal(rates, expected)


def test_fx_rate_loader_requires_pre_downloaded_csv(tmp_path):
    dates = pd.date_range("2024-01-02", periods=2)
    universe = Universe.__new__(Universe)
    universe.data_dir = tmp_path

    with pytest.raises(FileNotFoundError, match="download_fx_rates"):
        universe._load_fx_rates({"USD"}, dates[0], dates[-1])


def test_fx_rate_loader_rejects_insufficient_date_coverage(tmp_path):
    dates = pd.date_range("2024-01-02", periods=2)
    pd.DataFrame({"USD": [2.5]}, index=dates[1:]).to_csv(
        tmp_path / "fx_rates.csv", index_label="Date"
    )
    universe = Universe.__new__(Universe)
    universe.data_dir = tmp_path

    with pytest.raises(ValueError, match="deckt den Zeitraum"):
        universe._load_fx_rates({"USD"}, dates[0], dates[-1])
