"""Tests for the market simulator."""
import numpy as np
import pandas as pd
import pytest

from trading.simulator import DEFAULT_SYMBOLS, MarketSimulator


class TestMarketSimulator:
    @pytest.fixture
    def sim(self):
        return MarketSimulator(n_bars=200, seed=0)

    def test_all_symbols_generated(self, sim):
        for sym in DEFAULT_SYMBOLS:
            df = sim.get_prices(sym)
            assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_ohlcv_columns(self, sim):
        df = sim.get_prices("EURUSD")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_correct_number_of_bars(self, sim):
        df = sim.get_prices("EURUSD")
        assert len(df) == 200

    def test_high_ge_low(self, sim):
        for sym in DEFAULT_SYMBOLS:
            df = sim.get_prices(sym)
            assert (df["high"] >= df["low"]).all()

    def test_positive_prices(self, sim):
        for sym in DEFAULT_SYMBOLS:
            df = sim.get_prices(sym)
            assert (df[["open", "high", "low", "close"]] > 0).all().all()

    def test_get_close_returns_array(self, sim):
        close = sim.get_close("GBPUSD")
        assert isinstance(close, np.ndarray)
        assert len(close) == 200

    def test_unknown_symbol_raises(self, sim):
        with pytest.raises(KeyError):
            sim.get_prices("INVALID")

    def test_regenerate_changes_data(self):
        sim = MarketSimulator(n_bars=100, seed=1)
        close_before = sim.get_close("EURUSD").copy()
        sim.regenerate(seed=999)
        close_after = sim.get_close("EURUSD")
        assert not np.allclose(close_before, close_after)

    def test_reproducibility(self):
        sim1 = MarketSimulator(n_bars=100, seed=42)
        sim2 = MarketSimulator(n_bars=100, seed=42)
        np.testing.assert_array_equal(
            sim1.get_close("EURUSD"), sim2.get_close("EURUSD")
        )
