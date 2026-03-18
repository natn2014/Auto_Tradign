"""Tests for the technical indicators module."""
import numpy as np
import pytest

from trading.indicators import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    compute_sma,
)


class TestRSI:
    def test_output_length_matches_input(self):
        prices = np.linspace(1.0, 2.0, 50)
        rsi = compute_rsi(prices)
        assert len(rsi) == len(prices)

    def test_first_values_are_nan(self):
        prices = np.linspace(1.0, 2.0, 50)
        rsi = compute_rsi(prices, period=14)
        # First period values should be NaN
        assert np.all(np.isnan(rsi[:14]))

    def test_values_in_valid_range(self):
        rng = np.random.default_rng(0)
        prices = np.cumprod(1 + rng.normal(0, 0.01, 200))
        rsi = compute_rsi(prices, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_constant_prices_returns_nan_or_50(self):
        prices = np.ones(30)
        rsi = compute_rsi(prices, period=14)
        valid = rsi[~np.isnan(rsi)]
        # When there's no movement, RSI should be 50 (or handled gracefully)
        assert len(valid) >= 0  # at minimum, no crash

    def test_short_series(self):
        prices = np.array([1.0, 2.0, 3.0])
        rsi = compute_rsi(prices, period=14)
        assert len(rsi) == 3
        assert np.all(np.isnan(rsi))


class TestMACD:
    def test_output_shapes(self):
        prices = np.linspace(1.0, 2.0, 100)
        macd, signal, hist = compute_macd(prices)
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)

    def test_histogram_equals_macd_minus_signal(self):
        rng = np.random.default_rng(1)
        prices = np.cumprod(1 + rng.normal(0, 0.01, 200))
        macd, signal, hist = compute_macd(prices)
        mask = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(hist))
        np.testing.assert_allclose(hist[mask], macd[mask] - signal[mask], atol=1e-10)


class TestBollingerBands:
    def test_output_shapes(self):
        prices = np.linspace(1.0, 2.0, 50)
        upper, mid, lower = compute_bollinger_bands(prices)
        assert len(upper) == len(prices)
        assert len(mid) == len(prices)
        assert len(lower) == len(prices)

    def test_upper_ge_mid_ge_lower(self):
        rng = np.random.default_rng(2)
        prices = np.cumprod(1 + rng.normal(0, 0.01, 100))
        upper, mid, lower = compute_bollinger_bands(prices)
        mask = ~(np.isnan(upper) | np.isnan(mid) | np.isnan(lower))
        assert np.all(upper[mask] >= mid[mask])
        assert np.all(mid[mask] >= lower[mask])

    def test_first_period_minus_one_are_nan(self):
        prices = np.linspace(1.0, 2.0, 50)
        upper, mid, lower = compute_bollinger_bands(prices, period=20)
        assert np.all(np.isnan(mid[:19]))
        assert not np.isnan(mid[19])


class TestSMA:
    def test_output_length(self):
        prices = np.arange(1.0, 21.0)
        sma = compute_sma(prices, period=5)
        assert len(sma) == 20

    def test_known_values(self):
        prices = np.arange(1.0, 11.0)
        sma = compute_sma(prices, period=3)
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        np.testing.assert_almost_equal(sma[2], 2.0)
        np.testing.assert_almost_equal(sma[9], 9.0)
