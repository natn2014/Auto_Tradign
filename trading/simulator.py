"""Market data simulator using Geometric Brownian Motion (GBM).

Generates realistic synthetic OHLCV price data for multiple currency pairs
so the trading agent can be trained and tested without requiring live feeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Default set of simulated currency pairs (mirrors XM-style symbols)
DEFAULT_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF",
    "EURGBP",
]

# Approximate initial prices for each symbol
_BASE_PRICES: dict[str, float] = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2700,
    "USDJPY": 149.50,
    "AUDUSD": 0.6550,
    "USDCAD": 1.3600,
    "NZDUSD": 0.6100,
    "USDCHF": 0.8950,
    "EURGBP": 0.8550,
}

# Approximate annualised volatility for each symbol
_VOLATILITIES: dict[str, float] = {
    "EURUSD": 0.07,
    "GBPUSD": 0.08,
    "USDJPY": 0.09,
    "AUDUSD": 0.10,
    "USDCAD": 0.07,
    "NZDUSD": 0.10,
    "USDCHF": 0.07,
    "EURGBP": 0.05,
}


class MarketSimulator:
    """Generates synthetic OHLCV bars using GBM.

    Parameters
    ----------
    symbols:
        List of symbol names to simulate.  Must be a subset of
        ``DEFAULT_SYMBOLS`` unless you also supply ``base_prices`` /
        ``volatilities``.
    n_bars:
        Number of 1-minute bars to pre-generate.
    seed:
        Optional random seed for reproducibility.
    base_prices:
        Override the base (starting) price per symbol.
    volatilities:
        Override the annualised volatility per symbol.
    """

    BARS_PER_YEAR = 252 * 390  # ~1-minute bars in a trading year

    def __init__(
        self,
        symbols: list[str] | None = None,
        n_bars: int = 2000,
        seed: int | None = None,
        base_prices: dict[str, float] | None = None,
        volatilities: dict[str, float] | None = None,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.n_bars = n_bars
        self.rng = np.random.default_rng(seed)

        self._base_prices = {**_BASE_PRICES, **(base_prices or {})}
        self._volatilities = {**_VOLATILITIES, **(volatilities or {})}

        # Pre-generated data: symbol -> DataFrame with OHLCV columns
        self._data: dict[str, pd.DataFrame] = {}
        self._generate_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(self, symbol: str) -> pd.DataFrame:
        """Return the full OHLCV DataFrame for *symbol*."""
        if symbol not in self._data:
            raise KeyError(f"Unknown symbol: {symbol!r}")
        return self._data[symbol]

    def get_close(self, symbol: str) -> np.ndarray:
        """Return the close-price array for *symbol*."""
        return self._data[symbol]["close"].to_numpy()

    def regenerate(self, seed: int | None = None) -> None:
        """Re-generate all price series with an optional new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._generate_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_all(self) -> None:
        for symbol in self.symbols:
            self._data[symbol] = self._generate_symbol(symbol)

    def _generate_symbol(self, symbol: str) -> pd.DataFrame:
        s0 = self._base_prices.get(symbol, 1.0)
        sigma = self._volatilities.get(symbol, 0.08)
        mu = 0.0  # zero drift for simulation neutrality

        dt = 1.0 / self.BARS_PER_YEAR
        sqrt_dt = np.sqrt(dt)

        n = self.n_bars
        z = self.rng.standard_normal(n)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        closes = s0 * np.exp(np.cumsum(log_returns))

        # Build OHLC from close series
        noise = sigma * sqrt_dt * np.abs(self.rng.standard_normal(n))
        highs = closes * (1.0 + noise)
        lows = closes * (1.0 - noise)
        # Open: previous close (first open == s0)
        opens = np.empty(n)
        opens[0] = s0
        opens[1:] = closes[:-1]

        volume = (self.rng.integers(100, 10000, size=n)).astype(float)

        timestamps = pd.date_range(
            start="2024-01-01 00:00", periods=n, freq="1min"
        )

        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            },
            index=timestamps,
        )
