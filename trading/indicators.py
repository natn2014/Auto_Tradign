"""Technical indicators for trading analysis."""
import numpy as np


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index (RSI).

    Args:
        prices: Array of closing prices.
        period: Look-back period (default 14).

    Returns:
        RSI values in the range [0, 100]. Values before ``period + 1`` are NaN.
    """
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(prices), np.nan)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """Compute MACD indicator.

    Args:
        prices: Array of closing prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram) as np.ndarray each.
    """

    def _ema(data: np.ndarray, span: int) -> np.ndarray:
        k = 2.0 / (span + 1)
        out = np.full(len(data), np.nan)
        start = 0
        while start < len(data) and np.isnan(data[start]):
            start += 1
        if start >= len(data):
            return out
        out[start] = data[start]
        for i in range(start + 1, len(data)):
            if np.isnan(data[i]):
                out[i] = np.nan
            else:
                out[i] = data[i] * k + out[i - 1] * (1 - k)
        return out

    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple:
    """Compute Bollinger Bands.

    Args:
        prices: Array of closing prices.
        period: Moving average period.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band) as np.ndarray each.
    """
    n = len(prices)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        middle[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s

    return upper, middle, lower


def compute_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Simple Moving Average.

    Args:
        prices: Array of closing prices.
        period: Window size.

    Returns:
        SMA values; first ``period - 1`` entries are NaN.
    """
    sma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1 : i + 1])
    return sma
