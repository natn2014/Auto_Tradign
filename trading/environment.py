"""Reinforcement-learning trading environment.

Compatible with the Gymnasium API (``gymnasium.Env``).  The agent interacts
with a simulated market and is rewarded for profitable trades while being
penalised more heavily for losses (asymmetric reward shaping to *minimise
loss*).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trading.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_sma,
)
from trading.simulator import MarketSimulator


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
HOLD = 0
BUY = 1
SELL = 2

SPREAD_PIPS = 2  # typical XM spread in pips
PIP_VALUE = 0.0001  # value of 1 pip for most forex pairs


class TradingEnvironment(gym.Env):
    """Single-symbol trading environment for RL training.

    Observations
    ------------
    A flat vector of normalised features:

    * 20 normalised close-price returns
    * RSI (normalised to [0, 1])
    * MACD histogram (normalised)
    * Bollinger-band width (normalised)
    * Current position encoded as [long, flat, short] one-hot (3 values)
    * Normalised unrealised P&L

    Actions
    -------
    0 – HOLD
    1 – BUY  (open long / close short)
    2 – SELL (open short / close long)

    Reward shaping
    --------------
    * Realised P&L at close is credited/debited.
    * Holding a losing position incurs a small per-step holding cost.
    * Loss multiplier (``loss_penalty``) amplifies negative rewards so the
      agent is incentivised to cut losses early – the core XM "minimise loss"
      requirement.
    """

    metadata = {"render_modes": ["human"]}

    # Number of past returns included in the observation
    WINDOW = 20
    # Feature count: WINDOW returns + RSI + MACD hist + BB width + 3 position + PnL
    N_FEATURES = WINDOW + 1 + 1 + 1 + 3 + 1

    def __init__(
        self,
        simulator: MarketSimulator,
        symbol: str,
        initial_balance: float = 10_000.0,
        lot_size: float = 0.01,  # micro lot
        loss_penalty: float = 2.0,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()

        self.simulator = simulator
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.loss_penalty = loss_penalty

        self._prices = simulator.get_close(symbol)
        self._n = len(self._prices)
        self._max_steps = max_steps or (self._n - self.WINDOW - 1)

        # Gymnasium spaces
        obs_low = np.full(self.N_FEATURES, -np.inf, dtype=np.float32)
        obs_high = np.full(self.N_FEATURES, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Will be initialised in reset()
        self._step = 0
        self._balance = initial_balance
        self._position = 0   # +1 long, 0 flat, -1 short
        self._entry_price = 0.0
        self._equity_curve: list[float] = []
        self._trade_log: list[dict[str, Any]] = []

        # Pre-compute full indicator arrays once
        self._rsi = compute_rsi(self._prices, period=14)
        self._macd_hist = compute_macd(self._prices)[2]
        _bb_upper, _, _bb_lower = compute_bollinger_bands(self._prices, period=20)
        self._bb_width = _bb_upper - _bb_lower
        self._sma20 = compute_sma(self._prices, 20)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        self._balance = self.initial_balance
        self._position = 0
        self._entry_price = 0.0
        self._equity_curve = [self.initial_balance]
        self._trade_log = []
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self._prices[self._step + self.WINDOW]
        spread = SPREAD_PIPS * PIP_VALUE

        reward = 0.0

        # ── Execute action ─────────────────────────────────────────────
        if action == BUY:
            if self._position == -1:  # close short
                pnl = (self._entry_price - current_price - spread) * self.lot_size * 100_000
                reward += self._shaped_reward(pnl)
                self._balance += pnl
                self._trade_log.append(
                    {"type": "close_short", "price": current_price, "pnl": pnl, "step": self._step}
                )
                self._position = 0
            if self._position == 0:  # open long
                self._position = 1
                self._entry_price = current_price + spread
                self._trade_log.append(
                    {"type": "open_long", "price": self._entry_price, "step": self._step}
                )

        elif action == SELL:
            if self._position == 1:  # close long
                pnl = (current_price - self._entry_price - spread) * self.lot_size * 100_000
                reward += self._shaped_reward(pnl)
                self._balance += pnl
                self._trade_log.append(
                    {"type": "close_long", "price": current_price, "pnl": pnl, "step": self._step}
                )
                self._position = 0
            if self._position == 0:  # open short
                self._position = -1
                self._entry_price = current_price - spread
                self._trade_log.append(
                    {"type": "open_short", "price": self._entry_price, "step": self._step}
                )

        # Holding cost: penalise unrealised loss per step
        if self._position != 0:
            unrealised = self._unrealised_pnl(current_price)
            if unrealised < 0:
                reward += 0.001 * self._shaped_reward(unrealised)

        equity = self._balance + self._unrealised_pnl(current_price)
        self._equity_curve.append(equity)

        self._step += 1
        done = self._step >= self._max_steps or self._balance <= 0

        obs = self._observation()
        info = {
            "balance": self._balance,
            "equity": equity,
            "position": self._position,
            "step": self._step,
        }
        return obs, float(reward), done, False, info

    def render(self) -> None:
        price = self._prices[self._step + self.WINDOW - 1]
        equity = self._equity_curve[-1] if self._equity_curve else self.initial_balance
        pos_str = {1: "LONG", 0: "FLAT", -1: "SHORT"}.get(self._position, "?")
        print(
            f"Step {self._step:5d} | Price {price:.5f} | "
            f"Position {pos_str:5s} | Equity {equity:.2f}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    @property
    def trade_log(self) -> list[dict[str, Any]]:
        return list(self._trade_log)

    @property
    def current_price(self) -> float:
        idx = min(self._step + self.WINDOW, self._n - 1)
        return float(self._prices[idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shaped_reward(self, pnl: float) -> float:
        """Apply asymmetric loss penalty to the raw P&L."""
        if pnl < 0:
            return pnl * self.loss_penalty
        return pnl

    def _unrealised_pnl(self, current_price: float) -> float:
        if self._position == 0:
            return 0.0
        if self._position == 1:
            return (current_price - self._entry_price) * self.lot_size * 100_000
        return (self._entry_price - current_price) * self.lot_size * 100_000

    def _observation(self) -> np.ndarray:
        idx = self._step + self.WINDOW
        if idx >= self._n:
            idx = self._n - 1

        # 20 normalised log-returns
        window_prices = self._prices[idx - self.WINDOW : idx]
        returns = np.diff(np.log(np.clip(window_prices, 1e-10, None)))
        # Pad to WINDOW if not enough data
        if len(returns) < self.WINDOW:
            returns = np.pad(returns, (self.WINDOW - len(returns), 0))

        # RSI normalised to [0, 1]
        rsi_val = self._rsi[idx]
        rsi_norm = (rsi_val / 100.0) if not np.isnan(rsi_val) else 0.5

        # MACD histogram normalised
        macd_val = self._macd_hist[idx]
        if np.isnan(macd_val):
            macd_val = 0.0
        price_scale = self._prices[idx] if self._prices[idx] != 0 else 1.0
        macd_norm = np.clip(macd_val / price_scale, -1.0, 1.0)

        # Bollinger Band width normalised
        bb_w = self._bb_width[idx]
        if np.isnan(bb_w):
            bb_w = 0.0
        bb_norm = np.clip(bb_w / price_scale, 0.0, 1.0)

        # Position one-hot
        pos_onehot = np.array([
            float(self._position == 1),
            float(self._position == 0),
            float(self._position == -1),
        ])

        # Normalised unrealised P&L
        current_price = self._prices[idx]
        upnl = self._unrealised_pnl(current_price) / self.initial_balance

        obs = np.concatenate([
            returns.astype(np.float32),
            [rsi_norm, macd_norm, bb_norm],
            pos_onehot,
            [upnl],
        ]).astype(np.float32)

        return obs
