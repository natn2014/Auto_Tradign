"""Tests for the RL trading environment."""
import numpy as np
import pytest

from trading.environment import BUY, HOLD, SELL, TradingEnvironment
from trading.simulator import MarketSimulator


@pytest.fixture
def env():
    sim = MarketSimulator(n_bars=300, seed=7)
    return TradingEnvironment(sim, "EURUSD", initial_balance=10_000.0)


class TestTradingEnvironment:
    def test_observation_space_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (env.N_FEATURES,)

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_action_space(self, env):
        assert env.action_space.n == 3

    def test_episode_runs_without_error(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        assert steps > 0

    def test_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(HOLD)
        assert isinstance(reward, float)

    def test_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(HOLD)
        for key in ("balance", "equity", "position", "step"):
            assert key in info

    def test_equity_curve_grows(self, env):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(HOLD)
        assert len(env.equity_curve) > 1

    def test_buy_then_sell_updates_balance(self, env):
        env.reset()
        # In XM-style forex trading, SELL on a long position:
        #   1. Closes the long trade (realises P&L), then
        #   2. Opens a new short position in the same step.
        env._position = 0
        env.step(BUY)
        assert env._position == 1   # opened long
        env.step(SELL)
        assert env._position == -1  # closed long, opened short

    def test_reset_clears_position(self, env):
        env.reset()
        env.step(BUY)
        env.reset()
        assert env._position == 0
        assert env._step == 0

    def test_loss_penalty_amplifies_negative_reward(self):
        sim = MarketSimulator(n_bars=300, seed=0)
        env_low = TradingEnvironment(sim, "EURUSD", loss_penalty=1.0)
        env_high = TradingEnvironment(sim, "EURUSD", loss_penalty=5.0)

        # A shaped reward for a loss should be larger in magnitude with higher penalty
        assert abs(env_high._shaped_reward(-1.0)) > abs(env_low._shaped_reward(-1.0))
        # A positive reward should be equal
        assert env_high._shaped_reward(1.0) == env_low._shaped_reward(1.0)
