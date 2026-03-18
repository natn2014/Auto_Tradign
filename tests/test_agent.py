"""Tests for the DQN trading agent."""
import numpy as np
import pytest
import torch

from trading.agent import DQNAgent, DQNNetwork, ReplayBuffer
from trading.simulator import MarketSimulator


@pytest.fixture
def sim():
    return MarketSimulator(n_bars=300, seed=42)


@pytest.fixture
def agent(sim):
    return DQNAgent(
        sim,
        symbols=["EURUSD", "GBPUSD"],
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9,
        batch_size=16,
        target_update_freq=5,
    )


class TestDQNNetwork:
    def test_forward_shape(self):
        net = DQNNetwork(n_inputs=28, n_actions=3, hidden=64)
        x = torch.randn(4, 28)
        out = net(x)
        assert out.shape == (4, 3)


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(10), 0, 0.0, np.zeros(10), False)
        assert len(buf) == 50
        sample = buf.sample(16)
        assert len(sample) == 16

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(5), 0, 0.0, np.zeros(5), False)
        assert len(buf) == 10


class TestDQNAgent:
    def test_select_symbol_returns_valid(self, agent):
        sym = agent.select_symbol()
        assert sym in ["EURUSD", "GBPUSD"]

    def test_select_action_valid_range(self, agent):
        obs = np.zeros(agent.envs["EURUSD"].N_FEATURES, dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(obs)
            assert action in (0, 1, 2)

    def test_train_episode_returns_dict(self, agent):
        result = agent.train_episode()
        assert "symbol" in result
        assert "total_reward" in result
        assert "final_equity" in result
        assert "epsilon" in result

    def test_epsilon_decays_over_episodes(self, agent):
        eps_before = agent.epsilon
        for _ in range(5):
            agent.train_episode()
        assert agent.epsilon < eps_before

    def test_run_simulation_returns_equity_curve(self, agent):
        equity, trades = agent.run_simulation("EURUSD")
        assert len(equity) > 1
        assert isinstance(trades, list)

    def test_save_load_roundtrip(self, agent, tmp_path):
        path = str(tmp_path / "model.pth")
        agent.save(path)
        # Mutate weights
        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.zero_()
        agent.load(path)
        # Weights should be restored (not all zero)
        total = sum(p.abs().sum().item() for p in agent.policy_net.parameters())
        assert total > 0
