"""Deep Q-Network (DQN) trading agent with self-select algorithm.

The *self-select algorithm* mirrors the XM auto-selection concept: at each
decision step the agent scores all available symbols using a lightweight
volatility/momentum heuristic and focuses training/trading on the symbol that
currently offers the best risk-adjusted opportunity.
"""
from __future__ import annotations

import collections
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading.environment import TradingEnvironment
from trading.simulator import MarketSimulator


# ──────────────────────────────────────────────────────────────────────────────
# Neural-network model
# ──────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """Three-layer fully connected Q-network.

    Args:
        n_inputs: Dimension of the observation vector.
        n_actions: Number of discrete actions.
        hidden: Width of hidden layers.
    """

    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent that trains and trades across multiple simulated symbols.

    The *self-select* step picks the symbol with the highest recent
    volatility-adjusted momentum so the agent concentrates on the most
    actionable market condition – consistent with the XM "self-select"
    philosophy of routing orders to the best available instrument.

    Parameters
    ----------
    simulator:
        Shared ``MarketSimulator`` instance.
    symbols:
        List of symbols to trade.
    initial_balance:
        Starting account balance per episode.
    lr:
        Learning rate for the Adam optimiser.
    gamma:
        Discount factor.
    epsilon_start / epsilon_end / epsilon_decay:
        Epsilon-greedy exploration schedule.
    batch_size:
        Mini-batch size for each gradient update.
    target_update_freq:
        How many gradient steps between target-network sync operations.
    loss_penalty:
        Asymmetric reward multiplier for losses (passed to the environment).
    """

    def __init__(
        self,
        simulator: MarketSimulator,
        symbols: list[str] | None = None,
        initial_balance: float = 10_000.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        loss_penalty: float = 2.0,
    ) -> None:
        self.simulator = simulator
        self.symbols = symbols or simulator.symbols
        self.initial_balance = initial_balance
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.loss_penalty = loss_penalty

        # Build one environment per symbol
        self.envs: dict[str, TradingEnvironment] = {
            sym: TradingEnvironment(
                simulator, sym, initial_balance=initial_balance, loss_penalty=loss_penalty
            )
            for sym in self.symbols
        }

        # Single shared Q-network (shared weights across symbols)
        sample_env = next(iter(self.envs.values()))
        n_obs = sample_env.N_FEATURES
        n_act = sample_env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(n_obs, n_act).to(self.device)
        self.target_net = DQNNetwork(n_obs, n_act).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        self._train_steps = 0
        self._episode_rewards: list[float] = []
        self._selected_symbols: list[str] = []

        # Training-progress callbacks (set by the UI)
        self.on_step_callback: Any = None  # callable(info_dict) | None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def last_selected_symbol(self) -> str | None:
        """Return the most recently self-selected symbol, or None."""
        return self._selected_symbols[-1] if self._selected_symbols else None

    # ------------------------------------------------------------------
    # Self-select algorithm
    # ------------------------------------------------------------------

    def select_symbol(self) -> str:
        """Choose the most promising symbol to trade next.

        Scores each symbol by its recent volatility-adjusted momentum:

            score = |momentum| / (volatility + eps)

        A higher score means the price is trending strongly relative to
        noise, which gives the agent a clearer signal.
        """
        best_sym = self.symbols[0]
        best_score = -np.inf

        for sym in self.symbols:
            prices = self.simulator.get_close(sym)
            if len(prices) < 20:
                continue
            recent = prices[-20:]
            returns = np.diff(np.log(recent + 1e-10))
            momentum = np.mean(returns)
            volatility = np.std(returns) + 1e-9
            score = abs(momentum) / volatility
            if score > best_score:
                best_score = score
                best_sym = sym

        self._selected_symbols.append(best_sym)
        return best_sym

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(t)
            return int(q_vals.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_episode(self) -> dict[str, float]:
        """Run one full training episode on the self-selected symbol.

        Returns
        -------
        dict with keys: ``symbol``, ``total_reward``, ``final_equity``,
        ``epsilon``, ``n_trades``.
        """
        symbol = self.select_symbol()
        env = self.envs[symbol]
        state, _ = env.reset()

        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            self.replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(self.replay) >= self.batch_size:
                loss = self._update()
                self._train_steps += 1
                if self._train_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.on_step_callback is not None:
                self.on_step_callback(
                    {
                        "symbol": symbol,
                        "step": info["step"],
                        "equity": info["equity"],
                        "balance": info["balance"],
                        "position": info["position"],
                        "epsilon": self.epsilon,
                        "total_reward": total_reward,
                    }
                )

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        result = {
            "symbol": symbol,
            "total_reward": total_reward,
            "final_equity": env.equity_curve[-1],
            "epsilon": self.epsilon,
            "n_trades": len([t for t in env.trade_log if "close" in t.get("type", "")]),
        }
        self._episode_rewards.append(total_reward)
        return result

    def run_simulation(self, symbol: str | None = None) -> tuple[list[float], list[dict]]:
        """Run a single simulation episode using the greedy policy.

        Returns
        -------
        (equity_curve, trade_log)
        """
        sym = symbol or self.select_symbol()
        env = self.envs[sym]
        state, _ = env.reset()
        done = False

        # Use greedy policy (epsilon=0)
        saved_eps = self.epsilon
        self.epsilon = 0.0

        while not done:
            action = self.select_action(state)
            state, _, done, _, _ = env.step(action)

        self.epsilon = saved_eps
        return env.equity_curve, env.trade_log

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy network weights."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy network weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update(self) -> float:
        """One gradient-descent step on a sampled mini-batch."""
        batch = self.replay.sample(self.batch_size)

        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q values (double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())
