# Auto Trading AI

A Python desktop application that trains a Deep Q-Network (DQN) reinforcement-learning agent to trade simulated forex markets, with a real-time PySide6 GUI for monitoring training and running simulations.

---

## Features

- **DQN trading agent** — three-layer fully connected Q-network with a replay buffer, target network, and Double-DQN updates.
- **Self-select algorithm** — at every decision step the agent scores all available symbols by volatility-adjusted momentum and focuses on the most actionable instrument.
- **Asymmetric reward shaping** — configurable loss-penalty multiplier incentivises the agent to cut losses early.
- **Market simulator** — synthetic OHLCV price series generated via Geometric Brownian Motion (GBM) for eight forex pairs.
- **Technical indicators** — RSI, MACD, Bollinger Bands, and SMA computed from close prices and fed to the observation vector.
- **PySide6 GUI** — dark-themed desktop window with a live equity / price chart (pyqtgraph) and a control panel for adjusting hyperparameters, starting/stopping training, running simulations, and saving/loading model weights.

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| PySide6 | ≥ 6.8.0 |
| pyqtgraph | ≥ 0.13.7 |
| NumPy | ≥ 1.26.0 |
| pandas | ≥ 2.2.0 |
| PyTorch | ≥ 2.1.0 |
| Gymnasium | ≥ 1.0.0 |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/natn2014/Auto_Tradign.git
cd Auto_Tradign

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Launch the GUI

```bash
python main.py
```

The application opens a 1280 × 720 window. Use the left-hand **Control Panel** to:

1. **Symbol** — choose a specific forex pair or leave on *Auto (self-select)* to let the agent pick dynamically.
2. **Episodes** — number of training episodes (1 – 10 000).
3. **Loss Penalty** — multiplier applied to negative rewards (higher = more loss-averse agent).
4. **Learning Rate** — Adam optimiser learning rate.
5. Click **▶ Start Training** to begin training in a background thread.
6. Click **▶ Run Simulation** to run a single greedy-policy episode and plot the equity curve with trade markers.
7. Click **■ Stop** to interrupt training gracefully.
8. Use **💾 Save** / **📂 Load** to persist or restore trained model weights (`.pth` files).

The right-hand chart updates live during training, showing price history and per-episode equity.

---

## Project Structure

```
Auto_Tradign/
├── main.py                  # Entry point — creates QApplication and MainWindow
├── requirements.txt
├── trading/
│   ├── simulator.py         # GBM-based market simulator (OHLCV data generation)
│   ├── indicators.py        # RSI, MACD, Bollinger Bands, SMA
│   ├── environment.py       # Gymnasium-compatible single-symbol trading environment
│   └── agent.py             # DQNNetwork, ReplayBuffer, DQNAgent (self-select)
├── ui/
│   ├── main_window.py       # MainWindow, TrainingWorker, SimulationWorker
│   ├── chart_widget.py      # pyqtgraph-based equity / price chart
│   └── control_panel.py     # Hyperparameter controls and status labels
└── tests/
    ├── test_indicators.py
    ├── test_simulator.py
    ├── test_environment.py
    └── test_agent.py
```

---

## Architecture Overview

```
MarketSimulator  ──►  TradingEnvironment  ──►  DQNAgent
    (GBM)              (Gymnasium API)        (DQN + self-select)
                             │
                    Technical indicators
                  (RSI, MACD, BB, SMA)
```

- **`MarketSimulator`** pre-generates 2 000 synthetic 1-minute bars for each symbol using GBM with per-pair volatility parameters.
- **`TradingEnvironment`** wraps a single symbol, exposes a 27-dimensional observation vector (20 log-returns + RSI + MACD histogram + BB width + position one-hot + unrealised P&L), and supports three discrete actions: HOLD (0), BUY (1), SELL (2).
- **`DQNAgent`** maintains one shared Q-network across all symbols, uses the self-select algorithm to pick the most promising symbol each episode, and trains via Double-DQN with a circular replay buffer.

---

## Running Tests

```bash
python -m pytest tests/
```

---

## License

This project is provided for educational and research purposes.