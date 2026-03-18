"""Main application window for the AI trading simulator."""
from __future__ import annotations

import os

from PySide6.QtCore import QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from trading.agent import DQNAgent
from trading.simulator import MarketSimulator
from ui.chart_widget import TradingChartWidget
from ui.control_panel import ControlPanel


# ──────────────────────────────────────────────────────────────────────────────
# Background worker thread
# ──────────────────────────────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """Runs DQN training episodes in a background thread.

    Signals
    -------
    episode_done(episode_number, result_dict)
    step_update(info_dict)
    finished_all
    """

    episode_done = Signal(int, dict)
    step_update = Signal(dict)
    finished_all = Signal()

    def __init__(
        self,
        agent: DQNAgent,
        n_episodes: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._agent = agent
        self._n_episodes = n_episodes
        self._stop_flag = False

    def request_stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:  # noqa: D102
        self._agent.on_step_callback = lambda info: self.step_update.emit(info)

        for ep in range(1, self._n_episodes + 1):
            if self._stop_flag:
                break
            result = self._agent.train_episode()
            self.episode_done.emit(ep, result)

        self._agent.on_step_callback = None
        self.finished_all.emit()


class SimulationWorker(QThread):
    """Runs a simulation episode in a background thread."""

    simulation_done = Signal(list, list)  # equity_curve, trade_log
    finished_all = Signal()

    def __init__(
        self,
        agent: DQNAgent,
        symbol: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._agent = agent
        self._symbol = symbol if symbol and symbol != "Auto (self-select)" else None

    def run(self) -> None:  # noqa: D102
        equity_curve, trade_log = self._agent.run_simulation(self._symbol)
        self.simulation_done.emit(equity_curve, trade_log)
        self.finished_all.emit()


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window."""

    WINDOW_TITLE = "Auto Trading AI — Simulation Mode"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1280, 720)

        # Initialise trading components
        self._simulator = MarketSimulator(n_bars=2000, seed=42)
        self._agent = DQNAgent(self._simulator)

        self._worker: QThread | None = None
        self._current_episode = 0

        self._setup_ui()
        self._setup_style()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        self._control_panel = ControlPanel()
        self._control_panel.setMinimumWidth(220)
        self._control_panel.setMaximumWidth(300)
        splitter.addWidget(self._control_panel)

        self._chart = TradingChartWidget()
        splitter.addWidget(self._chart)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready — configure parameters and click Start Training or Run Simulation.")

    def _setup_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
                color: #bac2de;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QLabel { color: #cdd6f4; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 3px;
                padding: 2px 4px;
            }
            QSplitter::handle { background: #313244; width: 2px; }
            QStatusBar { background: #181825; color: #6c7086; }
        """)

    def _connect_signals(self) -> None:
        cp = self._control_panel
        cp.train_requested.connect(self._on_train_requested)
        cp.simulate_requested.connect(self._on_simulate_requested)
        cp.stop_requested.connect(self._on_stop_requested)
        cp.save_requested.connect(self._on_save_requested)
        cp.load_requested.connect(self._on_load_requested)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_train_requested(self, n_episodes: int) -> None:
        if self._worker and self._worker.isRunning():
            return

        # Recreate agent if loss-penalty or LR changed
        self._agent = DQNAgent(
            self._simulator,
            loss_penalty=self._control_panel.loss_penalty,
            lr=self._control_panel.learning_rate,
        )

        self._current_episode = 0
        self._chart.clear()
        self._control_panel.set_buttons_training(True)
        self._control_panel.set_status("Training…")
        self.statusBar().showMessage(f"Training for {n_episodes} episodes…")

        self._worker = TrainingWorker(self._agent, n_episodes, self)
        self._worker.episode_done.connect(self._on_episode_done)
        self._worker.step_update.connect(self._on_step_update)
        self._worker.finished_all.connect(self._on_training_finished)
        self._worker.start()

    @Slot(str)
    def _on_simulate_requested(self, symbol: str) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._chart.clear()
        self._control_panel.set_buttons_training(True)
        self._control_panel.set_status("Simulating…")
        self.statusBar().showMessage("Running simulation with current policy…")

        sym = None if symbol == "Auto (self-select)" else symbol
        self._worker = SimulationWorker(self._agent, sym, self)
        self._worker.simulation_done.connect(self._on_simulation_done)
        self._worker.finished_all.connect(
            lambda: self._control_panel.set_buttons_training(False)
        )
        self._worker.finished_all.connect(
            lambda: self._control_panel.set_status("Idle")
        )
        self._worker.start()

    @Slot()
    def _on_stop_requested(self) -> None:
        if self._worker and hasattr(self._worker, "request_stop"):
            self._worker.request_stop()
        self._control_panel.set_status("Stopping…")

    @Slot()
    def _on_save_requested(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "model.pth", "PyTorch Model (*.pth)"
        )
        if path:
            self._agent.save(path)
            self.statusBar().showMessage(f"Model saved to {path}")

    @Slot()
    def _on_load_requested(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "PyTorch Model (*.pth)"
        )
        if path and os.path.isfile(path):
            self._agent.load(path)
            self.statusBar().showMessage(f"Model loaded from {path}")

    @Slot(int, dict)
    def _on_episode_done(self, episode: int, result: dict) -> None:
        self._current_episode = episode
        self._control_panel.set_episode(episode)
        self._control_panel.set_epsilon(result.get("epsilon", 0))
        self._control_panel.set_selected_symbol(result.get("symbol", "—"))
        equity = result.get("final_equity", self._agent.initial_balance)
        self._control_panel.set_equity(equity)
        self._control_panel.set_last_reward(result.get("total_reward", 0))

        # Update equity chart with this episode's final equity
        self._chart.append_equity(equity)
        self.statusBar().showMessage(
            f"Episode {episode} | Symbol: {result.get('symbol')} | "
            f"Equity: ${equity:,.2f} | ε={result.get('epsilon', 0):.4f}"
        )

    @Slot(dict)
    def _on_step_update(self, info: dict) -> None:
        # Update price chart live during training
        env = self._agent.envs.get(info.get("symbol", ""))
        if env:
            self._chart.append_price(env.current_price)

    @Slot()
    def _on_training_finished(self) -> None:
        self._control_panel.set_buttons_training(False)
        self._control_panel.set_status("Idle")
        self.statusBar().showMessage(
            f"Training complete — {self._current_episode} episodes finished."
        )

    @Slot(list, list)
    def _on_simulation_done(self, equity_curve: list[float], trade_log: list[dict]) -> None:
        # Show equity curve
        self._chart.set_equity_curve(equity_curve)

        # Show price history
        sym = self._agent.last_selected_symbol or self._agent.symbols[0]
        prices = self._simulator.get_close(sym)
        for p in prices[-len(equity_curve):]:
            self._chart.append_price(float(p))

        # Add trade markers
        for trade in trade_log:
            t_type = trade.get("type", "")
            step = trade.get("step", 0)
            price = trade.get("price", 0.0)
            if "open_long" in t_type:
                self._chart.add_trade_marker(step, price, is_buy=True)
            elif "open_short" in t_type:
                self._chart.add_trade_marker(step, price, is_buy=False)

        n_trades = len([t for t in trade_log if "close" in t.get("type", "")])
        final_equity = equity_curve[-1] if equity_curve else self._agent.initial_balance
        pnl = final_equity - self._agent.initial_balance
        self._control_panel.set_equity(final_equity)
        self.statusBar().showMessage(
            f"Simulation done — {n_trades} trades | "
            f"P&L: ${pnl:+,.2f} | Final equity: ${final_equity:,.2f}"
        )
