"""Control panel widget for the trading application."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from trading.simulator import DEFAULT_SYMBOLS


class ControlPanel(QWidget):
    """Left-side panel with training / simulation controls and status labels.

    Signals
    -------
    train_requested(n_episodes):
        Emitted when the user clicks "Start Training".
    simulate_requested(symbol):
        Emitted when the user clicks "Run Simulation".
    stop_requested:
        Emitted when the user clicks "Stop".
    save_requested:
        Emitted when the user clicks "Save Model".
    load_requested:
        Emitted when the user clicks "Load Model".
    """

    train_requested = Signal(int)
    simulate_requested = Signal(str)
    stop_requested = Signal()
    save_requested = Signal()
    load_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self._lbl_status.setText(text)

    def set_episode(self, ep: int) -> None:
        self._lbl_episode.setText(str(ep))

    def set_epsilon(self, eps: float) -> None:
        self._lbl_epsilon.setText(f"{eps:.4f}")

    def set_equity(self, equity: float) -> None:
        color = "#a6e3a1" if equity >= 0 else "#f38ba8"
        self._lbl_equity.setText(f"<span style='color:{color}'>${equity:,.2f}</span>")

    def set_last_reward(self, reward: float) -> None:
        color = "#a6e3a1" if reward >= 0 else "#f38ba8"
        self._lbl_reward.setText(
            f"<span style='color:{color}'>{reward:+.4f}</span>"
        )

    def set_selected_symbol(self, symbol: str) -> None:
        self._lbl_symbol.setText(symbol)

    def set_buttons_training(self, is_training: bool) -> None:
        self._btn_train.setEnabled(not is_training)
        self._btn_simulate.setEnabled(not is_training)
        self._btn_stop.setEnabled(is_training)

    @property
    def n_episodes(self) -> int:
        return self._spin_episodes.value()

    @property
    def selected_symbol(self) -> str:
        return self._combo_symbol.currentText()

    @property
    def loss_penalty(self) -> float:
        return self._spin_loss_penalty.value()

    @property
    def learning_rate(self) -> float:
        return self._spin_lr.value()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── Parameters group ────────────────────────────────────────
        grp_params = QGroupBox("Parameters")
        form = QFormLayout(grp_params)

        self._combo_symbol = QComboBox()
        self._combo_symbol.addItem("Auto (self-select)")
        for sym in DEFAULT_SYMBOLS:
            self._combo_symbol.addItem(sym)
        form.addRow("Symbol:", self._combo_symbol)

        self._spin_episodes = QSpinBox()
        self._spin_episodes.setRange(1, 10_000)
        self._spin_episodes.setValue(50)
        form.addRow("Episodes:", self._spin_episodes)

        self._spin_loss_penalty = QDoubleSpinBox()
        self._spin_loss_penalty.setRange(1.0, 10.0)
        self._spin_loss_penalty.setSingleStep(0.5)
        self._spin_loss_penalty.setValue(2.0)
        self._spin_loss_penalty.setToolTip(
            "Penalty multiplier for losing trades (higher = more loss-averse)"
        )
        form.addRow("Loss Penalty:", self._spin_loss_penalty)

        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setDecimals(5)
        self._spin_lr.setRange(1e-5, 1e-1)
        self._spin_lr.setSingleStep(1e-4)
        self._spin_lr.setValue(1e-3)
        form.addRow("Learning Rate:", self._spin_lr)

        root.addWidget(grp_params)

        # ── Buttons ─────────────────────────────────────────────────
        grp_ctrl = QGroupBox("Control")
        vb = QVBoxLayout(grp_ctrl)

        self._btn_train = QPushButton("▶  Start Training")
        self._btn_train.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; padding:6px; border-radius:4px;}"
            "QPushButton:hover { background:#45475a; }"
        )
        self._btn_train.clicked.connect(
            lambda: self.train_requested.emit(self._spin_episodes.value())
        )
        vb.addWidget(self._btn_train)

        self._btn_simulate = QPushButton("▶  Run Simulation")
        self._btn_simulate.setStyleSheet(self._btn_train.styleSheet())
        self._btn_simulate.clicked.connect(
            lambda: self.simulate_requested.emit(self._combo_symbol.currentText())
        )
        vb.addWidget(self._btn_simulate)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background:#313244; color:#f38ba8; padding:6px; border-radius:4px;}"
            "QPushButton:hover { background:#45475a; }"
        )
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        vb.addWidget(self._btn_stop)

        btn_row = QHBoxLayout()
        self._btn_save = QPushButton("💾 Save")
        self._btn_load = QPushButton("📂 Load")
        for btn in (self._btn_save, self._btn_load):
            btn.setStyleSheet(self._btn_train.styleSheet())
        self._btn_save.clicked.connect(self.save_requested.emit)
        self._btn_load.clicked.connect(self.load_requested.emit)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_load)
        vb.addLayout(btn_row)

        root.addWidget(grp_ctrl)

        # ── Status group ─────────────────────────────────────────────
        grp_status = QGroupBox("Status")
        form2 = QFormLayout(grp_status)

        self._lbl_status = QLabel("Idle")
        form2.addRow("State:", self._lbl_status)

        self._lbl_episode = QLabel("0")
        form2.addRow("Episode:", self._lbl_episode)

        self._lbl_epsilon = QLabel("1.0000")
        form2.addRow("Epsilon:", self._lbl_epsilon)

        self._lbl_symbol = QLabel("—")
        form2.addRow("Symbol:", self._lbl_symbol)

        self._lbl_equity = QLabel("$10,000.00")
        form2.addRow("Equity:", self._lbl_equity)

        self._lbl_reward = QLabel("+0.0000")
        form2.addRow("Last Reward:", self._lbl_reward)

        root.addWidget(grp_status)
        root.addStretch()
