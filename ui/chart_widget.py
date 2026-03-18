"""Real-time chart widget using pyqtgraph.

Displays:
* Price history candlestick / line chart
* Equity (profit/loss) curve
* Trade entry/exit markers
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Use a dark theme that matches trading platforms
pg.setConfigOption("background", "#1e1e2e")
pg.setConfigOption("foreground", "#cdd6f4")


class CandlestickItem(pg.GraphicsObject):
    """Simple candlestick chart item for pyqtgraph."""

    def __init__(self, data: list[tuple]) -> None:
        """
        Args:
            data: List of (index, open, high, low, close) tuples.
        """
        super().__init__()
        self.data = data
        self._picture: pg.QtGui.QPicture | None = None
        self._generate_picture()

    def update_data(self, data: list[tuple]) -> None:
        self.data = data
        self._picture = None
        self._generate_picture()
        self.update()

    def _generate_picture(self) -> None:
        import pyqtgraph.Qt.QtGui as QtGui

        self._picture = QtGui.QPicture()
        p = QtGui.QPainter(self._picture)
        p.setPen(pg.mkPen("w"))
        w = 0.4

        for t, o, h, l, c in self.data:
            if c >= o:
                p.setBrush(pg.mkBrush("#a6e3a1"))  # green
                p.setPen(pg.mkPen("#a6e3a1"))
            else:
                p.setBrush(pg.mkBrush("#f38ba8"))  # red
                p.setPen(pg.mkPen("#f38ba8"))

            # High-low line
            p.drawLine(
                pg.QtCore.QPointF(t, l),
                pg.QtCore.QPointF(t, h),
            )
            # Body
            p.drawRect(
                pg.QtCore.QRectF(t - w, min(o, c), w * 2, abs(c - o) or 0.0001)
            )

        p.end()

    def paint(self, p, *args) -> None:  # noqa: ANN001
        if self._picture:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self):  # noqa: N802
        if self._picture:
            return pg.QtCore.QRectF(self._picture.boundingRect())
        return pg.QtCore.QRectF()


class TradingChartWidget(QWidget):
    """Widget containing price chart and equity P&L chart."""

    MAX_CANDLES = 100  # visible candle window

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

        self._price_history: list[float] = []
        self._equity_history: list[float] = []
        self._buy_markers: list[tuple[int, float]] = []
        self._sell_markers: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_price(self, price: float) -> None:
        """Add a single close price to the price chart."""
        self._price_history.append(price)
        self._refresh_price()

    def set_equity_curve(self, equity_curve: list[float]) -> None:
        """Replace the equity curve and refresh the P&L chart."""
        self._equity_history = list(equity_curve)
        self._refresh_equity()

    def append_equity(self, equity: float) -> None:
        """Append one equity value and refresh the P&L chart."""
        self._equity_history.append(equity)
        self._refresh_equity()

    def add_trade_marker(self, step: int, price: float, is_buy: bool) -> None:
        """Add a buy (▲) or sell (▼) marker on the price chart."""
        if is_buy:
            self._buy_markers.append((step, price))
        else:
            self._sell_markers.append((step, price))
        self._refresh_markers()

    def clear(self) -> None:
        """Reset all chart data."""
        self._price_history.clear()
        self._equity_history.clear()
        self._buy_markers.clear()
        self._sell_markers.clear()
        self._price_curve.setData([], [])
        self._equity_curve_plot.setData([], [])
        self._buy_scatter.setData([], [])
        self._sell_scatter.setData([], [])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Graphics layout
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # ── Price chart (top, larger) ───────────────────────────────
        self._price_plot: pg.PlotItem = self._glw.addPlot(row=0, col=0)
        self._price_plot.setLabel("left", "Price")
        self._price_plot.showGrid(x=True, y=True, alpha=0.3)
        self._price_plot.addLegend()

        self._price_curve = self._price_plot.plot(
            pen=pg.mkPen("#89b4fa", width=1.5), name="Price"
        )

        # Buy / sell scatter items
        self._buy_scatter = pg.ScatterPlotItem(
            symbol="t1",  # triangle up
            size=12,
            brush=pg.mkBrush("#a6e3a1"),
            pen=pg.mkPen(None),
        )
        self._sell_scatter = pg.ScatterPlotItem(
            symbol="t",  # triangle down
            size=12,
            brush=pg.mkBrush("#f38ba8"),
            pen=pg.mkPen(None),
        )
        self._price_plot.addItem(self._buy_scatter)
        self._price_plot.addItem(self._sell_scatter)

        # ── Equity / P&L chart (bottom) ─────────────────────────────
        self._equity_plot: pg.PlotItem = self._glw.addPlot(row=1, col=0)
        self._equity_plot.setLabel("left", "Equity ($)")
        self._equity_plot.setLabel("bottom", "Step")
        self._equity_plot.showGrid(x=True, y=True, alpha=0.3)

        self._equity_curve_plot = self._equity_plot.plot(
            pen=pg.mkPen("#cba6f7", width=1.5), name="Equity"
        )

        # Zero line
        self._zero_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#6c7086", style=Qt.DashLine)
        )
        self._equity_plot.addItem(self._zero_line)

        # Link x-axes
        self._equity_plot.setXLink(self._price_plot)

        # Row stretch: price chart 2x taller
        self._glw.ci.layout.setRowStretchFactor(0, 2)
        self._glw.ci.layout.setRowStretchFactor(1, 1)

    def _refresh_price(self) -> None:
        n = len(self._price_history)
        xs = np.arange(n, dtype=float)
        self._price_curve.setData(xs, self._price_history)

    def _refresh_equity(self) -> None:
        n = len(self._equity_history)
        if n == 0:
            return
        xs = np.arange(n, dtype=float)
        self._equity_curve_plot.setData(xs, self._equity_history)
        # Update zero line to initial balance
        self._zero_line.setValue(self._equity_history[0])

    def _refresh_markers(self) -> None:
        if self._buy_markers:
            bx, by = zip(*self._buy_markers)
            self._buy_scatter.setData(list(bx), list(by))
        if self._sell_markers:
            sx, sy = zip(*self._sell_markers)
            self._sell_scatter.setData(list(sx), list(sy))
