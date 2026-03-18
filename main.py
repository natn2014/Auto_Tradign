"""Entry point for the Auto Trading AI application.

Run with:
    python main.py
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Auto Trading AI")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
